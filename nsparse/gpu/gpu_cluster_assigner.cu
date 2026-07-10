/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/gpu/gpu_cluster_assigner.h"

#include <cusparse.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "cuda_runtime.h"
#include "nsparse/gpu/gpu_common.cuh"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse::detail {
namespace {

// One thread per document row: argmax over the row's n_clusters scores, ties
// broken to the lowest cluster index (strict-greater) to match the CPU path.
__global__ void row_argmax_kernel(const float* __restrict__ scores, int n_rows,
                                  int n_clusters,
                                  int32_t* __restrict__ best_cluster) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    const float* row_scores = scores + static_cast<size_t>(row) * n_clusters;
    float best = row_scores[0];
    int best_j = 0;
    for (int j = 1; j < n_clusters; ++j) {
        if (row_scores[j] > best) {
            best = row_scores[j];
            best_j = j;
        }
    }
    best_cluster[row] = best_j;
}

// Gather the CSR of the document sub-matrix A from the resident corpus, so the
// bulk doc data never re-crosses PCIe per list. One thread per output row.
__global__ void gather_csr_kernel(const int32_t* __restrict__ corpus_indptr,
                                  const int32_t* __restrict__ corpus_indices,
                                  const float* __restrict__ corpus_values,
                                  const int32_t* __restrict__ docs, int n_docs,
                                  const int32_t* __restrict__ a_row_ptr,
                                  int32_t* __restrict__ a_col,
                                  float* __restrict__ a_val) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_docs) {
        return;
    }
    const int32_t d = docs[row];
    const int32_t src = corpus_indptr[d];
    const int32_t len = corpus_indptr[d + 1] - src;
    const int32_t dst = a_row_ptr[row];
    for (int32_t t = 0; t < len; ++t) {
        a_col[dst + t] = corpus_indices[src + t];
        a_val[dst + t] = corpus_values[src + t];
    }
}

// Scatter centroid rows into the dense B matrix (dim x n_clusters, row-major,
// ldb = n_clusters). One thread per centroid; each owns a disjoint column, so
// no races. B must be zeroed first.
__global__ void scatter_dense_kernel(const int32_t* __restrict__ corpus_indptr,
                                     const int32_t* __restrict__ corpus_indices,
                                     const float* __restrict__ corpus_values,
                                     const int32_t* __restrict__ centroids,
                                     int n_clusters, int ldb,
                                     float* __restrict__ b) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_clusters) {
        return;
    }
    const int32_t c = centroids[j];
    const int32_t src = corpus_indptr[c];
    const int32_t len = corpus_indptr[c + 1] - src;
    for (int32_t t = 0; t < len; ++t) {
        const int32_t col = corpus_indices[src + t];
        b[static_cast<size_t>(col) * ldb + j] = corpus_values[src + t];
    }
}

// Per-thread GPU resources: an independent cuSPARSE handle + stream (handles
// are not shareable across threads) and scratch buffers reused across lists,
// grown on demand so steady-state builds do no per-list cudaMalloc.
struct ThreadCtx {
    cusparseHandle_t handle{};
    cudaStream_t stream{};
    bool init = false;

    int32_t* d_docs = nullptr;       size_t docs_cap = 0;
    int32_t* d_centroids = nullptr;  size_t cent_cap = 0;
    int32_t* d_a_row_ptr = nullptr;  size_t rowptr_cap = 0;
    int32_t* d_a_col = nullptr;      size_t col_cap = 0;
    float* d_a_val = nullptr;        size_t val_cap = 0;
    float* d_b = nullptr;            size_t b_cap = 0;
    float* d_c = nullptr;            size_t c_cap = 0;
    int32_t* d_best = nullptr;       size_t best_cap = 0;
    void* d_spmm = nullptr;          size_t spmm_cap = 0;

    ~ThreadCtx() {
        cudaFree(d_docs);
        cudaFree(d_centroids);
        cudaFree(d_a_row_ptr);
        cudaFree(d_a_col);
        cudaFree(d_a_val);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaFree(d_best);
        cudaFree(d_spmm);
        if (init) {
            cusparseDestroy(handle);
            cudaStreamDestroy(stream);
        }
    }
};

// Lazily created once per worker thread; destroyed at thread exit.
thread_local std::unique_ptr<ThreadCtx> t_ctx;

ThreadCtx& thread_ctx() {
    if (t_ctx == nullptr) {
        t_ctx = std::make_unique<ThreadCtx>();
        check_cuda(cudaStreamCreate(&t_ctx->stream), "cudaStreamCreate");
        check_cusparse(cusparseCreate(&t_ctx->handle), "cusparseCreate");
        check_cusparse(cusparseSetStream(t_ctx->handle, t_ctx->stream),
                       "cusparseSetStream");
        t_ctx->init = true;
    }
    return *t_ctx;
}

bool gpu_present() {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    return status == cudaSuccess && device_count > 0;
}

}  // namespace

GpuClusterAssigner& GpuClusterAssigner::instance() {
    static GpuClusterAssigner singleton;
    return singleton;
}

bool GpuClusterAssigner::available() {
    static const bool present = gpu_present();
    return present;
}

void GpuClusterAssigner::assign(const SparseVectors* vectors,
                                const std::vector<idx_t>& docs,
                                std::vector<std::vector<idx_t>>& clusters) {
    const size_t n_docs = docs.size();
    const size_t n_clusters = clusters.size();
    if (n_docs == 0 || n_clusters == 0) {
        return;
    }

    const idx_t* indptr = vectors->indptr_data();
    const size_t dim = vectors->get_dimension();

    // Centroids are clusters[j].front(); collect them and record which input
    // docs are centroids so they are not re-added (matches the CPU path).
    std::vector<int32_t> centroid_docs(n_clusters);
    absl::flat_hash_set<idx_t> centroid_set;
    centroid_set.reserve(n_clusters);
    for (size_t j = 0; j < n_clusters; ++j) {
        centroid_docs[j] = clusters[j].front();
        centroid_set.insert(clusters[j].front());
    }

    // Row pointers of A (host); column indices/values are gathered on-device.
    std::vector<int32_t> h_a_row_ptr(n_docs + 1, 0);
    for (size_t i = 0; i < n_docs; ++i) {
        const idx_t d = docs[i];
        h_a_row_ptr[i + 1] =
            h_a_row_ptr[i] + static_cast<int32_t>(indptr[d + 1] - indptr[d]);
    }
    const int64_t nnz_a = h_a_row_ptr[n_docs];

    const DeviceCorpus& corpus = GpuCorpus::instance().ensure_resident(vectors);
    ThreadCtx& ctx = thread_ctx();
    cudaStream_t stream = ctx.stream;

    // Upload only the small per-list metadata (doc/centroid ids).
    ensure_capacity(&ctx.d_docs, ctx.docs_cap, n_docs * sizeof(int32_t));
    ensure_capacity(&ctx.d_centroids, ctx.cent_cap, n_clusters * sizeof(int32_t));
    ensure_capacity(&ctx.d_best, ctx.best_cap, n_docs * sizeof(int32_t));
    ensure_capacity(&ctx.d_b, ctx.b_cap, dim * n_clusters * sizeof(float));

    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_docs, docs.data(),
                                       n_docs * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_centroids, centroid_docs.data(),
                                       n_clusters * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));

    // Build dense B (dim x n_clusters, row-major) from centroid rows.
    constexpr int kBlock = 256;
    NSPARSE_CUDA_CHECK(
        cudaMemsetAsync(ctx.d_b, 0, dim * n_clusters * sizeof(float), stream));
    const int cent_grid = static_cast<int>((n_clusters + kBlock - 1) / kBlock);
    scatter_dense_kernel<<<cent_grid, kBlock, 0, stream>>>(
        corpus.indptr, corpus.indices, corpus.values, ctx.d_centroids,
        static_cast<int>(n_clusters), static_cast<int>(n_clusters), ctx.d_b);
    NSPARSE_CUDA_CHECK(cudaGetLastError());

    // Gather A's CSR, then C = A * B (SpMM) and per-row argmax.
    ensure_capacity(&ctx.d_a_row_ptr, ctx.rowptr_cap,
                    (n_docs + 1) * sizeof(int32_t));
    ensure_capacity(&ctx.d_a_col, ctx.col_cap,
                    static_cast<size_t>(nnz_a) * sizeof(int32_t));
    ensure_capacity(&ctx.d_a_val, ctx.val_cap,
                    static_cast<size_t>(nnz_a) * sizeof(float));
    ensure_capacity(&ctx.d_c, ctx.c_cap, n_docs * n_clusters * sizeof(float));

    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_a_row_ptr, h_a_row_ptr.data(),
                                       (n_docs + 1) * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    const int docs_grid = static_cast<int>((n_docs + kBlock - 1) / kBlock);
    gather_csr_kernel<<<docs_grid, kBlock, 0, stream>>>(
        corpus.indptr, corpus.indices, corpus.values, ctx.d_docs,
        static_cast<int>(n_docs), ctx.d_a_row_ptr, ctx.d_a_col, ctx.d_a_val);
    NSPARSE_CUDA_CHECK(cudaGetLastError());

    cusparseSpMatDescr_t mat_a;
    cusparseDnMatDescr_t mat_b;
    cusparseDnMatDescr_t mat_c;
    NSPARSE_CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_a, static_cast<int64_t>(n_docs), static_cast<int64_t>(dim), nnz_a,
        ctx.d_a_row_ptr, ctx.d_a_col, ctx.d_a_val, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    NSPARSE_CUSPARSE_CHECK(cusparseCreateDnMat(
        &mat_b, static_cast<int64_t>(dim), static_cast<int64_t>(n_clusters),
        static_cast<int64_t>(n_clusters), ctx.d_b, CUDA_R_32F,
        CUSPARSE_ORDER_ROW));
    NSPARSE_CUSPARSE_CHECK(cusparseCreateDnMat(
        &mat_c, static_cast<int64_t>(n_docs), static_cast<int64_t>(n_clusters),
        static_cast<int64_t>(n_clusters), ctx.d_c, CUDA_R_32F,
        CUSPARSE_ORDER_ROW));

    const float alpha_v = 1.0F;
    const float beta_v = 0.0F;
    size_t buffer_size = 0;
    NSPARSE_CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        ctx.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_v, mat_a, mat_b, &beta_v,
        mat_c, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));
    ensure_capacity(reinterpret_cast<char**>(&ctx.d_spmm), ctx.spmm_cap,
                    buffer_size == 0 ? 1 : buffer_size);
    NSPARSE_CUSPARSE_CHECK(cusparseSpMM(
        ctx.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_v, mat_a, mat_b, &beta_v,
        mat_c, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, ctx.d_spmm));

    row_argmax_kernel<<<docs_grid, kBlock, 0, stream>>>(
        ctx.d_c, static_cast<int>(n_docs), static_cast<int>(n_clusters),
        ctx.d_best);
    NSPARSE_CUDA_CHECK(cudaGetLastError());

    cusparseDestroySpMat(mat_a);
    cusparseDestroyDnMat(mat_b);
    cusparseDestroyDnMat(mat_c);

    std::vector<int32_t> h_best(n_docs);
    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(h_best.data(), ctx.d_best,
                                       n_docs * sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, stream));
    NSPARSE_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Append assignments, skipping docs that are themselves a centroid.
    for (size_t i = 0; i < n_docs; ++i) {
        const idx_t d = docs[i];
        if (centroid_set.contains(d)) {
            continue;
        }
        clusters[h_best[i]].push_back(d);
    }
}

bool should_offload_assignment_to_gpu(size_t n_docs, size_t n_clusters) {
    return GpuClusterAssigner::available() && n_docs > 0 && n_clusters >= 2;
}

}  // namespace nsparse::detail
