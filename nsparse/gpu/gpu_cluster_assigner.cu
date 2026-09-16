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
// TScore is float for the unquantized path, int64_t for the 8-bit path (the
// int64 dot mirrors the CPU reference); the argmax is identical either way.
template <class TScore>
__global__ void row_argmax_kernel(const TScore* __restrict__ scores, int n_rows,
                                  int n_clusters,
                                  int32_t* __restrict__ best_cluster) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    const TScore* row_scores = scores + static_cast<size_t>(row) * n_clusters;
    TScore best = row_scores[0];
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
// TVal is the corpus value width (float or int8_t).
template <class TVal>
__global__ void gather_csr_kernel(const int32_t* __restrict__ corpus_indptr,
                                  const int32_t* __restrict__ corpus_indices,
                                  const TVal* __restrict__ corpus_values,
                                  const int32_t* __restrict__ docs, int n_docs,
                                  const int32_t* __restrict__ a_row_ptr,
                                  int32_t* __restrict__ a_col,
                                  TVal* __restrict__ a_val) {
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
// no races. B must be zeroed first. TVal is the corpus value width.
template <class TVal>
__global__ void scatter_dense_kernel(const int32_t* __restrict__ corpus_indptr,
                                     const int32_t* __restrict__ corpus_indices,
                                     const TVal* __restrict__ corpus_values,
                                     const int32_t* __restrict__ centroids,
                                     int n_clusters, int ldb,
                                     TVal* __restrict__ b) {
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

// SpMM for 8-bit codes: C[row][j] = sum over doc row's corpus nonzeros of
// code * B[col][j], read straight from the resident corpus (no CSR gather).
// Used instead of cuSPARSE: its int8 SpMM is SIGNED and would misread the
// unsigned [0,255] codes, and it offers no unsigned-int8 path. The accumulator
// is int64 to match the CPU reference for every input: a single product fits
// int32 (<= 255*255), but a doc overlapping a centroid in more than
// INT32_MAX/65025 (~33k) high-value terms -- reachable once the vocabulary
// exceeds ~33k -- would overflow an int32 sum and diverge from the CPU int64
// argmax. One block per doc row; threads stripe over the clusters.
__global__ void assign_u8_spmm_kernel(
    const int32_t* __restrict__ corpus_indptr,
    const int32_t* __restrict__ corpus_indices,
    const uint8_t* __restrict__ corpus_values,
    const int32_t* __restrict__ docs, int n_docs,
    const uint8_t* __restrict__ b, int n_clusters, int ldb,
    int64_t* __restrict__ c) {
    const int row = blockIdx.x;
    if (row >= n_docs) {
        return;
    }
    const int32_t d = docs[row];
    const int32_t start = corpus_indptr[d];
    const int32_t end = corpus_indptr[d + 1];
    int64_t* c_row = c + static_cast<size_t>(row) * n_clusters;
    for (int j = threadIdx.x; j < n_clusters; j += blockDim.x) {
        int64_t acc = 0;
        for (int32_t p = start; p < end; ++p) {
            const int32_t col = corpus_indices[p];
            acc += static_cast<int64_t>(corpus_values[p]) *
                   static_cast<int64_t>(b[static_cast<size_t>(col) * ldb + j]);
        }
        c_row[j] = acc;
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
    // Value/dense/output scratch are byte-capacity (float, or uint8 + int64).
    void* d_a_val = nullptr;         size_t val_cap = 0;
    void* d_b = nullptr;             size_t b_cap = 0;
    void* d_c = nullptr;             size_t c_cap = 0;
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

// Own a cuSPARSE matrix descriptor so a throw between create and use unwinds
// through the destroy instead of leaking the handle.
struct SpMatGuard {
    cusparseSpMatDescr_t desc{};
    ~SpMatGuard() {
        if (desc != nullptr) cusparseDestroySpMat(desc);
    }
};
struct DnMatGuard {
    cusparseDnMatDescr_t desc{};
    ~DnMatGuard() {
        if (desc != nullptr) cusparseDestroyDnMat(desc);
    }
};

// cuSPARSE SpMM core for the unquantized (float) path: build dense B and CSR A,
// run C = A * B, and argmax each row into ctx.d_best (copied to h_best).
// Templated so the types are explicit, but instantiated only as <float,float>;
// 8-bit codes cannot use cuSPARSE (its int8 SpMM is signed) and go through
// assign_u8 instead. The d_docs, d_centroids and d_a_row_ptr scratch are
// uploaded by the caller.
template <class TVal, class TScore>
void assign_spmm(const DeviceCorpus& corpus, ThreadCtx& ctx, int n_docs,
                 int n_clusters, int dim, int64_t nnz_a,
                 cudaDataType_t val_type, cudaDataType_t compute_type,
                 std::vector<int32_t>& h_best) {
    cudaStream_t stream = ctx.stream;
    constexpr int kBlock = 256;

    // Dense B (dim x n_clusters, row-major) from centroid rows.
    const size_t b_bytes = static_cast<size_t>(dim) * n_clusters * sizeof(TVal);
    ensure_capacity(reinterpret_cast<char**>(&ctx.d_b), ctx.b_cap, b_bytes);
    NSPARSE_CUDA_CHECK(cudaMemsetAsync(ctx.d_b, 0, b_bytes, stream));
    const int cent_grid = (n_clusters + kBlock - 1) / kBlock;
    scatter_dense_kernel<TVal><<<cent_grid, kBlock, 0, stream>>>(
        corpus.indptr, corpus.indices,
        static_cast<const TVal*>(corpus.values), ctx.d_centroids, n_clusters,
        n_clusters, static_cast<TVal*>(ctx.d_b));
    NSPARSE_CUDA_CHECK(cudaGetLastError());

    // CSR A gathered from the resident corpus, then C = A * B.
    ensure_capacity(reinterpret_cast<char**>(&ctx.d_a_val), ctx.val_cap,
                    static_cast<size_t>(nnz_a) * sizeof(TVal));
    ensure_capacity(reinterpret_cast<char**>(&ctx.d_c), ctx.c_cap,
                    static_cast<size_t>(n_docs) * n_clusters * sizeof(TScore));
    const int docs_grid = (n_docs + kBlock - 1) / kBlock;
    gather_csr_kernel<TVal><<<docs_grid, kBlock, 0, stream>>>(
        corpus.indptr, corpus.indices,
        static_cast<const TVal*>(corpus.values), ctx.d_docs, n_docs,
        ctx.d_a_row_ptr, ctx.d_a_col, static_cast<TVal*>(ctx.d_a_val));
    NSPARSE_CUDA_CHECK(cudaGetLastError());

    SpMatGuard a_guard;
    DnMatGuard b_guard;
    DnMatGuard c_guard;
    NSPARSE_CUSPARSE_CHECK(cusparseCreateCsr(
        &a_guard.desc, n_docs, dim, nnz_a, ctx.d_a_row_ptr, ctx.d_a_col,
        ctx.d_a_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, val_type));
    NSPARSE_CUSPARSE_CHECK(cusparseCreateDnMat(&b_guard.desc, dim, n_clusters,
                                               n_clusters, ctx.d_b, val_type,
                                               CUSPARSE_ORDER_ROW));
    NSPARSE_CUSPARSE_CHECK(
        cusparseCreateDnMat(&c_guard.desc, n_docs, n_clusters, n_clusters,
                            ctx.d_c, compute_type, CUSPARSE_ORDER_ROW));

    const TScore alpha_v = 1;
    const TScore beta_v = 0;
    size_t buffer_size = 0;
    NSPARSE_CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        ctx.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_v, a_guard.desc, b_guard.desc,
        &beta_v, c_guard.desc, compute_type, CUSPARSE_SPMM_ALG_DEFAULT,
        &buffer_size));
    ensure_capacity(reinterpret_cast<char**>(&ctx.d_spmm), ctx.spmm_cap,
                    buffer_size == 0 ? 1 : buffer_size);
    NSPARSE_CUSPARSE_CHECK(cusparseSpMM(
        ctx.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_v, a_guard.desc, b_guard.desc,
        &beta_v, c_guard.desc, compute_type, CUSPARSE_SPMM_ALG_DEFAULT,
        ctx.d_spmm));

    row_argmax_kernel<TScore><<<docs_grid, kBlock, 0, stream>>>(
        static_cast<const TScore*>(ctx.d_c), n_docs, n_clusters, ctx.d_best);
    NSPARSE_CUDA_CHECK(cudaGetLastError());
    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(h_best.data(), ctx.d_best,
                                       static_cast<size_t>(n_docs) *
                                           sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, stream));
    NSPARSE_CUDA_CHECK(cudaStreamSynchronize(stream));
}

// 8-bit path: dense B (uint8) from centroids + the custom int32 SpMM kernel +
// argmax. Separate from assign_spmm because cuSPARSE cannot do unsigned int8.
void assign_u8(const DeviceCorpus& corpus, ThreadCtx& ctx, int n_docs,
               int n_clusters, int dim, std::vector<int32_t>& h_best) {
    cudaStream_t stream = ctx.stream;
    constexpr int kBlock = 256;

    const size_t b_bytes = static_cast<size_t>(dim) * n_clusters;
    ensure_capacity(reinterpret_cast<char**>(&ctx.d_b), ctx.b_cap, b_bytes);
    NSPARSE_CUDA_CHECK(cudaMemsetAsync(ctx.d_b, 0, b_bytes, stream));
    const int cent_grid = (n_clusters + kBlock - 1) / kBlock;
    scatter_dense_kernel<uint8_t><<<cent_grid, kBlock, 0, stream>>>(
        corpus.indptr, corpus.indices,
        static_cast<const uint8_t*>(corpus.values), ctx.d_centroids, n_clusters,
        n_clusters, static_cast<uint8_t*>(ctx.d_b));
    NSPARSE_CUDA_CHECK(cudaGetLastError());

    ensure_capacity(reinterpret_cast<char**>(&ctx.d_c), ctx.c_cap,
                    static_cast<size_t>(n_docs) * n_clusters * sizeof(int64_t));
    assign_u8_spmm_kernel<<<n_docs, kBlock, 0, stream>>>(
        corpus.indptr, corpus.indices,
        static_cast<const uint8_t*>(corpus.values), ctx.d_docs, n_docs,
        static_cast<const uint8_t*>(ctx.d_b), n_clusters, n_clusters,
        static_cast<int64_t*>(ctx.d_c));
    NSPARSE_CUDA_CHECK(cudaGetLastError());

    const int docs_grid = (n_docs + kBlock - 1) / kBlock;
    row_argmax_kernel<int64_t><<<docs_grid, kBlock, 0, stream>>>(
        static_cast<const int64_t*>(ctx.d_c), n_docs, n_clusters, ctx.d_best);
    NSPARSE_CUDA_CHECK(cudaGetLastError());
    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(h_best.data(), ctx.d_best,
                                       static_cast<size_t>(n_docs) *
                                           sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, stream));
    NSPARSE_CUDA_CHECK(cudaStreamSynchronize(stream));
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

    const offset_t* indptr = vectors->indptr_data();
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

    // Upload the small per-list metadata (doc/centroid ids, row pointers) and
    // size the index scratch; the value/dense/output buffers are sized per
    // width inside assign_spmm.
    ensure_capacity(&ctx.d_docs, ctx.docs_cap, n_docs * sizeof(int32_t));
    ensure_capacity(&ctx.d_centroids, ctx.cent_cap, n_clusters * sizeof(int32_t));
    ensure_capacity(&ctx.d_best, ctx.best_cap, n_docs * sizeof(int32_t));
    ensure_capacity(&ctx.d_a_row_ptr, ctx.rowptr_cap,
                    (n_docs + 1) * sizeof(int32_t));
    ensure_capacity(&ctx.d_a_col, ctx.col_cap,
                    static_cast<size_t>(nnz_a) * sizeof(int32_t));

    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_docs, docs.data(),
                                       n_docs * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_centroids, centroid_docs.data(),
                                       n_clusters * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_a_row_ptr, h_a_row_ptr.data(),
                                       (n_docs + 1) * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));

    // Float corpus -> cuSPARSE float SpMM; 8-bit codes -> the custom uint8
    // int64-accumulate kernel (matches the CPU int64 argmax for any input).
    // Only these two widths reach here (ensure_resident rejects others).
    std::vector<int32_t> h_best(n_docs);
    if (corpus.element_size == U32) {
        assign_spmm<float, float>(corpus, ctx, static_cast<int>(n_docs),
                                  static_cast<int>(n_clusters),
                                  static_cast<int>(dim), nnz_a, CUDA_R_32F,
                                  CUDA_R_32F, h_best);
    } else {
        assign_u8(corpus, ctx, static_cast<int>(n_docs),
                  static_cast<int>(n_clusters), static_cast<int>(dim), h_best);
    }

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
