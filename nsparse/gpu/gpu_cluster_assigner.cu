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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "cuda_runtime.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse::detail {
namespace {

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error in " << what << ": " << cudaGetErrorString(status);
        throw std::runtime_error(oss.str());
    }
}

void check_cusparse(cusparseStatus_t status, const char* what) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "cuSPARSE error in " << what << ": "
            << cusparseGetErrorString(status);
        throw std::runtime_error(oss.str());
    }
}

#define NSPARSE_CUDA_CHECK(expr) check_cuda((expr), #expr)
#define NSPARSE_CUSPARSE_CHECK(expr) check_cusparse((expr), #expr)

// Optional phase profiling, enabled by NSPARSE_GPU_PROFILE=1. Accumulates
// wall-time (ns) across all list calls into a few buckets so the per-phase
// split of the GPU path can be inspected without an external profiler. Cheap
// (a handful of atomics per list) and compiled to near-nothing when disabled.
struct Profile {
    std::atomic<int64_t> host_prep{0};   // row_ptr/centroid host setup
    std::atomic<int64_t> gpu_exec{0};     // upload+kernels+SpMM+sync
    std::atomic<int64_t> host_assign{0};  // scatter results into clusters
    std::atomic<int64_t> calls{0};
    bool enabled = false;

    Profile() {
        const char* v = std::getenv("NSPARSE_GPU_PROFILE");
        enabled = (v != nullptr && v[0] == '1');
    }
    ~Profile() {
        if (!enabled) return;
        std::fprintf(stderr,
                     "[nsparse gpu] calls=%lld host_prep=%.3fs gpu_exec=%.3fs "
                     "host_assign=%.3fs\n",
                     static_cast<long long>(calls.load()),
                     host_prep.load() / 1e9, gpu_exec.load() / 1e9,
                     host_assign.load() / 1e9);
    }
};
Profile g_profile;

inline int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// One thread per document row. Scans the n_clusters scores for that row and
// selects the argmax, breaking ties toward the lowest cluster index (strict
// greater-than update) to match the CPU reference in map_docs_to_clusters().
__global__ void row_argmax_kernel(const float* __restrict__ scores,
                                  int n_rows, int n_clusters,
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

// Builds the compact CSR values/columns of the document sub-matrix A by
// gathering rows straight from the resident corpus, so the large doc data never
// crosses the PCIe bus per list. One thread per output row.
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


// Scatters centroid rows from the resident corpus into the dense B matrix
// (dim x n_clusters, row-major, ldb = n_clusters). Column j of B is centroid j.
// One thread per centroid; each writes a disjoint column, so there are no
// races. B must be zeroed first.
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

// =========================================================================
// GPU max-pool for summarize(). For one cluster (posting list) this computes,
// per term, the maximum weight across all documents in the cluster, then
// compacts the touched terms. SPLADE weights are >= 0, so a non-negative float
// compares monotonically as a reinterpreted int32 — allowing atomicMax on the
// integer view of the accumulator.
//
// Buffers (all sized to corpus dim, reused across clusters, left clean on exit
// so no per-cluster dim-wide memset is needed):
//   acc[dim]   : int32 view of the running max weight per term (0 == absent)
//   seen[dim]  : 0/1 first-touch flag used to append each term once to touched
//   touched[]  : compact list of term ids that appeared in this cluster
//   n_touched  : length of touched[] (atomic counter)
// =========================================================================

// Batched per-list max-pool: ONE block per cluster, so an entire inverted
// list's clusters are summarized in a single kernel launch with a single
// stream sync (instead of two blocking syncs per cluster, which serialized ~27M
// GPU round-trips across the build). Clusters are independent and each is owned
// by one block, so only intra-block synchronization is needed.
//
// Buffers (per thread/list, reused across lists, kept clean via touched-reset):
//   acc[n_clusters*dim], seen[n_clusters*dim] : per-cluster per-term max / flag,
//       partitioned so block b owns region [b*dim, (b+1)*dim). Start zeroed and
//       are restored to zero for touched slots before the kernel returns.
//   out_term / out_val : compacted results, laid out per cluster using
//       out_base[b] (the input nnz prefix, an upper bound on distinct terms).
//   out_count[b] : number of distinct terms produced for cluster b.
//   out_sum[b]   : sum of the cluster's max values.
__global__ void summarize_list_kernel(
    const int32_t* __restrict__ corpus_indptr,
    const int32_t* __restrict__ corpus_indices,
    const float* __restrict__ corpus_values,
    const int32_t* __restrict__ docs, const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ out_base, int dim,
    int32_t* __restrict__ acc, int32_t* __restrict__ seen,
    int32_t* __restrict__ out_term, float* __restrict__ out_val,
    int32_t* __restrict__ out_count, float* __restrict__ out_sum) {
    const int b = blockIdx.x;  // one block per cluster
    const int start = offsets[b];
    const int end = offsets[b + 1];
    const int64_t base = static_cast<int64_t>(b) * dim;
    const int32_t obase = out_base[b];

    __shared__ int s_count;
    if (threadIdx.x == 0) {
        s_count = 0;
    }
    __syncthreads();

    // Scatter-max over the cluster's documents. Each distinct term is appended
    // once (via seen[]) to this cluster's output segment at out_term[obase..].
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        const int32_t d = docs[i];
        const int32_t src = corpus_indptr[d];
        const int32_t len = corpus_indptr[d + 1] - src;
        for (int32_t t = 0; t < len; ++t) {
            const int32_t term = corpus_indices[src + t];
            const float v = corpus_values[src + t];
            // Non-negative float -> monotonic int bits; atomicMax on int view.
            atomicMax(&acc[base + term], __float_as_int(v));
            if (atomicCAS(&seen[base + term], 0, 1) == 0) {
                const int slot = atomicAdd(&s_count, 1);
                out_term[obase + slot] = term;
            }
        }
    }
    __syncthreads();

    // Compact: read each touched term's max, accumulate the sum, and reset
    // acc/seen for reuse. Sum via a block-wide atomic into a shared scalar.
    __shared__ float s_sum;
    if (threadIdx.x == 0) {
        s_sum = 0.0F;
        out_count[b] = s_count;
    }
    __syncthreads();

    for (int k = threadIdx.x; k < s_count; k += blockDim.x) {
        const int32_t term = out_term[obase + k];
        const float v = __int_as_float(acc[base + term]);
        out_val[obase + k] = v;
        atomicAdd(&s_sum, v);
        acc[base + term] = 0;   // reset for next list reusing this buffer
        seen[base + term] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        out_sum[b] = s_sum;
    }
}

// The full corpus, uploaded to the GPU once and shared read-only across all
// threads/lists. Indices are widened to int32 on upload to feed both the gather
// kernel and cuSPARSE's 32-bit CSR indices.
struct DeviceCorpus {
    int32_t* indptr = nullptr;   // n_vectors + 1
    int32_t* indices = nullptr;  // nnz
    float* values = nullptr;     // nnz
    int64_t nnz = 0;
    size_t dim = 0;
    size_t n_vectors = 0;
    // Identity of the resident corpus. Pointer alone is unsafe: a new
    // SparseVectors can reuse a freed one's address, so the residency check
    // also compares n_vectors and nnz.
    const SparseVectors* owner = nullptr;

    bool matches(const SparseVectors* v, size_t nv, int64_t nz) const {
        return owner == v && n_vectors == nv && nnz == nz;
    }

    void free() {
        cudaFree(indptr);
        cudaFree(indices);
        cudaFree(values);
        indptr = nullptr;
        indices = nullptr;
        values = nullptr;
        owner = nullptr;
        n_vectors = 0;
        nnz = 0;
    }
};

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

    // summarize() batched per-list max-pool scratch. acc/seen are partitioned
    // as n_clusters x dim and kept clean between lists (each cluster resets the
    // slots it touched). Sized to the largest list seen so far.
    int32_t* d_sum_docs = nullptr;    size_t sum_docs_cap = 0;
    int32_t* d_offsets = nullptr;     size_t offsets_cap = 0;
    int32_t* d_out_base = nullptr;    size_t out_base_cap = 0;
    int32_t* d_acc = nullptr;         size_t acc_cap = 0;
    int32_t* d_seen = nullptr;        size_t seen_cap = 0;
    int32_t* d_out_term = nullptr;    size_t out_term_cap = 0;
    float* d_out_val = nullptr;       size_t out_val_cap = 0;
    int32_t* d_out_count = nullptr;   size_t out_count_cap = 0;
    float* d_out_sum = nullptr;       size_t out_sum_cap = 0;

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
        cudaFree(d_sum_docs);
        cudaFree(d_offsets);
        cudaFree(d_out_base);
        cudaFree(d_acc);
        cudaFree(d_seen);
        cudaFree(d_out_term);
        cudaFree(d_out_val);
        cudaFree(d_out_count);
        cudaFree(d_out_sum);
        if (init) {
            cusparseDestroy(handle);
            cudaStreamDestroy(stream);
        }
    }
};

// Reallocate *ptr only when it must grow. Capacity tracked in bytes.
template <class T>
void ensure_capacity(T** ptr, size_t& cap_bytes, size_t need_bytes) {
    if (cap_bytes < need_bytes) {
        cudaFree(*ptr);
        void* p = nullptr;
        check_cuda(cudaMalloc(&p, need_bytes), "cudaMalloc(ensure_capacity)");
        *ptr = static_cast<T*>(p);
        cap_bytes = need_bytes;
    }
}

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

}  // namespace

struct GpuClusterAssigner::Impl {
    std::mutex corpus_mutex;  // guards residency check/upload only
    DeviceCorpus corpus;
    bool usable = false;
};

GpuClusterAssigner::GpuClusterAssigner() : impl_(new Impl()) {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        return;  // impl_->usable stays false; callers fall back to CPU
    }
    impl_->usable = true;
}

GpuClusterAssigner::~GpuClusterAssigner() {
    if (impl_ != nullptr) {
        impl_->corpus.free();
        delete impl_;
    }
}

GpuClusterAssigner& GpuClusterAssigner::instance() {
    static GpuClusterAssigner singleton;
    return singleton;
}

bool GpuClusterAssigner::available() {
    return instance().impl_ != nullptr && instance().impl_->usable;
}

// Uploads the corpus to the GPU once. Safe to call from many threads; the first
// caller uploads while the rest wait, and subsequent calls are a cheap pointer
// comparison. Returns a stable view (device pointers are not mutated once set).
// Declared as a member so it can touch the private Impl; the header exposes it
// as an opaque const void* to avoid leaking CUDA types.
const void* GpuClusterAssigner::ensure_corpus_resident(
    const SparseVectors* vectors) {
    Impl* impl = impl_;
    const size_t n_vectors = vectors->num_vectors();
    const idx_t* indptr = vectors->indptr_data();
    const term_t* indices = vectors->indices_data();
    const float* values = vectors->values_data_float();
    const int64_t nnz = indptr[n_vectors];

    std::lock_guard<std::mutex> lock(impl->corpus_mutex);
    if (impl->corpus.matches(vectors, n_vectors, nnz)) {
        return &impl->corpus;
    }
    impl->corpus.free();

    // Widen indices (term_t == uint16) to int32 for the gather kernel and
    // cuSPARSE. Done once per corpus.
    std::vector<int32_t> indices32(static_cast<size_t>(nnz));
    for (int64_t i = 0; i < nnz; ++i) {
        indices32[i] = static_cast<int32_t>(indices[i]);
    }

    check_cuda(cudaMalloc(&impl->corpus.indptr,
                          (n_vectors + 1) * sizeof(int32_t)),
               "cudaMalloc(corpus.indptr)");
    check_cuda(cudaMalloc(&impl->corpus.indices, nnz * sizeof(int32_t)),
               "cudaMalloc(corpus.indices)");
    check_cuda(cudaMalloc(&impl->corpus.values, nnz * sizeof(float)),
               "cudaMalloc(corpus.values)");
    check_cuda(cudaMemcpy(impl->corpus.indptr, indptr,
                          (n_vectors + 1) * sizeof(int32_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(corpus.indptr)");
    check_cuda(cudaMemcpy(impl->corpus.indices, indices32.data(),
                          nnz * sizeof(int32_t), cudaMemcpyHostToDevice),
               "cudaMemcpy(corpus.indices)");
    check_cuda(cudaMemcpy(impl->corpus.values, values, nnz * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(corpus.values)");

    impl->corpus.nnz = nnz;
    impl->corpus.dim = vectors->get_dimension();
    impl->corpus.n_vectors = n_vectors;
    impl->corpus.owner = vectors;
    return &impl->corpus;
}

void GpuClusterAssigner::assign(const SparseVectors* vectors,
                                const std::vector<idx_t>& docs,
                                std::vector<std::vector<idx_t>>& clusters) {
    const size_t n_docs = docs.size();
    const size_t n_clusters = clusters.size();
    if (n_docs == 0 || n_clusters == 0) {
        return;
    }

    const bool prof = g_profile.enabled;
    const int64_t t0 = prof ? now_ns() : 0;

    const idx_t* indptr = vectors->indptr_data();
    const size_t dim = vectors->get_dimension();

    // Centroids are the first element of each cluster. Collect them (host) and
    // mark which input docs are centroids so they are not re-added, as on CPU.
    std::vector<int32_t> centroid_docs(n_clusters);
    absl::flat_hash_set<idx_t> centroid_set;
    centroid_set.reserve(n_clusters);
    for (size_t j = 0; j < n_clusters; ++j) {
        centroid_docs[j] = clusters[j].front();
        centroid_set.insert(clusters[j].front());
    }

    // Row pointers of the document sub-matrix A (host-side; the actual column
    // indices and values are gathered on-device from the resident corpus).
    std::vector<int32_t> h_a_row_ptr(n_docs + 1, 0);
    for (size_t i = 0; i < n_docs; ++i) {
        const idx_t d = docs[i];
        h_a_row_ptr[i + 1] =
            h_a_row_ptr[i] + static_cast<int32_t>(indptr[d + 1] - indptr[d]);
    }
    const int64_t nnz_a = h_a_row_ptr[n_docs];

    const int64_t t1 = prof ? now_ns() : 0;

    const DeviceCorpus& corpus =
        *static_cast<const DeviceCorpus*>(ensure_corpus_resident(vectors));
    ThreadCtx& ctx = thread_ctx();
    cudaStream_t stream = ctx.stream;

    // Upload only the small per-list metadata (doc/centroid ids). The actual
    // doc/centroid payloads live on the GPU (resident corpus).
    ensure_capacity(&ctx.d_docs, ctx.docs_cap, n_docs * sizeof(int32_t));
    ensure_capacity(&ctx.d_centroids, ctx.cent_cap,
                    n_clusters * sizeof(int32_t));
    ensure_capacity(&ctx.d_best, ctx.best_cap, n_docs * sizeof(int32_t));
    ensure_capacity(&ctx.d_b, ctx.b_cap, dim * n_clusters * sizeof(float));

    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_docs, docs.data(),
                                       n_docs * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_centroids, centroid_docs.data(),
                                       n_clusters * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));

    // Build dense B (dim x n_clusters, row-major) from centroid rows. The
    // fused kernel below reads B but not A's CSR separately; docs are gathered
    // directly from the corpus inside the fused kernel.
    constexpr int kBlock = 256;
    NSPARSE_CUDA_CHECK(cudaMemsetAsync(
        ctx.d_b, 0, dim * n_clusters * sizeof(float), stream));
    const int cent_grid =
        static_cast<int>((n_clusters + kBlock - 1) / kBlock);
    scatter_dense_kernel<<<cent_grid, kBlock, 0, stream>>>(
        corpus.indptr, corpus.indices, corpus.values, ctx.d_centroids,
        static_cast<int>(n_clusters), static_cast<int>(n_clusters), ctx.d_b);
    NSPARSE_CUDA_CHECK(cudaGetLastError());

    // cuSPARSE SpMM path: gathers A from corpus, then runs SpMM + argmax.
    ensure_capacity(&ctx.d_a_row_ptr, ctx.rowptr_cap,
                    (n_docs + 1) * sizeof(int32_t));
    ensure_capacity(&ctx.d_a_col, ctx.col_cap,
                    static_cast<size_t>(nnz_a) * sizeof(int32_t));
    ensure_capacity(&ctx.d_a_val, ctx.val_cap,
                    static_cast<size_t>(nnz_a) * sizeof(float));
    ensure_capacity(&ctx.d_c, ctx.c_cap,
                    n_docs * n_clusters * sizeof(float));

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
        &mat_a, static_cast<int64_t>(n_docs), static_cast<int64_t>(dim),
        nnz_a, ctx.d_a_row_ptr, ctx.d_a_col, ctx.d_a_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));
    NSPARSE_CUSPARSE_CHECK(cusparseCreateDnMat(
        &mat_b, static_cast<int64_t>(dim),
        static_cast<int64_t>(n_clusters),
        static_cast<int64_t>(n_clusters), ctx.d_b, CUDA_R_32F,
        CUSPARSE_ORDER_ROW));
    NSPARSE_CUSPARSE_CHECK(cusparseCreateDnMat(
        &mat_c, static_cast<int64_t>(n_docs),
        static_cast<int64_t>(n_clusters),
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

    const int64_t t2 = prof ? now_ns() : 0;

    // Append assignments on the host, skipping documents that are themselves a
    // centroid (matching the CPU reference exactly).
    for (size_t i = 0; i < n_docs; ++i) {
        const idx_t d = docs[i];
        if (centroid_set.contains(d)) {
            continue;
        }
        clusters[h_best[i]].push_back(d);
    }

    if (prof) {
        const int64_t t3 = now_ns();
        g_profile.host_prep.fetch_add(t1 - t0, std::memory_order_relaxed);
        g_profile.gpu_exec.fetch_add(t2 - t1, std::memory_order_relaxed);
        g_profile.host_assign.fetch_add(t3 - t2, std::memory_order_relaxed);
        g_profile.calls.fetch_add(1, std::memory_order_relaxed);
    }
}

bool GpuClusterAssigner::summarize_list_maxpool(
    const SparseVectors* vectors, const idx_t* docs, const idx_t* offsets,
    size_t n_clusters, std::vector<ClusterSummary>& out) {
    if (!impl_->usable || n_clusters == 0) {
        return false;
    }
    const size_t dim = vectors->get_dimension();
    const idx_t* indptr = vectors->indptr_data();

    // The flattened doc array spans offsets[0]..offsets[n_clusters]. Compute,
    // per cluster, the nnz prefix (out_base) used to place each cluster's
    // compacted output in a private segment; the total nnz is the output cap.
    const size_t n_docs = static_cast<size_t>(offsets[n_clusters] - offsets[0]);
    if (n_docs == 0) {
        return false;
    }
    std::vector<int32_t> h_out_base(n_clusters + 1, 0);
    for (size_t b = 0; b < n_clusters; ++b) {
        int64_t nnz_b = 0;
        for (idx_t i = offsets[b]; i < offsets[b + 1]; ++i) {
            const idx_t d = docs[i];
            nnz_b += indptr[d + 1] - indptr[d];
        }
        h_out_base[b + 1] = h_out_base[b] + static_cast<int32_t>(nnz_b);
    }
    const int64_t total_nnz = h_out_base[n_clusters];
    if (total_nnz == 0) {
        return false;
    }

    const DeviceCorpus& corpus =
        *static_cast<const DeviceCorpus*>(ensure_corpus_resident(vectors));
    ThreadCtx& ctx = thread_ctx();
    cudaStream_t stream = ctx.stream;

    // acc/seen are partitioned as n_clusters x dim; must start (and remain)
    // zeroed. Zero only when (re)grown — the kernel restores touched slots.
    const size_t acc_bytes = n_clusters * dim * sizeof(int32_t);
    const bool acc_grew = ctx.acc_cap < acc_bytes;
    ensure_capacity(&ctx.d_acc, ctx.acc_cap, acc_bytes);
    ensure_capacity(&ctx.d_seen, ctx.seen_cap, acc_bytes);
    if (acc_grew) {
        NSPARSE_CUDA_CHECK(cudaMemsetAsync(ctx.d_acc, 0, ctx.acc_cap, stream));
        NSPARSE_CUDA_CHECK(
            cudaMemsetAsync(ctx.d_seen, 0, ctx.seen_cap, stream));
    }
    ensure_capacity(&ctx.d_sum_docs, ctx.sum_docs_cap,
                    n_docs * sizeof(int32_t));
    ensure_capacity(&ctx.d_offsets, ctx.offsets_cap,
                    (n_clusters + 1) * sizeof(int32_t));
    ensure_capacity(&ctx.d_out_base, ctx.out_base_cap,
                    (n_clusters + 1) * sizeof(int32_t));
    ensure_capacity(&ctx.d_out_term, ctx.out_term_cap,
                    static_cast<size_t>(total_nnz) * sizeof(int32_t));
    ensure_capacity(&ctx.d_out_val, ctx.out_val_cap,
                    static_cast<size_t>(total_nnz) * sizeof(float));
    ensure_capacity(&ctx.d_out_count, ctx.out_count_cap,
                    n_clusters * sizeof(int32_t));
    ensure_capacity(&ctx.d_out_sum, ctx.out_sum_cap,
                    n_clusters * sizeof(float));

    // Offsets are relative to offsets[0]; rebase to 0 for the flat doc upload.
    std::vector<int32_t> h_offsets(n_clusters + 1);
    const idx_t base0 = offsets[0];
    for (size_t b = 0; b <= n_clusters; ++b) {
        h_offsets[b] = static_cast<int32_t>(offsets[b] - base0);
    }

    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_sum_docs, docs + base0,
                                       n_docs * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_offsets, h_offsets.data(),
                                       (n_clusters + 1) * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_out_base, h_out_base.data(),
                                       (n_clusters + 1) * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));

    // One block per cluster.
    constexpr int kBlock = 128;
    summarize_list_kernel<<<static_cast<int>(n_clusters), kBlock, 0, stream>>>(
        corpus.indptr, corpus.indices, corpus.values, ctx.d_sum_docs,
        ctx.d_offsets, ctx.d_out_base, static_cast<int>(dim), ctx.d_acc,
        ctx.d_seen, ctx.d_out_term, ctx.d_out_val, ctx.d_out_count,
        ctx.d_out_sum);
    NSPARSE_CUDA_CHECK(cudaGetLastError());

    // Copy back the compact outputs (single sync for the whole list).
    std::vector<int32_t> h_term(total_nnz);
    std::vector<float> h_val(total_nnz);
    std::vector<int32_t> h_count(n_clusters);
    std::vector<float> h_sum(n_clusters);
    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(h_term.data(), ctx.d_out_term,
                                       total_nnz * sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, stream));
    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(h_val.data(), ctx.d_out_val,
                                       total_nnz * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(h_count.data(), ctx.d_out_count,
                                       n_clusters * sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, stream));
    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(h_sum.data(), ctx.d_out_sum,
                                       n_clusters * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
    NSPARSE_CUDA_CHECK(cudaStreamSynchronize(stream));

    out.resize(n_clusters);
    for (size_t b = 0; b < n_clusters; ++b) {
        const int32_t base = h_out_base[b];
        const int32_t cnt = h_count[b];
        ClusterSummary& cs = out[b];
        cs.terms.resize(cnt);
        cs.values.resize(cnt);
        for (int32_t k = 0; k < cnt; ++k) {
            cs.terms[k] = static_cast<term_t>(h_term[base + k]);
            cs.values[k] = h_val[base + k];
        }
        cs.sum = h_sum[b];
    }
    return true;
}

bool should_offload_assignment_to_gpu(size_t n_docs, size_t n_clusters) {
    if (!GpuClusterAssigner::available()) {
        return false;
    }
    // With on-device gather/scatter and per-thread streams the offload is
    // efficient, but very small lists still favor the CPU. Overridable for
    // tuning/benchmarking.
    size_t min_docs = 1024;
    if (const char* env = std::getenv("NSPARSE_GPU_MIN_DOCS")) {
        min_docs = static_cast<size_t>(std::strtoull(env, nullptr, 10));
    }
    return n_docs >= min_docs && n_clusters >= 2;
}

bool should_offload_summarize_to_gpu() {
    if (!GpuClusterAssigner::available()) {
        return false;
    }
    // On by default when a GPU is present. The max-pool is batched as one
    // kernel launch per inverted list (all clusters at once), so it avoids the
    // per-cluster sync storm that made a naive offload far slower. Opt out with
    // NSPARSE_GPU_SUMMARIZE=0.
    if (const char* env = std::getenv("NSPARSE_GPU_SUMMARIZE")) {
        return env[0] != '0';
    }
    return true;
}

}  // namespace nsparse::detail
