/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/gpu/gpu_summarizer.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "cuda_runtime.h"
#include "nsparse/gpu/gpu_common.cuh"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse::detail {
namespace {

// Batched per-list max-pool: one block per cluster, so a whole inverted list is
// summarized in one launch + one sync (a per-cluster launch is dominated by GPU
// round-trip overhead). Clusters are independent, each owned by one block, so
// only intra-block sync is needed. SPLADE weights are >= 0, so a non-negative
// float orders monotonically as its reinterpreted int32 — allowing atomicMax on
// the accumulator's int view.
//
// Buffers (per thread, reused across lists, left clean via touched-reset):
//   acc[n_clusters*dim] / seen[n_clusters*dim] : per-cluster per-term max / flag,
//       block b owning region [b*dim, (b+1)*dim). Start zeroed; touched slots
//       are restored to zero before return.
//   out_term / out_val : compacted results, per cluster at out_base[b].
//   out_count[b] / out_sum[b] : distinct terms and sum of maxes for cluster b.
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

    // Scatter-max over the cluster's documents; each distinct term is appended
    // once (via seen[]) to this cluster's output segment.
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        const int32_t d = docs[i];
        const int32_t src = corpus_indptr[d];
        const int32_t len = corpus_indptr[d + 1] - src;
        for (int32_t t = 0; t < len; ++t) {
            const int32_t term = corpus_indices[src + t];
            const float v = corpus_values[src + t];
            atomicMax(&acc[base + term], __float_as_int(v));
            if (atomicCAS(&seen[base + term], 0, 1) == 0) {
                out_term[obase + atomicAdd(&s_count, 1)] = term;
            }
        }
    }
    __syncthreads();

    // Read each touched term's max, accumulate the sum, and reset acc/seen for
    // reuse by the next list.
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
        acc[base + term] = 0;
        seen[base + term] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        out_sum[b] = s_sum;
    }
}

// Per-thread scratch (reused across lists, grown on demand). acc/seen are
// n_clusters x dim and kept zeroed between lists via the kernel's touched-reset.
struct ThreadCtx {
    cudaStream_t stream{};
    bool init = false;

    int32_t* d_docs = nullptr;      size_t docs_cap = 0;
    int32_t* d_offsets = nullptr;   size_t offsets_cap = 0;
    int32_t* d_out_base = nullptr;  size_t out_base_cap = 0;
    int32_t* d_acc = nullptr;       size_t acc_cap = 0;
    int32_t* d_seen = nullptr;      size_t seen_cap = 0;
    int32_t* d_out_term = nullptr;  size_t out_term_cap = 0;
    float* d_out_val = nullptr;     size_t out_val_cap = 0;
    int32_t* d_out_count = nullptr; size_t out_count_cap = 0;
    float* d_out_sum = nullptr;     size_t out_sum_cap = 0;

    ~ThreadCtx() {
        cudaFree(d_docs);
        cudaFree(d_offsets);
        cudaFree(d_out_base);
        cudaFree(d_acc);
        cudaFree(d_seen);
        cudaFree(d_out_term);
        cudaFree(d_out_val);
        cudaFree(d_out_count);
        cudaFree(d_out_sum);
        if (init) {
            cudaStreamDestroy(stream);
        }
    }
};

thread_local std::unique_ptr<ThreadCtx> t_ctx;

ThreadCtx& thread_ctx() {
    if (t_ctx == nullptr) {
        t_ctx = std::make_unique<ThreadCtx>();
        check_cuda(cudaStreamCreate(&t_ctx->stream), "cudaStreamCreate");
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

GpuSummarizer& GpuSummarizer::instance() {
    static GpuSummarizer singleton;
    return singleton;
}

bool GpuSummarizer::available() {
    static const bool present = gpu_present();
    return present;
}

bool GpuSummarizer::summarize_list(const SparseVectors* vectors,
                                   const idx_t* docs, const idx_t* offsets,
                                   size_t n_clusters,
                                   std::vector<ClusterSummary>& out) {
    if (!available() || n_clusters == 0) {
        return false;
    }
    const size_t dim = vectors->get_dimension();
    const idx_t* indptr = vectors->indptr_data();

    const size_t n_docs = static_cast<size_t>(offsets[n_clusters] - offsets[0]);
    if (n_docs == 0) {
        return false;
    }
    // Per-cluster nnz prefix: places each cluster's compact output in a private
    // segment; the total is the output capacity.
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

    const DeviceCorpus& corpus = GpuCorpus::instance().ensure_resident(vectors);
    ThreadCtx& ctx = thread_ctx();
    cudaStream_t stream = ctx.stream;

    // acc/seen are n_clusters x dim; must start (and remain) zeroed. Zero only
    // when (re)grown — the kernel restores touched slots.
    const size_t acc_bytes = n_clusters * dim * sizeof(int32_t);
    const bool acc_grew = ctx.acc_cap < acc_bytes;
    ensure_capacity(&ctx.d_acc, ctx.acc_cap, acc_bytes);
    ensure_capacity(&ctx.d_seen, ctx.seen_cap, acc_bytes);
    if (acc_grew) {
        NSPARSE_CUDA_CHECK(cudaMemsetAsync(ctx.d_acc, 0, ctx.acc_cap, stream));
        NSPARSE_CUDA_CHECK(cudaMemsetAsync(ctx.d_seen, 0, ctx.seen_cap, stream));
    }
    ensure_capacity(&ctx.d_docs, ctx.docs_cap, n_docs * sizeof(int32_t));
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
    ensure_capacity(&ctx.d_out_sum, ctx.out_sum_cap, n_clusters * sizeof(float));

    // Rebase offsets to 0 for the flat doc upload.
    std::vector<int32_t> h_offsets(n_clusters + 1);
    const idx_t base0 = offsets[0];
    for (size_t b = 0; b <= n_clusters; ++b) {
        h_offsets[b] = static_cast<int32_t>(offsets[b] - base0);
    }

    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_docs, docs + base0,
                                       n_docs * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_offsets, h_offsets.data(),
                                       (n_clusters + 1) * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    NSPARSE_CUDA_CHECK(cudaMemcpyAsync(ctx.d_out_base, h_out_base.data(),
                                       (n_clusters + 1) * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));

    constexpr int kBlock = 128;
    summarize_list_kernel<<<static_cast<int>(n_clusters), kBlock, 0, stream>>>(
        corpus.indptr, corpus.indices, corpus.values, ctx.d_docs, ctx.d_offsets,
        ctx.d_out_base, static_cast<int>(dim), ctx.d_acc, ctx.d_seen,
        ctx.d_out_term, ctx.d_out_val, ctx.d_out_count, ctx.d_out_sum);
    NSPARSE_CUDA_CHECK(cudaGetLastError());

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

bool should_offload_summarize_to_gpu() {
    if (!GpuSummarizer::available()) {
        return false;
    }
    const char* env = std::getenv("NSPARSE_GPU_SUMMARIZE");
    return env != nullptr && env[0] == '1';
}

}  // namespace nsparse::detail
