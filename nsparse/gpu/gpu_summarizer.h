/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef GPU_SUMMARIZER_H
#define GPU_SUMMARIZER_H

#include <vector>

#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse::detail {

/**
 * @brief GPU per-term max-pool for the summarize() step of the build() path.
 *
 * Computes, per cluster and per term, the maximum weight over the cluster's
 * documents. The tie-order-sensitive sort/truncate stays on the CPU, so this
 * offloads only the max-pool. A whole inverted list is processed in one kernel
 * launch (one block per cluster): clusters average a handful of documents, so a
 * per-cluster launch is dominated by GPU round-trip overhead. float (U32) only.
 *
 * Compiled only with -DNSPARSE_ENABLE_GPU=ON (defines NSPARSE_WITH_GPU). Shares
 * the resident corpus with GpuClusterAssigner via GpuCorpus.
 */
class GpuSummarizer {
public:
    /// Process-wide singleton owning the per-thread CUDA streams.
    static GpuSummarizer& instance();

    /// True when built with GPU support and a usable device is present.
    static bool available();

    /// Per-cluster max-pool result. Terms are in touched-append order (not
    /// sorted); the CPU sort/truncate handles ordering.
    struct ClusterSummary {
        std::vector<term_t> terms;
        std::vector<float> values;
        float sum = 0.0F;
    };

    /**
     * @brief Max-pool a whole inverted list's clusters in one launch.
     *
     * @param vectors     corpus of sparse vectors (CSR-backed).
     * @param docs        flattened doc ids for the list.
     * @param offsets     cluster b owns docs[offsets[b]..offsets[b+1]).
     * @param n_clusters  offsets.size() - 1.
     * @param out         resized to n_clusters; out[b] is cluster b's result.
     * @return false (caller falls back to CPU) if unavailable or list empty.
     */
    bool summarize_list(const SparseVectors* vectors, const idx_t* docs,
                        const idx_t* offsets, size_t n_clusters,
                        std::vector<ClusterSummary>& out);

    GpuSummarizer(const GpuSummarizer&) = delete;
    GpuSummarizer& operator=(const GpuSummarizer&) = delete;

private:
    GpuSummarizer() = default;
};

/**
 * @brief Whether to run summarize()'s max-pool on the GPU. Opt-in via
 * NSPARSE_GPU_SUMMARIZE=1. Default (unset) uses the CPU flat-array path, which
 * is faster on high-core hosts and bit-identical. The GPU offload is a net win
 * on GPU-rich / low-core hosts where CPU summarize dominates wall-time.
 */
bool should_offload_summarize_to_gpu();

}  // namespace nsparse::detail

#endif  // GPU_SUMMARIZER_H
