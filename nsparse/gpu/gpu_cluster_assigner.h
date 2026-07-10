/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef GPU_CLUSTER_ASSIGNER_H
#define GPU_CLUSTER_ASSIGNER_H

#include <vector>

#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse::detail {

/**
 * @brief GPU k-means document-to-cluster assignment for the build() path.
 *
 * Per inverted list, scoring every document against every centroid is a
 * sparse-times-dense product (documents as CSR, centroids as dense), computed
 * with cuSPARSE SpMM plus a per-row argmax. Output matches the scalar CPU
 * map_docs_to_clusters(): centroids are not re-added, ties break to the lowest
 * cluster index. float (U32) weights only; search() is unaffected (CPU).
 *
 * Compiled only with -DNSPARSE_ENABLE_GPU=ON (defines NSPARSE_WITH_GPU); the
 * declarations stay visible so callers guard on a single available() check.
 */
class GpuClusterAssigner {
public:
    /// Process-wide singleton owning the per-thread cuSPARSE handles/streams.
    static GpuClusterAssigner& instance();

    /// True when built with GPU support and a usable device is present.
    static bool available();

    /**
     * @brief Assign each non-centroid document to its best cluster on the GPU.
     *
     * @param vectors  corpus of sparse vectors (CSR-backed).
     * @param docs     document ids in this inverted list.
     * @param clusters in/out; clusters[j].front() is centroid j, assigned docs
     *                 are appended. Documents that are a centroid are skipped.
     */
    void assign(const SparseVectors* vectors, const std::vector<idx_t>& docs,
                std::vector<std::vector<idx_t>>& clusters);

    GpuClusterAssigner(const GpuClusterAssigner&) = delete;
    GpuClusterAssigner& operator=(const GpuClusterAssigner&) = delete;

private:
    GpuClusterAssigner() = default;
};

/**
 * @brief Whether to run assignment on the GPU: built with GPU support, a device
 * is present, and the list has >= 2 clusters. No runtime size threshold.
 */
bool should_offload_assignment_to_gpu(size_t n_docs, size_t n_clusters);

}  // namespace nsparse::detail

#endif  // GPU_CLUSTER_ASSIGNER_H
