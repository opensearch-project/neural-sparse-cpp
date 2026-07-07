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
 * @brief GPU-accelerated k-means document-to-cluster assignment used by the
 *        index-building (build()) path.
 *
 * The assignment step of Seismic clustering computes, for every document, the
 * dot product against each cluster centroid and picks the highest-scoring
 * cluster. Across a whole inverted list this is a sparse-times-dense matrix
 * product (documents as a CSR matrix, centroids as a dense matrix), which maps
 * directly onto NVIDIA cuSPARSE SpMM followed by a per-row argmax.
 *
 * This class mirrors the CPU map_docs_to_clusters() semantics exactly for
 * float (U32) weights:
 *   - documents equal to a cluster centroid are left untouched (not re-added),
 *   - ties are broken toward the lowest cluster index (strict-greater update).
 *
 * The search() path is unaffected and always runs on the CPU.
 *
 * The implementation lives in gpu_cluster_assigner.cu and is only compiled when
 * the project is configured with -DNSPARSE_ENABLE_CUDA=ON (which defines
 * NSPARSE_WITH_CUDA). The declarations below are always visible so that callers
 * can guard usage with a single runtime available() check.
 */
class GpuClusterAssigner {
public:
    /// Process-wide singleton. Owns the cuSPARSE handle and serializes GPU
    /// access so it is safe to call from OpenMP worker threads.
    static GpuClusterAssigner& instance();

    /// True when the library was built with CUDA support and a usable GPU is
    /// present. When false, callers must use the CPU path.
    static bool available();

    /**
     * @brief Assign each non-centroid document to its best cluster on the GPU.
     *
     * For every document id in @p docs (interpreted as a row of @p vectors),
     * computes the dot product against each cluster centroid
     * (clusters[j].front()) and appends the document to the highest-scoring
     * cluster. Documents that are themselves a centroid are skipped, matching
     * the CPU reference.
     *
     * Only float (U32) weights are supported on the GPU. Callers must check
     * vectors->get_element_size() == U32 and fall back to the CPU path
     * otherwise.
     *
     * @param vectors  full corpus of sparse vectors (CSR-backed).
     * @param docs     document ids belonging to this inverted list.
     * @param clusters in/out clusters; clusters[j].front() is the centroid of
     *                 cluster j and assigned documents are appended.
     */
    void assign(const SparseVectors* vectors, const std::vector<idx_t>& docs,
                std::vector<std::vector<idx_t>>& clusters);

    GpuClusterAssigner(const GpuClusterAssigner&) = delete;
    GpuClusterAssigner& operator=(const GpuClusterAssigner&) = delete;

private:
    GpuClusterAssigner();
    ~GpuClusterAssigner();

    struct Impl;
    // Uploads the corpus to the GPU on first use (or when it changes) and
    // returns an opaque handle to the resident device buffers. Thread-safe.
    const void* ensure_corpus_resident(const SparseVectors* vectors);

    Impl* impl_;
};

/**
 * @brief Heuristic gate for the automatic GPU path in map_docs_to_clusters().
 *
 * The per-inverted-list problems in a Seismic build are individually small and
 * the CPU already parallelizes across lists, so offloading tiny problems to the
 * GPU loses to transfer/launch overhead. This returns true only once the
 * problem is large enough to benefit. The threshold on the number of documents
 * can be overridden with the NSPARSE_GPU_MIN_DOCS environment variable.
 */
bool should_offload_assignment_to_gpu(size_t n_docs, size_t n_clusters);

}  // namespace nsparse::detail

#endif  // GPU_CLUSTER_ASSIGNER_H
