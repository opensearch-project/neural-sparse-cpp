/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/cluster/kmeans_utils.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#ifdef NSPARSE_WITH_GPU
#include "nsparse/gpu/gpu_cluster_assigner.h"
#include "nsparse/gpu/gpu_diagnostics.h"
#endif

namespace nsparse::detail {
namespace {

// Cluster index within one posting list's clustering. Independent of
// InvertedListClusters::cluster_id_t (the narrower on-disk type): clusters here
// can be as many as the documents being clustered.
using local_cluster_id_t = uint32_t;

// Similarity accumulator: exact for quantized weights, whose products can exceed
// float's 24-bit mantissa (65535^2 alone is ~4.3e9), and float for float weights.
template <class T>
using accumulator_t = std::conditional_t<std::is_same_v<T, float>, float, int64_t>;

// Term-major (CSC) index over the cluster centroids: for each term, the list of
// clusters whose centroid contains it, with the centroid's weight.
template <class T>
struct CentroidIndex {
    std::vector<idx_t> term_ptr;              // size n_cols + 1
    std::vector<local_cluster_id_t> cluster;  // size = total centroid postings
    std::vector<T> weight;                    // parallel to cluster
    size_t n_cols = 0;                        // terms [0, n_cols) are indexed
};

// Build the CSC over centroids only. Centroids are sparse (one document each),
// so this is proportional to their combined non-zeros — not to
// dimension x n_clusters.
template <class T>
CentroidIndex<T> build_centroid_index(
    const SparseVectors* vectors,
    const std::vector<std::vector<idx_t>>& clusters) {
    const idx_t* indptr = vectors->indptr_data();
    const term_t* indices = vectors->indices_data();
    const T* values = vectors->typed_values_data<T>();
    const size_t n_clusters = clusters.size();

    // Column count comes from the centroids' own terms rather than the
    // configured dimension, which callers may leave at 0 (unset).
    size_t max_term = 0;
    size_t nnz = 0;
    for (const auto& cluster : clusters) {
        const idx_t centroid = cluster.at(0);
        for (idx_t j = indptr[centroid]; j < indptr[centroid + 1]; ++j) {
            max_term = std::max<size_t>(max_term, indices[j]);
            ++nnz;
        }
    }

    CentroidIndex<T> index;
    index.n_cols = nnz == 0 ? 0 : max_term + 1;
    index.term_ptr.assign(index.n_cols + 1, 0);
    for (const auto& cluster : clusters) {
        const idx_t centroid = cluster.at(0);
        for (idx_t j = indptr[centroid]; j < indptr[centroid + 1]; ++j) {
            index.term_ptr[indices[j] + 1]++;
        }
    }
    for (size_t t = 0; t < index.n_cols; ++t) {
        index.term_ptr[t + 1] += index.term_ptr[t];
    }

    index.cluster.resize(nnz);
    index.weight.resize(nnz);
    std::vector<idx_t> cursor(index.term_ptr.begin(), index.term_ptr.end() - 1);
    for (size_t c = 0; c < n_clusters; ++c) {
        const idx_t centroid = clusters[c].at(0);
        for (idx_t j = indptr[centroid]; j < indptr[centroid + 1]; ++j) {
            const idx_t pos = cursor[indices[j]]++;
            index.cluster[pos] = static_cast<local_cluster_id_t>(c);
            index.weight[pos] = values[j];
        }
    }
    return index;
}

// Assign each doc to its most similar centroid by transposing the comparison:
// walk the doc's own terms and accumulate into only those clusters whose
// centroid shares a term, instead of scoring the doc against a dense
// dimension x n_clusters centroid matrix. Cost falls from
// O(n_docs * doc_nnz * n_clusters) to O(shared postings), and scratch from
// dimension * n_clusters floats to the centroids' non-zeros — the dense matrix,
// hundreds of megabytes at a realistic dimension and cluster count, was rebuilt
// for every posting list and dominated build time.
template <class T>
void map_docs_to_clusters_typed(const SparseVectors* vectors,
                                const std::vector<idx_t>& docs,
                                std::vector<std::vector<idx_t>>& clusters) {
    const idx_t* indptr = vectors->indptr_data();
    const term_t* indices = vectors->indices_data();
    const T* values = vectors->typed_values_data<T>();
    const size_t n_clusters = clusters.size();

    const CentroidIndex<T> index = build_centroid_index<T>(vectors, clusters);

    // Centroids are already members of their own cluster and must not be added
    // again. Sorted for binary search.
    std::vector<idx_t> centroid_ids;
    centroid_ids.reserve(n_clusters);
    for (const auto& cluster : clusters) {
        centroid_ids.push_back(cluster.at(0));
    }
    std::ranges::sort(centroid_ids);

    using acc_t = accumulator_t<T>;
    std::vector<acc_t> similarities(n_clusters, acc_t(0));
    for (const idx_t doc_id : docs) {
        if (std::ranges::binary_search(centroid_ids, doc_id)) {
            continue;
        }
        std::ranges::fill(similarities, acc_t(0));
        for (idx_t j = indptr[doc_id]; j < indptr[doc_id + 1]; ++j) {
            const size_t term = indices[j];
            if (term >= index.n_cols) {
                continue;  // no centroid carries this term
            }
            const acc_t doc_value = static_cast<acc_t>(values[j]);
            for (idx_t p = index.term_ptr[term]; p < index.term_ptr[term + 1];
                 ++p) {
                similarities[index.cluster[p]] +=
                    doc_value * static_cast<acc_t>(index.weight[p]);
            }
        }
        // Strict >, ascending: ties go to the lowest cluster index, and an
        // all-zero row lands in cluster 0, matching the previous behaviour.
        size_t best_cluster = 0;
        acc_t best = similarities[0];
        for (size_t c = 1; c < n_clusters; ++c) {
            if (similarities[c] > best) {
                best = similarities[c];
                best_cluster = c;
            }
        }
        clusters[best_cluster].push_back(doc_id);
    }
}

}  // namespace

void map_docs_to_clusters(const SparseVectors* vectors,
                          const std::vector<idx_t>& docs,
                          std::vector<std::vector<idx_t>>& clusters) {
    if (vectors == nullptr) {
        throw std::runtime_error("vectors is nullptr");
    }
    if (clusters.empty() || docs.empty()) {
        return;
    }
#ifdef NSPARSE_WITH_GPU
    // GPU assignment (cuSPARSE) for float (U32) weights. It matches the CPU
    // path below (skips centroids, ties to the lowest cluster index); on any
    // failure we fall through to CPU. No AVX-512 exclusion is needed: there is
    // now a single CPU path and it skips centroids like the GPU one.
    if (vectors->get_element_size() == U32 &&
        should_offload_assignment_to_gpu(docs.size(), clusters.size())) {
        try {
            GpuClusterAssigner::instance().assign(vectors, docs, clusters);
            return;
        } catch (const std::exception& e) {
            // Fall back to CPU on any GPU error (OOM, driver, etc.).
            warn_gpu_fallback_once("assignment", e.what());
        }
    }
#endif
    const auto element_size = vectors->get_element_size();
    if (element_size == U32) {
        map_docs_to_clusters_typed<float>(vectors, docs, clusters);
    } else if (element_size == U16) {
        map_docs_to_clusters_typed<uint16_t>(vectors, docs, clusters);
    } else {
        map_docs_to_clusters_typed<uint8_t>(vectors, docs, clusters);
    }
}

}  // namespace nsparse::detail