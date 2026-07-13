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
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/distance_simd.h"

namespace nsparse::detail {

template <typename T>
static void map_docs_to_clusters_sparse_invindex(
    const SparseVectors* vectors, const std::vector<idx_t>& docs,
    std::vector<std::vector<idx_t>>& clusters) {
    size_t n_clusters = clusters.size();
    size_t n_docs = docs.size();

    const offset_t* indptr = vectors->indptr_data();
    const term_t* indices = vectors->indices_data();
    const auto* values = reinterpret_cast<const T*>(vectors->values_data());
    size_t dimension = vectors->get_dimension();
    if (dimension == 0) {
        term_t max_term = 0;
        for (size_t c = 0; c < n_clusters; ++c) {
            idx_t centroid_doc = clusters[c].at(0);
            const offset_t start = indptr[centroid_doc];
            const offset_t end = indptr[centroid_doc + 1];
            for (offset_t j = start; j < end; ++j) {
                max_term = std::max(max_term, indices[j]);
            }
        }
        dimension = static_cast<size_t>(max_term) + 1;
    }

    // Build flattened inverted index: contiguous arrays with offset table
    // First pass: count entries per term
    std::vector<uint32_t> term_counts(dimension, 0);
    size_t total_entries = 0;
    for (size_t c = 0; c < n_clusters; ++c) {
        idx_t centroid_doc = clusters[c].at(0);
        const offset_t start = indptr[centroid_doc];
        const offset_t end = indptr[centroid_doc + 1];
        for (offset_t j = start; j < end; ++j) {
            term_counts[indices[j]]++;
            total_entries++;
        }
    }

    // Build offset table (prefix sum)
    std::vector<uint32_t> offsets(dimension + 1);
    offsets[0] = 0;
    for (size_t i = 0; i < dimension; ++i) {
        offsets[i + 1] = offsets[i] + term_counts[i];
    }

    // Second pass: fill flattened arrays
    std::vector<uint16_t> flat_cids(total_entries);
    std::vector<float> flat_vals(total_entries);
    std::vector<uint32_t> write_pos(dimension, 0);
    for (size_t c = 0; c < n_clusters; ++c) {
        idx_t centroid_doc = clusters[c].at(0);
        const offset_t start = indptr[centroid_doc];
        const offset_t end = indptr[centroid_doc + 1];
        for (offset_t j = start; j < end; ++j) {
            term_t term = indices[j];
            uint32_t pos = offsets[term] + write_pos[term]++;
            flat_cids[pos] = static_cast<uint16_t>(c);
            float val;
            if constexpr (std::is_same_v<T, float>) {
                val = values[j];
            } else {
                val = static_cast<float>(values[j]);
            }
            flat_vals[pos] = val;
        }
    }

    // Build sorted centroid list for binary search lookup
    std::vector<idx_t> centroid_ids;
    centroid_ids.reserve(n_clusters);
    for (size_t c = 0; c < n_clusters; ++c) {
        centroid_ids.push_back(clusters[c].at(0));
    }
    std::sort(centroid_ids.begin(), centroid_ids.end());

    // Sort doc IDs for cache-friendly CSR access
    std::vector<size_t> sorted_order(n_docs);
    std::iota(sorted_order.begin(), sorted_order.end(), 0);
    std::sort(sorted_order.begin(), sorted_order.end(),
              [&docs](size_t a, size_t b) { return docs[a] < docs[b]; });

    // Assign each doc to nearest centroid via sparse-sparse dot product
    // Uses dirty-centroid tracking to avoid full zeroing and full argmax
    std::vector<float> scores(n_clusters, 0.0f);
    std::vector<uint16_t> dirty(n_clusters);
    std::vector<bool> seen(n_clusters, false);
    size_t n_dirty = 0;

    for (size_t si = 0; si < n_docs; ++si) {
        size_t orig_idx = sorted_order[si];
        idx_t doc_id = docs[orig_idx];

        if (std::binary_search(centroid_ids.begin(), centroid_ids.end(),
                               doc_id))
            continue;

        n_dirty = 0;
        const offset_t start = indptr[doc_id];
        const offset_t end = indptr[doc_id + 1];
        for (offset_t j = start; j < end; ++j) {
            term_t term = indices[j];
            float doc_val;
            if constexpr (std::is_same_v<T, float>) {
                doc_val = values[j];
            } else {
                doc_val = static_cast<float>(values[j]);
            }
            const uint32_t inv_start = offsets[term];
            const uint32_t inv_end = offsets[term + 1];
            for (uint32_t k = inv_start; k < inv_end; ++k) {
                uint16_t cid = flat_cids[k];
                if (!seen[cid]) {
                    seen[cid] = true;
                    dirty[n_dirty++] = cid;
                }
                scores[cid] += doc_val * flat_vals[k];
            }
        }

        // Partial argmax over dirty centroids only
        uint16_t best = dirty[0];
        float best_score = scores[best];
        for (size_t k = 1; k < n_dirty; ++k) {
            uint16_t cid = dirty[k];
            if (scores[cid] > best_score) {
                best_score = scores[cid];
                best = cid;
            }
        }
        clusters[best].push_back(doc_id);

        // Reset only dirty entries
        for (size_t k = 0; k < n_dirty; ++k) {
            scores[dirty[k]] = 0.0f;
            seen[dirty[k]] = false;
        }
    }
}

void map_docs_to_clusters(const SparseVectors* vectors,
                          const std::vector<idx_t>& docs,
                          std::vector<std::vector<idx_t>>& clusters) {
    if (vectors == nullptr) {
        throw std::runtime_error("vectors is nullptr");
    }
    size_t n_clusters = clusters.size();
    size_t n_docs = docs.size();
    if (n_clusters == 0 || n_docs == 0) {
        return;
    }
    const auto element_size = vectors->get_element_size();
    if (element_size == U32) {
        map_docs_to_clusters_sparse_invindex<float>(vectors, docs, clusters);
    } else if (element_size == U16) {
        map_docs_to_clusters_sparse_invindex<uint16_t>(vectors, docs, clusters);
    } else {
        map_docs_to_clusters_sparse_invindex<uint8_t>(vectors, docs, clusters);
    }
}

}  // namespace nsparse::detail