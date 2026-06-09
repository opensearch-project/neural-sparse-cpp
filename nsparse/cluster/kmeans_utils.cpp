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
#if defined(__AVX512F__)
#include <memory>
#include <type_traits>

#include "nsparse/utils/dense_vector_matrix.h"
#endif

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

#if defined(__AVX512F__)

template <typename T>
static std::unique_ptr<DenseVectorMatrixT<T>>
initialize_cluster_representatives(
    const std::vector<std::vector<T>>& dense_centroids,
    size_t center_dimension) {
    size_t cluster_count = dense_centroids.size();
    auto representatives = std::make_unique<DenseVectorMatrixT<T>>(
        center_dimension,  // Number of dimensions
        cluster_count      // Number of clusters
    );

    // Fill the representatives matrix
    T* data = representatives->data();
    for (size_t dim = 0; dim < center_dimension; ++dim) {
        for (size_t cluster_idx = 0; cluster_idx < cluster_count;
             ++cluster_idx) {
            data[dim * cluster_count + cluster_idx] =
                dense_centroids[cluster_idx][dim];
        }
    }

    return representatives;
}

template <typename T>
static std::vector<std::vector<T>> centroids_to_dense(
    const std::vector<std::vector<idx_t>>& clusters,
    const SparseVectors* vectors) {
    std::vector<std::vector<T>> dense_centroids;
    dense_centroids.reserve(clusters.size());
    for (const auto& cluster : clusters) {
        if constexpr (std::is_same_v<T, float>) {
            const auto& dense = vectors->get_dense_vector_float(cluster.at(0));
            dense_centroids.emplace_back(dense);
        } else {
            const auto& raw_dense = vectors->get_dense_vector(cluster.at(0));
            std::vector<T> typed_dense(raw_dense.size() / sizeof(T));
            std::memcpy(typed_dense.data(), raw_dense.data(), raw_dense.size());
            dense_centroids.emplace_back(std::move(typed_dense));
        }
    }
    return dense_centroids;
}

template <typename T>
static size_t get_dense_vector_max_dimension(
    const std::vector<std::vector<T>>& dense) {
    size_t max_dimension = 0;
    for (const auto& centroid : dense) {
        max_dimension = std::max<size_t>(max_dimension, centroid.size());
    }
    return max_dimension;
}

// Unified template for map_docs_to_clusters_avx512
template <typename T>
static void map_docs_to_clusters_avx512_impl(
    const SparseVectors* vectors, const std::vector<idx_t>& docs,
    std::vector<std::vector<idx_t>>& clusters) {
    size_t n_clusters = clusters.size();
    size_t n_docs = docs.size();

    auto dense_centroids = centroids_to_dense<T>(clusters, vectors);
    size_t max_dimension = vectors->get_dimension() == 0
                               ? get_dense_vector_max_dimension(dense_centroids)
                               : vectors->get_dimension();
    size_t center_dimension = (n_clusters > 0) ? max_dimension : 0;
    auto cluster_representatives =
        initialize_cluster_representatives(dense_centroids, center_dimension);
    dense_centroids = std::vector<std::vector<T>>();

    const offset_t* indptr = vectors->indptr_data();
    const term_t* indices = vectors->indices_data();
    const T* values = reinterpret_cast<const T*>(vectors->values_data());

    for (size_t i = 0; i < n_docs; ++i) {
        idx_t doc_id = docs[i];
        const offset_t start = indptr[doc_id];
        const size_t len = indptr[doc_id + 1] - start;

        auto similarities = dot_product_sparse_matrix(
            indices + start, values + start, len, *cluster_representatives);

        size_t best_cluster = argmax_typed(similarities);
        clusters[best_cluster].push_back(doc_id);
    }
}

static void map_docs_to_clusters_avx512(
    const SparseVectors* vectors, const std::vector<idx_t>& docs,
    std::vector<std::vector<idx_t>>& clusters) {
    if (vectors == nullptr) {
        throw std::runtime_error("vectors is nullptr");
    }

    const auto element_size = vectors->get_element_size();

    if (element_size == U32) {
        map_docs_to_clusters_avx512_impl<float>(vectors, docs, clusters);
    } else if (element_size == U16) {
        map_docs_to_clusters_avx512_impl<uint16_t>(vectors, docs, clusters);
    } else {
        map_docs_to_clusters_avx512_impl<uint8_t>(vectors, docs, clusters);
    }
}

#endif

inline static float dot_product_typed_dense(const term_t* indices,
                                            const uint8_t* values,
                                            const uint8_t* dense, size_t offset,
                                            size_t len, size_t element_size) {
    if (element_size == U32) {
        // start is element index, need byte offset for float access
        const auto* float_values =
            reinterpret_cast<const float*>(values) + offset;
        const auto* float_dense = reinterpret_cast<const float*>(dense);
        return dot_product_float_dense(indices + offset, float_values, len,
                                       float_dense);
    }
    if (element_size == U16) {
        const auto* uint16_values =
            reinterpret_cast<const uint16_t*>(values) + offset;
        const auto* uint16_dense = reinterpret_cast<const uint16_t*>(dense);
        return dot_product_uint16_dense(indices + offset, uint16_values, len,
                                        uint16_dense);
    }
    return dot_product_uint8_dense(indices + offset, values + offset, len,
                                   dense);
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
    if (vectors->get_dimension() > 0) {
        const auto element_size = vectors->get_element_size();
        if (element_size == U32) {
            map_docs_to_clusters_sparse_invindex<float>(vectors, docs,
                                                        clusters);
        } else if (element_size == U16) {
            map_docs_to_clusters_sparse_invindex<uint16_t>(vectors, docs,
                                                            clusters);
        } else {
            map_docs_to_clusters_sparse_invindex<uint8_t>(vectors, docs,
                                                          clusters);
        }
    } else {
#if defined(__AVX512F__)
        map_docs_to_clusters_avx512(vectors, docs, clusters);
#else
        const auto element_size = vectors->get_element_size();
        if (element_size == U32) {
            map_docs_to_clusters_sparse_invindex<float>(vectors, docs,
                                                        clusters);
        } else if (element_size == U16) {
            map_docs_to_clusters_sparse_invindex<uint16_t>(vectors, docs,
                                                            clusters);
        } else {
            map_docs_to_clusters_sparse_invindex<uint8_t>(vectors, docs,
                                                          clusters);
        }
#endif
    }
}

}  // namespace nsparse::detail