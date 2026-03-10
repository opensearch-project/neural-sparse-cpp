/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/cluster/kmeans_utils.h"

#include <cstdint>
#include <cstring>
#include <limits>
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

    const idx_t* indptr = vectors->indptr_data();
    const term_t* indices = vectors->indices_data();
    const T* values = reinterpret_cast<const T*>(vectors->values_data());

    for (size_t i = 0; i < n_docs; ++i) {
        idx_t doc_id = docs[i];
        const idx_t start = indptr[doc_id];
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
#if defined(__AVX512F__)
    map_docs_to_clusters_avx512(vectors, docs, clusters);
    return;
#else

    const idx_t* indptr = vectors->indptr_data();
    const term_t* indices = vectors->indices_data();
    const uint8_t* values = vectors->values_data();
    const auto element_size = vectors->get_element_size();
    for (size_t i = 0; i < n_docs; ++i) {
        // get_dense_vector returns uint8_t buffer with element_size bytes per
        // value
        const auto& vec = vectors->get_dense_vector(docs[i]);
        float max_similarity = std::numeric_limits<float>::lowest();
        size_t best_cluster = 0;
        bool is_centroid = false;
        for (size_t j = 0; j < n_clusters; ++j) {
            idx_t centroid_doc = clusters[j].at(0);
            if (docs[i] == centroid_doc) {
                is_centroid = true;
                break;
            }
            const idx_t start = indptr[centroid_doc];
            const size_t len = indptr[centroid_doc + 1] - start;
            float similarity = dot_product_typed_dense(
                indices, values, vec.data(), start, len, element_size);
            if (similarity > max_similarity) {
                max_similarity = similarity;
                best_cluster = j;
            }
        }
        if (!is_centroid) {
            clusters[best_cluster].push_back(docs[i]);
        }
    }
#endif
}

}  // namespace nsparse::detail