/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef SEISMIC_COMMON_H
#define SEISMIC_COMMON_H

#include <memory>
#include <numeric>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/cluster/random_kmeans.h"
#include "nsparse/id_selector.h"
#include "nsparse/invlists/inverted_lists.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/distance_simd.h"

namespace nsparse {
struct SeismicClusterParameters {
    int lambda;
    int beta;
    float alpha;
};

namespace detail {

constexpr int kDefaultLambda = -1;
constexpr float kDefaultPostingPruneRatio = 0.0005F;
constexpr int kDefaultPostingMinimumLength = 160;
constexpr float kDefaultBetaRatio = 0.1F;
constexpr int kDefaultBeta = -1;
constexpr float kDefaultAlpha = 0.4F;

constexpr SeismicClusterParameters kDefaultSeismicClusterParams = {
    .lambda = kDefaultLambda, .beta = kDefaultBeta, .alpha = kDefaultAlpha};

inline std::vector<float> calculate_summary_scores(
    const size_t element_size, const SparseVectors* summaries,
    const std::vector<uint8_t>& dense) {
    std::vector<float> summary_scores;
    if (element_size == U32) {
        summary_scores = dot_product_float_vectors_dense(
            summaries, reinterpret_cast<const float*>(dense.data()));
    } else if (element_size == U16) {
        summary_scores = dot_product_uint16_vectors_dense(
            summaries, reinterpret_cast<const uint16_t*>(dense.data()));
    } else {
        summary_scores =
            dot_product_uint8_vectors_dense(summaries, dense.data());
    }
    return summary_scores;
}

inline float compute_similarity(idx_t doc_id, const idx_t* indptr,
                                const term_t* indices, const uint8_t* values,
                                const uint8_t* dense, size_t element_size) {
    const idx_t start = indptr[doc_id];
    const size_t len = indptr[doc_id + 1] - start;
    float score = 0.0F;
    if (element_size == U32) {
        const auto* float_values =
            reinterpret_cast<const float*>(values + start * sizeof(float));
        const auto* float_dense = reinterpret_cast<const float*>(dense);
        score = dot_product_float_dense(indices + start, float_values, len,
                                        float_dense);
    } else if (element_size == U16) {
        // start is element index, need to convert to byte offset for
        // uint16_t access
        const auto* int16_values = reinterpret_cast<const uint16_t*>(
            values + start * sizeof(uint16_t));
        const auto* int16_dense = reinterpret_cast<const uint16_t*>(dense);
        score = dot_product_uint16_dense(indices + start, int16_values, len,
                                         int16_dense);
    } else {
        score = dot_product_uint8_dense(indices + start, values + start, len,
                                        dense);
    }
    return score;
}

inline std::vector<size_t> reorder_clusters(
    const std::vector<float>& summary_scores, bool first_list) {
    std::vector<size_t> cluster_order(summary_scores.size());
    std::iota(cluster_order.begin(), cluster_order.end(), 0);
    if (first_list) {
        std::ranges::sort(cluster_order, [&](size_t a, size_t b) {
            return summary_scores[a] > summary_scores[b];
        });
    }
    return cluster_order;
}

inline bool should_run_exact_match(const IDSelector* id_selector, int k,
                                   const SparseVectors* queries) {
    if (id_selector == nullptr) {
        return false;
    }
    const auto* id_selector_enumerable =
        dynamic_cast<const IDSelectorEnumerable*>(id_selector);
    if (id_selector_enumerable == nullptr) {
        return false;
    }
    return id_selector_enumerable->size() <= k;
}

inline int calculate_lambda(int lambda, size_t n_vectors) {
    if (lambda == kDefaultLambda) {
        return std::max(static_cast<int>(kDefaultPostingPruneRatio *
                                         static_cast<float>(n_vectors)),
                        kDefaultPostingMinimumLength);
    }
    return lambda;
}

inline int calculate_beta(int beta, int lambda) {
    if (beta == kDefaultBeta) {
        return static_cast<int>(static_cast<float>(lambda) * kDefaultBetaRatio);
    }
    return beta;
}

inline std::vector<InvertedListClusters> build_inverted_lists_clusters(
    const SparseVectors* vectors, const SparseVectorsConfig& config,
    const SeismicClusterParameters& seismic_cluster_params) {
    // build inverted index
    std::unique_ptr<ArrayInvertedLists> inverted_lists =
        ArrayInvertedLists::build_inverted_lists(config.dimension,
                                                 config.element_size, vectors);
    int lambda =
        calculate_lambda(seismic_cluster_params.lambda, vectors->num_vectors());
    int beta = calculate_beta(seismic_cluster_params.beta, lambda);
    size_t inverted_lists_size = inverted_lists->size();
    std::vector<InvertedListClusters> clustered_inverted_lists(
        inverted_lists_size);
#pragma omp parallel for schedule(dynamic, 64)
    for (int64_t idx = 0; idx < static_cast<int64_t>(inverted_lists_size);
         ++idx) {
        auto& invlist = (*inverted_lists)[idx];
        const auto& doc_ids = invlist.prune_and_keep_doc_ids(lambda);
        InvertedListClusters inverted_list_clusters(
            detail::RandomKMeans::train(vectors, doc_ids, beta));
        inverted_list_clusters.summarize(vectors, seismic_cluster_params.alpha);
        clustered_inverted_lists[idx] = std::move(inverted_list_clusters);
        invlist.clear();
    }
    return clustered_inverted_lists;
}

}  // namespace detail
}  // namespace nsparse

#endif  // SEISMIC_COMMON_H