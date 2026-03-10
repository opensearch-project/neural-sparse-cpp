/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef EXACT_MATCH_INDEX_H
#define EXACT_MATCH_INDEX_H
#include <vector>

#include "nsparse/id_selector.h"
#include "nsparse/seismic_common.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/utils/ranker.h"

namespace nsparse::detail {

class ExactMatcher {
public:
    static auto single_query(const SparseVectors* vectors,
                             const IDSelectorEnumerable* id_selector,
                             const uint8_t* dense, size_t element_size, int k)
        -> pair_of_score_id_vector_t {
        const auto* indptr = vectors->indptr_data();
        const auto* indices = vectors->indices_data();
        const auto* values = vectors->values_data();
        detail::TopKHolder<idx_t> holder(k);
        auto ids = id_selector->ordered_ids();
        for (auto doc_id : ids) {
            auto score = detail::compute_similarity(
                doc_id, indptr, indices, values, dense, element_size);
            holder.add(score, doc_id);
        }
        auto [scores, labels] = holder.top_k_items_descending();
        scores.resize(k, -1.0F);
        labels.resize(k, INVALID_IDX);
        return {scores, labels};
    }

    static auto search(const SparseVectors* vectors,
                       const IDSelectorEnumerable* id_selector,
                       const SparseVectors* queries, size_t element_size, int k)
        -> pair_of_score_id_vectors_t {
        size_t n_queries = queries->num_vectors();
        std::vector<std::vector<float>> result_distances(n_queries);
        std::vector<std::vector<idx_t>> result_labels(n_queries);
        for (idx_t query_idx = 0; query_idx < n_queries; ++query_idx) {
            const auto& dense = queries->get_dense_vector(query_idx);
            auto [distances, labels] = single_query(
                vectors, id_selector, dense.data(), element_size, k);
            result_distances[query_idx] = std::move(distances);
            result_labels[query_idx] = std::move(labels);
        }
        return {result_distances, result_labels};
    }
};
}  // namespace nsparse::detail

#endif  // EXACT_MATCH_INDEX_H