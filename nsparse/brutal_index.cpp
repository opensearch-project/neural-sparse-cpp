/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/brutal_index.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"
#include "nsparse/utils/distance.h"
#include "nsparse/utils/ranker.h"

namespace nsparse {

BrutalIndex::BrutalIndex(int dim) : Index(dim) {}

void BrutalIndex::add(idx_t n, const idx_t* indptr, const term_t* indices,
                      const float* values) {
    throw_if_not_positive(n);
    throw_if_any_null(indptr, indices, values);

    size_t indptr_size = n + 1;
    size_t nnz = indptr[n];  // Total non-zeros
    constexpr int element_size = U32;
    if (vectors_ == nullptr) {
        vectors_ = std::unique_ptr<SparseVectors>(
            new SparseVectors({.element_size = element_size,
                               .dimension = static_cast<size_t>(dimension_)}));
    }
    vectors_->add_vectors(indptr, indptr_size, indices, nnz,
                          reinterpret_cast<const uint8_t*>(values),
                          nnz * element_size);
}

auto BrutalIndex::search(idx_t n, const idx_t* indptr, const term_t* indices,
                         const float* values, int k,
                         SearchParameters* search_parameters)
    -> pair_of_score_id_vectors_t {
    if (vectors_ == nullptr || n == 0) {
        return {std::vector<std::vector<float>>(n),
                std::vector<std::vector<idx_t>>(n)};
    }
    size_t indptr_size = n + 1;
    size_t nnz = indptr[n];  // Total non-zeros
    // Create query vectors from input
    SparseVectors query_vectors({.element_size = ElementSize::U32,
                                 .dimension = static_cast<size_t>(dimension_)});
    query_vectors.add_vectors(indptr, indptr_size, indices, nnz,
                              reinterpret_cast<const uint8_t*>(values),
                              nnz * U32);
    std::vector<std::vector<float>> result_distances(n);
    std::vector<std::vector<idx_t>> result_labels(n);
    // For each query vector
#pragma omp parallel for
    for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
        auto dense = query_vectors.get_dense_vector_float(query_idx);
        auto [distances, labels] = single_query(dense, k);
        result_distances[query_idx] = std::move(distances);
        result_labels[query_idx] = std::move(labels);
    }

    return {result_distances, result_labels};
}

auto BrutalIndex::single_query(const std::vector<float>& dense, int k)
    -> pair_of_score_id_vector_t {
    detail::DedupeTopKHolder<idx_t> holder(k);
    size_t num_docs = vectors_->num_vectors();
    if (num_docs == 0) {
        return {{}, {}};
    }

    // Hoist pointer fetches outside the loop for hot path optimization
    const auto& [indptr, indices, values] = vectors_->get_all_data();

    for (size_t i = 0; i < num_docs; ++i) {
        const idx_t start = indptr[i];
        const size_t len = indptr[i + 1] - start;
        float score = detail::dot_product_float_dense(
            indices + start, values + start, len, dense.data());
        holder.add(score, static_cast<idx_t>(i));
    }
    auto [labels, scores] =
        holder.top_k_descending_with_scores_and_padding(INVALID_IDX, -1.0F);
    return {scores, labels};
}

const SparseVectors* BrutalIndex::get_vectors() const { return vectors_.get(); }

}  // namespace nsparse