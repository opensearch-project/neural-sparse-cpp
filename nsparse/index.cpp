/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/index.h"

#include <algorithm>

#include "nsparse/types.h"
#include "nsparse/utils/checks.h"

namespace nsparse {

Index::Index(int dim) : dimension_(dim) {}

void Index::build() { throw_not_implemented(); }

void Index::search(idx_t n, const idx_t* indptr, const term_t* indices,
                   const float* values, int k, float* distances, idx_t* labels,
                   SearchParameters* search_parameters) {
    throw_if_not_positive(n);
    throw_if_not_positive(k);
    throw_if_any_null(indptr, indices, values, labels, distances);

    auto [result_distances, result_labels] =
        search(n, indptr, indices, values, k, search_parameters);

    idx_t* dest_labels = labels;
    float* dest_distances = distances;
    for (size_t i = 0; i < result_labels.size(); ++i) {
        dest_distances =
            std::ranges::copy(result_distances[i], dest_distances).out;
        dest_labels = std::ranges::copy(result_labels[i], dest_labels).out;
    }
}

auto Index::search(idx_t n, const idx_t* indptr, const term_t* indices,
                   const float* values, int k,
                   SearchParameters* search_parameters)
    -> pair_of_score_id_vectors_t {
    throw_not_implemented("search not implementted in Index");
}

void Index::add_with_ids(idx_t n, const idx_t* indptr, const term_t* indices,
                         const float* values, const idx_t* ids) {
    throw_not_implemented("add_with_ids not implemented in Index");
}

}  // namespace nsparse