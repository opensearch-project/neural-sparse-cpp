/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <vector>

namespace nsparse {

using idx_t = int32_t;
using term_t = uint16_t;
using weight_t = float;

template <class T>
using pair_of_score_id_vector_t_t =
    std::pair<std::vector<float>, std::vector<T>>;
using pair_of_score_id_vector_t = pair_of_score_id_vector_t_t<idx_t>;
using pair_of_score_id_vectors_t =
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<idx_t>>>;
constexpr idx_t INVALID_IDX = -1;
}  // namespace nsparse

#endif  // TYPES_H