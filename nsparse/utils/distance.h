/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef DISTANCE_H
#define DISTANCE_H
#include <cstddef>
#include <vector>

#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse::detail {

/**
 * @brief sparse X dense (raw pointer version for hot paths)
 *
 * @param indices pointer to sparse indices
 * @param weights pointer to sparse weights
 * @param len number of elements
 * @param dense dense vector is expected to be of size max_index+1
 * @return float
 */
inline float dot_product_float_dense(const term_t* indices,
                                     const float* weights, size_t len,
                                     const float* dense) {
    float result = 0.0F;
    for (size_t i = 0; i < len; ++i) {
        result += weights[i] * dense[indices[i]];
    }
    return result;
}

inline float dot_product_uint8_dense(const term_t* indices,
                                     const uint8_t* weights, size_t len,
                                     const uint8_t* dense) {
    int result = 0;
    for (size_t i = 0; i < len; ++i) {
        result += weights[i] * dense[indices[i]];
    }
    return static_cast<float>(result);
}

inline float dot_product_uint16_dense(const term_t* indices,
                                      const uint16_t* weights, size_t len,
                                      const uint16_t* dense) {
    int64_t result = 0;
    for (size_t i = 0; i < len; ++i) {
        result += static_cast<int64_t>(weights[i]) * dense[indices[i]];
    }
    return static_cast<float>(result);
}

template <class T>
inline auto dot_product_vectors_dense(const SparseVectors* vectors,
                                      const T* dense) -> std::vector<float> {
    size_t n_vectors = vectors->num_vectors();
    std::vector<float> results(n_vectors, 0.0F);

    const auto* indptr = vectors->indptr_data();
    const auto* indices = vectors->indices_data();
    // values_data() returns uint8_t*, indptr stores element indices
    // so we need to access values at byte offset = element_index * sizeof(T)
    const auto* values = vectors->values_data();

    for (size_t i = 0; i < n_vectors; ++i) {
        const idx_t start = indptr[i];
        const idx_t end = indptr[i + 1];
        const size_t len = end - start;
        const term_t* idx_ptr = indices + start;
        // Cast to T* at the correct byte offset
        const T* val_ptr =
            reinterpret_cast<const T*>(values + (start * sizeof(T)));

        for (size_t j = 0; j < len; ++j) {
            auto index = idx_ptr[j];
            if (dense[index] == 0) {
                continue;
            }
            results[i] += static_cast<float>(val_ptr[j]) * dense[index];
        }
    }
    return results;
}

inline auto dot_product_float_vectors_dense(const SparseVectors* vectors,
                                            const float* dense)
    -> std::vector<float> {
    return dot_product_vectors_dense<float>(vectors, dense);
}

inline auto dot_product_uint8_vectors_dense(const SparseVectors* vectors,
                                            const uint8_t* dense)
    -> std::vector<float> {
    return dot_product_vectors_dense<uint8_t>(vectors, dense);
}

inline auto dot_product_uint16_vectors_dense(const SparseVectors* vectors,
                                             const uint16_t* dense)
    -> std::vector<float> {
    return dot_product_vectors_dense<uint16_t>(vectors, dense);
}

}  // namespace nsparse::detail

#endif  // DISTANCE_H