/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef DISTANCE_AVX512_H
#define DISTANCE_AVX512_H

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/dense_vector_matrix.h"
#include "nsparse/utils/prefetch.h"

namespace nsparse::detail {

// Scalar fallback for argmax
inline size_t argmax_scalar(const std::vector<float>& values) {
    if (values.empty()) {
        return 0;
    }
    size_t max_idx = 0;
    float max_val = values[0];
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] > max_val) {
            max_val = values[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// matrix is a dimension * num_clusters matrix (raw pointer version)
inline std::vector<float> dot_product_sparse_matrix(
    const term_t* indices, const float* weights, size_t len,
    const DenseVectorMatrix& matrix) {
    size_t rows =
        matrix.get_rows();  // this is actually dimension of dense vector
    size_t dimension = matrix.get_dimension();  // this is number of vectors
    size_t num_clusters = dimension;

    std::vector<float> similarities(num_clusters, 0.0F);
    for (size_t i = 0; i < len; ++i) {
        size_t dim = indices[i];
        if (dim >= rows) {
            continue;
        }
        float doc_value = weights[i];

        const float* centroid_values = matrix.data() + dim * num_clusters;

        size_t centroid_idx = 0;

        const __m512 doc_vec_512 = _mm512_set1_ps(doc_value);

        for (; centroid_idx + 32 <= num_clusters; centroid_idx += 32) {
            __m512 centroid_vec_0 =
                _mm512_loadu_ps(centroid_values + centroid_idx);
            __m512 centroid_vec_1 =
                _mm512_loadu_ps(centroid_values + centroid_idx + 16);

            __m512 current_0 = _mm512_loadu_ps(&similarities[centroid_idx]);
            __m512 current_1 =
                _mm512_loadu_ps(&similarities[centroid_idx + 16]);

            current_0 = _mm512_fmadd_ps(centroid_vec_0, doc_vec_512, current_0);
            current_1 = _mm512_fmadd_ps(centroid_vec_1, doc_vec_512, current_1);

            _mm512_storeu_ps(&similarities[centroid_idx], current_0);
            _mm512_storeu_ps(&similarities[centroid_idx + 16], current_1);
        }

        for (; centroid_idx + 16 <= num_clusters; centroid_idx += 16) {
            __m512 centroid_vec =
                _mm512_loadu_ps(centroid_values + centroid_idx);
            __m512 current = _mm512_loadu_ps(&similarities[centroid_idx]);

            current = _mm512_fmadd_ps(centroid_vec, doc_vec_512, current);

            _mm512_storeu_ps(&similarities[centroid_idx], current);
        }

        const __m256 doc_vec_256 = _mm256_set1_ps(doc_value);

        for (; centroid_idx + 8 <= num_clusters; centroid_idx += 8) {
            __m256 centroid_vec =
                _mm256_loadu_ps(centroid_values + centroid_idx);
            __m256 current = _mm256_loadu_ps(&similarities[centroid_idx]);

            current = _mm256_fmadd_ps(centroid_vec, doc_vec_256, current);

            _mm256_storeu_ps(&similarities[centroid_idx], current);
        }

        for (; centroid_idx < num_clusters; ++centroid_idx) {
            similarities[centroid_idx] +=
                doc_value * centroid_values[centroid_idx];
        }
    }

    return similarities;
}

// uint16_t version of dot_product_sparse_matrix
inline std::vector<int64_t> dot_product_sparse_matrix(
    const term_t* indices, const uint16_t* weights, size_t len,
    const DenseVectorMatrixT<uint16_t>& matrix) {
    size_t rows = matrix.get_rows();
    size_t num_clusters = matrix.get_dimension();

    std::vector<int64_t> similarities(num_clusters, 0);
    for (size_t i = 0; i < len; ++i) {
        size_t dim = indices[i];
        if (dim >= rows) {
            continue;
        }
        uint16_t doc_value = weights[i];
        const uint16_t* centroid_values = matrix.data() + dim * num_clusters;

        size_t centroid_idx = 0;

        // Process 8 uint16_t at a time using AVX2
        for (; centroid_idx + 8 <= num_clusters; centroid_idx += 8) {
            __m128i centroid_vec =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    centroid_values + centroid_idx));
            __m128i doc_vec = _mm_set1_epi16(static_cast<int16_t>(doc_value));

            // Zero-extend to 32-bit for multiplication
            __m256i centroid_32 = _mm256_cvtepu16_epi32(centroid_vec);
            __m256i doc_32 = _mm256_cvtepu16_epi32(doc_vec);

            // Multiply
            __m256i prod = _mm256_mullo_epi32(centroid_32, doc_32);

            // Load current similarities and add
            // Since similarities are int64_t, we need to handle this carefully
            for (int j = 0; j < 8; ++j) {
                similarities[centroid_idx + j] +=
                    static_cast<int64_t>(doc_value) *
                    static_cast<int64_t>(centroid_values[centroid_idx + j]);
            }
        }

        // Scalar tail
        for (; centroid_idx < num_clusters; ++centroid_idx) {
            similarities[centroid_idx] +=
                static_cast<int64_t>(doc_value) *
                static_cast<int64_t>(centroid_values[centroid_idx]);
        }
    }

    return similarities;
}

// uint8_t version of dot_product_sparse_matrix
inline std::vector<int32_t> dot_product_sparse_matrix(
    const term_t* indices, const uint8_t* weights, size_t len,
    const DenseVectorMatrixT<uint8_t>& matrix) {
    size_t rows = matrix.get_rows();
    size_t num_clusters = matrix.get_dimension();

    std::vector<int32_t> similarities(num_clusters, 0);
    for (size_t i = 0; i < len; ++i) {
        size_t dim = indices[i];
        if (dim >= rows) {
            continue;
        }
        uint8_t doc_value = weights[i];
        const uint8_t* centroid_values = matrix.data() + dim * num_clusters;

        size_t centroid_idx = 0;

        // Process 16 uint8_t at a time using AVX2
        for (; centroid_idx + 16 <= num_clusters; centroid_idx += 16) {
            __m128i centroid_vec =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    centroid_values + centroid_idx));
            __m128i doc_vec = _mm_set1_epi8(static_cast<int8_t>(doc_value));

            // Zero-extend to 16-bit
            __m256i centroid_16 = _mm256_cvtepu8_epi16(centroid_vec);
            __m256i doc_16 = _mm256_cvtepu8_epi16(doc_vec);

            // Multiply (16-bit)
            __m256i prod = _mm256_mullo_epi16(centroid_16, doc_16);

            // Extract and accumulate to 32-bit similarities
            __m128i prod_lo = _mm256_extracti128_si256(prod, 0);
            __m128i prod_hi = _mm256_extracti128_si256(prod, 1);

            // Extend to 32-bit and add (use unsigned extension since
            // uint8*uint8 can be up to 65025, exceeding signed 16-bit max)
            __m256i prod_lo_32 = _mm256_cvtepu16_epi32(prod_lo);
            __m256i prod_hi_32 = _mm256_cvtepu16_epi32(prod_hi);

            __m256i current_lo = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&similarities[centroid_idx]));
            __m256i current_hi =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    &similarities[centroid_idx + 8]));

            current_lo = _mm256_add_epi32(current_lo, prod_lo_32);
            current_hi = _mm256_add_epi32(current_hi, prod_hi_32);

            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(&similarities[centroid_idx]),
                current_lo);
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(&similarities[centroid_idx + 8]),
                current_hi);
        }

        // Scalar tail
        for (; centroid_idx < num_clusters; ++centroid_idx) {
            similarities[centroid_idx] +=
                static_cast<int32_t>(doc_value) *
                static_cast<int32_t>(centroid_values[centroid_idx]);
        }
    }

    return similarities;
}

// Templated argmax for different similarity types
template <typename T>
inline size_t argmax_typed(const std::vector<T>& values) {
    if (values.empty()) {
        return 0;
    }
    size_t max_idx = 0;
    T max_val = values[0];
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] > max_val) {
            max_val = values[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// Find the index of the maximum value in a vector using SIMD
inline size_t argmax_simd(const std::vector<float>& values) {
    if (values.empty()) {
        return 0;
    }

    size_t size = values.size();
    const float* data = values.data();

    // Handle small arrays with scalar code
    if (size < 16) {
        return argmax_scalar(values);
    }

    // SIMD processing - initialize with first 16 elements
    size_t i = 0;
    __m512 max_vec = _mm512_loadu_ps(data);
    __m512i idx_vec =
        _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m512i offset_vec = _mm512_set_epi32(31, 30, 29, 28, 27, 26, 25, 24, 23,
                                          22, 21, 20, 19, 18, 17, 16);
    __m512i increment = _mm512_set1_epi32(16);

    i = 16;

    // Process 16 floats at a time
    for (; i + 16 <= size; i += 16) {
        __m512 current = _mm512_loadu_ps(data + i);
        __mmask16 mask = _mm512_cmp_ps_mask(current, max_vec, _CMP_GT_OQ);
        max_vec = _mm512_mask_blend_ps(mask, max_vec, current);
        idx_vec = _mm512_mask_blend_epi32(mask, idx_vec, offset_vec);
        offset_vec = _mm512_add_epi32(offset_vec, increment);
    }

    // Horizontal reduction to find max and its index
    alignas(64) float max_vals[16];
    alignas(64) int indices[16];
    _mm512_store_ps(max_vals, max_vec);
    _mm512_store_si512(reinterpret_cast<__m512i*>(indices), idx_vec);

    float final_max = max_vals[0];
    size_t final_idx = indices[0];
    for (int j = 1; j < 16; ++j) {
        if (max_vals[j] > final_max) {
            final_max = max_vals[j];
            final_idx = indices[j];
        }
    }

    // Handle remaining elements
    for (; i < size; ++i) {
        if (data[i] > final_max) {
            final_max = data[i];
            final_idx = i;
        }
    }

    return final_idx;
}

// Specialization of argmax_typed for float that uses SIMD
template <>
inline size_t argmax_typed<float>(const std::vector<float>& values) {
    return argmax_simd(values);
}

// AVX512 version of dot_product_float_dense (raw pointer version for hot paths)
// Computes dot product between sparse vector (indices + weights) and dense
// vector.
inline float dot_product_float_dense(const term_t* indices,
                                     const float* weights, size_t len,
                                     const float* dense) {
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;

    // Process 16 floats at a time using AVX512
    for (; i + 16 <= len; i += 16) {
        // Load 16 uint16_t indices (256 bits) and zero-extend to 32-bit
        __m256i idx16 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices + i));
        __m512i idx = _mm512_cvtepu16_epi32(idx16);

        // Gather 16 values from dense vector using indices
        __m512 dense_vals = _mm512_i32gather_ps(idx, dense, sizeof(float));

        // Load 16 weights (aligned load - weights must be 64-byte aligned)
        __m512 weight_vals = _mm512_loadu_ps(weights + i);

        // Fused multiply-add: sum += weights * dense_vals
        sum = _mm512_fmadd_ps(weight_vals, dense_vals, sum);
    }

    // Horizontal sum of the 16 floats in sum register
    float result = _mm512_reduce_add_ps(sum);

    // Handle remaining elements with scalar code
    for (; i < len; ++i) {
        result += weights[i] * dense[indices[i]];
    }

    return result;
}

inline float dot_product_uint16_dense(const term_t* indices,
                                      const uint16_t* values, size_t len,
                                      const uint16_t* dense) {
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;

    // Process 16 elements at a time using AVX512
    for (; i + 16 <= len; i += 16) {
        // Load 16 uint16_t indices and zero-extend to 32-bit
        __m256i idx16 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices + i));
        __m512i idx = _mm512_cvtepu16_epi32(idx16);

        // Gather 16 uint16 from dense, treating as int32 with scale=2
        __m512i dense_i32 = _mm512_i32gather_epi32(idx, dense, 2);
        // Mask to keep only the low 16 bits of each 32-bit element
        dense_i32 = _mm512_and_si512(dense_i32, _mm512_set1_epi32(0xFFFF));
        // Convert to float
        __m512 dense_f = _mm512_cvtepi32_ps(dense_i32);

        // Load 16 uint16 values and convert to float
        __m256i vals_16 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values + i));
        __m512i vals_32 = _mm512_cvtepu16_epi32(vals_16);
        __m512 vals_f = _mm512_cvtepi32_ps(vals_32);

        // FMA: sum += vals * dense
        sum = _mm512_fmadd_ps(vals_f, dense_f, sum);
    }

    // Horizontal sum
    float result = _mm512_reduce_add_ps(sum);

    // Scalar tail
    for (; i < len; ++i) {
        result += static_cast<float>(values[i]) *
                  static_cast<float>(dense[indices[i]]);
    }

    return result;
}

inline float dot_product_uint8_dense(const term_t* indices,
                                     const uint8_t* values, size_t len,
                                     const uint8_t* dense) {
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;

    // Process 16 elements at a time using AVX512
    for (; i + 16 <= len; i += 16) {
        // Load 16 uint16_t indices and zero-extend to 32-bit
        __m256i idx16 =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices + i));
        __m512i idx = _mm512_cvtepu16_epi32(idx16);

        // Gather 16 bytes from dense, treating as int32 with scale=1
        // This loads 4 bytes per index but we only need the lowest byte
        __m512i dense_i32 = _mm512_i32gather_epi32(idx, dense, 1);
        // Mask to keep only the low byte of each 32-bit element
        dense_i32 = _mm512_and_si512(dense_i32, _mm512_set1_epi32(0xFF));
        // Convert to float
        __m512 dense_f = _mm512_cvtepi32_ps(dense_i32);

        // Load 16 uint8 values and convert to float
        __m128i vals_8 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(values + i));
        __m512i vals_32 = _mm512_cvtepu8_epi32(vals_8);
        __m512 vals_f = _mm512_cvtepi32_ps(vals_32);

        // FMA: sum += vals * dense
        sum = _mm512_fmadd_ps(vals_f, dense_f, sum);
    }

    // Horizontal sum
    float result = _mm512_reduce_add_ps(sum);

    // Scalar tail
    for (; i < len; ++i) {
        result += static_cast<float>(values[i]) *
                  static_cast<float>(dense[indices[i]]);
    }

    return result;
}

inline auto dot_product_float_vectors_dense(const SparseVectors* vectors,
                                            const float* dense)
    -> std::vector<float> {
    size_t n_vectors = vectors->num_vectors();
    std::vector<float> results(n_vectors, 0.0F);

    const auto& [indptr, indices, values] = vectors->get_all_data();

    for (size_t i = 0; i < n_vectors; ++i) {
        const idx_t start = indptr[i];
        const idx_t end = indptr[i + 1];
        const size_t len = end - start;

        if (i + 1 < n_vectors) {
            const idx_t next_start = indptr[i + 1];
            const size_t next_len = indptr[i + 2] - next_start;
            prefetch_vector(indices + next_start, values + next_start,
                            next_len);
        }
        results[i] = dot_product_float_dense(indices + start, values + start,
                                             len, dense);
    }
    return results;
}

inline auto dot_product_uint8_vectors_dense(const SparseVectors* vectors,
                                            const uint8_t* dense)
    -> std::vector<float> {
    size_t n_vectors = vectors->num_vectors();
    std::vector<float> results(n_vectors, 0);

    const idx_t* indptr = vectors->indptr_data();
    const term_t* indices = vectors->indices_data();
    const uint8_t* values = vectors->typed_values_data<uint8_t>();

    for (size_t i = 0; i < n_vectors; ++i) {
        const idx_t start = indptr[i];
        const idx_t end = indptr[i + 1];
        const size_t len = end - start;

        if (i + 1 < n_vectors) {
            const idx_t next_start = indptr[i + 1];
            const size_t next_len = indptr[i + 2] - next_start;
            prefetch_vector(indices + next_start, values + next_start,
                            next_len);
        }
        results[i] = dot_product_uint8_dense(indices + start, values + start,
                                             len, dense);
    }
    return results;
}

inline auto dot_product_uint16_vectors_dense(const SparseVectors* vectors,
                                             const uint16_t* dense)
    -> std::vector<float> {
    size_t n_vectors = vectors->num_vectors();
    std::vector<float> results(n_vectors, 0);

    const idx_t* indptr = vectors->indptr_data();
    const term_t* indices = vectors->indices_data();
    const uint16_t* values = vectors->typed_values_data<uint16_t>();

    for (size_t i = 0; i < n_vectors; ++i) {
        const idx_t start = indptr[i];
        const idx_t end = indptr[i + 1];
        const size_t len = end - start;

        if (i + 1 < n_vectors) {
            const idx_t next_start = indptr[i + 1];
            const size_t next_len = indptr[i + 2] - next_start;
            prefetch_vector(indices + next_start, values + next_start,
                            next_len);
        }
        results[i] = dot_product_uint16_dense(indices + start, values + start,
                                              len, dense);
    }
    return results;
}

}  // namespace nsparse::detail

#endif  // DISTANCE_AVX512_H