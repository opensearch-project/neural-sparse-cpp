/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef DISTANCE_SVE_H
#define DISTANCE_SVE_H

#include <arm_sve.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/dense_vector_matrix.h"
#include "nsparse/utils/prefetch.h"

namespace nsparse::detail {

// Scalar fallback for argmax
inline size_t argmax_scalar_sve(const std::vector<float>& values) {
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
    size_t rows = matrix.get_rows();
    size_t num_clusters = matrix.get_dimension();

    std::vector<float> similarities(num_clusters, 0.0F);

    for (size_t i = 0; i < len; ++i) {
        size_t dim = indices[i];
        if (dim >= rows) {
            continue;
        }
        float doc_value = weights[i];
        const float* centroid_values = matrix.data() + dim * num_clusters;

        size_t centroid_idx = 0;
        svfloat32_t doc_vec = svdup_f32(doc_value);

        // SVE loop with predication
        while (centroid_idx < num_clusters) {
            svbool_t pg = svwhilelt_b32(centroid_idx, num_clusters);

            svfloat32_t centroid_vec =
                svld1_f32(pg, centroid_values + centroid_idx);
            svfloat32_t current = svld1_f32(pg, &similarities[centroid_idx]);

            current = svmla_f32_m(pg, current, centroid_vec, doc_vec);

            svst1_f32(pg, &similarities[centroid_idx], current);

            centroid_idx += svcntw();
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

        // Scalar fallback for int64 accumulation (SVE int64 ops are complex)
        for (size_t centroid_idx = 0; centroid_idx < num_clusters;
             ++centroid_idx) {
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

        // Convert doc_value to float for SIMD
        svfloat32_t doc_vec = svdup_f32(static_cast<float>(doc_value));

        while (centroid_idx < num_clusters) {
            svbool_t pg = svwhilelt_b32(centroid_idx, num_clusters);

            // Load uint8 values and convert to float
            svuint8_t centroid_u8 =
                svld1_u8(svwhilelt_b8(centroid_idx, num_clusters),
                         centroid_values + centroid_idx);

            // Unpack uint8 -> uint16 -> uint32 -> float (process in chunks)
            size_t vl = svcntw();
            for (size_t j = 0; j < vl && centroid_idx + j < num_clusters; ++j) {
                similarities[centroid_idx + j] +=
                    static_cast<int32_t>(doc_value) *
                    static_cast<int32_t>(centroid_values[centroid_idx + j]);
            }

            centroid_idx += vl;
        }
    }

    return similarities;
}

// Templated argmax for different similarity types
template <typename T>
inline size_t argmax_typed_sve(const std::vector<T>& values) {
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

// Find the index of the maximum value in a vector using SVE
inline size_t argmax_simd_sve(const std::vector<float>& values) {
    if (values.empty()) {
        return 0;
    }

    size_t size = values.size();
    const float* data = values.data();

    // Handle small arrays with scalar code
    size_t vl = svcntw();
    if (size < vl) {
        return argmax_scalar_sve(values);
    }

    // Initialize with first vector
    svbool_t pg_first = svwhilelt_b32(static_cast<size_t>(0), size);
    svfloat32_t max_vec = svld1_f32(pg_first, data);
    svuint32_t idx_vec = svindex_u32(0, 1);
    size_t i = vl;

    // Process vectors
    while (i < size) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t current = svld1_f32(pg, data + i);
        svuint32_t current_idx = svindex_u32(static_cast<uint32_t>(i), 1);

        // Compare and select
        svbool_t cmp = svcmpgt_f32(pg, current, max_vec);
        max_vec = svsel_f32(cmp, current, max_vec);
        idx_vec = svsel_u32(cmp, current_idx, idx_vec);

        i += vl;
    }

    // Horizontal reduction
    float final_max = svmaxv_f32(svptrue_b32(), max_vec);

    // Find the index of the max value
    svbool_t max_mask =
        svcmpeq_f32(svptrue_b32(), max_vec, svdup_f32(final_max));
    uint32_t final_idx = svminv_u32(max_mask, idx_vec);

    return static_cast<size_t>(final_idx);
}

// Specialization of argmax_typed for float that uses SIMD
template <>
inline size_t argmax_typed_sve<float>(const std::vector<float>& values) {
    return argmax_simd_sve(values);
}

// SVE version of dot_product_float_dense
// Computes dot product between sparse vector (indices + weights) and dense
// vector.
inline float dot_product_float_dense(const term_t* indices,
                                     const float* weights, size_t len,
                                     const float* dense) {
    svfloat32_t sum = svdup_f32(0.0F);
    size_t i = 0;
    size_t vl = svcntw();

    // Process using SVE gather
    while (i + vl <= len) {
        svbool_t pg = svptrue_b32();

        // Load indices (uint16_t) and zero-extend to uint32
        svuint16_t idx16 = svld1_u16(svptrue_b16(), indices + i);
        svuint32_t idx = svunpklo_u32(idx16);

        // Gather from dense vector
        svfloat32_t dense_vals = svld1_gather_u32index_f32(pg, dense, idx);

        // Load weights
        svfloat32_t weight_vals = svld1_f32(pg, weights + i);

        // FMA: sum += weights * dense_vals
        sum = svmla_f32_m(pg, sum, weight_vals, dense_vals);

        i += vl;
    }

    // Horizontal sum
    float result = svaddv_f32(svptrue_b32(), sum);

    // Handle remaining elements with scalar code
    for (; i < len; ++i) {
        result += weights[i] * dense[indices[i]];
    }

    return result;
}

// SVE version for uint16_t - convert to float for SIMD performance
// Note: SVE gather with u32index scales by 4 bytes, but we need 2-byte scaling.
// Use svld1_gather_u32offset with manually scaled offsets (idx * 2).
inline float dot_product_uint16_dense(const term_t* indices,
                                      const uint16_t* values, size_t len,
                                      const uint16_t* dense) {
    svfloat32_t sum = svdup_f32(0.0F);
    size_t i = 0;
    size_t vl = svcntw();

    while (i + vl <= len) {
        svbool_t pg = svptrue_b32();

        // Load indices and zero-extend to uint32
        svuint16_t idx16 = svld1_u16(svptrue_b16(), indices + i);
        svuint32_t idx = svunpklo_u32(idx16);

        // Scale indices by 2 to get byte offsets for uint16 elements
        svuint32_t byte_offsets = svlsl_n_u32_z(pg, idx, 1);  // idx * 2

        // Gather uint16 from dense using byte offsets
        svuint32_t dense_u32 = svld1_gather_u32offset_u32(
            pg, reinterpret_cast<const uint32_t*>(dense), byte_offsets);
        // Mask to get only lower 16 bits
        dense_u32 = svand_u32_z(pg, dense_u32, svdup_u32(0xFFFF));
        svfloat32_t dense_f = svcvt_f32_u32_z(pg, dense_u32);

        // Load uint16 values and convert to float
        svuint16_t vals_16 = svld1_u16(svptrue_b16(), values + i);
        svuint32_t vals_32 = svunpklo_u32(vals_16);
        svfloat32_t vals_f = svcvt_f32_u32_z(pg, vals_32);

        // FMA: sum += vals * dense
        sum = svmla_f32_m(pg, sum, vals_f, dense_f);

        i += vl;
    }

    // Horizontal sum
    float result = svaddv_f32(svptrue_b32(), sum);

    // Scalar tail
    for (; i < len; ++i) {
        result += static_cast<float>(values[i]) *
                  static_cast<float>(dense[indices[i]]);
    }

    return result;
}

// SVE version for uint8_t - convert to float for SIMD performance
// Note: SVE gather doesn't support byte-granularity like AVX512's scale=1,
// so we use svld1_gather_u32offset with byte offsets instead of u32index.
inline float dot_product_uint8_dense(const term_t* indices,
                                     const uint8_t* values, size_t len,
                                     const uint8_t* dense) {
    svfloat32_t sum = svdup_f32(0.0F);
    size_t i = 0;
    size_t vl = svcntw();

    while (i + vl <= len) {
        svbool_t pg = svptrue_b32();

        // Load indices and zero-extend to uint32 (these are byte offsets)
        svuint16_t idx16 = svld1_u16(svptrue_b16(), indices + i);
        svuint32_t idx = svunpklo_u32(idx16);

        // Gather uint8 from dense using byte offsets
        // svld1_gather_u32offset treats idx as byte offsets, not element
        // indices
        svuint32_t dense_u32 = svld1_gather_u32offset_u32(
            pg, reinterpret_cast<const uint32_t*>(dense), idx);
        // Mask to keep only the lowest byte (the gathered uint8 value)
        dense_u32 = svand_u32_z(pg, dense_u32, svdup_u32(0xFF));
        svfloat32_t dense_f = svcvt_f32_u32_z(pg, dense_u32);

        // Load uint8 values and convert to float
        svuint8_t vals_8 = svld1_u8(svptrue_b8(), values + i);
        svuint16_t vals_16 = svunpklo_u16(vals_8);
        svuint32_t vals_32 = svunpklo_u32(vals_16);
        svfloat32_t vals_f = svcvt_f32_u32_z(pg, vals_32);

        // FMA: sum += vals * dense
        sum = svmla_f32_m(pg, sum, vals_f, dense_f);

        i += vl;
    }

    // Horizontal sum
    float result = svaddv_f32(svptrue_b32(), sum);

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

#endif  // DISTANCE_SVE_H
