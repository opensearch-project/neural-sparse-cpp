/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef DISTANCE_NEON_H
#define DISTANCE_NEON_H

#include <arm_neon.h>

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
    size_t rows = matrix.get_rows();
    size_t dimension = matrix.get_dimension();
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

        const float32x4_t doc_vec = vdupq_n_f32(doc_value);

        // Process 16 floats at a time (4x NEON registers)
        for (; centroid_idx + 16 <= num_clusters; centroid_idx += 16) {
            float32x4_t centroid_vec_0 =
                vld1q_f32(centroid_values + centroid_idx);
            float32x4_t centroid_vec_1 =
                vld1q_f32(centroid_values + centroid_idx + 4);
            float32x4_t centroid_vec_2 =
                vld1q_f32(centroid_values + centroid_idx + 8);
            float32x4_t centroid_vec_3 =
                vld1q_f32(centroid_values + centroid_idx + 12);

            float32x4_t current_0 = vld1q_f32(&similarities[centroid_idx]);
            float32x4_t current_1 = vld1q_f32(&similarities[centroid_idx + 4]);
            float32x4_t current_2 = vld1q_f32(&similarities[centroid_idx + 8]);
            float32x4_t current_3 = vld1q_f32(&similarities[centroid_idx + 12]);

            current_0 = vfmaq_f32(current_0, centroid_vec_0, doc_vec);
            current_1 = vfmaq_f32(current_1, centroid_vec_1, doc_vec);
            current_2 = vfmaq_f32(current_2, centroid_vec_2, doc_vec);
            current_3 = vfmaq_f32(current_3, centroid_vec_3, doc_vec);

            vst1q_f32(&similarities[centroid_idx], current_0);
            vst1q_f32(&similarities[centroid_idx + 4], current_1);
            vst1q_f32(&similarities[centroid_idx + 8], current_2);
            vst1q_f32(&similarities[centroid_idx + 12], current_3);
        }

        // Process 4 floats at a time
        for (; centroid_idx + 4 <= num_clusters; centroid_idx += 4) {
            float32x4_t centroid_vec =
                vld1q_f32(centroid_values + centroid_idx);
            float32x4_t current = vld1q_f32(&similarities[centroid_idx]);

            current = vfmaq_f32(current, centroid_vec, doc_vec);

            vst1q_f32(&similarities[centroid_idx], current);
        }

        // Scalar tail
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

    // Use 32-bit accumulators for SIMD, convert to 64-bit at the end
    // This is safe as long as we don't overflow 32 bits during accumulation
    // Max per-element product: 65535 * 65535 = ~4.3B (fits in 32 bits)
    // We need to be careful about accumulation overflow
    std::vector<uint32_t> sims32(num_clusters, 0);

    for (size_t i = 0; i < len; ++i) {
        size_t dim = indices[i];
        if (dim >= rows) {
            continue;
        }
        uint16_t doc_value = weights[i];
        const uint16_t* centroid_values = matrix.data() + dim * num_clusters;

        size_t centroid_idx = 0;

        // Broadcast doc_value to all lanes (as 32-bit for multiplication)
        uint32x4_t doc_vec = vdupq_n_u32(doc_value);

        // Process 8 uint16_t at a time using NEON
        for (; centroid_idx + 8 <= num_clusters; centroid_idx += 8) {
            // Load 8 uint16 values
            uint16x8_t centroid_vec = vld1q_u16(centroid_values + centroid_idx);

            // Widen to 32-bit
            uint32x4_t centroid_lo = vmovl_u16(vget_low_u16(centroid_vec));
            uint32x4_t centroid_hi = vmovl_u16(vget_high_u16(centroid_vec));

            // Load current accumulators
            uint32x4_t acc_lo = vld1q_u32(&sims32[centroid_idx]);
            uint32x4_t acc_hi = vld1q_u32(&sims32[centroid_idx + 4]);

            // Multiply and accumulate
            acc_lo = vmlaq_u32(acc_lo, centroid_lo, doc_vec);
            acc_hi = vmlaq_u32(acc_hi, centroid_hi, doc_vec);

            // Store back
            vst1q_u32(&sims32[centroid_idx], acc_lo);
            vst1q_u32(&sims32[centroid_idx + 4], acc_hi);
        }

        // Scalar tail
        for (; centroid_idx < num_clusters; ++centroid_idx) {
            sims32[centroid_idx] +=
                static_cast<uint32_t>(doc_value) *
                static_cast<uint32_t>(centroid_values[centroid_idx]);
        }
    }

    // Convert to int64_t result
    std::vector<int64_t> similarities(num_clusters);
    for (size_t i = 0; i < num_clusters; ++i) {
        similarities[i] = static_cast<int64_t>(sims32[i]);
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

        uint16x8_t doc_vec_16 = vdupq_n_u16(doc_value);

        // Process 16 uint8_t at a time using NEON
        for (; centroid_idx + 16 <= num_clusters; centroid_idx += 16) {
            uint8x16_t centroid_vec = vld1q_u8(centroid_values + centroid_idx);

            // Widen to 16-bit
            uint16x8_t centroid_lo = vmovl_u8(vget_low_u8(centroid_vec));
            uint16x8_t centroid_hi = vmovl_u8(vget_high_u8(centroid_vec));

            // Multiply
            uint16x8_t prod_lo = vmulq_u16(centroid_lo, doc_vec_16);
            uint16x8_t prod_hi = vmulq_u16(centroid_hi, doc_vec_16);

            // Widen to 32-bit and accumulate
            int32x4_t current_0 = vld1q_s32(&similarities[centroid_idx]);
            int32x4_t current_1 = vld1q_s32(&similarities[centroid_idx + 4]);
            int32x4_t current_2 = vld1q_s32(&similarities[centroid_idx + 8]);
            int32x4_t current_3 = vld1q_s32(&similarities[centroid_idx + 12]);

            current_0 = vaddq_s32(
                current_0,
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(prod_lo))));
            current_1 = vaddq_s32(
                current_1,
                vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(prod_lo))));
            current_2 = vaddq_s32(
                current_2,
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(prod_hi))));
            current_3 = vaddq_s32(
                current_3,
                vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(prod_hi))));

            vst1q_s32(&similarities[centroid_idx], current_0);
            vst1q_s32(&similarities[centroid_idx + 4], current_1);
            vst1q_s32(&similarities[centroid_idx + 8], current_2);
            vst1q_s32(&similarities[centroid_idx + 12], current_3);
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

// Find the index of the maximum value in a vector using SIMD (NEON)
inline size_t argmax_simd(const std::vector<float>& values) {
    if (values.empty()) {
        return 0;
    }

    size_t size = values.size();
    const float* data = values.data();

    // Handle small arrays with scalar code
    if (size < 4) {
        return argmax_scalar(values);
    }

    // SIMD processing - initialize with first 4 elements
    size_t i = 0;
    float32x4_t max_vec = vld1q_f32(data);
    uint32x4_t idx_vec = {0, 1, 2, 3};
    uint32x4_t offset_vec = {4, 5, 6, 7};
    uint32x4_t increment = vdupq_n_u32(4);

    i = 4;

    // Process 4 floats at a time
    for (; i + 4 <= size; i += 4) {
        float32x4_t current = vld1q_f32(data + i);
        uint32x4_t cmp = vcgtq_f32(current, max_vec);
        max_vec = vbslq_f32(cmp, current, max_vec);
        idx_vec = vbslq_u32(cmp, offset_vec, idx_vec);
        offset_vec = vaddq_u32(offset_vec, increment);
    }

    // Horizontal reduction to find max and its index
    float max_vals[4];
    uint32_t indices[4];
    vst1q_f32(max_vals, max_vec);
    vst1q_u32(indices, idx_vec);

    float final_max = max_vals[0];
    size_t final_idx = indices[0];
    for (int j = 1; j < 4; ++j) {
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

// NEON version of dot_product_float_dense (raw pointer version for hot paths)
// Computes dot product between sparse vector (indices + weights) and dense
// vector.
inline float dot_product_float_dense(const term_t* indices,
                                     const float* weights, size_t len,
                                     const float* dense) {
    float32x4_t sum = vdupq_n_f32(0.0F);
    size_t i = 0;

    // Process 4 floats at a time using NEON
    for (; i + 4 <= len; i += 4) {
        // Manual gather for 4 elements
        float gathered[4];
        for (int j = 0; j < 4; ++j) {
            gathered[j] = dense[indices[i + j]];
        }

        float32x4_t dense_vals = vld1q_f32(gathered);
        float32x4_t weight_vals = vld1q_f32(weights + i);

        // Fused multiply-add: sum += weights * dense_vals
        sum = vfmaq_f32(sum, weight_vals, dense_vals);
    }

    // Horizontal sum of the 4 floats in sum register
    float32x2_t sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    sum2 = vpadd_f32(sum2, sum2);
    float result = vget_lane_f32(sum2, 0);

    // Handle remaining elements with scalar code
    for (; i < len; ++i) {
        result += weights[i] * dense[indices[i]];
    }

    return result;
}

inline float dot_product_uint16_dense(const term_t* indices,
                                      const uint16_t* values, size_t len,
                                      const uint16_t* dense) {
    // Use integer accumulation to avoid expensive float conversions
    uint64x2_t sum_lo = vdupq_n_u64(0);
    uint64x2_t sum_hi = vdupq_n_u64(0);
    size_t i = 0;

    // Process 8 elements at a time using NEON with integer math
    for (; i + 8 <= len; i += 8) {
        // Manual gather for 8 uint16 elements from dense
        uint16_t gathered[8];
        for (int j = 0; j < 8; ++j) {
            gathered[j] = dense[indices[i + j]];
        }

        // Load 8 uint16 values
        uint16x8_t vals = vld1q_u16(values + i);
        uint16x8_t dense_vals = vld1q_u16(gathered);

        // Widen to 32-bit and multiply
        uint32x4_t vals_lo = vmovl_u16(vget_low_u16(vals));
        uint32x4_t vals_hi = vmovl_u16(vget_high_u16(vals));
        uint32x4_t dense_lo = vmovl_u16(vget_low_u16(dense_vals));
        uint32x4_t dense_hi = vmovl_u16(vget_high_u16(dense_vals));

        uint32x4_t prod_lo = vmulq_u32(vals_lo, dense_lo);
        uint32x4_t prod_hi = vmulq_u32(vals_hi, dense_hi);

        // Accumulate to 64-bit to avoid overflow
        sum_lo = vaddw_u32(sum_lo, vget_low_u32(prod_lo));
        sum_lo = vaddw_u32(sum_lo, vget_high_u32(prod_lo));
        sum_hi = vaddw_u32(sum_hi, vget_low_u32(prod_hi));
        sum_hi = vaddw_u32(sum_hi, vget_high_u32(prod_hi));
    }

    // Horizontal sum
    uint64x2_t total = vaddq_u64(sum_lo, sum_hi);
    uint64_t result_int = vgetq_lane_u64(total, 0) + vgetq_lane_u64(total, 1);

    // Scalar tail
    for (; i < len; ++i) {
        result_int += static_cast<uint64_t>(values[i]) *
                      static_cast<uint64_t>(dense[indices[i]]);
    }

    return static_cast<float>(result_int);
}

inline float dot_product_uint8_dense(const term_t* indices,
                                     const uint8_t* values, size_t len,
                                     const uint8_t* dense) {
    float32x4_t sum = vdupq_n_f32(0.0F);
    size_t i = 0;

    // Process 4 elements at a time using NEON
    for (; i + 4 <= len; i += 4) {
        // Manual gather for 4 uint8 elements from dense
        float gathered[4];
        for (int j = 0; j < 4; ++j) {
            gathered[j] = static_cast<float>(dense[indices[i + j]]);
        }
        float32x4_t dense_f = vld1q_f32(gathered);

        // Load 4 uint8 values and convert to float
        uint8x8_t vals_8 = vld1_u8(values + i);
        uint16x8_t vals_16 = vmovl_u8(vals_8);
        uint32x4_t vals_32 = vmovl_u16(vget_low_u16(vals_16));
        float32x4_t vals_f = vcvtq_f32_u32(vals_32);

        // FMA: sum += vals * dense
        sum = vfmaq_f32(sum, vals_f, dense_f);
    }

    // Horizontal sum
    float32x2_t sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    sum2 = vpadd_f32(sum2, sum2);
    float result = vget_lane_f32(sum2, 0);

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

#endif  // DISTANCE_NEON_H
