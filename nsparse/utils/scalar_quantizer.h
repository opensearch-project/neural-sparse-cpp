/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef SCALAR_QUANTIZER_H
#define SCALAR_QUANTIZER_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#endif

namespace nsparse {

/// Quantization type for scalar quantizer
enum class QuantizerType : uint8_t {
    QT_8bit,   // 8-bit quantization
    QT_16bit,  // 16-bit quantization
};

/// Scalar Quantizer using min-max quantization for sparse vector values
/// Quantizes individual float values to 8-bit or 16-bit integers
class ScalarQuantizer {
public:
    ScalarQuantizer() : qtype_(QuantizerType::QT_8bit), vmin_(0), vmax_(1) {}

    ScalarQuantizer(QuantizerType qtype, float vmin, float vmax)
        : qtype_(qtype), vmin_(vmin), vmax_(vmax) {
        if (vmax <= vmin) {
            throw std::invalid_argument("vmax must be greater than vmin");
        }
    }

    /// Returns bytes per quantized value
    [[nodiscard]] size_t bytes_per_value() const {
        return (qtype_ == QuantizerType::QT_8bit) ? 1 : 2;
    }

    /// Encode array of values
    void encode(const float* vals, uint8_t* codes, size_t n) const {
        if (qtype_ == QuantizerType::QT_8bit) {
            encode_8bit_batch(vals, codes, n);
        } else {
            auto* codes16 = reinterpret_cast<uint16_t*>(codes);
            for (size_t i = 0; i < n; i++) {
                codes16[i] = encode_16bit(vals[i]);
            }
        }
    }

    /// Decode array of values
    void decode(const uint8_t* codes, float* vals, size_t n) const {
        if (qtype_ == QuantizerType::QT_8bit) {
            for (size_t i = 0; i < n; i++) {
                vals[i] = decode_8bit(codes[i]);
            }
        } else {
            const auto* codes16 = reinterpret_cast<const uint16_t*>(codes);
            for (size_t i = 0; i < n; i++) {
                vals[i] = decode_16bit(codes16[i]);
            }
        }
    }

    QuantizerType get_quantizer_type() const { return qtype_; }
    float get_min() const { return vmin_; }
    float get_max() const { return vmax_; }

    /// Decode a quantized dot product score back to approximate original scale
    /// Uses this quantizer for ingest and query_sq for query quantization
    [[nodiscard]] float decode_dot_product(
        float quantized_score, const ScalarQuantizer& query_sq) const {
        const float ingest_range = vmax_ - vmin_;
        const float query_range = query_sq.get_max() - query_sq.get_min();
        const float max_q =
            (qtype_ == QuantizerType::QT_8bit) ? kMax8bit : kMax16bit;
        const float scale = (ingest_range * query_range) / (max_q * max_q);
        return quantized_score * scale;
    }

private:
    void encode_8bit_batch(const float* vals, uint8_t* codes, size_t n) const {
#if defined(__AVX512F__) && defined(__AVX512BW__)
        const float scale = kMax8bit / (vmax_ - vmin_);
        const __m512 v_vmin = _mm512_set1_ps(vmin_);
        const __m512 v_scale = _mm512_set1_ps(scale);
        const __m512 v_zero = _mm512_setzero_ps();
        const __m512 v_max = _mm512_set1_ps(kMax8bit);

        size_t i = 0;
        // Process 64 floats (4x16) per iteration → 64 uint8 output
        for (; i + 64 <= n; i += 64) {
            // Load and quantize 4 groups of 16 floats
            __m512 f0 = _mm512_loadu_ps(vals + i);
            __m512 f1 = _mm512_loadu_ps(vals + i + 16);
            __m512 f2 = _mm512_loadu_ps(vals + i + 32);
            __m512 f3 = _mm512_loadu_ps(vals + i + 48);

            // (val - vmin) * scale, clamped to [0, 255]
            f0 = _mm512_min_ps(v_max, _mm512_max_ps(v_zero,
                _mm512_mul_ps(_mm512_sub_ps(f0, v_vmin), v_scale)));
            f1 = _mm512_min_ps(v_max, _mm512_max_ps(v_zero,
                _mm512_mul_ps(_mm512_sub_ps(f1, v_vmin), v_scale)));
            f2 = _mm512_min_ps(v_max, _mm512_max_ps(v_zero,
                _mm512_mul_ps(_mm512_sub_ps(f2, v_vmin), v_scale)));
            f3 = _mm512_min_ps(v_max, _mm512_max_ps(v_zero,
                _mm512_mul_ps(_mm512_sub_ps(f3, v_vmin), v_scale)));

            // Convert to int32 with rounding
            __m512i i0 = _mm512_cvt_roundps_epi32(f0,
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m512i i1 = _mm512_cvt_roundps_epi32(f1,
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m512i i2 = _mm512_cvt_roundps_epi32(f2,
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m512i i3 = _mm512_cvt_roundps_epi32(f3,
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

            // Pack 32-bit → 16-bit (saturating): 16+16 → 32 per pack
            __m512i s01 = _mm512_packs_epi32(i0, i1);
            __m512i s23 = _mm512_packs_epi32(i2, i3);

            // Pack 16-bit → 8-bit (unsigned saturating): 32+32 → 64
            __m512i bytes = _mm512_packus_epi16(s01, s23);

            // The packs interleave lanes, need permutation to restore order
            // After packs_epi32(A,B): [A0..A3,B0..B3, A4..A7,B4..B7,
            //                          A8..A11,B8..B11, A12..A15,B12..B15]
            // After packus_epi16(AB,CD): complex lane interleaving
            // Use a permute to fix the order
            const __m512i perm = _mm512_set_epi32(
                15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
            bytes = _mm512_permutexvar_epi32(perm, bytes);

            _mm512_storeu_si512(codes + i, bytes);
        }

        // Process remaining 16 floats at a time
        for (; i + 16 <= n; i += 16) {
            __m512 f = _mm512_loadu_ps(vals + i);
            f = _mm512_min_ps(v_max, _mm512_max_ps(v_zero,
                _mm512_mul_ps(_mm512_sub_ps(f, v_vmin), v_scale)));
            __m512i iv = _mm512_cvt_roundps_epi32(f,
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            // Extract and store scalar (small tail, perf not critical)
            alignas(64) int32_t tmp[16];
            _mm512_store_si512(tmp, iv);
            for (int j = 0; j < 16; ++j) {
                codes[i + j] = static_cast<uint8_t>(tmp[j]);
            }
        }

        // Scalar tail
        for (; i < n; i++) {
            codes[i] = encode_8bit(vals[i]);
        }
#else
        for (size_t i = 0; i < n; i++) {
            codes[i] = encode_8bit(vals[i]);
        }
#endif
    }

    /// Encode a single float value to 8-bit
    [[nodiscard]] uint8_t encode_8bit(float val) const {
        float scaled = (val - vmin_) * (kMax8bit / (vmax_ - vmin_));
        scaled = std::max(0.0F, std::min(kMax8bit, scaled));
        return static_cast<uint8_t>(std::lround(scaled));
    }

    /// Decode 8-bit back to float
    [[nodiscard]] float decode_8bit(uint8_t code) const {
        return vmin_ + (static_cast<float>(code) * (vmax_ - vmin_) / kMax8bit);
    }

    /// Encode a single float value to 16-bit
    [[nodiscard]] uint16_t encode_16bit(float val) const {
        float scaled = (val - vmin_) * (kMax16bit / (vmax_ - vmin_));
        scaled = std::max(0.0F, std::min(kMax16bit, scaled));
        return static_cast<uint16_t>(std::lround(scaled));
    }

    /// Decode 16-bit back to float
    [[nodiscard]] float decode_16bit(uint16_t code) const {
        return vmin_ + (static_cast<float>(code) * (vmax_ - vmin_) / kMax16bit);
    }

    static constexpr float kMax8bit = std::numeric_limits<uint8_t>::max();
    static constexpr float kMax16bit = std::numeric_limits<uint16_t>::max();

    QuantizerType qtype_;
    float vmin_;  // minimum value for quantization range
    float vmax_;  // maximum value for quantization range
};

}  // namespace nsparse

#endif  // SCALAR_QUANTIZER_H
