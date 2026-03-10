/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/utils/scalar_quantizer.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

using nsparse::QuantizerType;
using nsparse::ScalarQuantizer;

// Constructor tests
TEST(ScalarQuantizer, default_constructor) {
    ScalarQuantizer sq;
    ASSERT_EQ(sq.get_quantizer_type(), QuantizerType::QT_8bit);
    ASSERT_FLOAT_EQ(sq.get_min(), 0.0F);
    ASSERT_FLOAT_EQ(sq.get_max(), 1.0F);
}

TEST(ScalarQuantizer, constructor_8bit) {
    ScalarQuantizer sq(QuantizerType::QT_8bit, -1.0F, 1.0F);
    ASSERT_EQ(sq.get_quantizer_type(), QuantizerType::QT_8bit);
    ASSERT_FLOAT_EQ(sq.get_min(), -1.0F);
    ASSERT_FLOAT_EQ(sq.get_max(), 1.0F);
}

TEST(ScalarQuantizer, constructor_16bit) {
    ScalarQuantizer sq(QuantizerType::QT_16bit, 0.0F, 100.0F);
    ASSERT_EQ(sq.get_quantizer_type(), QuantizerType::QT_16bit);
    ASSERT_FLOAT_EQ(sq.get_min(), 0.0F);
    ASSERT_FLOAT_EQ(sq.get_max(), 100.0F);
}

TEST(ScalarQuantizer, constructor_throws_when_vmax_equals_vmin) {
    ASSERT_THROW(ScalarQuantizer(QuantizerType::QT_8bit, 1.0F, 1.0F),
                 std::invalid_argument);
}

TEST(ScalarQuantizer, constructor_throws_when_vmax_less_than_vmin) {
    ASSERT_THROW(ScalarQuantizer(QuantizerType::QT_8bit, 2.0F, 1.0F),
                 std::invalid_argument);
}

// bytes_per_value tests
TEST(ScalarQuantizer, bytes_per_value_8bit) {
    ScalarQuantizer sq(QuantizerType::QT_8bit, 0.0F, 1.0F);
    ASSERT_EQ(sq.bytes_per_value(), 1);
}

TEST(ScalarQuantizer, bytes_per_value_16bit) {
    ScalarQuantizer sq(QuantizerType::QT_16bit, 0.0F, 1.0F);
    ASSERT_EQ(sq.bytes_per_value(), 2);
}

// 8-bit encode/decode tests
TEST(ScalarQuantizer, encode_decode_8bit_min_value) {
    ScalarQuantizer sq(QuantizerType::QT_8bit, 0.0F, 1.0F);
    float val = 0.0F;
    uint8_t code;
    sq.encode(&val, &code, 1);
    ASSERT_EQ(code, 0);

    float decoded;
    sq.decode(&code, &decoded, 1);
    ASSERT_FLOAT_EQ(decoded, 0.0F);
}

TEST(ScalarQuantizer, encode_decode_8bit_max_value) {
    ScalarQuantizer sq(QuantizerType::QT_8bit, 0.0F, 1.0F);
    float val = 1.0F;
    uint8_t code;
    sq.encode(&val, &code, 1);
    ASSERT_EQ(code, 255);

    float decoded;
    sq.decode(&code, &decoded, 1);
    ASSERT_FLOAT_EQ(decoded, 1.0F);
}

TEST(ScalarQuantizer, encode_decode_8bit_mid_value) {
    ScalarQuantizer sq(QuantizerType::QT_8bit, 0.0F, 1.0F);
    float val = 0.5F;
    uint8_t code;
    sq.encode(&val, &code, 1);
    // 0.5 * 255 = 127.5, rounds to 128
    ASSERT_EQ(code, 128);

    float decoded;
    sq.decode(&code, &decoded, 1);
    // 128 / 255 ≈ 0.502
    ASSERT_NEAR(decoded, 0.5F, 0.01F);
}

TEST(ScalarQuantizer, encode_8bit_clamps_below_min) {
    ScalarQuantizer sq(QuantizerType::QT_8bit, 0.0F, 1.0F);
    float val = -0.5F;
    uint8_t code;
    sq.encode(&val, &code, 1);
    ASSERT_EQ(code, 0);
}

TEST(ScalarQuantizer, encode_8bit_clamps_above_max) {
    ScalarQuantizer sq(QuantizerType::QT_8bit, 0.0F, 1.0F);
    float val = 1.5F;
    uint8_t code;
    sq.encode(&val, &code, 1);
    ASSERT_EQ(code, 255);
}

// 16-bit encode/decode tests
TEST(ScalarQuantizer, encode_decode_16bit_min_value) {
    ScalarQuantizer sq(QuantizerType::QT_16bit, 0.0F, 1.0F);
    float val = 0.0F;
    std::vector<uint8_t> codes(2);
    sq.encode(&val, codes.data(), 1);

    auto code = *reinterpret_cast<uint16_t*>(codes.data());
    ASSERT_EQ(code, 0);

    float decoded;
    sq.decode(codes.data(), &decoded, 1);
    ASSERT_FLOAT_EQ(decoded, 0.0F);
}

TEST(ScalarQuantizer, encode_decode_16bit_max_value) {
    ScalarQuantizer sq(QuantizerType::QT_16bit, 0.0F, 1.0F);
    float val = 1.0F;
    std::vector<uint8_t> codes(2);
    sq.encode(&val, codes.data(), 1);

    auto code = *reinterpret_cast<uint16_t*>(codes.data());
    ASSERT_EQ(code, 65535);

    float decoded;
    sq.decode(codes.data(), &decoded, 1);
    ASSERT_FLOAT_EQ(decoded, 1.0F);
}

TEST(ScalarQuantizer, encode_decode_16bit_mid_value) {
    ScalarQuantizer sq(QuantizerType::QT_16bit, 0.0F, 1.0F);
    float val = 0.5F;
    std::vector<uint8_t> codes(2);
    sq.encode(&val, codes.data(), 1);

    auto code = *reinterpret_cast<uint16_t*>(codes.data());
    // 0.5 * 65535 = 32767.5, rounds to 32768
    ASSERT_EQ(code, 32768);

    float decoded;
    sq.decode(codes.data(), &decoded, 1);
    ASSERT_NEAR(decoded, 0.5F, 0.001F);
}

TEST(ScalarQuantizer, encode_16bit_clamps_below_min) {
    ScalarQuantizer sq(QuantizerType::QT_16bit, 0.0F, 1.0F);
    float val = -0.5F;
    std::vector<uint8_t> codes(2);
    sq.encode(&val, codes.data(), 1);

    auto code = *reinterpret_cast<uint16_t*>(codes.data());
    ASSERT_EQ(code, 0);
}

TEST(ScalarQuantizer, encode_16bit_clamps_above_max) {
    ScalarQuantizer sq(QuantizerType::QT_16bit, 0.0F, 1.0F);
    float val = 1.5F;
    std::vector<uint8_t> codes(2);
    sq.encode(&val, codes.data(), 1);

    auto code = *reinterpret_cast<uint16_t*>(codes.data());
    ASSERT_EQ(code, 65535);
}

// Array encode/decode tests
TEST(ScalarQuantizer, encode_decode_8bit_array) {
    ScalarQuantizer sq(QuantizerType::QT_8bit, 0.0F, 1.0F);
    std::vector<float> vals = {0.0F, 0.25F, 0.5F, 0.75F, 1.0F};
    std::vector<uint8_t> codes(vals.size());

    sq.encode(vals.data(), codes.data(), vals.size());

    std::vector<float> decoded(vals.size());
    sq.decode(codes.data(), decoded.data(), vals.size());

    for (size_t i = 0; i < vals.size(); i++) {
        ASSERT_NEAR(decoded[i], vals[i], 0.01F);
    }
}

TEST(ScalarQuantizer, encode_decode_16bit_array) {
    ScalarQuantizer sq(QuantizerType::QT_16bit, 0.0F, 1.0F);
    std::vector<float> vals = {0.0F, 0.25F, 0.5F, 0.75F, 1.0F};
    std::vector<uint8_t> codes(vals.size() * 2);

    sq.encode(vals.data(), codes.data(), vals.size());

    std::vector<float> decoded(vals.size());
    sq.decode(codes.data(), decoded.data(), vals.size());

    for (size_t i = 0; i < vals.size(); i++) {
        ASSERT_NEAR(decoded[i], vals[i], 0.001F);
    }
}

// Custom range tests
TEST(ScalarQuantizer, encode_decode_8bit_negative_range) {
    ScalarQuantizer sq(QuantizerType::QT_8bit, -10.0F, 10.0F);
    std::vector<float> vals = {-10.0F, -5.0F, 0.0F, 5.0F, 10.0F};
    std::vector<uint8_t> codes(vals.size());

    sq.encode(vals.data(), codes.data(), vals.size());

    std::vector<float> decoded(vals.size());
    sq.decode(codes.data(), decoded.data(), vals.size());

    for (size_t i = 0; i < vals.size(); i++) {
        ASSERT_NEAR(decoded[i], vals[i], 0.1F);
    }
}

TEST(ScalarQuantizer, encode_decode_16bit_large_range) {
    ScalarQuantizer sq(QuantizerType::QT_16bit, -1000.0F, 1000.0F);
    std::vector<float> vals = {-1000.0F, -500.0F, 0.0F, 500.0F, 1000.0F};
    std::vector<uint8_t> codes(vals.size() * 2);

    sq.encode(vals.data(), codes.data(), vals.size());

    std::vector<float> decoded(vals.size());
    sq.decode(codes.data(), decoded.data(), vals.size());

    for (size_t i = 0; i < vals.size(); i++) {
        ASSERT_NEAR(decoded[i], vals[i], 0.1F);
    }
}

// Precision comparison: 16-bit should be more precise than 8-bit
TEST(ScalarQuantizer, precision_16bit_better_than_8bit) {
    ScalarQuantizer sq8(QuantizerType::QT_8bit, 0.0F, 1.0F);
    ScalarQuantizer sq16(QuantizerType::QT_16bit, 0.0F, 1.0F);

    float val = 0.123456F;

    uint8_t code8;
    sq8.encode(&val, &code8, 1);
    float decoded8;
    sq8.decode(&code8, &decoded8, 1);

    std::vector<uint8_t> code16(2);
    sq16.encode(&val, code16.data(), 1);
    float decoded16;
    sq16.decode(code16.data(), &decoded16, 1);

    float error8 = std::abs(decoded8 - val);
    float error16 = std::abs(decoded16 - val);

    ASSERT_LT(error16, error8);
}

// decode_dot_product tests
TEST(ScalarQuantizer, decode_dot_product_8bit_same_quantizer) {
    ScalarQuantizer sq(QuantizerType::QT_8bit, 0.0F, 1.0F);
    // For range [0,1] and 8-bit: scale = (1*1)/(255*255) = 1/65025
    float quantized_score = 65025.0F;  // max possible: 255 * 255
    float decoded = sq.decode_dot_product(quantized_score, sq);
    ASSERT_NEAR(decoded, 1.0F, 0.001F);
}

TEST(ScalarQuantizer, decode_dot_product_8bit_zero_score) {
    ScalarQuantizer sq(QuantizerType::QT_8bit, 0.0F, 1.0F);
    float decoded = sq.decode_dot_product(0.0F, sq);
    ASSERT_FLOAT_EQ(decoded, 0.0F);
}

TEST(ScalarQuantizer, decode_dot_product_8bit_different_ranges) {
    ScalarQuantizer ingest_sq(QuantizerType::QT_8bit, 0.0F, 2.0F);
    ScalarQuantizer query_sq(QuantizerType::QT_8bit, 0.0F, 4.0F);
    // scale = (2 * 4) / (255 * 255) = 8 / 65025
    float quantized_score = 65025.0F;
    float decoded = ingest_sq.decode_dot_product(quantized_score, query_sq);
    ASSERT_NEAR(decoded, 8.0F, 0.001F);
}

TEST(ScalarQuantizer, decode_dot_product_16bit_same_quantizer) {
    ScalarQuantizer sq(QuantizerType::QT_16bit, 0.0F, 1.0F);
    // For range [0,1] and 16-bit: scale = 1/(65535*65535)
    float quantized_score = 65535.0F * 65535.0F;
    float decoded = sq.decode_dot_product(quantized_score, sq);
    ASSERT_NEAR(decoded, 1.0F, 0.001F);
}

TEST(ScalarQuantizer, decode_dot_product_16bit_different_ranges) {
    ScalarQuantizer ingest_sq(QuantizerType::QT_16bit, 0.0F, 10.0F);
    ScalarQuantizer query_sq(QuantizerType::QT_16bit, 0.0F, 5.0F);
    // scale = (10 * 5) / (65535 * 65535) = 50 / 4295098225
    float quantized_score = 65535.0F * 65535.0F;
    float decoded = ingest_sq.decode_dot_product(quantized_score, query_sq);
    ASSERT_NEAR(decoded, 50.0F, 0.01F);
}

TEST(ScalarQuantizer, decode_dot_product_preserves_relative_ordering) {
    ScalarQuantizer sq(QuantizerType::QT_8bit, 0.0F, 1.0F);
    float score1 = 1000.0F;
    float score2 = 2000.0F;
    float score3 = 3000.0F;

    float decoded1 = sq.decode_dot_product(score1, sq);
    float decoded2 = sq.decode_dot_product(score2, sq);
    float decoded3 = sq.decode_dot_product(score3, sq);

    ASSERT_LT(decoded1, decoded2);
    ASSERT_LT(decoded2, decoded3);
}
