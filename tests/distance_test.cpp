/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/distance_simd.h"

using nsparse::idx_t;
using nsparse::SparseVectors;
using nsparse::SparseVectorsConfig;
using nsparse::term_t;
using nsparse::detail::dot_product_float_dense;
using nsparse::detail::dot_product_uint16_dense;
using nsparse::detail::dot_product_uint8_dense;

// dot_product_float_dense (raw pointer version) tests
TEST(DotProductFloatDense, empty_vector) {
    std::vector<term_t> indices;
    std::vector<float> weights;
    std::vector<float> dense(10, 1.0F);

    float result = dot_product_float_dense(indices.data(), weights.data(), 0,
                                           dense.data());

    ASSERT_FLOAT_EQ(result, 0.0F);
}

TEST(DotProductFloatDense, single_element) {
    std::vector<term_t> indices = {3};
    std::vector<float> weights = {2.0F};
    std::vector<float> dense = {0.0F, 0.0F, 0.0F, 5.0F, 0.0F};

    float result = dot_product_float_dense(indices.data(), weights.data(), 1,
                                           dense.data());

    ASSERT_FLOAT_EQ(result, 10.0F);  // 2.0 * 5.0
}

TEST(DotProductFloatDense, multiple_elements) {
    std::vector<term_t> indices = {0, 2, 4};
    std::vector<float> weights = {1.0F, 2.0F, 3.0F};
    std::vector<float> dense = {1.0F, 0.0F, 2.0F, 0.0F, 3.0F};

    float result = dot_product_float_dense(indices.data(), weights.data(), 3,
                                           dense.data());

    // 1.0*1.0 + 2.0*2.0 + 3.0*3.0 = 1 + 4 + 9 = 14
    ASSERT_FLOAT_EQ(result, 14.0F);
}

TEST(DotProductFloatDense, sparse_with_zeros_in_dense) {
    std::vector<term_t> indices = {0, 1, 2};
    std::vector<float> weights = {1.0F, 2.0F, 3.0F};
    std::vector<float> dense = {1.0F, 0.0F, 2.0F};

    float result = dot_product_float_dense(indices.data(), weights.data(), 3,
                                           dense.data());

    // 1.0*1.0 + 2.0*0.0 + 3.0*2.0 = 1 + 0 + 6 = 7
    ASSERT_FLOAT_EQ(result, 7.0F);
}

// dot_product_uint8_dense tests
TEST(DotProductUint8Dense, empty_vector) {
    std::vector<term_t> indices;
    std::vector<uint8_t> weights;
    std::vector<uint8_t> dense(10, 1);

    float result = dot_product_uint8_dense(indices.data(), weights.data(), 0,
                                           dense.data());

    ASSERT_FLOAT_EQ(result, 0.0F);
}

TEST(DotProductUint8Dense, single_element) {
    std::vector<term_t> indices = {2};
    std::vector<uint8_t> weights = {10};
    std::vector<uint8_t> dense = {0, 0, 20, 0};

    float result = dot_product_uint8_dense(indices.data(), weights.data(), 1,
                                           dense.data());

    ASSERT_FLOAT_EQ(result, 200.0F);  // 10 * 20
}

TEST(DotProductUint8Dense, multiple_elements) {
    std::vector<term_t> indices = {0, 1, 2};
    std::vector<uint8_t> weights = {1, 2, 3};
    std::vector<uint8_t> dense = {4, 5, 6};

    float result = dot_product_uint8_dense(indices.data(), weights.data(), 3,
                                           dense.data());

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    ASSERT_FLOAT_EQ(result, 32.0F);
}

TEST(DotProductUint8Dense, max_values) {
    std::vector<term_t> indices = {0, 1};
    std::vector<uint8_t> weights = {255, 255};
    std::vector<uint8_t> dense = {255, 255};

    float result = dot_product_uint8_dense(indices.data(), weights.data(), 2,
                                           dense.data());

    // 255*255 + 255*255 = 65025 + 65025 = 130050
    ASSERT_FLOAT_EQ(result, 130050.0F);
}

// dot_product_uint16_dense tests
TEST(DotProductUint16Dense, empty_vector) {
    std::vector<term_t> indices;
    std::vector<uint16_t> weights;
    std::vector<uint16_t> dense(10, 1);

    float result = dot_product_uint16_dense(indices.data(), weights.data(), 0,
                                            dense.data());

    ASSERT_FLOAT_EQ(result, 0.0F);
}

TEST(DotProductUint16Dense, single_element) {
    std::vector<term_t> indices = {1};
    std::vector<uint16_t> weights = {1000};
    std::vector<uint16_t> dense = {0, 2000};

    float result = dot_product_uint16_dense(indices.data(), weights.data(), 1,
                                            dense.data());

    ASSERT_FLOAT_EQ(result, 2000000.0F);  // 1000 * 2000
}

TEST(DotProductUint16Dense, multiple_elements) {
    std::vector<term_t> indices = {0, 1, 2};
    std::vector<uint16_t> weights = {100, 200, 300};
    std::vector<uint16_t> dense = {10, 20, 30};

    float result = dot_product_uint16_dense(indices.data(), weights.data(), 3,
                                            dense.data());

    // 100*10 + 200*20 + 300*30 = 1000 + 4000 + 9000 = 14000
    ASSERT_FLOAT_EQ(result, 14000.0F);
}

TEST(DotProductUint16Dense, large_values_no_overflow) {
    std::vector<term_t> indices = {0, 1};
    std::vector<uint16_t> weights = {65535, 65535};
    std::vector<uint16_t> dense = {65535, 65535};

    float result = dot_product_uint16_dense(indices.data(), weights.data(), 2,
                                            dense.data());

    // 65535*65535 * 2 = 4294836225 * 2 = 8589672450
    ASSERT_FLOAT_EQ(result, 8589672450.0F);
}

// dot_product_float_dense (SparseVectors version) tests
TEST(DotProductFloatDenseSparseVectors, empty_vectors) {
    SparseVectors vectors(
        SparseVectorsConfig{.element_size = 4, .dimension = 5});
    std::vector<float> dense = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};

    auto results = nsparse::detail::dot_product_float_vectors_dense(
        &vectors, dense.data());

    ASSERT_TRUE(results.empty());
}

TEST(DotProductFloatDenseSparseVectors, single_vector) {
    SparseVectors vectors(
        SparseVectorsConfig{.element_size = 4, .dimension = 5});
    std::vector<term_t> indices = {1, 3};
    std::vector<uint8_t> weights(8);
    float w1 = 2.0F, w2 = 3.0F;
    std::memcpy(weights.data(), &w1, 4);
    std::memcpy(weights.data() + 4, &w2, 4);
    vectors.add_vector(indices.data(), indices.size(), weights.data(),
                       weights.size());

    std::vector<float> dense = {0.0F, 4.0F, 0.0F, 5.0F, 0.0F};

    auto results = nsparse::detail::dot_product_float_vectors_dense(
        &vectors, dense.data());

    ASSERT_EQ(results.size(), 1);
    // 2.0*4.0 + 3.0*5.0 = 8 + 15 = 23
    ASSERT_FLOAT_EQ(results[0], 23.0F);
}

TEST(DotProductFloatDenseSparseVectors, multiple_vectors) {
    SparseVectors vectors(
        SparseVectorsConfig{.element_size = 4, .dimension = 4});

    // Vector 0: indices [0, 2], weights [1.0, 2.0]
    std::vector<term_t> indices0 = {0, 2};
    std::vector<uint8_t> weights0(8);
    float v0w0 = 1.0F, v0w1 = 2.0F;
    std::memcpy(weights0.data(), &v0w0, 4);
    std::memcpy(weights0.data() + 4, &v0w1, 4);
    vectors.add_vector(indices0.data(), indices0.size(), weights0.data(),
                       weights0.size());

    // Vector 1: indices [1, 3], weights [3.0, 4.0]
    std::vector<term_t> indices1 = {1, 3};
    std::vector<uint8_t> weights1(8);
    float v1w0 = 3.0F, v1w1 = 4.0F;
    std::memcpy(weights1.data(), &v1w0, 4);
    std::memcpy(weights1.data() + 4, &v1w1, 4);
    vectors.add_vector(indices1.data(), indices1.size(), weights1.data(),
                       weights1.size());

    std::vector<float> dense = {1.0F, 2.0F, 3.0F, 4.0F};

    auto results = nsparse::detail::dot_product_float_vectors_dense(
        &vectors, dense.data());

    ASSERT_EQ(results.size(), 2);
    // Vector 0: 1.0*1.0 + 2.0*3.0 = 1 + 6 = 7
    ASSERT_FLOAT_EQ(results[0], 7.0F);
    // Vector 1: 3.0*2.0 + 4.0*4.0 = 6 + 16 = 22
    ASSERT_FLOAT_EQ(results[1], 22.0F);
}

TEST(DotProductFloatDenseSparseVectors, skips_zero_dense_values) {
    SparseVectors vectors(
        SparseVectorsConfig{.element_size = 4, .dimension = 4});

    std::vector<term_t> indices = {0, 1, 2};
    std::vector<uint8_t> weights(12);
    float w0 = 1.0F, w1 = 2.0F, w2 = 3.0F;
    std::memcpy(weights.data(), &w0, 4);
    std::memcpy(weights.data() + 4, &w1, 4);
    std::memcpy(weights.data() + 8, &w2, 4);
    vectors.add_vector(indices.data(), indices.size(), weights.data(),
                       weights.size());

    // Dense has zero at index 1
    std::vector<float> dense = {5.0F, 0.0F, 7.0F, 0.0F};

    auto results = nsparse::detail::dot_product_float_vectors_dense(
        &vectors, dense.data());

    ASSERT_EQ(results.size(), 1);
    // 1.0*5.0 + 2.0*0.0 + 3.0*7.0 = 5 + 0 + 21 = 26
    ASSERT_FLOAT_EQ(results[0], 26.0F);
}

// dot_product_dense template tests
TEST(DotProductDenseTemplate, uint8_values) {
    SparseVectors vectors(
        SparseVectorsConfig{.element_size = 1, .dimension = 4});

    std::vector<term_t> indices = {0, 2};
    std::vector<uint8_t> weights = {10, 20};
    vectors.add_vector(indices.data(), indices.size(), weights.data(),
                       weights.size());

    std::vector<uint8_t> dense = {5, 0, 3, 0};

    auto results = nsparse::detail::dot_product_uint8_vectors_dense(
        &vectors, dense.data());

    ASSERT_EQ(results.size(), 1);
    // 10*5 + 20*3 = 50 + 60 = 110
    ASSERT_FLOAT_EQ(results[0], 110.0F);
}

TEST(DotProductDenseTemplate, uint16_values) {
    SparseVectors vectors(
        SparseVectorsConfig{.element_size = 2, .dimension = 4});

    std::vector<term_t> indices = {1, 3};
    std::vector<uint8_t> weights(4);
    uint16_t w0 = 100, w1 = 200;
    std::memcpy(weights.data(), &w0, 2);
    std::memcpy(weights.data() + 2, &w1, 2);
    vectors.add_vector(indices.data(), indices.size(), weights.data(),
                       weights.size());

    std::vector<uint16_t> dense = {0, 10, 0, 20};

    auto results = nsparse::detail::dot_product_uint16_vectors_dense(
        &vectors, dense.data());

    ASSERT_EQ(results.size(), 1);
    // 100*10 + 200*20 = 1000 + 4000 = 5000
    ASSERT_FLOAT_EQ(results[0], 5000.0F);
}

TEST(DotProductDenseTemplate, skips_zero_dense_values) {
    SparseVectors vectors(
        SparseVectorsConfig{.element_size = 1, .dimension = 4});

    std::vector<term_t> indices = {0, 1, 2};
    std::vector<uint8_t> weights = {1, 2, 3};
    vectors.add_vector(indices.data(), indices.size(), weights.data(),
                       weights.size());

    std::vector<uint8_t> dense = {10, 0, 30, 0};

    auto results = nsparse::detail::dot_product_uint8_vectors_dense(
        &vectors, dense.data());

    ASSERT_EQ(results.size(), 1);
    // 1*10 + 2*0 + 3*30 = 10 + 0 + 90 = 100
    ASSERT_FLOAT_EQ(results[0], 100.0F);
}

TEST(DotProductDenseTemplate, multiple_vectors) {
    SparseVectors vectors(
        SparseVectorsConfig{.element_size = 1, .dimension = 3});

    std::vector<term_t> indices0 = {0, 1};
    std::vector<uint8_t> weights0 = {1, 2};
    vectors.add_vector(indices0.data(), indices0.size(), weights0.data(),
                       weights0.size());

    std::vector<term_t> indices1 = {1, 2};
    std::vector<uint8_t> weights1 = {3, 4};
    vectors.add_vector(indices1.data(), indices1.size(), weights1.data(),
                       weights1.size());

    std::vector<uint8_t> dense = {10, 20, 30};

    auto results = nsparse::detail::dot_product_uint8_vectors_dense(
        &vectors, dense.data());

    ASSERT_EQ(results.size(), 2);
    // Vector 0: 1*10 + 2*20 = 10 + 40 = 50
    ASSERT_FLOAT_EQ(results[0], 50.0F);
    // Vector 1: 3*20 + 4*30 = 60 + 120 = 180
    ASSERT_FLOAT_EQ(results[1], 180.0F);
}
