/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/sparse_vectors.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "nsparse/io/buffered_io.h"
#include "nsparse/types.h"

// Constructor tests
TEST(SparseVectors, default_constructor) {
    nsparse::SparseVectors vectors;
    ASSERT_EQ(vectors.num_vectors(), 0);
}

TEST(SparseVectors, config_constructor) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 10});
    ASSERT_EQ(vectors.num_vectors(), 0);
    ASSERT_EQ(vectors.get_dimension(), 10);
    ASSERT_EQ(vectors.get_element_size(), nsparse::U32);
}

TEST(SparseVectors, config_constructor_throws_on_zero_dimension) {
    ASSERT_THROW(
        nsparse::SparseVectors({.element_size = nsparse::U32, .dimension = 0}),
        std::invalid_argument);
}

// add_vector tests (single vector)
TEST(SparseVectors, add_vector_single_float) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 5});
    std::vector<nsparse::term_t> indices = {0, 2, 4};
    std::vector<float> values = {1.0F, 2.0F, 3.0F};

    vectors.add_vector(indices.data(), indices.size(),
                       reinterpret_cast<const uint8_t*>(values.data()),
                       values.size() * sizeof(float));

    ASSERT_EQ(vectors.num_vectors(), 1);
}

TEST(SparseVectors, add_vector_single_uint8) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U8, .dimension = 5});
    std::vector<nsparse::term_t> indices = {1, 3};
    std::vector<uint8_t> values = {100, 200};

    vectors.add_vector(indices, values);

    ASSERT_EQ(vectors.num_vectors(), 1);
}

TEST(SparseVectors, add_vector_multiple) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 10});

    std::vector<nsparse::term_t> indices1 = {0, 1};
    std::vector<float> values1 = {1.0F, 2.0F};
    vectors.add_vector(indices1.data(), indices1.size(),
                       reinterpret_cast<const uint8_t*>(values1.data()),
                       values1.size() * sizeof(float));

    std::vector<nsparse::term_t> indices2 = {2, 3, 4};
    std::vector<float> values2 = {3.0F, 4.0F, 5.0F};
    vectors.add_vector(indices2.data(), indices2.size(),
                       reinterpret_cast<const uint8_t*>(values2.data()),
                       values2.size() * sizeof(float));

    ASSERT_EQ(vectors.num_vectors(), 2);
}

// add_vectors tests (batch)
TEST(SparseVectors, add_vectors_batch_float) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 10});

    // Two vectors: [0,1] and [2,3,4]
    std::vector<nsparse::idx_t> indptr = {0, 2, 5};
    std::vector<nsparse::term_t> indices = {0, 1, 2, 3, 4};
    std::vector<float> values = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};

    vectors.add_vectors(indptr.data(), indptr.size(), indices.data(),
                        indices.size(),
                        reinterpret_cast<const uint8_t*>(values.data()),
                        values.size() * sizeof(float));

    ASSERT_EQ(vectors.num_vectors(), 2);
}

TEST(SparseVectors, add_vectors_batch_uint8) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U8, .dimension = 10});

    std::vector<nsparse::idx_t> indptr = {0, 2, 4};
    std::vector<nsparse::term_t> indices = {0, 1, 2, 3};
    std::vector<uint8_t> values = {10, 20, 30, 40};

    vectors.add_vectors(indptr, indices, values);

    ASSERT_EQ(vectors.num_vectors(), 2);
}

TEST(SparseVectors, add_vectors_empty_indptr) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 10});

    std::vector<nsparse::idx_t> indptr = {0};  // Less than 2 elements
    std::vector<nsparse::term_t> indices = {};
    std::vector<uint8_t> values = {};

    vectors.add_vectors(indptr, indices, values);

    ASSERT_EQ(vectors.num_vectors(), 0);
}

TEST(SparseVectors, add_vectors_throws_on_size_mismatch) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 10});

    std::vector<nsparse::idx_t> indptr = {0, 2};
    std::vector<nsparse::term_t> indices = {0, 1};
    std::vector<uint8_t> values = {1, 2};  // Should be 8 bytes for 2 floats

    ASSERT_THROW(vectors.add_vectors(indptr, indices, values),
                 std::invalid_argument);
}

// get_dense_vector_float tests
TEST(SparseVectors, get_dense_vector_float_single) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 5});
    std::vector<nsparse::term_t> indices = {0, 2, 4};
    std::vector<float> values = {1.0F, 2.0F, 3.0F};
    vectors.add_vector(indices.data(), indices.size(),
                       reinterpret_cast<const uint8_t*>(values.data()),
                       values.size() * sizeof(float));

    auto dense = vectors.get_dense_vector_float(0);

    ASSERT_EQ(dense.size(), 5);
    ASSERT_FLOAT_EQ(dense[0], 1.0F);
    ASSERT_FLOAT_EQ(dense[1], 0.0F);
    ASSERT_FLOAT_EQ(dense[2], 2.0F);
    ASSERT_FLOAT_EQ(dense[3], 0.0F);
    ASSERT_FLOAT_EQ(dense[4], 3.0F);
}

TEST(SparseVectors, get_dense_vector_float_uint8_element) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U8, .dimension = 3});
    std::vector<nsparse::term_t> indices = {0, 2};
    std::vector<uint8_t> values = {100, 200};
    vectors.add_vector(indices, values);

    auto dense = vectors.get_dense_vector_float(0);

    ASSERT_EQ(dense.size(), 3);
    ASSERT_FLOAT_EQ(dense[0], 100.0F);
    ASSERT_FLOAT_EQ(dense[1], 0.0F);
    ASSERT_FLOAT_EQ(dense[2], 200.0F);
}

TEST(SparseVectors, get_dense_vector_float_uint16_element) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U16, .dimension = 3});
    std::vector<nsparse::term_t> indices = {0, 2};
    std::vector<uint16_t> values = {1000, 2000};
    vectors.add_vector(indices.data(), indices.size(),
                       reinterpret_cast<const uint8_t*>(values.data()),
                       values.size() * sizeof(uint16_t));

    auto dense = vectors.get_dense_vector_float(0);

    ASSERT_EQ(dense.size(), 3);
    ASSERT_FLOAT_EQ(dense[0], 1000.0F);
    ASSERT_FLOAT_EQ(dense[1], 0.0F);
    ASSERT_FLOAT_EQ(dense[2], 2000.0F);
}

TEST(SparseVectors, get_dense_vector_float_multiple_vectors) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 3});

    std::vector<nsparse::term_t> indices1 = {0};
    std::vector<float> values1 = {1.0F};
    vectors.add_vector(indices1.data(), indices1.size(),
                       reinterpret_cast<const uint8_t*>(values1.data()),
                       values1.size() * sizeof(float));

    std::vector<nsparse::term_t> indices2 = {1, 2};
    std::vector<float> values2 = {2.0F, 3.0F};
    vectors.add_vector(indices2.data(), indices2.size(),
                       reinterpret_cast<const uint8_t*>(values2.data()),
                       values2.size() * sizeof(float));

    auto dense0 = vectors.get_dense_vector_float(0);
    ASSERT_FLOAT_EQ(dense0[0], 1.0F);
    ASSERT_FLOAT_EQ(dense0[1], 0.0F);
    ASSERT_FLOAT_EQ(dense0[2], 0.0F);

    auto dense1 = vectors.get_dense_vector_float(1);
    ASSERT_FLOAT_EQ(dense1[0], 0.0F);
    ASSERT_FLOAT_EQ(dense1[1], 2.0F);
    ASSERT_FLOAT_EQ(dense1[2], 3.0F);
}

TEST(SparseVectors, get_dense_vector_float_out_of_range) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 5});
    std::vector<nsparse::term_t> indices = {0};
    std::vector<float> values = {1.0F};
    vectors.add_vector(indices.data(), indices.size(),
                       reinterpret_cast<const uint8_t*>(values.data()),
                       values.size() * sizeof(float));

    ASSERT_THROW(vectors.get_dense_vector_float(1), std::out_of_range);
    ASSERT_THROW(vectors.get_dense_vector_float(-1), std::out_of_range);
}

// get_dense_vector tests (raw bytes)
TEST(SparseVectors, get_dense_vector_uint8) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U8, .dimension = 4});
    std::vector<nsparse::term_t> indices = {1, 3};
    std::vector<uint8_t> values = {50, 150};
    vectors.add_vector(indices, values);

    auto dense = vectors.get_dense_vector(0);

    ASSERT_EQ(dense.size(), 4);
    ASSERT_EQ(dense[0], 0);
    ASSERT_EQ(dense[1], 50);
    ASSERT_EQ(dense[2], 0);
    ASSERT_EQ(dense[3], 150);
}

TEST(SparseVectors, get_dense_vector_uint16) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U16, .dimension = 3});
    std::vector<nsparse::term_t> indices = {0, 2};
    std::vector<uint16_t> values = {1000, 2000};
    vectors.add_vector(indices.data(), indices.size(),
                       reinterpret_cast<const uint8_t*>(values.data()),
                       values.size() * sizeof(uint16_t));

    auto dense = vectors.get_dense_vector(0);

    ASSERT_EQ(dense.size(), 3 * sizeof(uint16_t));
    const auto* typed = reinterpret_cast<const uint16_t*>(dense.data());
    ASSERT_EQ(typed[0], 1000);
    ASSERT_EQ(typed[1], 0);
    ASSERT_EQ(typed[2], 2000);
}

TEST(SparseVectors, get_dense_vector_out_of_range) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U8, .dimension = 5});
    std::vector<nsparse::term_t> indices = {0};
    std::vector<uint8_t> values = {1};
    vectors.add_vector(indices, values);

    ASSERT_THROW(vectors.get_dense_vector(1), std::out_of_range);
}

// Data accessor tests
TEST(SparseVectors, indptr_data) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 5});
    std::vector<nsparse::term_t> indices = {0, 1};
    std::vector<float> values = {1.0F, 2.0F};
    vectors.add_vector(indices.data(), indices.size(),
                       reinterpret_cast<const uint8_t*>(values.data()),
                       values.size() * sizeof(float));

    const nsparse::idx_t* indptr = vectors.indptr_data();
    ASSERT_EQ(indptr[0], 0);
    ASSERT_EQ(indptr[1], 2);
}

TEST(SparseVectors, indices_data) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 5});
    std::vector<nsparse::term_t> indices = {2, 4};
    std::vector<float> values = {1.0F, 2.0F};
    vectors.add_vector(indices.data(), indices.size(),
                       reinterpret_cast<const uint8_t*>(values.data()),
                       values.size() * sizeof(float));

    const nsparse::term_t* idx = vectors.indices_data();
    ASSERT_EQ(idx[0], 2);
    ASSERT_EQ(idx[1], 4);
}

TEST(SparseVectors, values_data_float) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 5});
    std::vector<nsparse::term_t> indices = {0, 1};
    std::vector<float> values = {1.5F, 2.5F};
    vectors.add_vector(indices.data(), indices.size(),
                       reinterpret_cast<const uint8_t*>(values.data()),
                       values.size() * sizeof(float));

    const float* vals = vectors.values_data_float();
    ASSERT_FLOAT_EQ(vals[0], 1.5F);
    ASSERT_FLOAT_EQ(vals[1], 2.5F);
}

TEST(SparseVectors, typed_values_data) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U16, .dimension = 5});
    std::vector<nsparse::term_t> indices = {0, 1};
    std::vector<uint16_t> values = {1000, 2000};
    vectors.add_vector(indices.data(), indices.size(),
                       reinterpret_cast<const uint8_t*>(values.data()),
                       values.size() * sizeof(uint16_t));

    const uint16_t* vals = vectors.typed_values_data<uint16_t>();
    ASSERT_EQ(vals[0], 1000);
    ASSERT_EQ(vals[1], 2000);
}

TEST(SparseVectors, get_all_data) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 5});
    std::vector<nsparse::term_t> indices = {0, 1};
    std::vector<float> values = {1.0F, 2.0F};
    vectors.add_vector(indices.data(), indices.size(),
                       reinterpret_cast<const uint8_t*>(values.data()),
                       values.size() * sizeof(float));

    auto data = vectors.get_all_data();
    ASSERT_NE(data.indptr_data, nullptr);
    ASSERT_NE(data.indices_data, nullptr);
    ASSERT_NE(data.values_data, nullptr);
    ASSERT_EQ(data.indptr_data[0], 0);
    ASSERT_EQ(data.indices_data[0], 0);
    ASSERT_FLOAT_EQ(data.values_data[0], 1.0F);
}

// Serialization tests
TEST(SparseVectors, serialize_deserialize_empty) {
    nsparse::SparseVectors original;

    nsparse::BufferedIOWriter writer;
    original.serialize(&writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::SparseVectors loaded;
    loaded.deserialize(&reader);

    ASSERT_EQ(loaded.num_vectors(), 0);
}

TEST(SparseVectors, serialize_deserialize_float) {
    nsparse::SparseVectors original(
        {.element_size = nsparse::U32, .dimension = 5});
    std::vector<nsparse::term_t> indices = {0, 2, 4};
    std::vector<float> values = {1.0F, 2.0F, 3.0F};
    original.add_vector(indices.data(), indices.size(),
                        reinterpret_cast<const uint8_t*>(values.data()),
                        values.size() * sizeof(float));

    nsparse::BufferedIOWriter writer;
    original.serialize(&writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::SparseVectors loaded;
    loaded.deserialize(&reader);

    ASSERT_EQ(loaded.num_vectors(), 1);
    ASSERT_EQ(loaded.get_dimension(), 5);
    ASSERT_EQ(loaded.get_element_size(), nsparse::U32);

    auto dense = loaded.get_dense_vector_float(0);
    ASSERT_FLOAT_EQ(dense[0], 1.0F);
    ASSERT_FLOAT_EQ(dense[2], 2.0F);
    ASSERT_FLOAT_EQ(dense[4], 3.0F);
}

TEST(SparseVectors, serialize_deserialize_multiple_vectors) {
    nsparse::SparseVectors original(
        {.element_size = nsparse::U8, .dimension = 4});

    std::vector<nsparse::term_t> indices1 = {0, 1};
    std::vector<uint8_t> values1 = {10, 20};
    original.add_vector(indices1, values1);

    std::vector<nsparse::term_t> indices2 = {2, 3};
    std::vector<uint8_t> values2 = {30, 40};
    original.add_vector(indices2, values2);

    nsparse::BufferedIOWriter writer;
    original.serialize(&writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::SparseVectors loaded;
    loaded.deserialize(&reader);

    ASSERT_EQ(loaded.num_vectors(), 2);

    auto dense0 = loaded.get_dense_vector(0);
    ASSERT_EQ(dense0[0], 10);
    ASSERT_EQ(dense0[1], 20);

    auto dense1 = loaded.get_dense_vector(1);
    ASSERT_EQ(dense1[2], 30);
    ASSERT_EQ(dense1[3], 40);
}

// Copy/move tests
TEST(SparseVectors, copy_constructor) {
    nsparse::SparseVectors original(
        {.element_size = nsparse::U32, .dimension = 5});
    std::vector<nsparse::term_t> indices = {0, 1};
    std::vector<float> values = {1.0F, 2.0F};
    original.add_vector(indices.data(), indices.size(),
                        reinterpret_cast<const uint8_t*>(values.data()),
                        values.size() * sizeof(float));

    nsparse::SparseVectors copy(original);

    ASSERT_EQ(copy.num_vectors(), 1);
    auto dense = copy.get_dense_vector_float(0);
    ASSERT_FLOAT_EQ(dense[0], 1.0F);
    ASSERT_FLOAT_EQ(dense[1], 2.0F);
}

TEST(SparseVectors, move_constructor) {
    nsparse::SparseVectors original(
        {.element_size = nsparse::U32, .dimension = 5});
    std::vector<nsparse::term_t> indices = {0, 1};
    std::vector<float> values = {1.0F, 2.0F};
    original.add_vector(indices.data(), indices.size(),
                        reinterpret_cast<const uint8_t*>(values.data()),
                        values.size() * sizeof(float));

    nsparse::SparseVectors moved(std::move(original));

    ASSERT_EQ(moved.num_vectors(), 1);
    auto dense = moved.get_dense_vector_float(0);
    ASSERT_FLOAT_EQ(dense[0], 1.0F);
    ASSERT_FLOAT_EQ(dense[1], 2.0F);
}
