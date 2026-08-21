/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/invlists/inverted_lists.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "nsparse/io/buffered_io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/mmap_cursor.h"

namespace {
// A list's doc ids, copied out: the stored buffer is read-only, and the callers
// below sort what they get.
std::vector<nsparse::idx_t> doc_ids_of(const nsparse::InvertedList& list) {
    const auto& doc_ids = list.get_doc_ids();
    return {doc_ids.begin(), doc_ids.end()};
}
}  // namespace

// InvertedList tests
TEST(InvertedList, constructor) {
    nsparse::InvertedList list(nsparse::U32);
    ASSERT_TRUE(list.get_doc_ids().empty());
    ASSERT_TRUE(list.get_codes().empty());
}

TEST(InvertedList, add_entries_empty) {
    nsparse::InvertedList list(nsparse::U32);
    list.add_entries(0, nullptr, nullptr);
    ASSERT_TRUE(list.get_doc_ids().empty());
    ASSERT_TRUE(list.get_codes().empty());
}

TEST(InvertedList, add_entries_single_float) {
    nsparse::InvertedList list(nsparse::U32);
    nsparse::idx_t doc_id = 42;
    float value = 1.5F;
    list.add_entries(1, &doc_id, reinterpret_cast<const uint8_t*>(&value));

    ASSERT_EQ(list.get_doc_ids().size(), 1);
    ASSERT_EQ(list.get_doc_ids()[0], 42);
    ASSERT_EQ(list.get_codes().size(), sizeof(float));

    float stored = *reinterpret_cast<const float*>(list.get_codes().data());
    ASSERT_FLOAT_EQ(stored, 1.5F);
}

TEST(InvertedList, add_entries_multiple_float) {
    nsparse::InvertedList list(nsparse::U32);
    std::vector<nsparse::idx_t> doc_ids = {1, 2, 3};
    std::vector<float> values = {1.0F, 2.0F, 3.0F};

    list.add_entries(3, doc_ids.data(),
                     reinterpret_cast<const uint8_t*>(values.data()));

    ASSERT_EQ(list.get_doc_ids().size(), 3);
    ASSERT_EQ(list.get_doc_ids()[0], 1);
    ASSERT_EQ(list.get_doc_ids()[1], 2);
    ASSERT_EQ(list.get_doc_ids()[2], 3);

    ASSERT_EQ(list.get_codes().size(), 3 * sizeof(float));
    const auto* stored =
        reinterpret_cast<const float*>(list.get_codes().data());
    ASSERT_FLOAT_EQ(stored[0], 1.0F);
    ASSERT_FLOAT_EQ(stored[1], 2.0F);
    ASSERT_FLOAT_EQ(stored[2], 3.0F);
}

TEST(InvertedList, add_entries_uint8) {
    nsparse::InvertedList list(nsparse::U8);
    std::vector<nsparse::idx_t> doc_ids = {10, 20};
    std::vector<uint8_t> values = {100, 200};

    list.add_entries(2, doc_ids.data(), values.data());

    ASSERT_EQ(list.get_doc_ids().size(), 2);
    ASSERT_EQ(list.get_codes().size(), 2);
    ASSERT_EQ(list.get_codes()[0], 100);
    ASSERT_EQ(list.get_codes()[1], 200);
}

TEST(InvertedList, add_entries_uint16) {
    nsparse::InvertedList list(nsparse::U16);
    std::vector<nsparse::idx_t> doc_ids = {5, 6};
    std::vector<uint16_t> values = {1000, 2000};

    list.add_entries(2, doc_ids.data(),
                     reinterpret_cast<const uint8_t*>(values.data()));

    ASSERT_EQ(list.get_doc_ids().size(), 2);
    ASSERT_EQ(list.get_codes().size(), 2 * sizeof(uint16_t));

    const auto* stored =
        reinterpret_cast<const uint16_t*>(list.get_codes().data());
    ASSERT_EQ(stored[0], 1000);
    ASSERT_EQ(stored[1], 2000);
}

TEST(InvertedList, add_entries_accumulates) {
    nsparse::InvertedList list(nsparse::U32);

    nsparse::idx_t doc1 = 1;
    float val1 = 1.0F;
    list.add_entries(1, &doc1, reinterpret_cast<const uint8_t*>(&val1));

    nsparse::idx_t doc2 = 2;
    float val2 = 2.0F;
    list.add_entries(1, &doc2, reinterpret_cast<const uint8_t*>(&val2));

    ASSERT_EQ(list.get_doc_ids().size(), 2);
    ASSERT_EQ(list.get_doc_ids()[0], 1);
    ASSERT_EQ(list.get_doc_ids()[1], 2);
}

TEST(InvertedList, clear) {
    nsparse::InvertedList list(nsparse::U32);
    std::vector<nsparse::idx_t> doc_ids = {1, 2, 3};
    std::vector<float> values = {1.0F, 2.0F, 3.0F};
    list.add_entries(3, doc_ids.data(),
                     reinterpret_cast<const uint8_t*>(values.data()));

    list.clear();

    ASSERT_TRUE(list.get_doc_ids().empty());
    ASSERT_TRUE(list.get_codes().empty());
}

TEST(InvertedList, prune_and_keep_doc_ids_empty_list) {
    nsparse::InvertedList list(nsparse::U32);
    auto result = list.prune_and_keep_doc_ids(5);
    ASSERT_TRUE(result.empty());
}

TEST(InvertedList, prune_and_keep_doc_ids_lambda_zero) {
    nsparse::InvertedList list(nsparse::U32);
    std::vector<nsparse::idx_t> doc_ids = {1, 2, 3};
    std::vector<float> values = {1.0F, 2.0F, 3.0F};
    list.add_entries(3, doc_ids.data(),
                     reinterpret_cast<const uint8_t*>(values.data()));

    auto result = list.prune_and_keep_doc_ids(0);
    ASSERT_EQ(result.size(), 3);  // Returns all when lambda <= 0
}

TEST(InvertedList, prune_and_keep_doc_ids_lambda_exceeds_size) {
    nsparse::InvertedList list(nsparse::U32);
    std::vector<nsparse::idx_t> doc_ids = {1, 2};
    std::vector<float> values = {1.0F, 2.0F};
    list.add_entries(2, doc_ids.data(),
                     reinterpret_cast<const uint8_t*>(values.data()));

    auto result = list.prune_and_keep_doc_ids(10);
    ASSERT_EQ(result.size(), 2);  // Returns all when lambda >= n_docs
}

TEST(InvertedList, prune_and_keep_doc_ids_keeps_top_values_float) {
    nsparse::InvertedList list(nsparse::U32);
    // doc_ids: 10, 20, 30, 40 with values: 1.0, 4.0, 2.0, 3.0
    // Top 2 by value: doc 20 (4.0), doc 40 (3.0)
    std::vector<nsparse::idx_t> doc_ids = {10, 20, 30, 40};
    std::vector<float> values = {1.0F, 4.0F, 2.0F, 3.0F};
    list.add_entries(4, doc_ids.data(),
                     reinterpret_cast<const uint8_t*>(values.data()));

    auto result = list.prune_and_keep_doc_ids(2);

    ASSERT_EQ(result.size(), 2);
    // Should contain doc 20 and doc 40 (highest values)
    std::ranges::sort(result);
    ASSERT_EQ(result[0], 20);
    ASSERT_EQ(result[1], 40);
}

TEST(InvertedList, prune_and_keep_doc_ids_keeps_top_values_uint8) {
    nsparse::InvertedList list(nsparse::U8);
    // doc_ids: 1, 2, 3 with values: 50, 200, 100
    // Top 2 by value: doc 2 (200), doc 3 (100)
    std::vector<nsparse::idx_t> doc_ids = {1, 2, 3};
    std::vector<uint8_t> values = {50, 200, 100};
    list.add_entries(3, doc_ids.data(), values.data());

    auto result = list.prune_and_keep_doc_ids(2);

    ASSERT_EQ(result.size(), 2);
    std::ranges::sort(result);
    ASSERT_EQ(result[0], 2);
    ASSERT_EQ(result[1], 3);
}

TEST(InvertedList, prune_and_keep_doc_ids_keeps_top_values_uint16) {
    nsparse::InvertedList list(nsparse::U16);
    // doc_ids: 1, 2, 3 with values: 500, 2000, 1000
    // Top 2 by value: doc 2 (2000), doc 3 (1000)
    std::vector<nsparse::idx_t> doc_ids = {1, 2, 3};
    std::vector<uint16_t> values = {500, 2000, 1000};
    list.add_entries(3, doc_ids.data(),
                     reinterpret_cast<const uint8_t*>(values.data()));

    auto result = list.prune_and_keep_doc_ids(2);

    ASSERT_EQ(result.size(), 2);
    std::ranges::sort(result);
    ASSERT_EQ(result[0], 2);
    ASSERT_EQ(result[1], 3);
}

TEST(InvertedList, move_constructor) {
    nsparse::InvertedList list(nsparse::U32);
    nsparse::idx_t doc_id = 42;
    float value = 1.5F;
    list.add_entries(1, &doc_id, reinterpret_cast<const uint8_t*>(&value));

    nsparse::InvertedList moved(std::move(list));

    ASSERT_EQ(moved.get_doc_ids().size(), 1);
    ASSERT_EQ(moved.get_doc_ids()[0], 42);
}

// ArrayInvertedLists tests
TEST(ArrayInvertedLists, constructor) {
    nsparse::ArrayInvertedLists lists(10, nsparse::U32);
    ASSERT_EQ(lists.get_n_term(), 10);
    ASSERT_EQ(lists.get_element_size(), nsparse::U32);
    ASSERT_EQ(lists.size(), 10);
}

TEST(ArrayInvertedLists, add_entries_single_term) {
    nsparse::ArrayInvertedLists lists(5, nsparse::U32);
    std::vector<nsparse::idx_t> doc_ids = {1, 2, 3};
    std::vector<float> values = {1.0F, 2.0F, 3.0F};

    lists.add_entries(2, 3, doc_ids.data(),
                      reinterpret_cast<const uint8_t*>(values.data()));

    ASSERT_EQ(lists[2].get_doc_ids().size(), 3);
    ASSERT_EQ(lists[0].get_doc_ids().size(), 0);
    ASSERT_EQ(lists[1].get_doc_ids().size(), 0);
}

TEST(ArrayInvertedLists, add_entries_multiple_terms) {
    nsparse::ArrayInvertedLists lists(3, nsparse::U32);

    nsparse::idx_t doc1 = 10;
    float val1 = 1.0F;
    lists.add_entries(0, 1, &doc1, reinterpret_cast<const uint8_t*>(&val1));

    nsparse::idx_t doc2 = 20;
    float val2 = 2.0F;
    lists.add_entries(1, 1, &doc2, reinterpret_cast<const uint8_t*>(&val2));

    nsparse::idx_t doc3 = 30;
    float val3 = 3.0F;
    lists.add_entries(2, 1, &doc3, reinterpret_cast<const uint8_t*>(&val3));

    ASSERT_EQ(lists[0].get_doc_ids()[0], 10);
    ASSERT_EQ(lists[1].get_doc_ids()[0], 20);
    ASSERT_EQ(lists[2].get_doc_ids()[0], 30);
}

TEST(ArrayInvertedLists, add_entries_out_of_range_throws) {
    nsparse::ArrayInvertedLists lists(5, nsparse::U32);
    nsparse::idx_t doc_id = 1;
    float value = 1.0F;

    ASSERT_THROW(lists.add_entries(5, 1, &doc_id,
                                   reinterpret_cast<const uint8_t*>(&value)),
                 std::invalid_argument);

    ASSERT_THROW(lists.add_entries(100, 1, &doc_id,
                                   reinterpret_cast<const uint8_t*>(&value)),
                 std::invalid_argument);
}

TEST(ArrayInvertedLists, add_entry_single) {
    nsparse::ArrayInvertedLists lists(3, nsparse::U32);
    float value = 5.0F;

    lists.add_entry(1, 42, reinterpret_cast<const uint8_t*>(&value));

    ASSERT_EQ(lists[1].get_doc_ids().size(), 1);
    ASSERT_EQ(lists[1].get_doc_ids()[0], 42);
}

TEST(ArrayInvertedLists, operator_bracket_const) {
    nsparse::ArrayInvertedLists lists(3, nsparse::U32);
    nsparse::idx_t doc_id = 1;
    float value = 1.0F;
    lists.add_entries(0, 1, &doc_id, reinterpret_cast<const uint8_t*>(&value));

    const auto& const_lists = lists;
    ASSERT_EQ(const_lists[0].get_doc_ids().size(), 1);
}

TEST(ArrayInvertedLists, iterator) {
    nsparse::ArrayInvertedLists lists(3, nsparse::U32);

    int count = 0;
    for (auto& list : lists) {
        (void)list;
        count++;
    }
    ASSERT_EQ(count, 3);
}

TEST(ArrayInvertedLists, const_iterator) {
    nsparse::ArrayInvertedLists lists(3, nsparse::U32);

    const auto& const_lists = lists;
    int count = 0;
    for (const auto& list : const_lists) {
        (void)list;
        count++;
    }
    ASSERT_EQ(count, 3);
}

// build_inverted_lists tests
TEST(ArrayInvertedLists, build_inverted_lists_empty_vectors) {
    nsparse::SparseVectorsConfig config{.element_size = nsparse::U32,
                                        .dimension = 10};
    nsparse::SparseVectors vectors(config);

    auto invlists = nsparse::ArrayInvertedLists::build_inverted_lists(
        10, nsparse::U32, &vectors);

    ASSERT_NE(invlists, nullptr);
    ASSERT_EQ(invlists->get_n_term(), 10);
    ASSERT_EQ(invlists->get_element_size(), nsparse::U32);
    for (size_t i = 0; i < 10; ++i) {
        ASSERT_TRUE((*invlists)[i].get_doc_ids().empty());
    }
}

TEST(ArrayInvertedLists, build_inverted_lists_single_doc_single_term) {
    nsparse::SparseVectorsConfig config{.element_size = nsparse::U32,
                                        .dimension = 5};
    nsparse::SparseVectors vectors(config);

    std::vector<nsparse::term_t> indices = {2};
    float value = 3.5F;
    std::vector<uint8_t> weights(
        reinterpret_cast<uint8_t*>(&value),
        reinterpret_cast<uint8_t*>(&value) + sizeof(float));
    vectors.add_vector(indices, weights);

    auto invlists = nsparse::ArrayInvertedLists::build_inverted_lists(
        5, nsparse::U32, &vectors);

    ASSERT_EQ(invlists->size(), 5);
    // Only term 2 should have an entry
    ASSERT_EQ((*invlists)[2].get_doc_ids().size(), 1);
    ASSERT_EQ((*invlists)[2].get_doc_ids()[0], 0);  // doc_id = 0

    // Other terms should be empty
    ASSERT_TRUE((*invlists)[0].get_doc_ids().empty());
    ASSERT_TRUE((*invlists)[1].get_doc_ids().empty());
    ASSERT_TRUE((*invlists)[3].get_doc_ids().empty());
    ASSERT_TRUE((*invlists)[4].get_doc_ids().empty());
}

TEST(ArrayInvertedLists, build_inverted_lists_single_doc_multiple_terms) {
    nsparse::SparseVectorsConfig config{.element_size = nsparse::U32,
                                        .dimension = 5};
    nsparse::SparseVectors vectors(config);

    std::vector<nsparse::term_t> indices = {0, 2, 4};
    std::vector<float> values = {1.0F, 2.0F, 3.0F};
    std::vector<uint8_t> weights(reinterpret_cast<uint8_t*>(values.data()),
                                 reinterpret_cast<uint8_t*>(values.data()) +
                                     values.size() * sizeof(float));
    vectors.add_vector(indices, weights);

    auto invlists = nsparse::ArrayInvertedLists::build_inverted_lists(
        5, nsparse::U32, &vectors);

    // Terms 0, 2, 4 should have doc 0
    ASSERT_EQ((*invlists)[0].get_doc_ids().size(), 1);
    ASSERT_EQ((*invlists)[0].get_doc_ids()[0], 0);
    ASSERT_EQ((*invlists)[2].get_doc_ids().size(), 1);
    ASSERT_EQ((*invlists)[2].get_doc_ids()[0], 0);
    ASSERT_EQ((*invlists)[4].get_doc_ids().size(), 1);
    ASSERT_EQ((*invlists)[4].get_doc_ids()[0], 0);

    // Terms 1, 3 should be empty
    ASSERT_TRUE((*invlists)[1].get_doc_ids().empty());
    ASSERT_TRUE((*invlists)[3].get_doc_ids().empty());
}

TEST(ArrayInvertedLists, build_inverted_lists_multiple_docs_same_term) {
    nsparse::SparseVectorsConfig config{.element_size = nsparse::U32,
                                        .dimension = 3};
    nsparse::SparseVectors vectors(config);

    // Doc 0: term 1
    std::vector<nsparse::term_t> indices1 = {1};
    float val1 = 1.0F;
    std::vector<uint8_t> weights1(
        reinterpret_cast<uint8_t*>(&val1),
        reinterpret_cast<uint8_t*>(&val1) + sizeof(float));
    vectors.add_vector(indices1, weights1);

    // Doc 1: term 1
    std::vector<nsparse::term_t> indices2 = {1};
    float val2 = 2.0F;
    std::vector<uint8_t> weights2(
        reinterpret_cast<uint8_t*>(&val2),
        reinterpret_cast<uint8_t*>(&val2) + sizeof(float));
    vectors.add_vector(indices2, weights2);

    // Doc 2: term 1
    std::vector<nsparse::term_t> indices3 = {1};
    float val3 = 3.0F;
    std::vector<uint8_t> weights3(
        reinterpret_cast<uint8_t*>(&val3),
        reinterpret_cast<uint8_t*>(&val3) + sizeof(float));
    vectors.add_vector(indices3, weights3);

    auto invlists = nsparse::ArrayInvertedLists::build_inverted_lists(
        3, nsparse::U32, &vectors);

    // Term 1 should have all 3 docs
    ASSERT_EQ((*invlists)[1].get_doc_ids().size(), 3);
    auto doc_ids = doc_ids_of((*invlists)[1]);
    std::ranges::sort(doc_ids);
    ASSERT_EQ(doc_ids[0], 0);
    ASSERT_EQ(doc_ids[1], 1);
    ASSERT_EQ(doc_ids[2], 2);

    // Other terms should be empty
    ASSERT_TRUE((*invlists)[0].get_doc_ids().empty());
    ASSERT_TRUE((*invlists)[2].get_doc_ids().empty());
}

TEST(ArrayInvertedLists, build_inverted_lists_multiple_docs_different_terms) {
    nsparse::SparseVectorsConfig config{.element_size = nsparse::U32,
                                        .dimension = 4};
    nsparse::SparseVectors vectors(config);

    // Doc 0: terms 0, 1
    std::vector<nsparse::term_t> indices1 = {0, 1};
    std::vector<float> vals1 = {1.0F, 2.0F};
    std::vector<uint8_t> weights1(reinterpret_cast<uint8_t*>(vals1.data()),
                                  reinterpret_cast<uint8_t*>(vals1.data()) +
                                      vals1.size() * sizeof(float));
    vectors.add_vector(indices1, weights1);

    // Doc 1: terms 1, 2
    std::vector<nsparse::term_t> indices2 = {1, 2};
    std::vector<float> vals2 = {3.0F, 4.0F};
    std::vector<uint8_t> weights2(reinterpret_cast<uint8_t*>(vals2.data()),
                                  reinterpret_cast<uint8_t*>(vals2.data()) +
                                      vals2.size() * sizeof(float));
    vectors.add_vector(indices2, weights2);

    // Doc 2: terms 2, 3
    std::vector<nsparse::term_t> indices3 = {2, 3};
    std::vector<float> vals3 = {5.0F, 6.0F};
    std::vector<uint8_t> weights3(reinterpret_cast<uint8_t*>(vals3.data()),
                                  reinterpret_cast<uint8_t*>(vals3.data()) +
                                      vals3.size() * sizeof(float));
    vectors.add_vector(indices3, weights3);

    auto invlists = nsparse::ArrayInvertedLists::build_inverted_lists(
        4, nsparse::U32, &vectors);

    // Term 0: doc 0
    ASSERT_EQ((*invlists)[0].get_doc_ids().size(), 1);
    ASSERT_EQ((*invlists)[0].get_doc_ids()[0], 0);

    // Term 1: docs 0, 1
    ASSERT_EQ((*invlists)[1].get_doc_ids().size(), 2);
    auto term1_docs = doc_ids_of((*invlists)[1]);
    std::ranges::sort(term1_docs);
    ASSERT_EQ(term1_docs[0], 0);
    ASSERT_EQ(term1_docs[1], 1);

    // Term 2: docs 1, 2
    ASSERT_EQ((*invlists)[2].get_doc_ids().size(), 2);
    auto term2_docs = doc_ids_of((*invlists)[2]);
    std::ranges::sort(term2_docs);
    ASSERT_EQ(term2_docs[0], 1);
    ASSERT_EQ(term2_docs[1], 2);

    // Term 3: doc 2
    ASSERT_EQ((*invlists)[3].get_doc_ids().size(), 1);
    ASSERT_EQ((*invlists)[3].get_doc_ids()[0], 2);
}

TEST(ArrayInvertedLists, build_inverted_lists_uint8_element_size) {
    nsparse::SparseVectorsConfig config{.element_size = nsparse::U8,
                                        .dimension = 3};
    nsparse::SparseVectors vectors(config);

    std::vector<nsparse::term_t> indices = {0, 2};
    std::vector<uint8_t> weights = {100, 200};
    vectors.add_vector(indices, weights);

    auto invlists = nsparse::ArrayInvertedLists::build_inverted_lists(
        3, nsparse::U8, &vectors);

    ASSERT_EQ(invlists->get_element_size(), nsparse::U8);
    ASSERT_EQ((*invlists)[0].get_doc_ids().size(), 1);
    ASSERT_EQ((*invlists)[2].get_doc_ids().size(), 1);
    ASSERT_TRUE((*invlists)[1].get_doc_ids().empty());
}

TEST(ArrayInvertedLists, build_inverted_lists_uint16_element_size) {
    nsparse::SparseVectorsConfig config{.element_size = nsparse::U16,
                                        .dimension = 3};
    nsparse::SparseVectors vectors(config);

    std::vector<nsparse::term_t> indices = {1};
    uint16_t value = 1000;
    std::vector<uint8_t> weights(
        reinterpret_cast<uint8_t*>(&value),
        reinterpret_cast<uint8_t*>(&value) + sizeof(uint16_t));
    vectors.add_vector(indices, weights);

    auto invlists = nsparse::ArrayInvertedLists::build_inverted_lists(
        3, nsparse::U16, &vectors);

    ASSERT_EQ(invlists->get_element_size(), nsparse::U16);
    ASSERT_EQ((*invlists)[1].get_doc_ids().size(), 1);
    ASSERT_EQ((*invlists)[1].get_doc_ids()[0], 0);
}

TEST(ArrayInvertedLists, build_inverted_lists_preserves_values) {
    nsparse::SparseVectorsConfig config{.element_size = nsparse::U32,
                                        .dimension = 3};
    nsparse::SparseVectors vectors(config);

    std::vector<nsparse::term_t> indices = {1};
    float value = 42.5F;
    std::vector<uint8_t> weights(
        reinterpret_cast<uint8_t*>(&value),
        reinterpret_cast<uint8_t*>(&value) + sizeof(float));
    vectors.add_vector(indices, weights);

    auto invlists = nsparse::ArrayInvertedLists::build_inverted_lists(
        3, nsparse::U32, &vectors);

    const auto& codes = (*invlists)[1].get_codes();
    ASSERT_EQ(codes.size(), sizeof(float));
    float stored = *reinterpret_cast<const float*>(codes.data());
    ASSERT_FLOAT_EQ(stored, 42.5F);
}

// Tests for newly added InvertedList functions

TEST(InvertedList, get_value_float_all_element_sizes) {
    // U32 (float)
    {
        nsparse::InvertedList list(nsparse::U32);
        std::vector<nsparse::idx_t> doc_ids = {1, 2, 3};
        std::vector<float> values = {1.5F, 2.7F, 0.3F};
        list.add_entries(3, doc_ids.data(),
                         reinterpret_cast<const uint8_t*>(values.data()));

        EXPECT_FLOAT_EQ(list.get_value_float(0), 1.5F);
        EXPECT_FLOAT_EQ(list.get_value_float(1), 2.7F);
        EXPECT_FLOAT_EQ(list.get_value_float(2), 0.3F);
    }
    // U16
    {
        nsparse::InvertedList list(nsparse::U16);
        std::vector<nsparse::idx_t> doc_ids = {10, 20};
        std::vector<uint16_t> values = {500, 1000};
        list.add_entries(2, doc_ids.data(),
                         reinterpret_cast<const uint8_t*>(values.data()));

        EXPECT_FLOAT_EQ(list.get_value_float(0), 500.0F);
        EXPECT_FLOAT_EQ(list.get_value_float(1), 1000.0F);
    }
    // U8
    {
        nsparse::InvertedList list(nsparse::U8);
        std::vector<nsparse::idx_t> doc_ids = {5, 6};
        std::vector<uint8_t> values = {42, 255};
        list.add_entries(2, doc_ids.data(), values.data());

        EXPECT_FLOAT_EQ(list.get_value_float(0), 42.0F);
        EXPECT_FLOAT_EQ(list.get_value_float(1), 255.0F);
    }
}

TEST(InvertedList, max_value_returns_maximum) {
    nsparse::InvertedList list(nsparse::U32);
    std::vector<nsparse::idx_t> doc_ids = {1, 2, 3, 4};
    std::vector<float> values = {0.5F, 3.7F, 1.2F, 2.9F};
    list.add_entries(4, doc_ids.data(),
                     reinterpret_cast<const uint8_t*>(values.data()));

    EXPECT_FLOAT_EQ(list.max_value(), 3.7F);

    // Empty list returns 0
    nsparse::InvertedList empty_list(nsparse::U32);
    EXPECT_FLOAT_EQ(empty_list.max_value(), 0.0F);
}

TEST(InvertedList, size_returns_doc_count) {
    nsparse::InvertedList list(nsparse::U32);
    EXPECT_EQ(list.size(), 0);

    std::vector<nsparse::idx_t> doc_ids = {1, 2, 3};
    std::vector<float> values = {1.0F, 2.0F, 3.0F};
    list.add_entries(3, doc_ids.data(),
                     reinterpret_cast<const uint8_t*>(values.data()));
    EXPECT_EQ(list.size(), 3);

    // Add more entries
    nsparse::idx_t doc4 = 4;
    float val4 = 4.0F;
    list.add_entries(1, &doc4, reinterpret_cast<const uint8_t*>(&val4));
    EXPECT_EQ(list.size(), 4);

    list.clear();
    EXPECT_EQ(list.size(), 0);
}

// ============== serialization ==============

namespace {

// A three-term, mixed-length set of lists: two populated (one of odd length, so
// the codes that follow its doc ids need padding) and one empty.
std::unique_ptr<nsparse::ArrayInvertedLists> lists_for_io() {
    nsparse::SparseVectorsConfig config{.element_size = nsparse::U32,
                                        .dimension = 3};
    nsparse::SparseVectors vectors(config);
    // doc0: {0: 1.5, 2: 2.5}, doc1: {0: 0.5}, doc2: {0: 3.5}
    std::vector<nsparse::idx_t> indptr = {0, 2, 3, 4};
    std::vector<nsparse::term_t> indices = {0, 2, 0, 0};
    std::vector<float> values = {1.5F, 2.5F, 0.5F, 3.5F};
    vectors.add_vectors(indptr.data(), indptr.size(), indices.data(),
                        indices.size(),
                        reinterpret_cast<const uint8_t*>(values.data()),
                        values.size() * sizeof(float));
    return nsparse::ArrayInvertedLists::build_inverted_lists(3, nsparse::U32,
                                                             &vectors);
}

void expect_same_lists(const nsparse::ArrayInvertedLists& actual,
                       const nsparse::ArrayInvertedLists& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    ASSERT_EQ(actual.get_element_size(), expected.get_element_size());
    for (size_t term = 0; term < expected.size(); ++term) {
        const auto& want = expected[term];
        const auto& got = actual[term];
        ASSERT_EQ(got.size(), want.size()) << "term " << term;
        for (size_t i = 0; i < want.size(); ++i) {
            EXPECT_EQ(got.get_doc_ids()[i], want.get_doc_ids()[i])
                << "term " << term << " entry " << i;
            EXPECT_FLOAT_EQ(got.get_value_float(i), want.get_value_float(i))
                << "term " << term << " entry " << i;
        }
    }
}

}  // namespace

TEST(ArrayInvertedLists, read_matches_what_serialize_wrote) {
    auto original = lists_for_io();

    nsparse::BufferedIOWriter writer;
    original->serialize(&writer);

    nsparse::BufferedIOReader reader(writer.data());
    auto loaded = nsparse::ArrayInvertedLists::read(&reader, nsparse::U32);

    expect_same_lists(*loaded, *original);
    // Copied out, so nothing points into the writer's buffer.
    EXPECT_TRUE((*loaded)[0].get_doc_ids().owns());
}

// The point of the mapped path: the arrays point into the serialized bytes
// rather than into fresh allocations.
TEST(ArrayInvertedLists, map_borrows_from_the_serialized_bytes) {
    auto original = lists_for_io();

    nsparse::BufferedIOWriter writer;
    original->serialize(&writer);
    const auto& bytes = writer.data();

    nsparse::MmapCursor cursor(bytes.data(), bytes.size());
    auto mapped = nsparse::ArrayInvertedLists::map(&cursor, nsparse::U32);

    expect_same_lists(*mapped, *original);
    // Every array in the file has been consumed.
    EXPECT_EQ(cursor.remaining(), 0);

    const auto& doc_ids = (*mapped)[0].get_doc_ids();
    const auto& codes = (*mapped)[0].get_codes();
    EXPECT_FALSE(doc_ids.owns());
    EXPECT_FALSE(codes.owns());
    const auto* base = bytes.data();
    const auto* end = base + bytes.size();
    const auto* doc_ids_bytes =
        reinterpret_cast<const uint8_t*>(doc_ids.data());
    EXPECT_GE(doc_ids_bytes, base);
    EXPECT_LT(doc_ids_bytes, end);
    // A doc id array is 4-byte aligned and so are the codes, so the codes of a
    // list follow its doc ids with no padding between them.
    EXPECT_EQ(codes.data(), doc_ids_bytes + doc_ids.byte_size());
}

TEST(ArrayInvertedLists, map_rejects_a_truncated_buffer) {
    auto original = lists_for_io();

    nsparse::BufferedIOWriter writer;
    original->serialize(&writer);
    const auto& bytes = writer.data();

    nsparse::MmapCursor cursor(bytes.data(), bytes.size() - sizeof(float));
    ASSERT_THROW(nsparse::ArrayInvertedLists::map(&cursor, nsparse::U32),
                 std::runtime_error);
}

// A term count the rest of the buffer cannot hold is rejected before it is
// allocated, rather than after a multi-gigabyte reserve.
TEST(ArrayInvertedLists, map_rejects_an_implausible_term_count) {
    std::vector<uint8_t> bytes(2 * sizeof(size_t));
    // size_t{1}, not 1UL: unsigned long is 32 bits on Windows, where the shift
    // folded to a term count small enough for the guard to let through.
    const size_t n_term = size_t{1} << 40;
    const size_t element_size = nsparse::U32;
    std::memcpy(bytes.data(), &n_term, sizeof(size_t));
    std::memcpy(bytes.data() + sizeof(size_t), &element_size, sizeof(size_t));

    nsparse::MmapCursor cursor(bytes.data(), bytes.size());
    ASSERT_THROW(nsparse::ArrayInvertedLists::map(&cursor, nsparse::U32),
                 std::runtime_error);
}

// The stored width sets how many code bytes each list holds, so reading at
// another one desynchronizes at the first list. Both readers therefore reject
// it up front rather than on the garbage that follows.
TEST(ArrayInvertedLists, both_reads_reject_a_foreign_element_width) {
    auto original = lists_for_io();

    nsparse::BufferedIOWriter writer;
    original->serialize(&writer);
    const auto& bytes = writer.data();

    nsparse::BufferedIOReader reader(bytes);
    ASSERT_THROW(nsparse::ArrayInvertedLists::read(&reader, nsparse::U16),
                 std::runtime_error);
    // Nothing past the two header fields was consumed.
    EXPECT_EQ(reader.pos(), 2 * sizeof(size_t));

    nsparse::MmapCursor cursor(bytes.data(), bytes.size());
    ASSERT_THROW(nsparse::ArrayInvertedLists::map(&cursor, nsparse::U16),
                 std::runtime_error);
    EXPECT_EQ(cursor.pos(), 2 * sizeof(size_t));
}

TEST(ArrayInvertedLists, mapped_lists_cannot_be_appended_to) {
    auto original = lists_for_io();

    nsparse::BufferedIOWriter writer;
    original->serialize(&writer);
    const auto& bytes = writer.data();

    nsparse::MmapCursor cursor(bytes.data(), bytes.size());
    auto mapped = nsparse::ArrayInvertedLists::map(&cursor, nsparse::U32);

    nsparse::idx_t doc_id = 99;
    float value = 1.0F;
    ASSERT_THROW(mapped->add_entries(0, 1, &doc_id,
                                     reinterpret_cast<const uint8_t*>(&value)),
                 std::runtime_error);
}

TEST(InvertedList, set_entries_rejects_a_mismatched_code_count) {
    nsparse::InvertedList list(nsparse::U32);
    // Two doc ids but only one float's worth of codes.
    ASSERT_THROW(list.set_entries({1, 2}, std::vector<uint8_t>(sizeof(float))),
                 std::invalid_argument);
}
