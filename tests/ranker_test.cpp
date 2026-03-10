/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/utils/ranker.h"

#include <gtest/gtest.h>

TEST(TopKHolder, constructor_throw_for_invalid_k) {
    ASSERT_THROW(nsparse::detail::TopKHolder<int> holder(0),
                 std::invalid_argument);
}

TEST(TopKHolder, full) {
    nsparse::detail::TopKHolder<int> holder(2);
    holder.add(1.0F, 1);
    ASSERT_FALSE(holder.full());
    holder.add(2.0F, 2);
    ASSERT_TRUE(holder.full());
    holder.add(3.0F, 3);
    ASSERT_TRUE(holder.full());
}

TEST(TopKHolder, empty) {
    nsparse::detail::TopKHolder<int> holder(2);
    ASSERT_TRUE(holder.empty());
    holder.add(1.0F, 1);
    ASSERT_FALSE(holder.empty());
}

TEST(TopKHolder, size) {
    nsparse::detail::TopKHolder<int> holder(2);
    ASSERT_EQ(holder.size(), 0);
    holder.add(1.0F, 1);
    ASSERT_EQ(holder.size(), 1);
    holder.add(2.0F, 2);
    ASSERT_EQ(holder.size(), 2);
    holder.add(3.0F, 3);
    ASSERT_EQ(holder.size(), 2);
}

TEST(TopKHolder, peek_score) {
    nsparse::detail::TopKHolder<int> holder(3);
    holder.add(5.0F, 1);
    ASSERT_FLOAT_EQ(holder.peek_score(), 5.0F);
    holder.add(3.0F, 2);
    ASSERT_FLOAT_EQ(holder.peek_score(), 3.0F);
    holder.add(7.0F, 3);
    ASSERT_FLOAT_EQ(holder.peek_score(), 3.0F);
    holder.add(10.0F, 4);
    ASSERT_FLOAT_EQ(holder.peek_score(), 5.0F);
}

TEST(TopKHolder, add) {
    nsparse::detail::TopKHolder<int> holder(3);
    holder.add(1.0F, 10);
    holder.add(5.0F, 50);
    holder.add(3.0F, 30);
    ASSERT_EQ(holder.size(), 3);

    holder.add(0.5F, 5);
    ASSERT_EQ(holder.size(), 3);
    ASSERT_FLOAT_EQ(holder.peek_score(), 1.0F);

    holder.add(4.0F, 40);
    ASSERT_EQ(holder.size(), 3);
    ASSERT_FLOAT_EQ(holder.peek_score(), 3.0F);
}

TEST(TopKHolder, top_k) {
    nsparse::detail::TopKHolder<int> holder(3);
    holder.add(5.0F, 50);
    holder.add(1.0F, 10);
    holder.add(3.0F, 30);

    auto result = holder.top_k();
    ASSERT_EQ(result.size(), 3);
    ASSERT_EQ(result[0], 10);
    ASSERT_EQ(result[1], 30);
    ASSERT_EQ(result[2], 50);
}

TEST(TopKHolder, top_k_descending) {
    nsparse::detail::TopKHolder<int> holder(3);
    holder.add(5.0F, 50);
    holder.add(1.0F, 10);

    auto result = holder.top_k_descending();
    ASSERT_EQ(result.size(), 2);  // no padding, returns actual size
    ASSERT_EQ(result[0], 50);     // highest score first
    ASSERT_EQ(result[1], 10);
}

TEST(TopKHolder, top_k_descending_with_padding) {
    nsparse::detail::TopKHolder<int> holder(3);
    holder.add(5.0F, 50);
    holder.add(1.0F, 10);

    auto result = holder.top_k_descending_with_padding(-1);
    ASSERT_EQ(result.size(), 3);  // padded to k
    ASSERT_EQ(result[0], 50);     // highest score first
    ASSERT_EQ(result[1], 10);
    ASSERT_EQ(result[2], -1);  // padded value
}

// DedupeTopKHolder tests
TEST(DedupeTopKHolder, full) {
    nsparse::detail::DedupeTopKHolder<int> holder(2);
    holder.add(1.0F, 1, 10);
    ASSERT_FALSE(holder.full());
    holder.add(2.0F, 2, 20);
    ASSERT_TRUE(holder.full());
    holder.add(3.0F, 3, 30);
    ASSERT_TRUE(holder.full());
}

TEST(DedupeTopKHolder, empty) {
    nsparse::detail::DedupeTopKHolder<int> holder(2);
    ASSERT_TRUE(holder.empty());
    holder.add(1.0F, 1, 10);
    ASSERT_FALSE(holder.empty());
}

TEST(DedupeTopKHolder, size) {
    nsparse::detail::DedupeTopKHolder<int> holder(2);
    ASSERT_EQ(holder.size(), 0);
    holder.add(1.0F, 1, 10);
    ASSERT_EQ(holder.size(), 1);
    holder.add(2.0F, 2, 20);
    ASSERT_EQ(holder.size(), 2);
    holder.add(3.0F, 3, 30);
    ASSERT_EQ(holder.size(), 2);
}

TEST(DedupeTopKHolder, peek_score) {
    nsparse::detail::DedupeTopKHolder<int> holder(3);
    holder.add(5.0F, 1, 10);
    ASSERT_FLOAT_EQ(holder.peek_score(), 5.0F);
    holder.add(3.0F, 2, 20);
    ASSERT_FLOAT_EQ(holder.peek_score(), 3.0F);
    holder.add(7.0F, 3, 30);
    ASSERT_FLOAT_EQ(holder.peek_score(), 3.0F);
    holder.add(10.0F, 4, 40);
    ASSERT_FLOAT_EQ(holder.peek_score(), 5.0F);
}

TEST(DedupeTopKHolder, add_deduplicates) {
    nsparse::detail::DedupeTopKHolder<int> holder(3);
    holder.add(5.0F, 1, 10);
    holder.add(3.0F, 1, 20);  // same id, should be ignored
    ASSERT_EQ(holder.size(), 1);

    holder.add(7.0F, 2, 30);
    holder.add(9.0F, 2, 40);  // same id, should be ignored
    ASSERT_EQ(holder.size(), 2);
}

TEST(DedupeTopKHolder, add_with_two_args) {
    nsparse::detail::DedupeTopKHolder<size_t> holder(3);
    holder.add(5.0F, 1);
    holder.add(3.0F, 2);
    holder.add(7.0F, 3);
    ASSERT_EQ(holder.size(), 3);
    ASSERT_TRUE(holder.full());
}

TEST(DedupeTopKHolder, top_k) {
    nsparse::detail::DedupeTopKHolder<int> holder(3);
    holder.add(5.0F, 1, 50);
    holder.add(1.0F, 2, 10);
    holder.add(3.0F, 3, 30);

    auto result = holder.top_k();
    ASSERT_EQ(result.size(), 3);
    // ascending order (lowest first from min-heap)
    ASSERT_EQ(result[0], 10);
    ASSERT_EQ(result[1], 30);
    ASSERT_EQ(result[2], 50);
}

TEST(DedupeTopKHolder, top_k_descending) {
    nsparse::detail::DedupeTopKHolder<int> holder(3);
    holder.add(5.0F, 1, 50);
    holder.add(1.0F, 2, 10);

    auto result = holder.top_k_descending();
    ASSERT_EQ(result.size(), 2);  // no padding, returns actual size
    // descending order (highest first)
    ASSERT_EQ(result[0], 50);
    ASSERT_EQ(result[1], 10);
}

TEST(DedupeTopKHolder, top_k_descending_with_padding) {
    nsparse::detail::DedupeTopKHolder<int> holder(3);
    holder.add(5.0F, 1, 50);
    holder.add(1.0F, 2, 10);

    auto result = holder.top_k_descending_with_padding(-1);
    ASSERT_EQ(result.size(), 3);  // padded to k
    // descending order (highest first)
    ASSERT_EQ(result[0], 50);
    ASSERT_EQ(result[1], 10);
    ASSERT_EQ(result[2], -1);  // padded value
}
