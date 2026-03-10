/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/utils/vector_process.h"

#include <gtest/gtest.h>

#include <vector>

TEST(TopKTokens, basic) {
    std::vector<nsparse::term_t> indices = {0, 1, 2, 3, 4};
    std::vector<float> weights = {1.0F, 5.0F, 3.0F, 2.0F, 4.0F};

    auto result =
        nsparse::detail::top_k_tokens(indices.data(), weights.data(), 5, 3);

    ASSERT_EQ(result.size(), 3);
    ASSERT_EQ(result[0], 1);  // weight 5.0 (highest)
    ASSERT_EQ(result[1], 4);  // weight 4.0
    ASSERT_EQ(result[2], 2);  // weight 3.0
}

TEST(TopKTokens, k_equals_size) {
    std::vector<nsparse::term_t> indices = {10, 20, 30};
    std::vector<float> weights = {3.0F, 1.0F, 2.0F};

    auto result =
        nsparse::detail::top_k_tokens(indices.data(), weights.data(), 3, 3);

    // When k >= size, returns all indices in original order (early return)
    ASSERT_EQ(result.size(), 3);
    ASSERT_EQ(result[0], 10);
    ASSERT_EQ(result[1], 20);
    ASSERT_EQ(result[2], 30);
}

TEST(TopKTokens, k_greater_than_size) {
    std::vector<nsparse::term_t> indices = {5, 6};
    std::vector<float> weights = {2.0F, 1.0F};

    auto result =
        nsparse::detail::top_k_tokens(indices.data(), weights.data(), 2, 5);

    // When k >= size, returns all indices in original order (early return)
    ASSERT_EQ(result.size(), 2);
    ASSERT_EQ(result[0], 5);
    ASSERT_EQ(result[1], 6);
}

TEST(TopKTokens, single_element) {
    std::vector<nsparse::term_t> indices = {42};
    std::vector<float> weights = {7.0F};

    auto result =
        nsparse::detail::top_k_tokens(indices.data(), weights.data(), 1, 1);

    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0], 42);
}

TEST(TopKTokens, duplicate_weights) {
    std::vector<nsparse::term_t> indices = {1, 2, 3, 4};
    std::vector<float> weights = {5.0F, 5.0F, 5.0F, 5.0F};

    auto result =
        nsparse::detail::top_k_tokens(indices.data(), weights.data(), 4, 2);

    ASSERT_EQ(result.size(), 2);
}
