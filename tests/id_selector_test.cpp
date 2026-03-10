/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/id_selector.h"

#include <vector>

#include "gtest/gtest.h"

namespace nsparse {

// --- SetIDSelector tests ---

TEST(SetIDSelector, empty_set_contains_nothing) {
    SetIDSelector selector(0, nullptr);
    EXPECT_FALSE(selector.is_member(0));
    EXPECT_FALSE(selector.is_member(1));
    EXPECT_FALSE(selector.is_member(-1));
}

TEST(SetIDSelector, single_element) {
    idx_t id = 42;
    SetIDSelector selector(1, &id);
    EXPECT_TRUE(selector.is_member(42));
    EXPECT_FALSE(selector.is_member(0));
    EXPECT_FALSE(selector.is_member(43));
}

TEST(SetIDSelector, multiple_elements) {
    std::vector<idx_t> ids = {1, 5, 10, 100};
    SetIDSelector selector(ids.size(), ids.data());
    for (idx_t id : ids) {
        EXPECT_TRUE(selector.is_member(id));
    }
    EXPECT_FALSE(selector.is_member(0));
    EXPECT_FALSE(selector.is_member(2));
    EXPECT_FALSE(selector.is_member(99));
}

TEST(SetIDSelector, duplicate_ids) {
    std::vector<idx_t> ids = {3, 3, 3};
    SetIDSelector selector(ids.size(), ids.data());
    EXPECT_TRUE(selector.is_member(3));
    EXPECT_FALSE(selector.is_member(0));
}

TEST(SetIDSelector, negative_ids) {
    std::vector<idx_t> ids = {-5, -1, 0, 7};
    SetIDSelector selector(ids.size(), ids.data());
    EXPECT_TRUE(selector.is_member(-5));
    EXPECT_TRUE(selector.is_member(-1));
    EXPECT_TRUE(selector.is_member(0));
    EXPECT_TRUE(selector.is_member(7));
    EXPECT_FALSE(selector.is_member(-2));
}

TEST(SetIDSelector, operator_call_delegates_to_is_member) {
    std::vector<idx_t> ids = {10, 20};
    SetIDSelector selector(ids.size(), ids.data());
    EXPECT_TRUE(selector(10));
    EXPECT_TRUE(selector(20));
    EXPECT_FALSE(selector(15));
}

// --- NotIDSelector tests ---

TEST(NotIDSelector, negates_delegate) {
    std::vector<idx_t> ids = {1, 2, 3};
    SetIDSelector inner(ids.size(), ids.data());
    NotIDSelector selector(&inner);
    EXPECT_FALSE(selector.is_member(1));
    EXPECT_FALSE(selector.is_member(2));
    EXPECT_FALSE(selector.is_member(3));
    EXPECT_TRUE(selector.is_member(0));
    EXPECT_TRUE(selector.is_member(4));
}

TEST(NotIDSelector, negates_empty_set_accepts_all) {
    SetIDSelector inner(0, nullptr);
    NotIDSelector selector(&inner);
    EXPECT_TRUE(selector.is_member(0));
    EXPECT_TRUE(selector.is_member(42));
    EXPECT_TRUE(selector.is_member(-1));
}

TEST(NotIDSelector, operator_call_delegates_to_is_member) {
    idx_t id = 5;
    SetIDSelector inner(1, &id);
    NotIDSelector selector(&inner);
    EXPECT_FALSE(selector(5));
    EXPECT_TRUE(selector(6));
}

TEST(NotIDSelector, double_negation) {
    std::vector<idx_t> ids = {10, 20};
    SetIDSelector base(ids.size(), ids.data());
    NotIDSelector not_selector(&base);
    NotIDSelector double_not(&not_selector);
    EXPECT_TRUE(double_not.is_member(10));
    EXPECT_TRUE(double_not.is_member(20));
    EXPECT_FALSE(double_not.is_member(15));
}

}  // namespace nsparse
