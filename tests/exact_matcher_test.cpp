/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/exact_matcher.h"

#include <gtest/gtest.h>

#include "nsparse/id_selector.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace {

// Helper: build a SparseVectors with uint8 element size containing the given
// sparse vectors. Each vector is a list of (index, weight) pairs.
nsparse::SparseVectors make_u8_vectors(
    size_t dimension,
    const std::vector<std::vector<std::pair<nsparse::term_t, uint8_t>>>& vecs) {
    nsparse::SparseVectors sv(nsparse::SparseVectorsConfig{
        .element_size = nsparse::U8, .dimension = dimension});
    for (const auto& vec : vecs) {
        std::vector<nsparse::term_t> indices;
        std::vector<uint8_t> weights;
        for (auto [idx, w] : vec) {
            indices.push_back(idx);
            weights.push_back(w);
        }
        sv.add_vector(indices, weights);
    }
    return sv;
}

}  // namespace

TEST(ExactMatcher, single_query_returns_top_k_by_score) {
    // 3 documents in dimension 4, uint8 weights
    // doc0: index 0 -> 2, index 1 -> 3
    // doc1: index 1 -> 5
    // doc2: index 0 -> 1, index 2 -> 4
    auto vectors =
        make_u8_vectors(4, {{{0, 2}, {1, 3}}, {{1, 5}}, {{0, 1}, {2, 4}}});

    // query dense vector: [1, 2, 3, 0]
    // doc0 score: 2*1 + 3*2 = 8
    // doc1 score: 5*2 = 10
    // doc2 score: 1*1 + 4*3 = 13
    std::vector<uint8_t> dense = {1, 2, 3, 0};

    std::vector<nsparse::idx_t> ids = {0, 1, 2};
    nsparse::ArrayIDSelector selector(ids.size(), ids.data());

    auto [scores, labels] = nsparse::detail::ExactMatcher::single_query(
        &vectors, &selector, dense.data(), nsparse::U8, 2);

    ASSERT_EQ(scores.size(), 2);
    ASSERT_EQ(labels.size(), 2);
    // top-2 descending: doc2 (13), doc1 (10)
    EXPECT_EQ(labels[0], 2);
    EXPECT_EQ(labels[1], 1);
    EXPECT_FLOAT_EQ(scores[0], 13.0F);
    EXPECT_FLOAT_EQ(scores[1], 10.0F);
}

TEST(ExactMatcher, single_query_pads_to_k_when_fewer_docs) {
    // Only 1 document, but k=3
    auto vectors = make_u8_vectors(3, {{{0, 5}, {1, 2}}});

    std::vector<uint8_t> dense = {1, 1, 0};
    // doc0 score: 5*1 + 2*1 = 7

    std::vector<nsparse::idx_t> ids = {0};
    nsparse::ArrayIDSelector selector(ids.size(), ids.data());

    auto [scores, labels] = nsparse::detail::ExactMatcher::single_query(
        &vectors, &selector, dense.data(), nsparse::U8, 3);

    ASSERT_EQ(scores.size(), 3);
    ASSERT_EQ(labels.size(), 3);
    EXPECT_EQ(labels[0], 0);
    EXPECT_FLOAT_EQ(scores[0], 7.0F);
    // padded entries
    EXPECT_EQ(labels[1], nsparse::INVALID_IDX);
    EXPECT_FLOAT_EQ(scores[1], -1.0F);
    EXPECT_EQ(labels[2], nsparse::INVALID_IDX);
    EXPECT_FLOAT_EQ(scores[2], -1.0F);
}

TEST(ExactMatcher, single_query_subset_of_ids) {
    // 3 documents but selector only includes doc0 and doc2
    auto vectors = make_u8_vectors(3, {{{0, 1}}, {{0, 10}}, {{0, 5}}});

    std::vector<uint8_t> dense = {1, 0, 0};
    // doc0 score: 1, doc1 score: 10 (excluded), doc2 score: 5

    std::vector<nsparse::idx_t> ids = {0, 2};
    nsparse::ArrayIDSelector selector(ids.size(), ids.data());

    auto [scores, labels] = nsparse::detail::ExactMatcher::single_query(
        &vectors, &selector, dense.data(), nsparse::U8, 3);

    ASSERT_EQ(scores.size(), 3);
    ASSERT_EQ(labels.size(), 3);
    // top descending: doc2 (5), doc0 (1), then padding
    EXPECT_EQ(labels[0], 2);
    EXPECT_FLOAT_EQ(scores[0], 5.0F);
    EXPECT_EQ(labels[1], 0);
    EXPECT_FLOAT_EQ(scores[1], 1.0F);
    EXPECT_EQ(labels[2], nsparse::INVALID_IDX);
    EXPECT_FLOAT_EQ(scores[2], -1.0F);
}

TEST(ExactMatcher, single_query_zero_scores) {
    // Documents with no overlap with query
    auto vectors = make_u8_vectors(4, {{{2, 3}}, {{3, 7}}});

    // query only has weight on index 0
    std::vector<uint8_t> dense = {10, 0, 0, 0};

    std::vector<nsparse::idx_t> ids = {0, 1};
    nsparse::ArrayIDSelector selector(ids.size(), ids.data());

    auto [scores, labels] = nsparse::detail::ExactMatcher::single_query(
        &vectors, &selector, dense.data(), nsparse::U8, 2);

    ASSERT_EQ(scores.size(), 2);
    // Both scores should be 0
    EXPECT_FLOAT_EQ(scores[0], 0.0F);
    EXPECT_FLOAT_EQ(scores[1], 0.0F);
}

TEST(ExactMatcher, search_multiple_queries) {
    // 2 documents, dimension 3
    // doc0: index 0 -> 2
    // doc1: index 1 -> 3
    auto vectors = make_u8_vectors(3, {{{0, 2}}, {{1, 3}}});

    // 2 queries as sparse vectors
    // query0: index 0 -> 1  => doc0=2, doc1=0
    // query1: index 1 -> 2  => doc0=0, doc1=6
    auto queries = make_u8_vectors(3, {{{0, 1}}, {{1, 2}}});

    std::vector<nsparse::idx_t> ids = {0, 1};
    nsparse::ArrayIDSelector selector(ids.size(), ids.data());

    auto [result_distances, result_labels] =
        nsparse::detail::ExactMatcher::search(&vectors, &selector, &queries,
                                              nsparse::U8, 2);

    ASSERT_EQ(result_distances.size(), 2);
    ASSERT_EQ(result_labels.size(), 2);

    // query0: top descending doc0(2), doc1(0)
    EXPECT_EQ(result_labels[0][0], 0);
    EXPECT_FLOAT_EQ(result_distances[0][0], 2.0F);
    EXPECT_EQ(result_labels[0][1], 1);
    EXPECT_FLOAT_EQ(result_distances[0][1], 0.0F);

    // query1: top descending doc1(6), doc0(0)
    EXPECT_EQ(result_labels[1][0], 1);
    EXPECT_FLOAT_EQ(result_distances[1][0], 6.0F);
    EXPECT_EQ(result_labels[1][1], 0);
    EXPECT_FLOAT_EQ(result_distances[1][1], 0.0F);
}

TEST(ExactMatcher, search_with_k_greater_than_docs) {
    auto vectors = make_u8_vectors(2, {{{0, 1}}});
    auto queries = make_u8_vectors(2, {{{0, 3}}});

    std::vector<nsparse::idx_t> ids = {0};
    nsparse::ArrayIDSelector selector(ids.size(), ids.data());

    auto [result_distances, result_labels] =
        nsparse::detail::ExactMatcher::search(&vectors, &selector, &queries,
                                              nsparse::U8, 3);

    ASSERT_EQ(result_distances.size(), 1);
    ASSERT_EQ(result_labels[0].size(), 3);
    EXPECT_EQ(result_labels[0][0], 0);
    EXPECT_FLOAT_EQ(result_distances[0][0], 3.0F);
    EXPECT_EQ(result_labels[0][1], nsparse::INVALID_IDX);
    EXPECT_EQ(result_labels[0][2], nsparse::INVALID_IDX);
}

TEST(ExactMatcher, single_query_with_set_id_selector) {
    auto vectors = make_u8_vectors(3, {{{0, 1}}, {{1, 2}}, {{2, 3}}});

    std::vector<uint8_t> dense = {1, 1, 1};
    // doc0=1, doc1=2, doc2=3

    std::vector<nsparse::idx_t> ids = {1, 2};
    nsparse::SetIDSelector selector(ids.size(), ids.data());

    auto [scores, labels] = nsparse::detail::ExactMatcher::single_query(
        &vectors, &selector, dense.data(), nsparse::U8, 2);

    ASSERT_EQ(labels.size(), 2);
    EXPECT_EQ(labels[0], 2);
    EXPECT_FLOAT_EQ(scores[0], 3.0F);
    EXPECT_EQ(labels[1], 1);
    EXPECT_FLOAT_EQ(scores[1], 2.0F);
}
