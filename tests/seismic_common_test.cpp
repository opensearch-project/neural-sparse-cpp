/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/seismic_common.h"

#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "nsparse/id_selector.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse {
namespace detail {

// ---- calculate_summary_scores tests ----

TEST(CalculateSummaryScores, float_element_size) {
    // dimension=4, element_size=U32 (float)
    SparseVectorsConfig cfg{.element_size = U32, .dimension = 4};
    SparseVectors summaries(cfg);

    // One sparse vector: indices {1, 3}, values {2.0f, 3.0f}
    std::vector<term_t> indices = {1, 3};
    std::vector<uint8_t> weights(2 * sizeof(float));
    float w0 = 2.0F;
    float w1 = 3.0F;
    std::memcpy(weights.data(), &w0, sizeof(float));
    std::memcpy(weights.data() + sizeof(float), &w1, sizeof(float));
    summaries.add_vector(indices.data(), indices.size(), weights.data(),
                         weights.size());

    // Dense vector: [0.0, 1.0, 0.0, 2.0] as floats in uint8_t buffer
    std::vector<uint8_t> dense(4 * sizeof(float));
    float d[4] = {0.0F, 1.0F, 0.0F, 2.0F};
    std::memcpy(dense.data(), d, sizeof(d));

    auto scores = calculate_summary_scores(U32, &summaries, dense);
    ASSERT_EQ(scores.size(), 1);
    // 2.0*1.0 + 3.0*2.0 = 8.0
    EXPECT_FLOAT_EQ(scores[0], 8.0F);
}

TEST(CalculateSummaryScores, uint16_element_size) {
    SparseVectorsConfig cfg{.element_size = U16, .dimension = 4};
    SparseVectors summaries(cfg);

    std::vector<term_t> indices = {0, 2};
    std::vector<uint8_t> weights(2 * sizeof(uint16_t));
    uint16_t w0 = 10;
    uint16_t w1 = 20;
    std::memcpy(weights.data(), &w0, sizeof(uint16_t));
    std::memcpy(weights.data() + sizeof(uint16_t), &w1, sizeof(uint16_t));
    summaries.add_vector(indices.data(), indices.size(), weights.data(),
                         weights.size());

    std::vector<uint8_t> dense(4 * sizeof(uint16_t), 0);
    uint16_t d[4] = {3, 0, 5, 0};
    std::memcpy(dense.data(), d, sizeof(d));

    auto scores = calculate_summary_scores(U16, &summaries, dense);
    ASSERT_EQ(scores.size(), 1);
    // 10*3 + 20*5 = 130
    EXPECT_FLOAT_EQ(scores[0], 130.0F);
}

TEST(CalculateSummaryScores, uint8_element_size) {
    SparseVectorsConfig cfg{.element_size = U8, .dimension = 4};
    SparseVectors summaries(cfg);

    std::vector<term_t> indices = {0, 1};
    std::vector<uint8_t> weights = {5, 10};
    summaries.add_vector(indices.data(), indices.size(), weights.data(),
                         weights.size());

    std::vector<uint8_t> dense = {2, 3, 0, 0};

    auto scores = calculate_summary_scores(U8, &summaries, dense);
    ASSERT_EQ(scores.size(), 1);
    // 5*2 + 10*3 = 40
    EXPECT_FLOAT_EQ(scores[0], 40.0F);
}

TEST(CalculateSummaryScores, multiple_vectors) {
    SparseVectorsConfig cfg{.element_size = U8, .dimension = 3};
    SparseVectors summaries(cfg);

    // Vector 0: index {0}, weight {4}
    std::vector<term_t> idx0 = {0};
    std::vector<uint8_t> w0 = {4};
    summaries.add_vector(idx0.data(), idx0.size(), w0.data(), w0.size());

    // Vector 1: index {1, 2}, weight {2, 3}
    std::vector<term_t> idx1 = {1, 2};
    std::vector<uint8_t> w1 = {2, 3};
    summaries.add_vector(idx1.data(), idx1.size(), w1.data(), w1.size());

    std::vector<uint8_t> dense = {10, 20, 30};

    auto scores = calculate_summary_scores(U8, &summaries, dense);
    ASSERT_EQ(scores.size(), 2);
    // vec0: 4*10 = 40
    EXPECT_FLOAT_EQ(scores[0], 40.0F);
    // vec1: 2*20 + 3*30 = 130
    EXPECT_FLOAT_EQ(scores[1], 130.0F);
}

// ---- compute_similarity tests ----

TEST(ComputeSimilarity, float_element_size) {
    // 2 docs: doc0 has indices {1}, values {2.0f}; doc1 has indices {0,2},
    // values {1.0f, 3.0f}
    std::vector<offset_t> indptr = {0, 1, 3};
    std::vector<term_t> indices = {1, 0, 2};

    // Values stored as raw bytes
    std::vector<uint8_t> values(3 * sizeof(float));
    float v[3] = {2.0F, 1.0F, 3.0F};
    std::memcpy(values.data(), v, sizeof(v));

    // Dense: [5.0, 10.0, 7.0]
    std::vector<uint8_t> dense(3 * sizeof(float));
    float d[3] = {5.0F, 10.0F, 7.0F};
    std::memcpy(dense.data(), d, sizeof(d));

    // doc0: 2.0*10.0 = 20.0
    float score0 = compute_similarity(0, indptr.data(), indices.data(),
                                      values.data(), dense.data(), U32);
    EXPECT_FLOAT_EQ(score0, 20.0F);

    // doc1: 1.0*5.0 + 3.0*7.0 = 26.0
    float score1 = compute_similarity(1, indptr.data(), indices.data(),
                                      values.data(), dense.data(), U32);
    EXPECT_FLOAT_EQ(score1, 26.0F);
}

TEST(ComputeSimilarity, uint16_element_size) {
    std::vector<offset_t> indptr = {0, 2};
    std::vector<term_t> indices = {0, 1};

    std::vector<uint8_t> values(2 * sizeof(uint16_t));
    uint16_t v[2] = {100, 200};
    std::memcpy(values.data(), v, sizeof(v));

    std::vector<uint8_t> dense(2 * sizeof(uint16_t));
    uint16_t d[2] = {3, 4};
    std::memcpy(dense.data(), d, sizeof(d));

    // 100*3 + 200*4 = 1100
    float score = compute_similarity(0, indptr.data(), indices.data(),
                                     values.data(), dense.data(), U16);
    EXPECT_FLOAT_EQ(score, 1100.0F);
}

TEST(ComputeSimilarity, uint8_element_size) {
    std::vector<offset_t> indptr = {0, 3};
    std::vector<term_t> indices = {0, 1, 2};
    std::vector<uint8_t> values = {2, 3, 4};
    std::vector<uint8_t> dense = {10, 20, 30};

    // 2*10 + 3*20 + 4*30 = 200
    float score = compute_similarity(0, indptr.data(), indices.data(),
                                     values.data(), dense.data(), U8);
    EXPECT_FLOAT_EQ(score, 200.0F);
}

TEST(ComputeSimilarity, empty_doc) {
    std::vector<offset_t> indptr = {0, 0};
    std::vector<term_t> indices = {};
    std::vector<uint8_t> values = {};
    std::vector<uint8_t> dense = {1, 2, 3};

    float score = compute_similarity(0, indptr.data(), indices.data(),
                                     values.data(), dense.data(), U8);
    EXPECT_FLOAT_EQ(score, 0.0F);
}

// ---- reorder_clusters tests ----

TEST(ReorderClusters, first_list_sorts_descending) {
    std::vector<float> scores = {1.0F, 3.0F, 2.0F, 5.0F, 4.0F};
    auto order = reorder_clusters(scores, true);
    ASSERT_EQ(order.size(), 5);
    // Sorted by descending score: indices 3(5.0), 4(4.0), 1(3.0), 2(2.0),
    // 0(1.0)
    EXPECT_EQ(order[0], 3);
    EXPECT_EQ(order[1], 4);
    EXPECT_EQ(order[2], 1);
    EXPECT_EQ(order[3], 2);
    EXPECT_EQ(order[4], 0);
}

TEST(ReorderClusters, not_first_list_preserves_order) {
    std::vector<float> scores = {1.0F, 3.0F, 2.0F};
    auto order = reorder_clusters(scores, false);
    ASSERT_EQ(order.size(), 3);
    // Should be identity: 0, 1, 2
    EXPECT_EQ(order[0], 0);
    EXPECT_EQ(order[1], 1);
    EXPECT_EQ(order[2], 2);
}

TEST(ReorderClusters, empty_scores) {
    std::vector<float> scores = {};
    auto order = reorder_clusters(scores, true);
    EXPECT_TRUE(order.empty());
}

TEST(ReorderClusters, single_element) {
    std::vector<float> scores = {42.0F};
    auto order = reorder_clusters(scores, true);
    ASSERT_EQ(order.size(), 1);
    EXPECT_EQ(order[0], 0);
}

TEST(ReorderClusters, equal_scores_stable) {
    std::vector<float> scores = {5.0F, 5.0F, 5.0F};
    auto order = reorder_clusters(scores, true);
    ASSERT_EQ(order.size(), 3);
    // All equal, any permutation is valid, just check all indices present
    std::vector<bool> seen(3, false);
    for (auto idx : order) {
        seen[idx] = true;
    }
    EXPECT_TRUE(seen[0]);
    EXPECT_TRUE(seen[1]);
    EXPECT_TRUE(seen[2]);
}

// ---- should_run_exact_match tests ----

TEST(ShouldRunExactMatch, null_selector_returns_false) {
    EXPECT_FALSE(should_run_exact_match(nullptr, 10, nullptr));
}

TEST(ShouldRunExactMatch, non_enumerable_selector_returns_false) {
    // NotIDSelector is not IDSelectorEnumerable
    std::vector<idx_t> ids = {1};
    SetIDSelector inner(ids.size(), ids.data());
    NotIDSelector selector(&inner);
    EXPECT_FALSE(should_run_exact_match(&selector, 10, nullptr));
}

TEST(ShouldRunExactMatch, enumerable_size_less_than_k_returns_true) {
    std::vector<idx_t> ids = {1, 2, 3};
    SetIDSelector selector(ids.size(), ids.data());
    // size=3 <= k=5
    EXPECT_TRUE(should_run_exact_match(&selector, 5, nullptr));
}

TEST(ShouldRunExactMatch, enumerable_size_equal_to_k_returns_true) {
    std::vector<idx_t> ids = {1, 2, 3};
    SetIDSelector selector(ids.size(), ids.data());
    // size=3 <= k=3
    EXPECT_TRUE(should_run_exact_match(&selector, 3, nullptr));
}

TEST(ShouldRunExactMatch, enumerable_size_greater_than_k_returns_false) {
    std::vector<idx_t> ids = {1, 2, 3, 4, 5};
    SetIDSelector selector(ids.size(), ids.data());
    // size=5 > k=2
    EXPECT_FALSE(should_run_exact_match(&selector, 2, nullptr));
}

TEST(ShouldRunExactMatch, array_selector_also_enumerable) {
    std::vector<idx_t> ids = {10, 20};
    ArrayIDSelector selector(ids.size(), ids.data());
    // ArrayIDSelector is IDSelectorEnumerable, size=2 <= k=5
    EXPECT_TRUE(should_run_exact_match(&selector, 5, nullptr));
}

// ---- build_inverted_lists_clusters dimension boundary test ----

TEST(BuildInvertedListsClusters, dimension_65536_no_overflow) {
    // Regression test: when dimension=65536, batch_end=65536 must not be
    // cast to term_t (uint16_t) which would overflow to 0.
    SparseVectorsConfig cfg{.element_size = U8, .dimension = 65536};
    SparseVectors vectors(cfg);

    // 10 vectors, each with 5 entries at high term IDs (near 65535)
    std::vector<idx_t> indptr = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
    std::vector<term_t> indices;
    std::vector<uint8_t> values;
    for (int doc = 0; doc < 10; ++doc) {
        for (int j = 0; j < 5; ++j) {
            indices.push_back(static_cast<term_t>(65530 + j));
            values.push_back(static_cast<uint8_t>(100 + j));
        }
    }
    vectors.add_vectors(indptr.data(), indptr.size(), indices.data(),
                        indices.size(), values.data(), values.size());
    ASSERT_EQ(vectors.num_vectors(), 10);

    SeismicClusterParameters params = {.lambda = 10, .beta = 2, .alpha = 0.4F};
    auto result = build_inverted_lists_clusters(&vectors, cfg, params);
    ASSERT_EQ(result.size(), 65536);

    // Terms 65530..65534 should have non-empty clusters
    size_t non_empty = 0;
    for (size_t i = 65530; i < 65535; ++i) {
        if (result[i].cluster_size() > 0) {
            non_empty++;
        }
    }
    EXPECT_GT(non_empty, 0)
        << "Expected non-empty inverted lists for high term IDs";
}


// ---- resolve_build_mem_budget tests ----

TEST(ResolveBuildMemBudget, explicit_budget_used_verbatim) {
    // A caller-supplied budget must be honored exactly, regardless of platform
    // or available memory.
    const size_t requested = 3ULL * 1024 * 1024 * 1024;  // 3 GB
    EXPECT_EQ(resolve_build_mem_budget(requested), requested);
}

TEST(ResolveBuildMemBudget, explicit_small_budget_survives) {
    // Small explicit budgets are returned verbatim (not clamped to the
    // fallback), so behavior is deterministic on non-Linux platforms too.
    const size_t tiny = 123456;
    EXPECT_EQ(resolve_build_mem_budget(tiny), tiny);
}

TEST(ResolveBuildMemBudget, auto_detect_is_nonzero) {
    // With 0 (auto), the result must always be a usable, non-zero budget:
    //  - Linux: detected MemAvailable minus reserve (or fallback if tiny)
    //  - non-Linux: the fixed fallback
    // It must never be 0 (which would divide-by-zero in batch sizing).
    EXPECT_GT(resolve_build_mem_budget(0), 0u);
}

TEST(ResolveBuildMemBudget, auto_detect_matches_query_or_fallback) {
    // The auto path is a pure function of query_available_memory_bytes():
    //   available > reserve -> available - reserve
    //   otherwise           -> fixed fallback
    const size_t available = query_available_memory_bytes();
    const size_t budget = resolve_build_mem_budget(0);
    if (available > kBuildMemReserveBytes) {
        EXPECT_EQ(budget, available - kBuildMemReserveBytes);
    } else {
        EXPECT_EQ(budget, kBuildMemFallbackBytes);
    }
}

}  // namespace detail
}  // namespace nsparse
