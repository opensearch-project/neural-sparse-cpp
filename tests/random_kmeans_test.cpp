/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/cluster/random_kmeans.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace {

nsparse::SparseVectors create_float_vectors(
    const std::vector<std::vector<nsparse::term_t>>& indices_list,
    const std::vector<std::vector<float>>& values_list) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 10});
    for (size_t i = 0; i < indices_list.size(); ++i) {
        const auto& indices = indices_list[i];
        const auto& values = values_list[i];
        vectors.add_vector(indices.data(), indices.size(),
                           reinterpret_cast<const uint8_t*>(values.data()),
                           values.size() * sizeof(float));
    }
    return vectors;
}

}  // namespace

TEST(RandomKMeansTrain, throws_on_null_vectors) {
    std::vector<nsparse::idx_t> doc_ids = {0, 1};
    ASSERT_THROW(nsparse::detail::RandomKMeans::train(nullptr, doc_ids, 2),
                 std::invalid_argument);
}

TEST(RandomKMeansTrain, empty_doc_ids_returns_empty) {
    auto vectors = create_float_vectors({{0}}, {{1.0F}});
    std::vector<nsparse::idx_t> doc_ids = {};

    auto clusters = nsparse::detail::RandomKMeans::train(&vectors, doc_ids, 2);

    ASSERT_TRUE(clusters.empty());
}

TEST(RandomKMeansTrain, single_doc_single_cluster) {
    auto vectors = create_float_vectors({{0}}, {{1.0F}});
    std::vector<nsparse::idx_t> doc_ids = {0};

    auto clusters = nsparse::detail::RandomKMeans::train(&vectors, doc_ids, 1);

    ASSERT_EQ(clusters.size(), 1);
    ASSERT_EQ(clusters[0].size(), 1);
    ASSERT_EQ(clusters[0][0], 0);
}

TEST(RandomKMeansTrain, n_clusters_capped_to_n_docs) {
    auto vectors = create_float_vectors({{0}, {1}}, {{1.0F}, {1.0F}});
    std::vector<nsparse::idx_t> doc_ids = {0, 1};

    // Request more clusters than docs
    auto clusters = nsparse::detail::RandomKMeans::train(&vectors, doc_ids, 10);

    // Should cap to 2 clusters
    ASSERT_EQ(clusters.size(), 2);
}

TEST(RandomKMeansTrain, zero_n_clusters_uses_sqrt) {
    // Create 9 vectors
    std::vector<std::vector<nsparse::term_t>> indices_list;
    std::vector<std::vector<float>> values_list;
    for (int i = 0; i < 9; ++i) {
        indices_list.push_back({static_cast<nsparse::term_t>(i)});
        values_list.push_back({1.0F});
    }
    auto vectors = create_float_vectors(indices_list, values_list);

    std::vector<nsparse::idx_t> doc_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8};

    // n_clusters = 0 should use sqrt(9) = 3
    auto clusters = nsparse::detail::RandomKMeans::train(&vectors, doc_ids, 0);

    ASSERT_EQ(clusters.size(), 3);
}

TEST(RandomKMeansTrain, all_docs_assigned_no_duplicates_no_missing) {
    auto vectors = create_float_vectors(
        {{0}, {1}, {0}, {1}, {0}}, {{1.0F}, {1.0F}, {0.9F}, {0.8F}, {0.7F}});

    std::vector<nsparse::idx_t> doc_ids = {0, 1, 2, 3, 4};

    auto clusters = nsparse::detail::RandomKMeans::train(&vectors, doc_ids, 2);

    // Collect all docs from clusters
    std::vector<nsparse::idx_t> all_docs;
    for (const auto& cluster : clusters) {
        for (const auto& doc : cluster) {
            all_docs.push_back(doc);
        }
    }

    std::sort(all_docs.begin(), all_docs.end());

    // No duplicates
    for (size_t i = 1; i < all_docs.size(); ++i) {
        ASSERT_NE(all_docs[i], all_docs[i - 1])
            << "Duplicate doc: " << all_docs[i];
    }

    // No missing
    std::vector<nsparse::idx_t> expected = {0, 1, 2, 3, 4};
    ASSERT_EQ(all_docs, expected);
}

TEST(RandomKMeansTrain, each_cluster_has_centroid_at_position_zero) {
    auto vectors = create_float_vectors({{0}, {1}, {2}, {3}},
                                        {{1.0F}, {1.0F}, {1.0F}, {1.0F}});

    std::vector<nsparse::idx_t> doc_ids = {0, 1, 2, 3};

    auto clusters = nsparse::detail::RandomKMeans::train(&vectors, doc_ids, 2);

    // Each cluster should have at least one element (the centroid)
    for (const auto& cluster : clusters) {
        ASSERT_GE(cluster.size(), 1);
    }
}
