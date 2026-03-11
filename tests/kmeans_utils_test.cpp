/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/cluster/kmeans_utils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace {

// Helper to create SparseVectors with float values
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

TEST(MapDocsToClusters, throws_on_null_vectors) {
    std::vector<nsparse::idx_t> docs = {0, 1};
    std::vector<std::vector<nsparse::idx_t>> clusters = {{0}, {1}};
    ASSERT_THROW(nsparse::detail::map_docs_to_clusters(nullptr, docs, clusters),
                 std::runtime_error);
}

TEST(MapDocsToClusters, empty_docs_no_change) {
    auto vectors =
        create_float_vectors({{0, 1}, {1, 2}}, {{1.0F, 2.0F}, {3.0F, 4.0F}});
    std::vector<nsparse::idx_t> docs = {};
    std::vector<std::vector<nsparse::idx_t>> clusters = {{0}, {1}};

    nsparse::detail::map_docs_to_clusters(&vectors, docs, clusters);

    ASSERT_EQ(clusters[0].size(), 1);
    ASSERT_EQ(clusters[1].size(), 1);
}

TEST(MapDocsToClusters, empty_clusters_no_crash) {
    auto vectors = create_float_vectors({{0, 1}}, {{1.0F, 2.0F}});
    std::vector<nsparse::idx_t> docs = {0};
    std::vector<std::vector<nsparse::idx_t>> clusters = {};

    // Should not crash with empty clusters
    nsparse::detail::map_docs_to_clusters(&vectors, docs, clusters);
    ASSERT_TRUE(clusters.empty());
}

TEST(MapDocsToClusters, single_cluster_all_docs_assigned) {
    // 3 vectors: centroid at index 0, docs 1 and 2 to be assigned
    auto vectors = create_float_vectors(
        {{0, 1}, {0, 1}, {0, 1}}, {{1.0F, 0.0F}, {0.8F, 0.2F}, {0.9F, 0.1F}});

    std::vector<nsparse::idx_t> docs = {1, 2};
    std::vector<std::vector<nsparse::idx_t>> clusters = {{0}};  // centroid is 0

    nsparse::detail::map_docs_to_clusters(&vectors, docs, clusters);

    ASSERT_EQ(clusters[0].size(), 3);  // centroid + 2 docs
    ASSERT_EQ(clusters[0][0], 0);      // centroid unchanged
    // docs 1 and 2 should be added
    ASSERT_TRUE(std::find(clusters[0].begin(), clusters[0].end(), 1) !=
                clusters[0].end());
    ASSERT_TRUE(std::find(clusters[0].begin(), clusters[0].end(), 2) !=
                clusters[0].end());
}

TEST(MapDocsToClusters, two_clusters_docs_assigned_to_nearest) {
    // Create vectors where similarity determines cluster assignment
    // Centroid 0: strong on term 0
    // Centroid 1: strong on term 1
    // Doc 2: similar to centroid 0
    // Doc 3: similar to centroid 1
    auto vectors = create_float_vectors({{0}, {1}, {0}, {1}},
                                        {{1.0F}, {1.0F}, {0.9F}, {0.8F}});

    std::vector<nsparse::idx_t> docs = {2, 3};
    std::vector<std::vector<nsparse::idx_t>> clusters = {{0}, {1}};

    nsparse::detail::map_docs_to_clusters(&vectors, docs, clusters);

    // Doc 2 (term 0) has similarity with centroid 0 (term 0), 0 with centroid 1
    // Doc 3 (term 1) has similarity with centroid 1 (term 1), 0 with centroid 0
    ASSERT_EQ(clusters[0].size(), 2);  // centroid 0 + doc 2
    ASSERT_EQ(clusters[1].size(), 2);  // centroid 1 + doc 3
    ASSERT_EQ(clusters[0][1], 2);
    ASSERT_EQ(clusters[1][1], 3);
}

TEST(MapDocsToClusters, doc_assigned_to_highest_similarity_cluster) {
    // Centroid 0: terms {0, 1} with values {1.0, 1.0}
    // Centroid 1: terms {2, 3} with values {1.0, 1.0}
    // Doc 2: terms {0, 1, 2} with values {0.5, 0.5, 0.1} - more similar to c0
    auto vectors =
        create_float_vectors({{0, 1}, {2, 3}, {0, 1, 2}},
                             {{1.0F, 1.0F}, {1.0F, 1.0F}, {0.5F, 0.5F, 0.1F}});

    std::vector<nsparse::idx_t> docs = {2};
    std::vector<std::vector<nsparse::idx_t>> clusters = {{0}, {1}};

    nsparse::detail::map_docs_to_clusters(&vectors, docs, clusters);

    // Doc 2 has dot product 1.0 with centroid 0, 0.1 with centroid 1
    ASSERT_EQ(clusters[0].size(), 2);  // centroid + doc 2
    ASSERT_EQ(clusters[1].size(), 1);  // just centroid
    ASSERT_EQ(clusters[0][1], 2);
}

TEST(MapDocsToClusters, centroid_position_preserved) {
    auto vectors =
        create_float_vectors({{0}, {1}, {0}}, {{1.0F}, {1.0F}, {0.5F}});

    std::vector<nsparse::idx_t> docs = {2};
    std::vector<std::vector<nsparse::idx_t>> clusters = {{0}, {1}};

    nsparse::detail::map_docs_to_clusters(&vectors, docs, clusters);

    // Centroids should remain at position 0
    ASSERT_EQ(clusters[0][0], 0);
    ASSERT_EQ(clusters[1][0], 1);
}

TEST(MapDocsToClusters,
     no_duplicate_and_no_missing_docs_when_centroids_in_docs) {
    // Create 5 vectors: 0 and 1 are centroids, 2-4 are regular docs
    // Centroid 0: strong on term 0
    // Centroid 1: strong on term 1
    // Doc 2: similar to centroid 0
    // Doc 3: similar to centroid 1
    // Doc 4: similar to centroid 0
    auto vectors = create_float_vectors(
        {{0}, {1}, {0}, {1}, {0}}, {{1.0F}, {1.0F}, {0.9F}, {0.8F}, {0.7F}});

    // Include centroids (0, 1) in the docs list - this tests the duplicate
    // avoidance logic
    std::vector<nsparse::idx_t> docs = {0, 1, 2, 3, 4};
    std::vector<std::vector<nsparse::idx_t>> clusters = {{0}, {1}};

    nsparse::detail::map_docs_to_clusters(&vectors, docs, clusters);

    // Collect all docs from all clusters
    std::vector<nsparse::idx_t> all_docs_in_clusters;
    for (const auto& cluster : clusters) {
        for (const auto& doc : cluster) {
            all_docs_in_clusters.push_back(doc);
        }
    }

    // Sort for easier comparison
    std::sort(all_docs_in_clusters.begin(), all_docs_in_clusters.end());

    // Verify no duplicates: after sorting, no adjacent elements should be equal
    for (size_t i = 1; i < all_docs_in_clusters.size(); ++i) {
        ASSERT_NE(all_docs_in_clusters[i], all_docs_in_clusters[i - 1])
            << "Duplicate doc found: " << all_docs_in_clusters[i];
    }

    // Verify no missing docs: all input docs should be present
    std::vector<nsparse::idx_t> expected_docs = {0, 1, 2, 3, 4};
    ASSERT_EQ(all_docs_in_clusters.size(), expected_docs.size());
    ASSERT_EQ(all_docs_in_clusters, expected_docs);
}
