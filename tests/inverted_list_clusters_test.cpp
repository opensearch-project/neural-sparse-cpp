/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/cluster/inverted_list_clusters.h"

#include <gtest/gtest.h>

#include <vector>

#include "nsparse/io/buffered_io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace {

nsparse::SparseVectors create_float_vectors(
    const std::vector<std::vector<nsparse::term_t>>& indices_list,
    const std::vector<std::vector<float>>& values_list, size_t dimension = 10) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = dimension});
    for (size_t i = 0; i < indices_list.size(); ++i) {
        const auto& indices = indices_list[i];
        const auto& values = values_list[i];
        vectors.add_vector(indices.data(), indices.size(),
                           reinterpret_cast<const uint8_t*>(values.data()),
                           values.size() * sizeof(float));
    }
    return vectors;
}

nsparse::SparseVectors create_uint8_vectors(
    const std::vector<std::vector<nsparse::term_t>>& indices_list,
    const std::vector<std::vector<uint8_t>>& values_list,
    size_t dimension = 10) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U8, .dimension = dimension});
    for (size_t i = 0; i < indices_list.size(); ++i) {
        const auto& indices = indices_list[i];
        const auto& values = values_list[i];
        vectors.add_vector(indices.data(), indices.size(), values.data(),
                           values.size());
    }
    return vectors;
}

nsparse::SparseVectors create_uint16_vectors(
    const std::vector<std::vector<nsparse::term_t>>& indices_list,
    const std::vector<std::vector<uint16_t>>& values_list,
    size_t dimension = 10) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U16, .dimension = dimension});
    for (size_t i = 0; i < indices_list.size(); ++i) {
        const auto& indices = indices_list[i];
        const auto& values = values_list[i];
        vectors.add_vector(indices.data(), indices.size(),
                           reinterpret_cast<const uint8_t*>(values.data()),
                           values.size() * sizeof(uint16_t));
    }
    return vectors;
}

}  // namespace

// Constructor tests
TEST(InvertedListClusters, default_constructor) {
    nsparse::InvertedListClusters clusters;
    ASSERT_EQ(clusters.cluster_size(), 0);
}

TEST(InvertedListClusters, constructor_with_empty_docs) {
    std::vector<std::vector<nsparse::idx_t>> docs = {};
    nsparse::InvertedListClusters clusters(docs);
    ASSERT_EQ(clusters.cluster_size(), 0);
}

TEST(InvertedListClusters, constructor_with_single_cluster) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1, 2}};
    nsparse::InvertedListClusters clusters(docs);

    auto doc_span = clusters.get_docs(0);
    ASSERT_EQ(doc_span.size(), 3);
    ASSERT_EQ(doc_span[0], 0);
    ASSERT_EQ(doc_span[1], 1);
    ASSERT_EQ(doc_span[2], 2);
}

TEST(InvertedListClusters, constructor_with_multiple_clusters) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2, 3, 4}, {5}};
    nsparse::InvertedListClusters clusters(docs);

    auto doc_span0 = clusters.get_docs(0);
    ASSERT_EQ(doc_span0.size(), 2);
    ASSERT_EQ(doc_span0[0], 0);
    ASSERT_EQ(doc_span0[1], 1);

    auto doc_span1 = clusters.get_docs(1);
    ASSERT_EQ(doc_span1.size(), 3);
    ASSERT_EQ(doc_span1[0], 2);
    ASSERT_EQ(doc_span1[1], 3);
    ASSERT_EQ(doc_span1[2], 4);

    auto doc_span2 = clusters.get_docs(2);
    ASSERT_EQ(doc_span2.size(), 1);
    ASSERT_EQ(doc_span2[0], 5);
}

TEST(InvertedListClusters, constructor_with_empty_cluster) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {}, {2}};
    nsparse::InvertedListClusters clusters(docs);

    auto doc_span0 = clusters.get_docs(0);
    ASSERT_EQ(doc_span0.size(), 2);

    auto doc_span1 = clusters.get_docs(1);
    ASSERT_EQ(doc_span1.size(), 0);

    auto doc_span2 = clusters.get_docs(2);
    ASSERT_EQ(doc_span2.size(), 1);
}

// Copy constructor tests
TEST(InvertedListClusters, copy_constructor_without_summaries) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2, 3}};
    nsparse::InvertedListClusters original(docs);

    nsparse::InvertedListClusters copy(original);

    auto doc_span = copy.get_docs(0);
    ASSERT_EQ(doc_span.size(), 2);
    ASSERT_EQ(doc_span[0], 0);
    ASSERT_EQ(doc_span[1], 1);
}

TEST(InvertedListClusters, copy_constructor_with_summaries) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2}};
    nsparse::InvertedListClusters original(docs);

    auto vectors = create_float_vectors(
        {{0, 1}, {0, 2}, {1, 2}}, {{1.0F, 2.0F}, {1.5F, 1.0F}, {3.0F, 2.0F}});
    original.summarize(&vectors, 1.0F);

    nsparse::InvertedListClusters copy(original);

    ASSERT_EQ(copy.cluster_size(), 2);
    ASSERT_EQ(copy.summaries().num_vectors(), 2);
}

// Copy assignment tests
TEST(InvertedListClusters, copy_assignment_without_summaries) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2, 3}};
    nsparse::InvertedListClusters original(docs);

    nsparse::InvertedListClusters copy;
    copy = original;

    auto doc_span = copy.get_docs(1);
    ASSERT_EQ(doc_span.size(), 2);
    ASSERT_EQ(doc_span[0], 2);
    ASSERT_EQ(doc_span[1], 3);
}

TEST(InvertedListClusters, copy_assignment_self) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}};
    nsparse::InvertedListClusters clusters(docs);

    clusters = clusters;

    auto doc_span = clusters.get_docs(0);
    ASSERT_EQ(doc_span.size(), 2);
}

// Summarize tests
TEST(InvertedListClusters, summarize_float_single_cluster) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}};
    nsparse::InvertedListClusters clusters(docs);

    // Vector 0: term 0 -> 1.0, term 1 -> 2.0
    // Vector 1: term 0 -> 3.0, term 1 -> 1.0
    // Summary should have max: term 0 -> 3.0, term 1 -> 2.0
    auto vectors =
        create_float_vectors({{0, 1}, {0, 1}}, {{1.0F, 2.0F}, {3.0F, 1.0F}});

    clusters.summarize(&vectors, 1.0F);

    ASSERT_EQ(clusters.cluster_size(), 1);
    const auto& summaries = clusters.summaries();
    ASSERT_EQ(summaries.num_vectors(), 1);

    auto dense = summaries.get_dense_vector_float(0);
    ASSERT_FLOAT_EQ(dense[0], 3.0F);  // max(1.0, 3.0)
    ASSERT_FLOAT_EQ(dense[1], 2.0F);  // max(2.0, 1.0)
}

TEST(InvertedListClusters, summarize_float_multiple_clusters) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2}};
    nsparse::InvertedListClusters clusters(docs);

    // Cluster 0: vectors 0,1 -> term 0: max(1.0, 2.0)=2.0
    // Cluster 1: vector 2 -> term 1: 3.0
    auto vectors =
        create_float_vectors({{0}, {0}, {1}}, {{1.0F}, {2.0F}, {3.0F}});

    clusters.summarize(&vectors, 1.0F);

    ASSERT_EQ(clusters.cluster_size(), 2);

    auto dense0 = clusters.summaries().get_dense_vector_float(0);
    ASSERT_FLOAT_EQ(dense0[0], 2.0F);  // max(1.0, 2.0)

    auto dense1 = clusters.summaries().get_dense_vector_float(1);
    ASSERT_FLOAT_EQ(dense1[1], 3.0F);
}

TEST(InvertedListClusters, summarize_uint8) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}};
    nsparse::InvertedListClusters clusters(docs);

    // Vector 0: term 0 -> 10, term 1 -> 20
    // Vector 1: term 0 -> 30, term 1 -> 10
    // Summary: term 0 -> max(10,30)=30, term 1 -> max(20,10)=20
    auto vectors = create_uint8_vectors({{0, 1}, {0, 1}}, {{10, 20}, {30, 10}});

    clusters.summarize(&vectors, 1.0F);

    ASSERT_EQ(clusters.cluster_size(), 1);

    auto dense = clusters.summaries().get_dense_vector(0);
    ASSERT_EQ(dense[0], 30);  // max(10, 30)
    ASSERT_EQ(dense[1], 20);  // max(20, 10)
}

TEST(InvertedListClusters, summarize_uint16) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}};
    nsparse::InvertedListClusters clusters(docs);

    // Vector 0: term 0 -> 100, term 1 -> 200
    // Vector 1: term 0 -> 300, term 1 -> 100
    // Summary: term 0 -> max(100,300)=300, term 1 -> max(200,100)=200
    auto vectors =
        create_uint16_vectors({{0, 1}, {0, 1}}, {{100, 200}, {300, 100}});

    clusters.summarize(&vectors, 1.0F);

    ASSERT_EQ(clusters.cluster_size(), 1);

    auto dense = clusters.summaries().get_dense_vector(0);
    auto* values = reinterpret_cast<const uint16_t*>(dense.data());
    ASSERT_EQ(values[0], 300);  // max(100, 300)
    ASSERT_EQ(values[1], 200);  // max(200, 100)
}

TEST(InvertedListClusters, summarize_with_alpha_pruning) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0}};
    nsparse::InvertedListClusters clusters(docs);

    // Vector with terms: 0->10.0, 1->5.0, 2->3.0, 3->2.0
    // Total sum = 20.0
    // With alpha=0.5, should keep terms until cumsum >= 10.0
    // Sorted by value desc: term0(10), term1(5), term2(3), term3(2)
    // cumsum after term0: 10/20 = 0.5 >= 0.5, so keep only term0
    auto vectors =
        create_float_vectors({{0, 1, 2, 3}}, {{10.0F, 5.0F, 3.0F, 2.0F}});

    clusters.summarize(&vectors, 0.5F);

    ASSERT_EQ(clusters.cluster_size(), 1);
    const auto& summaries = clusters.summaries();
    auto dense = summaries.get_dense_vector_float(0);
    // Only term 0 should be non-zero
    ASSERT_GT(dense[0], 0.0F);
}

TEST(InvertedListClusters, summarize_replaces_existing) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0}};
    nsparse::InvertedListClusters clusters(docs);

    auto vectors1 = create_float_vectors({{0}}, {{1.0F}});
    clusters.summarize(&vectors1, 1.0F);
    ASSERT_EQ(clusters.cluster_size(), 1);

    auto vectors2 = create_float_vectors({{0}, {1}}, {{2.0F}, {3.0F}});
    std::vector<std::vector<nsparse::idx_t>> docs2 = {{0, 1}};
    nsparse::InvertedListClusters clusters2(docs2);
    clusters2.summarize(&vectors2, 1.0F);
    ASSERT_EQ(clusters2.cluster_size(), 1);
}

// Serialization tests
TEST(InvertedListClusters, serialize_deserialize_empty) {
    nsparse::InvertedListClusters original;

    nsparse::BufferedIOWriter writer;
    original.serialize(&writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::InvertedListClusters loaded;
    loaded.deserialize(&reader);

    ASSERT_EQ(loaded.cluster_size(), 0);
}

TEST(InvertedListClusters, serialize_deserialize_without_summaries) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1, 2}, {3, 4}};
    nsparse::InvertedListClusters original(docs);

    nsparse::BufferedIOWriter writer;
    original.serialize(&writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::InvertedListClusters loaded;
    loaded.deserialize(&reader);

    ASSERT_EQ(loaded.cluster_size(), 0);  // No summaries

    auto doc_span0 = loaded.get_docs(0);
    ASSERT_EQ(doc_span0.size(), 3);
    ASSERT_EQ(doc_span0[0], 0);
    ASSERT_EQ(doc_span0[1], 1);
    ASSERT_EQ(doc_span0[2], 2);

    auto doc_span1 = loaded.get_docs(1);
    ASSERT_EQ(doc_span1.size(), 2);
    ASSERT_EQ(doc_span1[0], 3);
    ASSERT_EQ(doc_span1[1], 4);
}

TEST(InvertedListClusters, serialize_deserialize_with_summaries) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2}};
    nsparse::InvertedListClusters original(docs);

    auto vectors = create_float_vectors(
        {{0, 1}, {0, 2}, {1, 2}}, {{1.0F, 2.0F}, {1.5F, 1.0F}, {3.0F, 2.0F}});
    original.summarize(&vectors, 1.0F);

    nsparse::BufferedIOWriter writer;
    original.serialize(&writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::InvertedListClusters loaded;
    loaded.deserialize(&reader);

    ASSERT_EQ(loaded.cluster_size(), 2);

    auto doc_span0 = loaded.get_docs(0);
    ASSERT_EQ(doc_span0.size(), 2);

    auto doc_span1 = loaded.get_docs(1);
    ASSERT_EQ(doc_span1.size(), 1);
}

// Move constructor/assignment tests
TEST(InvertedListClusters, move_constructor) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2, 3}};
    nsparse::InvertedListClusters original(docs);

    nsparse::InvertedListClusters moved(std::move(original));

    auto doc_span = moved.get_docs(0);
    ASSERT_EQ(doc_span.size(), 2);
    ASSERT_EQ(doc_span[0], 0);
    ASSERT_EQ(doc_span[1], 1);
}

TEST(InvertedListClusters, move_assignment) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2, 3}};
    nsparse::InvertedListClusters original(docs);

    nsparse::InvertedListClusters moved;
    moved = std::move(original);

    auto doc_span = moved.get_docs(1);
    ASSERT_EQ(doc_span.size(), 2);
    ASSERT_EQ(doc_span[0], 2);
    ASSERT_EQ(doc_span[1], 3);
}
