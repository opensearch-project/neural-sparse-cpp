/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/io/seismic_invlists_writer.h"

#include <gtest/gtest.h>

#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
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

}  // namespace

TEST(SeismicInvertedListsWriter, serialize_deserialize_empty) {
    std::vector<nsparse::InvertedListClusters> original;
    nsparse::SeismicInvertedListsWriter writer(original);

    nsparse::BufferedIOWriter io_writer;
    writer.serialize(&io_writer);

    nsparse::BufferedIOReader io_reader(io_writer.data());
    nsparse::SeismicInvertedListsWriter reader;
    reader.deserialize(&io_reader);

    auto result = reader.release();
    ASSERT_TRUE(result.empty());
}

TEST(SeismicInvertedListsWriter, serialize_deserialize_single_cluster) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2}};
    nsparse::InvertedListClusters clusters(docs);

    std::vector<nsparse::InvertedListClusters> original;
    original.push_back(std::move(clusters));

    nsparse::SeismicInvertedListsWriter writer(original);

    nsparse::BufferedIOWriter io_writer;
    writer.serialize(&io_writer);

    nsparse::BufferedIOReader io_reader(io_writer.data());
    nsparse::SeismicInvertedListsWriter reader;
    reader.deserialize(&io_reader);

    auto result = reader.release();
    ASSERT_EQ(result.size(), 1);

    auto doc_span0 = result[0].get_docs(0);
    ASSERT_EQ(doc_span0.size(), 2);
    ASSERT_EQ(doc_span0[0], 0);
    ASSERT_EQ(doc_span0[1], 1);

    auto doc_span1 = result[0].get_docs(1);
    ASSERT_EQ(doc_span1.size(), 1);
    ASSERT_EQ(doc_span1[0], 2);
}

TEST(SeismicInvertedListsWriter, serialize_deserialize_multiple_clusters) {
    std::vector<nsparse::InvertedListClusters> original;

    std::vector<std::vector<nsparse::idx_t>> docs1 = {{0, 1}};
    original.emplace_back(docs1);

    std::vector<std::vector<nsparse::idx_t>> docs2 = {{2, 3, 4}};
    original.emplace_back(docs2);

    std::vector<std::vector<nsparse::idx_t>> docs3 = {{5}};
    original.emplace_back(docs3);

    nsparse::SeismicInvertedListsWriter writer(original);

    nsparse::BufferedIOWriter io_writer;
    writer.serialize(&io_writer);

    nsparse::BufferedIOReader io_reader(io_writer.data());
    nsparse::SeismicInvertedListsWriter reader;
    reader.deserialize(&io_reader);

    auto result = reader.release();
    ASSERT_EQ(result.size(), 3);

    ASSERT_EQ(result[0].get_docs(0).size(), 2);
    ASSERT_EQ(result[1].get_docs(0).size(), 3);
    ASSERT_EQ(result[2].get_docs(0).size(), 1);
}

TEST(SeismicInvertedListsWriter, serialize_deserialize_with_summaries) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2}};
    nsparse::InvertedListClusters clusters(docs);

    auto vectors = create_float_vectors(
        {{0, 1}, {0, 2}, {1, 2}}, {{1.0F, 2.0F}, {1.5F, 1.0F}, {3.0F, 2.0F}});
    clusters.summarize(&vectors, 1.0F);

    std::vector<nsparse::InvertedListClusters> original;
    original.push_back(std::move(clusters));

    nsparse::SeismicInvertedListsWriter writer(original);

    nsparse::BufferedIOWriter io_writer;
    writer.serialize(&io_writer);

    nsparse::BufferedIOReader io_reader(io_writer.data());
    nsparse::SeismicInvertedListsWriter reader;
    reader.deserialize(&io_reader);

    auto result = reader.release();
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0].cluster_size(), 2);
}

// summaries_only omits the doc-id membership: the stream must be smaller yet
// still round-trip through the unchanged reader with the summaries intact.
TEST(SeismicInvertedListsWriter, summaries_only_drops_membership_keeps_summaries) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2}};
    nsparse::InvertedListClusters clusters(docs);
    auto vectors = create_float_vectors(
        {{0, 1}, {0, 2}, {1, 2}}, {{1.0F, 2.0F}, {1.5F, 1.0F}, {3.0F, 2.0F}});
    clusters.summarize(&vectors, 1.0F);
    std::vector<nsparse::InvertedListClusters> original;
    original.push_back(std::move(clusters));

    nsparse::BufferedIOWriter full_writer;
    nsparse::SeismicInvertedListsWriter(original).serialize(&full_writer);
    nsparse::BufferedIOWriter summaries_writer;
    nsparse::SeismicInvertedListsWriter(original, /*summaries_only=*/true)
        .serialize(&summaries_writer);

    // Dropping docs_/offsets_ makes the summaries-only stream strictly smaller.
    EXPECT_LT(summaries_writer.size(), full_writer.size());

    // Both streams round-trip through the same (unchanged) reader.
    nsparse::BufferedIOReader full_reader(full_writer.data());
    nsparse::SeismicInvertedListsWriter full_rt;
    full_rt.deserialize(&full_reader);
    auto full_lists = full_rt.release();

    nsparse::BufferedIOReader summaries_reader(summaries_writer.data());
    nsparse::SeismicInvertedListsWriter summaries_rt;
    summaries_rt.deserialize(&summaries_reader);
    auto summaries_lists = summaries_rt.release();

    ASSERT_EQ(full_lists.size(), 1U);
    ASSERT_EQ(summaries_lists.size(), 1U);
    // The cluster count (a stored scalar) survives; only membership was dropped.
    EXPECT_EQ(summaries_lists[0].cluster_size(), full_lists[0].cluster_size());

    // Summaries are byte-preserved: the same query scores both identically.
    std::vector<nsparse::term_t> q_idx = {0, 1, 2};
    std::vector<float> q_val = {1.0F, 1.0F, 1.0F};
    std::vector<float> full_scores;
    std::vector<float> summaries_scores;
    full_lists[0].score_summaries_transposed(
        q_idx.data(), reinterpret_cast<const uint8_t*>(q_val.data()),
        q_idx.size(), full_scores);
    summaries_lists[0].score_summaries_transposed(
        q_idx.data(), reinterpret_cast<const uint8_t*>(q_val.data()),
        q_idx.size(), summaries_scores);
    EXPECT_EQ(summaries_scores, full_scores);
}

// release() hands over what deserialize() read; it is not a way to take back a
// vector handed in for writing, which the writer only borrows.
TEST(SeismicInvertedListsWriter, release_moves_deserialized_data) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}};
    nsparse::InvertedListClusters clusters(docs);

    std::vector<nsparse::InvertedListClusters> original;
    original.push_back(std::move(clusters));

    nsparse::BufferedIOWriter io_writer;
    nsparse::SeismicInvertedListsWriter(original).serialize(&io_writer);

    nsparse::BufferedIOReader io_reader(io_writer.data());
    nsparse::SeismicInvertedListsWriter reader;
    reader.deserialize(&io_reader);

    auto released = reader.release();
    ASSERT_EQ(released.size(), 1);
    ASSERT_EQ(released[0].get_docs(0).size(), 2);
    // Moved out, so the writer no longer holds it.
    ASSERT_TRUE(reader.release().empty());
}

// Writing borrows: the caller's vector is left intact, not emptied.
TEST(SeismicInvertedListsWriter, serialize_leaves_the_source_intact) {
    std::vector<std::vector<nsparse::idx_t>> docs = {{0, 1}, {2}};
    std::vector<nsparse::InvertedListClusters> original;
    original.emplace_back(docs);

    nsparse::BufferedIOWriter io_writer;
    nsparse::SeismicInvertedListsWriter writer(original);
    writer.serialize(&io_writer);

    ASSERT_EQ(original.size(), 1);
    ASSERT_EQ(original[0].get_docs(0).size(), 2);
    ASSERT_GT(io_writer.size(), 0);
}
