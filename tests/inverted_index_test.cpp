/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/inverted_index.h"

#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "nsparse/index.h"
#include "nsparse/io/buffered_io.h"
#include "nsparse/io/index_io.h"
#include "nsparse/types.h"

namespace nsparse {
namespace {

// Helper to add docs from map format: {{term: value, ...}, ...}
void add_docs(InvertedIndex& index,
              const std::vector<std::map<int, float>>& docs) {
    std::vector<idx_t> indptr;
    std::vector<term_t> indices;
    std::vector<float> values;

    indptr.push_back(0);
    for (const auto& doc : docs) {
        for (const auto& [term, value] : doc) {
            indices.push_back(static_cast<term_t>(term));
            values.push_back(value);
        }
        indptr.push_back(static_cast<idx_t>(indices.size()));
    }

    index.add(static_cast<idx_t>(docs.size()), indptr.data(), indices.data(),
              values.data());
}

// ============== Constructor tests ==============

TEST(InvertedIndexConstructor, sets_dimension) {
    InvertedIndex index(100);
    EXPECT_EQ(index.get_dimension(), 100);
}

TEST(InvertedIndexConstructor, id_returns_INVT) {
    InvertedIndex index(5);
    std::array<char, 4> expected = {'I', 'N', 'V', 'T'};
    EXPECT_EQ(index.id(), expected);
}

// ============== add() tests ==============

TEST(InvertedIndexAdd, add_single_vector) {
    InvertedIndex index(5);
    add_docs(index, {{{0, 1.0F}, {1, 0.5F}}});
    // InvertedIndex doesn't expose get_vectors(), so num_vectors() returns 0.
    // Just verify no crash and build works after add.
    index.build();
}

TEST(InvertedIndexAdd, add_multiple_vectors) {
    InvertedIndex index(5);
    add_docs(index, {{{0, 1.0F}, {1, 0.5F}}, {{2, 0.8F}, {3, 0.6F}}});
    index.build();
}

// ============== search() tests ==============

TEST(InvertedIndexSearch, search_returns_empty_when_not_built) {
    InvertedIndex index(5);
    Index* idx = &index;

    std::vector<idx_t> query_indptr = {0, 2};
    std::vector<term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 0.5F};
    std::vector<idx_t> labels(5, -1);
    std::vector<float> distances(5, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 5, distances.data(), labels.data());

    for (const auto& label : labels) {
        EXPECT_EQ(label, -1);
    }
}

TEST(InvertedIndexSearch, search_finds_matching_doc) {
    InvertedIndex index(3);
    Index* idx = &index;

    add_docs(index, {{{0, 1.0F}, {1, 0.5F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(1, -1);
    std::vector<float> distances(1, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data());

    EXPECT_EQ(labels[0], 0);
    EXPECT_FLOAT_EQ(distances[0], 1.0F);
}

TEST(InvertedIndexSearch, search_multiple_queries) {
    InvertedIndex index(4);
    Index* idx = &index;

    add_docs(index, {{{0, 1.0F}, {1, 0.5F}}, {{2, 0.8F}, {3, 0.6F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1, 2};
    std::vector<term_t> query_indices = {0, 2};
    std::vector<float> query_values = {1.0F, 1.0F};
    std::vector<idx_t> labels(2, -1);
    std::vector<float> distances(2, -1.0F);

    idx->search(2, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data());

    EXPECT_EQ(labels[0], 0);
    EXPECT_EQ(labels[1], 1);
}

TEST(InvertedIndexSearch, search_respects_k_limit) {
    InvertedIndex index(3);
    Index* idx = &index;

    add_docs(index, {{{0, 1.0F}}, {{0, 0.9F}}, {{0, 0.8F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(2, -1);
    std::vector<float> distances(2, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 2, distances.data(), labels.data());

    // Top 2 by score: doc0 (1.0), doc1 (0.9)
    EXPECT_EQ(labels[0], 0);
    EXPECT_EQ(labels[1], 1);
}

TEST(InvertedIndexSearch, search_returns_results_sorted_by_score) {
    InvertedIndex index(3);
    Index* idx = &index;

    // doc0: term0=0.3, doc1: term0=1.0, doc2: term0=0.5
    add_docs(index, {{{0, 0.3F}}, {{0, 1.0F}}, {{0, 0.5F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(3, -1);
    std::vector<float> distances(3, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 3, distances.data(), labels.data());

    // Sorted by score descending: doc1 (1.0), doc2 (0.5), doc0 (0.3)
    EXPECT_EQ(labels[0], 1);
    EXPECT_EQ(labels[1], 2);
    EXPECT_EQ(labels[2], 0);
}

TEST(InvertedIndexSearch, search_with_no_matching_term) {
    InvertedIndex index(5);
    Index* idx = &index;

    add_docs(index, {{{0, 1.0F}, {1, 0.5F}}});
    index.build();

    // Query with term 3 which no doc has
    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {3};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(1, -1);
    std::vector<float> distances(1, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data());

    EXPECT_EQ(labels[0], -1);
}

TEST(InvertedIndexSearch, search_multi_term_dot_product) {
    InvertedIndex index(4);
    Index* idx = &index;

    // doc0: term0=1.0, term1=0.5
    // doc1: term0=0.8, term2=0.6
    // doc2: term1=0.9, term3=0.7
    add_docs(index, {{{0, 1.0F}, {1, 0.5F}},
                     {{0, 0.8F}, {2, 0.6F}},
                     {{1, 0.9F}, {3, 0.7F}}});
    index.build();

    // Query: term0=1.0, term1=0.8
    // Scores: doc0 = 1.0*1.0 + 0.5*0.8 = 1.4
    //         doc1 = 0.8*1.0 = 0.8
    //         doc2 = 0.9*0.8 = 0.72
    std::vector<idx_t> query_indptr = {0, 2};
    std::vector<term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 0.8F};
    std::vector<idx_t> labels(3, -1);
    std::vector<float> distances(3, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 3, distances.data(), labels.data());

    EXPECT_EQ(labels[0], 0);
    EXPECT_EQ(labels[1], 1);
    EXPECT_EQ(labels[2], 2);
    EXPECT_FLOAT_EQ(distances[0], 1.4F);
    EXPECT_FLOAT_EQ(distances[1], 0.8F);
    EXPECT_FLOAT_EQ(distances[2], 0.72F);
}

TEST(InvertedIndexSearch, search_k_larger_than_num_docs) {
    InvertedIndex index(3);
    Index* idx = &index;

    add_docs(index, {{{0, 1.0F}}, {{0, 0.5F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(5, -1);
    std::vector<float> distances(5, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 5, distances.data(), labels.data());

    EXPECT_EQ(labels[0], 0);
    EXPECT_EQ(labels[1], 1);
    // Remaining padded with INVALID_IDX (-1)
    EXPECT_EQ(labels[2], -1);
    EXPECT_EQ(labels[3], -1);
    EXPECT_EQ(labels[4], -1);
}

// ============== write_index/read_index tests ==============

TEST(InvertedIndexIO, write_and_read_empty_index) {
    InvertedIndex original(100);

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    Index* loaded = read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_dimension(), 100);
    EXPECT_EQ(loaded->id(), original.id());

    delete loaded;
}

TEST(InvertedIndexIO, write_and_read_built_index) {
    InvertedIndex original(4);

    add_docs(original, {{{0, 1.0F}, {1, 0.5F}}, {{2, 0.8F}, {3, 0.6F}}});
    original.build();

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    Index* loaded = read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_dimension(), 4);

    delete loaded;
}

TEST(InvertedIndexIO, write_and_read_search_produces_same_results) {
    InvertedIndex original(4);

    // doc0: term0=1.0, term1=0.5
    // doc1: term0=0.8, term2=0.6
    // doc2: term1=0.9, term3=0.7
    add_docs(original, {{{0, 1.0F}, {1, 0.5F}},
                        {{0, 0.8F}, {2, 0.6F}},
                        {{1, 0.9F}, {3, 0.7F}}});
    original.build();

    // Search on original
    std::vector<idx_t> query_indptr = {0, 2};
    std::vector<term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 0.8F};
    std::vector<idx_t> labels_original(3, -1);
    std::vector<float> distances_original(3, -1.0F);

    Index* idx_original = &original;
    idx_original->search(1, query_indptr.data(), query_indices.data(),
                         query_values.data(), 3, distances_original.data(),
                         labels_original.data());

    // Write and read
    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    Index* loaded = read_index(&reader);

    // Search on loaded
    std::vector<idx_t> labels_loaded(3, -1);
    std::vector<float> distances_loaded(3, -1.0F);
    loaded->search(1, query_indptr.data(), query_indices.data(),
                   query_values.data(), 3, distances_loaded.data(),
                   labels_loaded.data());

    // Results should be identical
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(labels_original[i], labels_loaded[i]);
        EXPECT_FLOAT_EQ(distances_original[i], distances_loaded[i]);
    }

    delete loaded;
}

TEST(InvertedIndexIO, write_and_read_with_empty_posting_lists) {
    InvertedIndex original(5);

    // Only terms 0 and 4 have docs, terms 1-3 are empty
    add_docs(original, {{{0, 1.0F}}, {{4, 0.5F}}});
    original.build();

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    Index* loaded = read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_dimension(), 5);

    // Verify search still works — query term 0
    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(1, -1);
    std::vector<float> distances(1, -1.0F);

    loaded->search(1, query_indptr.data(), query_indices.data(),
                   query_values.data(), 1, distances.data(), labels.data());

    EXPECT_EQ(labels[0], 0);

    delete loaded;
}

TEST(InvertedIndexIO, write_and_read_multiple_docs_per_term) {
    InvertedIndex original(3);

    // 4 docs all sharing term 0
    add_docs(original, {{{0, 1.0F}}, {{0, 0.9F}}, {{0, 0.8F}}, {{0, 0.7F}}});
    original.build();

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    Index* loaded = read_index(&reader);

    // Search for all 4 docs
    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(4, -1);
    std::vector<float> distances(4, -1.0F);

    loaded->search(1, query_indptr.data(), query_indices.data(),
                   query_values.data(), 4, distances.data(), labels.data());

    // All 4 docs should be found, sorted by score descending
    EXPECT_EQ(labels[0], 0);
    EXPECT_EQ(labels[1], 1);
    EXPECT_EQ(labels[2], 2);
    EXPECT_EQ(labels[3], 3);

    delete loaded;
}

TEST(InvertedIndexIO, roundtrip_preserves_fourcc) {
    InvertedIndex original(10);

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    Index* loaded = read_index(&reader);

    EXPECT_EQ(loaded->id(), (std::array<char, 4>{'I', 'N', 'V', 'T'}));

    delete loaded;
}

}  // namespace
}  // namespace nsparse
