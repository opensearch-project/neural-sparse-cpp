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

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nsparse/id_selector.h"
#include "nsparse/index.h"
#include "nsparse/io/buffered_io.h"
#include "nsparse/io/index_io.h"
#include "nsparse/types.h"
#include "nsparse/utils/csr_layout.h"

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
    index.build();
    EXPECT_EQ(index.num_vectors(), 1);
}

TEST(InvertedIndexAdd, add_multiple_vectors) {
    InvertedIndex index(5);
    add_docs(index, {{{0, 1.0F}, {1, 0.5F}}, {{2, 0.8F}, {3, 0.6F}}});
    index.build();
    EXPECT_EQ(index.num_vectors(), 2);
}

// A document with no terms is still a document, and build() dropping it from
// the posting lists does not change that.
TEST(InvertedIndexAdd, add_counts_empty_vectors) {
    InvertedIndex index(5);
    add_docs(index, {{{0, 1.0F}}, {}, {{2, 0.8F}}, {}});
    index.build();
    EXPECT_EQ(index.num_vectors(), 4);
}

// build() consumes the vectors, so a second call has nothing to work from --
// and for a mapped index it would drop the file the posting lists borrow from.
TEST(InvertedIndexAdd, build_without_vectors_is_a_no_op) {
    InvertedIndex index(5);
    ASSERT_NO_THROW(index.build());

    add_docs(index, {{{0, 1.0F}}});
    index.build();
    ASSERT_NO_THROW(index.build());
    EXPECT_EQ(index.num_vectors(), 1);
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

// Regression test: score_essential_terms must skip docs before window_base.
// When scorers span multiple windows (>4096 docs) and re-partitioning changes
// first_essential, a newly-essential scorer may have its cursor pointing at
// docs before the current window. Without the bounds check, this produces a
// negative slot index causing SIGBUS / out-of-bounds write.
TEST(InvertedIndexSearch, search_multi_window_no_crash) {
    // Use 2 terms. term0 has docs in early range, term1 has docs spread across
    // multiple windows (>4096). This forces the BMW algorithm to process
    // multiple scoring windows and potentially re-partition essential terms.
    constexpr int kDim = 2;
    constexpr int kNumDocs = 6000;  // > kScoreWindowSize (4096)
    InvertedIndex index(kDim);

    // Build docs: every doc has term0, every 3rd doc also has term1.
    // This creates different posting list lengths that affect max_score
    // partitioning.
    std::vector<std::map<int, float>> docs;
    docs.reserve(kNumDocs);
    for (int i = 0; i < kNumDocs; ++i) {
        std::map<int, float> doc;
        doc[0] = 0.1F + static_cast<float>(i % 10) * 0.01F;
        if (i % 3 == 0) {
            doc[1] = 0.5F + static_cast<float>(i % 7) * 0.05F;
        }
        docs.push_back(doc);
    }
    add_docs(index, docs);
    index.build();

    // Query both terms — forces multi-term scoring across multiple windows.
    std::vector<idx_t> query_indptr = {0, 2};
    std::vector<term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 1.0F};

    constexpr int kTopK = 10;
    std::vector<idx_t> labels(kTopK, -1);
    std::vector<float> distances(kTopK, -1.0F);

    Index* idx = &index;
    // The main assertion is that this does not crash (SIGBUS).
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), kTopK, distances.data(), labels.data());

    // Verify we got valid results.
    EXPECT_GE(labels[0], 0);
    EXPECT_GT(distances[0], 0.0F);

    // Results should be sorted by score descending.
    for (int i = 1; i < kTopK; ++i) {
        EXPECT_GE(distances[i - 1], distances[i]);
    }
}

// ============== IDSelector tests ==============

TEST(InvertedIndexSearch, search_with_id_selector_filters_results) {
    InvertedIndex index(3);
    Index* idx = &index;

    // doc0: term0=0.3, doc1: term0=1.0, doc2: term0=0.5
    add_docs(index, {{{0, 0.3F}}, {{0, 1.0F}}, {{0, 0.5F}}});
    index.build();

    // Only allow doc0 and doc2.
    std::vector<idx_t> allowed_ids = {0, 2};
    SetIDSelector selector(allowed_ids.size(), allowed_ids.data());
    SearchParameters params;
    params.set_id_selector(&selector);

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(3, -1);
    std::vector<float> distances(3, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 3, distances.data(), labels.data(),
                &params);

    // doc1 (highest score 1.0) is filtered out.
    EXPECT_EQ(labels[0], 2);
    EXPECT_EQ(labels[1], 0);
    EXPECT_EQ(labels[2], -1);
    EXPECT_FLOAT_EQ(distances[0], 0.5F);
    EXPECT_FLOAT_EQ(distances[1], 0.3F);
}

// The selector is read off the base, so any subclass works — the JNI layer
// passes a seismic one.
TEST(InvertedIndexSearch,
     search_honors_selector_on_search_parameters_subclass) {
    struct DerivedSearchParameters : SearchParameters {};

    InvertedIndex index(3);
    Index* idx = &index;

    add_docs(index, {{{0, 0.3F}}, {{0, 1.0F}}, {{0, 0.5F}}});
    index.build();

    std::vector<idx_t> allowed_ids = {0};
    SetIDSelector selector(allowed_ids.size(), allowed_ids.data());
    DerivedSearchParameters params;
    params.set_id_selector(&selector);

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(3, -1);
    std::vector<float> distances(3, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 3, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 0);
    EXPECT_EQ(labels[1], -1);
    EXPECT_EQ(labels[2], -1);
}

TEST(InvertedIndexSearch, search_with_id_selector_matching_nothing) {
    InvertedIndex index(3);
    Index* idx = &index;

    add_docs(index, {{{0, 0.3F}}, {{0, 1.0F}}});
    index.build();

    // No id in the selector exists in the index.
    std::vector<idx_t> allowed_ids = {7, 8};
    SetIDSelector selector(allowed_ids.size(), allowed_ids.data());
    SearchParameters params;
    params.set_id_selector(&selector);

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(2, -1);
    std::vector<float> distances(2, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 2, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], -1);
    EXPECT_EQ(labels[1], -1);
}

// Non-enumerable selectors work too: filtering goes through is_member.
TEST(InvertedIndexSearch, search_with_non_enumerable_id_selector) {
    InvertedIndex index(3);
    Index* idx = &index;

    add_docs(index, {{{0, 0.3F}}, {{0, 1.0F}}, {{0, 0.5F}}});
    index.build();

    std::vector<idx_t> denied_ids = {1};
    SetIDSelector denied(denied_ids.size(), denied_ids.data());
    NotIDSelector selector(&denied);
    SearchParameters params;
    params.set_id_selector(&selector);

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(3, -1);
    std::vector<float> distances(3, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 3, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 2);
    EXPECT_EQ(labels[1], 0);
    EXPECT_EQ(labels[2], -1);
}

// Multi-term path: the excluded doc would otherwise set the heap threshold.
TEST(InvertedIndexSearch, search_with_id_selector_multi_term) {
    InvertedIndex index(4);
    Index* idx = &index;

    // doc0: 1.4, doc1: 0.8, doc2: 0.72 for the query below.
    add_docs(index, {{{0, 1.0F}, {1, 0.5F}},
                     {{0, 0.8F}, {2, 0.6F}},
                     {{1, 0.9F}, {3, 0.7F}}});
    index.build();

    std::vector<idx_t> allowed_ids = {1, 2};
    SetIDSelector selector(allowed_ids.size(), allowed_ids.data());
    SearchParameters params;
    params.set_id_selector(&selector);

    std::vector<idx_t> query_indptr = {0, 2};
    std::vector<term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 0.8F};
    std::vector<idx_t> labels(3, -1);
    std::vector<float> distances(3, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 3, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 1);
    EXPECT_EQ(labels[1], 2);
    EXPECT_EQ(labels[2], -1);
    EXPECT_FLOAT_EQ(distances[0], 0.8F);
    EXPECT_FLOAT_EQ(distances[1], 0.72F);
}

// Allowed ids in several scoring windows, exercising window advancement.
TEST(InvertedIndexSearch, search_with_id_selector_multi_window) {
    constexpr int kDim = 2;
    constexpr int kNumDocs = 6000;  // > kScoreWindowSize (4096)
    InvertedIndex index(kDim);

    std::vector<std::map<int, float>> docs;
    docs.reserve(kNumDocs);
    for (int i = 0; i < kNumDocs; ++i) {
        std::map<int, float> doc;
        doc[0] = 0.1F + static_cast<float>(i % 10) * 0.01F;
        if (i % 3 == 0) {
            doc[1] = 0.5F + static_cast<float>(i % 7) * 0.05F;
        }
        docs.push_back(doc);
    }
    add_docs(index, docs);
    index.build();

    std::vector<idx_t> allowed_ids = {5, 4095, 4096, 4099, 5999};
    SetIDSelector selector(allowed_ids.size(), allowed_ids.data());
    SearchParameters params;
    params.set_id_selector(&selector);

    std::vector<idx_t> query_indptr = {0, 2};
    std::vector<term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 1.0F};

    constexpr int kTopK = 10;
    std::vector<idx_t> labels(kTopK, -1);
    std::vector<float> distances(kTopK, -1.0F);

    Index* idx = &index;
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), kTopK, distances.data(), labels.data(),
                &params);

    int found = 0;
    for (int i = 0; i < kTopK; ++i) {
        if (labels[i] == -1) {
            continue;
        }
        ++found;
        EXPECT_TRUE(selector.is_member(labels[i])) << "doc " << labels[i];
    }
    EXPECT_EQ(found, static_cast<int>(allowed_ids.size()));
}

// A term past the index's dimension has no posting list. It used to be indexed
// into the list vector regardless, reading out of bounds.
TEST(InvertedIndexSearch, search_ignores_query_terms_beyond_the_dimension) {
    InvertedIndex index(3);
    Index* idx = &index;

    add_docs(index, {{{0, 1.0F}}, {{0, 0.5F}}});
    index.build();

    // Term 0 exists; term 9 is past the dimension entirely.
    std::vector<idx_t> query_indptr = {0, 2};
    std::vector<term_t> query_indices = {0, 9};
    std::vector<float> query_values = {1.0F, 1.0F};
    std::vector<idx_t> labels(2, -1);
    std::vector<float> distances(2, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 2, distances.data(), labels.data());

    EXPECT_EQ(labels[0], 0);
    EXPECT_EQ(labels[1], 1);
    EXPECT_FLOAT_EQ(distances[0], 1.0F);
    EXPECT_FLOAT_EQ(distances[1], 0.5F);
}

// ============== mapped (mmap) read tests ==============

namespace {

// Index file removed on destruction. read_index/write_index take char*, not
// const char*, hence the owned string.
class TempIndexFile {
public:
    explicit TempIndexFile(const std::string& name)
        : path_(std::filesystem::temp_directory_path() / name) {
        std::filesystem::remove(path_);
    }

    // A destructor cannot forward an exception, and Windows refuses to delete a
    // file while a mapping over it is still open.
    ~TempIndexFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    TempIndexFile(const TempIndexFile&) = delete;
    TempIndexFile& operator=(const TempIndexFile&) = delete;

    char* c_str() { return path_str_.data(); }

private:
    std::filesystem::path path_;
    std::string path_str_ = path_.string();
};

std::unique_ptr<InvertedIndex> built_index() {
    auto index = std::make_unique<InvertedIndex>(4);
    // doc0: term0=1.0, term1=0.5
    // doc1: term0=0.8, term2=0.6
    // doc2: term1=0.9, term3=0.7
    add_docs(*index, {{{0, 1.0F}, {1, 0.5F}},
                      {{0, 0.8F}, {2, 0.6F}},
                      {{1, 0.9F}, {3, 0.7F}}});
    index->build();
    return index;
}

// (scores, labels) for one query, so two residencies can be compared outright.
pair_of_score_id_vector_t search_one(Index* index,
                                     const std::vector<term_t>& terms,
                                     const std::vector<float>& weights, int k) {
    std::vector<idx_t> indptr = {0, static_cast<idx_t>(terms.size())};
    std::vector<float> distances(k, -1.0F);
    std::vector<idx_t> labels(k, detail::INVALID_IDX);
    index->search(1, indptr.data(), terms.data(), weights.data(), k,
                  distances.data(), labels.data());
    return {distances, labels};
}

// The stored element width, counting past the fourcc and dimension read_index
// consumes, and past the document count and term count write_index puts ahead
// of it.
constexpr size_t kElementSizeOffset =
    sizeof(uint32_t) + sizeof(int) + (2 * sizeof(size_t));

template <class T>
T read_field(const std::string& path, size_t offset) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(offset));
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

// The load-time guard rejects what the writer cannot produce, so the only way
// to reach it is to overwrite a field of an otherwise valid file in place.
template <class T>
void patch_field(const std::string& path, size_t offset, T value) {
    std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out);
    out.seekp(static_cast<std::streamoff>(offset));
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

}  // namespace

TEST(InvertedIndexMmapIO, mapped_read_matches_the_copying_read) {
    TempIndexFile file("nsparse_invt_mapped.idx");

    auto source = built_index();
    write_index(source.get(), file.c_str());

    // One file, read both ways: kUseMmap is all that separates the residencies.
    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    std::unique_ptr<Index> copied(read_index(file.c_str()));

    ASSERT_NE(mapped, nullptr);
    ASSERT_NE(copied, nullptr);
    EXPECT_EQ(mapped->num_vectors(), source->num_vectors());
    EXPECT_EQ(copied->num_vectors(), source->num_vectors());

    // Single-term queries, then a multi-term one that exercises the pruning
    // bound off the borrowed per-term maxima.
    for (term_t term = 0; term < 4; ++term) {
        EXPECT_EQ(search_one(mapped.get(), {term}, {1.0F}, 3),
                  search_one(copied.get(), {term}, {1.0F}, 3))
            << "term " << term;
    }
    EXPECT_EQ(search_one(mapped.get(), {0, 1}, {1.0F, 0.8F}, 3),
              search_one(copied.get(), {0, 1}, {1.0F, 0.8F}, 3));
    EXPECT_EQ(search_one(mapped.get(), {0, 1}, {1.0F, 0.8F}, 3),
              search_one(source.get(), {0, 1}, {1.0F, 0.8F}, 3));
}

// Posting lists span several scoring windows here, so the mapped arrays are
// walked well past their first page.
TEST(InvertedIndexMmapIO, mapped_read_matches_across_scoring_windows) {
    constexpr int kNumDocs = 6000;  // > kScoreWindowSize (4096)
    TempIndexFile file("nsparse_invt_mapped_windows.idx");

    InvertedIndex source(2);
    std::vector<std::map<int, float>> docs;
    docs.reserve(kNumDocs);
    for (int i = 0; i < kNumDocs; ++i) {
        std::map<int, float> doc;
        doc[0] = 0.1F + static_cast<float>(i % 10) * 0.01F;
        if (i % 3 == 0) {
            doc[1] = 0.5F + static_cast<float>(i % 7) * 0.05F;
        }
        docs.push_back(doc);
    }
    add_docs(source, docs);
    source.build();
    write_index(&source, file.c_str());

    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));

    ASSERT_NE(mapped, nullptr);
    EXPECT_EQ(mapped->num_vectors(), static_cast<size_t>(kNumDocs));
    EXPECT_EQ(search_one(mapped.get(), {0, 1}, {1.0F, 1.0F}, 10),
              search_one(&source, {0, 1}, {1.0F, 1.0F}, 10));
}

// A stream has no file to map, so the flag alone must not send read_index down
// the mapped path.
TEST(InvertedIndexMmapIO, buffered_read_copies_even_when_the_flag_is_set) {
    auto source = built_index();

    BufferedIOWriter writer;
    write_index(source.get(), &writer);

    BufferedIOReader reader(writer.data());
    std::unique_ptr<Index> loaded(read_index(&reader, IndexIoFlag::kUseMmap));

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->num_vectors(), source->num_vectors());
    EXPECT_EQ(search_one(loaded.get(), {0}, {1.0F}, 3),
              search_one(source.get(), {0}, {1.0F}, 3));
}

TEST(InvertedIndexMmapIO, mapped_read_of_an_index_written_before_build) {
    TempIndexFile file("nsparse_invt_empty_mapped.idx");
    InvertedIndex source(5);
    write_index(&source, file.c_str());

    std::unique_ptr<Index> loaded(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_dimension(), 5);
    EXPECT_EQ(loaded->num_vectors(), 0);
    // No posting lists, so every query comes back empty rather than crashing.
    const auto [distances, labels] = search_one(loaded.get(), {0}, {1.0F}, 2);
    EXPECT_EQ(labels[0], detail::INVALID_IDX);
}

// The values are read back as float, at the width add() encoded them with, so a
// file declaring another width would be strided halfway into each value. Only a
// corrupt file can say so, and both readers take the field from the file
// themselves.
TEST(InvertedIndexMmapIO, both_reads_reject_a_foreign_stored_element_width) {
    TempIndexFile file("nsparse_invt_element_width.idx");
    auto source = built_index();
    write_index(source.get(), file.c_str());
    const std::string path = file.c_str();

    // Fails loudly if the layout moves, rather than patching some other field.
    ASSERT_EQ(read_field<size_t>(path, kElementSizeOffset), size_t{U32});
    patch_field(path, kElementSizeOffset, size_t{U16});

    for (const int flags : {0, static_cast<int>(IndexIoFlag::kUseMmap)}) {
        try {
            std::unique_ptr<Index> loaded(read_index(file.c_str(), flags));
            ADD_FAILURE() << "accepted the file, flags " << flags;
        } catch (const std::runtime_error& error) {
            // The message is checked, not just the type: reading on past a
            // guard that was dropped throws too, somewhere downstream, and that
            // must not read as a pass.
            EXPECT_NE(
                std::string(error.what()).find("stored posting element width"),
                std::string::npos)
                << "flags " << flags << ": " << error.what();
        }
    }
}

// The other mmap entry point: read_csr borrows a native-layout CSR instead of
// copying it in through add(), and build() then owns the postings it derives.
TEST(InvertedIndexMmapIO, build_from_a_mapped_csr_matches_an_added_index) {
    TempIndexFile interchange("nsparse_invt_source.csr");
    TempIndexFile native("nsparse_invt_source.csr.mcsr");

    // The same three documents built_index() adds.
    const std::vector<int64_t> header = {3, 4, 6};
    const std::vector<int64_t> indptr = {0, 2, 4, 6};
    const std::vector<int32_t> indices = {0, 1, 0, 2, 1, 3};
    const std::vector<float> values = {1.0F, 0.5F, 0.8F, 0.6F, 0.9F, 0.7F};
    {
        std::ofstream out(interchange.c_str(), std::ios::binary);
        const auto write_all = [&out](const void* data, size_t bytes) {
            out.write(static_cast<const char*>(data),
                      static_cast<std::streamsize>(bytes));
        };
        write_all(header.data(), header.size() * sizeof(int64_t));
        write_all(indptr.data(), indptr.size() * sizeof(int64_t));
        write_all(indices.data(), indices.size() * sizeof(int32_t));
        write_all(values.data(), values.size() * sizeof(float));
    }
    csr_layout::convert(interchange.c_str(), native.c_str());

    InvertedIndex mapped_source(4);
    mapped_source.read_csr(native.c_str(), Residency::kMmap);
    mapped_source.build();

    // read_csr's mapped residency never goes through add(), so the count has to
    // come off the vectors at build time.
    EXPECT_EQ(mapped_source.num_vectors(), 3);

    auto added = built_index();
    for (term_t term = 0; term < 4; ++term) {
        EXPECT_EQ(search_one(&mapped_source, {term}, {1.0F}, 3),
                  search_one(added.get(), {term}, {1.0F}, 3))
            << "term " << term;
    }
    EXPECT_EQ(search_one(&mapped_source, {0, 1}, {1.0F, 0.8F}, 3),
              search_one(added.get(), {0, 1}, {1.0F, 0.8F}, 3));
}

// The mapped path parses before it commits, so a file cut short leaves nothing
// behind rather than an index pointing past the mapping.
TEST(InvertedIndexMmapIO, mapped_read_rejects_a_truncated_file) {
    TempIndexFile file("nsparse_invt_truncated.idx");
    auto source = built_index();
    write_index(source.get(), file.c_str());

    const std::string path = file.c_str();
    const auto full_size = std::filesystem::file_size(path);
    std::filesystem::resize_file(path, full_size - sizeof(float));

    ASSERT_THROW(
        std::unique_ptr<Index>(read_index(file.c_str(), IndexIoFlag::kUseMmap)),
        std::runtime_error);
}

}  // namespace
}  // namespace nsparse
