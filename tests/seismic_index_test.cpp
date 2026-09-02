/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/seismic_index.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/id_selector.h"
#include "nsparse/index.h"
#include "nsparse/io/buffered_io.h"
#include "nsparse/io/index_io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/csr_layout.h"
#include "tests/csr_interchange_test_util.h"

namespace nsparse {
namespace {

// Testable subclass that exposes protected members
class TestableSeismicIndex : public SeismicIndex {
public:
    using SeismicIndex::add;
    using SeismicIndex::SeismicIndex;

    TestableSeismicIndex(int lambda, int beta, float alpha, int dim)
        : SeismicIndex(dim, {.lambda = lambda, .beta = beta, .alpha = alpha}) {}

    std::vector<InvertedListClusters>& get_clustered_inverted_lists() {
        return clustered_inverted_lists;
    }

    // Helper to add docs from map format: {{term: value, ...}, ...}
    void add_docs(const std::vector<std::map<int, float>>& docs) {
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

        SeismicIndex::add(static_cast<idx_t>(docs.size()), indptr.data(),
                          indices.data(), values.data());
    }

    void set_clustered_inverted_lists(
        std::vector<InvertedListClusters> inv_lists) {
        clustered_inverted_lists = std::move(inv_lists);
    }
};

// ============== build() tests ==============

TEST(SeismicIndexBuild, build_creates_clustered_inverted_lists) {
    TestableSeismicIndex index(2, 2, 0.5F, 3);

    // doc0 has terms {0,1}, doc1 has terms {0,1,2}, doc2 has {1,2}
    // With lambda=2 and beta=2, each term's posting list is clustered
    index.add_docs({{{0, 1.0F}, {1, 0.5F}},
                    {{0, 0.8F}, {1, 0.6F}, {2, 0.7F}},
                    {{1, 0.9F}, {2, 0.4F}}});
    index.build();

    auto& inv_lists = index.get_clustered_inverted_lists();
    // Size equals dimension (3)
    EXPECT_EQ(inv_lists.size(), 3);
    // Term 0: 2 docs -> with beta=2, expect 2 clusters
    EXPECT_EQ(inv_lists.at(0).cluster_size(), 2);
    // Term 1: 3 docs -> with beta=2, expect 2 clusters
    EXPECT_EQ(inv_lists.at(1).cluster_size(), 2);
    // Term 2: 2 docs -> with beta=2, expect 2 clusters
    EXPECT_EQ(inv_lists.at(2).cluster_size(), 2);
}

TEST(SeismicIndexBuild, build_with_single_vector) {
    TestableSeismicIndex index(1, 1, 0.5F, 3);

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}});
    index.build();

    auto& inv_lists = index.get_clustered_inverted_lists();
    EXPECT_EQ(inv_lists.size(), 3);
    // Term 0 and 1 each have 1 doc, term 2 has none
    EXPECT_EQ(inv_lists[0].cluster_size(), 1);
    EXPECT_EQ(inv_lists[1].cluster_size(), 1);
    EXPECT_EQ(inv_lists[2].cluster_size(), 0);
}

TEST(SeismicIndexBuild, build_populates_inverted_lists_for_each_term) {
    TestableSeismicIndex index(10, 2, 0.5F, 4);

    // 4 docs, each with one unique term
    index.add_docs({{{0, 1.0F}}, {{1, 1.0F}}, {{2, 1.0F}}, {{3, 1.0F}}});
    index.build();

    auto& inv_lists = index.get_clustered_inverted_lists();
    EXPECT_EQ(inv_lists.size(), 4);
    // Each term has only 1 doc, so 1 cluster each
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(inv_lists[i].cluster_size(), 1);
    }
}

TEST(SeismicIndexBuild, build_with_multiple_docs_same_term) {
    TestableSeismicIndex index(10, 2, 0.5F, 3);

    // 3 docs all with term 0
    index.add_docs({{{0, 1.0F}}, {{0, 0.9F}}, {{0, 0.8F}}});
    index.build();

    auto& inv_lists = index.get_clustered_inverted_lists();
    EXPECT_EQ(inv_lists.size(), 3);
    // Term 0 has 3 docs, beta=2 so expect 2 clusters
    EXPECT_EQ(inv_lists[0].cluster_size(), 2);
    // Terms 1 and 2 have no docs
    EXPECT_EQ(inv_lists[1].cluster_size(), 0);
    EXPECT_EQ(inv_lists[2].cluster_size(), 0);
}

TEST(SeismicIndexBuild, build_creates_summaries_for_clusters) {
    TestableSeismicIndex index(10, 2, 0.5F, 3);

    // 2 docs both with term 0
    index.add_docs({{{0, 1.0F}, {1, 0.5F}}, {{0, 0.8F}, {2, 0.6F}}});
    index.build();

    auto& inv_lists = index.get_clustered_inverted_lists();
    // Term 0 has 2 docs -> 2 clusters, each with a summary.
    EXPECT_EQ(inv_lists[0].cluster_size(), 2);
}

// ============== Constructor tests ==============

TEST(SeismicIndexConstructor, constructor_with_dim_only) {
    SeismicIndex index(100);
    EXPECT_EQ(index.get_dimension(), 100);
}

TEST(SeismicIndexConstructor, constructor_with_all_params) {
    SeismicIndex index(50, {.lambda = 5, .beta = 3, .alpha = 0.6F});
    EXPECT_EQ(index.get_dimension(), 50);
}

// ============== add() tests ==============

TEST(SeismicIndexAdd, add_creates_vectors) {
    TestableSeismicIndex index(5);

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}});

    EXPECT_NE(index.get_vectors(), nullptr);
    EXPECT_EQ(index.get_vectors()->num_vectors(), 1);
}

TEST(SeismicIndexAdd, add_multiple_vectors) {
    TestableSeismicIndex index(5);

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}, {{2, 0.8F}, {3, 0.6F}}});

    EXPECT_EQ(index.get_vectors()->num_vectors(), 2);
}

TEST(SeismicIndexAdd, add_preserves_vector_data) {
    TestableSeismicIndex index(5);

    index.add_docs({{{0, 1.0F}, {2, 2.0F}, {4, 3.0F}}});

    const auto* vecs = index.get_vectors();
    EXPECT_EQ(vecs->num_vectors(), 1);
    auto dense = vecs->get_dense_vector_float(0);
    EXPECT_EQ(dense.size(), 5);
    EXPECT_FLOAT_EQ(dense[0], 1.0F);
    EXPECT_FLOAT_EQ(dense[2], 2.0F);
    EXPECT_FLOAT_EQ(dense[4], 3.0F);
}

// ============== get_vectors() tests ==============

TEST(SeismicIndexGetVectors, returns_null_before_add) {
    TestableSeismicIndex index(5);
    EXPECT_EQ(index.get_vectors(), nullptr);
}

TEST(SeismicIndexGetVectors, returns_vectors_after_add) {
    TestableSeismicIndex index(5);

    index.add_docs({{{0, 1.0F}}});

    EXPECT_NE(index.get_vectors(), nullptr);
}

// ============== SeismicSearchParameters tests ==============

TEST(SeismicSearchParameters, default_values) {
    SeismicSearchParameters params;
    EXPECT_EQ(params.cut, 10);
    EXPECT_FLOAT_EQ(params.heap_factor, 1.0F);
}

TEST(SeismicSearchParameters, custom_values) {
    SeismicSearchParameters params(20, 0.5F);
    EXPECT_EQ(params.cut, 20);
    EXPECT_FLOAT_EQ(params.heap_factor, 0.5F);
}

// ============== search() tests ==============

TEST(SeismicIndexSearch, search_returns_empty_when_no_vectors) {
    SeismicIndex index(5);
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

TEST(SeismicIndexSearch, search_finds_matching_doc) {
    TestableSeismicIndex index(10, 2, 0.5F, 3);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(1, -1);
    std::vector<float> distances(1, -1.0F);

    SeismicSearchParameters params(5, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 0);
}

TEST(SeismicIndexSearch, search_multiple_queries) {
    TestableSeismicIndex index(10, 2, 0.5F, 4);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}, {{2, 0.8F}, {3, 0.6F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1, 2};
    std::vector<term_t> query_indices = {0, 2};
    std::vector<float> query_values = {1.0F, 1.0F};
    std::vector<idx_t> labels(2, -1);
    std::vector<float> distances(2, -1.0F);

    SeismicSearchParameters params(5, 1.0F);
    idx->search(2, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 0);
    EXPECT_EQ(labels[1], 1);
}

TEST(SeismicIndexSearch, search_respects_k_limit) {
    TestableSeismicIndex index(10, 2, 0.5F, 3);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}}, {{0, 0.9F}}, {{0, 0.8F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(2, -1);
    std::vector<float> distances(2, -1.0F);

    SeismicSearchParameters params(5, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 2, distances.data(), labels.data(),
                &params);

    // Top 2 by score: doc0 (1.0), doc1 (0.9)
    EXPECT_EQ(labels[0], 0);
    EXPECT_EQ(labels[1], 1);
}

TEST(SeismicIndexSearch, search_with_no_matching_term) {
    TestableSeismicIndex index(10, 2, 0.5F, 5);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}});
    index.build();

    // Query with term {3} which doesn't exist in any doc
    // Term 3's inverted list is empty, so no docs should be visited
    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {3};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(1, -1);
    std::vector<float> distances(1, -1.0F);

    SeismicSearchParameters params(5, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data(),
                &params);

    // No docs have term 3, so inverted list for term 3 is empty
    // TopKHolder::top_k_descending_with_padding() pads with -1
    // when fewer than k results are found
    EXPECT_EQ(labels[0], -1);
}

TEST(SeismicIndexSearch, search_returns_results_sorted_by_score) {
    TestableSeismicIndex index(10, 3, 0.5F, 3);
    Index* idx = &index;

    // doc0: term0=0.3, doc1: term0=1.0, doc2: term0=0.5
    index.add_docs({{{0, 0.3F}}, {{0, 1.0F}}, {{0, 0.5F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(3, -1);
    std::vector<float> distances(3, -1.0F);

    SeismicSearchParameters params(5, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 3, distances.data(), labels.data(),
                &params);

    // Sorted by score descending: doc1 (1.0), doc2 (0.5), doc0 (0.3)
    EXPECT_EQ(labels[0], 1);
    EXPECT_EQ(labels[1], 2);
    EXPECT_EQ(labels[2], 0);
}

TEST(SeismicIndexSearch, search_with_default_parameters) {
    TestableSeismicIndex index(10, 2, 0.5F, 3);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 2};
    std::vector<term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 0.5F};
    std::vector<idx_t> labels(1, -1);
    std::vector<float> distances(1, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data(),
                nullptr);

    EXPECT_EQ(labels[0], 0);
}

// ============== Complex search tests ==============

TEST(SeismicIndexSearch, lambda_prunes_posting_list) {
    // lambda=2 means only top 2 docs by value are kept per term's posting list
    TestableSeismicIndex index(2, 1, 0.5F, 3);
    Index* idx = &index;

    // 4 docs all with term 0, values: 0.1, 0.2, 0.3, 0.4
    // After lambda=2 pruning, only doc3 (0.4) and doc2 (0.3) remain in term 0's
    // list
    index.add_docs({{{0, 0.1F}}, {{0, 0.2F}}, {{0, 0.3F}}, {{0, 0.4F}}});
    index.build();

    // Verify posting list was pruned
    auto& inv_lists = index.get_clustered_inverted_lists();
    // Term 0's cluster should only have 2 docs (lambda=2)
    size_t total_docs = 0;
    for (size_t c = 0; c < inv_lists[0].cluster_size(); ++c) {
        total_docs += inv_lists[0].get_docs(c).size();
    }
    EXPECT_EQ(total_docs, 2);

    // Search should only find doc0 and doc1, not doc2 or doc3
    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(4, -1);
    std::vector<float> distances(4, -1.0F);

    SeismicSearchParameters params(5, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 4, distances.data(), labels.data(),
                &params);

    // Only 2 docs in posting list, so only 2 results, rest padded with -1
    EXPECT_EQ(labels[0], 3);   // doc3 (score 0.4)
    EXPECT_EQ(labels[1], 2);   // doc2 (score 0.3)
    EXPECT_EQ(labels[2], -1);  // padded
    EXPECT_EQ(labels[3], -1);  // padded
}

TEST(SeismicIndexSearch, cut_prunes_query_tokens) {
    // cut=1 means only top 1 query token by weight is used
    TestableSeismicIndex index(10, 2, 0.5F, 4);
    Index* idx = &index;

    // doc0 has term 0, doc1 has term 1, doc2 has term 2, doc3 has term 3
    index.add_docs({{{0, 1.0F}}, {{1, 1.0F}}, {{2, 1.0F}}, {{3, 1.0F}}});
    index.build();

    // Query has 3 terms: term0 (0.1), term1 (0.5), term2 (0.9)
    // With cut=1, only term2 (highest weight 0.9) is used
    std::vector<idx_t> query_indptr = {0, 3};
    std::vector<term_t> query_indices = {0, 1, 2};
    std::vector<float> query_values = {0.1F, 0.5F, 0.9F};
    std::vector<idx_t> labels(1, -1);
    std::vector<float> distances(1, -1.0F);

    SeismicSearchParameters params(1, 1.0F);  // cut=1
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data(),
                &params);

    // Only term2 is used, so only doc2 is found
    EXPECT_EQ(labels[0], 2);
}

TEST(SeismicIndexSearch, large_heap_factor_includes_all_clusters) {
    // With very large heap_factor, cluster pruning condition is never met
    // so all clusters are processed across multiple terms
    TestableSeismicIndex index(10, 2, 0.5F, 5);
    Index* idx = &index;

    // 6 docs with multiple terms
    // doc0: term0=1.0, term1=0.5
    // doc1: term0=0.9
    // doc2: term1=0.8, term2=0.3
    // doc3: term2=0.7
    // doc4: term3=0.6
    // doc5: term0=0.1, term3=0.4
    index.add_docs({{{0, 1.0F}, {1, 0.5F}},
                    {{0, 0.9F}},
                    {{1, 0.8F}, {2, 0.3F}},
                    {{2, 0.7F}},
                    {{3, 0.6F}},
                    {{0, 0.1F}, {3, 0.4F}}});
    index.build();

    // Query with terms 0, 1, 2 (weights 1.0, 0.8, 0.5)
    std::vector<idx_t> query_indptr = {0, 3};
    std::vector<term_t> query_indices = {0, 1, 2};
    std::vector<float> query_values = {1.0F, 0.8F, 0.5F};
    std::vector<idx_t> labels(6, -1);
    std::vector<float> distances(6, -1.0F);

    // heap_factor=1000.0 ensures all clusters across all query terms are
    // processed
    SeismicSearchParameters params(5, 1000.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 6, distances.data(), labels.data(),
                &params);

    // Scores (dot product of doc vector with query vector):
    // doc0: 1.0*1.0 + 0.5*0.8 = 1.4
    // doc1: 0.9*1.0 = 0.9
    // doc2: 0.8*0.8 + 0.3*0.5 = 0.79
    // doc3: 0.7*0.5 = 0.35
    // doc4: 0 (no matching terms)
    // doc5: 0.1*1.0 = 0.1
    // Sorted: doc0(1.4), doc1(0.9), doc2(0.79), doc3(0.35), doc5(0.1), doc4(0)
    EXPECT_EQ(labels[0], 0);
    EXPECT_EQ(labels[1], 1);
    EXPECT_EQ(labels[2], 2);
    EXPECT_EQ(labels[3], 3);
    EXPECT_EQ(labels[4], 5);
}

TEST(SeismicIndexSearch, small_heap_factor_prunes_clusters) {
    // With very small heap_factor, cluster pruning happens early
    // Fewer docs are visited compared to large heap_factor
    TestableSeismicIndex index(10, 2, 0.5F, 5);
    Index* idx = &index;

    // 6 docs with multiple terms
    // doc0: term0=1.0, term1=0.5
    // doc1: term0=0.9
    // doc2: term1=0.8, term2=0.3
    // doc3: term2=0.7
    // doc4: term3=0.6
    // doc5: term0=0.1, term3=0.4
    index.add_docs({{{0, 1.0F}, {1, 0.5F}},
                    {{0, 0.9F}},
                    {{1, 0.8F}, {2, 0.3F}},
                    {{2, 0.7F}},
                    {{3, 0.6F}},
                    {{0, 0.1F}, {3, 0.4F}}});
    index.build();

    // Query with terms 0, 1, 2 (weights 1.0, 0.8, 0.5)
    std::vector<idx_t> query_indptr = {0, 3};
    std::vector<term_t> query_indices = {0, 1, 2};
    std::vector<float> query_values = {1.0F, 0.8F, 0.5F};

    // With small heap_factor=0.001, cluster pruning kicks in early
    // After processing first cluster of first term, heap fills up
    // Subsequent clusters may be skipped due to low summary_score * heap_factor
    std::vector<idx_t> labels_small(3, -1);
    std::vector<float> distances_small(3, -1.0F);
    SeismicSearchParameters params_small(5, 0.001F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 3, distances_small.data(),
                labels_small.data(), &params_small);

    // With large heap_factor=1000, all clusters across all terms are processed
    std::vector<idx_t> labels_large(6, -1);
    std::vector<float> distances_large(6, -1.0F);
    SeismicSearchParameters params_large(5, 1000.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 6, distances_large.data(),
                labels_large.data(), &params_large);

    // Large heap_factor should find all docs with matching terms, sorted by
    // score Scores: doc0=1.4, doc1=0.9, doc2=0.79, doc3=0.35, doc5=0.1, doc4=0
    EXPECT_EQ(labels_large[0], 0);
    EXPECT_EQ(labels_large[1], 1);
    EXPECT_EQ(labels_large[2], 2);
    EXPECT_EQ(labels_large[3], 3);
    EXPECT_EQ(labels_large[4], 5);

    // Small heap_factor returns valid results but may miss some docs
    // due to early cluster pruning
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(labels_small[i], 0);
        EXPECT_LE(labels_small[i], 5);
    }
}

// ============== Integration-style tests ==============

TEST(SeismicIndexSearch, heap_factor_controls_result_count_large_dataset) {
    constexpr int kDocCount = 100;
    constexpr int kDimension = 5000;
    constexpr int kLambda = 100;
    constexpr int kBeta = 10;
    constexpr float kAlpha = 0.5F;

    TestableSeismicIndex index(kLambda, kBeta, kAlpha, kDimension);

    // Generate 100 docs where all docs share some common terms
    // This creates dense posting lists that benefit from cluster pruning
    std::vector<std::map<int, float>> docs;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> value_dist(0.1F, 1.0F);

    for (int doc_id = 0; doc_id < kDocCount; ++doc_id) {
        std::map<int, float> doc;
        // All docs have query terms with varying weights
        doc[1000] = value_dist(rng);
        doc[2000] = value_dist(rng);
        doc[3000] = value_dist(rng);
        doc[4000] = value_dist(rng);
        // Add some unique terms per doc
        doc[doc_id] = value_dist(rng);
        docs.push_back(doc);
    }

    index.add_docs(docs);
    index.build();

    // Query with 4 terms (similar to OpenSearch test)
    std::vector<idx_t> query_indptr = {0, 4};
    std::vector<term_t> query_indices = {1000, 2000, 3000, 4000};
    std::vector<float> query_values = {0.12F, 0.64F, 0.87F, 0.53F};

    // Test with very small heap_factor - aggressive pruning
    // cut=2 means only top 2 query terms by weight are used
    std::vector<idx_t> labels_small(kDocCount, -1);
    std::vector<float> distances_small(kDocCount, -1.0F);
    SeismicSearchParameters params_small(2, 0.000001F);
    Index* idx = &index;
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), kDocCount, distances_small.data(),
                labels_small.data(), &params_small);

    // Count valid results with small heap_factor
    int count_small = 0;
    for (int i = 0; i < kDocCount; ++i) {
        if (labels_small[i] >= 0 && labels_small[i] < kDocCount) {
            ++count_small;
        }
    }

    // Test with very large heap_factor - no pruning
    // cut=4 means all query terms are used
    std::vector<idx_t> labels_large(kDocCount, -1);
    std::vector<float> distances_large(kDocCount, -1.0F);
    SeismicSearchParameters params_large(4, 100000.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), kDocCount, distances_large.data(),
                labels_large.data(), &params_large);

    // Count valid results with large heap_factor
    int count_large = 0;
    for (int i = 0; i < kDocCount; ++i) {
        if (labels_large[i] >= 0 && labels_large[i] < kDocCount) {
            ++count_large;
        }
    }

    // With small heap_factor, we should get fewer or equal results
    EXPECT_LE(count_small, kDocCount);
    EXPECT_GT(count_small, 0);

    // With large heap_factor, we should get all docs (all have query terms)
    EXPECT_EQ(count_large, kDocCount);

    // Small heap_factor with cut=2 should return fewer results than
    // large heap_factor with cut=4 (more query terms = more docs found)
    EXPECT_LE(count_small, count_large);
}

// ============== write_index/read_index tests ==============

TEST(SeismicIndexIO, write_and_read_empty_index) {
    SeismicIndex original(100);

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    Index* loaded = read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_dimension(), 100);
    EXPECT_EQ(loaded->id(), original.id());
    EXPECT_EQ(loaded->get_vectors(), nullptr);

    delete loaded;
}

TEST(SeismicIndexIO, write_and_read_with_vectors) {
    TestableSeismicIndex original(10, 2, 0.5F, 5);

    original.add_docs({{{0, 1.0F}, {1, 0.5F}}, {{2, 0.8F}, {3, 0.6F}}});

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    Index* loaded = read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_dimension(), 5);

    const auto* vecs = loaded->get_vectors();
    ASSERT_NE(vecs, nullptr);
    EXPECT_EQ(vecs->num_vectors(), 2);

    delete loaded;
}

TEST(SeismicIndexIO, write_and_read_preserves_vector_data) {
    TestableSeismicIndex original(10, 2, 0.5F, 5);

    original.add_docs({{{0, 1.0F}, {2, 2.0F}, {4, 3.0F}}});

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    Index* loaded = read_index(&reader);

    const auto* vecs = loaded->get_vectors();
    ASSERT_NE(vecs, nullptr);

    auto dense = vecs->get_dense_vector_float(0);
    EXPECT_EQ(dense.size(), 5);
    EXPECT_FLOAT_EQ(dense[0], 1.0F);
    EXPECT_FLOAT_EQ(dense[2], 2.0F);
    EXPECT_FLOAT_EQ(dense[4], 3.0F);

    delete loaded;
}

TEST(SeismicIndexIO, write_and_read_built_index) {
    TestableSeismicIndex original(10, 2, 0.5F, 4);

    original.add_docs({{{0, 1.0F}, {1, 0.5F}}, {{2, 0.8F}, {3, 0.6F}}});
    original.build();

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    Index* loaded = read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_dimension(), 4);
    EXPECT_EQ(loaded->get_vectors()->num_vectors(), 2);

    delete loaded;
}

TEST(SeismicIndexIO, write_and_read_search_produces_same_results) {
    TestableSeismicIndex original(10, 2, 0.5F, 4);

    // doc0: term0=1.0, term1=0.5
    // doc1: term0=0.8, term2=0.6
    // doc2: term1=0.9, term3=0.7
    original.add_docs({{{0, 1.0F}, {1, 0.5F}},
                       {{0, 0.8F}, {2, 0.6F}},
                       {{1, 0.9F}, {3, 0.7F}}});
    original.build();

    // Search on original
    std::vector<idx_t> query_indptr = {0, 2};
    std::vector<term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 0.8F};
    std::vector<idx_t> labels_original(3, -1);
    std::vector<float> distances_original(3, -1.0F);

    SeismicSearchParameters params(5, 1000.0F);
    Index* idx_original = &original;
    idx_original->search(1, query_indptr.data(), query_indices.data(),
                         query_values.data(), 3, distances_original.data(),
                         labels_original.data(), &params);

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
                   labels_loaded.data(), &params);

    // Results should be identical
    EXPECT_EQ(labels_original[0], labels_loaded[0]);
    EXPECT_EQ(labels_original[1], labels_loaded[1]);
    EXPECT_EQ(labels_original[2], labels_loaded[2]);

    delete loaded;
}

TEST(SeismicIndexIO, write_and_read_multiple_terms) {
    TestableSeismicIndex original(10, 2, 0.5F, 5);

    // 4 docs with multiple terms
    original.add_docs({{{0, 1.0F}, {1, 0.5F}},
                       {{1, 0.8F}, {2, 0.6F}},
                       {{2, 0.7F}, {3, 0.4F}},
                       {{3, 0.9F}, {4, 0.3F}}});
    original.build();

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    Index* loaded = read_index(&reader);

    EXPECT_EQ(loaded->get_vectors()->num_vectors(), 4);

    // Verify search works on loaded index
    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(1, -1);
    std::vector<float> distances(1, -1.0F);

    SeismicSearchParameters params(5, 1.0F);
    loaded->search(1, query_indptr.data(), query_indices.data(),
                   query_values.data(), 1, distances.data(), labels.data(),
                   &params);

    EXPECT_EQ(labels[0], 0);

    delete loaded;
}

// ============== IDSelector tests ==============

TEST(SeismicIndexSearch, search_exact_match_with_small_selector) {
    TestableSeismicIndex index(10, 3, 0.5F, 3);
    Index* idx = &index;

    // doc0: term0=0.3, doc1: term0=1.0, doc2: term0=0.5, doc3: term0=0.8
    index.add_docs({{{0, 0.3F}}, {{0, 1.0F}}, {{0, 0.5F}}, {{0, 0.8F}}});
    index.build();

    // Selector size (2) <= k (2), triggers exact match path
    std::vector<idx_t> allowed_ids = {1, 3};
    ArrayIDSelector selector(allowed_ids.size(), allowed_ids.data());

    SeismicSearchParameters params(5, 1.0F);
    params.set_id_selector(&selector);

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(2, -1);
    std::vector<float> distances(2, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 2, distances.data(), labels.data(),
                &params);

    // Exact match should return doc1 (score 1.0) and doc3 (score 0.8)
    EXPECT_EQ(labels[0], 1);
    EXPECT_EQ(labels[1], 3);
    EXPECT_GT(distances[0], distances[1]);
}

TEST(SeismicIndexSearch, search_with_id_selector_filters_results) {
    TestableSeismicIndex index(10, 3, 0.5F, 3);
    Index* idx = &index;

    // doc0: term0=0.3, doc1: term0=1.0, doc2: term0=0.5
    index.add_docs({{{0, 0.3F}}, {{0, 1.0F}}, {{0, 0.5F}}});
    index.build();

    // Only allow doc0 and doc2 via IDSelector
    std::vector<idx_t> allowed_ids = {0, 2};
    SetIDSelector selector(allowed_ids.size(), allowed_ids.data());

    SeismicSearchParameters params(5, 1.0F);
    params.set_id_selector(&selector);

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(3, -1);
    std::vector<float> distances(3, -1.0F);

    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 3, distances.data(), labels.data(),
                &params);

    // doc1 (highest score 1.0) should be excluded
    // Results: doc2 (0.5), doc0 (0.3), then padding
    EXPECT_EQ(labels[0], 2);
    EXPECT_EQ(labels[1], 0);
    EXPECT_EQ(labels[2], -1);
}

// ============== mapped read_index tests ==============

namespace {

// Index file removed on destruction.
class TempIndexFile {
public:
    explicit TempIndexFile(const std::string& name)
        : path_(std::filesystem::temp_directory_path() / name) {
        std::filesystem::remove(path_);
    }
    ~TempIndexFile() { std::filesystem::remove(path_); }

    TempIndexFile(const TempIndexFile&) = delete;
    TempIndexFile& operator=(const TempIndexFile&) = delete;

    // write_index/read_index take char*, not const char*.
    char* c_str() { return path_str_.data(); }

private:
    std::filesystem::path path_{};
    std::string path_str_ = path_.string();
};

// A built index with a handful of docs. Residency is a property of the reader,
// not of the file, so the written bytes are the same either way.
std::unique_ptr<TestableSeismicIndex> built_index() {
    auto index = std::make_unique<TestableSeismicIndex>(
        5, SeismicClusterParameters{.lambda = 10, .beta = 2, .alpha = 0.5F});
    index->add_docs({{{0, 1.0F}, {2, 2.0F}},
                     {{1, 0.5F}, {3, 1.5F}},
                     {{0, 0.3F}, {4, 2.5F}},
                     {{2, 1.0F}, {3, 0.7F}, {4, 1.2F}}});
    index->build();
    return index;
}

std::vector<idx_t> search_top(Index* index, term_t term, int k) {
    std::vector<idx_t> indptr = {0, 1};
    std::vector<term_t> indices = {term};
    std::vector<float> values = {1.0F};
    std::vector<idx_t> labels(k, detail::INVALID_IDX);
    std::vector<float> distances(k, -1.0F);
    index->search(1, indptr.data(), indices.data(), values.data(), k,
                  distances.data(), labels.data());
    return labels;
}

// SparseVectors' element_size, counting past the header read_index consumes and
// past the vector count and dimension serialize() wrote ahead of it.
constexpr size_t kElementSizeOffset = kIndexHeaderSize + (2 * sizeof(size_t));

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

// Both readers of a corrupted file, each of which has to reject it on its own.
// The message is checked, not just the type: reading on past a guard that was
// dropped throws too, somewhere downstream, and that must not read as a pass.
void expect_both_reads_rejected(char* path, const char* fragment) {
    for (const int flags : {0, static_cast<int>(IndexIoFlag::kUseMmap)}) {
        try {
            std::unique_ptr<Index> loaded(read_index(path, flags));
            ADD_FAILURE() << "accepted the file, flags " << flags;
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find(fragment),
                      std::string::npos)
                << "flags " << flags << ": " << error.what();
        }
    }
}

// Raw-array CSR corpus. A reproducible random corpus with distinct, ascending
// terms per row (CSR convention) and values in (0, 1].
struct RawCsr {
    std::vector<idx_t> indptr;
    std::vector<term_t> indices;
    std::vector<float> values;
    idx_t n = 0;
};

RawCsr make_raw_corpus(idx_t rows, int dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> nnz_dist(3, 12);
    std::uniform_int_distribution<int> term_dist(0, dim - 1);
    std::uniform_real_distribution<float> val_dist(0.05F, 1.0F);
    RawCsr c;
    c.n = rows;
    c.indptr.push_back(0);
    for (idx_t r = 0; r < rows; ++r) {
        std::set<int> terms;
        const int nnz = nnz_dist(rng);
        while (static_cast<int>(terms.size()) < nnz) {
            terms.insert(term_dist(rng));
        }
        for (const int t : terms) {
            c.indices.push_back(static_cast<term_t>(t));
            c.values.push_back(val_dist(rng));
        }
        c.indptr.push_back(static_cast<idx_t>(c.indices.size()));
    }
    return c;
}

// Batch search over the corpus rows as queries; returns per-query {labels,
// scores} for a bit-exact comparison.
std::pair<std::vector<idx_t>, std::vector<float>> search_corpus(
    Index* index, const RawCsr& queries, int k) {
    std::vector<idx_t> labels(static_cast<size_t>(queries.n) * k,
                              detail::INVALID_IDX);
    std::vector<float> distances(static_cast<size_t>(queries.n) * k, -1.0F);
    index->search(queries.n, queries.indptr.data(), queries.indices.data(),
                  queries.values.data(), k, distances.data(), labels.data());
    return {labels, distances};
}

}  // namespace

TEST(SeismicIndexMmapIO, mapped_read_matches_the_copying_read) {
    TempIndexFile file("nsparse_seis_mapped.idx");

    auto source = built_index();
    write_index(source.get(), file.c_str());

    // One file, read both ways: kUseMmap is all that separates the residencies.
    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    std::unique_ptr<Index> copied(read_index(file.c_str()));

    ASSERT_NE(mapped, nullptr);
    ASSERT_NE(copied, nullptr);
    ASSERT_EQ(mapped->get_vectors()->num_vectors(),
              copied->get_vectors()->num_vectors());
    for (term_t term = 0; term < 5; ++term) {
        EXPECT_EQ(search_top(mapped.get(), term, 4),
                  search_top(copied.get(), term, 4))
            << "term " << term;
    }
}

// The point of the mapped path: the arrays point into the file, not into fresh
// allocations.
TEST(SeismicIndexMmapIO, mapped_read_borrows_from_the_file) {
    TempIndexFile file("nsparse_seis_borrow.idx");
    auto source = built_index();
    write_index(source.get(), file.c_str());

    std::unique_ptr<Index> loaded(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));

    const auto* vectors = loaded->get_vectors();
    ASSERT_NE(vectors, nullptr);
    // A borrowed CSR keeps the file's contiguity: indices sit right after
    // indptr, which is only true of the mapping, not of separate allocations.
    const auto* indptr_bytes =
        reinterpret_cast<const uint8_t*>(vectors->indptr_data());
    const auto* indices_bytes =
        reinterpret_cast<const uint8_t*>(vectors->indices_data());
    EXPECT_EQ(indices_bytes - indptr_bytes,
              static_cast<ptrdiff_t>((vectors->num_vectors() + 1) *
                                     sizeof(idx_t)));
}

// Building from a native CSR borrowed via mmap (read_csr(kMmap)) must match
// building the same corpus fed through add(): convert -> read_csr(kMmap) ->
// build -> persist -> mmap-reload -> search is bit-exact to the add()-fed build.
// This pins the mmap-CSR build path for the regular (unquantized) index; its
// num_vectors() derives from get_vectors(), so no count fix is needed here.
TEST(SeismicIndexMmapIO, mmap_csr_build_matches_add_build) {
    constexpr int kDim = 128;
    const RawCsr corpus = make_raw_corpus(800, kDim, /*seed=*/11);
    const RawCsr queries = make_raw_corpus(30, kDim, /*seed=*/12);
    const SeismicClusterParameters params{
        .lambda = 20, .beta = 4, .alpha = 0.5F, .seed = 42};

    // Reference: the same corpus fed through add().
    SeismicIndex added(kDim, params);
    added.add(corpus.n, corpus.indptr.data(), corpus.indices.data(),
              corpus.values.data());
    added.build();
    const auto expected = search_corpus(&added, queries, 10);

    // Under test: build from a native CSR of the same corpus borrowed via mmap,
    // then persist and reload.
    csr_test::TempCsrFiles csr("nsparse_seis_src");
    csr_test::write_interchange_csr(csr.interchange(), corpus, kDim);
    csr_layout::convert(csr.interchange(), csr.native());

    SeismicIndex mapped_build(kDim, params);
    mapped_build.read_csr(csr.native().c_str(), Residency::kMmap);
    ASSERT_EQ(mapped_build.num_vectors(), static_cast<size_t>(corpus.n));
    mapped_build.build();

    TempIndexFile file("nsparse_seis_mmapbuild.idx");
    write_index(&mapped_build, file.c_str());
    std::unique_ptr<Index> reloaded(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(reloaded, nullptr);
    EXPECT_EQ(reloaded->num_vectors(), static_cast<size_t>(corpus.n));

    const auto got = search_corpus(reloaded.get(), queries, 10);
    EXPECT_EQ(got.first, expected.first) << "labels differ";
    ASSERT_EQ(got.second.size(), expected.second.size());
    for (size_t i = 0; i < got.second.size(); ++i) {
        EXPECT_FLOAT_EQ(got.second[i], expected.second[i]) << "score at " << i;
    }
}

// A stream has no file to map, so the flag alone must not send read_index down
// the mapped path.
TEST(SeismicIndexMmapIO, buffered_read_copies_even_when_the_flag_is_set) {
    auto source = built_index();

    BufferedIOWriter writer;
    write_index(source.get(), &writer);

    BufferedIOReader reader(writer.data());
    std::unique_ptr<Index> loaded(read_index(&reader, IndexIoFlag::kUseMmap));

    ASSERT_NE(loaded, nullptr);
    const auto* vectors = loaded->get_vectors();
    ASSERT_NE(vectors, nullptr);
    EXPECT_EQ(vectors->num_vectors(), 4);
    // Copied out, so nothing points into the writer's buffer.
    const auto* base = writer.data().data();
    const auto* end = base + writer.data().size();
    const auto* indptr =
        reinterpret_cast<const uint8_t*>(vectors->indptr_data());
    EXPECT_TRUE(indptr < base || indptr >= end);
}

// The values are read back as float, at the width add() encoded them with, so
// a file declaring another width would be strided past the end of the array or
// halfway into each value. Only a corrupt file can say so, and both readers
// take the field from the file themselves.
TEST(SeismicIndexMmapIO, both_reads_reject_a_foreign_stored_element_width) {
    TempIndexFile file("nsparse_seis_element_width.idx");
    auto source = built_index();
    write_index(source.get(), file.c_str());
    const std::string path = file.c_str();

    // Fails loudly if the layout moves, rather than patching some other field.
    ASSERT_EQ(read_field<size_t>(path, kElementSizeOffset), size_t{U32});
    patch_field(path, kElementSizeOffset, size_t{U16});

    expect_both_reads_rejected(file.c_str(),
                               "element size is not the seismic index's");
}

TEST(SeismicIndexMmapIO, mapped_read_of_an_empty_index) {
    TempIndexFile file("nsparse_seis_empty_mapped.idx");
    SeismicIndex source(5, {.lambda = 10, .beta = 2, .alpha = 0.5F});
    write_index(&source, file.c_str());

    std::unique_ptr<Index> loaded(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_vectors(), nullptr);
}

// ============== seed tests ==============

namespace {

// One corpus, so the seed is the only thing that can differ between builds.
std::vector<std::map<int, float>> seed_test_docs(int n_docs, int dim) {
    std::vector<std::map<int, float>> docs;
    docs.reserve(n_docs);
    for (int i = 0; i < n_docs; ++i) {
        docs.push_back({{i % dim, 1.0F + static_cast<float>(i % 13)},
                        {(i * 7 + 3) % dim, 0.5F + static_cast<float>(i % 5)},
                        {(i * 11 + 5) % dim, 0.25F + static_cast<float>(i % 3)}});
    }
    return docs;
}

// Search output of a freshly built index, which is what a caller can observe.
std::pair<std::vector<float>, std::vector<idx_t>> build_and_search(int seed) {
    constexpr int kDim = 64;
    constexpr int kDocs = 400;
    constexpr int kTopK = 10;

    TestableSeismicIndex index(kDim, {.lambda = 25, .beta = 4, .alpha = 0.4F,
                                      .seed = seed});
    index.add_docs(seed_test_docs(kDocs, kDim));
    index.build();

    std::vector<idx_t> query_indptr = {0, 3};
    std::vector<term_t> query_indices = {1, 17, 33};
    std::vector<float> query_values = {1.0F, 0.7F, 0.4F};
    std::vector<float> distances(kTopK);
    std::vector<idx_t> labels(kTopK);
    // Through the base: SeismicIndex redeclares search() as protected, which
    // hides the public Index overload by name.
    Index& searchable = index;
    searchable.search(1, query_indptr.data(), query_indices.data(),
                      query_values.data(), kTopK, distances.data(),
                      labels.data(), nullptr);
    return {distances, labels};
}

}  // namespace

TEST(SeismicIndexSeed, same_seed_builds_an_identical_index) {
    ASSERT_EQ(build_and_search(42), build_and_search(42));
}

TEST(SeismicIndexSeed, different_seeds_build_different_indexes) {
    ASSERT_NE(build_and_search(42), build_and_search(43));
}

}  // namespace
}  // namespace nsparse
