/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/seismic_scalar_quantized_index.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/id_selector.h"
#include "nsparse/index.h"
#include "nsparse/io/buffered_io.h"
#include "nsparse/io/index_io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/scalar_quantizer.h"
#include "tests/csr_interchange_test_util.h"

namespace nsparse {
namespace {

// Testable subclass that exposes protected members
class TestableSeismicSQIndex : public SeismicScalarQuantizedIndex {
public:
    using SeismicScalarQuantizedIndex::add;
    using SeismicScalarQuantizedIndex::SeismicScalarQuantizedIndex;

    TestableSeismicSQIndex(QuantizerType qt, float vmin, float vmax, int lambda,
                           int beta, float alpha, int dim)
        : SeismicScalarQuantizedIndex(
              qt, vmin, vmax, {.lambda = lambda, .beta = beta, .alpha = alpha},
              dim) {}

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

        SeismicScalarQuantizedIndex::add(static_cast<idx_t>(docs.size()),
                                         indptr.data(), indices.data(),
                                         values.data());
    }
};

// ============== build() tests ==============

TEST(SeismicSQIndexBuild, build_creates_clustered_inverted_lists_8bit) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 2, 2, 0.5F,
                                 3);

    index.add_docs({{{0, 1.0F}, {1, 0.5F}},
                    {{0, 0.8F}, {1, 0.6F}, {2, 0.7F}},
                    {{1, 0.9F}, {2, 0.4F}}});
    index.build();

    auto& inv_lists = index.get_clustered_inverted_lists();
    EXPECT_EQ(inv_lists.size(), 3);
    EXPECT_EQ(inv_lists.at(0).cluster_size(), 2);
    EXPECT_EQ(inv_lists.at(1).cluster_size(), 2);
    EXPECT_EQ(inv_lists.at(2).cluster_size(), 2);
}

TEST(SeismicSQIndexBuild, build_creates_clustered_inverted_lists_16bit) {
    TestableSeismicSQIndex index(QuantizerType::QT_16bit, 0.0F, 1.0F, 2, 2,
                                 0.5F, 3);

    index.add_docs({{{0, 1.0F}, {1, 0.5F}},
                    {{0, 0.8F}, {1, 0.6F}, {2, 0.7F}},
                    {{1, 0.9F}, {2, 0.4F}}});
    index.build();

    auto& inv_lists = index.get_clustered_inverted_lists();
    EXPECT_EQ(inv_lists.size(), 3);
    EXPECT_EQ(inv_lists.at(0).cluster_size(), 2);
    EXPECT_EQ(inv_lists.at(1).cluster_size(), 2);
    EXPECT_EQ(inv_lists.at(2).cluster_size(), 2);
}

TEST(SeismicSQIndexBuild, build_with_single_vector) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 1, 1, 0.5F,
                                 3);

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}});
    index.build();

    auto& inv_lists = index.get_clustered_inverted_lists();
    EXPECT_EQ(inv_lists.size(), 3);
    EXPECT_EQ(inv_lists[0].cluster_size(), 1);
    EXPECT_EQ(inv_lists[1].cluster_size(), 1);
    EXPECT_EQ(inv_lists[2].cluster_size(), 0);
}

TEST(SeismicSQIndexBuild, build_populates_inverted_lists_for_each_term) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 4);

    index.add_docs({{{0, 1.0F}}, {{1, 1.0F}}, {{2, 1.0F}}, {{3, 1.0F}}});
    index.build();

    auto& inv_lists = index.get_clustered_inverted_lists();
    EXPECT_EQ(inv_lists.size(), 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(inv_lists[i].cluster_size(), 1);
    }
}

TEST(SeismicSQIndexBuild, build_with_multiple_docs_same_term) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 3);

    index.add_docs({{{0, 1.0F}}, {{0, 0.9F}}, {{0, 0.8F}}});
    index.build();

    auto& inv_lists = index.get_clustered_inverted_lists();
    EXPECT_EQ(inv_lists.size(), 3);
    EXPECT_EQ(inv_lists[0].cluster_size(), 2);
    EXPECT_EQ(inv_lists[1].cluster_size(), 0);
    EXPECT_EQ(inv_lists[2].cluster_size(), 0);
}

// ============== Constructor tests ==============

TEST(SeismicSQIndexConstructor, constructor_with_dim_only) {
    SeismicScalarQuantizedIndex index(100);
    EXPECT_EQ(index.get_dimension(), 100);
}

TEST(SeismicSQIndexConstructor, constructor_with_all_params_8bit) {
    SeismicScalarQuantizedIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                      {.lambda = 5, .beta = 3, .alpha = 0.6F},
                                      50);
    EXPECT_EQ(index.get_dimension(), 50);
    EXPECT_EQ(index.get_scalar_quantizer().get_quantizer_type(),
              QuantizerType::QT_8bit);
}

TEST(SeismicSQIndexConstructor, constructor_with_all_params_16bit) {
    SeismicScalarQuantizedIndex index(QuantizerType::QT_16bit, 0.0F, 2.0F,
                                      {.lambda = 5, .beta = 3, .alpha = 0.6F},
                                      50);
    EXPECT_EQ(index.get_dimension(), 50);
    EXPECT_EQ(index.get_scalar_quantizer().get_quantizer_type(),
              QuantizerType::QT_16bit);
}

// ============== add() tests ==============

TEST(SeismicSQIndexAdd, add_creates_vectors) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 5, 2, 0.5F,
                                 5);

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}});

    EXPECT_NE(index.get_vectors(), nullptr);
    EXPECT_EQ(index.get_vectors()->num_vectors(), 1);
}

TEST(SeismicSQIndexAdd, add_multiple_vectors) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 5, 2, 0.5F,
                                 5);

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}, {{2, 0.8F}, {3, 0.6F}}});

    EXPECT_EQ(index.get_vectors()->num_vectors(), 2);
}

TEST(SeismicSQIndexAdd, add_quantizes_values_8bit) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 5, 2, 0.5F,
                                 5);

    index.add_docs({{{0, 1.0F}, {2, 0.5F}}});

    const auto* vecs = index.get_vectors();
    EXPECT_EQ(vecs->get_element_size(), U8);
}

TEST(SeismicSQIndexAdd, add_quantizes_values_16bit) {
    TestableSeismicSQIndex index(QuantizerType::QT_16bit, 0.0F, 1.0F, 5, 2,
                                 0.5F, 5);

    index.add_docs({{{0, 1.0F}, {2, 0.5F}}});

    const auto* vecs = index.get_vectors();
    EXPECT_EQ(vecs->get_element_size(), U16);
}

// ============== get_vectors() tests ==============

TEST(SeismicSQIndexGetVectors, returns_null_before_add) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 5, 2, 0.5F,
                                 5);
    EXPECT_EQ(index.get_vectors(), nullptr);
}

TEST(SeismicSQIndexGetVectors, returns_vectors_after_add) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 5, 2, 0.5F,
                                 5);

    index.add_docs({{{0, 1.0F}}});

    EXPECT_NE(index.get_vectors(), nullptr);
}

// ============== SeismicSQSearchParameters tests ==============

TEST(SeismicSQSearchParameters, constructor_sets_values) {
    SeismicSQSearchParameters params(0.0F, 1.0F, 20, 0.5F);
    EXPECT_FLOAT_EQ(params.vmin, 0.0F);
    EXPECT_FLOAT_EQ(params.vmax, 1.0F);
    EXPECT_EQ(params.cut, 20);
    EXPECT_FLOAT_EQ(params.heap_factor, 0.5F);
}

// ============== search() tests ==============

TEST(SeismicSQIndexSearch, search_returns_empty_when_no_vectors) {
    SeismicScalarQuantizedIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                      {.lambda = 5, .beta = 2, .alpha = 0.5F},
                                      5);
    Index* idx = &index;

    std::vector<idx_t> query_indptr = {0, 2};
    std::vector<term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 0.5F};
    std::vector<idx_t> labels(5, -1);
    std::vector<float> distances(5, -1.0F);

    SeismicSearchParameters params(10, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 5, distances.data(), labels.data(),
                &params);

    for (const auto& label : labels) {
        EXPECT_EQ(label, -1);
    }
}

TEST(SeismicSQIndexSearch, search_finds_matching_doc_8bit) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 3);
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

TEST(SeismicSQIndexSearch, search_finds_matching_doc_16bit) {
    TestableSeismicSQIndex index(QuantizerType::QT_16bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 3);
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

TEST(SeismicSQIndexSearch, search_multiple_queries) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 4);
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

TEST(SeismicSQIndexSearch, search_respects_k_limit) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 3);
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

    EXPECT_EQ(labels[0], 0);
    EXPECT_EQ(labels[1], 1);
}

TEST(SeismicSQIndexSearch, search_with_default_parameters) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 3);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 2};
    std::vector<term_t> query_indices = {0, 1};
    std::vector<float> query_values = {1.0F, 0.5F};
    std::vector<idx_t> labels(1, -1);
    std::vector<float> distances(1, -1.0F);

    SeismicSearchParameters params(10, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 0);
}

// ============== Complex search tests ==============

TEST(SeismicSQIndexSearch, lambda_prunes_posting_list) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 2, 1, 0.5F,
                                 3);
    Index* idx = &index;

    index.add_docs({{{0, 0.1F}}, {{0, 0.2F}}, {{0, 0.3F}}, {{0, 0.4F}}});
    index.build();

    auto& inv_lists = index.get_clustered_inverted_lists();
    size_t total_docs = 0;
    for (size_t c = 0; c < inv_lists[0].cluster_size(); ++c) {
        total_docs += inv_lists[0].get_docs(c).size();
    }
    EXPECT_EQ(total_docs, 2);

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(4, -1);
    std::vector<float> distances(4, -1.0F);

    SeismicSearchParameters params(5, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 4, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 3);
    EXPECT_EQ(labels[1], 2);
}

TEST(SeismicSQIndexSearch, cut_prunes_query_tokens) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 4);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}}, {{1, 1.0F}}, {{2, 1.0F}}, {{3, 1.0F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 3};
    std::vector<term_t> query_indices = {0, 1, 2};
    std::vector<float> query_values = {0.1F, 0.5F, 0.9F};
    std::vector<idx_t> labels(1, -1);
    std::vector<float> distances(1, -1.0F);

    SeismicSearchParameters params(1, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 2);
}

TEST(SeismicSQIndexSearch, large_heap_factor_includes_all_clusters) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 5);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}, {1, 0.5F}},
                    {{0, 0.9F}},
                    {{1, 0.8F}, {2, 0.3F}},
                    {{2, 0.7F}},
                    {{3, 0.6F}},
                    {{0, 0.1F}, {3, 0.4F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 3};
    std::vector<term_t> query_indices = {0, 1, 2};
    std::vector<float> query_values = {1.0F, 0.8F, 0.5F};
    std::vector<idx_t> labels(6, -1);
    std::vector<float> distances(6, -1.0F);

    SeismicSearchParameters params(5, 1000.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 6, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 0);
    EXPECT_EQ(labels[1], 1);
    EXPECT_EQ(labels[2], 2);
    EXPECT_EQ(labels[3], 3);
    EXPECT_EQ(labels[4], 5);
}

// ============== Search with SeismicSQSearchParameters tests ==============

TEST(SeismicSQIndexSearch, search_with_sq_search_parameters) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 3);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};
    std::vector<idx_t> labels(1, -1);
    std::vector<float> distances(1, -1.0F);

    SeismicSQSearchParameters params(0.0F, 1.0F, 5, 1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 1, distances.data(), labels.data(),
                &params);

    EXPECT_EQ(labels[0], 0);
}

// ============== write_index/read_index tests ==============

TEST(SeismicSQIndexIO, write_and_read_empty_index) {
    SeismicScalarQuantizedIndex original(
        QuantizerType::QT_8bit, 0.0F, 1.0F,
        {.lambda = 5, .beta = 2, .alpha = 0.5F}, 100);

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

TEST(SeismicSQIndexIO, write_and_read_with_vectors) {
    TestableSeismicSQIndex original(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                    0.5F, 5);

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

TEST(SeismicSQIndexIO, write_and_read_built_index) {
    TestableSeismicSQIndex original(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                    0.5F, 4);

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

TEST(SeismicSQIndexIO, write_and_read_search_produces_same_results) {
    TestableSeismicSQIndex original(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                    0.5F, 4);

    original.add_docs({{{0, 1.0F}, {1, 0.5F}},
                       {{0, 0.8F}, {2, 0.6F}},
                       {{1, 0.9F}, {3, 0.7F}}});
    original.build();

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

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    Index* loaded = read_index(&reader);

    std::vector<idx_t> labels_loaded(3, -1);
    std::vector<float> distances_loaded(3, -1.0F);
    loaded->search(1, query_indptr.data(), query_indices.data(),
                   query_values.data(), 3, distances_loaded.data(),
                   labels_loaded.data(), &params);

    EXPECT_EQ(labels_original[0], labels_loaded[0]);
    EXPECT_EQ(labels_original[1], labels_loaded[1]);
    EXPECT_EQ(labels_original[2], labels_loaded[2]);

    delete loaded;
}

TEST(SeismicSQIndexIO, write_and_read_preserves_quantizer_type_8bit) {
    TestableSeismicSQIndex original(QuantizerType::QT_8bit, 0.1F, 0.9F, 10, 2,
                                    0.5F, 5);

    original.add_docs({{{0, 0.5F}}});

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    auto* loaded =
        dynamic_cast<SeismicScalarQuantizedIndex*>(read_index(&reader));

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_scalar_quantizer().get_quantizer_type(),
              QuantizerType::QT_8bit);
    EXPECT_FLOAT_EQ(loaded->get_scalar_quantizer().get_min(), 0.1F);
    EXPECT_FLOAT_EQ(loaded->get_scalar_quantizer().get_max(), 0.9F);

    delete loaded;
}

TEST(SeismicSQIndexIO, write_and_read_preserves_quantizer_type_16bit) {
    TestableSeismicSQIndex original(QuantizerType::QT_16bit, 0.0F, 2.0F, 10, 2,
                                    0.5F, 5);

    original.add_docs({{{0, 1.0F}}});

    BufferedIOWriter writer;
    write_index(&original, &writer);

    BufferedIOReader reader(writer.data());
    auto* loaded =
        dynamic_cast<SeismicScalarQuantizedIndex*>(read_index(&reader));

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_scalar_quantizer().get_quantizer_type(),
              QuantizerType::QT_16bit);
    EXPECT_FLOAT_EQ(loaded->get_scalar_quantizer().get_min(), 0.0F);
    EXPECT_FLOAT_EQ(loaded->get_scalar_quantizer().get_max(), 2.0F);

    delete loaded;
}

// ============== Integration-style tests ==============

TEST(SeismicSQIndexSearch, heap_factor_controls_result_count_large_dataset) {
    constexpr int kDocCount = 100;
    constexpr int kDimension = 5000;
    constexpr int kLambda = 100;
    constexpr int kBeta = 10;
    constexpr float kAlpha = 0.5F;

    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, kLambda,
                                 kBeta, kAlpha, kDimension);

    std::vector<std::map<int, float>> docs;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> value_dist(0.1F, 1.0F);

    for (int doc_id = 0; doc_id < kDocCount; ++doc_id) {
        std::map<int, float> doc;
        doc[1000] = value_dist(rng);
        doc[2000] = value_dist(rng);
        doc[3000] = value_dist(rng);
        doc[4000] = value_dist(rng);
        doc[doc_id] = value_dist(rng);
        docs.push_back(doc);
    }

    index.add_docs(docs);
    index.build();

    std::vector<idx_t> query_indptr = {0, 4};
    std::vector<term_t> query_indices = {1000, 2000, 3000, 4000};
    std::vector<float> query_values = {0.12F, 0.64F, 0.87F, 0.53F};

    std::vector<idx_t> labels_small(kDocCount, -1);
    std::vector<float> distances_small(kDocCount, -1.0F);
    SeismicSearchParameters params_small(2, 0.000001F);
    Index* idx = &index;
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), kDocCount, distances_small.data(),
                labels_small.data(), &params_small);

    int count_small = 0;
    for (int i = 0; i < kDocCount; ++i) {
        if (labels_small[i] >= 0 && labels_small[i] < kDocCount) {
            ++count_small;
        }
    }

    std::vector<idx_t> labels_large(kDocCount, -1);
    std::vector<float> distances_large(kDocCount, -1.0F);
    SeismicSearchParameters params_large(4, 100000.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), kDocCount, distances_large.data(),
                labels_large.data(), &params_large);

    int count_large = 0;
    for (int i = 0; i < kDocCount; ++i) {
        if (labels_large[i] >= 0 && labels_large[i] < kDocCount) {
            ++count_large;
        }
    }

    EXPECT_LE(count_small, kDocCount);
    EXPECT_GT(count_small, 0);
    EXPECT_EQ(count_large, kDocCount);
    EXPECT_LE(count_small, count_large);
}

// ============== IDSelector tests ==============

TEST(SeismicSQIndexSearch, search_exact_match_with_small_selector) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 3,
                                 0.5F, 3);
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

    // Exact match should return doc1 and doc3, sorted by score descending
    EXPECT_EQ(labels[0], 1);
    EXPECT_EQ(labels[1], 3);
    EXPECT_GT(distances[0], distances[1]);
}

TEST(SeismicSQIndexSearch, search_exact_match_scores_match_normal_path_scores) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 3,
                                 0.5F, 3);
    Index* idx = &index;

    // doc0: term0=1.0, doc1: term0=0.5, doc2: term0=0.8
    index.add_docs({{{0, 1.0F}}, {{0, 0.5F}}, {{0, 0.8F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};

    // Normal path (no selector): k=3, all 3 docs returned
    std::vector<idx_t> labels_normal(3, -1);
    std::vector<float> distances_normal(3, -1.0F);
    SeismicSearchParameters params_normal(5, 1000.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 3, distances_normal.data(),
                labels_normal.data(), &params_normal);

    // Exact match path: selector size (2) <= k (2)
    std::vector<idx_t> allowed_ids = {0, 2};
    ArrayIDSelector selector(allowed_ids.size(), allowed_ids.data());
    SeismicSearchParameters params_filtered(5, 1000.0F);
    params_filtered.set_id_selector(&selector);

    std::vector<idx_t> labels_filtered(2, -1);
    std::vector<float> distances_filtered(2, -1.0F);
    idx->search(1, query_indptr.data(), query_indices.data(),
                query_values.data(), 2, distances_filtered.data(),
                labels_filtered.data(), &params_filtered);

    // Find doc0's score from the normal path
    float doc0_score_normal = -1.0F;
    for (int i = 0; i < 3; ++i) {
        if (labels_normal[i] == 0) {
            doc0_score_normal = distances_normal[i];
            break;
        }
    }
    // Find doc0's score from the exact match path
    float doc0_score_filtered = -1.0F;
    for (int i = 0; i < 2; ++i) {
        if (labels_filtered[i] == 0) {
            doc0_score_filtered = distances_filtered[i];
            break;
        }
    }

    ASSERT_GE(doc0_score_normal, 0.0F);
    ASSERT_GE(doc0_score_filtered, 0.0F);
    // Scores must match — both paths should decode quantized dot products
    EXPECT_FLOAT_EQ(doc0_score_normal, doc0_score_filtered);
    // Sanity: decoded score should be in a reasonable range (not raw quantized)
    EXPECT_LT(doc0_score_filtered, 10.0F);
}

TEST(SeismicSQIndexSearch, search_with_id_selector_filters_results) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 3,
                                 0.5F, 3);
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

    // doc1 (highest score) should be excluded
    // Results: doc2, doc0, then padding
    EXPECT_EQ(labels[0], 2);
    EXPECT_EQ(labels[1], 0);
    EXPECT_EQ(labels[2], -1);
}

// Parameters carrying no cut / heap factor fall back to the defaults, as they
// do for SeismicIndex, rather than being rejected or dereferenced as null. Null
// used to throw here; the results it now returns must be the default ones, not
// merely non-empty.
TEST(SeismicSQIndexSearch, search_without_seismic_parameters_uses_defaults) {
    TestableSeismicSQIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F, 10, 2,
                                 0.5F, 3);
    Index* idx = &index;

    index.add_docs({{{0, 1.0F}, {1, 0.5F}}, {{0, 0.2F}}});
    index.build();

    std::vector<idx_t> query_indptr = {0, 1};
    std::vector<term_t> query_indices = {0};
    std::vector<float> query_values = {1.0F};

    auto search_with = [&](SearchParameters* params) {
        std::vector<idx_t> labels(2, detail::INVALID_IDX);
        std::vector<float> distances(2, -1.0F);
        idx->search(1, query_indptr.data(), query_indices.data(),
                    query_values.data(), 2, distances.data(), labels.data(),
                    params);
        return std::make_pair(distances, labels);
    };

    SeismicSearchParameters explicit_defaults;
    const auto expected = search_with(&explicit_defaults);
    EXPECT_EQ(expected.second[0], 0);

    EXPECT_EQ(search_with(nullptr), expected);
    // An id selector holder, carrying no cut / heap factor of its own.
    SearchParameters plain_params;
    EXPECT_EQ(search_with(&plain_params), expected);
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
std::unique_ptr<TestableSeismicSQIndex> built_index(QuantizerType qtype) {
    auto index = std::make_unique<TestableSeismicSQIndex>(qtype, 0.0F, 1.0F, 10,
                                                          2, 0.5F, 5);
    index->add_docs({{{0, 1.0F}, {2, 0.9F}},
                     {{1, 0.5F}, {3, 0.7F}},
                     {{0, 0.3F}, {4, 0.8F}},
                     {{2, 0.6F}, {3, 0.4F}, {4, 0.2F}}});
    index->build();
    return index;
}

std::vector<idx_t> search_top(Index* index, term_t term, int k) {
    std::vector<idx_t> indptr = {0, 1};
    std::vector<term_t> indices = {term};
    std::vector<float> values = {1.0F};
    std::vector<idx_t> labels(k, detail::INVALID_IDX);
    std::vector<float> distances(k, -1.0F);
    SeismicSearchParameters params(5, 1.0F);
    index->search(1, indptr.data(), indices.data(), values.data(), k,
                  distances.data(), labels.data(), &params);
    return labels;
}

// Labels and scores together: a mapped read that returned subtly wrong values
// while preserving the ranking would pass a labels-only comparison.
std::pair<std::vector<float>, std::vector<idx_t>> search_scored(Index* index,
                                                                term_t term,
                                                                int k) {
    std::vector<idx_t> indptr = {0, 1};
    std::vector<term_t> indices = {term};
    std::vector<float> values = {1.0F};
    std::vector<idx_t> labels(k, detail::INVALID_IDX);
    std::vector<float> distances(k, -1.0F);
    SeismicSearchParameters params(5, 1.0F);
    index->search(1, indptr.data(), indices.data(), values.data(), k,
                  distances.data(), labels.data(), &params);
    return {distances, labels};
}

// Every byte of the stored CSR, not just its shape: the mapped reader borrows
// these arrays in place, so a wrong offset or a skipped pad shows up here.
void expect_same_vectors(const SparseVectors* lhs, const SparseVectors* rhs) {
    ASSERT_NE(lhs, nullptr);
    ASSERT_NE(rhs, nullptr);
    ASSERT_EQ(lhs->num_vectors(), rhs->num_vectors());
    ASSERT_EQ(lhs->get_element_size(), rhs->get_element_size());
    ASSERT_EQ(lhs->get_dimension(), rhs->get_dimension());

    const size_t rows = lhs->num_vectors();
    for (size_t i = 0; i <= rows; ++i) {
        ASSERT_EQ(lhs->indptr_data()[i], rhs->indptr_data()[i])
            << "indptr[" << i << "]";
    }
    const size_t nnz = static_cast<size_t>(lhs->indptr_data()[rows]);
    for (size_t i = 0; i < nnz; ++i) {
        ASSERT_EQ(lhs->indices_data()[i], rhs->indices_data()[i])
            << "indices[" << i << "]";
    }
    for (size_t i = 0; i < nnz * lhs->get_element_size(); ++i) {
        ASSERT_EQ(lhs->values_data()[i], rhs->values_data()[i])
            << "values byte " << i;
    }
}

#if defined(__linux__)
// How many mappings of `path` the process holds, straight from the kernel's
// view. The only way to tell a released mapping from a leaked one.
size_t count_mappings_of(const std::string& path) {
    std::ifstream maps("/proc/self/maps");
    std::string line;
    size_t count = 0;
    while (std::getline(maps, line)) {
        if (line.find(path) != std::string::npos) {
            ++count;
        }
    }
    return count;
}
#endif

#if !defined(_WIN32)
// Pins NSPARSE_MMAP_ADVISE for a test that depends on how the file is mapped,
// and restores whatever the environment had.
class ScopedMmapAdvise {
public:
    explicit ScopedMmapAdvise(const char* mode) {
        const char* previous = std::getenv(kVariable);
        if (previous != nullptr) {
            previous_ = previous;
        }
        ::setenv(kVariable, mode, 1);
    }
    ~ScopedMmapAdvise() {
        if (previous_.has_value()) {
            ::setenv(kVariable, previous_->c_str(), 1);
        } else {
            ::unsetenv(kVariable);
        }
    }

    ScopedMmapAdvise(const ScopedMmapAdvise&) = delete;
    ScopedMmapAdvise& operator=(const ScopedMmapAdvise&) = delete;

private:
    static constexpr const char* kVariable = "NSPARSE_MMAP_ADVISE";
    std::optional<std::string> previous_;
};
#endif

// Where read_index leaves off before the payload. SESQ's payload opens with the
// quantization header write_quantization_header wrote.
constexpr size_t kQuantizerTypeOffset = kIndexHeaderSize;
constexpr size_t kVminOffset = kQuantizerTypeOffset + sizeof(QuantizerType);

template <class T>
T read_field(const std::string& path, size_t offset) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(offset));
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

// A load-time guard rejects what the writer cannot produce, so the only way to
// reach one is to overwrite a field of an otherwise valid file in place.
template <class T>
void patch_field(const std::string& path, size_t offset, T value) {
    std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out);
    out.seekp(static_cast<std::streamoff>(offset));
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

// Both readers of a corrupted file, each of which has to reject it on its own.
// The message is checked, not just the type: reading on past a guard that was
// dropped throws too, somewhere downstream, and that must not read as a pass.
template <class Error>
void expect_both_reads_rejected(char* path, const char* fragment) {
    for (const int flags : {0, static_cast<int>(IndexIoFlag::kUseMmap)}) {
        try {
            std::unique_ptr<Index> loaded(read_index(path, flags));
            ADD_FAILURE() << "accepted the file, flags " << flags;
        } catch (const Error& error) {
            EXPECT_NE(std::string(error.what()).find(fragment),
                      std::string::npos)
                << "flags " << flags << ": " << error.what();
        }
    }
}

}  // namespace

class SeismicSQIndexMmapIO : public testing::TestWithParam<QuantizerType> {};

INSTANTIATE_TEST_SUITE_P(QuantizerTypes, SeismicSQIndexMmapIO,
                         testing::Values(QuantizerType::QT_8bit,
                                         QuantizerType::QT_16bit),
                         [](const testing::TestParamInfo<QuantizerType>& info) {
                             return info.param == QuantizerType::QT_8bit
                                        ? "bit8"
                                        : "bit16";
                         });

TEST_P(SeismicSQIndexMmapIO, mapped_read_matches_the_copying_read) {
    TempIndexFile file("nsparse_sesq_mapped.idx");

    auto source = built_index(GetParam());
    write_index(source.get(), file.c_str());

    // One file, read both ways: kUseMmap is all that separates the residencies.
    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    std::unique_ptr<Index> copied(read_index(file.c_str()));

    ASSERT_NE(mapped, nullptr);
    ASSERT_NE(copied, nullptr);
    ASSERT_EQ(mapped->get_vectors()->num_vectors(),
              copied->get_vectors()->num_vectors());
    // The quantizer header is part of the mapped payload, not just the copied
    // one: without it the codes would be decoded against the wrong range.
    const auto* mapped_sq =
        dynamic_cast<SeismicScalarQuantizedIndex*>(mapped.get());
    ASSERT_NE(mapped_sq, nullptr);
    EXPECT_EQ(mapped_sq->get_scalar_quantizer().get_quantizer_type(),
              GetParam());
    EXPECT_EQ(mapped->get_vectors()->get_element_size(),
              mapped_sq->get_scalar_quantizer().bytes_per_value());

    // The borrowed CSR must be the copied one byte for byte, not merely the
    // same shape.
    expect_same_vectors(mapped->get_vectors(), copied->get_vectors());

    for (term_t term = 0; term < 5; ++term) {
        const auto [mapped_scores, mapped_labels] =
            search_scored(mapped.get(), term, 4);
        const auto [copied_scores, copied_labels] =
            search_scored(copied.get(), term, 4);
        EXPECT_EQ(mapped_labels, copied_labels) << "term " << term;
        ASSERT_EQ(mapped_scores.size(), copied_scores.size());
        for (size_t i = 0; i < mapped_scores.size(); ++i) {
            EXPECT_FLOAT_EQ(mapped_scores[i], copied_scores[i])
                << "term " << term << " score " << i;
        }
    }
}

// The quantized payload opens with a 9-byte quantizer header (a 1-byte enum and
// two floats), which shifts every array behind it relative to the float layout
// -- and the padding io_align inserts is computed from those absolute offsets.
// An odd nnz and an empty row move the offsets again; 8-bit codes cannot catch
// a values-array padding bug on their own, since alignment 1 makes the padding
// always zero.
TEST_P(SeismicSQIndexMmapIO, mapped_read_matches_the_copying_read_odd_layout) {
    TempIndexFile file("nsparse_sesq_odd_layout.idx");

    // nnz = 5 (odd), with an empty row in the middle.
    TestableSeismicSQIndex source(GetParam(), 0.0F, 1.0F, 10, 2, 0.5F, 5);
    source.add_docs(
        {{{0, 1.0F}, {3, 0.9F}}, {}, {{1, 0.4F}}, {{2, 0.7F}, {4, 0.25F}}});
    source.build();
    write_index(&source, file.c_str());

    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    std::unique_ptr<Index> copied(read_index(file.c_str()));
    ASSERT_NE(mapped, nullptr);
    ASSERT_NE(copied, nullptr);

    ASSERT_EQ(mapped->get_vectors()->indptr_data()[4], 5);
    expect_same_vectors(mapped->get_vectors(), copied->get_vectors());
    for (term_t term = 0; term < 5; ++term) {
        EXPECT_EQ(search_top(mapped.get(), term, 4),
                  search_top(copied.get(), term, 4))
            << "term " << term;
    }
}

// What the padding exists for: the values array is reinterpreted at the code
// width in place, so a mapped start that is not a multiple of that width is UB
// on x86 and faults on ARM. MmapCursor rejects a misaligned array, but only for
// types whose alignof says so -- values are bytes on the wire, so this is the
// only check that the element-width padding actually landed.
TEST_P(SeismicSQIndexMmapIO, mapped_values_are_aligned_for_the_code_width) {
    TempIndexFile file("nsparse_sesq_align.idx");
    TestableSeismicSQIndex source(GetParam(), 0.0F, 1.0F, 10, 2, 0.5F, 5);
    source.add_docs({{{0, 1.0F}, {3, 0.9F}}, {}, {{1, 0.4F}}});  // nnz = 3, odd
    source.build();
    write_index(&source, file.c_str());

    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(mapped, nullptr);
    const auto* vectors = mapped->get_vectors();
    ASSERT_NE(vectors, nullptr);
    const auto element_size = vectors->get_element_size();
    EXPECT_EQ(
        reinterpret_cast<uintptr_t>(vectors->values_data()) % element_size, 0U);
    EXPECT_EQ(
        reinterpret_cast<uintptr_t>(vectors->indptr_data()) % alignof(idx_t),
        0U);
    EXPECT_EQ(
        reinterpret_cast<uintptr_t>(vectors->indices_data()) % alignof(term_t),
        0U);
}

// Writer and copying reader must agree on every pad, or the mapped reader --
// which computes the same padding from the same absolute offsets -- borrows
// from the wrong place. Consuming exactly what was written is that agreement.
TEST_P(SeismicSQIndexMmapIO, copying_read_consumes_exactly_what_was_written) {
    TestableSeismicSQIndex source(GetParam(), 0.0F, 1.0F, 10, 2, 0.5F, 5);
    source.add_docs({{{0, 1.0F}, {3, 0.9F}}, {}, {{1, 0.4F}}});  // nnz = 3, odd
    source.build();

    BufferedIOWriter writer;
    write_index(&source, &writer);
    const size_t written = writer.size();

    BufferedIOReader reader(writer.data());
    std::unique_ptr<Index> loaded(read_index(&reader));
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(reader.pos(), written);
}

// bytes_per_value() reads anything that is not QT_8bit as 16-bit, so an
// undefined type would pick an element width instead of being rejected, and the
// codes behind it would be strided at that width. Both readers parse the
// quantizer header themselves, so the guard has to hold on both.
TEST_P(SeismicSQIndexMmapIO, both_reads_reject_an_unknown_quantizer_type) {
    TempIndexFile file("nsparse_sesq_unknown_type.idx");
    auto source = built_index(GetParam());
    write_index(source.get(), file.c_str());
    const std::string path = file.c_str();

    // Fails loudly if the layout moves, rather than patching some other field.
    ASSERT_EQ(read_field<QuantizerType>(path, kQuantizerTypeOffset),
              GetParam());
    patch_field(path, kQuantizerTypeOffset, static_cast<QuantizerType>(2));

    expect_both_reads_rejected<std::runtime_error>(file.c_str(),
                                                  "unknown quantizer type");
}

// A type that is a valid enum but not the width the values were encoded at:
// search would stride the stored codes at the quantizer's width and read past
// the end of the array, or halfway into each code.
TEST_P(SeismicSQIndexMmapIO, both_reads_reject_a_type_the_stored_width_denies) {
    TempIndexFile file("nsparse_sesq_width_mismatch.idx");
    auto source = built_index(GetParam());
    write_index(source.get(), file.c_str());
    const std::string path = file.c_str();

    ASSERT_EQ(read_field<QuantizerType>(path, kQuantizerTypeOffset),
              GetParam());
    const auto other_width = GetParam() == QuantizerType::QT_8bit
                                 ? QuantizerType::QT_16bit
                                 : QuantizerType::QT_8bit;
    patch_field(path, kQuantizerTypeOffset, other_width);

    expect_both_reads_rejected<std::runtime_error>(
        file.c_str(), "element size disagrees with its quantizer type");
}

// The stored range reaches the quantizer's constructor, which is what rejects a
// NaN bound -- an ordered comparison cannot, and the range would then decode
// every score to NaN while the labels still looked ordered.
TEST_P(SeismicSQIndexMmapIO, both_reads_reject_a_non_finite_quantizer_range) {
    TempIndexFile file("nsparse_sesq_nan_range.idx");
    auto source = built_index(GetParam());
    write_index(source.get(), file.c_str());
    const std::string path = file.c_str();

    ASSERT_FLOAT_EQ(read_field<float>(path, kVminOffset), 0.0F);
    patch_field(path, kVminOffset, std::numeric_limits<float>::quiet_NaN());

    expect_both_reads_rejected<std::invalid_argument>(file.c_str(),
                                                     "must be finite");
}

// The point of the mapped path: the arrays point into the file, not into fresh
// allocations.
TEST(SeismicSQIndexMmapIOSingle, mapped_read_borrows_from_the_file) {
    TempIndexFile file("nsparse_sesq_borrow.idx");
    auto source = built_index(QuantizerType::QT_8bit);
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
    EXPECT_EQ(
        indices_bytes - indptr_bytes,
        static_cast<ptrdiff_t>((vectors->num_vectors() + 1) * sizeof(idx_t)));
}

// A stream has no file to map, so the flag alone must not send read_index down
// the mapped path.
TEST(SeismicSQIndexMmapIOSingle,
     buffered_read_copies_even_when_the_flag_is_set) {
    auto source = built_index(QuantizerType::QT_8bit);

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

TEST(SeismicSQIndexMmapIOSingle, mapped_read_of_an_empty_index) {
    TempIndexFile file("nsparse_sesq_empty_mapped.idx");
    SeismicScalarQuantizedIndex source(QuantizerType::QT_16bit, 0.25F, 2.0F,
                                       {.lambda = 10, .beta = 2, .alpha = 0.5F},
                                       5);
    write_index(&source, file.c_str());

    std::unique_ptr<Index> loaded(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->get_vectors(), nullptr);
    const auto* loaded_sq =
        dynamic_cast<SeismicScalarQuantizedIndex*>(loaded.get());
    ASSERT_NE(loaded_sq, nullptr);
    EXPECT_EQ(loaded_sq->get_scalar_quantizer().get_quantizer_type(),
              QuantizerType::QT_16bit);
    EXPECT_FLOAT_EQ(loaded_sq->get_scalar_quantizer().get_min(), 0.25F);
    EXPECT_FLOAT_EQ(loaded_sq->get_scalar_quantizer().get_max(), 2.0F);
}

// The mapping must go away with the index that owns it. MmapFile unmaps in its
// destructor and drops the fd right after mmap(), but nothing held that to
// account: a mapping kept alive by a stray copy, or an fd left open, only shows
// up as a process that grows until it cannot open another index.
TEST(SeismicSQIndexMmapIOSingle, mapped_index_releases_the_mapping_and_the_fd) {
#if !defined(_WIN32)
    // Both checks below assume a mapping of the file. The "hugetlb" advise mode
    // copies the file into an anonymous region instead, which the kernel does
    // not report as a mapping of the path at all, so an ambient
    // NSPARSE_MMAP_ADVISE would turn the count into a false failure.
    const ScopedMmapAdvise advise("hugepage");
#endif
    TempIndexFile file("nsparse_sesq_release.idx");
    auto source = built_index(QuantizerType::QT_8bit);
    write_index(source.get(), file.c_str());
    const std::string path = file.c_str();

#if defined(__linux__)
    const size_t before = count_mappings_of(path);
    {
        std::unique_ptr<Index> loaded(
            read_index(file.c_str(), IndexIoFlag::kUseMmap));
        ASSERT_NE(loaded, nullptr);
        ASSERT_NE(loaded->get_vectors(), nullptr);
        EXPECT_GT(count_mappings_of(path), before) << "index is not mapped";
    }
    EXPECT_EQ(count_mappings_of(path), before)
        << "mapping outlived the index that owns it";
#endif

    // Open and destroy far more times than the default fd limit: a leaked
    // descriptor per mapped read fails here with "failed to open", and a leaked
    // mapping shows up on Linux in the check above.
    for (int i = 0; i < 2048; ++i) {
        std::unique_ptr<Index> loaded(
            read_index(file.c_str(), IndexIoFlag::kUseMmap));
        ASSERT_NE(loaded, nullptr) << "iteration " << i;
        ASSERT_EQ(loaded->get_vectors()->num_vectors(), 4) << "iteration " << i;
    }
}

// The mapping outlives every borrower: searching a mapped index after the
// MmapFile has been moved into it, and destroying in the base-class order, must
// not touch unmapped memory. Runs the search twice so a use-after-unmap has
// something to read.
TEST(SeismicSQIndexMmapIOSingle, mapped_index_stays_valid_for_its_whole_life) {
    TempIndexFile file("nsparse_sesq_lifetime.idx");
    auto source = built_index(QuantizerType::QT_8bit);
    write_index(source.get(), file.c_str());

    std::unique_ptr<Index> loaded(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(loaded, nullptr);
    const auto first = search_top(loaded.get(), 2, 4);
    // Deleting the file leaves the mapping intact on POSIX; the pages are the
    // kernel's now, not the directory entry's.
    const auto second = search_top(loaded.get(), 2, 4);
    EXPECT_EQ(first, second);
    loaded.reset();  // must not fault, and must not double-unmap
}

// Building from a native codes CSR borrowed via mmap must match building the
// same corpus fed through add(): the upstream writes codes at the quantizer's
// width, read_csr(kMmap) borrows them in place, and build/write/reload/search
// is bit-exact to the add()-fed build. This is the quantized analog of the
// unquantized mmap-CSR build; the two builds see identical codes because add()
// runs the same ScalarQuantizer::encode the test wrote into the file.
TEST(SeismicSQIndexMmapIOSingle, mmap_codes_csr_build_matches_add_build) {
    constexpr int kDim = 5;
    const SeismicClusterParameters params{
        .lambda = 10, .beta = 2, .alpha = 0.5F, .seed = 42};
    // A small reproducible corpus, held as raw CSR arrays so it can be fed both
    // ways from the same bytes.
    const std::vector<idx_t> indptr = {0, 2, 4, 6, 9};
    const std::vector<term_t> indices = {0, 2, 1, 3, 0, 4, 2, 3, 4};
    const std::vector<float> values = {1.0F, 0.9F, 0.5F, 0.7F, 0.3F,
                                       0.8F, 0.6F, 0.4F, 0.2F};
    const idx_t n = static_cast<idx_t>(indptr.size()) - 1;

    // Reference: the float corpus fed through add(), which quantizes it.
    SeismicScalarQuantizedIndex added(QuantizerType::QT_8bit, 0.0F, 1.0F, params,
                                      kDim);
    added.add(n, indptr.data(), indices.data(), values.data());
    added.build();

    // Under test: the same corpus pre-quantized to codes and borrowed via mmap.
    const ScalarQuantizer sq(QuantizerType::QT_8bit, 0.0F, 1.0F);
    const size_t element_size = sq.bytes_per_value();
    std::vector<uint8_t> codes(indices.size() * element_size);
    sq.encode(values.data(), codes.data(), indices.size());

    csr_test::TempCsrFiles csr("nsparse_sesq_codes");
    csr_test::write_native_codes_csr(csr.native(), indptr, indices, codes, kDim,
                                     element_size);

    SeismicScalarQuantizedIndex mapped(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                       params, kDim);
    mapped.read_csr(csr.native().c_str(), Residency::kMmap);
    ASSERT_EQ(mapped.num_vectors(), static_cast<size_t>(n));
    mapped.build();

    TempIndexFile file("nsparse_sesq_mmapbuild.idx");
    write_index(&mapped, file.c_str());
    std::unique_ptr<Index> reloaded(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(reloaded, nullptr);

    for (term_t term = 0; term < kDim; ++term) {
        EXPECT_EQ(search_scored(reloaded.get(), term, 4),
                  search_scored(&added, term, 4))
            << "mismatch at term " << term;
    }
}

// A codes CSR whose value width does not match the index's quantizer is
// rejected by read_mcsr's native-layout size check rather than misread: here
// 16-bit-wide values (element_size 2) are fed to an 8-bit index.
TEST(SeismicSQIndexMmapIOSingle, mmap_codes_csr_wrong_width_is_rejected) {
    constexpr int kDim = 5;
    const std::vector<idx_t> indptr = {0, 2, 4};
    const std::vector<term_t> indices = {0, 2, 1, 3};
    std::vector<uint8_t> wide_codes(indices.size() * 2);
    csr_test::TempCsrFiles csr("nsparse_sesq_wrongwidth");
    csr_test::write_native_codes_csr(csr.native(), indptr, indices, wide_codes,
                                     kDim, /*element_size=*/2);

    SeismicScalarQuantizedIndex eight(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                      {.lambda = 10, .beta = 2, .alpha = 0.5F},
                                      kDim);
    EXPECT_THROW(eight.read_csr(csr.native().c_str(), Residency::kMmap),
                 std::invalid_argument);
}

}  // namespace
}  // namespace nsparse
