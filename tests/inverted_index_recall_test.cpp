/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nsparse/inverted_index.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "nsparse/types.h"

namespace nsparse {
namespace {

// Brute-force top-k: compute dot product for all docs and return top-k doc ids.
std::vector<idx_t> brute_force_top_k(
    const std::vector<std::vector<std::pair<term_t, float>>>& docs,
    const std::vector<std::pair<term_t, float>>& query, int k) {
    std::vector<std::pair<float, idx_t>> scores;
    for (idx_t d = 0; d < static_cast<idx_t>(docs.size()); ++d) {
        float score = 0.0F;
        // Build a quick lookup for doc terms.
        std::map<term_t, float> doc_map;
        for (auto& [t, v] : docs[d]) {
            doc_map[t] = v;
        }
        for (auto& [qt, qv] : query) {
            auto it = doc_map.find(qt);
            if (it != doc_map.end()) {
                score += qv * it->second;
            }
        }
        if (score > 0.0F) {
            scores.emplace_back(score, d);
        }
    }
    // Sort descending by score, then ascending by doc_id for tie-breaking.
    std::sort(scores.begin(), scores.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return a.second < b.second;
              });
    std::vector<idx_t> result;
    for (int i = 0; i < k && i < static_cast<int>(scores.size()); ++i) {
        result.push_back(scores[i].second);
    }
    return result;
}

// Generate random sparse documents.
struct RandomSparseData {
    int dim;
    int n_docs;
    int n_queries;
    int avg_nnz;
    std::vector<std::vector<std::pair<term_t, float>>> docs;
    std::vector<std::vector<std::pair<term_t, float>>> queries;
};

RandomSparseData generate_random_data(int dim, int n_docs, int n_queries,
                                       int avg_nnz, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> nnz_dist(
        std::max(1, avg_nnz / 2), avg_nnz * 2);
    std::uniform_int_distribution<int> term_dist(0, dim - 1);
    std::uniform_real_distribution<float> val_dist(0.01F, 2.0F);

    RandomSparseData data;
    data.dim = dim;
    data.n_docs = n_docs;
    data.n_queries = n_queries;
    data.avg_nnz = avg_nnz;

    // Generate documents.
    for (int d = 0; d < n_docs; ++d) {
        int nnz = nnz_dist(rng);
        std::set<term_t> used_terms;
        std::vector<std::pair<term_t, float>> doc;
        for (int j = 0; j < nnz; ++j) {
            term_t t = static_cast<term_t>(term_dist(rng));
            if (used_terms.insert(t).second) {
                doc.emplace_back(t, val_dist(rng));
            }
        }
        // Sort by term for consistent ordering.
        std::sort(doc.begin(), doc.end());
        data.docs.push_back(std::move(doc));
    }

    // Generate queries.
    for (int q = 0; q < n_queries; ++q) {
        int nnz = nnz_dist(rng);
        std::set<term_t> used_terms;
        std::vector<std::pair<term_t, float>> query;
        for (int j = 0; j < nnz; ++j) {
            term_t t = static_cast<term_t>(term_dist(rng));
            if (used_terms.insert(t).second) {
                query.emplace_back(t, val_dist(rng));
            }
        }
        std::sort(query.begin(), query.end());
        data.queries.push_back(std::move(query));
    }

    return data;
}

void add_docs_from_data(
    InvertedIndex& index,
    const std::vector<std::vector<std::pair<term_t, float>>>& docs) {
    std::vector<idx_t> indptr;
    std::vector<term_t> indices;
    std::vector<float> values;

    indptr.push_back(0);
    for (const auto& doc : docs) {
        for (const auto& [term, value] : doc) {
            indices.push_back(term);
            values.push_back(value);
        }
        indptr.push_back(static_cast<idx_t>(indices.size()));
    }

    index.add(static_cast<idx_t>(docs.size()), indptr.data(), indices.data(),
              values.data());
}

float compute_recall(const std::vector<idx_t>& predicted,
                     const std::vector<idx_t>& ground_truth) {
    std::set<idx_t> gt_set(ground_truth.begin(), ground_truth.end());
    int hits = 0;
    for (auto id : predicted) {
        if (id != INVALID_IDX && gt_set.count(id)) {
            ++hits;
        }
    }
    return ground_truth.empty()
               ? 1.0F
               : static_cast<float>(hits) / static_cast<float>(gt_set.size());
}

// Test recall with various dataset sizes and k values.
class InvertedIndexRecallTest
    : public ::testing::TestWithParam<std::tuple<int, int, int, int>> {};

TEST_P(InvertedIndexRecallTest, recall_should_be_perfect) {
    auto [n_docs, dim, avg_nnz, k] = GetParam();
    int n_queries = 50;

    auto data = generate_random_data(dim, n_docs, n_queries, avg_nnz);

    InvertedIndex index(dim);
    add_docs_from_data(index, data.docs);
    index.build();
    Index* idx = &index;

    float total_recall = 0.0F;
    int n_valid_queries = 0;

    for (int q = 0; q < n_queries; ++q) {
        const auto& query = data.queries[q];
        if (query.empty()) continue;

        // Build query CSR.
        std::vector<idx_t> q_indptr = {0,
                                        static_cast<idx_t>(query.size())};
        std::vector<term_t> q_indices;
        std::vector<float> q_values;
        for (auto& [t, v] : query) {
            q_indices.push_back(t);
            q_values.push_back(v);
        }

        std::vector<idx_t> labels(k, INVALID_IDX);
        std::vector<float> distances(k, -1.0F);

        idx->search(1, q_indptr.data(), q_indices.data(), q_values.data(), k,
                     distances.data(), labels.data());

        // Brute-force ground truth.
        auto gt = brute_force_top_k(data.docs, query, k);

        float recall = compute_recall(
            std::vector<idx_t>(labels.begin(), labels.end()), gt);

        total_recall += recall;
        ++n_valid_queries;

        // Print failing queries for debugging.
        if (recall < 1.0F) {
            std::cout << "Query " << q << ": recall=" << recall
                      << " predicted=[";
            for (int i = 0; i < k; ++i) {
                if (i > 0) std::cout << ",";
                std::cout << labels[i];
            }
            std::cout << "] gt=[";
            for (size_t i = 0; i < gt.size(); ++i) {
                if (i > 0) std::cout << ",";
                std::cout << gt[i];
            }
            std::cout << "]" << std::endl;

            // Print scores for debugging.
            std::cout << "  predicted_scores=[";
            for (int i = 0; i < k; ++i) {
                if (i > 0) std::cout << ",";
                std::cout << distances[i];
            }
            std::cout << "]" << std::endl;

            // Print ground truth scores.
            std::cout << "  gt_scores=[";
            for (size_t i = 0; i < gt.size(); ++i) {
                if (i > 0) std::cout << ",";
                // Compute score for gt[i].
                float score = 0.0F;
                std::map<term_t, float> doc_map;
                for (auto& [t, v] : data.docs[gt[i]]) doc_map[t] = v;
                for (auto& [qt, qv] : query) {
                    auto it = doc_map.find(qt);
                    if (it != doc_map.end()) score += qv * it->second;
                }
                std::cout << score;
            }
            std::cout << "]" << std::endl;
        }
    }

    float avg_recall =
        n_valid_queries > 0 ? total_recall / n_valid_queries : 1.0F;
    std::cout << "Average recall: " << avg_recall << " (n_docs=" << n_docs
              << ", dim=" << dim << ", avg_nnz=" << avg_nnz << ", k=" << k
              << ")" << std::endl;

    // InvertedIndex should have PERFECT recall (it's an exact algorithm).
    EXPECT_FLOAT_EQ(avg_recall, 1.0F)
        << "InvertedIndex recall should be 1.0 but got " << avg_recall;
}

INSTANTIATE_TEST_SUITE_P(
    RecallTests, InvertedIndexRecallTest,
    ::testing::Values(
        // (n_docs, dim, avg_nnz, k)
        // Small: single window, no pruning.
        std::make_tuple(100, 500, 10, 10),
        // Medium: multiple windows, pruning active.
        std::make_tuple(5000, 1000, 20, 10),
        // Large: many windows, aggressive pruning.
        std::make_tuple(10000, 2000, 30, 10),
        // Large with higher k.
        std::make_tuple(10000, 2000, 30, 50),
        // Dense queries, many terms.
        std::make_tuple(5000, 500, 50, 10),
        // Very sparse queries and docs.
        std::make_tuple(10000, 5000, 5, 10)));

}  // namespace
}  // namespace nsparse
