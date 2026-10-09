/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef DISK_SEISMIC_TEST_UTIL_H
#define DISK_SEISMIC_TEST_UTIL_H

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "nsparse/index.h"
#include "nsparse/io/index_io.h"
#include "nsparse/seismic_index.h"
#include "nsparse/types.h"
#include "tests/csr_interchange_test_util.h"

// Shared fixtures for the two disk-resident index test suites (DiskSeismicIndex
// and DiskSeismicScalarQuantizedIndex): the same random corpus, temp file, and
// batch-search harness, so each suite adds only its own assertions.
namespace nsparse::disk_seismic_test {

// Fixed cluster params + seed so builds are reproducible and (where both
// indexes are compared) they select the same candidate blocks.
inline constexpr int kLambda = 32;
inline constexpr int kBeta = 8;
inline constexpr float kAlpha = 0.4F;
inline constexpr int kSeed = 42;
inline constexpr int kDim = 400;

// K' large enough to select every candidate block (saturates the budget).
inline constexpr int kAllBlocks = 1'000'000;

inline SeismicClusterParameters cluster_params() {
    return {.lambda = kLambda, .beta = kBeta, .alpha = kAlpha, .seed = kSeed};
}

struct CSR {
    std::vector<offset_t> indptr;
    std::vector<term_t> indices;
    std::vector<float> values;
    idx_t n = 0;
};

// A reproducible random sparse corpus: each row has a random number of distinct
// terms (sorted ascending, CSR convention) with values in (0, 1] -- so the
// default quantizer range [0, 1] covers them without clipping.
inline CSR make_corpus(idx_t rows, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> nnz_dist(5, 25);
    std::uniform_int_distribution<int> term_dist(0, kDim - 1);
    std::uniform_real_distribution<float> val_dist(0.05F, 1.0F);
    CSR c;
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
        c.indptr.push_back(static_cast<offset_t>(c.indices.size()));
    }
    return c;
}

// A corpus whose last `n_victims` docs are pruned from every posting list, so
// they end up in the remainder store rather than an inline block: each victim
// carries only term 0 at a weight below every filler, and term 0 has far more
// than kLambda higher-weight fillers (each filler also owns a unique term, so
// it always survives via its own single-doc list). Victim ids are
// [n_fillers, n_fillers + n_victims). Requires n_fillers > kLambda.
inline CSR make_corpus_with_remainder(int n_fillers, int n_victims) {
    CSR c;
    c.n = n_fillers + n_victims;
    c.indptr.push_back(0);
    for (int r = 0; r < n_fillers; ++r) {
        c.indices.push_back(0);
        c.values.push_back(1.0F);
        c.indices.push_back(static_cast<term_t>(r + 1));
        c.values.push_back(1.0F);
        c.indptr.push_back(static_cast<offset_t>(c.indices.size()));
    }
    for (int v = 0; v < n_victims; ++v) {
        c.indices.push_back(0);
        c.values.push_back(0.01F);
        c.indptr.push_back(static_cast<offset_t>(c.indices.size()));
    }
    return c;
}

inline void add_corpus(Index& index, const CSR& c) {
    index.add(c.n, c.indptr.data(), c.indices.data(), c.values.data());
}

// The mmap-CSR build helpers are generic (see csr_interchange_test_util.h);
// re-exported here so the disk suite reaches them via `disk_seismic_test`.
using csr_test::TempCsrFiles;
using csr_test::write_interchange_csr;
using csr_test::write_native_codes_csr;

// Index file removed on destruction. write_index/read_index take char*.
class TempIndexFile {
public:
    explicit TempIndexFile(const std::string& name)
        : path_(std::filesystem::temp_directory_path() / name) {
        std::filesystem::remove(path_);
    }
    ~TempIndexFile() { std::filesystem::remove(path_); }
    TempIndexFile(const TempIndexFile&) = delete;
    TempIndexFile& operator=(const TempIndexFile&) = delete;
    char* c_str() { return path_str_.data(); }
    std::uintmax_t size() const { return std::filesystem::file_size(path_); }

private:
    std::filesystem::path path_{};
    std::string path_str_ = path_.string();
};

using ScoreIds =
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<idx_t>>>;

inline ScoreIds search_all(Index& index, const CSR& queries, int k,
                           SearchParameters* params) {
    std::vector<float> distances(static_cast<size_t>(queries.n) * k);
    std::vector<idx_t> labels(static_cast<size_t>(queries.n) * k);
    index.search(queries.n, queries.indptr.data(), queries.indices.data(),
                 queries.values.data(), k, distances.data(), labels.data(),
                 params);
    ScoreIds out;
    out.first.resize(queries.n);
    out.second.resize(queries.n);
    for (idx_t q = 0; q < queries.n; ++q) {
        for (int j = 0; j < k; ++j) {
            out.first[q].push_back(distances[static_cast<size_t>(q) * k + j]);
            out.second[q].push_back(labels[static_cast<size_t>(q) * k + j]);
        }
    }
    return out;
}

// Asserts every selector member appears in each query's results.
inline void expect_all_members_returned(const ScoreIds& got,
                                        const std::vector<idx_t>& members) {
    for (size_t q = 0; q < got.second.size(); ++q) {
        const std::unordered_set<idx_t> returned(got.second[q].begin(),
                                                 got.second[q].end());
        for (const idx_t member : members) {
            EXPECT_GT(returned.count(member), 0U)
                << "query " << q << " dropped selector member " << member;
        }
    }
}

// Same ranking, allowing docs with equal scores to come back in either order.
// Quantized dot products are small integers, so ties at the k-th place are
// routine and which tied doc lands inside k is not part of what a format change
// promises. Scores must still match rank for rank; within each run of equal
// scores the label sets must match, except the last run, which k may have cut
// through -- there only the count is comparable.
inline void expect_same_ranking_modulo_ties(const ScoreIds& a,
                                           const ScoreIds& b) {
    ASSERT_EQ(a.second.size(), b.second.size());
    for (size_t q = 0; q < a.second.size(); ++q) {
        ASSERT_EQ(a.first[q].size(), b.first[q].size());
        for (size_t j = 0; j < a.first[q].size(); ++j) {
            EXPECT_FLOAT_EQ(a.first[q][j], b.first[q][j])
                << "score differs at query " << q << " rank " << j;
        }
        size_t run_start = 0;
        for (size_t j = 1; j <= a.first[q].size(); ++j) {
            const bool run_ends =
                j == a.first[q].size() || a.first[q][j] != a.first[q][run_start];
            if (!run_ends) {
                continue;
            }
            const std::multiset<idx_t> got(a.second[q].begin() + run_start,
                                           a.second[q].begin() + j);
            const std::multiset<idx_t> want(b.second[q].begin() + run_start,
                                            b.second[q].begin() + j);
            if (j == a.first[q].size()) {
                EXPECT_EQ(got.size(), want.size())
                    << "trailing tie run differs in size at query " << q;
            } else {
                EXPECT_EQ(got, want)
                    << "tie run at query " << q << " rank " << run_start
                    << " holds different docs";
            }
            run_start = j;
        }
    }
}

inline void expect_same_results(const ScoreIds& a, const ScoreIds& b) {
    ASSERT_EQ(a.second.size(), b.second.size());
    for (size_t q = 0; q < a.second.size(); ++q) {
        EXPECT_EQ(a.second[q], b.second[q]) << "labels differ at query " << q;
        ASSERT_EQ(a.first[q].size(), b.first[q].size());
        for (size_t j = 0; j < a.first[q].size(); ++j) {
            EXPECT_FLOAT_EQ(a.first[q][j], b.first[q][j])
                << "score differs at query " << q << " rank " << j;
        }
    }
}

}  // namespace nsparse::disk_seismic_test

#endif  // DISK_SEISMIC_TEST_UTIL_H
