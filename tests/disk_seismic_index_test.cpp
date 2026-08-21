/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/disk_seismic_index.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "nsparse/index_factory.h"
#include "nsparse/io/buffered_io.h"
#include "nsparse/io/index_io.h"
#include "nsparse/seismic_index.h"
#include "nsparse/types.h"

namespace nsparse {
namespace {

// Fixed cluster params + seed so SeismicIndex and DiskSeismicIndex build
// bit-identical clusters and their results can be compared exactly.
constexpr int kLambda = 32;
constexpr int kBeta = 8;
constexpr float kAlpha = 0.4F;
constexpr int kSeed = 42;
constexpr int kDim = 400;

SeismicClusterParameters cluster_params() {
    return {.lambda = kLambda, .beta = kBeta, .alpha = kAlpha, .seed = kSeed};
}

struct CSR {
    std::vector<idx_t> indptr;
    std::vector<term_t> indices;
    std::vector<float> values;
    idx_t n = 0;
};

// A reproducible random sparse corpus: each row has a random number of distinct
// terms (sorted ascending, CSR convention) with values in (0, 1].
CSR make_corpus(idx_t rows, unsigned seed) {
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
        while (static_cast<int>(terms.size()) < nnz)
            terms.insert(term_dist(rng));
        for (const int t : terms) {
            c.indices.push_back(static_cast<term_t>(t));
            c.values.push_back(val_dist(rng));
        }
        c.indptr.push_back(static_cast<idx_t>(c.indices.size()));
    }
    return c;
}

void add_corpus(Index& index, const CSR& c) {
    index.add(c.n, c.indptr.data(), c.indices.data(), c.values.data());
}

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

private:
    std::filesystem::path path_{};
    std::string path_str_ = path_.string();
};

using ScoreIds =
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<idx_t>>>;

// K' large enough to select every candidate block (saturates the budget).
constexpr int kAllBlocks = 1'000'000;

ScoreIds search_all(Index& index, const CSR& queries, int k,
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

void expect_same_results(const ScoreIds& a, const ScoreIds& b) {
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

}  // namespace

// Bit-exact anchor: reading a selected block's vectors from the inline mapping
// (fwd_) gives the same result as reading them from the in-RAM CSR (vectors_)
// of a fresh build, at the same K' -- same clusters, same selected blocks, same
// docs and dots.
TEST(DiskSeismicIndex, MappedReloadMatchesFreshBuild) {
    const CSR corpus = make_corpus(1500, /*seed=*/1);
    const CSR queries = make_corpus(40, /*seed=*/2);

    DiskSeismicIndex disk(kDim, cluster_params());
    add_corpus(disk, corpus);
    disk.build();
    DiskSeismicSearchParameters params(/*cut=*/10, /*k_prime=*/32);
    const ScoreIds fresh = search_all(disk, queries, 10, &params);  // vectors_

    TempIndexFile file("nsparse_disk_seismic_parity.idx");
    write_index(&disk, file.c_str());
    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(mapped, nullptr);
    EXPECT_EQ(mapped->num_vectors(), static_cast<size_t>(corpus.n));
    expect_same_results(search_all(*mapped, queries, 10, &params),
                        fresh);  // fwd_
}

// K' is a monotonic depth budget: the global top-K' blocks are nested by
// summary score, so a larger budget scores a superset of docs and every rank's
// score can only improve. This also proves the parameter is actually consumed.
TEST(DiskSeismicIndex, BlockBudgetIsMonotone) {
    const CSR corpus = make_corpus(1500, /*seed=*/7);
    const CSR queries = make_corpus(40, /*seed=*/8);
    DiskSeismicIndex disk(kDim, cluster_params());
    add_corpus(disk, corpus);
    disk.build();
    TempIndexFile file("nsparse_disk_seismic_monotone.idx");
    write_index(&disk, file.c_str());
    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(mapped, nullptr);

    ScoreIds prev;
    bool have_prev = false;
    bool any_improved = false;
    for (const int kp : {1, 4, 16, 64, kAllBlocks}) {
        DiskSeismicSearchParameters params(/*cut=*/10, kp);
        const ScoreIds cur = search_all(*mapped, queries, 10, &params);
        if (have_prev) {
            for (size_t q = 0; q < cur.first.size(); ++q) {
                for (size_t j = 0; j < cur.first[q].size(); ++j) {
                    EXPECT_GE(cur.first[q][j], prev.first[q][j] - 1e-6F)
                        << "rank " << j << " regressed as K' grew, query " << q;
                    if (cur.first[q][j] > prev.first[q][j] + 1e-6F) {
                        any_improved = true;
                    }
                }
            }
        }
        prev = cur;
        have_prev = true;
    }
    EXPECT_TRUE(any_improved) << "varying K' had no effect on any result";
}

// Once K' reaches the candidate-block count, a larger budget changes nothing.
TEST(DiskSeismicIndex, BlockBudgetSaturates) {
    const CSR corpus = make_corpus(1200, /*seed=*/3);
    const CSR queries = make_corpus(30, /*seed=*/4);
    DiskSeismicIndex disk(kDim, cluster_params());
    add_corpus(disk, corpus);
    disk.build();
    TempIndexFile file("nsparse_disk_seismic_saturate.idx");
    write_index(&disk, file.c_str());
    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(mapped, nullptr);
    DiskSeismicSearchParameters big(/*cut=*/10, kAllBlocks);
    DiskSeismicSearchParameters bigger(/*cut=*/10, kAllBlocks * 2);
    expect_same_results(search_all(*mapped, queries, 10, &big),
                        search_all(*mapped, queries, 10, &bigger));
}

// K' (block budget) must be positive.
TEST(DiskSeismicIndex, RejectsNonPositiveBlockBudget) {
    const CSR corpus = make_corpus(300, /*seed=*/5);
    const CSR queries = make_corpus(2, /*seed=*/6);
    DiskSeismicIndex disk(kDim, cluster_params());
    add_corpus(disk, corpus);
    disk.build();
    std::vector<float> distances(2 * 10);
    std::vector<idx_t> labels(2 * 10);
    Index& idx = disk;  // search() is a public method of the Index base
    for (const int bad : {0, -1}) {
        DiskSeismicSearchParameters params(/*cut=*/10, bad);
        EXPECT_THROW(idx.search(queries.n, queries.indptr.data(),
                                queries.indices.data(), queries.values.data(),
                                10, distances.data(), labels.data(), &params),
                     std::invalid_argument);
    }
}

// mmap-only: the copying read path is unsupported.
TEST(DiskSeismicIndex, CopyingReadThrows) {
    const CSR corpus = make_corpus(300, /*seed=*/5);
    DiskSeismicIndex disk(kDim, cluster_params());
    add_corpus(disk, corpus);
    disk.build();
    TempIndexFile file("nsparse_disk_seismic_copying.idx");
    write_index(&disk, file.c_str());
    // No kUseMmap -> falls to the copying reader, which throws.
    EXPECT_THROW(read_index(file.c_str(), /*io_flags=*/0), std::runtime_error);
    // A stream reader cannot be mapped either, so the flag still throws.
    BufferedIOWriter writer;
    write_index(&disk, &writer);
    BufferedIOReader reader(writer.data());
    EXPECT_THROW(read_index(&reader, IndexIoFlag::kUseMmap),
                 std::runtime_error);
}

// An empty index (no docs) round-trips: count 0, no results, no crash.
TEST(DiskSeismicIndex, EmptyIndexRoundTrip) {
    DiskSeismicIndex disk(kDim, cluster_params());  // no add(), no build()
    TempIndexFile file("nsparse_disk_seismic_empty.idx");
    write_index(&disk, file.c_str());
    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(mapped, nullptr);
    EXPECT_EQ(mapped->num_vectors(), 0U);

    const CSR queries = make_corpus(3, /*seed=*/6);
    std::vector<float> distances(3 * 5, -1.0F);
    std::vector<idx_t> labels(3 * 5, detail::INVALID_IDX);
    SeismicSearchParameters params(10, 1.0F);
    mapped->search(queries.n, queries.indptr.data(), queries.indices.data(),
                   queries.values.data(), 5, distances.data(), labels.data(),
                   &params);  // must not crash; nothing found
}

// index_factory understands the "disk_seismic" descriptor.
TEST(DiskSeismicIndex, FactoryCreatesIt) {
    std::unique_ptr<Index> index(
        index_factory(kDim, "disk_seismic,lambda=32|beta=8|alpha=0.4|seed=42"));
    ASSERT_NE(index, nullptr);
    EXPECT_EQ(index->id(), DiskSeismicIndex::name);
}

}  // namespace nsparse
