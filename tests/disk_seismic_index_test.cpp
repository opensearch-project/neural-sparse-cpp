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

#include <cstdint>
#include <memory>
#include <vector>

#include "nsparse/index_factory.h"
#include "nsparse/io/buffered_io.h"
#include "nsparse/io/index_io.h"
#include "nsparse/seismic_index.h"
#include "nsparse/types.h"
#include "nsparse/utils/csr_layout.h"
#include "tests/disk_seismic_test_util.h"

namespace nsparse {
using namespace disk_seismic_test;  // NOLINT(build/namespaces)

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

// Building from a native CSR borrowed via mmap must match building the same
// corpus fed through add(): convert -> read_csr(kMmap) -> build -> persist ->
// mmap-reload -> search is bit-exact to the add()-fed build. This is also the
// regression guard for the num_vectors_ desync -- read_csr(kMmap) borrows
// vectors_ without running add(), so unless num_vectors_ is synced the written
// index records nv=0 and a reloaded search returns nothing.
TEST(DiskSeismicIndex, MmapCsrBuildMatchesAddBuild) {
    const CSR corpus = make_corpus(1500, /*seed=*/1);
    const CSR queries = make_corpus(40, /*seed=*/2);
    DiskSeismicSearchParameters params(/*cut=*/10, /*k_prime=*/32);

    // Reference: the same corpus fed through add(), then reloaded via mmap.
    DiskSeismicIndex added(kDim, cluster_params());
    add_corpus(added, corpus);
    added.build();
    TempIndexFile added_file("nsparse_disk_seismic_addbuild.idx");
    write_index(&added, added_file.c_str());
    std::unique_ptr<Index> added_mapped(
        read_index(added_file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(added_mapped, nullptr);
    const ScoreIds fresh = search_all(*added_mapped, queries, 10, &params);

    // Under test: build from a native CSR of the same corpus, borrowed via mmap.
    TempCsrFiles csr("nsparse_disk_seismic_source");
    write_interchange_csr(csr.interchange(), corpus, kDim);
    csr_layout::convert(csr.interchange(), csr.native());

    DiskSeismicIndex mapped_build(kDim, cluster_params());
    mapped_build.read_csr(csr.native().c_str(), Residency::kMmap);
    // num_vectors_ must reflect the borrowed vectors (regression guard).
    ASSERT_EQ(mapped_build.num_vectors(), static_cast<size_t>(corpus.n));
    mapped_build.build();
    TempIndexFile built_file("nsparse_disk_seismic_mmapbuild.idx");
    write_index(&mapped_build, built_file.c_str());
    std::unique_ptr<Index> built_mapped(
        read_index(built_file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(built_mapped, nullptr);
    // The persisted doc count must survive the mmap-CSR build.
    EXPECT_EQ(built_mapped->num_vectors(), static_cast<size_t>(corpus.n));
    expect_same_results(search_all(*built_mapped, queries, 10, &params), fresh);
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
