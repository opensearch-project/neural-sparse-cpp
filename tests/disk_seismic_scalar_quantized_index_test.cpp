/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/disk_seismic_scalar_quantized_index.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "nsparse/disk_seismic_index.h"
#include "nsparse/disk_seismic_search.h"
#include "nsparse/id_selector.h"
#include "nsparse/index_factory.h"
#include "nsparse/io/buffered_io.h"
#include "nsparse/io/index_io.h"
#include "nsparse/types.h"
#include "nsparse/utils/scalar_quantizer.h"
#include "tests/disk_seismic_test_util.h"

namespace nsparse {
using namespace disk_seismic_test;  // NOLINT(build/namespaces)
namespace {

// The quantizer header (type u8 + vmin f32 + vmax f32) is the first thing
// write_index writes after the common IndexHeader, so the type byte sits right
// at kIndexHeaderSize.
constexpr size_t kQuantizerTypeOffset = kIndexHeaderSize;

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

// disk_seismic_sq is mmap-only, so the copying read throws "mmap-only" before
// any quantizer guard runs; only the mmap read reaches the guard. The message
// is checked, not just the type, so a guard silently dropped (and something
// downstream throwing instead) does not read as a pass.
void expect_mmap_read_rejected(char* path, const char* fragment) {
    try {
        std::unique_ptr<Index> loaded(read_index(path, IndexIoFlag::kUseMmap));
        ADD_FAILURE() << "accepted the corrupted file";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find(fragment), std::string::npos)
            << error.what();
    }
}

// Mean fraction of `reference`'s valid labels recovered by `got`, per query.
double recall(const ScoreIds& got, const ScoreIds& reference) {
    double sum = 0.0;
    const size_t nq = reference.second.size();
    for (size_t q = 0; q < nq; ++q) {
        std::unordered_set<idx_t> hits(got.second[q].begin(),
                                       got.second[q].end());
        size_t found = 0;
        size_t total = 0;
        for (const idx_t label : reference.second[q]) {
            if (label == detail::INVALID_IDX) {
                continue;
            }
            ++total;
            if (hits.count(label) > 0) {
                ++found;
            }
        }
        if (total > 0) {
            sum += static_cast<double>(found) / static_cast<double>(total);
        }
    }
    return nq > 0 ? sum / static_cast<double>(nq) : 1.0;
}

}  // namespace

// Bit-exact anchor: reading a selected block's codes from the inline mapping
// (fwd_) gives the same result as reading them from the in-RAM CSR (vectors_)
// of a fresh build, at the same K' -- same clusters, same blocks, same codes
// and integer dots, same decode.
TEST(DiskSeismicSQIndex, MappedReloadMatchesFreshBuild8bit) {
    const CSR corpus = make_corpus(1500, /*seed=*/1);
    const CSR queries = make_corpus(40, /*seed=*/2);

    DiskSeismicScalarQuantizedIndex disk(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                         cluster_params(), kDim);
    add_corpus(disk, corpus);
    disk.build();
    DiskSeismicSearchParameters params(/*cut=*/25, /*k_prime=*/32);
    const ScoreIds fresh = search_all(disk, queries, 10, &params);  // vectors_

    TempIndexFile file("nsparse_disk_seismic_sq_parity8.idx");
    write_index(&disk, file.c_str());
    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(mapped, nullptr);
    EXPECT_EQ(mapped->num_vectors(), static_cast<size_t>(corpus.n));
    expect_same_results(search_all(*mapped, queries, 10, &params),
                        fresh);  // fwd_
}

// An enumerable selector of size <= k must return every member, not just those
// the block budget scores. Many-block corpus + tiny k' so the budget cannot
// incidentally cover the scattered members. In-RAM path.
TEST(DiskSeismicSQIndex, SmallSelectorReturnsAllMembersInMemory) {
    const CSR corpus = make_corpus(2000, /*seed=*/1);
    const CSR queries = make_corpus(20, /*seed=*/2);
    DiskSeismicScalarQuantizedIndex disk(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                         cluster_params(), kDim);
    add_corpus(disk, corpus);
    disk.build();

    std::vector<idx_t> members = {5, 100, 250, 500, 900, 1200, 1600, 1999};
    SetIDSelector selector(members.size(), members.data());
    DiskSeismicSearchParameters params(/*cut=*/10, /*k_prime=*/1);
    params.set_id_selector(&selector);

    expect_all_members_returned(search_all(disk, queries, 10, &params),
                                members);
}

// Same contract on the mmap-loaded serialized index.
TEST(DiskSeismicSQIndex, SmallSelectorReturnsAllMembersMmap) {
    const CSR corpus = make_corpus(2000, /*seed=*/1);
    const CSR queries = make_corpus(20, /*seed=*/2);
    DiskSeismicScalarQuantizedIndex disk(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                         cluster_params(), kDim);
    add_corpus(disk, corpus);
    disk.build();
    TempIndexFile file("nsparse_dssq_exact_match.idx");
    write_index(&disk, file.c_str());
    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(mapped, nullptr);

    std::vector<idx_t> members = {5, 100, 250, 500, 900, 1200, 1600, 1999};
    SetIDSelector selector(members.size(), members.data());
    DiskSeismicSearchParameters params(/*cut=*/10, /*k_prime=*/1);
    params.set_id_selector(&selector);

    expect_all_members_returned(search_all(*mapped, queries, 10, &params),
                                members);
}

// The mapped path (directory + remainder) must return the same members and
// scores as the in-RAM path.
TEST(DiskSeismicSQIndex, ExactMatchMappedMatchesInMemory) {
    const CSR corpus = make_corpus(2000, /*seed=*/1);
    const CSR queries = make_corpus(20, /*seed=*/2);
    DiskSeismicScalarQuantizedIndex disk(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                         cluster_params(), kDim);
    add_corpus(disk, corpus);
    disk.build();
    std::vector<idx_t> members = {5, 100, 250, 500, 900, 1200, 1600, 1999};
    SetIDSelector selector(members.size(), members.data());
    DiskSeismicSearchParameters params(/*cut=*/10, /*k_prime=*/1);
    params.set_id_selector(&selector);

    const ScoreIds in_memory = search_all(disk, queries, 10, &params);
    TempIndexFile file("nsparse_dssq_exact_parity.idx");
    write_index(&disk, file.c_str());
    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(mapped, nullptr);
    expect_same_results(search_all(*mapped, queries, 10, &params), in_memory);
}

// Docs pruned from every block must still be returned and scored, through the
// remainder store. The victims are provably fully pruned (see the helper), so
// with k_prime=1 they can only surface via the mapped exact-match path.
TEST(DiskSeismicSQIndex, RemainderPathReturnsFullyPrunedMembers) {
    const int n_fillers = 40;
    const int n_victims = 3;
    const CSR corpus = make_corpus_with_remainder(n_fillers, n_victims);
    const CSR queries = make_corpus(5, /*seed=*/3);
    DiskSeismicScalarQuantizedIndex disk(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                         cluster_params(), kDim);
    add_corpus(disk, corpus);
    disk.build();
    std::vector<idx_t> members;
    for (int v = 0; v < n_victims; ++v) {
        members.push_back(n_fillers + v);
    }
    SetIDSelector selector(members.size(), members.data());
    DiskSeismicSearchParameters params(/*cut=*/10, /*k_prime=*/1);
    params.set_id_selector(&selector);

    const ScoreIds in_memory = search_all(disk, queries, 10, &params);
    TempIndexFile file("nsparse_dssq_remainder.idx");
    write_index(&disk, file.c_str());
    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(mapped, nullptr);
    const ScoreIds got = search_all(*mapped, queries, 10, &params);
    expect_all_members_returned(got, members);
    expect_same_results(got, in_memory);
}

// A selector member outside the index (id >= num_vectors) is skipped, not read
// out of bounds, on both the in-RAM and the mapped exact-match paths.
TEST(DiskSeismicSQIndex, ExactMatchSkipsOutOfRangeSelectorMember) {
    const CSR corpus = make_corpus(2000, /*seed=*/1);
    const CSR queries = make_corpus(20, /*seed=*/2);
    DiskSeismicScalarQuantizedIndex disk(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                         cluster_params(), kDim);
    add_corpus(disk, corpus);
    disk.build();
    std::vector<idx_t> members = {5, 100, static_cast<idx_t>(corpus.n) + 50};
    const std::vector<idx_t> valid = {5, 100};
    SetIDSelector selector(members.size(), members.data());
    DiskSeismicSearchParameters params(/*cut=*/10, /*k_prime=*/1);
    params.set_id_selector(&selector);

    expect_all_members_returned(search_all(disk, queries, 10, &params), valid);
    TempIndexFile file("nsparse_dssq_oob.idx");
    write_index(&disk, file.c_str());
    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(mapped, nullptr);
    expect_all_members_returned(search_all(*mapped, queries, 10, &params),
                                valid);
}

// Same fresh-vs-mapped parity at 16-bit (the other quantizer width).
TEST(DiskSeismicSQIndex, MappedReloadMatchesFreshBuild16bit) {
    const CSR corpus = make_corpus(1500, /*seed=*/1);
    const CSR queries = make_corpus(40, /*seed=*/2);

    DiskSeismicScalarQuantizedIndex disk(QuantizerType::QT_16bit, 0.0F, 1.0F,
                                         cluster_params(), kDim);
    add_corpus(disk, corpus);
    disk.build();
    DiskSeismicSearchParameters params(/*cut=*/25, /*k_prime=*/32);
    const ScoreIds fresh = search_all(disk, queries, 10, &params);

    TempIndexFile file("nsparse_disk_seismic_sq_parity16.idx");
    write_index(&disk, file.c_str());
    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(mapped, nullptr);
    expect_same_results(search_all(*mapped, queries, 10, &params), fresh);
}

// add() encodes to the quantizer's element width (1 byte for 8-bit, 2 for
// 16-bit), so the forward index and summaries are stored as codes.
TEST(DiskSeismicSQIndex, AddEncodesToCodeWidth) {
    const CSR corpus = make_corpus(50, /*seed=*/9);
    DiskSeismicScalarQuantizedIndex eight(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                          cluster_params(), kDim);
    add_corpus(eight, corpus);
    ASSERT_NE(eight.get_vectors(), nullptr);
    EXPECT_EQ(eight.get_vectors()->get_element_size(), U8);

    DiskSeismicScalarQuantizedIndex sixteen(QuantizerType::QT_16bit, 0.0F, 1.0F,
                                            cluster_params(), kDim);
    add_corpus(sixteen, corpus);
    ASSERT_NE(sixteen.get_vectors(), nullptr);
    EXPECT_EQ(sixteen.get_vectors()->get_element_size(), U16);
}

// With the cut covering every query term and the budget saturated, both indexes
// score the identical candidate set, so any ranking difference is quantization
// alone. 8-bit over the corpus's [0.05, 1] range recovers nearly all of float's
// top-k. The floor is conservative, not a target.
TEST(DiskSeismicSQIndex, RecallCloseToFloat) {
    const CSR corpus = make_corpus(1500, /*seed=*/11);
    const CSR queries = make_corpus(40, /*seed=*/12);
    DiskSeismicSearchParameters params(/*cut=*/25, kAllBlocks);

    DiskSeismicIndex fdisk(kDim, cluster_params());
    add_corpus(fdisk, corpus);
    fdisk.build();
    const ScoreIds fref = search_all(fdisk, queries, 10, &params);

    DiskSeismicScalarQuantizedIndex qdisk(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                          cluster_params(), kDim);
    add_corpus(qdisk, corpus);
    qdisk.build();
    const ScoreIds qres = search_all(qdisk, queries, 10, &params);

    EXPECT_GE(recall(qres, fref), 0.80);
}

// The whole point: codes shrink the on-disk index versus float (forward vals
// 4B/nnz -> 1B or 2B/nnz, summaries likewise), for the same corpus and
// clusters. Both widths must win, so a 16-bit-specific layout or padding
// regression cannot inflate the file past float unnoticed.
TEST(DiskSeismicSQIndex, SmallerThanFloat) {
    const CSR corpus = make_corpus(1500, /*seed=*/13);

    DiskSeismicIndex fdisk(kDim, cluster_params());
    add_corpus(fdisk, corpus);
    fdisk.build();
    TempIndexFile ffile("nsparse_disk_seismic_sq_float.idx");
    write_index(&fdisk, ffile.c_str());
    const auto float_size = ffile.size();

    for (const QuantizerType qt :
         {QuantizerType::QT_8bit, QuantizerType::QT_16bit}) {
        DiskSeismicScalarQuantizedIndex qdisk(qt, 0.0F, 1.0F, cluster_params(),
                                              kDim);
        add_corpus(qdisk, corpus);
        qdisk.build();
        TempIndexFile qfile("nsparse_disk_seismic_sq_codes.idx");
        write_index(&qdisk, qfile.c_str());
        EXPECT_LT(qfile.size(), float_size)
            << "quantized index (" << qfile.size()
            << ") not smaller than float (" << float_size << ") at width "
            << (qt == QuantizerType::QT_8bit ? 1 : 2);
    }
}

// K' (block budget) must be positive.
TEST(DiskSeismicSQIndex, RejectsNonPositiveBlockBudget) {
    const CSR corpus = make_corpus(300, /*seed=*/5);
    const CSR queries = make_corpus(2, /*seed=*/6);
    DiskSeismicScalarQuantizedIndex disk(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                         cluster_params(), kDim);
    add_corpus(disk, corpus);
    disk.build();
    std::vector<float> distances(2 * 10);
    std::vector<idx_t> labels(2 * 10);
    Index& idx = disk;
    for (const int bad : {0, -1}) {
        DiskSeismicSearchParameters params(/*cut=*/10, bad);
        EXPECT_THROW(idx.search(queries.n, queries.indptr.data(),
                                queries.indices.data(), queries.values.data(),
                                10, distances.data(), labels.data(), &params),
                     std::invalid_argument);
    }
}

// mmap-only: the copying read path is unsupported.
TEST(DiskSeismicSQIndex, CopyingReadThrows) {
    const CSR corpus = make_corpus(300, /*seed=*/5);
    DiskSeismicScalarQuantizedIndex disk(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                         cluster_params(), kDim);
    add_corpus(disk, corpus);
    disk.build();
    TempIndexFile file("nsparse_disk_seismic_sq_copying.idx");
    write_index(&disk, file.c_str());
    EXPECT_THROW(read_index(file.c_str(), /*io_flags=*/0), std::runtime_error);
    // A stream reader cannot be mapped either, so the flag still throws.
    BufferedIOWriter writer;
    write_index(&disk, &writer);
    BufferedIOReader reader(writer.data());
    EXPECT_THROW(read_index(&reader, IndexIoFlag::kUseMmap),
                 std::runtime_error);
}

// bytes_per_value() reads anything that is not QT_8bit as 16-bit, so an
// undefined type would pick an element width instead of being rejected, and the
// codes behind it would be strided at that width. The mmap reader parses the
// quantizer header itself, so the guard has to hold there.
TEST(DiskSeismicSQIndex, MmapReadRejectsUnknownQuantizerType) {
    const CSR corpus = make_corpus(300, /*seed=*/21);
    DiskSeismicScalarQuantizedIndex disk(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                         cluster_params(), kDim);
    add_corpus(disk, corpus);
    disk.build();
    TempIndexFile file("nsparse_disk_seismic_sq_unknown_type.idx");
    write_index(&disk, file.c_str());

    // Fails loudly if the layout moves, rather than patching some other field.
    ASSERT_EQ(read_field<QuantizerType>(file.c_str(), kQuantizerTypeOffset),
              QuantizerType::QT_8bit);
    patch_field(std::string(file.c_str()), kQuantizerTypeOffset,
                static_cast<QuantizerType>(2));

    expect_mmap_read_rejected(file.c_str(), "unknown quantizer type");
}

// A type that is a valid enum but not the width the codes were stored at:
// search would stride the stored codes at the quantizer's width and read past
// the end of the array, or halfway into each code.
TEST(DiskSeismicSQIndex, MmapReadRejectsWidthMismatch) {
    const CSR corpus = make_corpus(300, /*seed=*/22);
    DiskSeismicScalarQuantizedIndex disk(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                         cluster_params(), kDim);
    add_corpus(disk, corpus);
    disk.build();
    TempIndexFile file("nsparse_disk_seismic_sq_width_mismatch.idx");
    write_index(&disk, file.c_str());

    ASSERT_EQ(read_field<QuantizerType>(file.c_str(), kQuantizerTypeOffset),
              QuantizerType::QT_8bit);
    // Valid enum, but the codes were written at 8-bit width.
    patch_field(std::string(file.c_str()), kQuantizerTypeOffset,
                QuantizerType::QT_16bit);

    expect_mmap_read_rejected(file.c_str(),
                              "element size disagrees with its quantizer type");
}

// An empty index (no docs) round-trips: count 0, no results, no crash. The
// quantizer header still writes and reads back.
TEST(DiskSeismicSQIndex, EmptyIndexRoundTrip) {
    DiskSeismicScalarQuantizedIndex disk(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                         cluster_params(), kDim);
    TempIndexFile file("nsparse_disk_seismic_sq_empty.idx");
    write_index(&disk, file.c_str());
    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(mapped, nullptr);
    EXPECT_EQ(mapped->num_vectors(), 0U);

    const CSR queries = make_corpus(3, /*seed=*/6);
    std::vector<float> distances(3 * 5, -1.0F);
    std::vector<idx_t> labels(3 * 5, detail::INVALID_IDX);
    DiskSeismicSearchParameters params(10, 50);
    mapped->search(queries.n, queries.indptr.data(), queries.indices.data(),
                   queries.values.data(), 5, distances.data(), labels.data(),
                   &params);  // must not crash; nothing found
}

// index_factory understands the "disk_seismic_sq" descriptor and its quantizer
// parameter.
TEST(DiskSeismicSQIndex, FactoryCreatesIt) {
    std::unique_ptr<Index> eight(index_factory(
        kDim, "disk_seismic_sq,quantizer=8bit|lambda=32|beta=8|seed=42"));
    ASSERT_NE(eight, nullptr);
    EXPECT_EQ(eight->id(), DiskSeismicScalarQuantizedIndex::name);

    std::unique_ptr<Index> sixteen(index_factory(
        kDim, "disk_seismic_sq,quantizer=16bit|lambda=32|beta=8|seed=42"));
    ASSERT_NE(sixteen, nullptr);
    auto* typed =
        dynamic_cast<DiskSeismicScalarQuantizedIndex*>(sixteen.get());
    ASSERT_NE(typed, nullptr);
    EXPECT_EQ(typed->get_scalar_quantizer().get_quantizer_type(),
              QuantizerType::QT_16bit);
}

// Building the disk quantized index from a native codes CSR borrowed via mmap
// must match the add()-fed build: the upstream writes codes at the quantizer's
// width, read_csr(kMmap) borrows them, and build -> persist -> mmap-reload ->
// search is bit-exact to feeding the same corpus (quantized by add()) then
// building. Both see identical codes because add() runs the same
// ScalarQuantizer::encode the test wrote. Also asserts a width-mismatched codes
// file is rejected rather than misread.
TEST(DiskSeismicSQIndex, MmapCodesCsrBuildMatchesAddBuild) {
    const CSR corpus = make_corpus(1500, /*seed=*/1);
    const CSR queries = make_corpus(40, /*seed=*/2);
    DiskSeismicSearchParameters params(/*cut=*/25, /*k_prime=*/32);

    // Reference: float corpus fed through add() (which quantizes), reloaded.
    DiskSeismicScalarQuantizedIndex added(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                          cluster_params(), kDim);
    add_corpus(added, corpus);
    added.build();
    TempIndexFile added_file("nsparse_dssq_addbuild.idx");
    write_index(&added, added_file.c_str());
    std::unique_ptr<Index> added_mapped(
        read_index(added_file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(added_mapped, nullptr);
    const ScoreIds fresh = search_all(*added_mapped, queries, 10, &params);

    // Under test: the same corpus pre-quantized to codes, borrowed via mmap.
    const ScalarQuantizer sq(QuantizerType::QT_8bit, 0.0F, 1.0F);
    const size_t element_size = sq.bytes_per_value();
    std::vector<uint8_t> codes(corpus.indices.size() * element_size);
    sq.encode(corpus.values.data(), codes.data(), corpus.indices.size());
    TempCsrFiles csr("nsparse_dssq_codes");
    write_native_codes_csr(csr.native(), corpus.indptr, corpus.indices, codes,
                           kDim, element_size);

    DiskSeismicScalarQuantizedIndex mapped(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                           cluster_params(), kDim);
    mapped.read_csr(csr.native().c_str(), Residency::kMmap);
    ASSERT_EQ(mapped.num_vectors(), static_cast<size_t>(corpus.n));
    mapped.build();
    TempIndexFile built_file("nsparse_dssq_mmapbuild.idx");
    write_index(&mapped, built_file.c_str());
    std::unique_ptr<Index> built_mapped(
        read_index(built_file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(built_mapped, nullptr);
    EXPECT_EQ(built_mapped->num_vectors(), static_cast<size_t>(corpus.n));
    expect_same_results(search_all(*built_mapped, queries, 10, &params), fresh);

    // A codes file whose width does not match the index's quantizer is rejected
    // by the native-layout size check: 16-bit-wide values fed to an 8-bit index.
    std::vector<uint8_t> wide_codes(corpus.indices.size() * 2);
    TempCsrFiles wrong("nsparse_dssq_wrongwidth");
    write_native_codes_csr(wrong.native(), corpus.indptr, corpus.indices,
                           wide_codes, kDim, /*element_size=*/2);
    DiskSeismicScalarQuantizedIndex eight(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                          cluster_params(), kDim);
    EXPECT_THROW(eight.read_csr(wrong.native().c_str(), Residency::kMmap),
                 std::invalid_argument);
}

// --- Truncated inline forward index (inline_max_nnz) ---

// The anchor for the whole scheme: truncating the extra inline copies changes
// only how candidates are *ranked* before re-scoring, so with every block
// selected and a re-scoring depth past the candidate pool the result must be
// bit-identical to keeping every copy whole.
TEST(DiskSeismicSQIndex, TruncatedInlineMatchesWholeWhenEverythingIsRescored) {
    const CSR corpus = make_corpus(1500, /*seed=*/1);
    const CSR queries = make_corpus(40, /*seed=*/2);
    // A limit well below the corpus's 5..25 nnz per doc, so most copies are cut.
    constexpr uint32_t kInlineMaxNnz = 6;

    DiskSeismicScalarQuantizedIndex whole(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                          cluster_params(), kDim);
    add_corpus(whole, corpus);
    whole.build();
    TempIndexFile whole_file("nsparse_dssq_trunc_whole.idx");
    write_index(&whole, whole_file.c_str());
    std::unique_ptr<Index> whole_mapped(
        read_index(whole_file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(whole_mapped, nullptr);

    DiskSeismicScalarQuantizedIndex cut(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                        cluster_params(), kDim, kInlineMaxNnz);
    add_corpus(cut, corpus);
    cut.build();
    TempIndexFile cut_file("nsparse_dssq_trunc_cut.idx");
    write_index(&cut, cut_file.c_str());
    std::unique_ptr<Index> cut_mapped(
        read_index(cut_file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(cut_mapped, nullptr);

    // Same clusters (fixed seed) and every block selected, so both see the same
    // candidate pool.
    DiskSeismicSearchParameters params(/*cut=*/10, /*k_prime=*/kAllBlocks);
    DiskSeismicSearchParameters rescore_all(/*cut=*/10, /*k_prime=*/kAllBlocks,
                                            /*rescore=*/100000);
    expect_same_ranking_modulo_ties(
        search_all(*cut_mapped, queries, 10, &rescore_all),
        search_all(*whole_mapped, queries, 10, &params));
    EXPECT_LT(cut_file.size(), whole_file.size());
}

// The whole point: the file gets smaller, and the truncation survives the round
// trip so a reader knows it must re-score.
TEST(DiskSeismicSQIndex, TruncatedInlineShrinksTheFile) {
    const CSR corpus = make_corpus(1200, /*seed=*/7);
    std::uintmax_t previous = 0;
    // Tighter limits are monotonically smaller files; 0 keeps every copy whole.
    for (const uint32_t limit : {0U, 12U, 6U, 3U}) {
        DiskSeismicScalarQuantizedIndex index(QuantizerType::QT_8bit, 0.0F,
                                              1.0F, cluster_params(), kDim,
                                              limit);
        add_corpus(index, corpus);
        index.build();
        TempIndexFile file("nsparse_dssq_trunc_size.idx");
        write_index(&index, file.c_str());
        if (previous != 0) {
            EXPECT_LT(file.size(), previous) << "limit " << limit;
        }
        previous = file.size();
        std::unique_ptr<Index> mapped(
            read_index(file.c_str(), IndexIoFlag::kUseMmap));
        ASSERT_NE(mapped, nullptr);
        // Searching a truncated index still returns k results.
        const CSR queries = make_corpus(5, /*seed=*/8);
        DiskSeismicSearchParameters params(/*cut=*/10, /*k_prime=*/32);
        const ScoreIds got = search_all(*mapped, queries, 10, &params);
        for (const auto& ids : got.second) {
            EXPECT_EQ(ids.size(), 10U);
        }
    }
}

// Docs that no block holds live in the remainder store, which is never
// truncated -- so the exact-match path over them keeps working, and the
// re-scoring pass can still resolve them.
TEST(DiskSeismicSQIndex, TruncationLeavesRemainderDocsWhole) {
    const CSR corpus = make_corpus_with_remainder(/*n_fillers=*/64,
                                                  /*n_victims=*/3);
    const std::vector<idx_t> victims = {64, 65, 66};
    DiskSeismicScalarQuantizedIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                          cluster_params(), kDim,
                                          /*inline_max_nnz=*/1);
    add_corpus(index, corpus);
    index.build();
    TempIndexFile file("nsparse_dssq_trunc_remainder.idx");
    write_index(&index, file.c_str());
    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(mapped, nullptr);

    CSR queries;
    queries.n = 1;
    queries.indptr = {0, 1};
    queries.indices = {0};
    queries.values = {1.0F};
    SetIDSelector selector(victims.size(), victims.data());
    DiskSeismicSearchParameters params(/*cut=*/10, /*k_prime=*/32);
    params.set_id_selector(&selector);
    expect_all_members_returned(search_all(*mapped, queries, 10, &params),
                                victims);
}

// A truncated index built through the factory string, since that is how the
// knob reaches production callers.
TEST(DiskSeismicSQIndex, FactoryAcceptsInlineMaxNnz) {
    const CSR corpus = make_corpus(600, /*seed=*/3);
    std::unique_ptr<Index> index(index_factory(
        kDim,
        "disk_seismic_sq,quantizer=8bit|vmin=0.0|vmax=1.0|lambda=32|beta=8|"
        "alpha=0.4|seed=42|inline_max_nnz=6"));
    ASSERT_NE(index, nullptr);
    add_corpus(*index, corpus);
    index->build();
    TempIndexFile file("nsparse_dssq_trunc_factory.idx");
    write_index(index.get(), file.c_str());
    std::unique_ptr<Index> mapped(
        read_index(file.c_str(), IndexIoFlag::kUseMmap));
    ASSERT_NE(mapped, nullptr);
    EXPECT_EQ(mapped->num_vectors(), static_cast<size_t>(corpus.n));
}

// --- The rescore search parameter ---

// Builds, writes, and mmap-reloads a seeded 8-bit index over `corpus`, with
// inline_max_nnz = `limit` (0 = every copy whole). Truncation only exists on
// disk, so every rescore test has to go through a reload.
std::unique_ptr<Index> mapped_sq(const CSR& corpus, uint32_t limit,
                                 TempIndexFile& file) {
    DiskSeismicScalarQuantizedIndex index(QuantizerType::QT_8bit, 0.0F, 1.0F,
                                          cluster_params(), kDim, limit);
    add_corpus(index, corpus);
    index.build();
    write_index(&index, file.c_str());
    return std::unique_ptr<Index>(read_index(file.c_str(), IndexIoFlag::kUseMmap));
}

TEST(DiskSeismicSearchParameters, CarriesRescoreDepth) {
    EXPECT_EQ(DiskSeismicSearchParameters().rescore, kDefaultRescoreDepth);
    EXPECT_EQ(DiskSeismicSearchParameters(3, 50).rescore, kDefaultRescoreDepth);
    EXPECT_EQ(DiskSeismicSearchParameters(3, 50, 77).rescore, 77);
    EXPECT_EQ(DiskSeismicSQSearchParameters(0.0F, 1.0F, 3, 50).rescore,
              kDefaultRescoreDepth);
    EXPECT_EQ(DiskSeismicSQSearchParameters(0.0F, 1.0F, 3, 50, 77).rescore, 77);

    // resolve_cut_and_budget is what search reads it through: a disk parameter
    // carries its own depth, anything else gets the default.
    const DiskSeismicSQSearchParameters sq(0.0F, 1.0F, 4, 60, 77);
    const detail::DiskSeismicCutBudget from_sq =
        detail::resolve_cut_and_budget(&sq);
    EXPECT_EQ(from_sq.cut, 4);
    EXPECT_EQ(from_sq.k_prime, 60);
    EXPECT_EQ(from_sq.rescore, 77);
    const SeismicSearchParameters plain(5, 1.0F);
    EXPECT_EQ(detail::resolve_cut_and_budget(&plain).rescore,
              kDefaultRescoreDepth);
    EXPECT_EQ(detail::resolve_cut_and_budget(nullptr).rescore,
              kDefaultRescoreDepth);
}

// On an index whose copies are all whole there is nothing to re-score, so the
// depth must not change a single result.
TEST(DiskSeismicSQIndex, RescoreIsIgnoredWithoutTruncation) {
    const CSR corpus = make_corpus(1200, /*seed=*/11);
    const CSR queries = make_corpus(30, /*seed=*/12);
    TempIndexFile file("nsparse_dssq_rescore_whole.idx");
    std::unique_ptr<Index> mapped = mapped_sq(corpus, 0, file);
    ASSERT_NE(mapped, nullptr);
    DiskSeismicSearchParameters shallow(/*cut=*/10, /*k_prime=*/32, /*rescore=*/10);
    DiskSeismicSearchParameters deep(/*cut=*/10, /*k_prime=*/32, /*rescore=*/100000);
    expect_same_results(search_all(*mapped, queries, 10, &shallow),
                        search_all(*mapped, queries, 10, &deep));
}

// After re-scoring, a returned score is the doc's true score, not the lower
// bound its truncated copy gave -- even at depth k, where the second pass
// cannot change which docs come back, only what they score.
TEST(DiskSeismicSQIndex, RescoredScoresAreExact) {
    const CSR corpus = make_corpus(1200, /*seed=*/13);
    const CSR queries = make_corpus(20, /*seed=*/14);
    constexpr int kK = 10;
    TempIndexFile cut_file("nsparse_dssq_rescore_exact_cut.idx");
    std::unique_ptr<Index> cut = mapped_sq(corpus, /*limit=*/4, cut_file);
    TempIndexFile whole_file("nsparse_dssq_rescore_exact_whole.idx");
    std::unique_ptr<Index> whole = mapped_sq(corpus, 0, whole_file);
    ASSERT_NE(cut, nullptr);
    ASSERT_NE(whole, nullptr);

    for (const int depth : {kK, 50}) {
        DiskSeismicSearchParameters params(/*cut=*/10, /*k_prime=*/32, depth);
        const ScoreIds got = search_all(*cut, queries, kK, &params);
        for (idx_t q = 0; q < queries.n; ++q) {
            // An enumerable selector of size <= k takes the exact-match path,
            // which scores exactly the named docs against their whole vectors
            // in the untruncated index: an independent oracle.
            std::vector<idx_t> members;
            for (const idx_t id : got.second[q]) {
                if (id != detail::INVALID_IDX) {
                    members.push_back(id);
                }
            }
            SetIDSelector selector(members.size(), members.data());
            DiskSeismicSearchParameters exact(/*cut=*/10, /*k_prime=*/32);
            exact.set_id_selector(&selector);
            CSR one;
            one.n = 1;
            one.indptr = {0, queries.indptr[q + 1] - queries.indptr[q]};
            one.indices.assign(queries.indices.begin() + queries.indptr[q],
                               queries.indices.begin() + queries.indptr[q + 1]);
            one.values.assign(queries.values.begin() + queries.indptr[q],
                              queries.values.begin() + queries.indptr[q + 1]);
            const ScoreIds want = search_all(*whole, one, kK, &exact);
            for (size_t j = 0; j < members.size(); ++j) {
                const auto at = std::find(want.second[0].begin(),
                                          want.second[0].end(), members[j]);
                ASSERT_NE(at, want.second[0].end());
                EXPECT_FLOAT_EQ(got.first[q][j],
                                want.first[0][at - want.second[0].begin()])
                    << "depth " << depth << " query " << q << " doc "
                    << members[j] << " returned a lower bound";
            }
        }
    }
}

// Fewer than k re-scored candidates could not fill the result, so a depth below
// k behaves exactly like depth k.
TEST(DiskSeismicSQIndex, RescoreBelowKIsRaisedToK) {
    const CSR corpus = make_corpus(1200, /*seed=*/15);
    const CSR queries = make_corpus(30, /*seed=*/16);
    TempIndexFile file("nsparse_dssq_rescore_floor.idx");
    std::unique_ptr<Index> mapped = mapped_sq(corpus, /*limit=*/4, file);
    ASSERT_NE(mapped, nullptr);
    DiskSeismicSearchParameters at_k(/*cut=*/10, /*k_prime=*/32, /*rescore=*/10);
    const ScoreIds want = search_all(*mapped, queries, 10, &at_k);
    for (const int below : {0, 1, 9}) {
        DiskSeismicSearchParameters params(/*cut=*/10, /*k_prime=*/32, below);
        expect_same_results(search_all(*mapped, queries, 10, &params), want);
    }
}

// A deeper second pass re-scores a superset of candidates, and the k best exact
// scores of a superset are never worse -- so each query's summed top-k score is
// non-decreasing in the depth. (Recall itself can only tie-break differently.)
TEST(DiskSeismicSQIndex, DeeperRescoreNeverLowersTheResult) {
    const CSR corpus = make_corpus(1500, /*seed=*/17);
    const CSR queries = make_corpus(30, /*seed=*/18);
    TempIndexFile file("nsparse_dssq_rescore_monotone.idx");
    std::unique_ptr<Index> mapped = mapped_sq(corpus, /*limit=*/4, file);
    ASSERT_NE(mapped, nullptr);
    std::vector<double> previous(queries.n, -1.0);
    for (const int depth : {10, 20, 50, 200, 100000}) {
        DiskSeismicSearchParameters params(/*cut=*/10, /*k_prime=*/kAllBlocks,
                                           depth);
        const ScoreIds got = search_all(*mapped, queries, 10, &params);
        for (idx_t q = 0; q < queries.n; ++q) {
            double sum = 0.0;
            for (const float score : got.first[q]) {
                sum += score > 0.0F ? score : 0.0F;
            }
            EXPECT_GE(sum, previous[q] - 1e-4)
                << "depth " << depth << " lowered query " << q;
            previous[q] = sum;
        }
    }
}

// A negative depth is a caller bug, rejected on any disk index whether or not it
// would have been used, like a non-positive k_prime.
TEST(DiskSeismicSQIndex, RejectsNegativeRescore) {
    const CSR corpus = make_corpus(600, /*seed=*/19);
    const CSR queries = make_corpus(5, /*seed=*/20);
    for (const uint32_t limit : {0U, 4U}) {
        TempIndexFile file("nsparse_dssq_rescore_negative.idx");
        std::unique_ptr<Index> mapped = mapped_sq(corpus, limit, file);
        ASSERT_NE(mapped, nullptr);
        DiskSeismicSearchParameters params(/*cut=*/10, /*k_prime=*/32,
                                           /*rescore=*/-1);
        EXPECT_THROW(search_all(*mapped, queries, 10, &params),
                     std::invalid_argument)
            << "inline_max_nnz " << limit;
    }
}

// A malformed limit must fail the build spec, not quietly become "no
// truncation" -- std::stoul alone would wrap "-1" to ULONG_MAX.
TEST(DiskSeismicSQIndex, FactoryRejectsABadInlineMaxNnz) {
    const std::string base =
        "disk_seismic_sq,quantizer=8bit|vmin=0.0|vmax=1.0|lambda=32|beta=8|"
        "inline_max_nnz=";
    for (const char* bad : {"-1", "4294967296", "abc", ""}) {
        EXPECT_THROW(std::unique_ptr<Index>(
                         index_factory(kDim, (base + bad).c_str())),
                     std::exception)
            << "inline_max_nnz=" << bad;
    }
    EXPECT_NE(std::unique_ptr<Index>(index_factory(kDim, (base + "0").c_str())),
              nullptr);
    EXPECT_NE(std::unique_ptr<Index>(
                  index_factory(kDim, (base + "4294967295").c_str())),
              nullptr);
}

}  // namespace nsparse
