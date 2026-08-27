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

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "nsparse/disk_seismic_index.h"
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

}  // namespace nsparse
