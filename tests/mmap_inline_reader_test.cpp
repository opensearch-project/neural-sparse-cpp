/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/io/mmap_inline_reader.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/inline_forward_index.h"
#include "nsparse/io/file_io.h"
#include "nsparse/io/inline_forward_index_writer.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace {

constexpr float kAlpha = 0.5F;

nsparse::SparseVectors create_float_vectors(
    const std::vector<std::vector<nsparse::term_t>>& indices_list,
    const std::vector<std::vector<float>>& values_list, size_t dimension) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = dimension});
    for (size_t i = 0; i < indices_list.size(); ++i) {
        vectors.add_vector(
            indices_list[i].data(), indices_list[i].size(),
            reinterpret_cast<const uint8_t*>(values_list[i].data()),
            values_list[i].size() * sizeof(float));
    }
    return vectors;
}

nsparse::SparseVectors create_uint16_vectors(
    const std::vector<std::vector<nsparse::term_t>>& indices_list,
    const std::vector<std::vector<uint16_t>>& values_list, size_t dimension) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U16, .dimension = dimension});
    for (size_t i = 0; i < indices_list.size(); ++i) {
        vectors.add_vector(
            indices_list[i].data(), indices_list[i].size(),
            reinterpret_cast<const uint8_t*>(values_list[i].data()),
            values_list[i].size() * sizeof(uint16_t));
    }
    return vectors;
}

nsparse::SparseVectors create_uint8_vectors(
    const std::vector<std::vector<nsparse::term_t>>& indices_list,
    const std::vector<std::vector<uint8_t>>& values_list, size_t dimension) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U8, .dimension = dimension});
    for (size_t i = 0; i < indices_list.size(); ++i) {
        vectors.add_vector(indices_list[i].data(), indices_list[i].size(),
                           values_list[i].data(), values_list[i].size());
    }
    return vectors;
}

std::vector<nsparse::InvertedListClusters> build_lists(
    const std::vector<std::vector<std::vector<nsparse::idx_t>>>& layout,
    const nsparse::SparseVectors& vectors) {
    std::vector<nsparse::InvertedListClusters> lists;
    lists.reserve(layout.size());
    for (const auto& blocks : layout) {
        nsparse::InvertedListClusters list(blocks);
        list.summarize(&vectors, kAlpha);
        lists.push_back(std::move(list));
    }
    return lists;
}

// Writes an inline file to a unique temp path and removes it on destruction.
class TempInlineFile {
public:
    TempInlineFile(const std::string& name,
                   const std::vector<nsparse::InvertedListClusters>& lists,
                   const nsparse::SparseVectors& vectors,
                   const nsparse::InlineForwardIndexWriter& writer)
        : path_(std::filesystem::temp_directory_path() / name) {
        nsparse::FileIOWriter file_writer(path_str_.data());
        writer.write(lists, vectors, &file_writer);
        file_writer.close();  // flush before the reader mmaps it
    }

    // Non-throwing: a destructor cannot forward, and Windows refuses to delete
    // a file while a mapping over it is still open.
    ~TempInlineFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    TempInlineFile(const TempInlineFile&) = delete;
    TempInlineFile& operator=(const TempInlineFile&) = delete;

    // Not path_.c_str(): path::value_type is wchar_t on Windows, so the
    // narrowed form has to be held somewhere with the same lifetime.
    const char* path() const { return path_str_.c_str(); }

private:
    std::filesystem::path path_;
    std::string path_str_ = path_.string();
};

// Overwrite `n` bytes at `off` in a file (used to corrupt a valid inline file).
void poke(const char* path, size_t off, const void* bytes, size_t n) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(file.is_open());
    file.seekp(static_cast<std::streamoff>(off));
    file.write(static_cast<const char*>(bytes),
               static_cast<std::streamsize>(n));
    ASSERT_TRUE(file.good());
}

size_t file_size(const char* path) {
    return static_cast<size_t>(std::filesystem::file_size(path));
}

// Assert a block's records reproduce the source documents exactly, in order.
void verify_block(const nsparse::MmapInlineReader& reader,
                  const nsparse::BlockView& view,
                  const std::vector<nsparse::idx_t>& expected_docs,
                  const nsparse::SparseVectors& vectors) {
    ASSERT_NE(view.data, nullptr);
    ASSERT_EQ(view.n_docs, expected_docs.size());

    const size_t element_size = reader.element_size();
    const auto* indptr = vectors.indptr_data();
    const auto* indices = vectors.indices_data();
    const auto* values = vectors.values_data();

    const uint8_t* cursor = view.records();
    const uint8_t* end = view.data + view.len;
    for (size_t i = 0; i < view.n_docs; ++i) {
        const nsparse::InlineDocRecord rec =
            nsparse::read_inline_record(cursor, end, element_size);
        EXPECT_EQ(rec.doc_id, static_cast<uint32_t>(expected_docs[i]));
        const nsparse::idx_t start = indptr[rec.doc_id];
        const uint32_t src_nnz =
            static_cast<uint32_t>(indptr[rec.doc_id + 1] - start);
        ASSERT_EQ(rec.nnz, src_nnz);
        EXPECT_EQ(0, std::memcmp(rec.comps, indices + start,
                                 rec.nnz * sizeof(nsparse::term_t)));
        EXPECT_EQ(
            0, std::memcmp(rec.vals,
                           values + static_cast<size_t>(start) * element_size,
                           static_cast<size_t>(rec.nnz) * element_size));
    }
    // The cursor consumed exactly the block's recorded payload length.
    EXPECT_EQ(static_cast<uint64_t>(cursor - view.data), view.len);
}

// A shared multi-list layout with docs duplicated across blocks and lists.
const std::vector<std::vector<std::vector<nsparse::idx_t>>> kLayout = {
    {{0, 1}, {2, 3, 4}}, {{3, 5}, {0}}};

nsparse::SparseVectors sample_float_vectors() {
    return create_float_vectors({{0, 3}, {1}, {2, 4, 6}, {0, 9}, {5}, {3, 7}},
                                {{1.0F, 2.0F},
                                 {3.0F},
                                 {1.5F, 2.5F, 3.5F},
                                 {4.0F, 5.0F},
                                 {6.0F},
                                 {7.0F, 8.0F}},
                                10);
}

}  // namespace

TEST(MmapInlineReader, RoundTripPageAligned) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    nsparse::InlineForwardIndexWriter writer;  // page-aligned default
    TempInlineFile file("nsparse_inline_page_aligned.bin", lists, vectors,
                        writer);

    nsparse::MmapInlineReader reader(file.path());
    EXPECT_EQ(reader.element_size(), nsparse::U32);
    EXPECT_EQ(reader.num_blocks(), 4U);
    EXPECT_EQ(reader.num_lists(), 2U);
    EXPECT_EQ(reader.page_size(),
              nsparse::InlineForwardIndexWriter::kDefaultPageSize);
    EXPECT_EQ(reader.num_blocks_in_list(0), 2U);
    EXPECT_EQ(reader.num_blocks_in_list(1), 2U);

    verify_block(reader, reader.block(0, 0), kLayout[0][0], vectors);
    verify_block(reader, reader.block(0, 1), kLayout[0][1], vectors);
    verify_block(reader, reader.block(1, 0), kLayout[1][0], vectors);
    verify_block(reader, reader.block(1, 1), kLayout[1][1], vectors);
    // Every mapped block starts on a page boundary.
    EXPECT_EQ(
        static_cast<size_t>(reader.block(1, 0).data - reader.block(0, 0).data) %
            reader.page_size(),
        0U);
}

TEST(MmapInlineReader, RoundTripPacked) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    nsparse::InlineForwardIndexWriter writer(
        nsparse::InlineForwardIndexWriter::kDefaultPageSize,
        nsparse::InlineLayout::kPacked);
    TempInlineFile file("nsparse_inline_packed.bin", lists, vectors, writer);

    nsparse::MmapInlineReader reader(file.path());
    EXPECT_EQ(reader.page_size(), 1U);
    EXPECT_EQ(reader.num_blocks(), 4U);
    verify_block(reader, reader.block(0, 0), kLayout[0][0], vectors);
    verify_block(reader, reader.block(0, 1), kLayout[0][1], vectors);
    verify_block(reader, reader.block(1, 0), kLayout[1][0], vectors);
    verify_block(reader, reader.block(1, 1), kLayout[1][1], vectors);
}

TEST(MmapInlineReader, ElementSizeU16) {
    auto vectors = create_uint16_vectors({{0, 3}, {1}, {2, 4, 6}},
                                         {{10, 20}, {30}, {15, 25, 35}}, 8);
    auto lists = build_lists({{{0, 1}, {2}}}, vectors);
    nsparse::InlineForwardIndexWriter writer;
    TempInlineFile file("nsparse_inline_u16.bin", lists, vectors, writer);

    nsparse::MmapInlineReader reader(file.path());
    EXPECT_EQ(reader.element_size(), nsparse::U16);
    verify_block(reader, reader.block(0, 0), {0, 1}, vectors);
    verify_block(reader, reader.block(0, 1), {2}, vectors);
}

TEST(MmapInlineReader, ElementSizeU8) {
    auto vectors = create_uint8_vectors({{0, 3}, {1}, {2, 4, 6}},
                                        {{10, 20}, {30}, {15, 25, 35}}, 8);
    auto lists = build_lists({{{0, 1}, {2}}}, vectors);
    nsparse::InlineForwardIndexWriter writer;
    TempInlineFile file("nsparse_inline_u8.bin", lists, vectors, writer);

    nsparse::MmapInlineReader reader(file.path());
    EXPECT_EQ(reader.element_size(), nsparse::U8);
    verify_block(reader, reader.block(0, 0), {0, 1}, vectors);
    verify_block(reader, reader.block(0, 1), {2}, vectors);
}

TEST(MmapInlineReader, SparseListsEmptyBlocksAndAbsentLookups) {
    // doc 0 has a zero-nnz vector; list 0 has a normal block + an empty block;
    // list 1 has no clusters (sparse pl); list 2 has one block.
    auto vectors =
        create_float_vectors({{}, {0, 3}, {1}}, {{}, {1.0F, 2.0F}, {3.0F}}, 8);
    std::vector<nsparse::InvertedListClusters> lists;
    const std::vector<std::vector<nsparse::idx_t>> blocks0 = {{0, 1}, {}};
    nsparse::InvertedListClusters list0(blocks0);
    list0.summarize(&vectors, kAlpha);
    lists.push_back(std::move(list0));
    lists.emplace_back();  // list 1: zero clusters
    const std::vector<std::vector<nsparse::idx_t>> blocks2 = {{2}};
    nsparse::InvertedListClusters list2(blocks2);
    list2.summarize(&vectors, kAlpha);
    lists.push_back(std::move(list2));

    nsparse::InlineForwardIndexWriter writer;
    TempInlineFile file("nsparse_inline_sparse.bin", lists, vectors, writer);

    nsparse::MmapInlineReader reader(file.path());
    EXPECT_EQ(reader.num_lists(), 3U);
    EXPECT_EQ(reader.num_blocks(), 3U);  // 2 (list0) + 0 (list1) + 1 (list2)
    EXPECT_EQ(reader.num_blocks_in_list(0), 2U);
    EXPECT_EQ(reader.num_blocks_in_list(1), 0U);
    EXPECT_EQ(reader.num_blocks_in_list(2), 1U);

    verify_block(reader, reader.block(0, 0), {0, 1},
                 vectors);                                  // incl. zero-nnz
    verify_block(reader, reader.block(0, 1), {}, vectors);  // empty block
    verify_block(reader, reader.block(2, 0), {2}, vectors);

    // Absent lookups return an empty view (data == nullptr).
    EXPECT_EQ(reader.block(1, 0).data, nullptr);  // empty posting list
    EXPECT_EQ(reader.block(0, 2).data, nullptr);  // block index out of range
    EXPECT_EQ(reader.block(9, 0).data, nullptr);  // posting list out of range
}

TEST(MmapInlineReader, MissingFileThrows) {
    const auto missing =
        std::filesystem::temp_directory_path() / "nsparse_inline_absent.bin";
    std::error_code ignored;
    std::filesystem::remove(missing, ignored);
    const std::string path = missing.string();
    EXPECT_THROW({ nsparse::MmapInlineReader reader(path.c_str()); },
                 std::runtime_error);
}

TEST(MmapInlineReader, TruncatedFileThrows) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    nsparse::InlineForwardIndexWriter writer;
    TempInlineFile file("nsparse_inline_truncated.bin", lists, vectors, writer);

    // Truncate below the minimum header + trailer size.
    std::filesystem::resize_file(file.path(), 8);
    EXPECT_THROW({ nsparse::MmapInlineReader reader(file.path()); },
                 std::runtime_error);
}

TEST(MmapInlineReader, MoveSemantics) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    nsparse::InlineForwardIndexWriter writer;
    TempInlineFile file("nsparse_inline_move_a.bin", lists, vectors, writer);
    TempInlineFile file2("nsparse_inline_move_b.bin", lists, vectors, writer);

    // Asserts a moved-from reader is fully inert. Buf's move keeps the source's
    // size()/data(), so this pins the explicit member reset that compensates;
    // without it block() would hand back a wild pointer (use-after-move).
    auto expect_inert = [](const nsparse::MmapInlineReader& r) {
        EXPECT_EQ(r.num_blocks(), 0U);
        EXPECT_EQ(r.num_lists(), 0U);
        EXPECT_EQ(r.element_size(), 0U);
        EXPECT_EQ(r.page_size(), 0U);
        EXPECT_EQ(r.num_blocks_in_list(0), 0U);
        EXPECT_EQ(r.num_blocks_in_list(0xFFFFFFFFu), 0U);  // uint32 wrap guard
        EXPECT_EQ(r.block(0, 0).data, nullptr);
    };

    nsparse::MmapInlineReader reader(file.path());
    nsparse::MmapInlineReader moved(std::move(reader));
    verify_block(moved, moved.block(0, 0), kLayout[0][0], vectors);
    expect_inert(reader);  // inert after move-construction

    nsparse::MmapInlineReader reader2(file2.path());
    reader2 = std::move(moved);
    verify_block(reader2, reader2.block(0, 0), kLayout[0][0], vectors);
    expect_inert(moved);  // inert after move-assignment (pins the op= reset)

    // Self-move-assignment is a safe no-op (aliased to dodge -Wself-move).
    nsparse::MmapInlineReader& alias = reader2;
    reader2 = std::move(alias);
    verify_block(reader2, reader2.block(0, 0), kLayout[0][0], vectors);
}

TEST(MmapInlineReader, NullPathThrows) {
    // A null path is a programming error, distinct from a bad file.
    EXPECT_THROW({ nsparse::MmapInlineReader reader(nullptr); },
                 std::invalid_argument);
}

TEST(MmapInlineReader, MultiPageBlock) {
    // 4 docs, nnz=2 each -> 20-byte records -> block payload 4 + 4*20 = 84
    // > 32.
    auto vectors = create_float_vectors(
        {{0, 1}, {2, 3}, {4, 5}, {6, 7}},
        {{1.0F, 2.0F}, {3.0F, 4.0F}, {5.0F, 6.0F}, {7.0F, 8.0F}}, 10);
    auto lists = build_lists({{{0, 1, 2, 3}}}, vectors);
    nsparse::InlineForwardIndexWriter writer(32);
    TempInlineFile file("nsparse_inline_multipage.bin", lists, vectors, writer);

    nsparse::MmapInlineReader reader(file.path());
    EXPECT_EQ(reader.page_size(), 32U);
    verify_block(reader, reader.block(0, 0), {0, 1, 2, 3}, vectors);
}

TEST(MmapInlineReader, CorruptElementSizeRejected) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    nsparse::InlineForwardIndexWriter writer;
    TempInlineFile file("nsparse_inline_corrupt_esz.bin", lists, vectors,
                        writer);

    const uint32_t bad = 3;  // element_size is the first header field
    poke(file.path(), 0, &bad, sizeof(bad));
    EXPECT_THROW({ nsparse::MmapInlineReader r(file.path()); },
                 std::runtime_error);
}

TEST(MmapInlineReader, CorruptHugeNListsRejected) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    nsparse::InlineForwardIndexWriter writer;
    TempInlineFile file("nsparse_inline_corrupt_nlists.bin", lists, vectors,
                        writer);

    // n_lists is the last 8 bytes (trailer). UINT64_MAX would overflow the
    // list-index allocation without the reader's bound check.
    const size_t size = file_size(file.path());
    const uint64_t huge = UINT64_MAX;
    poke(file.path(), size - sizeof(uint64_t), &huge, sizeof(huge));
    EXPECT_THROW({ nsparse::MmapInlineReader r(file.path()); },
                 std::runtime_error);
}

TEST(MmapInlineReader, CorruptNEntriesRejected) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    nsparse::InlineForwardIndexWriter writer;
    TempInlineFile file("nsparse_inline_corrupt_nentries.bin", lists, vectors,
                        writer);

    // n_entries is the middle trailer field (bytes [size-16, size-8)).
    const size_t size = file_size(file.path());
    const uint64_t wrong = 999;
    poke(file.path(), size - 2 * sizeof(uint64_t), &wrong, sizeof(wrong));
    EXPECT_THROW({ nsparse::MmapInlineReader r(file.path()); },
                 std::runtime_error);
}

TEST(MmapInlineReader, CorruptRecordBoundsChecked) {
    auto vectors =
        create_float_vectors({{0, 3}, {1}}, {{1.0F, 2.0F}, {3.0F}}, 8);
    auto lists = build_lists({{{0, 1}}}, vectors);
    // Packed: block 0 starts right after the header, so record offsets are
    // known.
    nsparse::InlineForwardIndexWriter writer(
        nsparse::InlineForwardIndexWriter::kDefaultPageSize,
        nsparse::InlineLayout::kPacked);
    TempInlineFile file("nsparse_inline_corrupt_record.bin", lists, vectors,
                        writer);

    // Corrupt record 0's nnz: header + n_docs(4) + doc_id(4).
    const size_t nnz_off =
        sizeof(nsparse::InlineForwardIndexHeader) + 2 * sizeof(uint32_t);
    const uint32_t huge = 0xFFFFFFFFu;
    poke(file.path(), nnz_off, &huge, sizeof(huge));

    // The block envelope is still valid, so open succeeds...
    nsparse::MmapInlineReader reader(file.path());
    const nsparse::BlockView view = reader.block(0, 0);
    ASSERT_NE(view.data, nullptr);
    // ...but parsing the corrupt record must not read past the block.
    const uint8_t* cursor = view.records();
    const uint8_t* end = view.data + view.len;
    EXPECT_THROW(
        nsparse::read_inline_record(cursor, end, reader.element_size()),
        std::runtime_error);
}
