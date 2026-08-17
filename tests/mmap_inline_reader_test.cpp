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

#include <cstddef>
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

using nsparse::detail::BlockView;
using nsparse::detail::MmapInlineReader;

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

nsparse::InlineForwardIndexTrailer read_trailer(const char* path) {
    nsparse::InlineForwardIndexTrailer trailer{};
    const size_t size = file_size(path);
    std::ifstream file(path, std::ios::binary);
    file.seekg(static_cast<std::streamoff>(size - sizeof(trailer)));
    file.read(reinterpret_cast<char*>(&trailer), sizeof(trailer));
    return trailer;
}

// The file offset of a packed file's first block (right after the header,
// rounded up to the block alignment).
uint64_t packed_first_block_off() {
    return nsparse::inline_align_up(sizeof(nsparse::InlineForwardIndexHeader),
                                    nsparse::kMinBlockAlign);
}

// Assert a block's SoA view reproduces the source documents exactly, in order,
// and that its sub-arrays are aligned for in-place typed reads.
void verify_block(const MmapInlineReader& reader, const BlockView& view,
                  const std::vector<nsparse::idx_t>& expected_docs,
                  const nsparse::SparseVectors& vectors) {
    ASSERT_FALSE(view.absent());
    ASSERT_EQ(view.n_docs, expected_docs.size());
    EXPECT_EQ(view.offsets[0], 0U);

    const size_t element_size = reader.element_size();
    const auto* indptr = vectors.indptr_data();
    const auto* indices = vectors.indices_data();
    const auto* values = vectors.values_data();

    for (uint32_t i = 0; i < view.n_docs; ++i) {
        EXPECT_EQ(view.doc_ids[i], static_cast<uint32_t>(expected_docs[i]));
        const nsparse::idx_t start = indptr[view.doc_ids[i]];
        const uint32_t src_nnz =
            static_cast<uint32_t>(indptr[view.doc_ids[i] + 1] - start);
        ASSERT_EQ(view.nnz(i), src_nnz);
        EXPECT_EQ(0, std::memcmp(view.doc_comps(i), indices + start,
                                 src_nnz * sizeof(nsparse::term_t)));
        EXPECT_EQ(
            0, std::memcmp(view.doc_vals(i, element_size),
                           values + static_cast<size_t>(start) * element_size,
                           static_cast<size_t>(src_nnz) * element_size));
    }
    // Every sub-array is aligned for its element type, so a consumer can read
    // it in place without an unaligned load.
    EXPECT_EQ(reinterpret_cast<uintptr_t>(view.doc_ids) % alignof(uint32_t),
              0U);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(view.offsets) % alignof(uint32_t),
              0U);
    EXPECT_EQ(
        reinterpret_cast<uintptr_t>(view.comps) % alignof(nsparse::term_t), 0U);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(view.vals) % element_size, 0U);
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

    MmapInlineReader reader(file.path());
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
}

TEST(MmapInlineReader, RoundTripPacked) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    nsparse::InlineForwardIndexWriter writer(
        nsparse::InlineForwardIndexWriter::kDefaultPageSize,
        nsparse::InlineLayout::kPacked);
    TempInlineFile file("nsparse_inline_packed.bin", lists, vectors, writer);

    MmapInlineReader reader(file.path());
    EXPECT_EQ(reader.page_size(), nsparse::kMinBlockAlign);
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

    MmapInlineReader reader(file.path());
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

    MmapInlineReader reader(file.path());
    EXPECT_EQ(reader.element_size(), nsparse::U8);
    verify_block(reader, reader.block(0, 0), {0, 1}, vectors);
    verify_block(reader, reader.block(0, 1), {2}, vectors);
}

TEST(MmapInlineReader, ValuesReadableAsTypedFloats) {
    // The float values in a block borrow directly as an aligned const float*.
    auto vectors = create_float_vectors(
        {{0, 3}, {1}, {2, 4}}, {{1.5F, 2.5F}, {3.5F}, {4.5F, 5.5F}}, 8);
    auto lists = build_lists({{{0, 1, 2}}}, vectors);
    nsparse::InlineForwardIndexWriter writer;
    TempInlineFile file("nsparse_inline_typed.bin", lists, vectors, writer);

    MmapInlineReader reader(file.path());
    const BlockView view = reader.block(0, 0);
    ASSERT_FALSE(view.absent());
    ASSERT_EQ(view.n_docs, 3U);

    const auto* indptr = vectors.indptr_data();
    const auto* values = reinterpret_cast<const float*>(vectors.values_data());
    for (uint32_t i = 0; i < view.n_docs; ++i) {
        const auto* doc_vals =
            reinterpret_cast<const float*>(view.doc_vals(i, nsparse::U32));
        const nsparse::idx_t start = indptr[view.doc_ids[i]];
        for (uint32_t j = 0; j < view.nnz(i); ++j) {
            EXPECT_FLOAT_EQ(doc_vals[j], values[start + j]);
        }
    }
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

    MmapInlineReader reader(file.path());
    EXPECT_EQ(reader.num_lists(), 3U);
    EXPECT_EQ(reader.num_blocks(), 3U);  // 2 (list0) + 0 (list1) + 1 (list2)
    EXPECT_EQ(reader.num_blocks_in_list(0), 2U);
    EXPECT_EQ(reader.num_blocks_in_list(1), 0U);
    EXPECT_EQ(reader.num_blocks_in_list(2), 1U);

    verify_block(reader, reader.block(0, 0), {0, 1},
                 vectors);  // incl. zero-nnz doc

    // The empty block is present (not absent) but has zero documents.
    const BlockView empty = reader.block(0, 1);
    EXPECT_FALSE(empty.absent());
    EXPECT_EQ(empty.n_docs, 0U);

    verify_block(reader, reader.block(2, 0), {2}, vectors);

    // Absent lookups return an absent view (doc_ids == nullptr).
    EXPECT_TRUE(reader.block(1, 0).absent());  // empty posting list
    EXPECT_TRUE(reader.block(0, 2).absent());  // block index out of range
    EXPECT_TRUE(reader.block(9, 0).absent());  // posting list out of range
}

TEST(MmapInlineReader, MissingFileThrows) {
    const auto missing =
        std::filesystem::temp_directory_path() / "nsparse_inline_absent.bin";
    std::error_code ignored;
    std::filesystem::remove(missing, ignored);
    const std::string path = missing.string();
    EXPECT_THROW({ MmapInlineReader reader(path.c_str()); },
                 std::runtime_error);
}

TEST(MmapInlineReader, NullPathThrows) {
    // A null path is a programming error, distinct from a bad file.
    EXPECT_THROW({ MmapInlineReader reader(nullptr); }, std::invalid_argument);
}

TEST(MmapInlineReader, TruncatedFileThrows) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    nsparse::InlineForwardIndexWriter writer;
    TempInlineFile file("nsparse_inline_truncated.bin", lists, vectors, writer);

    // Truncate below the minimum header + trailer size.
    std::filesystem::resize_file(file.path(), 8);
    EXPECT_THROW({ MmapInlineReader reader(file.path()); }, std::runtime_error);
}

TEST(MmapInlineReader, MoveSemantics) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    nsparse::InlineForwardIndexWriter writer;
    TempInlineFile file("nsparse_inline_move_a.bin", lists, vectors, writer);
    TempInlineFile file2("nsparse_inline_move_b.bin", lists, vectors, writer);

    // A moved-from reader is fully inert. Buf's move keeps the source's
    // size()/data(), so this pins the explicit member reset that compensates;
    // without it block() would hand back a wild pointer (use-after-move).
    auto expect_inert = [](const MmapInlineReader& r) {
        EXPECT_EQ(r.num_blocks(), 0U);
        EXPECT_EQ(r.num_lists(), 0U);
        EXPECT_EQ(r.element_size(), 0U);
        EXPECT_EQ(r.page_size(), 0U);
        EXPECT_EQ(r.num_blocks_in_list(0), 0U);
        EXPECT_EQ(r.num_blocks_in_list(0xFFFFFFFFu), 0U);  // uint32 wrap guard
        EXPECT_TRUE(r.block(0, 0).absent());
    };

    MmapInlineReader reader(file.path());
    MmapInlineReader moved(std::move(reader));
    verify_block(moved, moved.block(0, 0), kLayout[0][0], vectors);
    expect_inert(reader);  // inert after move-construction

    MmapInlineReader reader2(file2.path());
    reader2 = std::move(moved);
    verify_block(reader2, reader2.block(0, 0), kLayout[0][0], vectors);
    expect_inert(moved);  // inert after move-assignment (pins the op= reset)

    // Self-move-assignment is a safe no-op (aliased to dodge -Wself-move).
    MmapInlineReader& alias = reader2;
    reader2 = std::move(alias);
    verify_block(reader2, reader2.block(0, 0), kLayout[0][0], vectors);
}

TEST(MmapInlineReader, MultiPageBlock) {
    // 4 docs, nnz=2 each -> a block payload larger than a 32-byte page.
    auto vectors = create_float_vectors(
        {{0, 1}, {2, 3}, {4, 5}, {6, 7}},
        {{1.0F, 2.0F}, {3.0F, 4.0F}, {5.0F, 6.0F}, {7.0F, 8.0F}}, 10);
    auto lists = build_lists({{{0, 1, 2, 3}}}, vectors);
    nsparse::InlineForwardIndexWriter writer(32);
    TempInlineFile file("nsparse_inline_multipage.bin", lists, vectors, writer);

    MmapInlineReader reader(file.path());
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
    EXPECT_THROW({ MmapInlineReader r(file.path()); }, std::runtime_error);
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
    EXPECT_THROW({ MmapInlineReader r(file.path()); }, std::runtime_error);
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
    EXPECT_THROW({ MmapInlineReader r(file.path()); }, std::runtime_error);
}

TEST(MmapInlineReader, CorruptMisalignedBlockRejected) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    nsparse::InlineForwardIndexWriter writer(
        nsparse::InlineForwardIndexWriter::kDefaultPageSize,
        nsparse::InlineLayout::kPacked);
    TempInlineFile file("nsparse_inline_corrupt_align.bin", lists, vectors,
                        writer);

    // Nudge the first directory entry's byte_off off the alignment grid.
    const uint64_t dir_offset = read_trailer(file.path()).dir_offset;
    const size_t byte_off_field =
        dir_offset + offsetof(nsparse::InlineDirEntry, byte_off);
    const uint64_t misaligned = packed_first_block_off() + 1;
    poke(file.path(), byte_off_field, &misaligned, sizeof(misaligned));
    EXPECT_THROW({ MmapInlineReader r(file.path()); }, std::runtime_error);
}

// The corrupt-block tests below build a single packed block of two docs
// (doc0 nnz=2, doc1 nnz=1) so the block starts at a known offset and its
// [n_docs][doc_id[]][off[]] header offsets are deterministic.
namespace {
nsparse::SparseVectors two_doc_block_vectors() {
    return create_float_vectors({{0, 3}, {1}}, {{1.0F, 2.0F}, {3.0F}}, 8);
}
std::vector<nsparse::InvertedListClusters> two_doc_block_lists(
    const nsparse::SparseVectors& vectors) {
    return build_lists({{{0, 1}}}, vectors);
}
// Byte offset in the file of off[i] for the first packed block (2 docs).
uint64_t off_field_pos(uint32_t i) {
    return packed_first_block_off() +
           nsparse::inline_block_offsets(/*n_docs=*/2, /*total_nnz=*/0,
                                         nsparse::U32)
               .off +
           i * sizeof(uint32_t);
}
}  // namespace

TEST(MmapInlineReader, CorruptBlockNDocsMismatchRejected) {
    auto vectors = two_doc_block_vectors();
    auto lists = two_doc_block_lists(vectors);
    nsparse::InlineForwardIndexWriter writer(
        nsparse::InlineForwardIndexWriter::kDefaultPageSize,
        nsparse::InlineLayout::kPacked);
    TempInlineFile file("nsparse_inline_corrupt_ndocs.bin", lists, vectors,
                        writer);

    // The block's own n_docs field (first u32 of the block) disagrees with the
    // directory entry.
    const uint32_t wrong = 7;
    poke(file.path(), packed_first_block_off(), &wrong, sizeof(wrong));
    MmapInlineReader reader(file.path());  // envelope still valid
    EXPECT_THROW(reader.block(0, 0), std::runtime_error);
}

TEST(MmapInlineReader, CorruptBlockOffsetsStartNonZeroRejected) {
    auto vectors = two_doc_block_vectors();
    auto lists = two_doc_block_lists(vectors);
    nsparse::InlineForwardIndexWriter writer(
        nsparse::InlineForwardIndexWriter::kDefaultPageSize,
        nsparse::InlineLayout::kPacked);
    TempInlineFile file("nsparse_inline_corrupt_off0.bin", lists, vectors,
                        writer);

    const uint32_t nonzero = 5;
    poke(file.path(), off_field_pos(0), &nonzero, sizeof(nonzero));
    MmapInlineReader reader(file.path());
    EXPECT_THROW(reader.block(0, 0), std::runtime_error);
}

TEST(MmapInlineReader, CorruptBlockOffsetsNonMonotonicRejected) {
    auto vectors = two_doc_block_vectors();
    auto lists = two_doc_block_lists(vectors);
    nsparse::InlineForwardIndexWriter writer(
        nsparse::InlineForwardIndexWriter::kDefaultPageSize,
        nsparse::InlineLayout::kPacked);
    TempInlineFile file("nsparse_inline_corrupt_mono.bin", lists, vectors,
                        writer);

    // off[1] huge -> off[2] (total_nnz) is smaller, breaking monotonicity.
    const uint32_t huge = 0xFFFFFFFFu;
    poke(file.path(), off_field_pos(1), &huge, sizeof(huge));
    MmapInlineReader reader(file.path());
    EXPECT_THROW(reader.block(0, 0), std::runtime_error);
}

TEST(MmapInlineReader, CorruptBlockLengthMismatchRejected) {
    auto vectors = two_doc_block_vectors();
    auto lists = two_doc_block_lists(vectors);
    nsparse::InlineForwardIndexWriter writer(
        nsparse::InlineForwardIndexWriter::kDefaultPageSize,
        nsparse::InlineLayout::kPacked);
    TempInlineFile file("nsparse_inline_corrupt_len.bin", lists, vectors,
                        writer);

    // off[2] (total_nnz) inflated -> the computed payload length no longer
    // matches the directory's recorded len.
    const uint32_t inflated = 1000;
    poke(file.path(), off_field_pos(2), &inflated, sizeof(inflated));
    MmapInlineReader reader(file.path());
    EXPECT_THROW(reader.block(0, 0), std::runtime_error);
}

TEST(MmapInlineReader, CorruptDirectoryOrderRejected) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);  // entries (0,0)(0,1)(1,0)(1,1)
    nsparse::InlineForwardIndexWriter writer;
    TempInlineFile file("nsparse_inline_corrupt_order.bin", lists, vectors,
                        writer);

    // entry[1] should be (pl0, block1); give it a gap block index so the
    // directory is no longer gap-free and load rejects it.
    const uint64_t dir_offset = read_trailer(file.path()).dir_offset;
    const size_t block_field = dir_offset + sizeof(nsparse::InlineDirEntry) +
                               offsetof(nsparse::InlineDirEntry, block);
    const uint32_t bad_block = 5;
    poke(file.path(), block_field, &bad_block, sizeof(bad_block));
    EXPECT_THROW({ MmapInlineReader r(file.path()); }, std::runtime_error);
}
