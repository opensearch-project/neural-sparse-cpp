/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/inline_forward_index.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/io/buffered_io.h"
#include "nsparse/io/inline_forward_index_io.h"
#include "nsparse/io/io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/mmap_cursor.h"

namespace {

using nsparse::detail::BlockView;
using nsparse::detail::inline_align_up;
using nsparse::detail::inline_block_offsets;
using nsparse::detail::InlineDirEntry;
using nsparse::detail::InlineForwardIndex;
using nsparse::detail::InlineForwardIndexHeader;
using nsparse::detail::InlineForwardIndexTrailer;
using nsparse::detail::InlineLayout;
using nsparse::detail::kMinBlockAlign;

constexpr float kAlpha = 0.5F;

// serialize() prefixes the section with a u64 byte length, so the format body
// (header, blocks, directory, trailer) starts kLenPrefix into the buffer.
constexpr size_t kLenPrefix = sizeof(uint64_t);

template <class T>
T read_pod(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

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

// Serialize a write-mode index to a byte buffer (the on-disk form).
std::vector<uint8_t> serialize_bytes(
    const std::vector<nsparse::InvertedListClusters>& lists,
    const nsparse::SparseVectors& vectors,
    uint64_t page = InlineForwardIndex::kDefaultPageSize,
    InlineLayout layout = InlineLayout::kPageAligned) {
    InlineForwardIndex writer(lists, vectors, page, layout);
    nsparse::BufferedIOWriter buffer;
    writer.serialize(&buffer);
    return buffer.data();
}

// Read (borrow) an index back from bytes. The buffer (operator-new storage) is
// >= 8-aligned, so the block sub-arrays line up. `bytes` must outlive the
// index.
InlineForwardIndex read_index(const std::vector<uint8_t>& bytes) {
    nsparse::MmapCursor cursor(bytes.data(), bytes.size());
    InlineForwardIndex index;
    index.mmap_deserialize(&cursor);
    return index;
}

InlineForwardIndexHeader parse_header(const std::vector<uint8_t>& bin) {
    return read_pod<InlineForwardIndexHeader>(bin.data() + kLenPrefix);
}

std::vector<InlineDirEntry> parse_dir(const std::vector<uint8_t>& bin,
                                      InlineForwardIndexTrailer& trailer_out) {
    const size_t tsz = sizeof(InlineForwardIndexTrailer);
    // The trailer is the last bytes of the body, i.e. of the whole buffer.
    trailer_out =
        read_pod<InlineForwardIndexTrailer>(bin.data() + bin.size() - tsz);
    std::vector<InlineDirEntry> entries(trailer_out.n_entries);
    for (uint64_t i = 0; i < trailer_out.n_entries; ++i) {
        entries[i] = read_pod<InlineDirEntry>(bin.data() + kLenPrefix +
                                              trailer_out.dir_offset +
                                              i * sizeof(InlineDirEntry));
    }
    return entries;
}

// File offset of a body-relative offset within the serialized buffer.
size_t body_off(uint64_t body_relative) { return kLenPrefix + body_relative; }

// Assert a block view reproduces the source documents exactly, in order, and
// that its sub-arrays are aligned for in-place typed reads.
void verify_block(const InlineForwardIndex& index, const BlockView& view,
                  const std::vector<nsparse::idx_t>& expected_docs,
                  const nsparse::SparseVectors& vectors) {
    ASSERT_FALSE(view.absent());
    ASSERT_EQ(view.n_docs, expected_docs.size());
    EXPECT_EQ(view.offsets[0], 0U);

    const size_t element_size = index.element_size();
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
    EXPECT_EQ(reinterpret_cast<uintptr_t>(view.doc_ids) % alignof(uint32_t),
              0U);
    EXPECT_EQ(
        reinterpret_cast<uintptr_t>(view.comps) % alignof(nsparse::term_t), 0U);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(view.vals) % element_size, 0U);
}

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

// The file offset of the first packed block (right after the header, rounded
// up to the block alignment).
uint64_t packed_first_block_off() {
    return inline_align_up(sizeof(InlineForwardIndexHeader), kMinBlockAlign);
}

}  // namespace

// ==================== serialize() (on-disk format) ====================

TEST(InlineForwardIndex, SerializesHeaderTrailerAndDirectory) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    const auto bin = serialize_bytes(lists, vectors);

    const auto page = InlineForwardIndex::kDefaultPageSize;
    const auto header = parse_header(bin);
    EXPECT_EQ(header.element_size, nsparse::U32);
    EXPECT_EQ(header.n_blocks, 4U);
    EXPECT_EQ(header.page_size, page);

    InlineForwardIndexTrailer trailer{};
    auto entries = parse_dir(bin, trailer);
    EXPECT_EQ(trailer.n_lists, 2U);
    ASSERT_EQ(entries.size(), 4U);
    const std::vector<std::pair<uint32_t, uint32_t>> ids = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}};
    for (size_t i = 0; i < entries.size(); ++i) {
        EXPECT_EQ(entries[i].pl, ids[i].first);
        EXPECT_EQ(entries[i].block, ids[i].second);
        EXPECT_EQ(entries[i].byte_off % page, 0U);
    }
    EXPECT_EQ(entries[0].byte_off, page);
}

TEST(InlineForwardIndex, PackedLayoutMinimalAlignment) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    const auto packed =
        serialize_bytes(lists, vectors, InlineForwardIndex::kDefaultPageSize,
                        InlineLayout::kPacked);
    const auto padded = serialize_bytes(lists, vectors);  // page-aligned

    EXPECT_EQ(parse_header(packed).page_size, kMinBlockAlign);
    InlineForwardIndexTrailer trailer{};
    auto entries = parse_dir(packed, trailer);
    uint64_t running = sizeof(InlineForwardIndexHeader);
    for (const auto& entry : entries) {
        running = inline_align_up(running, kMinBlockAlign);
        EXPECT_EQ(entry.byte_off, running);
        running += entry.len;
    }
    EXPECT_LT(packed.size(), padded.size());  // strictly smaller than paged
}

TEST(InlineForwardIndex, SerializeRejectsUnsupportedElementSize) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U64, .dimension = 4});
    std::vector<nsparse::InvertedListClusters> lists;
    InlineForwardIndex writer(lists, vectors);
    nsparse::BufferedIOWriter buffer;
    EXPECT_THROW(writer.serialize(&buffer), std::invalid_argument);
}

TEST(InlineForwardIndex, SerializeRejectsOutOfRangeDocId) {
    auto vectors_full =
        create_float_vectors({{0}, {1}, {2}}, {{1.0F}, {2.0F}, {3.0F}}, 4);
    auto lists = build_lists({{{0, 1, 2}}}, vectors_full);
    auto vectors_small = create_float_vectors({{0}, {1}}, {{1.0F}, {2.0F}}, 4);
    InlineForwardIndex writer(lists, vectors_small);
    nsparse::BufferedIOWriter buffer;
    EXPECT_THROW(writer.serialize(&buffer), std::out_of_range);
}

// page_size only constrains the page-aligned layout, which has to be asked for:
// packed is the default, and there the alignment is kMinBlockAlign and page_size
// is ignored.
TEST(InlineForwardIndex, RejectsInvalidPageSize) {
    auto vectors = sample_float_vectors();
    EXPECT_THROW(InlineForwardIndex(build_lists(kLayout, vectors), vectors, 0,
                                    InlineLayout::kPageAligned),
                 std::invalid_argument);
    EXPECT_THROW(InlineForwardIndex(build_lists(kLayout, vectors), vectors, 100,
                                    InlineLayout::kPageAligned),
                 std::invalid_argument);
    EXPECT_NO_THROW(InlineForwardIndex(build_lists(kLayout, vectors), vectors,
                                       64, InlineLayout::kPageAligned));
    EXPECT_NO_THROW(
        InlineForwardIndex(build_lists(kLayout, vectors), vectors, 100));
}

// The two layouts are one format: the header records the alignment and a reader
// honours whatever the file declares, so a page-aligned file written before the
// default changed still loads, and loads to the same documents.
TEST(InlineForwardIndex, BothLayoutsReadBackTheSameBlocks) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    const auto padded =
        serialize_bytes(lists, vectors, InlineForwardIndex::kDefaultPageSize,
                        InlineLayout::kPageAligned);
    const auto packed =
        serialize_bytes(lists, vectors, InlineForwardIndex::kDefaultPageSize,
                        InlineLayout::kPacked);
    auto padded_index = read_index(padded);
    auto packed_index = read_index(packed);

    ASSERT_EQ(padded_index.num_blocks(), packed_index.num_blocks());
    ASSERT_EQ(padded_index.num_lists(), packed_index.num_lists());
    for (uint32_t pl = 0; pl < padded_index.num_lists(); ++pl) {
        const uint64_t blocks = padded_index.num_blocks_in_list(pl);
        ASSERT_EQ(blocks, packed_index.num_blocks_in_list(pl));
        for (uint32_t block = 0; block < blocks; ++block) {
            verify_block(packed_index, packed_index.block(pl, block),
                         kLayout[pl][block], vectors);
            verify_block(padded_index, padded_index.block(pl, block),
                         kLayout[pl][block], vectors);
        }
    }
}

// What a disk index gets when it does not ask: blocks packed on kMinBlockAlign,
// not padded to a page. This is the whole of the padding change, so it is the one
// thing a future edit to the default must trip over.
TEST(InlineForwardIndex, DefaultsToThePackedLayout) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    InlineForwardIndex writer(lists, vectors);
    nsparse::BufferedIOWriter buffer;
    writer.serialize(&buffer);
    const auto bin = buffer.data();

    EXPECT_EQ(parse_header(bin).page_size, kMinBlockAlign);
    // And it is smaller than the same lists written page-aligned -- padding is
    // all that differs.
    EXPECT_LT(bin.size(), serialize_bytes(lists, vectors,
                                          InlineForwardIndex::kDefaultPageSize,
                                          InlineLayout::kPageAligned)
                              .size());
}

TEST(InlineForwardIndex, SerializeOnReadModeIndexThrows) {
    InlineForwardIndex reader;  // default (read-mode) index
    nsparse::BufferedIOWriter buffer;
    EXPECT_THROW(reader.serialize(&buffer), std::logic_error);
}

TEST(InlineForwardIndex, DeserializeUnsupported) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    const auto bin = serialize_bytes(lists, vectors);
    nsparse::BufferedIOReader reader(bin.data(), bin.size());
    InlineForwardIndex index;
    EXPECT_THROW(index.deserialize(&reader), std::runtime_error);
}

// ==================== mmap_deserialize() + block() ====================

TEST(InlineForwardIndex, RoundTripPageAligned) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    const auto bin = serialize_bytes(lists, vectors);
    auto index = read_index(bin);

    EXPECT_EQ(index.element_size(), nsparse::U32);
    EXPECT_EQ(index.num_blocks(), 4U);
    EXPECT_EQ(index.num_lists(), 2U);
    EXPECT_EQ(index.page_size(), InlineForwardIndex::kDefaultPageSize);
    EXPECT_EQ(index.num_blocks_in_list(0), 2U);
    EXPECT_EQ(index.num_blocks_in_list(1), 2U);

    verify_block(index, index.block(0, 0), kLayout[0][0], vectors);
    verify_block(index, index.block(0, 1), kLayout[0][1], vectors);
    verify_block(index, index.block(1, 0), kLayout[1][0], vectors);
    verify_block(index, index.block(1, 1), kLayout[1][1], vectors);
}

TEST(InlineForwardIndex, RoundTripPacked) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    const auto bin =
        serialize_bytes(lists, vectors, InlineForwardIndex::kDefaultPageSize,
                        InlineLayout::kPacked);
    auto index = read_index(bin);
    EXPECT_EQ(index.page_size(), kMinBlockAlign);
    verify_block(index, index.block(0, 0), kLayout[0][0], vectors);
    verify_block(index, index.block(1, 1), kLayout[1][1], vectors);
}

TEST(InlineForwardIndex, ElementSizeU16) {
    auto vectors = create_uint16_vectors({{0, 3}, {1}, {2, 4, 6}},
                                         {{10, 20}, {30}, {15, 25, 35}}, 8);
    auto lists = build_lists({{{0, 1}, {2}}}, vectors);
    const auto bin = serialize_bytes(lists, vectors);
    auto index = read_index(bin);
    EXPECT_EQ(index.element_size(), nsparse::U16);
    verify_block(index, index.block(0, 0), {0, 1}, vectors);
    verify_block(index, index.block(0, 1), {2}, vectors);
}

TEST(InlineForwardIndex, ElementSizeU8) {
    auto vectors = create_uint8_vectors({{0, 3}, {1}, {2, 4, 6}},
                                        {{10, 20}, {30}, {15, 25, 35}}, 8);
    auto lists = build_lists({{{0, 1}, {2}}}, vectors);
    const auto bin = serialize_bytes(lists, vectors);
    auto index = read_index(bin);
    EXPECT_EQ(index.element_size(), nsparse::U8);
    verify_block(index, index.block(0, 0), {0, 1}, vectors);
    verify_block(index, index.block(0, 1), {2}, vectors);
}

TEST(InlineForwardIndex, ValuesReadableAsTypedFloats) {
    auto vectors = create_float_vectors(
        {{0, 3}, {1}, {2, 4}}, {{1.5F, 2.5F}, {3.5F}, {4.5F, 5.5F}}, 8);
    auto lists = build_lists({{{0, 1, 2}}}, vectors);
    const auto bin = serialize_bytes(lists, vectors);
    auto index = read_index(bin);
    const BlockView view = index.block(0, 0);
    ASSERT_FALSE(view.absent());
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

TEST(InlineForwardIndex, SparseListsEmptyBlocksAndAbsentLookups) {
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

    const auto bin = serialize_bytes(lists, vectors);
    auto index = read_index(bin);
    EXPECT_EQ(index.num_lists(), 3U);
    EXPECT_EQ(index.num_blocks(), 3U);
    EXPECT_EQ(index.num_blocks_in_list(1), 0U);

    verify_block(index, index.block(0, 0), {0, 1}, vectors);
    const BlockView empty = index.block(0, 1);
    EXPECT_FALSE(empty.absent());
    EXPECT_EQ(empty.n_docs, 0U);
    verify_block(index, index.block(2, 0), {2}, vectors);

    EXPECT_TRUE(index.block(1, 0).absent());  // empty posting list
    EXPECT_TRUE(index.block(0, 2).absent());  // block index out of range
    EXPECT_TRUE(index.block(9, 0).absent());  // posting list out of range
}

// Composes at an unaligned stream offset. A 4-byte leading component leaves
// serialize() at pos 4, so its pad_to(kMinBlockAlign) has real work to do -- a
// missing pad would put the section base off a boundary and make the block
// sub-arrays unreadable. Everything goes through one writer (the real
// composition path, not a manual splice), and reading back through one cursor
// the reader skips that padding and lands exactly on the trailing marker, with
// the leading marker untouched.
TEST(InlineForwardIndex, ComposesAtAnUnalignedStreamOffset) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);

    uint32_t lead_marker = 0x11223344;
    uint32_t suffix_marker = 0xABCDEF01;
    nsparse::BufferedIOWriter writer;
    writer.write(&lead_marker, sizeof(uint32_t), 1);
    ASSERT_EQ(writer.pos() % kMinBlockAlign, 4U);  // section starts unaligned
    InlineForwardIndex(lists, vectors).serialize(&writer);
    writer.write(&suffix_marker, sizeof(uint32_t), 1);
    const auto stream = writer.data();

    nsparse::MmapCursor cursor(stream.data(), stream.size());
    EXPECT_EQ(cursor.read_scalar<uint32_t>(), lead_marker);
    InlineForwardIndex index;
    index.mmap_deserialize(&cursor);  // skips pad, reads length, self-advances

    verify_block(index, index.block(0, 0), kLayout[0][0], vectors);
    EXPECT_EQ(cursor.read_scalar<uint32_t>(), suffix_marker);  // left just past
    EXPECT_EQ(cursor.remaining(), 0U);
}

TEST(InlineForwardIndex, MoveSemantics) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    const auto bin_a = serialize_bytes(lists, vectors);
    const auto bin_b = serialize_bytes(lists, vectors);

    auto expect_inert = [](const InlineForwardIndex& idx) {
        EXPECT_EQ(idx.num_blocks(), 0U);
        EXPECT_EQ(idx.num_lists(), 0U);
        EXPECT_EQ(idx.element_size(), 0U);
        EXPECT_EQ(idx.num_blocks_in_list(0), 0U);
        EXPECT_EQ(idx.num_blocks_in_list(0xFFFFFFFFu), 0U);
        EXPECT_TRUE(idx.block(0, 0).absent());
    };

    nsparse::MmapCursor ca(bin_a.data(), bin_a.size());
    InlineForwardIndex a;
    a.mmap_deserialize(&ca);
    InlineForwardIndex moved(std::move(a));
    verify_block(moved, moved.block(0, 0), kLayout[0][0], vectors);
    expect_inert(a);

    nsparse::MmapCursor cb(bin_b.data(), bin_b.size());
    InlineForwardIndex b;
    b.mmap_deserialize(&cb);
    b = std::move(moved);
    verify_block(b, b.block(0, 0), kLayout[0][0], vectors);
    expect_inert(moved);

    InlineForwardIndex& alias = b;
    b = std::move(alias);  // self-move is a safe no-op
    verify_block(b, b.block(0, 0), kLayout[0][0], vectors);
}

// ==================== corrupt-input rejection ====================

TEST(InlineForwardIndex, TruncatedSectionThrows) {
    auto vectors = sample_float_vectors();
    auto bin = serialize_bytes(build_lists(kLayout, vectors), vectors);
    nsparse::MmapCursor cursor(bin.data(), 8);  // far below header + trailer
    InlineForwardIndex index;
    EXPECT_THROW(index.mmap_deserialize(&cursor), std::runtime_error);
}

TEST(InlineForwardIndex, CorruptElementSizeRejected) {
    auto vectors = sample_float_vectors();
    auto bin = serialize_bytes(build_lists(kLayout, vectors), vectors);
    const uint32_t bad = 3;
    std::memcpy(bin.data() + kLenPrefix, &bad, sizeof(bad));  // element_size
    nsparse::MmapCursor cursor(bin.data(), bin.size());
    InlineForwardIndex index;
    EXPECT_THROW(index.mmap_deserialize(&cursor), std::runtime_error);
}

TEST(InlineForwardIndex, CorruptHugeNListsRejected) {
    auto vectors = sample_float_vectors();
    auto bin = serialize_bytes(build_lists(kLayout, vectors), vectors);
    const uint64_t huge = UINT64_MAX;  // n_lists = last 8 bytes (trailer)
    std::memcpy(bin.data() + bin.size() - sizeof(uint64_t), &huge,
                sizeof(huge));
    nsparse::MmapCursor cursor(bin.data(), bin.size());
    InlineForwardIndex index;
    EXPECT_THROW(index.mmap_deserialize(&cursor), std::runtime_error);
}

TEST(InlineForwardIndex, CorruptMisalignedBlockRejected) {
    auto vectors = sample_float_vectors();
    auto bin = serialize_bytes(build_lists(kLayout, vectors), vectors,
                               InlineForwardIndex::kDefaultPageSize,
                               InlineLayout::kPacked);
    InlineForwardIndexTrailer trailer{};
    parse_dir(bin, trailer);
    const size_t field =
        body_off(trailer.dir_offset + offsetof(InlineDirEntry, byte_off));
    const uint64_t misaligned = packed_first_block_off() + 1;
    std::memcpy(bin.data() + field, &misaligned, sizeof(misaligned));
    nsparse::MmapCursor cursor(bin.data(), bin.size());
    InlineForwardIndex index;
    EXPECT_THROW(index.mmap_deserialize(&cursor), std::runtime_error);
}

TEST(InlineForwardIndex, CorruptDirectoryOrderRejected) {
    auto vectors = sample_float_vectors();
    auto bin = serialize_bytes(build_lists(kLayout, vectors), vectors);
    InlineForwardIndexTrailer trailer{};
    parse_dir(bin, trailer);
    // entry[1] should be (pl0, block1); give it a gap block index.
    const size_t field = body_off(trailer.dir_offset + sizeof(InlineDirEntry) +
                                  offsetof(InlineDirEntry, block));
    const uint32_t bad_block = 5;
    std::memcpy(bin.data() + field, &bad_block, sizeof(bad_block));
    nsparse::MmapCursor cursor(bin.data(), bin.size());
    InlineForwardIndex index;
    EXPECT_THROW(index.mmap_deserialize(&cursor), std::runtime_error);
}

// The corrupt-block tests use a single packed block of two docs (doc0 nnz=2,
// doc1 nnz=1) so its [n_docs][doc_id[]][off[]] header offsets are
// deterministic.
namespace {
std::vector<uint8_t> two_doc_packed_bytes(const nsparse::SparseVectors& v) {
    return serialize_bytes(build_lists({{{0, 1}}}, v), v,
                           InlineForwardIndex::kDefaultPageSize,
                           InlineLayout::kPacked);
}
size_t off_field_pos(uint32_t i) {
    return body_off(packed_first_block_off() +
                    inline_block_offsets(2, 0, nsparse::U32).off +
                    i * sizeof(uint32_t));
}
}  // namespace

TEST(InlineForwardIndex, CorruptBlockNDocsMismatchRejected) {
    auto vectors =
        create_float_vectors({{0, 3}, {1}}, {{1.0F, 2.0F}, {3.0F}}, 8);
    auto bin = two_doc_packed_bytes(vectors);
    const uint32_t wrong = 7;
    std::memcpy(bin.data() + body_off(packed_first_block_off()), &wrong,
                sizeof(wrong));
    auto index = read_index(bin);  // envelope still valid
    EXPECT_THROW(index.block(0, 0), std::runtime_error);
}

TEST(InlineForwardIndex, CorruptBlockOffsetsStartNonZeroRejected) {
    auto vectors =
        create_float_vectors({{0, 3}, {1}}, {{1.0F, 2.0F}, {3.0F}}, 8);
    auto bin = two_doc_packed_bytes(vectors);
    const uint32_t nonzero = 5;
    std::memcpy(bin.data() + off_field_pos(0), &nonzero, sizeof(nonzero));
    auto index = read_index(bin);
    EXPECT_THROW(index.block(0, 0), std::runtime_error);
}

TEST(InlineForwardIndex, CorruptBlockOffsetsNonMonotonicRejected) {
    auto vectors =
        create_float_vectors({{0, 3}, {1}}, {{1.0F, 2.0F}, {3.0F}}, 8);
    auto bin = two_doc_packed_bytes(vectors);
    const uint32_t huge = 0xFFFFFFFFu;  // off[1] huge -> off[2] smaller
    std::memcpy(bin.data() + off_field_pos(1), &huge, sizeof(huge));
    auto index = read_index(bin);
    EXPECT_THROW(index.block(0, 0), std::runtime_error);
}

TEST(InlineForwardIndex, CorruptBlockLengthMismatchRejected) {
    auto vectors =
        create_float_vectors({{0, 3}, {1}}, {{1.0F, 2.0F}, {3.0F}}, 8);
    auto bin = two_doc_packed_bytes(vectors);
    const uint32_t inflated = 1000;  // total_nnz -> layout.end != len
    std::memcpy(bin.data() + off_field_pos(2), &inflated, sizeof(inflated));
    auto index = read_index(bin);
    EXPECT_THROW(index.block(0, 0), std::runtime_error);
}

// ==================== edge-case round-trips ====================

TEST(InlineForwardIndex, EmptyRoundTrip) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 4});
    std::vector<nsparse::InvertedListClusters> lists;  // no lists, no blocks
    for (auto layout : {InlineLayout::kPageAligned, InlineLayout::kPacked}) {
        const auto bin = serialize_bytes(
            lists, vectors, InlineForwardIndex::kDefaultPageSize, layout);
        auto index = read_index(bin);
        EXPECT_EQ(index.num_lists(), 0U);
        EXPECT_EQ(index.num_blocks(), 0U);
        EXPECT_TRUE(index.block(0, 0).absent());
    }
}

TEST(InlineForwardIndex, CustomPageSizeRoundTrip) {
    // 4 docs, nnz=2 each -> a block payload larger than the 64-byte page.
    auto vectors = create_float_vectors(
        {{0, 1}, {2, 3}, {4, 5}, {6, 7}},
        {{1.0F, 2.0F}, {3.0F, 4.0F}, {5.0F, 6.0F}, {7.0F, 8.0F}}, 10);
    auto lists = build_lists({{{0, 1, 2, 3}}}, vectors);
    const auto bin = serialize_bytes(lists, vectors, 64);
    InlineForwardIndexTrailer trailer{};
    auto entries = parse_dir(bin, trailer);
    ASSERT_EQ(entries.size(), 1U);
    EXPECT_EQ(entries[0].byte_off % 64U, 0U);
    EXPECT_GT(entries[0].len, 64U);  // payload spans past one page
    auto index = read_index(bin);
    EXPECT_EQ(index.page_size(), 64U);
    verify_block(index, index.block(0, 0), {0, 1, 2, 3}, vectors);
}

TEST(InlineForwardIndex, NullArgumentsRejected) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    InlineForwardIndex writer(lists, vectors);
    EXPECT_THROW(writer.serialize(nullptr), std::invalid_argument);
    InlineForwardIndex reader;
    EXPECT_THROW(reader.mmap_deserialize(nullptr), std::invalid_argument);
}

// --- Truncated extra copies (header max_doc_nnz) ---

// A doc in two blocks, truncated to its largest few components everywhere but
// the one block its locator names -- which is what lets a re-scoring reader
// recover the whole vector.
TEST(InlineForwardIndex, KeepsOneWholeCopyAndTruncatesTheRest) {
    constexpr uint32_t kKeep = 2;
    // Doc 0 has 5 components with distinct values, so "the largest 2" is
    // unambiguous: components 3 and 4 (values 0.9 and 0.8).
    auto vectors = create_float_vectors(
        {{0, 1, 2, 3, 4}, {1, 2}},
        {{0.1F, 0.2F, 0.3F, 0.9F, 0.8F}, {0.5F, 0.6F}}, 10);
    // Two posting lists, each with one block, both holding doc 0.
    auto lists = build_lists({{{0, 1}}, {{0}}}, vectors);
    // Doc 0's whole copy goes to list 0 block 0 slot 0, doc 1's to the same
    // block at slot 1.
    const std::vector<nsparse::detail::DocLocator> full_copies = {{0, 0, 0},
                                                                 {0, 0, 1}};

    InlineForwardIndex writer(lists, vectors,
                              InlineForwardIndex::kDefaultPageSize,
                              InlineLayout::kPacked, kKeep, &full_copies);
    nsparse::BufferedIOWriter buffer;
    writer.serialize(&buffer);
    const std::vector<uint8_t> bin = buffer.data();
    EXPECT_EQ(parse_header(bin).max_doc_nnz, kKeep);

    auto index = read_index(bin);
    EXPECT_EQ(index.max_doc_nnz(), kKeep);

    // The named block keeps doc 0 whole.
    const BlockView whole = index.block(0, 0);
    ASSERT_FALSE(whole.absent());
    ASSERT_EQ(whole.n_docs, 2U);
    EXPECT_EQ(whole.nnz(0), 5U);
    const std::vector<nsparse::term_t> all_comps(whole.doc_comps(0),
                                                 whole.doc_comps(0) + 5);
    EXPECT_EQ(all_comps, (std::vector<nsparse::term_t>{0, 1, 2, 3, 4}));
    // Doc 1 is shorter than the limit, so it is whole wherever it appears.
    EXPECT_EQ(whole.nnz(1), 2U);

    // The other copy keeps only the two largest, still in ascending component
    // order so the block stays readable as a CSR row.
    const BlockView cut = index.block(1, 0);
    ASSERT_FALSE(cut.absent());
    ASSERT_EQ(cut.n_docs, 1U);
    ASSERT_EQ(cut.nnz(0), kKeep);
    const std::vector<nsparse::term_t> kept(cut.doc_comps(0),
                                            cut.doc_comps(0) + kKeep);
    EXPECT_EQ(kept, (std::vector<nsparse::term_t>{3, 4}));
    const auto* kept_vals =
        reinterpret_cast<const float*>(cut.doc_vals(0, nsparse::U32));
    EXPECT_FLOAT_EQ(kept_vals[0], 0.9F);
    EXPECT_FLOAT_EQ(kept_vals[1], 0.8F);
}

// Ties at the cut are taken in stored order, so exactly max_doc_nnz components
// survive rather than every component that equals the boundary value.
TEST(InlineForwardIndex, TruncationKeepsExactlyTheLimitAcrossTies) {
    constexpr uint32_t kKeep = 3;
    // All five values equal: every component is a tie at the boundary.
    auto vectors = create_uint8_vectors({{0, 1, 2, 3, 4}, {7}},
                                        {{5, 5, 5, 5, 5}, {9}}, 10);
    auto lists = build_lists({{{0}}, {{0, 1}}}, vectors);
    // Doc 0's whole copy is in list 1, so list 0's copy is the truncated one.
    const std::vector<nsparse::detail::DocLocator> full_copies = {{1, 0, 0},
                                                                 {1, 0, 1}};
    InlineForwardIndex writer(lists, vectors,
                              InlineForwardIndex::kDefaultPageSize,
                              InlineLayout::kPacked, kKeep, &full_copies);
    nsparse::BufferedIOWriter buffer;
    writer.serialize(&buffer);
    const std::vector<uint8_t> bin = buffer.data();
    auto index = read_index(bin);

    const BlockView cut = index.block(0, 0);
    ASSERT_FALSE(cut.absent());
    ASSERT_EQ(cut.nnz(0), kKeep);
    const std::vector<nsparse::term_t> kept(cut.doc_comps(0),
                                            cut.doc_comps(0) + kKeep);
    EXPECT_EQ(kept, (std::vector<nsparse::term_t>{0, 1, 2}));
    EXPECT_EQ(index.block(1, 0).nnz(0), 5U);  // the whole copy
}

// Truncating shrinks the section; the default writes nothing truncated.
TEST(InlineForwardIndex, TruncationShrinksTheSectionAndDefaultsOff) {
    auto vectors = create_float_vectors(
        {{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}},
        {{0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F},
         {0.6F, 0.5F, 0.4F, 0.3F, 0.2F, 0.1F}},
        10);
    auto lists = build_lists({{{0, 1}}, {{0, 1}}}, vectors);
    const std::vector<nsparse::detail::DocLocator> full_copies = {{0, 0, 0},
                                                                 {0, 0, 1}};

    const auto whole =
        serialize_bytes(lists, vectors, InlineForwardIndex::kDefaultPageSize,
                        InlineLayout::kPacked);
    EXPECT_EQ(parse_header(whole).max_doc_nnz, 0U);
    EXPECT_EQ(read_index(whole).max_doc_nnz(), 0U);

    InlineForwardIndex writer(lists, vectors,
                              InlineForwardIndex::kDefaultPageSize,
                              InlineLayout::kPacked, /*max_doc_nnz=*/2,
                              &full_copies);
    nsparse::BufferedIOWriter buffer;
    writer.serialize(&buffer);
    EXPECT_LT(buffer.data().size(), whole.size());
}

// The two truncation arguments are meaningless apart, so neither is accepted
// alone, and a locator table has to cover every vector.
TEST(InlineForwardIndex, RejectsInconsistentTruncationArguments) {
    auto vectors = sample_float_vectors();
    auto lists = build_lists(kLayout, vectors);
    const std::vector<nsparse::detail::DocLocator> full_copies(
        vectors.num_vectors(), {0, 0, 0});
    const std::vector<nsparse::detail::DocLocator> too_short(1, {0, 0, 0});

    EXPECT_THROW(InlineForwardIndex(lists, vectors,
                                    InlineForwardIndex::kDefaultPageSize,
                                    InlineLayout::kPacked, /*max_doc_nnz=*/2,
                                    nullptr),
                 std::invalid_argument);
    EXPECT_THROW(InlineForwardIndex(lists, vectors,
                                    InlineForwardIndex::kDefaultPageSize,
                                    InlineLayout::kPacked, /*max_doc_nnz=*/0,
                                    &full_copies),
                 std::invalid_argument);
    EXPECT_THROW(InlineForwardIndex(lists, vectors,
                                    InlineForwardIndex::kDefaultPageSize,
                                    InlineLayout::kPacked, /*max_doc_nnz=*/2,
                                    &too_short),
                 std::invalid_argument);
}

// --- Whole-copy resolution (resolve_doc / resolve_docs / doc_slice) ---

// The locator table the index writer builds: each doc's first occurrence in
// (pl, block, slot) order, which is the copy a truncating writer keeps whole.
std::vector<nsparse::detail::DocLocator> first_occurrence_locators(
    const std::vector<std::vector<std::vector<nsparse::idx_t>>>& layout,
    size_t num_docs) {
    using nsparse::detail::DocLocator;
    std::vector<DocLocator> locators(num_docs,
                                     {DocLocator::kRemainder, 0, 0});
    std::vector<bool> seen(num_docs, false);
    for (size_t pl = 0; pl < layout.size(); ++pl) {
        for (size_t block = 0; block < layout[pl].size(); ++block) {
            for (size_t slot = 0; slot < layout[pl][block].size(); ++slot) {
                const auto doc = static_cast<size_t>(layout[pl][block][slot]);
                if (!seen[doc]) {
                    seen[doc] = true;
                    locators[doc] = {static_cast<uint32_t>(pl),
                                     static_cast<uint32_t>(block),
                                     static_cast<uint32_t>(slot)};
                }
            }
        }
    }
    return locators;
}

// 40 docs of 6 components each, every doc in three blocks, truncated to 2 --
// enough docs to cross resolve_docs' internal chunk boundary several times.
struct TruncatedFixture {
    static constexpr size_t kDocs = 40;
    static constexpr uint32_t kKeep = 2;
    nsparse::SparseVectors vectors;
    std::vector<std::vector<std::vector<nsparse::idx_t>>> layout;
    std::vector<nsparse::detail::DocLocator> locators;
    std::vector<uint8_t> bin;

    TruncatedFixture()
        : vectors(nsparse::SparseVectors(
              {.element_size = nsparse::U32, .dimension = 64})) {
        for (size_t doc = 0; doc < kDocs; ++doc) {
            std::vector<nsparse::term_t> comps;
            std::vector<float> vals;
            for (nsparse::term_t c = 0; c < 6; ++c) {
                comps.push_back(static_cast<nsparse::term_t>(doc % 8 + c * 9));
                vals.push_back(0.1F * static_cast<float>(c + 1) +
                               0.001F * static_cast<float>(doc));
            }
            vectors.add_vector(comps.data(), comps.size(),
                               reinterpret_cast<const uint8_t*>(vals.data()),
                               vals.size() * sizeof(float));
        }
        // Three lists; list p puts doc d in block (d + p) % 5.
        for (size_t pl = 0; pl < 3; ++pl) {
            std::vector<std::vector<nsparse::idx_t>> blocks(5);
            for (size_t doc = 0; doc < kDocs; ++doc) {
                blocks[(doc + pl) % 5].push_back(
                    static_cast<nsparse::idx_t>(doc));
            }
            layout.push_back(std::move(blocks));
        }
        locators = first_occurrence_locators(layout, kDocs);
        auto lists = build_lists(layout, vectors);
        InlineForwardIndex writer(lists, vectors,
                                  InlineForwardIndex::kDefaultPageSize,
                                  InlineLayout::kPacked, kKeep, &locators);
        nsparse::BufferedIOWriter buffer;
        writer.serialize(&buffer);
        bin = buffer.data();
    }
};

TEST(InlineForwardIndex, ResolveDocReturnsTheWholeCopy) {
    TruncatedFixture fx;
    auto index = read_index(fx.bin);
    const nsparse::detail::FullVectorStore store{
        .fwd = &index,
        .locators = fx.locators.data(),
        .num_locators = fx.locators.size(),
        .remainder = nullptr};
    const auto* indptr = fx.vectors.indptr_data();
    for (size_t doc = 0; doc < TruncatedFixture::kDocs; ++doc) {
        const nsparse::detail::DocSlice slice = nsparse::detail::resolve_doc(
            store, static_cast<nsparse::idx_t>(doc), nsparse::U32);
        const auto start = indptr[doc];
        ASSERT_EQ(slice.nnz, static_cast<size_t>(indptr[doc + 1] - start))
            << "doc " << doc << " resolved to a truncated copy";
        EXPECT_EQ(0, std::memcmp(slice.comps, fx.vectors.indices_data() + start,
                                 slice.nnz * sizeof(nsparse::term_t)));
        EXPECT_EQ(0, std::memcmp(slice.vals,
                                 fx.vectors.values_data() +
                                     static_cast<size_t>(start) * sizeof(float),
                                 slice.nnz * sizeof(float)));
    }
}

// The batched walk is an optimization of resolve_doc and must agree with it
// exactly -- same slices, in the caller's order, duplicates included -- across
// more docs than one internal chunk.
TEST(InlineForwardIndex, ResolveDocsMatchesResolveDocOneByOne) {
    TruncatedFixture fx;
    auto index = read_index(fx.bin);
    const nsparse::detail::FullVectorStore store{
        .fwd = &index,
        .locators = fx.locators.data(),
        .num_locators = fx.locators.size(),
        .remainder = nullptr};
    std::vector<nsparse::idx_t> ids;
    for (size_t i = 0; i < 3 * TruncatedFixture::kDocs; ++i) {
        ids.push_back(static_cast<nsparse::idx_t>((i * 17) %
                                                  TruncatedFixture::kDocs));
    }
    nsparse::detail::DocResolveScratch scratch;
    std::vector<nsparse::detail::DocSlice> slices;
    nsparse::detail::resolve_docs(store, ids.data(), ids.size(), nsparse::U32,
                                  &scratch, &slices);
    ASSERT_EQ(slices.size(), ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        const auto one = nsparse::detail::resolve_doc(store, ids[i],
                                                      nsparse::U32);
        EXPECT_EQ(slices[i].comps, one.comps) << "position " << i;
        EXPECT_EQ(slices[i].vals, one.vals) << "position " << i;
        EXPECT_EQ(slices[i].nnz, one.nnz) << "position " << i;
    }

    // Reusing the scratch for a shorter batch must not leave stale entries.
    nsparse::detail::resolve_docs(store, ids.data(), 3, nsparse::U32, &scratch,
                                  &slices);
    EXPECT_EQ(slices.size(), 3U);
    nsparse::detail::resolve_docs(store, ids.data(), 0, nsparse::U32, &scratch,
                                  &slices);
    EXPECT_TRUE(slices.empty());
}

TEST(InlineForwardIndex, ResolveRejectsDocsOutsideTheDirectory) {
    TruncatedFixture fx;
    auto index = read_index(fx.bin);
    const nsparse::detail::FullVectorStore store{
        .fwd = &index,
        .locators = fx.locators.data(),
        .num_locators = fx.locators.size(),
        .remainder = nullptr};
    const auto past_end = static_cast<nsparse::idx_t>(TruncatedFixture::kDocs);
    EXPECT_THROW(nsparse::detail::resolve_doc(store, past_end, nsparse::U32),
                 std::out_of_range);
    EXPECT_THROW(nsparse::detail::resolve_doc(store, -1, nsparse::U32),
                 std::out_of_range);

    const std::vector<nsparse::idx_t> ids = {0, 1, past_end};
    nsparse::detail::DocResolveScratch scratch;
    std::vector<nsparse::detail::DocSlice> slices;
    EXPECT_THROW(nsparse::detail::resolve_docs(store, ids.data(), ids.size(),
                                               nsparse::U32, &scratch, &slices),
                 std::out_of_range);
}

// A locator naming a slot its block does not have is a corrupt directory; both
// resolve paths must refuse it rather than read past the block.
TEST(InlineForwardIndex, ResolveRejectsALocatorPastItsBlock) {
    TruncatedFixture fx;
    auto index = read_index(fx.bin);
    std::vector<nsparse::detail::DocLocator> corrupt = fx.locators;
    corrupt[0].slot = 1000;
    const nsparse::detail::FullVectorStore store{
        .fwd = &index,
        .locators = corrupt.data(),
        .num_locators = corrupt.size(),
        .remainder = nullptr};
    EXPECT_THROW(nsparse::detail::resolve_doc(store, 0, nsparse::U32),
                 std::runtime_error);
    const std::vector<nsparse::idx_t> ids = {0};
    nsparse::detail::DocResolveScratch scratch;
    std::vector<nsparse::detail::DocSlice> slices;
    EXPECT_THROW(nsparse::detail::resolve_docs(store, ids.data(), ids.size(),
                                               nsparse::U32, &scratch, &slices),
                 std::runtime_error);

    const nsparse::detail::InlineDirEntry* entry = index.dir_entry(0, 0);
    ASSERT_NE(entry, nullptr);
    EXPECT_THROW(index.doc_slice(*entry, entry->n_docs), std::runtime_error);
    EXPECT_EQ(index.dir_entry(0, 99), nullptr);
    EXPECT_EQ(index.dir_entry(99, 0), nullptr);
}
