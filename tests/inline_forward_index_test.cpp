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
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/io/buffered_io.h"
#include "nsparse/io/inline_forward_index_writer.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace {

// Alpha for summarize(); its only effect here is to make cluster_size() reflect
// the block count (the writer reads the full member set via get_docs(), which
// is independent of summary pruning).
constexpr float kAlpha = 0.5F;

// Copy a trivially-copyable value out of a byte buffer (buffer offsets are not
// guaranteed aligned for the wider fields, so go through memcpy).
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

// Build one InvertedListClusters per posting list from an explicit block->docs
// layout and summarize each so cluster_size() reports the block count.
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

nsparse::InlineForwardIndexHeader parse_header(
    const std::vector<uint8_t>& bin) {
    nsparse::InlineForwardIndexHeader header{};
    EXPECT_GE(bin.size(), sizeof(header));
    if (bin.size() >= sizeof(header)) {
        header = read_pod<nsparse::InlineForwardIndexHeader>(bin.data());
    }
    return header;
}

// Read the trailer at EOF and the directory it points to. Bounds-checked so a
// regressed writer yields a clean EXPECT failure instead of reading past the
// buffer.
std::vector<nsparse::InlineDirEntry> parse_dir(
    const std::vector<uint8_t>& bin,
    nsparse::InlineForwardIndexTrailer& trailer_out) {
    trailer_out = nsparse::InlineForwardIndexTrailer{};
    std::vector<nsparse::InlineDirEntry> entries;
    const size_t tsz = sizeof(nsparse::InlineForwardIndexTrailer);
    EXPECT_GE(bin.size(), tsz);
    if (bin.size() < tsz) {
        return entries;
    }
    trailer_out = read_pod<nsparse::InlineForwardIndexTrailer>(
        bin.data() + bin.size() - tsz);
    // The directory occupies [dir_offset, bin.size() - tsz).
    EXPECT_LE(trailer_out.dir_offset, bin.size() - tsz);
    if (trailer_out.dir_offset > bin.size() - tsz) {
        return entries;
    }
    const uint64_t dir_bytes = (bin.size() - tsz) - trailer_out.dir_offset;
    EXPECT_EQ(trailer_out.n_entries * sizeof(nsparse::InlineDirEntry),
              dir_bytes);
    const uint64_t available = dir_bytes / sizeof(nsparse::InlineDirEntry);
    const uint64_t n = std::min<uint64_t>(trailer_out.n_entries, available);
    for (uint64_t i = 0; i < n; ++i) {
        entries.push_back(read_pod<nsparse::InlineDirEntry>(
            bin.data() + trailer_out.dir_offset +
            i * sizeof(nsparse::InlineDirEntry)));
    }
    return entries;
}

// Total file size the writer must produce for a given directory: the directory
// starts at trailer.dir_offset, holds n_entries entries, and is followed by the
// fixed trailer.
uint64_t expected_file_size(const nsparse::InlineForwardIndexTrailer& trailer) {
    return trailer.dir_offset +
           trailer.n_entries * sizeof(nsparse::InlineDirEntry) +
           sizeof(nsparse::InlineForwardIndexTrailer);
}

// Parse the structure-of-arrays block at entry.byte_off and assert every
// document's component ids and value bytes reproduce the source exactly, in
// order. Layout: [n_docs][doc_id[]][off[]][comps[]](pad)[vals[]].
void verify_block(const std::vector<uint8_t>& bin,
                  const nsparse::InlineDirEntry& entry,
                  const std::vector<nsparse::idx_t>& expected_docs,
                  const nsparse::SparseVectors& vectors) {
    ASSERT_LE(entry.byte_off + entry.len, bin.size());
    EXPECT_EQ(entry.reserved, 0U);  // reserved must be zero-filled
    const uint8_t* base = bin.data() + entry.byte_off;

    const uint32_t n_docs = read_pod<uint32_t>(base);
    EXPECT_EQ(n_docs, entry.n_docs);
    ASSERT_EQ(n_docs, expected_docs.size());

    // doc_id[] and off[] offsets are independent of total_nnz.
    const size_t off_pos = sizeof(uint32_t) + n_docs * sizeof(uint32_t);
    std::vector<uint32_t> off(n_docs + 1);
    for (uint32_t i = 0; i <= n_docs; ++i) {
        off[i] = read_pod<uint32_t>(base + off_pos + i * sizeof(uint32_t));
    }
    EXPECT_EQ(off[0], 0U);
    const uint32_t total_nnz = off[n_docs];
    const size_t element_size = vectors.get_element_size();
    const auto layout =
        nsparse::inline_block_offsets(n_docs, total_nnz, element_size);
    // The payload length recorded in the directory must be exact.
    EXPECT_EQ(layout.end, entry.len);

    const auto* indptr = vectors.indptr_data();
    const auto* indices = vectors.indices_data();
    const auto* values = vectors.values_data();

    for (uint32_t i = 0; i < n_docs; ++i) {
        const uint32_t doc_id =
            read_pod<uint32_t>(base + sizeof(uint32_t) + i * sizeof(uint32_t));
        EXPECT_EQ(doc_id, static_cast<uint32_t>(expected_docs[i]));
        ASSERT_LT(doc_id, vectors.num_vectors());  // guard before indexing
        const nsparse::idx_t start = indptr[doc_id];
        const uint32_t src_nnz =
            static_cast<uint32_t>(indptr[doc_id + 1] - start);
        const uint32_t nnz = off[i + 1] - off[i];  // offsets are monotonic
        ASSERT_EQ(nnz, src_nnz);

        // Each doc's comps/vals slice starts at its within-block offset off[i].
        EXPECT_EQ(0, std::memcmp(
                         base + layout.comps + off[i] * sizeof(nsparse::term_t),
                         indices + start, nnz * sizeof(nsparse::term_t)))
            << "component mismatch for doc " << doc_id;
        EXPECT_EQ(
            0, std::memcmp(base + layout.vals + off[i] * element_size,
                           values + static_cast<size_t>(start) * element_size,
                           static_cast<size_t>(nnz) * element_size))
            << "value bytes mismatch for doc " << doc_id;
    }
    // vals must start on an element_size boundary so it can be read in place.
    EXPECT_EQ((entry.byte_off + layout.vals) % element_size, 0U);
}

// Assert every byte in [from, to) of the buffer is zero (padding regions).
void expect_zero_range(const std::vector<uint8_t>& buf, size_t from,
                       size_t to) {
    ASSERT_LE(to, buf.size());
    for (size_t i = from; i < to; ++i) {
        EXPECT_EQ(buf[i], 0) << "non-zero padding byte at offset " << i;
    }
}

}  // namespace

TEST(InlineForwardIndex, HeaderTrailerAndDirectoryLayout) {
    // dimension 10; term ids < 10.
    auto vectors =
        create_float_vectors({{0, 3}, {1}, {2, 4, 6}, {0, 9}, {5}, {3, 7}},
                             {{1.0F, 2.0F},
                              {3.0F},
                              {1.5F, 2.5F, 3.5F},
                              {4.0F, 5.0F},
                              {6.0F},
                              {7.0F, 8.0F}},
                             10);
    // Two posting lists; doc 3 and doc 0 are duplicated across blocks/lists.
    const std::vector<std::vector<std::vector<nsparse::idx_t>>> layout = {
        {{0, 1}, {2, 3, 4}}, {{3, 5}, {0}}};
    auto lists = build_lists(layout, vectors);

    nsparse::InlineForwardIndexWriter writer;  // default 4 KiB page
    nsparse::BufferedIOWriter bin;
    writer.write(lists, vectors, &bin);

    const auto page = nsparse::InlineForwardIndexWriter::kDefaultPageSize;
    const auto header = parse_header(bin.data());
    EXPECT_EQ(header.element_size, nsparse::U32);
    EXPECT_EQ(header.n_blocks, 4U);
    EXPECT_EQ(header.page_size, page);

    nsparse::InlineForwardIndexTrailer trailer{};
    auto entries = parse_dir(bin.data(), trailer);
    EXPECT_EQ(trailer.n_lists, 2U);
    EXPECT_EQ(trailer.n_entries, 4U);
    ASSERT_EQ(entries.size(), 4U);

    // Entries are (pl, block) in ascending order, first block page-aligned, and
    // offsets strictly increasing and page-aligned.
    const std::vector<std::pair<uint32_t, uint32_t>> expected_ids = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}};
    uint64_t prev_off = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        EXPECT_EQ(entries[i].pl, expected_ids[i].first);
        EXPECT_EQ(entries[i].block, expected_ids[i].second);
        EXPECT_EQ(entries[i].byte_off % page, 0U);
        if (i == 0) {
            EXPECT_EQ(entries[i].byte_off, page);
        } else {
            EXPECT_GT(entries[i].byte_off, prev_off);
        }
        prev_off = entries[i].byte_off;
    }

    // Each block's byte_off equals the previous block's end rounded up.
    for (size_t i = 1; i < entries.size(); ++i) {
        const uint64_t prev_end = entries[i - 1].byte_off + entries[i - 1].len;
        EXPECT_EQ(entries[i].byte_off,
                  nsparse::inline_align_up(prev_end, page));
    }

    // The directory begins right after the last (padded) block, and the file
    // ends after the directory + trailer.
    const uint64_t last_end = entries.back().byte_off + entries.back().len;
    EXPECT_EQ(trailer.dir_offset, nsparse::inline_align_up(last_end, page));
    EXPECT_EQ(bin.data().size(), expected_file_size(trailer));
}

TEST(InlineForwardIndex, BlockContentsMatchForwardVectors) {
    auto vectors =
        create_float_vectors({{0, 3}, {1}, {2, 4, 6}, {0, 9}, {5}, {3, 7}},
                             {{1.0F, 2.0F},
                              {3.0F},
                              {1.5F, 2.5F, 3.5F},
                              {4.0F, 5.0F},
                              {6.0F},
                              {7.0F, 8.0F}},
                             10);
    const std::vector<std::vector<std::vector<nsparse::idx_t>>> layout = {
        {{0, 1}, {2, 3, 4}}, {{3, 5}, {0}}};
    auto lists = build_lists(layout, vectors);

    nsparse::InlineForwardIndexWriter writer;
    nsparse::BufferedIOWriter bin;
    writer.write(lists, vectors, &bin);

    nsparse::InlineForwardIndexTrailer trailer{};
    auto entries = parse_dir(bin.data(), trailer);
    ASSERT_EQ(entries.size(), 4U);

    // Verify every block's records reproduce the source vectors, and that a
    // duplicated doc (doc 3 in pl0/block1 and pl1/block0) is stored in full in
    // both.
    verify_block(bin.data(), entries[0], layout[0][0], vectors);
    verify_block(bin.data(), entries[1], layout[0][1], vectors);
    verify_block(bin.data(), entries[2], layout[1][0], vectors);
    verify_block(bin.data(), entries[3], layout[1][1], vectors);
}

TEST(InlineForwardIndex, ElementSizeVariantsU16) {
    auto vectors = create_uint16_vectors({{0, 3}, {1}, {2, 4, 6}},
                                         {{10, 20}, {30}, {15, 25, 35}}, 8);
    auto lists = build_lists({{{0, 1}, {2}}}, vectors);

    nsparse::InlineForwardIndexWriter writer;
    nsparse::BufferedIOWriter bin;
    writer.write(lists, vectors, &bin);

    const auto header = parse_header(bin.data());
    EXPECT_EQ(header.element_size, nsparse::U16);
    nsparse::InlineForwardIndexTrailer trailer{};
    auto entries = parse_dir(bin.data(), trailer);
    ASSERT_EQ(entries.size(), 2U);
    verify_block(bin.data(), entries[0], {0, 1}, vectors);
    verify_block(bin.data(), entries[1], {2}, vectors);
}

TEST(InlineForwardIndex, ElementSizeVariantsU8) {
    auto vectors = create_uint8_vectors({{0, 3}, {1}, {2, 4, 6}},
                                        {{10, 20}, {30}, {15, 25, 35}}, 8);
    auto lists = build_lists({{{0, 1}, {2}}}, vectors);

    nsparse::InlineForwardIndexWriter writer;
    nsparse::BufferedIOWriter bin;
    writer.write(lists, vectors, &bin);

    const auto header = parse_header(bin.data());
    EXPECT_EQ(header.element_size, nsparse::U8);
    nsparse::InlineForwardIndexTrailer trailer{};
    auto entries = parse_dir(bin.data(), trailer);
    ASSERT_EQ(entries.size(), 2U);
    verify_block(bin.data(), entries[0], {0, 1}, vectors);
    verify_block(bin.data(), entries[1], {2}, vectors);
}

TEST(InlineForwardIndex, SingleDocBlock) {
    auto vectors = create_float_vectors({{0, 1, 2}}, {{1.0F, 2.0F, 3.0F}}, 4);
    auto lists = build_lists({{{0}}}, vectors);

    nsparse::InlineForwardIndexWriter writer;
    nsparse::BufferedIOWriter bin;
    writer.write(lists, vectors, &bin);

    const auto header = parse_header(bin.data());
    EXPECT_EQ(header.n_blocks, 1U);
    nsparse::InlineForwardIndexTrailer trailer{};
    auto entries = parse_dir(bin.data(), trailer);
    ASSERT_EQ(entries.size(), 1U);
    EXPECT_EQ(entries[0].n_docs, 1U);
    verify_block(bin.data(), entries[0], {0}, vectors);
}

TEST(InlineForwardIndex, EmptyClusterProducesZeroDocBlock) {
    // A posting list whose second block has no member documents.
    auto vectors =
        create_float_vectors({{0, 1}, {2}}, {{1.0F, 2.0F}, {3.0F}}, 4);
    auto lists = build_lists({{{0, 1}, {}}}, vectors);

    nsparse::InlineForwardIndexWriter writer;
    nsparse::BufferedIOWriter bin;
    writer.write(lists, vectors, &bin);

    nsparse::InlineForwardIndexTrailer trailer{};
    auto entries = parse_dir(bin.data(), trailer);
    ASSERT_EQ(entries.size(), 2U);
    EXPECT_EQ(entries[1].n_docs, 0U);
    // Empty block: [n_docs=0][off[0]=0], no doc_id/comps/vals.
    EXPECT_EQ(entries[1].len,
              nsparse::inline_block_offsets(0, 0, nsparse::U32).end);
    verify_block(bin.data(), entries[0], {0, 1}, vectors);
    verify_block(bin.data(), entries[1], {}, vectors);
}

TEST(InlineForwardIndex, EmptyListsContributeNoBlocks) {
    // One posting list with docs, one posting list with zero clusters.
    auto vectors = create_float_vectors({{0}, {1}}, {{1.0F}, {2.0F}}, 4);
    std::vector<nsparse::InvertedListClusters> lists;
    nsparse::InvertedListClusters list0({{0, 1}});
    list0.summarize(&vectors, kAlpha);
    lists.push_back(std::move(list0));
    lists.emplace_back();  // default: cluster_size() == 0

    nsparse::InlineForwardIndexWriter writer;
    nsparse::BufferedIOWriter bin;
    writer.write(lists, vectors, &bin);

    const auto header = parse_header(bin.data());
    EXPECT_EQ(header.n_blocks, 1U);
    nsparse::InlineForwardIndexTrailer trailer{};
    auto entries = parse_dir(bin.data(), trailer);
    EXPECT_EQ(trailer.n_lists, 2U);
    ASSERT_EQ(entries.size(), 1U);
    EXPECT_EQ(entries[0].pl, 0U);
    verify_block(bin.data(), entries[0], {0, 1}, vectors);
}

TEST(InlineForwardIndex, NoListsWritesEmptyDirectory) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 4});
    std::vector<nsparse::InvertedListClusters> lists;

    nsparse::InlineForwardIndexWriter writer;
    nsparse::BufferedIOWriter bin;
    writer.write(lists, vectors, &bin);

    const auto page = nsparse::InlineForwardIndexWriter::kDefaultPageSize;
    const auto header = parse_header(bin.data());
    EXPECT_EQ(header.n_blocks, 0U);

    nsparse::InlineForwardIndexTrailer trailer{};
    auto entries = parse_dir(bin.data(), trailer);
    EXPECT_EQ(trailer.n_lists, 0U);
    EXPECT_EQ(trailer.n_entries, 0U);
    EXPECT_TRUE(entries.empty());
    // With no blocks the directory starts at the (still page-aligned) first
    // block offset; the file is header + pad + (empty dir) + trailer.
    EXPECT_EQ(trailer.dir_offset, page);
    EXPECT_EQ(bin.data().size(), expected_file_size(trailer));
    // The header padding is zero-filled (byte-reproducible artifact).
    expect_zero_range(bin.data(), sizeof(nsparse::InlineForwardIndexHeader),
                      page);
}

TEST(InlineForwardIndex, CustomPageSizeAlignsBlocks) {
    auto vectors = create_float_vectors(
        {{0, 3}, {1}, {2, 4, 6}, {0, 9}},
        {{1.0F, 2.0F}, {3.0F}, {1.5F, 2.5F, 3.5F}, {4.0F, 5.0F}}, 10);
    auto lists = build_lists({{{0, 1}, {2, 3}}}, vectors);

    nsparse::InlineForwardIndexWriter writer(64);  // small power-of-two page
    nsparse::BufferedIOWriter bin;
    writer.write(lists, vectors, &bin);

    const auto header = parse_header(bin.data());
    EXPECT_EQ(header.page_size, 64U);
    nsparse::InlineForwardIndexTrailer trailer{};
    auto entries = parse_dir(bin.data(), trailer);
    ASSERT_EQ(entries.size(), 2U);
    for (const auto& entry : entries) {
        EXPECT_EQ(entry.byte_off % 64U, 0U);
    }
    EXPECT_EQ(entries[0].byte_off, 64U);
    EXPECT_EQ(
        entries[1].byte_off,
        nsparse::inline_align_up(entries[0].byte_off + entries[0].len, 64));
    verify_block(bin.data(), entries[0], {0, 1}, vectors);
    verify_block(bin.data(), entries[1], {2, 3}, vectors);
}

TEST(InlineForwardIndex, MultiPageBlockSpansAndPadsCorrectly) {
    // Six docs, each nnz=2.
    auto vectors =
        create_float_vectors({{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}},
                             {{1.0F, 2.0F},
                              {3.0F, 4.0F},
                              {5.0F, 6.0F},
                              {7.0F, 8.0F},
                              {9.0F, 10.0F},
                              {11.0F, 12.0F}},
                             10);
    // block0 has 4 docs (total_nnz=8) -> SoA payload 88 bytes, spanning
    // multiple 32-byte pages; block1 has 2 docs (total_nnz=4) -> 48 bytes.
    auto lists = build_lists({{{0, 1, 2, 3}, {4, 5}}}, vectors);

    nsparse::InlineForwardIndexWriter writer(32);
    nsparse::BufferedIOWriter bin;
    writer.write(lists, vectors, &bin);

    nsparse::InlineForwardIndexTrailer trailer{};
    auto entries = parse_dir(bin.data(), trailer);
    ASSERT_EQ(entries.size(), 2U);

    const uint64_t len0 = nsparse::inline_block_offsets(4, 8, nsparse::U32).end;
    const uint64_t len1 = nsparse::inline_block_offsets(2, 4, nsparse::U32).end;
    EXPECT_EQ(len0, 88U);
    EXPECT_EQ(len1, 48U);

    // block0: page-aligned start, exact multi-page payload length.
    EXPECT_EQ(entries[0].byte_off, 32U);
    EXPECT_EQ(entries[0].len, len0);  // 88, > 2 pages
    EXPECT_GT(entries[0].len, 64U);
    // block1 begins at the next page boundary after block0's payload, which is
    // more than one page past block0's start.
    const uint64_t expected1 =
        nsparse::inline_align_up(entries[0].byte_off + entries[0].len, 32);
    EXPECT_EQ(entries[1].byte_off, expected1);
    EXPECT_EQ(entries[1].byte_off % 32U, 0U);
    EXPECT_GT(entries[1].byte_off - entries[0].byte_off, 32U);
    EXPECT_EQ(entries[1].len, len1);  // 48

    // Inter-block padding bytes are zero.
    expect_zero_range(bin.data(), entries[0].byte_off + entries[0].len,
                      entries[1].byte_off);

    verify_block(bin.data(), entries[0], {0, 1, 2, 3}, vectors);
    verify_block(bin.data(), entries[1], {4, 5}, vectors);

    // The directory starts at the last block's payload rounded up to a page,
    // the last block's trailing pad is zero, and the file ends after the
    // directory + trailer.
    const uint64_t last_end = entries[1].byte_off + entries[1].len;
    EXPECT_EQ(trailer.dir_offset, nsparse::inline_align_up(last_end, 32));
    expect_zero_range(bin.data(), last_end, trailer.dir_offset);
    EXPECT_EQ(bin.data().size(), expected_file_size(trailer));
}

TEST(InlineForwardIndex, ZeroNnzDocumentIsStored) {
    // doc 0 has an empty stored vector (nnz == 0); doc 1 is normal.
    auto vectors = create_float_vectors({{}, {0}}, {{}, {1.0F}}, 4);
    auto lists = build_lists({{{0, 1}}}, vectors);

    nsparse::InlineForwardIndexWriter writer;
    nsparse::BufferedIOWriter bin;
    writer.write(lists, vectors, &bin);

    nsparse::InlineForwardIndexTrailer trailer{};
    auto entries = parse_dir(bin.data(), trailer);
    ASSERT_EQ(entries.size(), 1U);
    // SoA: n_docs + doc_id[2] + off[3] + comps[1]=2B, pad 2B, vals[1]=4B.
    // = 4 + 8 + 12 + 2 + 2 + 4 = 32.
    EXPECT_EQ(entries[0].len, nsparse::inline_block_offsets(
                                  /*n_docs=*/2, /*total_nnz=*/1, nsparse::U32)
                                  .end);
    EXPECT_EQ(entries[0].len, 32U);
    verify_block(bin.data(), entries[0], {0, 1}, vectors);
}

TEST(InlineForwardIndex, RejectsUnsupportedElementSize) {
    // U64 (8) is in the ElementSize enum but not a supported inline width.
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U64, .dimension = 4});
    std::vector<nsparse::InvertedListClusters> lists;

    nsparse::InlineForwardIndexWriter writer;
    nsparse::BufferedIOWriter bin;
    EXPECT_THROW(writer.write(lists, vectors, &bin), std::invalid_argument);
}

TEST(InlineForwardIndex, RejectsNullWriter) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 4});
    std::vector<nsparse::InvertedListClusters> lists;

    nsparse::InlineForwardIndexWriter writer;
    EXPECT_THROW(writer.write(lists, vectors, nullptr), std::invalid_argument);
}

TEST(InlineForwardIndex, RejectsOutOfRangeDocId) {
    // Summarize the list against a 3-doc index, then write it against a 2-doc
    // index: block references doc 2, which is out of range for the writer's
    // vectors (simulating a stale/mismatched cluster structure).
    auto vectors_full =
        create_float_vectors({{0}, {1}, {2}}, {{1.0F}, {2.0F}, {3.0F}}, 4);
    auto lists = build_lists({{{0, 1, 2}}}, vectors_full);

    auto vectors_small = create_float_vectors({{0}, {1}}, {{1.0F}, {2.0F}}, 4);
    nsparse::InlineForwardIndexWriter writer;
    nsparse::BufferedIOWriter bin;
    EXPECT_THROW(writer.write(lists, vectors_small, &bin), std::out_of_range);
}

TEST(InlineForwardIndex, PackedLayoutMinimalAlignment) {
    auto vectors =
        create_float_vectors({{0, 3}, {1}, {2, 4, 6}, {0, 9}, {5}, {3, 7}},
                             {{1.0F, 2.0F},
                              {3.0F},
                              {1.5F, 2.5F, 3.5F},
                              {4.0F, 5.0F},
                              {6.0F},
                              {7.0F, 8.0F}},
                             10);
    const std::vector<std::vector<std::vector<nsparse::idx_t>>> layout = {
        {{0, 1}, {2, 3, 4}}, {{3, 5}, {0}}};
    auto lists = build_lists(layout, vectors);

    nsparse::InlineForwardIndexWriter writer(
        nsparse::InlineForwardIndexWriter::kDefaultPageSize,
        nsparse::InlineLayout::kPacked);
    nsparse::BufferedIOWriter bin;
    writer.write(lists, vectors, &bin);

    // Packed records the minimum block alignment as the effective alignment.
    const auto header = parse_header(bin.data());
    EXPECT_EQ(header.page_size, nsparse::kMinBlockAlign);
    EXPECT_EQ(header.n_blocks, 4U);
    nsparse::InlineForwardIndexTrailer trailer{};
    auto entries = parse_dir(bin.data(), trailer);
    ASSERT_EQ(entries.size(), 4U);

    // Blocks are back-to-back except for the minimal kMinBlockAlign padding:
    // the first follows the header (rounded up) and each next begins where the
    // previous ended, rounded up to kMinBlockAlign. Every block is aligned.
    uint64_t running = sizeof(nsparse::InlineForwardIndexHeader);
    for (const auto& entry : entries) {
        running = nsparse::inline_align_up(running, nsparse::kMinBlockAlign);
        EXPECT_EQ(entry.byte_off, running);
        EXPECT_EQ(entry.byte_off % nsparse::kMinBlockAlign, 0U);
        running += entry.len;
    }
    // The directory follows the last (aligned) block, then the trailer.
    running = nsparse::inline_align_up(running, nsparse::kMinBlockAlign);
    EXPECT_EQ(trailer.dir_offset, running);
    EXPECT_EQ(bin.data().size(), expected_file_size(trailer));

    verify_block(bin.data(), entries[0], layout[0][0], vectors);
    verify_block(bin.data(), entries[1], layout[0][1], vectors);
    verify_block(bin.data(), entries[2], layout[1][0], vectors);
    verify_block(bin.data(), entries[3], layout[1][1], vectors);

    // Packed is strictly smaller than the page-aligned layout for the same
    // data.
    nsparse::InlineForwardIndexWriter padded_writer;  // default: page-aligned
    nsparse::BufferedIOWriter padded_bin;
    padded_writer.write(lists, vectors, &padded_bin);
    EXPECT_LT(bin.data().size(), padded_bin.data().size());
}

TEST(InlineForwardIndex, PackedLayoutParityAcrossWidthsAndEdges) {
    // Packed path with a non-float width (u8), a zero-nnz document, an empty
    // (zero-doc) block, and two posting lists -- the edge cases the padded path
    // covers, exercised here back-to-back.
    auto vectors =
        create_uint8_vectors({{}, {0, 3}, {1}}, {{}, {10, 20}, {30}}, 8);
    // list0: a normal block (incl. the zero-nnz doc 0) then an empty block;
    // list1: a single-doc block.
    auto lists = build_lists({{{0, 1}, {}}, {{2}}}, vectors);

    nsparse::InlineForwardIndexWriter writer(
        nsparse::InlineForwardIndexWriter::kDefaultPageSize,
        nsparse::InlineLayout::kPacked);
    nsparse::BufferedIOWriter bin;
    writer.write(lists, vectors, &bin);

    const auto header = parse_header(bin.data());
    EXPECT_EQ(header.element_size, nsparse::U8);
    EXPECT_EQ(header.page_size, nsparse::kMinBlockAlign);
    nsparse::InlineForwardIndexTrailer trailer{};
    auto entries = parse_dir(bin.data(), trailer);
    EXPECT_EQ(trailer.n_lists, 2U);
    ASSERT_EQ(entries.size(), 3U);

    // Back-to-back on the kMinBlockAlign grid -- even with an empty block and a
    // zero-nnz doc in the mix.
    uint64_t running = sizeof(nsparse::InlineForwardIndexHeader);
    for (const auto& entry : entries) {
        running = nsparse::inline_align_up(running, nsparse::kMinBlockAlign);
        EXPECT_EQ(entry.byte_off, running);
        running += entry.len;
    }
    running = nsparse::inline_align_up(running, nsparse::kMinBlockAlign);
    EXPECT_EQ(trailer.dir_offset, running);
    EXPECT_EQ(bin.data().size(), expected_file_size(trailer));
    EXPECT_EQ(entries[1].n_docs, 0U);  // the empty block
    EXPECT_EQ(entries[1].len,
              nsparse::inline_block_offsets(0, 0, nsparse::U8).end);

    verify_block(bin.data(), entries[0], {0, 1}, vectors);
    verify_block(bin.data(), entries[1], {}, vectors);
    verify_block(bin.data(), entries[2], {2}, vectors);
}

TEST(InlineForwardIndex, PackedLayoutIgnoresPageSize) {
    // page_size values that are rejected for the page-aligned layout are
    // accepted (and unused) when packed.
    EXPECT_NO_THROW(
        nsparse::InlineForwardIndexWriter(0, nsparse::InlineLayout::kPacked));
    EXPECT_NO_THROW(
        nsparse::InlineForwardIndexWriter(7, nsparse::InlineLayout::kPacked));
}

TEST(InlineForwardIndex, RejectsInvalidPageSize) {
    EXPECT_THROW(nsparse::InlineForwardIndexWriter(0), std::invalid_argument);
    EXPECT_THROW(nsparse::InlineForwardIndexWriter(100),  // not power of two
                 std::invalid_argument);
    EXPECT_THROW(nsparse::InlineForwardIndexWriter(16),  // < header size
                 std::invalid_argument);
    EXPECT_NO_THROW(nsparse::InlineForwardIndexWriter(32));
    EXPECT_NO_THROW(nsparse::InlineForwardIndexWriter(4096));
}
