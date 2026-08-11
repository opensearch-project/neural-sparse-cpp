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

nsparse::InlineForwardIndexHeader parse_bin_header(
    const std::vector<uint8_t>& bin) {
    nsparse::InlineForwardIndexHeader header{};
    EXPECT_GE(bin.size(), sizeof(header));
    if (bin.size() >= sizeof(header)) {
        header = read_pod<nsparse::InlineForwardIndexHeader>(bin.data());
    }
    return header;
}

std::vector<nsparse::InlineDirEntry> parse_dir(
    const std::vector<uint8_t>& dir, nsparse::InlineDirHeader& header_out) {
    header_out = nsparse::InlineDirHeader{};
    std::vector<nsparse::InlineDirEntry> entries;
    EXPECT_GE(dir.size(), sizeof(header_out));
    if (dir.size() < sizeof(header_out)) {
        return entries;
    }
    header_out = read_pod<nsparse::InlineDirHeader>(dir.data());
    const size_t base = sizeof(header_out);
    EXPECT_EQ(dir.size(),
              base + header_out.n_entries * sizeof(nsparse::InlineDirEntry));
    // Bound the reads by what the buffer actually holds so a regressed writer
    // (inflated n_entries / truncated buffer) yields a clean EXPECT failure
    // instead of reading past the vector.
    const uint64_t available =
        (dir.size() - base) / sizeof(nsparse::InlineDirEntry);
    const uint64_t n = std::min<uint64_t>(header_out.n_entries, available);
    for (uint64_t i = 0; i < n; ++i) {
        entries.push_back(read_pod<nsparse::InlineDirEntry>(
            dir.data() + base + i * sizeof(nsparse::InlineDirEntry)));
    }
    return entries;
}

// Parse the block at entry.byte_off and assert every record reproduces the
// source document's component ids and value bytes exactly, in the order given.
void verify_block(const std::vector<uint8_t>& bin,
                  const nsparse::InlineDirEntry& entry,
                  const std::vector<nsparse::idx_t>& expected_docs,
                  const nsparse::SparseVectors& vectors) {
    ASSERT_LE(entry.byte_off + entry.len, bin.size());
    EXPECT_EQ(entry.reserved, 0U);  // reserved must be zero-filled
    const uint8_t* base = bin.data();
    size_t off = entry.byte_off;

    const uint32_t n_docs = read_pod<uint32_t>(base + off);
    off += sizeof(uint32_t);
    EXPECT_EQ(n_docs, entry.n_docs);
    ASSERT_EQ(n_docs, expected_docs.size());

    const size_t element_size = vectors.get_element_size();
    const auto* indptr = vectors.indptr_data();
    const auto* indices = vectors.indices_data();
    const auto* values = vectors.values_data();

    for (uint32_t i = 0; i < n_docs; ++i) {
        const uint32_t doc_id = read_pod<uint32_t>(base + off);
        off += sizeof(uint32_t);
        const uint32_t nnz = read_pod<uint32_t>(base + off);
        off += sizeof(uint32_t);

        EXPECT_EQ(doc_id, static_cast<uint32_t>(expected_docs[i]));
        ASSERT_LT(doc_id, vectors.num_vectors());  // guard before indexing
        const nsparse::idx_t start = indptr[doc_id];
        const uint32_t src_nnz =
            static_cast<uint32_t>(indptr[doc_id + 1] - start);
        ASSERT_EQ(nnz, src_nnz);

        EXPECT_EQ(0, std::memcmp(base + off, indices + start,
                                 nnz * sizeof(nsparse::term_t)))
            << "component mismatch for doc " << doc_id;
        off += nnz * sizeof(nsparse::term_t);

        EXPECT_EQ(
            0, std::memcmp(base + off,
                           values + static_cast<size_t>(start) * element_size,
                           static_cast<size_t>(nnz) * element_size))
            << "value bytes mismatch for doc " << doc_id;
        off += static_cast<size_t>(nnz) * element_size;
    }
    // The payload length recorded in the directory must be exact.
    EXPECT_EQ(off - entry.byte_off, entry.len);
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

TEST(InlineForwardIndex, HeaderAndDirectoryLayout) {
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
    nsparse::BufferedIOWriter dir;
    writer.write(lists, vectors, &bin, &dir);

    const auto header = parse_bin_header(bin.data());
    EXPECT_EQ(header.magic, nsparse::InlineForwardIndexHeader::kMagic);
    EXPECT_EQ(header.element_size, nsparse::U32);
    EXPECT_EQ(header.n_blocks, 4U);
    EXPECT_EQ(header.page_size,
              nsparse::InlineForwardIndexWriter::kDefaultPageSize);

    nsparse::InlineDirHeader dir_header{};
    auto entries = parse_dir(dir.data(), dir_header);
    EXPECT_EQ(dir_header.magic, nsparse::InlineDirHeader::kMagic);
    EXPECT_EQ(dir_header.element_size, nsparse::U32);
    EXPECT_EQ(dir_header.n_lists, 2U);
    EXPECT_EQ(dir_header.n_entries, 4U);
    EXPECT_EQ(dir_header.page_size,
              nsparse::InlineForwardIndexWriter::kDefaultPageSize);
    ASSERT_EQ(entries.size(), 4U);

    // Entries are (pl, block) in ascending order, first block page-aligned, and
    // offsets strictly increasing and page-aligned.
    const std::vector<std::pair<uint32_t, uint32_t>> expected_ids = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}};
    uint64_t prev_off = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        EXPECT_EQ(entries[i].pl, expected_ids[i].first);
        EXPECT_EQ(entries[i].block, expected_ids[i].second);
        EXPECT_EQ(entries[i].byte_off %
                      nsparse::InlineForwardIndexWriter::kDefaultPageSize,
                  0U);
        if (i == 0) {
            EXPECT_EQ(entries[i].byte_off,
                      nsparse::InlineForwardIndexWriter::kDefaultPageSize);
        } else {
            EXPECT_GT(entries[i].byte_off, prev_off);
        }
        prev_off = entries[i].byte_off;
    }

    // Each block's byte_off must equal the previous block's end rounded up.
    for (size_t i = 1; i < entries.size(); ++i) {
        const uint64_t prev_end = entries[i - 1].byte_off + entries[i - 1].len;
        EXPECT_EQ(
            entries[i].byte_off,
            nsparse::inline_align_up(
                prev_end, nsparse::InlineForwardIndexWriter::kDefaultPageSize));
    }
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
    nsparse::BufferedIOWriter dir;
    writer.write(lists, vectors, &bin, &dir);

    nsparse::InlineDirHeader dir_header{};
    auto entries = parse_dir(dir.data(), dir_header);
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
    nsparse::BufferedIOWriter dir;
    writer.write(lists, vectors, &bin, &dir);

    const auto header = parse_bin_header(bin.data());
    EXPECT_EQ(header.element_size, nsparse::U16);
    nsparse::InlineDirHeader dir_header{};
    auto entries = parse_dir(dir.data(), dir_header);
    EXPECT_EQ(dir_header.element_size, nsparse::U16);
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
    nsparse::BufferedIOWriter dir;
    writer.write(lists, vectors, &bin, &dir);

    const auto header = parse_bin_header(bin.data());
    EXPECT_EQ(header.element_size, nsparse::U8);
    nsparse::InlineDirHeader dir_header{};
    auto entries = parse_dir(dir.data(), dir_header);
    EXPECT_EQ(dir_header.element_size, nsparse::U8);
    ASSERT_EQ(entries.size(), 2U);
    verify_block(bin.data(), entries[0], {0, 1}, vectors);
    verify_block(bin.data(), entries[1], {2}, vectors);
}

TEST(InlineForwardIndex, SingleDocBlock) {
    auto vectors = create_float_vectors({{0, 1, 2}}, {{1.0F, 2.0F, 3.0F}}, 4);
    auto lists = build_lists({{{0}}}, vectors);

    nsparse::InlineForwardIndexWriter writer;
    nsparse::BufferedIOWriter bin;
    nsparse::BufferedIOWriter dir;
    writer.write(lists, vectors, &bin, &dir);

    const auto header = parse_bin_header(bin.data());
    EXPECT_EQ(header.n_blocks, 1U);
    nsparse::InlineDirHeader dir_header{};
    auto entries = parse_dir(dir.data(), dir_header);
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
    nsparse::BufferedIOWriter dir;
    writer.write(lists, vectors, &bin, &dir);

    nsparse::InlineDirHeader dir_header{};
    auto entries = parse_dir(dir.data(), dir_header);
    ASSERT_EQ(entries.size(), 2U);
    EXPECT_EQ(entries[1].n_docs, 0U);
    EXPECT_EQ(entries[1].len, sizeof(uint32_t));  // just the n_docs prefix
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
    nsparse::BufferedIOWriter dir;
    writer.write(lists, vectors, &bin, &dir);

    const auto header = parse_bin_header(bin.data());
    EXPECT_EQ(header.n_blocks, 1U);
    nsparse::InlineDirHeader dir_header{};
    auto entries = parse_dir(dir.data(), dir_header);
    EXPECT_EQ(dir_header.n_lists, 2U);
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
    nsparse::BufferedIOWriter dir;
    writer.write(lists, vectors, &bin, &dir);

    const auto header = parse_bin_header(bin.data());
    EXPECT_EQ(header.n_blocks, 0U);
    // Header is padded out to a full page even with no blocks, and the pad
    // bytes are zero (the artifact must be byte-reproducible).
    EXPECT_EQ(bin.data().size(),
              nsparse::InlineForwardIndexWriter::kDefaultPageSize);
    expect_zero_range(bin.data(), sizeof(nsparse::InlineForwardIndexHeader),
                      nsparse::InlineForwardIndexWriter::kDefaultPageSize);
    nsparse::InlineDirHeader dir_header{};
    auto entries = parse_dir(dir.data(), dir_header);
    EXPECT_EQ(dir_header.n_lists, 0U);
    EXPECT_EQ(dir_header.n_entries, 0U);
    EXPECT_TRUE(entries.empty());
}

TEST(InlineForwardIndex, CustomPageSizeAlignsBlocks) {
    auto vectors = create_float_vectors(
        {{0, 3}, {1}, {2, 4, 6}, {0, 9}},
        {{1.0F, 2.0F}, {3.0F}, {1.5F, 2.5F, 3.5F}, {4.0F, 5.0F}}, 10);
    auto lists = build_lists({{{0, 1}, {2, 3}}}, vectors);

    nsparse::InlineForwardIndexWriter writer(64);  // small power-of-two page
    nsparse::BufferedIOWriter bin;
    nsparse::BufferedIOWriter dir;
    writer.write(lists, vectors, &bin, &dir);

    const auto header = parse_bin_header(bin.data());
    EXPECT_EQ(header.page_size, 64U);
    nsparse::InlineDirHeader dir_header{};
    auto entries = parse_dir(dir.data(), dir_header);
    ASSERT_EQ(entries.size(), 2U);
    for (const auto& e : entries) {
        EXPECT_EQ(e.byte_off % 64U, 0U);
    }
    EXPECT_EQ(entries[0].byte_off, 64U);
    EXPECT_EQ(
        entries[1].byte_off,
        nsparse::inline_align_up(entries[0].byte_off + entries[0].len, 64));
    verify_block(bin.data(), entries[0], {0, 1}, vectors);
    verify_block(bin.data(), entries[1], {2, 3}, vectors);
}

TEST(InlineForwardIndex, MultiPageBlockSpansAndPadsCorrectly) {
    // Six docs, each nnz=2 -> record = 8 + 2*2 + 2*4 = 20 bytes.
    auto vectors =
        create_float_vectors({{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}},
                             {{1.0F, 2.0F},
                              {3.0F, 4.0F},
                              {5.0F, 6.0F},
                              {7.0F, 8.0F},
                              {9.0F, 10.0F},
                              {11.0F, 12.0F}},
                             10);
    // block0 has 4 docs -> payload 4 + 4*20 = 84 bytes, spanning multiple
    // 32-byte pages; block1 has 2 docs -> payload 4 + 2*20 = 44 bytes.
    auto lists = build_lists({{{0, 1, 2, 3}, {4, 5}}}, vectors);

    nsparse::InlineForwardIndexWriter writer(32);
    nsparse::BufferedIOWriter bin;
    nsparse::BufferedIOWriter dir;
    writer.write(lists, vectors, &bin, &dir);

    nsparse::InlineDirHeader dir_header{};
    auto entries = parse_dir(dir.data(), dir_header);
    ASSERT_EQ(entries.size(), 2U);

    // block0: page-aligned start, exact multi-page payload length.
    EXPECT_EQ(entries[0].byte_off, 32U);
    EXPECT_EQ(entries[0].len, 4U + 4U * 20U);  // 84, > 2 pages
    EXPECT_GT(entries[0].len, 64U);
    // block1 begins at the next page boundary after block0's payload, which is
    // more than one page past block0's start.
    const uint64_t expected1 =
        nsparse::inline_align_up(entries[0].byte_off + entries[0].len, 32);
    EXPECT_EQ(entries[1].byte_off, expected1);
    EXPECT_EQ(entries[1].byte_off % 32U, 0U);
    EXPECT_GT(entries[1].byte_off - entries[0].byte_off, 32U);
    EXPECT_EQ(entries[1].len, 4U + 2U * 20U);  // 44

    // Inter-block padding bytes are zero.
    expect_zero_range(bin.data(), entries[0].byte_off + entries[0].len,
                      entries[1].byte_off);

    verify_block(bin.data(), entries[0], {0, 1, 2, 3}, vectors);
    verify_block(bin.data(), entries[1], {4, 5}, vectors);

    // The file ends exactly at the last block's payload rounded up to a page,
    // and every byte of the final trailing pad is zero (byte-reproducible).
    const uint64_t total =
        nsparse::inline_align_up(entries[1].byte_off + entries[1].len, 32);
    EXPECT_EQ(bin.data().size(), total);
    expect_zero_range(bin.data(), entries[1].byte_off + entries[1].len,
                      bin.data().size());
}

TEST(InlineForwardIndex, ZeroNnzDocumentIsStored) {
    // doc 0 has an empty stored vector (nnz == 0); doc 1 is normal.
    auto vectors = create_float_vectors({{}, {0}}, {{}, {1.0F}}, 4);
    auto lists = build_lists({{{0, 1}}}, vectors);

    nsparse::InlineForwardIndexWriter writer;
    nsparse::BufferedIOWriter bin;
    nsparse::BufferedIOWriter dir;
    writer.write(lists, vectors, &bin, &dir);

    nsparse::InlineDirHeader dir_header{};
    auto entries = parse_dir(dir.data(), dir_header);
    ASSERT_EQ(entries.size(), 1U);
    // n_docs(4) + doc0[doc_id(4)+nnz(4)] +
    // doc1[doc_id(4)+nnz(4)+comps(2)+val(4)]
    EXPECT_EQ(entries[0].len, 4U + 8U + 14U);
    verify_block(bin.data(), entries[0], {0, 1}, vectors);
}

TEST(InlineForwardIndex, RejectsUnsupportedElementSize) {
    // U64 (8) is in the ElementSize enum but not a supported inline width.
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U64, .dimension = 4});
    std::vector<nsparse::InvertedListClusters> lists;

    nsparse::InlineForwardIndexWriter writer;
    nsparse::BufferedIOWriter bin;
    nsparse::BufferedIOWriter dir;
    EXPECT_THROW(writer.write(lists, vectors, &bin, &dir),
                 std::invalid_argument);
}

TEST(InlineForwardIndex, RejectsNullWriter) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 4});
    std::vector<nsparse::InvertedListClusters> lists;

    nsparse::InlineForwardIndexWriter writer;
    nsparse::BufferedIOWriter dir;
    EXPECT_THROW(writer.write(lists, vectors, nullptr, &dir),
                 std::invalid_argument);
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
    nsparse::BufferedIOWriter dir;
    EXPECT_THROW(writer.write(lists, vectors_small, &bin, &dir),
                 std::out_of_range);
}

TEST(InlineForwardIndex, PackedLayoutHasNoPadding) {
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
    nsparse::BufferedIOWriter dir;
    writer.write(lists, vectors, &bin, &dir);

    // Effective alignment 1 is recorded in both headers.
    const auto header = parse_bin_header(bin.data());
    EXPECT_EQ(header.page_size, 1U);
    EXPECT_EQ(header.n_blocks, 4U);
    nsparse::InlineDirHeader dir_header{};
    auto entries = parse_dir(dir.data(), dir_header);
    EXPECT_EQ(dir_header.page_size, 1U);
    ASSERT_EQ(entries.size(), 4U);

    // Blocks are back-to-back: the first immediately follows the (unpadded)
    // header and each subsequent block begins exactly where the previous ended.
    uint64_t running = sizeof(nsparse::InlineForwardIndexHeader);
    for (const auto& e : entries) {
        EXPECT_EQ(e.byte_off, running);
        running += e.len;
    }
    // The whole file is header + payloads, with no padding anywhere.
    EXPECT_EQ(bin.data().size(), running);

    verify_block(bin.data(), entries[0], layout[0][0], vectors);
    verify_block(bin.data(), entries[1], layout[0][1], vectors);
    verify_block(bin.data(), entries[2], layout[1][0], vectors);
    verify_block(bin.data(), entries[3], layout[1][1], vectors);

    // Packed is strictly smaller than the page-aligned layout for the same
    // data.
    nsparse::InlineForwardIndexWriter padded_writer;  // default: page-aligned
    nsparse::BufferedIOWriter padded_bin;
    nsparse::BufferedIOWriter padded_dir;
    padded_writer.write(lists, vectors, &padded_bin, &padded_dir);
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
    nsparse::BufferedIOWriter dir;
    writer.write(lists, vectors, &bin, &dir);

    const auto header = parse_bin_header(bin.data());
    EXPECT_EQ(header.element_size, nsparse::U8);
    EXPECT_EQ(header.page_size, 1U);
    nsparse::InlineDirHeader dir_header{};
    auto entries = parse_dir(dir.data(), dir_header);
    EXPECT_EQ(dir_header.element_size, nsparse::U8);
    EXPECT_EQ(dir_header.n_lists, 2U);
    ASSERT_EQ(entries.size(), 3U);

    // Back-to-back from just after the header, no padding anywhere -- even with
    // an empty block (len == 4) and a zero-nnz record in the mix.
    uint64_t running = sizeof(nsparse::InlineForwardIndexHeader);
    for (const auto& e : entries) {
        EXPECT_EQ(e.byte_off, running);
        running += e.len;
    }
    EXPECT_EQ(bin.data().size(), running);
    EXPECT_EQ(entries[1].n_docs, 0U);             // the empty block
    EXPECT_EQ(entries[1].len, sizeof(uint32_t));  // just the n_docs prefix

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
