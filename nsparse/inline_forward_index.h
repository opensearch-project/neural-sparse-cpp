/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef INLINE_FORWARD_INDEX_H
#define INLINE_FORWARD_INDEX_H

#include <cstdint>
#include <type_traits>

// These are internal format details (not exposed to Python/JNI), hence detail.
namespace nsparse::detail {

/**
 * On-disk "inline" (block-contiguous) forward-index format: each doc's vector
 * is copied into every block (cluster) it belongs to, so a block is one
 * contiguous mmap read. Single file, host byte order, rebuildable (no
 * cross-version guarantee):
 *   [InlineForwardIndexHeader]  (padded to page_size when page-aligned)
 *   per block (structure-of-arrays, a within-block CSR):
 *       [u32 n_docs]
 *       [u32 doc_id[n_docs]]
 *       [u32 off[n_docs + 1]]           within-block offsets into comps/vals
 *       [u16 comps[total_nnz]]          term ids, total_nnz == off[n_docs]
 *       (pad to element_size)
 *       [vals[total_nnz * element_size]]
 *   [directory: n_entries x InlineDirEntry]  ((pl, block) -> byte range)
 *   [InlineForwardIndexTrailer]  (fixed size at EOF; locates the directory)
 * Blocks start on a kMinBlockAlign (or page_size) boundary and each array is
 * laid out so it can be read in place as a typed array: doc_id/off are u32,
 * comps is u16, and vals is padded up to element_size. comps and vals match the
 * element types compute_similarity consumes; off[] is a u32 within-block CSR
 * offset, capped at INT32_MAX by the writer so it converts to idx_t losslessly.
 * Component: io/inline_forward_index_io.h.
 */

// Block placement: page-aligned (padded to page_size, mmap-friendly; default)
// or packed (blocks back-to-back on the minimum kMinBlockAlign boundary that
// keeps the per-block arrays readable in place). The header page_size records
// the effective alignment.
enum class InlineLayout : uint8_t { kPageAligned, kPacked };

// Minimum block-start alignment. Blocks begin on this boundary even when
// packed, so doc_id/off (u32), comps (u16), and vals (<= 4-byte) sub-arrays all
// land on an address their element type can be loaded from.
inline constexpr uint64_t kMinBlockAlign = 8;

// Header at the start of the file (padded to page_size when page-aligned; only
// to kMinBlockAlign when packed).
struct InlineForwardIndexHeader {
    uint32_t element_size;  // 1, 2, or 4
    uint64_t n_blocks;
    uint64_t page_size;  // effective block alignment (>= kMinBlockAlign)
};
static_assert(sizeof(InlineForwardIndexHeader) == 24);
static_assert(std::is_standard_layout_v<InlineForwardIndexHeader>);
static_assert(std::is_trivially_copyable_v<InlineForwardIndexHeader>);

// One entry per block, grouped by (pl, block) ascending. Block indices are
// gap-free within a list (so (pl, block) -> entry is O(1)); pl ids may be
// sparse (an empty list emits no entries).
struct InlineDirEntry {
    uint32_t pl;
    uint32_t block;
    uint64_t byte_off;  // block offset (a multiple of the block alignment)
    uint64_t len;       // payload length, excluding trailing block padding
    uint32_t n_docs;
    uint32_t reserved;  // 0 (pads the entry to 8-byte alignment)
};
static_assert(sizeof(InlineDirEntry) == 32);
static_assert(std::is_standard_layout_v<InlineDirEntry>);
static_assert(std::is_trivially_copyable_v<InlineDirEntry>);

// Fixed-size trailer at EOF. A reader reads the last sizeof(trailer) bytes,
// then reads n_entries InlineDirEntry starting at dir_offset.
struct InlineForwardIndexTrailer {
    uint64_t dir_offset;  // byte offset where the directory array begins
    uint64_t n_entries;   // number of InlineDirEntry (== header n_blocks)
    uint64_t n_lists;     // posting lists covered
};
static_assert(sizeof(InlineForwardIndexTrailer) == 24);
static_assert(std::is_standard_layout_v<InlineForwardIndexTrailer>);
static_assert(std::is_trivially_copyable_v<InlineForwardIndexTrailer>);

// Width of a component id on the wire (term_t is uint16_t; kept independent so
// this format header carries no dependency on types.h).
inline constexpr uint64_t kInlineCompWidth = sizeof(uint16_t);

// Size of a block's leading [u32 n_docs] field: the fixed prefix before the
// doc_id[] array. Named so the layout math (inline_block_offsets) and the
// directory validator (each block's len must be at least this) agree.
inline constexpr uint64_t kInlineBlockPrefixSize = sizeof(uint32_t);

// Round offset up to the next multiple of alignment (> 0).
inline uint64_t inline_align_up(uint64_t offset, uint64_t alignment) {
    return ((offset + alignment - 1) / alignment) * alignment;
}

// Byte offsets, relative to a block's start, of each structure-of-arrays
// sub-array for a block holding `n_docs` documents and `total_nnz` nonzeros at
// `element_size`-byte values. Shared by the writer and reader so their layout
// can never drift. `end` is the block's payload length (the directory entry's
// len, excluding trailing block padding).
//
// n_docs and total_nnz are u32 on the wire (doc_id[] and off[] are uint32_t),
// so every product below stays within uint64 without an overflow check.
struct InlineBlockOffsets {
    uint64_t doc_ids;  // uint32_t[n_docs]
    uint64_t off;      // uint32_t[n_docs + 1]
    uint64_t comps;    // uint16_t[total_nnz]
    uint64_t vals;     // element_size bytes * total_nnz
    uint64_t end;
};

inline InlineBlockOffsets inline_block_offsets(uint64_t n_docs,
                                               uint64_t total_nnz,
                                               uint64_t element_size) {
    InlineBlockOffsets o{};
    o.doc_ids = kInlineBlockPrefixSize;  // just past the [u32 n_docs] prefix
    o.off = o.doc_ids + n_docs * sizeof(uint32_t);
    o.comps = o.off + (n_docs + 1) * sizeof(uint32_t);
    const uint64_t comps_end = o.comps + total_nnz * kInlineCompWidth;
    o.vals = inline_align_up(comps_end, element_size);
    o.end = o.vals + total_nnz * element_size;
    return o;
}

}  // namespace nsparse::detail

#endif  // INLINE_FORWARD_INDEX_H
