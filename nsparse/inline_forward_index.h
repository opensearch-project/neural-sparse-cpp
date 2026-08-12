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

namespace nsparse {

/**
 * On-disk "inline" (block-contiguous) forward-index format: each doc's vector
 * is copied into every block (cluster) it belongs to, so a block is one
 * contiguous mmap read. Single file, host byte order, rebuildable (no
 * cross-version guarantee):
 *   [InlineForwardIndexHeader]  (padded to page_size when page-aligned)
 *   per block: [u32 n_docs] then n_docs records
 *              [u32 doc_id][u32 nnz][u16 comps[nnz]][vals nnz*element_size]
 *   [directory: n_entries x InlineDirEntry]  ((pl, block) -> byte range)
 *   [InlineForwardIndexTrailer]  (fixed size at EOF; locates the directory)
 * Records hold exactly what compute_similarity reads and have no per-field
 * padding, so a reader must use unaligned/memcpy loads. Writer:
 * io/inline_forward_index_writer.h; reader is P1.
 */

// Block placement: page-aligned (padded, mmap-friendly; default) or packed
// (back-to-back, no padding). The header page_size records it (1 when packed).
enum class InlineLayout : uint8_t { kPageAligned, kPacked };

// Header at the start of the file (padded to page_size when page-aligned; none
// when packed).
struct InlineForwardIndexHeader {
    uint32_t element_size;  // 1, 2, or 4
    uint64_t n_blocks;
    uint64_t page_size;  // effective alignment; 1 when packed
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
    uint64_t byte_off;  // block offset (page-aligned unless packed)
    uint64_t len;       // payload length, excluding trailing padding
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

// Round offset up to the next multiple of alignment (> 0).
inline uint64_t inline_align_up(uint64_t offset, uint64_t alignment) {
    return ((offset + alignment - 1) / alignment) * alignment;
}

}  // namespace nsparse

#endif  // INLINE_FORWARD_INDEX_H
