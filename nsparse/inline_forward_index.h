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

#include <cstddef>
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
 *
 * A doc's vector is duplicated into every block that holds it -- about 13 copies
 * at the standard parameters -- which is what makes a block one contiguous read
 * and also what makes the section the bulk of the file. The header's
 * `max_doc_nnz` bounds the *extra* copies: when it is non-zero, one copy of each
 * doc stays whole and every other copy keeps only its largest max_doc_nnz
 * components. Blocks are then cheaper to score but their scores are lower bounds,
 * so a reader that sees max_doc_nnz != 0 must re-score its best candidates
 * against the whole copies (DocLocator finds them) before trusting the ranking.
 * max_doc_nnz == 0 means every copy is whole and no re-scoring is needed.
 */

// Locates the one copy of a doc's vector that is never truncated. When
// posting_list == kRemainder the vector is row `block` of the remainder store
// (docs that no block holds); otherwise it is slot `slot` of inline-forward
// block (posting_list, block). Serialized as a per-doc array, so a format struct
// rather than a search-side detail.
struct DocLocator {
    uint32_t posting_list;
    uint32_t block;
    uint32_t slot;
    static constexpr uint32_t kRemainder = UINT32_MAX;
};
static_assert(sizeof(DocLocator) == 12,
              "DocLocator is borrowed from the mapping");
static_assert(std::is_standard_layout_v<DocLocator>);
static_assert(std::is_trivially_copyable_v<DocLocator>);

// Block placement: page-aligned (padded to page_size) or packed (blocks
// back-to-back on the minimum kMinBlockAlign boundary that keeps the per-block
// arrays readable in place; the default). The header page_size records the
// effective alignment, and a reader honours whichever the file declares, so the
// two are the same format and either loads in a build that writes the other.
//
// Packed is the default because a block is far smaller than a page. At the
// standard SEISMIC parameters a block holds about ten documents -- a ~4.5 KB
// payload -- so page alignment spent about as many bytes on padding as on
// values: a third of a `disk_seismic_sq` file over MS MARCO, 24.3 GiB of 74.2.
// Those bytes were never read, only stored and cached, which is why packing them
// out costs no query CPU and shows up as a smaller file and a smaller resident
// set rather than as fewer page faults.
enum class InlineLayout : uint8_t { kPageAligned, kPacked };

// Minimum block-start alignment. Blocks begin on this boundary even when
// packed, so doc_id/off (u32), comps (u16), and vals (<= 4-byte) sub-arrays all
// land on an address their element type can be loaded from.
inline constexpr uint64_t kMinBlockAlign = 8;

// Header at the start of the file (padded to page_size when page-aligned; only
// to kMinBlockAlign when packed).
struct InlineForwardIndexHeader {
    uint32_t element_size;  // 1, 2, or 4
    // Components kept by a doc's truncated copies; 0 = every copy is whole.
    // Sits in what was padding after element_size, so the header keeps its size
    // and a file written before this field existed reads back as 0 (the writer
    // value-initializes the header, so those bytes were already zero).
    uint32_t max_doc_nnz;
    uint64_t n_blocks;
    uint64_t page_size;  // effective block alignment (>= kMinBlockAlign)
};
static_assert(sizeof(InlineForwardIndexHeader) == 24);
static_assert(offsetof(InlineForwardIndexHeader, n_blocks) == 8,
              "max_doc_nnz must occupy the old padding, not shift the fields");
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
