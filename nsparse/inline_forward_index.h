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

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/io/io.h"
#include "nsparse/sparse_vectors.h"

namespace nsparse {

/**
 * On-disk "inline" (block-contiguous, SPANN-style) forward index.
 *
 * SEISMIC scores a query by selecting a small set of blocks (clusters) and, for
 * each selected block, computing the exact similarity of every member document.
 * The in-RAM engine reads each document's sparse vector from a single shared
 * forward index (`SparseVectors`) indexed by doc id, so a block scan gathers
 * its members from scattered offsets. When the forward index lives on disk that
 * scatter turns into hundreds of small random reads per query.
 *
 * The inline layout removes the scatter: it stores a full copy of each
 * document's sparse vector *contiguously inside every block it belongs to*, so
 * a whole block is one contiguous byte range that can be read (or paged in via
 * mmap) in a single sequential access. A document that appears in N blocks is
 * duplicated N times -- trading storage (~5x on MS MARCO v1) for locality.
 *
 * This component is the *build-time* writer plus the shared on-disk format
 * definition. The query-time reader (`MmapInlineReader`) and the disk-resident
 * index (`DiskSeismicIndex`) are separate components that consume this format.
 *
 * File layout
 * -----------
 * Two files are produced. Both are written in host byte order (matching the
 * rest of the repo's serialization; no byte swapping) and are *derived,
 * rebuildable* artifacts with no cross-version stability guarantee.
 *
 *   inline.bin  (memory-mapped at query time; blocks page-aligned by default,
 *               or packed back-to-back when the Packed layout is selected)
 *     [InlineForwardIndexHeader]          -- then zero-padded to `page_size`
 *     per block, starting on a `page_size` boundary (packed: no padding):
 *       [uint32 n_docs]
 *       n_docs x record:
 *         [uint32 doc_id]
 *         [uint32 nnz]
 *         [term_t comps[nnz]]            -- component (term) ids, u16
 *         [uint8  vals[nnz * element_size]]  -- value bytes, element_size wide
 *       (zero-padded up to the next `page_size` boundary; no padding when
 *        packed, where page_size == 1)
 *
 *   inline.bin.dir  (loaded fully into RAM at query time; the block directory)
 *     [InlineDirHeader]
 *     n_entries x [InlineDirEntry]        -- one per block, sorted by
 *                                            (pl ascending, block ascending)
 *
 * A record holds exactly the data `detail::compute_similarity` consumes for a
 * document -- its component ids and its `element_size`-wide value bytes -- so a
 * reader has everything needed to score without consulting the RAM forward
 * index. Values are copied raw at their stored width: float (`element_size ==
 * 4`), or scalar-quantized u16 (`2`) / u8 (`1`); quantization needs no special
 * handling here.
 *
 * Alignment contract: records are packed with no per-record padding (to keep
 * the storage cost at ~4 bytes per non-zero, the point of the layout). In the
 * default kPageAligned layout the block start (the `n_docs` prefix) is
 * page-aligned; when packed even the block start is unaligned. Every other
 * field -- the per-record `doc_id`/`nnz`, `comps`, and `vals` -- follows a
 * variable-length predecessor and is sub-word aligned in general (down to
 * 1-byte alignment for u8 values with odd nnz). Either way a reader MUST treat
 * every field as unaligned: read the fixed fields with `memcpy`, and do NOT
 * hand a `vals` pointer straight to the typed SIMD `dot_product_*` helpers
 * (which assume a naturally aligned `const float*`/`const uint16_t*`); copy a
 * document's values into an aligned scratch buffer first, or use unaligned
 * loads.
 */

/// Format version. Bump on any incompatible change to the byte layout below.
inline constexpr uint32_t kInlineForwardIndexVersion = 1;

/// Default block alignment. 4 KiB matches the common OS page size so each block
/// begins on a page boundary and pages in cleanly under mmap.
inline constexpr uint64_t kDefaultInlinePageSize = 4096;

/// Magic tag identifying an inline forward-index data file (`inline.bin`).
inline constexpr std::array<char, 4> kInlineForwardIndexMagic = {'N', 'S', 'F',
                                                                 'I'};
/// Magic tag identifying an inline forward-index directory file
/// (`inline.bin.dir`).
inline constexpr std::array<char, 4> kInlineDirMagic = {'N', 'S', 'F', 'D'};

/**
 * Fixed-size header at the start of `inline.bin`. In the default kPageAligned
 * layout it is followed by zero padding up to `page_size` so the first block
 * begins on a page boundary; when packed (`page_size == 1`) no padding is
 * written and the first block follows the header immediately. Always locate the
 * first block from the stored `page_size`, never from `kDefaultInlinePageSize`.
 */
struct InlineForwardIndexHeader {
    uint32_t magic;         // fourcc(kInlineForwardIndexMagic)
    uint32_t version;       // kInlineForwardIndexVersion
    uint32_t element_size;  // bytes per stored value: 1, 2, or 4
    uint32_t reserved;      // 0 (keeps the u64s below 8-byte aligned)
    uint64_t n_blocks;      // total blocks written across all posting lists
    uint64_t page_size;     // effective block alignment: page_size, or 1 if
                            // blocks were packed with no padding
};
static_assert(sizeof(InlineForwardIndexHeader) == 32);
static_assert(std::is_standard_layout_v<InlineForwardIndexHeader>);
static_assert(std::is_trivially_copyable_v<InlineForwardIndexHeader>);

/**
 * Fixed-size header at the start of `inline.bin.dir`, followed by `n_entries`
 * `InlineDirEntry` records.
 */
struct InlineDirHeader {
    uint32_t magic;         // fourcc(kInlineDirMagic)
    uint32_t version;       // kInlineForwardIndexVersion
    uint32_t element_size;  // mirrors the .bin header
    uint32_t reserved;      // 0
    uint64_t n_lists;       // number of posting lists covered
    uint64_t n_entries;     // number of block entries (== .bin n_blocks)
    uint64_t page_size;     // mirrors the .bin header
};
static_assert(sizeof(InlineDirHeader) == 40);
static_assert(std::is_standard_layout_v<InlineDirHeader>);
static_assert(std::is_trivially_copyable_v<InlineDirHeader>);

/**
 * One directory entry per block. Entries are written grouped by posting list
 * (ascending `pl`) and, within a list, by ascending `block`. Block indices are
 * gap-free within a list, so a reader can build an O(1) `(pl, block) -> entry`
 * index in a single pass -- but `pl` ids may be sparse, because a posting list
 * with zero clusters emits no entries; size the per-list index from the dir
 * header's `n_lists` and tolerate absent `pl`s.
 */
struct InlineDirEntry {
    uint32_t pl;     // posting list id
    uint32_t block;  // block index within the posting list
    uint64_t
        byte_off;     // byte offset of the block within inline.bin: a multiple
                      // of the header page_size in the default kPageAligned
                      // layout, but back-to-back / byte-granular when packed
                      // (page_size == 1). Use the header page_size to tell.
    uint64_t len;     // exact payload length of the block (excludes any
                      // trailing alignment padding); spans the n_docs prefix
                      // plus all records
    uint32_t n_docs;  // document count in the block (mirrors the block prefix)
    uint32_t reserved;  // 0
};
static_assert(sizeof(InlineDirEntry) == 32);
static_assert(std::is_standard_layout_v<InlineDirEntry>);
static_assert(std::is_trivially_copyable_v<InlineDirEntry>);

/// Round `offset` up to the next multiple of `alignment` (`alignment` > 0).
inline uint64_t inline_align_up(uint64_t offset, uint64_t alignment) {
    return ((offset + alignment - 1) / alignment) * alignment;
}

/**
 * Block placement in `inline.bin`.
 *
 * - `kPageAligned` (default): every block starts on a `page_size` boundary,
 *   zero-padded to the next boundary. This is what makes the file mmap-friendly
 *   -- a block pages in cleanly and can be `madvise`d independently -- at the
 *   cost of up to `page_size - 1` padding bytes per block.
 * - `kPacked`: blocks are written back-to-back with no padding (effective
 *   alignment 1). This drops the padding overhead (the storage lever from the
 *   RFC / dev-plan P3.2) but blocks no longer start on page boundaries, so the
 *   mapping is less mmap-friendly. The `.dir` still carries exact `byte_off` /
 *   `len` per block, so a reader consumes both layouts identically; the
 * header's `page_size` records the effective alignment (`page_size` when
 * aligned, `1` when packed) so a reader can tell which was used.
 */
enum class InlineLayout : uint8_t { kPageAligned, kPacked };

/**
 * Build-time writer for the inline forward-index files.
 *
 * Stateless apart from the configured page size and layout; a single instance
 * may be reused across writes.
 */
class InlineForwardIndexWriter {
public:
    /**
     * @param page_size Block alignment in bytes when `layout` is
     *        `kPageAligned`. Must be a power of two and at least
     *        `sizeof(InlineForwardIndexHeader)`. Defaults to 4 KiB. Ignored
     *        (and left unvalidated) when `layout` is `kPacked`.
     * @param layout Block placement: page-aligned (padded, default) or packed
     *        (no padding). See `InlineLayout`.
     */
    explicit InlineForwardIndexWriter(
        uint64_t page_size = kDefaultInlinePageSize,
        InlineLayout layout = InlineLayout::kPageAligned);

    /**
     * Write the inline forward index (`.bin`) and its directory sidecar
     * (`.dir`).
     *
     * For every posting list, for every block in that list, a contiguous record
     * of the block's member documents is emitted, each document's vector copied
     * from `vectors` at its stored `element_size` width. The document set and
     * ordering of a block come from `lists[pl].get_docs(block)`, mirroring the
     * scoring path exactly.
     *
     * `lists` must already be built/summarized (so `cluster_size()` reflects
     * the block count).
     *
     * Throws `std::invalid_argument` if a writer is null or `vectors`'
     * `element_size` is not one of {1, 2, 4}, and `std::out_of_range` if a
     * block references a doc id outside `[0, vectors.num_vectors())` (checked
     * in Release too; may leave partial output already written to the sinks).
     *
     * @param lists       Per-posting-list clusters, as held by the index.
     * @param vectors     Forward index; its `element_size` drives the value
     *                    width written.
     * @param bin_writer  Sink for `inline.bin`.
     * @param dir_writer  Sink for `inline.bin.dir`.
     */
    void write(const std::vector<InvertedListClusters>& lists,
               const SparseVectors& vectors, IOWriter* bin_writer,
               IOWriter* dir_writer) const;

    uint64_t page_size() const { return page_size_; }
    InlineLayout layout() const { return layout_; }

    /// Effective block alignment in bytes: `page_size` when page-aligned, else
    /// 1 (packed). This is the value stored in the file headers' `page_size`.
    uint64_t alignment() const {
        return layout_ == InlineLayout::kPageAligned ? page_size_ : 1;
    }

private:
    uint64_t page_size_;
    InlineLayout layout_;
};

}  // namespace nsparse

#endif  // INLINE_FORWARD_INDEX_H
