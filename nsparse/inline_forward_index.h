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
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/io/io.h"
#include "nsparse/sparse_vectors.h"

namespace nsparse {

/**
 * On-disk "inline" (block-contiguous) forward index: the byte format and the
 * build-time writer. Each document's vector is copied contiguously into every
 * block (cluster) it belongs to, so a block is one contiguous read via mmap at
 * query time. Two files, host byte order, rebuildable (no cross-version
 * guarantee). inline.bin: a header, then per block a u32 n_docs and n_docs
 * records of [doc_id u32][nnz u32][comps u16 x nnz][vals element_size x nnz].
 * inline.bin.dir: a header, then one InlineDirEntry per block mapping
 * (pl, block) to a byte range in inline.bin. Records hold exactly what
 * compute_similarity reads and have no per-field padding, so a reader must use
 * unaligned/memcpy loads.
 */

// Block placement: page-aligned (padded, mmap-friendly; default) or packed
// (back-to-back, no padding). The header page_size records it (1 when packed).
enum class InlineLayout : uint8_t { kPageAligned, kPacked };

// Header of inline.bin (padded to page_size when page-aligned; none when
// packed).
struct InlineForwardIndexHeader {
    static constexpr uint32_t kMagic = 0x4946534E;  // "NSFI"

    uint32_t magic;
    uint32_t element_size;  // 1, 2, or 4
    uint64_t n_blocks;
    uint64_t page_size;  // effective alignment; 1 when packed
};
static_assert(sizeof(InlineForwardIndexHeader) == 24);
static_assert(std::is_standard_layout_v<InlineForwardIndexHeader>);
static_assert(std::is_trivially_copyable_v<InlineForwardIndexHeader>);

// Header of inline.bin.dir, followed by n_entries InlineDirEntry.
struct InlineDirHeader {
    static constexpr uint32_t kMagic = 0x4446534E;  // "NSFD"

    uint32_t magic;
    uint32_t element_size;
    uint64_t n_lists;
    uint64_t n_entries;  // == .bin n_blocks
    uint64_t page_size;
};
static_assert(sizeof(InlineDirHeader) == 32);
static_assert(std::is_standard_layout_v<InlineDirHeader>);
static_assert(std::is_trivially_copyable_v<InlineDirHeader>);

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

// Round offset up to the next multiple of alignment (> 0).
inline uint64_t inline_align_up(uint64_t offset, uint64_t alignment) {
    return ((offset + alignment - 1) / alignment) * alignment;
}

// Build-time writer for the inline forward-index files. Reusable across writes.
class InlineForwardIndexWriter {
public:
    static constexpr uint64_t kDefaultPageSize = 4096;  // common OS page size

    // page_size is the block alignment when page-aligned (power of two, at
    // least sizeof(InlineForwardIndexHeader)); ignored when packed.
    explicit InlineForwardIndexWriter(
        uint64_t page_size = kDefaultPageSize,
        InlineLayout layout = InlineLayout::kPageAligned);

    /**
     * Write inline.bin and inline.bin.dir: one record per member doc of every
     * block, copying element_size-wide value bytes from `vectors` (float and
     * quantized alike). `lists` must be built/summarized.
     *
     * Throws std::invalid_argument (null sink, or element_size not in {1,2,4})
     * or std::out_of_range (doc id outside [0, num_vectors)); may leave partial
     * output in the sinks.
     */
    void write(const std::vector<InvertedListClusters>& lists,
               const SparseVectors& vectors, IOWriter* bin_writer,
               IOWriter* dir_writer) const;

    uint64_t page_size() const { return page_size_; }
    InlineLayout layout() const { return layout_; }

    // Effective block alignment: page_size when page-aligned, else 1 (packed).
    uint64_t alignment() const {
        return layout_ == InlineLayout::kPageAligned ? page_size_ : 1;
    }

private:
    uint64_t page_size_;
    InlineLayout layout_;
};

}  // namespace nsparse

#endif  // INLINE_FORWARD_INDEX_H
