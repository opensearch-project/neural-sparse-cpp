/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef INLINE_FORWARD_INDEX_IO_H
#define INLINE_FORWARD_INDEX_IO_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/inline_forward_index.h"
#include "nsparse/io/io.h"
#include "nsparse/io/mmap_io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/buf.h"
#include "nsparse/utils/mmap_cursor.h"

// Internal serialization machinery; not exposed to Python/JNI, hence detail.
namespace nsparse::detail {

// A read-only, in-place view of one block's structure-of-arrays (a within-block
// CSR). Every array points into the mapping and is aligned for its element
// type, so it can be read directly. `doc_ids == nullptr` means the (pl, block)
// does not exist; a present block with n_docs == 0 has non-null (empty) arrays.
struct BlockView {
    uint32_t n_docs = 0;
    const uint32_t* doc_ids = nullptr;  // [n_docs] global doc ids
    const uint32_t* offsets = nullptr;  // [n_docs + 1] within-block CSR offsets
    const term_t* comps = nullptr;      // [offsets[n_docs]] component ids
    const uint8_t* vals = nullptr;      // element_size bytes * offsets[n_docs]

    bool absent() const { return doc_ids == nullptr; }

    // Component count / component ids / value bytes of the i-th doc (i <
    // n_docs).
    uint32_t nnz(uint32_t i) const { return offsets[i + 1] - offsets[i]; }
    const term_t* doc_comps(uint32_t i) const { return comps + offsets[i]; }
    const uint8_t* doc_vals(uint32_t i, size_t element_size) const {
        return vals + static_cast<size_t>(offsets[i]) * element_size;
    }
};

// The inline forward-index as a serialization component (format in
// inline_forward_index.h). Construct with the source lists + vectors to write
// it (serialize); default-construct to read it. On read the directory is copied
// into RAM and block payloads are borrowed in place, giving O(1) (pl, block)
// lookup. Move-only.
//
// The serialized form is self-delimiting: serialize() prefixes the section with
// its byte length, and mmap_deserialize() reads that length, so (like its
// sibling components) it advances a shared cursor exactly past itself -- no
// caller-scoped subcursor required. The mapping must outlive this object.
// mmap_deserialize throws std::runtime_error on a malformed section
// (fail-closed: block() never returns an out-of-bounds view).
//
// deserialize(IOReader*) is unsupported: the forward index is disk-resident, so
// it is only read by borrowing from a mapping.
class InlineForwardIndex : public MmapSerializable {
public:
    static constexpr uint64_t kDefaultPageSize = 4096;  // common OS page size

    InlineForwardIndex() = default;  // read mode; fill via mmap_deserialize

    // Write mode. `lists`/`vectors` must outlive any serialize() call.
    // page_size: block alignment when page-aligned (power of two, >= header
    // size); ignored when packed.
    InlineForwardIndex(const std::vector<InvertedListClusters>& lists,
                       const SparseVectors& vectors,
                       uint64_t page_size = kDefaultPageSize,
                       InlineLayout layout = InlineLayout::kPageAligned);

    // Explicit, not defaulted: Buf's move keeps the source's size()/data()
    // (only its owner moves), so the moved-from members are reset to stay
    // inert.
    InlineForwardIndex(InlineForwardIndex&& other) noexcept;
    InlineForwardIndex& operator=(InlineForwardIndex&& other) noexcept;
    InlineForwardIndex(const InlineForwardIndex&) = delete;
    InlineForwardIndex& operator=(const InlineForwardIndex&) = delete;

    void serialize(IOWriter* writer) const override;
    void deserialize(IOReader* reader) override;  // unsupported (throws)
    void mmap_deserialize(MmapCursor* cursor) override;

    size_t element_size() const { return element_size_; }
    uint64_t num_blocks() const { return entries_.size(); }
    // list_offset_ holds num_lists + 1 prefix-sum entries; empty when
    // moved-from.
    uint64_t num_lists() const {
        return list_offset_.empty() ? 0 : list_offset_.size() - 1;
    }
    uint64_t page_size() const { return page_size_; }
    uint64_t num_blocks_in_list(uint32_t pl) const;

    BlockView block(uint32_t pl, uint32_t block) const;

private:
    // page_size when page-aligned, else kMinBlockAlign (packed).
    uint64_t write_alignment() const {
        return write_layout_ == InlineLayout::kPageAligned ? write_page_size_
                                                           : kMinBlockAlign;
    }
    // Serialize the section body (header, blocks, directory, trailer); the
    // public serialize() wraps this with a length prefix.
    void write_body(IOWriter* writer) const;
    // Validate the section at [base, base + section_len) and load its directory
    // into RAM; sets
    // element_size_/page_size_/entries_/list_offset_/block_base_.
    void index_section(const uint8_t* base, size_t section_len);
    // Validate a block's interior and build an in-place SoA view.
    BlockView view_block(const InlineDirEntry& entry) const;

    // Write mode (borrowed source; null in read mode).
    const std::vector<InvertedListClusters>* lists_ = nullptr;
    const SparseVectors* vectors_ = nullptr;
    uint64_t write_page_size_ = kDefaultPageSize;
    InlineLayout write_layout_ = InlineLayout::kPageAligned;

    // Read mode.
    const uint8_t* block_base_ =
        nullptr;  // section start (blocks at +byte_off)
    size_t element_size_ = 0;
    uint64_t page_size_ = 0;
    Buf<InlineDirEntry> entries_;  // directory copied into RAM
    Buf<uint64_t> list_offset_;    // start of each list in entries_
};

}  // namespace nsparse::detail

#endif  // INLINE_FORWARD_INDEX_IO_H
