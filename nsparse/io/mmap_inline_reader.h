/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef MMAP_INLINE_READER_H
#define MMAP_INLINE_READER_H

#include <cstddef>
#include <cstdint>

#include "nsparse/inline_forward_index.h"
#include "nsparse/types.h"
#include "nsparse/utils/buf.h"
#include "nsparse/utils/mmap_file.h"

// Internal query-time machinery; not exposed to Python/JNI, hence detail.
namespace nsparse::detail {

// A read-only, in-place view of one block's structure-of-arrays (a within-block
// CSR). Every array points into the file mapping and is aligned for its element
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

// Query-time reader for the inline forward-index file (format in
// inline_forward_index.h). Maps the file read-only through MmapFile and copies
// the directory into RAM for O(1) (pl, block) lookup. Move-only.
//
// A block's sub-arrays are laid out and aligned so they are handed back as
// typed pointers into the mapping (no copy); block() validates the block's
// internal offsets against its recorded length before returning the view.
//
// The constructor throws std::invalid_argument on a null path (a programming
// error) and std::runtime_error on an unopenable or malformed file. block()
// likewise throws std::runtime_error if a looked-up block's interior is corrupt
// (fail-closed: it never returns an out-of-bounds view).
class MmapInlineReader {
public:
    explicit MmapInlineReader(const char* path);

    // Explicit, not defaulted: Buf's move keeps the source's size()/data()
    // (only its owner moves), so the moved-from members are reset below to stay
    // inert.
    MmapInlineReader(MmapInlineReader&& other) noexcept;
    MmapInlineReader& operator=(MmapInlineReader&& other) noexcept;
    MmapInlineReader(const MmapInlineReader&) = delete;
    MmapInlineReader& operator=(const MmapInlineReader&) = delete;

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
    void load_directory();  // validate header/trailer, load directory to RAM
    // Validate a block's internal structure and build an in-place SoA view.
    BlockView view_block(const InlineDirEntry& entry) const;

    MmapFile mapped_file_;  // read-only file mapping the blocks live in
    size_t element_size_ = 0;
    uint64_t page_size_ = 0;
    Buf<InlineDirEntry> entries_;  // directory copied into RAM
    Buf<uint64_t> list_offset_;    // start of each list in entries_
};

}  // namespace nsparse::detail

#endif  // MMAP_INLINE_READER_H
