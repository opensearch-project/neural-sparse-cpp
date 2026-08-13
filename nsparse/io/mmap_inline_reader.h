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
#include "nsparse/utils/buf.h"
#include "nsparse/utils/mmap_file.h"

namespace nsparse {

// A read-only view of one block inside the mapping. `data` points at the
// block's [n_docs] prefix; `data == nullptr` means the (pl, block) does not
// exist.
struct BlockView {
    const uint8_t* data = nullptr;
    uint64_t len = 0;
    uint32_t n_docs = 0;

    // First document record (just past the n_docs prefix).
    const uint8_t* records() const {
        return data != nullptr ? data + sizeof(uint32_t) : nullptr;
    }
};

// One document record. comps/vals point into the mapping and are NOT naturally
// aligned -- read them with unaligned/memcpy loads.
struct InlineDocRecord {
    uint32_t doc_id;
    uint32_t nnz;
    const uint8_t* comps;  // nnz x uint16_t
    const uint8_t* vals;   // nnz x element_size bytes
};

// Parse the record at `cursor` and advance it past the record. `end` bounds the
// block (data + len); throws std::runtime_error if the record would overrun it.
InlineDocRecord read_inline_record(const uint8_t*& cursor, const uint8_t* end,
                                   size_t element_size);

// Query-time reader for the inline forward-index file (format in
// inline_forward_index.h). Maps the file read-only through MmapFile and copies
// the directory into RAM for O(1) (pl, block) lookup. Move-only.
//
// The constructor throws std::invalid_argument on a null path (a programming
// error) and std::runtime_error on an unopenable or malformed file.
//
// Block payloads are compact, 2-byte-packed records read with unaligned loads,
// so they are handed out as raw BlockView spans into the mapping rather than
// borrowed through MmapCursor/io_align, which assume element-aligned arrays.
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
    void parse();  // validate header/trailer, load directory, build the index

    MmapFile mapped_file_;  // read-only file mapping the blocks live in
    size_t element_size_ = 0;
    uint64_t page_size_ = 0;
    Buf<InlineDirEntry> entries_;  // directory copied into RAM
    Buf<uint64_t> list_offset_;    // start of each list in entries_
};

}  // namespace nsparse

#endif  // MMAP_INLINE_READER_H
