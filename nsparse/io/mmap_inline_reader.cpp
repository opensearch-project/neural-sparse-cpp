/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/io/mmap_inline_reader.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nsparse/inline_forward_index.h"
#include "nsparse/types.h"
#include "nsparse/utils/buf.h"
#include "nsparse/utils/checks.h"
#include "nsparse/utils/mmap_file.h"

namespace nsparse {

// The record bounds math (nnz up to 2^32-1 times record width) assumes a
// 64-bit size_t so it cannot overflow.
static_assert(sizeof(size_t) >= 8, "MmapInlineReader assumes 64-bit size_t");

InlineDocRecord read_inline_record(const uint8_t*& cursor, const uint8_t* end,
                                   size_t element_size) {
    InlineDocRecord rec;
    if (end < cursor ||
        static_cast<size_t>(end - cursor) < 2 * sizeof(uint32_t)) {
        throw std::runtime_error("MmapInlineReader: truncated record header");
    }
    std::memcpy(&rec.doc_id, cursor, sizeof(uint32_t));
    cursor += sizeof(uint32_t);
    std::memcpy(&rec.nnz, cursor, sizeof(uint32_t));
    cursor += sizeof(uint32_t);
    const size_t payload = static_cast<size_t>(rec.nnz) * sizeof(term_t) +
                           static_cast<size_t>(rec.nnz) * element_size;
    if (static_cast<size_t>(end - cursor) < payload) {
        throw std::runtime_error("MmapInlineReader: truncated record payload");
    }
    rec.comps = cursor;
    cursor += static_cast<size_t>(rec.nnz) * sizeof(term_t);
    rec.vals = cursor;
    cursor += static_cast<size_t>(rec.nnz) * element_size;
    return rec;
}

MmapInlineReader::MmapInlineReader(const char* path) {
    throw_if_null(path, "MmapInlineReader: path must not be null");
    mapped_file_.open(path);  // throws on open/stat failure
    parse();                  // throws on a corrupt file; ~MmapFile unmaps
}

// Buf's move copies the source's data()/size() and moves only its owner, so a
// naive move would leave the moved-from reader reporting stale blocks. Reset
// the moved-from members explicitly to keep it inert (num_blocks()/block() ==
// 0).
MmapInlineReader::MmapInlineReader(MmapInlineReader&& other) noexcept
    : mapped_file_(std::move(other.mapped_file_)),
      element_size_(other.element_size_),
      page_size_(other.page_size_),
      entries_(std::move(other.entries_)),
      list_offset_(std::move(other.list_offset_)) {
    other.element_size_ = 0;
    other.page_size_ = 0;
    other.entries_ = Buf<InlineDirEntry>();
    other.list_offset_ = Buf<uint64_t>();
}

MmapInlineReader& MmapInlineReader::operator=(
    MmapInlineReader&& other) noexcept {
    if (this != &other) {
        mapped_file_ = std::move(other.mapped_file_);
        element_size_ = other.element_size_;
        page_size_ = other.page_size_;
        entries_ = std::move(other.entries_);
        list_offset_ = std::move(other.list_offset_);
        other.element_size_ = 0;
        other.page_size_ = 0;
        other.entries_ = Buf<InlineDirEntry>();
        other.list_offset_ = Buf<uint64_t>();
    }
    return *this;
}

void MmapInlineReader::parse() {
    const uint8_t* map = mapped_file_.data();
    const size_t map_size = mapped_file_.size();
    if (map_size <
        sizeof(InlineForwardIndexHeader) + sizeof(InlineForwardIndexTrailer)) {
        throw std::runtime_error("MmapInlineReader: file too small");
    }

    InlineForwardIndexHeader header;
    std::memcpy(&header, map, sizeof(header));
    if (header.element_size != 1 && header.element_size != 2 &&
        header.element_size != 4) {
        throw std::runtime_error("MmapInlineReader: bad element_size");
    }
    element_size_ = header.element_size;
    page_size_ = header.page_size;
    // page_size is either 1 (packed) or a power of two at least the header
    // size.
    if (page_size_ != 1 && (page_size_ < sizeof(InlineForwardIndexHeader) ||
                            (page_size_ & (page_size_ - 1)) != 0)) {
        throw std::runtime_error("MmapInlineReader: bad page_size");
    }

    InlineForwardIndexTrailer trailer;
    std::memcpy(&trailer, map + map_size - sizeof(trailer), sizeof(trailer));
    const uint64_t num_lists = trailer.n_lists;

    // The directory occupies [dir_offset, EOF - trailer) and holds exactly
    // n_entries == header.n_blocks entries.
    const size_t entry_size = sizeof(InlineDirEntry);
    const size_t dir_end = map_size - sizeof(trailer);
    if (trailer.n_entries != header.n_blocks ||
        trailer.dir_offset < sizeof(InlineForwardIndexHeader) ||
        trailer.dir_offset > dir_end) {
        throw std::runtime_error("MmapInlineReader: corrupt directory");
    }
    // Compare via division so an oversized n_entries can't overflow a multiply.
    const size_t dir_bytes = dir_end - trailer.dir_offset;
    if (dir_bytes % entry_size != 0 ||
        dir_bytes / entry_size != trailer.n_entries) {
        throw std::runtime_error("MmapInlineReader: corrupt directory");
    }

    std::vector<InlineDirEntry> entries(trailer.n_entries);
    if (dir_bytes > 0) {
        std::memcpy(entries.data(), map + trailer.dir_offset, dir_bytes);
    }

    // n_lists (attacker-controlled) sizes the per-list index and must stay in
    // the posting-list space (pl derives from term_t); this also keeps the +1
    // below from overflowing and bounds the allocation.
    if (num_lists > (static_cast<uint64_t>(1) << (8 * sizeof(term_t)))) {
        throw std::runtime_error("MmapInlineReader: implausible n_lists");
    }
    // Entries are grouped by (pl ascending, block ascending), so the per-list
    // prefix sum gives each list's start in entries_.
    std::vector<uint64_t> list_offset(num_lists + 1, 0);
    for (const InlineDirEntry& e : entries) {
        if (e.pl >= num_lists || e.len < sizeof(uint32_t) ||
            e.byte_off < sizeof(InlineForwardIndexHeader) ||
            e.byte_off > trailer.dir_offset ||
            trailer.dir_offset - e.byte_off < e.len) {
            throw std::runtime_error(
                "MmapInlineReader: corrupt directory entry");
        }
        ++list_offset[e.pl + 1];
    }
    for (uint64_t i = 0; i < num_lists; ++i) {
        list_offset[i + 1] += list_offset[i];
    }

    entries_ = Buf<InlineDirEntry>::own(std::move(entries));
    list_offset_ = Buf<uint64_t>::own(std::move(list_offset));
}

uint64_t MmapInlineReader::num_blocks_in_list(uint32_t pl) const {
    if (static_cast<uint64_t>(pl) + 1 >= list_offset_.size()) {
        return 0;
    }
    return list_offset_[pl + 1] - list_offset_[pl];
}

BlockView MmapInlineReader::block(uint32_t pl, uint32_t block) const {
    // An out-of-range pl makes num_blocks_in_list(pl) return 0, so this returns
    // {} before list_offset_[pl] is ever indexed.
    if (block >= num_blocks_in_list(pl)) {
        return {};
    }
    const InlineDirEntry& e = entries_[list_offset_[pl] + block];
    if (e.pl != pl || e.block != block) {
        return {};  // directory not in the expected (pl, block) order
    }
    return {mapped_file_.data() + e.byte_off, e.len, e.n_docs};
}

}  // namespace nsparse
