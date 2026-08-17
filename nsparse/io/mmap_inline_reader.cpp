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

namespace nsparse::detail {

MmapInlineReader::MmapInlineReader(const char* path) {
    throw_if_null(path, "MmapInlineReader: path must not be null");
    mapped_file_.open(path);  // throws on open/stat failure
    load_directory();         // throws on a corrupt file; ~MmapFile unmaps
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

void MmapInlineReader::load_directory() {
    const uint8_t* map = mapped_file_.data();
    const size_t map_size = mapped_file_.size();
    if (map_size <
        sizeof(InlineForwardIndexHeader) + sizeof(InlineForwardIndexTrailer)) {
        throw std::runtime_error("MmapInlineReader: file too small");
    }
    // Sub-arrays are read as typed pointers at 8-aligned block offsets, so the
    // mapping base must be >= 8-aligned too (MmapFile always page-aligns).
    if (reinterpret_cast<uintptr_t>(map) % kMinBlockAlign != 0) {
        throw std::runtime_error("MmapInlineReader: mapping is under-aligned");
    }

    InlineForwardIndexHeader header;
    std::memcpy(&header, map, sizeof(header));
    if (header.element_size != 1 && header.element_size != 2 &&
        header.element_size != 4) {
        throw std::runtime_error("MmapInlineReader: bad element_size");
    }
    element_size_ = header.element_size;
    page_size_ = header.page_size;
    // Effective block alignment: a power of two, at least kMinBlockAlign.
    if (page_size_ < kMinBlockAlign || (page_size_ & (page_size_ - 1)) != 0) {
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

    // n_lists comes from the file; cap it in the posting-list space (pl derives
    // from term_t) so num_lists + 1 can't overflow and the allocation is
    // bounded.
    if (num_lists > (static_cast<uint64_t>(1) << (8 * sizeof(term_t)))) {
        throw std::runtime_error("MmapInlineReader: implausible n_lists");
    }
    // Validate each entry, enforce the (pl, block) grouping, and count blocks
    // per list; the prefix sum then gives each list's start in entries_.
    std::vector<uint64_t> list_offset(num_lists + 1, 0);
    bool started = false;
    uint32_t prev_pl = 0;
    uint32_t prev_block = 0;
    for (const InlineDirEntry& entry : entries) {
        if (entry.pl >= num_lists || entry.len < sizeof(uint32_t) ||
            entry.byte_off % kMinBlockAlign != 0 ||
            entry.byte_off < sizeof(InlineForwardIndexHeader) ||
            entry.byte_off > trailer.dir_offset ||
            trailer.dir_offset - entry.byte_off < entry.len) {
            throw std::runtime_error(
                "MmapInlineReader: corrupt directory entry");
        }
        // Entries must be grouped by (pl ascending, block ascending, gap-free)
        // so (pl, block) -> entry is the O(1) prefix-sum lookup block() relies
        // on. A new pl (or the first entry) restarts block numbering at 0.
        if (started && entry.pl < prev_pl) {
            throw std::runtime_error(
                "MmapInlineReader: directory not ordered by (pl, block)");
        }
        const uint32_t expected_block =
            (!started || entry.pl != prev_pl) ? 0 : prev_block + 1;
        if (entry.block != expected_block) {
            throw std::runtime_error(
                "MmapInlineReader: directory not ordered by (pl, block)");
        }
        started = true;
        prev_pl = entry.pl;
        prev_block = entry.block;
        ++list_offset[entry.pl + 1];
    }
    for (uint64_t i = 0; i < num_lists; ++i) {
        list_offset[i + 1] += list_offset[i];
    }

    entries_ = Buf<InlineDirEntry>::own(std::move(entries));
    list_offset_ = Buf<uint64_t>::own(std::move(list_offset));
}

// Reads are hand-rolled rather than layered on MmapCursor/borrow_padded: the
// block is random-access (located via the directory, not a sequential stream),
// and validating it with one `layout.end == len` equality is tighter than
// per-array cursor bounds checks. The sub-arrays are still borrowed in place.
BlockView MmapInlineReader::view_block(const InlineDirEntry& entry) const {
    const uint8_t* base = mapped_file_.data() + entry.byte_off;
    const uint64_t len =
        entry.len;  // <= dir_offset - byte_off (in the mapping)

    uint32_t n_docs;
    std::memcpy(&n_docs, base, sizeof(uint32_t));  // len >= 4 checked at load
    if (n_docs != entry.n_docs) {
        throw std::runtime_error("MmapInlineReader: block n_docs mismatch");
    }
    // doc_id[] and off[] must fit before the offsets can be read; the comps
    // offset is where they end and is independent of total_nnz.
    const InlineBlockOffsets hdr =
        inline_block_offsets(n_docs, 0, element_size_);
    if (hdr.comps > len) {
        throw std::runtime_error(
            "MmapInlineReader: block header overruns block");
    }
    const auto* doc_ids = reinterpret_cast<const uint32_t*>(base + hdr.doc_ids);
    const auto* offsets = reinterpret_cast<const uint32_t*>(base + hdr.off);
    if (offsets[0] != 0) {
        throw std::runtime_error(
            "MmapInlineReader: block offsets must start 0");
    }
    for (uint32_t i = 0; i < n_docs; ++i) {
        if (offsets[i + 1] < offsets[i]) {
            throw std::runtime_error(
                "MmapInlineReader: block offsets not monotonic");
        }
    }
    const uint64_t total_nnz = offsets[n_docs];
    const InlineBlockOffsets layout =
        inline_block_offsets(n_docs, total_nnz, element_size_);
    if (layout.end != len) {
        throw std::runtime_error("MmapInlineReader: block length mismatch");
    }

    BlockView view;
    view.n_docs = n_docs;
    view.doc_ids = doc_ids;
    view.offsets = offsets;
    view.comps = reinterpret_cast<const term_t*>(base + layout.comps);
    view.vals = base + layout.vals;
    return view;
}

uint64_t MmapInlineReader::num_blocks_in_list(uint32_t pl) const {
    if (static_cast<uint64_t>(pl) + 1 >= list_offset_.size()) {
        return 0;
    }
    return list_offset_[pl + 1] - list_offset_[pl];
}

BlockView MmapInlineReader::block(uint32_t pl, uint32_t block) const {
    // An out-of-range pl makes num_blocks_in_list(pl) return 0, so this returns
    // an absent view before list_offset_[pl] is ever indexed.
    if (block >= num_blocks_in_list(pl)) {
        return {};
    }
    const InlineDirEntry& entry = entries_[list_offset_[pl] + block];
    if (entry.pl != pl || entry.block != block) {
        return {};  // directory not in the expected (pl, block) order
    }
    return view_block(entry);
}

}  // namespace nsparse::detail
