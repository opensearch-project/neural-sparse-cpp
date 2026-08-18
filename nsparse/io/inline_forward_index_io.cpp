/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/io/inline_forward_index_io.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/inline_forward_index.h"
#include "nsparse/io/buffered_io.h"
#include "nsparse/io/io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"
#include "nsparse/utils/mmap_cursor.h"

namespace nsparse::detail {

// The record bounds math (nnz up to 2^32-1 times record width) assumes a
// 64-bit size_t so it cannot overflow.
static_assert(sizeof(size_t) >= 8, "InlineForwardIndex assumes 64-bit size_t");

InlineForwardIndex::InlineForwardIndex(
    const std::vector<InvertedListClusters>& lists,
    const SparseVectors& vectors, uint64_t page_size, InlineLayout layout)
    : lists_(&lists),
      vectors_(&vectors),
      write_page_size_(page_size),
      write_layout_(layout) {
    if (layout == InlineLayout::kPageAligned) {
        const bool is_power_of_two =
            page_size != 0 && (page_size & (page_size - 1)) == 0;
        if (!is_power_of_two || page_size < sizeof(InlineForwardIndexHeader)) {
            throw std::invalid_argument(
                "InlineForwardIndex: page_size must be a power of two and at "
                "least sizeof(InlineForwardIndexHeader)");
        }
    }
}

// Buf's move copies the source's data()/size() and moves only its owner, so a
// naive move would leave the moved-from index reporting stale blocks. Reset the
// moved-from members explicitly to keep it inert.
InlineForwardIndex::InlineForwardIndex(InlineForwardIndex&& other) noexcept
    : lists_(other.lists_),
      vectors_(other.vectors_),
      write_page_size_(other.write_page_size_),
      write_layout_(other.write_layout_),
      block_base_(other.block_base_),
      element_size_(other.element_size_),
      page_size_(other.page_size_),
      entries_(std::move(other.entries_)),
      list_offset_(std::move(other.list_offset_)) {
    other.lists_ = nullptr;
    other.vectors_ = nullptr;
    other.block_base_ = nullptr;
    other.element_size_ = 0;
    other.page_size_ = 0;
    other.entries_ = Buf<InlineDirEntry>();
    other.list_offset_ = Buf<uint64_t>();
}

InlineForwardIndex& InlineForwardIndex::operator=(
    InlineForwardIndex&& other) noexcept {
    if (this != &other) {
        lists_ = other.lists_;
        vectors_ = other.vectors_;
        write_page_size_ = other.write_page_size_;
        write_layout_ = other.write_layout_;
        block_base_ = other.block_base_;
        element_size_ = other.element_size_;
        page_size_ = other.page_size_;
        entries_ = std::move(other.entries_);
        list_offset_ = std::move(other.list_offset_);
        other.lists_ = nullptr;
        other.vectors_ = nullptr;
        other.block_base_ = nullptr;
        other.element_size_ = 0;
        other.page_size_ = 0;
        other.entries_ = Buf<InlineDirEntry>();
        other.list_offset_ = Buf<uint64_t>();
    }
    return *this;
}

void InlineForwardIndex::serialize(IOWriter* writer) const {
    throw_if_null(writer, "writer cannot be null");
    if (lists_ == nullptr || vectors_ == nullptr) {
        throw std::logic_error(
            "InlineForwardIndex::serialize called on a read-mode index");
    }
    // Prefix the section with its own byte length so it self-delimits: the
    // reader reads the length, bounds a subcursor to the section, and advances
    // the shared cursor past it -- like its sibling components, no external
    // scoping needed.
    BufferedIOWriter body;
    write_body(&body);
    const std::vector<uint8_t>& bytes = body.data();
    uint64_t section_len = bytes.size();
    writer->write(&section_len, sizeof(section_len), 1);
    if (section_len > 0) {
        writer->write(const_cast<uint8_t*>(bytes.data()), 1, section_len);
    }
}

void InlineForwardIndex::write_body(IOWriter* writer) const {
    static_assert(sizeof(term_t) == kInlineCompWidth,
                  "component id width must match the inline format");

    const std::vector<InvertedListClusters>& lists = *lists_;
    const SparseVectors& vectors = *vectors_;
    const size_t element_size = vectors.get_element_size();
    if (element_size != 1 && element_size != 2 && element_size != 4) {
        throw std::invalid_argument(
            "InlineForwardIndex: element_size must be 1, 2, or 4, got " +
            std::to_string(element_size));
    }
    const idx_t* indptr = vectors.indptr_data();
    const term_t* indices = vectors.indices_data();
    const uint8_t* values = vectors.values_data();
    const size_t num_vectors = vectors.num_vectors();

    uint64_t n_blocks = 0;
    for (const auto& list : lists) {
        n_blocks += list.cluster_size();
    }

    const uint64_t align = write_alignment();
    const std::vector<uint8_t> zero_pad(align, 0);  // any pad < align

    InlineForwardIndexHeader header{};
    header.element_size = static_cast<uint32_t>(element_size);
    header.n_blocks = n_blocks;
    header.page_size = align;
    writer->write(&header, sizeof(header), 1);

    const uint64_t first_block_off =
        inline_align_up(sizeof(InlineForwardIndexHeader), align);
    const uint64_t header_pad =
        first_block_off - sizeof(InlineForwardIndexHeader);
    if (header_pad > 0) {
        writer->write(const_cast<uint8_t*>(zero_pad.data()), 1, header_pad);
    }
    uint64_t cur_off = first_block_off;

    std::vector<InlineDirEntry> entries;
    entries.reserve(n_blocks);
    std::vector<uint32_t> doc_ids;
    std::vector<uint32_t> offsets;
    for (size_t pl = 0; pl < lists.size(); ++pl) {
        const InvertedListClusters& list = lists[pl];
        const size_t n_clusters = list.cluster_size();
        for (size_t block = 0; block < n_clusters; ++block) {
            const std::span<const idx_t> docs = list.get_docs(block);
            const uint64_t block_off = cur_off;
            // n_docs and the within-block offsets are u32 on the wire.
            if (docs.size() > UINT32_MAX) {
                throw std::length_error(
                    "InlineForwardIndex: block has more than 2^32 docs");
            }
            const uint32_t n_docs = static_cast<uint32_t>(docs.size());

            // First pass: validate doc ids and build doc_id[]/off[] (the CSR
            // prefix sum). off[] is u32, so the running total must stay < 2^32
            // or it would wrap and desync the block from its recorded length.
            doc_ids.clear();
            offsets.assign(1, 0);
            doc_ids.reserve(n_docs);
            offsets.reserve(n_docs + 1);
            uint64_t total_nnz = 0;
            for (const idx_t doc_id : docs) {
                if (doc_id < 0 || static_cast<size_t>(doc_id) >= num_vectors) {
                    throw std::out_of_range(
                        "InlineForwardIndex: block references doc id " +
                        std::to_string(doc_id) + " outside [0, " +
                        std::to_string(num_vectors) + ")");
                }
                total_nnz +=
                    static_cast<uint64_t>(indptr[doc_id + 1] - indptr[doc_id]);
                if (total_nnz > UINT32_MAX) {
                    throw std::length_error(
                        "InlineForwardIndex: block total nnz exceeds the u32 "
                        "offset range");
                }
                doc_ids.push_back(static_cast<uint32_t>(doc_id));
                offsets.push_back(static_cast<uint32_t>(total_nnz));
            }
            const InlineBlockOffsets layout =
                inline_block_offsets(n_docs, total_nnz, element_size);

            // [n_docs][doc_id[]][off[]]
            writer->write(const_cast<uint32_t*>(&n_docs), sizeof(uint32_t), 1);
            if (n_docs > 0) {
                writer->write(doc_ids.data(), sizeof(uint32_t), n_docs);
            }
            writer->write(offsets.data(), sizeof(uint32_t), n_docs + 1);

            // comps[] then (pad to element_size) then vals[], each doc's slice
            // concatenated in block order.
            for (const idx_t doc_id : docs) {
                const idx_t start = indptr[doc_id];
                const size_t nnz = indptr[doc_id + 1] - start;
                if (nnz > 0) {
                    writer->write(const_cast<term_t*>(indices + start),
                                  sizeof(term_t), nnz);
                }
            }
            const uint64_t comps_end =
                layout.comps + total_nnz * sizeof(term_t);
            const uint64_t vals_pad = layout.vals - comps_end;
            if (vals_pad > 0) {
                writer->write(const_cast<uint8_t*>(zero_pad.data()), 1,
                              vals_pad);
            }
            for (const idx_t doc_id : docs) {
                const idx_t start = indptr[doc_id];
                const size_t nnz = indptr[doc_id + 1] - start;
                if (nnz > 0) {
                    writer->write(
                        const_cast<uint8_t*>(
                            values + static_cast<size_t>(start) * element_size),
                        1, nnz * element_size);
                }
            }

            const uint64_t block_len = layout.end;
            cur_off += block_len;

            const uint64_t padded = inline_align_up(cur_off, align);
            if (padded > cur_off) {
                writer->write(const_cast<uint8_t*>(zero_pad.data()), 1,
                              padded - cur_off);
                cur_off = padded;
            }

            InlineDirEntry entry{};
            entry.pl = static_cast<uint32_t>(pl);
            entry.block = static_cast<uint32_t>(block);
            entry.byte_off = block_off;
            entry.len = block_len;
            entry.n_docs = n_docs;
            entry.reserved = 0;
            entries.push_back(entry);
        }
    }

    // Directory then trailer (unaligned; the reader copies the dir to RAM).
    const uint64_t dir_offset = cur_off;
    if (!entries.empty()) {
        writer->write(entries.data(), sizeof(InlineDirEntry), entries.size());
    }
    InlineForwardIndexTrailer trailer{};
    trailer.dir_offset = dir_offset;
    trailer.n_entries = entries.size();
    trailer.n_lists = lists.size();
    writer->write(&trailer, sizeof(trailer), 1);
}

void InlineForwardIndex::deserialize(IOReader* /*reader*/) {
    // The forward index is disk-resident; it is read by borrowing from a
    // mapping (mmap_deserialize), not copied into RAM.
    throw std::runtime_error(
        "InlineForwardIndex: deserialize unsupported; use mmap_deserialize");
}

void InlineForwardIndex::mmap_deserialize(MmapCursor* cursor) {
    throw_if_null(cursor, "cursor must not be null");
    // Self-delimiting: read the section length, bound a subcursor to it (which
    // rejects a corrupt length overrunning the mapping), parse, then advance
    // the shared cursor past the whole component -- left just past, per the
    // base contract.
    const uint64_t section_len = cursor->read_scalar<uint64_t>();
    MmapCursor body = cursor->subcursor(cursor->pos(), section_len);
    index_section(body.current(), section_len);
    block_base_ = body.current();
    cursor->skip(section_len);
}

void InlineForwardIndex::index_section(const uint8_t* base,
                                       size_t section_len) {
    if (section_len <
        sizeof(InlineForwardIndexHeader) + sizeof(InlineForwardIndexTrailer)) {
        throw std::runtime_error("InlineForwardIndex: section too small");
    }
    // Block sub-arrays are read as typed pointers at 8-aligned block offsets,
    // so the section base must be >= 8-aligned too.
    if (reinterpret_cast<uintptr_t>(base) % kMinBlockAlign != 0) {
        throw std::runtime_error(
            "InlineForwardIndex: section is under-aligned");
    }

    InlineForwardIndexHeader header;
    std::memcpy(&header, base, sizeof(header));
    if (header.element_size != 1 && header.element_size != 2 &&
        header.element_size != 4) {
        throw std::runtime_error("InlineForwardIndex: bad element_size");
    }
    element_size_ = header.element_size;
    page_size_ = header.page_size;
    // Effective block alignment: a power of two, at least kMinBlockAlign.
    if (page_size_ < kMinBlockAlign || (page_size_ & (page_size_ - 1)) != 0) {
        throw std::runtime_error("InlineForwardIndex: bad page_size");
    }

    InlineForwardIndexTrailer trailer;
    std::memcpy(&trailer, base + section_len - sizeof(trailer),
                sizeof(trailer));
    const uint64_t num_lists = trailer.n_lists;

    const size_t entry_size = sizeof(InlineDirEntry);
    const size_t dir_end = section_len - sizeof(trailer);
    if (trailer.n_entries != header.n_blocks ||
        trailer.dir_offset < sizeof(InlineForwardIndexHeader) ||
        trailer.dir_offset > dir_end) {
        throw std::runtime_error("InlineForwardIndex: corrupt directory");
    }
    // Compare via division so an oversized n_entries can't overflow a multiply.
    const size_t dir_bytes = dir_end - trailer.dir_offset;
    if (dir_bytes % entry_size != 0 ||
        dir_bytes / entry_size != trailer.n_entries) {
        throw std::runtime_error("InlineForwardIndex: corrupt directory");
    }

    std::vector<InlineDirEntry> entries(trailer.n_entries);
    if (dir_bytes > 0) {
        std::memcpy(entries.data(), base + trailer.dir_offset, dir_bytes);
    }

    // n_lists comes from the file; cap it in the posting-list space (pl derives
    // from term_t) so num_lists + 1 can't overflow and the allocation is
    // bounded.
    if (num_lists > (static_cast<uint64_t>(1) << (8 * sizeof(term_t)))) {
        throw std::runtime_error("InlineForwardIndex: implausible n_lists");
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
                "InlineForwardIndex: corrupt directory entry");
        }
        // Grouped by (pl ascending, block ascending, gap-free) so (pl, block)
        // -> entry is the O(1) prefix-sum lookup block() relies on. A new pl
        // (or the first entry) restarts block numbering at 0.
        if (started && entry.pl < prev_pl) {
            throw std::runtime_error(
                "InlineForwardIndex: directory not ordered by (pl, block)");
        }
        const uint32_t expected_block =
            (!started || entry.pl != prev_pl) ? 0 : prev_block + 1;
        if (entry.block != expected_block) {
            throw std::runtime_error(
                "InlineForwardIndex: directory not ordered by (pl, block)");
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

// Reads are hand-rolled rather than layered on MmapCursor/borrow_padded: a
// block is random-access (located via the directory), and validating it with
// one `layout.end == len` equality is tighter than per-array cursor bounds
// checks. The sub-arrays are still borrowed in place.
BlockView InlineForwardIndex::view_block(const InlineDirEntry& entry) const {
    const uint8_t* base = block_base_ + entry.byte_off;
    const uint64_t len = entry.len;

    uint32_t n_docs;
    std::memcpy(&n_docs, base, sizeof(uint32_t));  // len >= 4 checked at load
    if (n_docs != entry.n_docs) {
        throw std::runtime_error("InlineForwardIndex: block n_docs mismatch");
    }
    // doc_id[] and off[] must fit before the offsets can be read; the comps
    // offset is where they end and is independent of total_nnz.
    const InlineBlockOffsets hdr =
        inline_block_offsets(n_docs, 0, element_size_);
    if (hdr.comps > len) {
        throw std::runtime_error(
            "InlineForwardIndex: block header overruns block");
    }
    const auto* doc_ids = reinterpret_cast<const uint32_t*>(base + hdr.doc_ids);
    const auto* offsets = reinterpret_cast<const uint32_t*>(base + hdr.off);
    if (offsets[0] != 0) {
        throw std::runtime_error(
            "InlineForwardIndex: block offsets must start 0");
    }
    for (uint32_t i = 0; i < n_docs; ++i) {
        if (offsets[i + 1] < offsets[i]) {
            throw std::runtime_error(
                "InlineForwardIndex: block offsets not monotonic");
        }
    }
    const uint64_t total_nnz = offsets[n_docs];
    const InlineBlockOffsets layout =
        inline_block_offsets(n_docs, total_nnz, element_size_);
    if (layout.end != len) {
        throw std::runtime_error("InlineForwardIndex: block length mismatch");
    }

    BlockView view;
    view.n_docs = n_docs;
    view.doc_ids = doc_ids;
    view.offsets = offsets;
    view.comps = reinterpret_cast<const term_t*>(base + layout.comps);
    view.vals = base + layout.vals;
    return view;
}

uint64_t InlineForwardIndex::num_blocks_in_list(uint32_t pl) const {
    if (static_cast<uint64_t>(pl) + 1 >= list_offset_.size()) {
        return 0;
    }
    return list_offset_[pl + 1] - list_offset_[pl];
}

BlockView InlineForwardIndex::block(uint32_t pl, uint32_t block) const {
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
