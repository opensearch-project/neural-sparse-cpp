/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/io/inline_forward_index_io.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/inline_forward_index.h"
#include "nsparse/io/align.h"
#include "nsparse/io/io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"
#include "nsparse/utils/mmap_cursor.h"
#include "nsparse/utils/prefetch.h"

namespace nsparse::detail {
namespace {

// One stored code, widened. Exact for the 1- and 2-byte code widths and for
// float, so comparing widened values orders them exactly as the stored ones.
double code_value(const uint8_t* values, size_t index, size_t element_size) {
    if (element_size == sizeof(float)) {
        float value = NAN;
        std::memcpy(&value, values + index * sizeof(float), sizeof(float));
        return value;
    }
    if (element_size == sizeof(uint16_t)) {
        uint16_t value = 0;
        std::memcpy(&value, values + index * sizeof(uint16_t),
                    sizeof(uint16_t));
        return value;
    }
    return values[index];
}

}  // namespace

// The record bounds math (nnz up to INT32_MAX times record width) assumes a
// 64-bit size_t so it cannot overflow.
static_assert(sizeof(size_t) >= 8, "InlineForwardIndex assumes 64-bit size_t");

DocSlice resolve_doc(const FullVectorStore& store, idx_t doc_id,
                     size_t element_size) {
    if (doc_id < 0 || static_cast<uint64_t>(doc_id) >= store.num_locators) {
        throw std::out_of_range(
            "DiskSeismic: doc id outside the locator directory");
    }
    const DocLocator locator = store.locators[doc_id];
    if (locator.posting_list == DocLocator::kRemainder) {
        if (store.remainder == nullptr ||
            locator.block >= store.remainder->num_vectors()) {
            throw std::runtime_error(
                "DiskSeismic: remainder locator out of range");
        }
        const offset_t* indptr = store.remainder->indptr_data();
        const offset_t start = indptr[locator.block];
        return {store.remainder->indices_data() + start,
                store.remainder->values_data() +
                    static_cast<size_t>(start) * element_size,
                static_cast<size_t>(indptr[locator.block + 1] - start)};
    }
    const BlockView view = store.fwd->block(locator.posting_list, locator.block);
    if (view.absent() || locator.slot >= view.n_docs ||
        view.doc_ids[locator.slot] != static_cast<uint32_t>(doc_id)) {
        throw std::runtime_error(
            "DiskSeismic: doc locator does not resolve to its doc");
    }
    return {view.doc_comps(locator.slot),
            view.doc_vals(locator.slot, element_size), view.nnz(locator.slot)};
}

void resolve_docs(const FullVectorStore& store, const idx_t* doc_ids,
                  size_t n_docs, size_t element_size,
                  DocResolveScratch* scratch, std::vector<DocSlice>* out) {
    // Docs per chunk. The walk is staged so each step's misses overlap, but a
    // core has only ~10-16 line-fill buffers: prefetching a whole 300-doc batch
    // in one sweep drops all but the first dozen and the loads miss anyway. A
    // chunk of this size keeps the outstanding prefetches inside that budget
    // while still overlapping enough of them to matter.
    constexpr size_t kChunk = 12;

    out->clear();
    out->reserve(n_docs);

    for (size_t begin = 0; begin < n_docs; begin += kChunk) {
        const size_t end = std::min(begin + kChunk, n_docs);
        const size_t len = end - begin;
        scratch->locators.assign(len, DocLocator{});
        scratch->entries.assign(len, nullptr);

        // Stage 1: the locator table, one random 12-byte read per doc.
        for (size_t i = 0; i < len; ++i) {
            const idx_t doc_id = doc_ids[begin + i];
            if (doc_id < 0 ||
                static_cast<uint64_t>(doc_id) >= store.num_locators) {
                throw std::out_of_range(
                    "DiskSeismic: doc id outside the locator directory");
            }
            NSPARSE_PREFETCH(&store.locators[doc_id], 0, 0);
        }
        // Stage 2: read the locators, prefetch each one's directory entry. A
        // remainder doc has no entry and is resolved singly in stage 4, being
        // rare enough (docs that no block holds) not to be worth pipelining.
        for (size_t i = 0; i < len; ++i) {
            scratch->locators[i] = store.locators[doc_ids[begin + i]];
            const DocLocator& locator = scratch->locators[i];
            if (locator.posting_list != DocLocator::kRemainder) {
                scratch->entries[i] =
                    store.fwd->dir_entry(locator.posting_list, locator.block);
                NSPARSE_PREFETCH(scratch->entries[i], 0, 0);
            }
        }
        // Stage 3: read the entries, prefetch the block prefix the slice math
        // reads ([n_docs][doc_id[]][off[]]).
        for (size_t i = 0; i < len; ++i) {
            if (scratch->entries[i] != nullptr) {
                NSPARSE_PREFETCH(
                    store.fwd->block_base() + scratch->entries[i]->byte_off, 0,
                    0);
            }
        }
        // Stage 4: slice. The rows themselves are left to the caller, which
        // prefetches them with its own bounded lookahead as it scores.
        for (size_t i = 0; i < len; ++i) {
            if (scratch->entries[i] == nullptr) {
                out->push_back(
                    resolve_doc(store, doc_ids[begin + i], element_size));
                continue;
            }
            const InlineDirEntry& entry = *scratch->entries[i];
            if (scratch->locators[i].slot >= entry.n_docs) {
                throw std::runtime_error(
                    "DiskSeismic: doc locator does not resolve to its doc");
            }
            out->push_back(store.fwd->doc_slice(entry, scratch->locators[i].slot));
        }
    }
}

InlineForwardIndex::CodeCut InlineForwardIndex::select_top_codes(
    const uint8_t* values, uint64_t nnz, uint64_t keep, size_t element_size,
    std::vector<double>* scratch) {
    // `keep` of `nnz` values, chosen largest-first. Rather than a list of
    // indices this returns the boundary value and how many of the values *at*
    // it are in: everything above the boundary is in, and the ties are taken in
    // stored order. That keeps the two emit passes to one comparison per value
    // and leaves the kept components in ascending component order, which the
    // format requires.
    scratch->clear();
    scratch->reserve(nnz);
    for (uint64_t i = 0; i < nnz; ++i) {
        scratch->push_back(code_value(values, i, element_size));
    }
    const auto boundary =
        scratch->begin() + static_cast<ptrdiff_t>(keep) - 1;
    std::nth_element(scratch->begin(), boundary, scratch->end(),
                     std::greater<>());
    const double cut = *boundary;
    // Nothing outside the first `keep` can exceed the boundary, so counting
    // there counts the whole array.
    uint32_t above = 0;
    for (uint64_t i = 0; i < keep; ++i) {
        above += (*scratch)[i] > cut ? 1 : 0;
    }
    return {cut, static_cast<uint32_t>(keep) - above, false};
}

void InlineForwardIndex::gather_selected(const term_t* comps,
                                         const uint8_t* values, uint64_t nnz,
                                         size_t element_size, CodeCut cut,
                                         std::vector<term_t>* comps_out,
                                         std::vector<uint8_t>* vals_out) {
    uint32_t quota = cut.ties;
    for (uint64_t i = 0; i < nnz; ++i) {
        const double value = code_value(values, i, element_size);
        if (value <= cut.boundary) {
            if (value < cut.boundary || quota == 0) {
                continue;
            }
            --quota;
        }
        if (comps_out != nullptr) {
            comps_out->push_back(comps[i]);
        }
        if (vals_out != nullptr) {
            const uint8_t* src = values + i * element_size;
            vals_out->insert(vals_out->end(), src, src + element_size);
        }
    }
}

InlineForwardIndex::InlineForwardIndex(
    const std::vector<InvertedListClusters>& lists,
    const SparseVectors& vectors, uint64_t page_size, InlineLayout layout,
    uint32_t max_doc_nnz, const std::vector<DocLocator>* full_copies)
    : lists_(&lists),
      vectors_(&vectors),
      write_page_size_(page_size),
      write_layout_(layout),
      write_max_doc_nnz_(max_doc_nnz),
      write_full_copies_(full_copies) {
    if (layout == InlineLayout::kPageAligned) {
        const bool is_power_of_two =
            page_size != 0 && (page_size & (page_size - 1)) == 0;
        if (!is_power_of_two || page_size < sizeof(InlineForwardIndexHeader)) {
            throw std::invalid_argument(
                "InlineForwardIndex: page_size must be a power of two and at "
                "least sizeof(InlineForwardIndexHeader)");
        }
    }
    // Truncating without knowing where each whole copy went would leave no way
    // to re-score, and a locator table with nothing truncated is a caller that
    // thinks it asked for truncation. Reject both rather than silently picking
    // one interpretation.
    if ((max_doc_nnz == 0) != (full_copies == nullptr)) {
        throw std::invalid_argument(
            "InlineForwardIndex: max_doc_nnz and the whole-copy locators must "
            "be given together");
    }
    if (full_copies != nullptr &&
        full_copies->size() != vectors.num_vectors()) {
        throw std::invalid_argument(
            "InlineForwardIndex: the whole-copy locators must have one entry "
            "per vector");
    }
}

uint64_t InlineForwardIndex::doc_stored_nnz(idx_t doc_id, uint64_t full_nnz,
                                           uint32_t pl, uint32_t block) const {
    if (write_max_doc_nnz_ == 0 || full_nnz <= write_max_doc_nnz_) {
        return full_nnz;
    }
    // count_block validated doc_id against num_vectors, and the constructor
    // checked the table covers every vector, so this index is in range.
    const DocLocator& locator =
        (*write_full_copies_)[static_cast<size_t>(doc_id)];
    if (locator.posting_list == pl && locator.block == block) {
        return full_nnz;  // the one copy that stays whole
    }
    return write_max_doc_nnz_;
}

// Buf's move copies the source's data()/size() and moves only its owner, so a
// naive move would leave the moved-from index reporting stale blocks. Reset the
// moved-from members explicitly to keep it inert.
InlineForwardIndex::InlineForwardIndex(InlineForwardIndex&& other) noexcept
    : lists_(other.lists_),
      vectors_(other.vectors_),
      write_page_size_(other.write_page_size_),
      write_layout_(other.write_layout_),
      write_max_doc_nnz_(other.write_max_doc_nnz_),
      write_full_copies_(other.write_full_copies_),
      block_base_(other.block_base_),
      element_size_(other.element_size_),
      page_size_(other.page_size_),
      max_doc_nnz_(other.max_doc_nnz_),
      entries_(std::move(other.entries_)),
      list_offset_(std::move(other.list_offset_)) {
    other.lists_ = nullptr;
    other.vectors_ = nullptr;
    other.write_max_doc_nnz_ = 0;
    other.write_full_copies_ = nullptr;
    other.block_base_ = nullptr;
    other.element_size_ = 0;
    other.page_size_ = 0;
    other.max_doc_nnz_ = 0;
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
        write_max_doc_nnz_ = other.write_max_doc_nnz_;
        write_full_copies_ = other.write_full_copies_;
        block_base_ = other.block_base_;
        element_size_ = other.element_size_;
        page_size_ = other.page_size_;
        max_doc_nnz_ = other.max_doc_nnz_;
        entries_ = std::move(other.entries_);
        list_offset_ = std::move(other.list_offset_);
        other.lists_ = nullptr;
        other.vectors_ = nullptr;
        other.write_max_doc_nnz_ = 0;
        other.write_full_copies_ = nullptr;
        other.block_base_ = nullptr;
        other.element_size_ = 0;
        other.page_size_ = 0;
        other.max_doc_nnz_ = 0;
        other.entries_ = Buf<InlineDirEntry>();
        other.list_offset_ = Buf<uint64_t>();
    }
    return *this;
}

InlineForwardIndex::BlockCounts InlineForwardIndex::count_block(
    std::span<const idx_t> docs, const offset_t* indptr, size_t num_vectors,
    uint32_t pl, uint32_t block) const {
    // n_docs and the within-block offsets are u32 on the wire.
    if (docs.size() > UINT32_MAX) {
        throw std::length_error(
            "InlineForwardIndex: block has more than 2^32 docs");
    }
    const uint32_t n_docs = static_cast<uint32_t>(docs.size());
    uint64_t total_nnz = 0;
    for (const idx_t doc_id : docs) {
        if (doc_id < 0 || static_cast<size_t>(doc_id) >= num_vectors) {
            throw std::out_of_range(
                "InlineForwardIndex: block references doc id " +
                std::to_string(doc_id) + " outside [0, " +
                std::to_string(num_vectors) + ")");
        }
        total_nnz += doc_stored_nnz(
            doc_id,
            static_cast<uint64_t>(indptr[doc_id + 1] - indptr[doc_id]), pl,
            block);
        // Capped at INT32_MAX (not the u32 max): off[] is uint32_t on the wire
        // but must stay reinterpretable as idx_t (int32_t) on the search path,
        // and a wrapped offset would also desync the block from its length.
        if (total_nnz > static_cast<uint64_t>(INT32_MAX)) {
            throw std::length_error(
                "InlineForwardIndex: block total nnz exceeds INT32_MAX");
        }
    }
    return {n_docs, total_nnz};
}

uint64_t InlineForwardIndex::section_length() const {
    const std::vector<InvertedListClusters>& lists = *lists_;
    const SparseVectors& vectors = *vectors_;
    const size_t element_size = vectors.get_element_size();
    if (element_size != 1 && element_size != 2 && element_size != 4) {
        throw std::invalid_argument(
            "InlineForwardIndex: element_size must be 1, 2, or 4, got " +
            std::to_string(element_size));
    }
    const offset_t* indptr = vectors.indptr_data();
    const size_t num_vectors = vectors.num_vectors();
    const uint64_t align = write_alignment();

    uint64_t n_blocks = 0;
    for (const auto& list : lists) {
        n_blocks += list.cluster_size();
    }

    // Mirror write_body's offset accounting exactly (both go through the shared
    // inline_block_offsets/inline_align_up), touching no payload. serialize()
    // asserts the streamed body matches this length, so any drift fails closed.
    uint64_t cur_off = inline_align_up(sizeof(InlineForwardIndexHeader), align);
    for (size_t pl = 0; pl < lists.size(); ++pl) {
        const InvertedListClusters& list = lists[pl];
        const size_t n_clusters = list.cluster_size();
        for (size_t block = 0; block < n_clusters; ++block) {
            const BlockCounts counts = count_block(
                list.get_docs(block), indptr, num_vectors,
                static_cast<uint32_t>(pl), static_cast<uint32_t>(block));
            cur_off += inline_block_offsets(counts.n_docs, counts.total_nnz,
                                            element_size)
                           .end;
            cur_off = inline_align_up(cur_off, align);
        }
    }
    const uint64_t dir_offset = cur_off;
    return dir_offset + n_blocks * sizeof(InlineDirEntry) +
           sizeof(InlineForwardIndexTrailer);
}

void InlineForwardIndex::serialize(IOWriter* writer) const {
    throw_if_null(writer, "writer cannot be null");
    if (lists_ == nullptr || vectors_ == nullptr) {
        throw std::logic_error(
            "InlineForwardIndex::serialize called on a read-mode index");
    }
    // Self-delimiting and composable at any stream offset: pad to the block
    // alignment so the section base is loadable wherever it lands, then write a
    // u64 length prefix. mmap_deserialize() skips the same padding, reads the
    // length, and advances the shared cursor exactly past the body. The length
    // comes from a payload-free sizing pass, so the (corpus-sized) body streams
    // straight to the writer rather than being buffered in RAM first.
    io_align::pad_to(writer, kMinBlockAlign);
    uint64_t section_len = section_length();
    writer->write(&section_len, sizeof(section_len), 1);
    const size_t body_start = writer->pos();
    write_body(writer);
    // The prefix must equal what the body actually wrote, or a reader would
    // trust a wrong length. Catches any sizing/emitting drift at the source.
    if (writer->pos() - body_start != section_len) {
        throw std::logic_error(
            "InlineForwardIndex: serialized body length disagrees with the "
            "computed section length");
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
    const offset_t* indptr = vectors.indptr_data();
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
    header.max_doc_nnz = write_max_doc_nnz_;
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
    // Per-slot selection, computed once and used by both the comps pass and the
    // vals pass so the two cannot disagree about which components the doc keeps.
    // A default-constructed CodeCut (whole == true) means the slot stores
    // everything, which is every slot when truncation is off.
    std::vector<CodeCut> cuts;
    std::vector<double> cut_scratch;
    std::vector<term_t> comps_scratch;
    std::vector<uint8_t> vals_scratch;
    for (size_t pl = 0; pl < lists.size(); ++pl) {
        const InvertedListClusters& list = lists[pl];
        const size_t n_clusters = list.cluster_size();
        for (size_t block = 0; block < n_clusters; ++block) {
            const std::span<const idx_t> docs = list.get_docs(block);
            const uint64_t block_off = cur_off;
            // Validate + count once (the same call section_length() used), then
            // build doc_id[]/off[] (the within-block CSR prefix sum) from the
            // now-safe docs. count_block keeps off[] within INT32_MAX, so the
            // running total below cannot wrap or desync the recorded length.
            const BlockCounts counts =
                count_block(docs, indptr, num_vectors,
                            static_cast<uint32_t>(pl),
                            static_cast<uint32_t>(block));
            const uint32_t n_docs = counts.n_docs;
            const uint64_t total_nnz = counts.total_nnz;
            doc_ids.clear();
            offsets.assign(1, 0);
            doc_ids.reserve(n_docs);
            offsets.reserve(n_docs + 1);
            cuts.clear();
            cuts.reserve(n_docs);
            uint64_t running = 0;
            for (const idx_t doc_id : docs) {
                const offset_t start = indptr[doc_id];
                const auto nnz =
                    static_cast<uint64_t>(indptr[doc_id + 1] - start);
                const uint64_t stored = doc_stored_nnz(
                    doc_id, nnz, static_cast<uint32_t>(pl),
                    static_cast<uint32_t>(block));
                cuts.push_back(
                    stored == nnz
                        ? CodeCut{}
                        : select_top_codes(
                              values + static_cast<size_t>(start) * element_size,
                              nnz, stored, element_size, &cut_scratch));
                running += stored;
                doc_ids.push_back(static_cast<uint32_t>(doc_id));
                offsets.push_back(static_cast<uint32_t>(running));
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
            for (uint32_t slot = 0; slot < n_docs; ++slot) {
                const idx_t doc_id = docs[slot];
                const offset_t start = indptr[doc_id];
                const auto nnz =
                    static_cast<size_t>(indptr[doc_id + 1] - start);
                if (nnz == 0) {
                    continue;
                }
                if (cuts[slot].whole) {
                    writer->write(const_cast<term_t*>(indices + start),
                                  sizeof(term_t), nnz);
                    continue;
                }
                comps_scratch.clear();
                gather_selected(
                    indices + start,
                    values + static_cast<size_t>(start) * element_size, nnz,
                    element_size, cuts[slot], &comps_scratch, nullptr);
                writer->write(comps_scratch.data(), sizeof(term_t),
                              comps_scratch.size());
            }
            const uint64_t comps_end =
                layout.comps + total_nnz * sizeof(term_t);
            const uint64_t vals_pad = layout.vals - comps_end;
            if (vals_pad > 0) {
                writer->write(const_cast<uint8_t*>(zero_pad.data()), 1,
                              vals_pad);
            }
            for (uint32_t slot = 0; slot < n_docs; ++slot) {
                const idx_t doc_id = docs[slot];
                const offset_t start = indptr[doc_id];
                const auto nnz =
                    static_cast<size_t>(indptr[doc_id + 1] - start);
                if (nnz == 0) {
                    continue;
                }
                const uint8_t* doc_vals =
                    values + static_cast<size_t>(start) * element_size;
                if (cuts[slot].whole) {
                    writer->write(const_cast<uint8_t*>(doc_vals), 1,
                                  nnz * element_size);
                    continue;
                }
                vals_scratch.clear();
                gather_selected(indices + start, doc_vals, nnz, element_size,
                                cuts[slot], nullptr, &vals_scratch);
                writer->write(vals_scratch.data(), 1, vals_scratch.size());
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
    // Mirror serialize(): consume the alignment padding, read the section
    // length, reject one that overruns the mapping, then borrow the body in
    // place and advance the shared cursor exactly past it (left just past, per
    // the base contract).
    io_align::skip_padding(cursor, kMinBlockAlign);
    const uint64_t section_len = cursor->read_scalar<uint64_t>();
    if (section_len > cursor->remaining()) {
        throw std::runtime_error(
            "InlineForwardIndex: section length overruns the mapping");
    }
    const uint8_t* base = cursor->current();
    load_directory(base, section_len);
    block_base_ = base;
    cursor->skip(section_len);
}

void InlineForwardIndex::load_directory(const uint8_t* base,
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
    max_doc_nnz_ = header.max_doc_nnz;
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
        if (entry.pl >= num_lists || entry.len < kInlineBlockPrefixSize ||
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
    // entry.len >= kInlineBlockPrefixSize was checked in load_directory.
    std::memcpy(&n_docs, base, sizeof(uint32_t));
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

const InlineDirEntry* InlineForwardIndex::dir_entry(uint32_t pl,
                                                   uint32_t block) const {
    // An out-of-range pl makes num_blocks_in_list(pl) return 0, so this returns
    // before list_offset_[pl] is ever indexed.
    if (block >= num_blocks_in_list(pl)) {
        return nullptr;
    }
    const InlineDirEntry& entry = entries_[list_offset_[pl] + block];
    if (entry.pl != pl || entry.block != block) {
        return nullptr;  // directory not in the expected (pl, block) order
    }
    return &entry;
}

DocSlice InlineForwardIndex::doc_slice(const InlineDirEntry& entry,
                                      uint32_t slot) const {
    if (slot >= entry.n_docs) {
        throw std::runtime_error("InlineForwardIndex: slot outside its block");
    }
    const uint8_t* base = block_base_ + entry.byte_off;
    // doc_id[] and off[] must fit before the offsets can be read; the comps
    // offset is where they end and is independent of total_nnz.
    const InlineBlockOffsets hdr =
        inline_block_offsets(entry.n_docs, 0, element_size_);
    if (hdr.comps > entry.len) {
        throw std::runtime_error(
            "InlineForwardIndex: block header overruns block");
    }
    const auto* offsets = reinterpret_cast<const uint32_t*>(base + hdr.off);
    const uint64_t total_nnz = offsets[entry.n_docs];
    const InlineBlockOffsets layout =
        inline_block_offsets(entry.n_docs, total_nnz, element_size_);
    // The same length equality block() checks, which is what makes total_nnz --
    // and so the arrays the slice points into -- trustworthy.
    if (layout.end != entry.len) {
        throw std::runtime_error("InlineForwardIndex: block length mismatch");
    }
    const uint32_t start = offsets[slot];
    const uint32_t end = offsets[slot + 1];
    if (start > end || end > total_nnz) {
        throw std::runtime_error(
            "InlineForwardIndex: block offsets not monotonic");
    }
    return {reinterpret_cast<const term_t*>(base + layout.comps) + start,
            base + layout.vals + static_cast<size_t>(start) * element_size_,
            end - start};
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
