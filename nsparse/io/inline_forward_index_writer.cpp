/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/io/inline_forward_index_writer.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/inline_forward_index.h"
#include "nsparse/io/io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"

namespace nsparse {

InlineForwardIndexWriter::InlineForwardIndexWriter(uint64_t page_size,
                                                   InlineLayout layout)
    : page_size_(page_size), layout_(layout) {
    if (layout_ == InlineLayout::kPageAligned) {
        const bool is_power_of_two =
            page_size != 0 && (page_size & (page_size - 1)) == 0;
        if (!is_power_of_two || page_size < sizeof(InlineForwardIndexHeader)) {
            throw std::invalid_argument(
                "InlineForwardIndexWriter: page_size must be a power of two "
                "and "
                "at least sizeof(InlineForwardIndexHeader)");
        }
    }
}

void InlineForwardIndexWriter::write(
    const std::vector<InvertedListClusters>& lists,
    const SparseVectors& vectors, IOWriter* writer) const {
    throw_if_null(writer, "writer cannot be null");

    static_assert(sizeof(term_t) == kInlineCompWidth,
                  "component id width must match the inline format");

    const size_t element_size = vectors.get_element_size();
    if (element_size != 1 && element_size != 2 && element_size != 4) {
        throw std::invalid_argument(
            "InlineForwardIndexWriter: element_size must be 1, 2, or 4, got " +
            std::to_string(element_size));
    }
    const idx_t* indptr = vectors.indptr_data();
    const term_t* indices = vectors.indices_data();
    const uint8_t* values = vectors.values_data();
    const size_t num_vectors = vectors.num_vectors();

    // The header is written first, so count blocks up front.
    uint64_t n_blocks = 0;
    for (const auto& list : lists) {
        n_blocks += list.cluster_size();
    }

    // page_size when page-aligned, else kMinBlockAlign (packed): blocks are
    // still aligned to 8 so their sub-arrays stay in-place readable.
    const uint64_t align = alignment();
    const std::vector<uint8_t> zero_pad(align, 0);  // any pad < align

    InlineForwardIndexHeader header{};
    header.element_size = static_cast<uint32_t>(element_size);
    header.n_blocks = n_blocks;
    header.page_size = align;
    writer->write(&header, sizeof(header), 1);

    // Pad so the first block starts on an alignment boundary.
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
    // Reused per block to hold the structure-of-arrays header (doc ids + the
    // within-block CSR offsets) before the comps/vals payloads.
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
                    "InlineForwardIndexWriter: block has more than 2^32 docs");
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
                // Fail loudly on a bad doc id (asserts are off in Release).
                if (doc_id < 0 || static_cast<size_t>(doc_id) >= num_vectors) {
                    throw std::out_of_range(
                        "InlineForwardIndexWriter: block references doc id " +
                        std::to_string(doc_id) + " outside [0, " +
                        std::to_string(num_vectors) + ")");
                }
                total_nnz +=
                    static_cast<uint64_t>(indptr[doc_id + 1] - indptr[doc_id]);
                if (total_nnz > UINT32_MAX) {
                    throw std::length_error(
                        "InlineForwardIndexWriter: block total nnz exceeds the "
                        "u32 offset range");
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

            // Pad to the next block-alignment boundary.
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

    // Directory then trailer (unaligned; the reader loads the dir to RAM).
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

}  // namespace nsparse
