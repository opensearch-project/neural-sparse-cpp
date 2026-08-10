/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/inline_forward_index.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/io/io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"

namespace nsparse {

InlineForwardIndexWriter::InlineForwardIndexWriter(uint64_t page_size,
                                                   InlineLayout layout)
    : page_size_(page_size), layout_(layout) {
    // page_size only governs alignment when padding is on; in packed mode it is
    // ignored, so don't reject an unused value.
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
    const SparseVectors& vectors, IOWriter* bin_writer,
    IOWriter* dir_writer) const {
    throw_if_null(bin_writer, "bin_writer cannot be null");
    throw_if_null(dir_writer, "dir_writer cannot be null");

    const size_t element_size = vectors.get_element_size();
    // Only float (4) and scalar-quantized u16 (2) / u8 (1) widths are
    // representable and scoreable; reject anything else up front so an
    // unconsumable file is never produced.
    if (element_size != 1 && element_size != 2 && element_size != 4) {
        throw std::invalid_argument(
            "InlineForwardIndexWriter: element_size must be 1, 2, or 4, got " +
            std::to_string(element_size));
    }
    const idx_t* indptr = vectors.indptr_data();
    const term_t* indices = vectors.indices_data();
    const uint8_t* values = vectors.values_data();
    const size_t num_vectors = vectors.num_vectors();

    // Count blocks up front so the .bin header is correct on first write (the
    // sinks are append-only streams).
    uint64_t n_blocks = 0;
    for (const auto& list : lists) {
        n_blocks += list.cluster_size();
    }

    // Effective block alignment: page_size when padding, else 1 (packed, where
    // every align_up below is a no-op and no padding is written).
    const uint64_t align = alignment();

    // A single reusable zero buffer covers both the post-header padding and any
    // per-block trailing padding (both strictly smaller than `align`).
    const std::vector<uint8_t> zero_pad(align, 0);

    // --- inline.bin header, padded (when page-aligned) so the first block
    // starts on an alignment() boundary; no padding is written when packed. ---
    InlineForwardIndexHeader header{};
    header.magic = fourcc(kInlineForwardIndexMagic);
    header.version = kInlineForwardIndexVersion;
    header.element_size = static_cast<uint32_t>(element_size);
    header.reserved = 0;
    header.n_blocks = n_blocks;
    header.page_size = align;
    bin_writer->write(&header, sizeof(header), 1);

    const uint64_t first_block_off =
        inline_align_up(sizeof(InlineForwardIndexHeader), align);
    const uint64_t header_pad =
        first_block_off - sizeof(InlineForwardIndexHeader);
    if (header_pad > 0) {
        bin_writer->write(const_cast<uint8_t*>(zero_pad.data()), 1, header_pad);
    }
    uint64_t cur_off = first_block_off;  // offset of the next byte to write

    // --- Blocks, each starting on an alignment() boundary (a page_size
    // boundary when page-aligned, or back-to-back when packed). ---
    std::vector<InlineDirEntry> entries;
    entries.reserve(n_blocks);

    for (size_t pl = 0; pl < lists.size(); ++pl) {
        const InvertedListClusters& list = lists[pl];
        const size_t n_clusters = list.cluster_size();
        for (size_t block = 0; block < n_clusters; ++block) {
            const std::span<const idx_t> docs = list.get_docs(block);
            const uint64_t block_off =
                cur_off;  // already aligned to alignment()

            const uint32_t n_docs = static_cast<uint32_t>(docs.size());
            bin_writer->write(const_cast<uint32_t*>(&n_docs), sizeof(uint32_t),
                              1);
            uint64_t block_len = sizeof(uint32_t);  // the n_docs prefix

            for (const idx_t doc_id : docs) {
                // The writer emits a durable artifact, so a bad/stale doc id
                // must fail loudly rather than read out of bounds and persist
                // corrupt bytes (asserts are compiled out in Release).
                if (doc_id < 0 || static_cast<size_t>(doc_id) >= num_vectors) {
                    throw std::out_of_range(
                        "InlineForwardIndexWriter: block references doc id " +
                        std::to_string(doc_id) + " outside [0, " +
                        std::to_string(num_vectors) + ")");
                }
                const idx_t start = indptr[doc_id];
                const uint32_t nnz =
                    static_cast<uint32_t>(indptr[doc_id + 1] - start);

                uint32_t doc_id_u32 = static_cast<uint32_t>(doc_id);
                bin_writer->write(&doc_id_u32, sizeof(uint32_t), 1);
                uint32_t nnz_field = nnz;
                bin_writer->write(&nnz_field, sizeof(uint32_t), 1);
                if (nnz > 0) {
                    bin_writer->write(const_cast<term_t*>(indices + start),
                                      sizeof(term_t), nnz);
                    bin_writer->write(
                        const_cast<uint8_t*>(
                            values + static_cast<size_t>(start) * element_size),
                        1, static_cast<size_t>(nnz) * element_size);
                }
                block_len += 2 * sizeof(uint32_t) +
                             static_cast<uint64_t>(nnz) * sizeof(term_t) +
                             static_cast<uint64_t>(nnz) * element_size;
            }
            cur_off += block_len;

            // Zero-pad up to the next alignment boundary so the following block
            // is aligned (no-op when packed).
            const uint64_t padded = inline_align_up(cur_off, align);
            const uint64_t pad = padded - cur_off;
            if (pad > 0) {
                bin_writer->write(const_cast<uint8_t*>(zero_pad.data()), 1,
                                  pad);
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

    // --- inline.bin.dir: header followed by the block directory. ---
    InlineDirHeader dir_header{};
    dir_header.magic = fourcc(kInlineDirMagic);
    dir_header.version = kInlineForwardIndexVersion;
    dir_header.element_size = static_cast<uint32_t>(element_size);
    dir_header.reserved = 0;
    dir_header.n_lists = lists.size();
    dir_header.n_entries = entries.size();
    dir_header.page_size = align;
    dir_writer->write(&dir_header, sizeof(dir_header), 1);
    if (!entries.empty()) {
        dir_writer->write(entries.data(), sizeof(InlineDirEntry),
                          entries.size());
    }
}

}  // namespace nsparse
