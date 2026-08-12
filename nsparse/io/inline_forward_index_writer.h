/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef INLINE_FORWARD_INDEX_WRITER_H
#define INLINE_FORWARD_INDEX_WRITER_H

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

// Header-only build-time writer for the inline forward-index file
class InlineForwardIndexWriter {
public:
    static constexpr uint64_t kDefaultPageSize = 4096;  // common OS page size

    // page_size: block alignment when page-aligned (power of two, >= header
    // size); ignored when packed.
    explicit InlineForwardIndexWriter(
        uint64_t page_size = kDefaultPageSize,
        InlineLayout layout = InlineLayout::kPageAligned)
        : page_size_(page_size), layout_(layout) {
        if (layout_ == InlineLayout::kPageAligned) {
            const bool is_power_of_two =
                page_size != 0 && (page_size & (page_size - 1)) == 0;
            if (!is_power_of_two ||
                page_size < sizeof(InlineForwardIndexHeader)) {
                throw std::invalid_argument(
                    "InlineForwardIndexWriter: page_size must be a power of "
                    "two "
                    "and at least sizeof(InlineForwardIndexHeader)");
            }
        }
    }

    // Write the whole file to `writer`: header, blocks, directory, trailer. One
    // record per member doc, copying element_size-wide value bytes from
    // `vectors` (float and quantized alike); `lists` must be built/summarized.
    // Throws std::invalid_argument (null sink / element_size not in {1,2,4}) or
    // std::out_of_range (doc id outside [0, num_vectors)).
    void write(const std::vector<InvertedListClusters>& lists,
               const SparseVectors& vectors, IOWriter* writer) const {
        throw_if_null(writer, "writer cannot be null");

        const size_t element_size = vectors.get_element_size();
        if (element_size != 1 && element_size != 2 && element_size != 4) {
            throw std::invalid_argument(
                "InlineForwardIndexWriter: element_size must be 1, 2, or 4, "
                "got " +
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

        // page_size when page-aligned, else 1 (packed: align_up is a no-op).
        const uint64_t align = alignment();
        const std::vector<uint8_t> zero_pad(align, 0);  // any pad < align

        InlineForwardIndexHeader header{};
        header.magic = InlineForwardIndexHeader::kMagic;
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
        for (size_t pl = 0; pl < lists.size(); ++pl) {
            const InvertedListClusters& list = lists[pl];
            const size_t n_clusters = list.cluster_size();
            for (size_t block = 0; block < n_clusters; ++block) {
                const std::span<const idx_t> docs = list.get_docs(block);
                const uint64_t block_off = cur_off;

                const uint32_t n_docs = static_cast<uint32_t>(docs.size());
                writer->write(const_cast<uint32_t*>(&n_docs), sizeof(uint32_t),
                              1);
                uint64_t block_len = sizeof(uint32_t);

                for (const idx_t doc_id : docs) {
                    // Fail loudly on a bad doc id (asserts are off in Release).
                    if (doc_id < 0 ||
                        static_cast<size_t>(doc_id) >= num_vectors) {
                        throw std::out_of_range(
                            "InlineForwardIndexWriter: block references doc "
                            "id " +
                            std::to_string(doc_id) + " outside [0, " +
                            std::to_string(num_vectors) + ")");
                    }
                    const idx_t start = indptr[doc_id];
                    const uint32_t nnz =
                        static_cast<uint32_t>(indptr[doc_id + 1] - start);

                    uint32_t doc_id_u32 = static_cast<uint32_t>(doc_id);
                    writer->write(&doc_id_u32, sizeof(uint32_t), 1);
                    uint32_t nnz_field = nnz;
                    writer->write(&nnz_field, sizeof(uint32_t), 1);
                    if (nnz > 0) {
                        writer->write(const_cast<term_t*>(indices + start),
                                      sizeof(term_t), nnz);
                        writer->write(const_cast<uint8_t*>(
                                          values + static_cast<size_t>(start) *
                                                       element_size),
                                      1,
                                      static_cast<size_t>(nnz) * element_size);
                    }
                    block_len += 2 * sizeof(uint32_t) +
                                 static_cast<uint64_t>(nnz) * sizeof(term_t) +
                                 static_cast<uint64_t>(nnz) * element_size;
                }
                cur_off += block_len;

                // Pad to the next alignment boundary (no-op when packed).
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
            writer->write(entries.data(), sizeof(InlineDirEntry),
                          entries.size());
        }
        InlineForwardIndexTrailer trailer{};
        trailer.magic = InlineForwardIndexTrailer::kMagic;
        trailer.reserved = 0;
        trailer.dir_offset = dir_offset;
        trailer.n_entries = entries.size();
        trailer.n_lists = lists.size();
        writer->write(&trailer, sizeof(trailer), 1);
    }

    uint64_t page_size() const { return page_size_; }
    InlineLayout layout() const { return layout_; }

    // page_size when page-aligned, else 1 (packed).
    uint64_t alignment() const {
        return layout_ == InlineLayout::kPageAligned ? page_size_ : 1;
    }

private:
    uint64_t page_size_;
    InlineLayout layout_;
};

}  // namespace nsparse

#endif  // INLINE_FORWARD_INDEX_WRITER_H
