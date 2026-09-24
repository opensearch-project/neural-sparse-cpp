/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

// Where the bytes of a serialized disk_seismic / disk_seismic_sq index go, and
// what a narrower encoding of each array would save. Read-only: it re-parses an
// existing .dat with the same components the index loads, so sizing a format
// change costs one pass over the file rather than a rebuild.
//
//   index_size_stats <index.dat>
//
// Prints the section split (summaries / inline forward index / doc directory),
// the inline forward index's per-array split, and simulated sizes for: packing
// the blocks (what a file written before that became the default still stands to
// save), delta-coded component ids, narrower doc-id and offset tables, and 4-bit
// values.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nsparse/inline_forward_index.h"
#include "nsparse/io/inline_forward_index_io.h"
#include "nsparse/io/seismic_invlists_writer.h"
#include "nsparse/types.h"
#include "nsparse/utils/mmap_cursor.h"
#include "nsparse/utils/mmap_file.h"
#include "nsparse/utils/scalar_quantizer.h"

using nsparse::MmapCursor;
using nsparse::MmapFile;
using nsparse::QuantizerType;
using nsparse::SeismicInvertedListsWriter;
using nsparse::detail::BlockView;
using nsparse::detail::InlineForwardIndex;

namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

double gib(uint64_t bytes) { return static_cast<double>(bytes) / kGiB; }

void line(const char* label, uint64_t bytes, uint64_t total) {
    std::printf("  %-34s %16llu  %8.3f GiB  %6.2f%%\n", label,
                static_cast<unsigned long long>(bytes), gib(bytes),
                total == 0 ? 0.0
                           : 100.0 * static_cast<double>(bytes) /
                                 static_cast<double>(total));
}

// Bytes a Lucene-style VByte (seven payload bits per byte, high bit continuing)
// takes. A sizing model only: nothing in the index writes this encoding.
uint64_t vbyte_len(uint64_t value) {
    uint64_t bytes = 1;
    while (value >= 0x80) {
        value >>= 7;
        ++bytes;
    }
    return bytes;
}

struct Stats {
    uint64_t n_blocks = 0;
    // Doc slots, i.e. postings: a doc counts once per block that holds it.
    uint64_t n_docs = 0;
    uint64_t nnz = 0;

    // The layout as stored, by array.
    uint64_t b_prefix = 0;
    uint64_t b_doc_ids = 0;
    uint64_t b_off = 0;
    uint64_t b_comps = 0;
    uint64_t b_vals = 0;
    uint64_t b_intra_pad = 0;  // comps -> vals alignment
    uint64_t b_inter_pad = 0;  // block -> block alignment
    uint64_t b_dir = 0;
    uint64_t b_payload = 0;  // sum of the directory entries' len

    // Encodings not taken, sized against what is stored.
    uint64_t c_comps_vbyte = 0;    // per-doc delta + VByte
    uint64_t c_comps_flagged = 0;  // per-doc delta, one width bit per value
    uint64_t c_docids_vbyte = 0;   // delta + VByte over sorted doc ids
    uint64_t c_off_vbyte = 0;      // per-doc nnz as VByte
    uint64_t c_off_u16 = 0;        // u16 offsets where a block permits
    uint64_t blocks_off_fits_u16 = 0;

    // Diagnostics.
    uint64_t docs_comps_unsorted = 0;
    uint64_t blocks_docids_unsorted = 0;
    uint64_t vals_nonzero = 0;
    uint64_t vals_le15 = 0;  // would fit 4 bits as stored
};

// Component ids of one doc: how much delta coding would buy, two ways, and
// whether they ascend (which is what makes the deltas small).
void size_doc_comps(const nsparse::term_t* comps, uint32_t len, Stats* stats) {
    if (len == 0) {
        return;
    }
    constexpr uint32_t kNarrowMax = 0xFF;
    constexpr uint64_t kFlagBits = 8;
    bool ascending = true;
    uint64_t vbytes = vbyte_len(comps[0]);
    // One flag bit per value, then one payload byte, or two where the difference
    // needs them.
    uint64_t flagged = ((len + kFlagBits - 1) / kFlagBits) + 1 +
                       (comps[0] > kNarrowMax ? 1 : 0);
    uint32_t prev = comps[0];
    for (uint32_t j = 1; j < len; ++j) {
        const uint32_t cur = comps[j];
        if (cur <= prev) {
            ascending = false;
        }
        const uint32_t gap = cur > prev ? cur - prev : 0;
        vbytes += vbyte_len(gap);
        flagged += gap > kNarrowMax ? 2 : 1;
        prev = cur;
    }
    if (!ascending) {
        stats->docs_comps_unsorted += 1;
    }
    stats->c_comps_vbyte += vbytes;
    stats->c_comps_flagged += flagged;
}

void accumulate(const BlockView& view, size_t element_size, uint64_t align,
                Stats* stats) {
    const uint64_t n_docs = view.n_docs;
    const uint64_t total_nnz = view.offsets[n_docs];
    stats->n_blocks += 1;
    stats->n_docs += n_docs;
    stats->nnz += total_nnz;

    const nsparse::detail::InlineBlockOffsets layout =
        nsparse::detail::inline_block_offsets(n_docs, total_nnz, element_size);
    stats->b_prefix += nsparse::detail::kInlineBlockPrefixSize;
    stats->b_doc_ids += n_docs * sizeof(uint32_t);
    stats->b_off += (n_docs + 1) * sizeof(uint32_t);
    stats->b_comps += total_nnz * sizeof(nsparse::term_t);
    stats->b_vals += total_nnz * element_size;
    stats->b_intra_pad +=
        layout.vals - (layout.comps + total_nnz * sizeof(nsparse::term_t));
    stats->b_payload += layout.end;
    stats->b_inter_pad +=
        nsparse::detail::inline_align_up(layout.end, align) - layout.end;

    for (uint32_t i = 0; i < view.n_docs; ++i) {
        stats->c_off_vbyte += vbyte_len(view.nnz(i));
        size_doc_comps(view.doc_comps(i), view.nnz(i), stats);
    }

    // Doc ids: do they ascend, and what would delta + VByte cost? The sizing
    // assumes a writer that sorts them within a block, which today's does not.
    bool ids_sorted = true;
    for (uint64_t i = 1; i < n_docs; ++i) {
        if (view.doc_ids[i] <= view.doc_ids[i - 1]) {
            ids_sorted = false;
            break;
        }
    }
    if (!ids_sorted) {
        stats->blocks_docids_unsorted += 1;
    }
    std::vector<uint32_t> ids(view.doc_ids, view.doc_ids + n_docs);
    std::sort(ids.begin(), ids.end());
    uint64_t id_bytes = n_docs == 0 ? 0 : vbyte_len(ids[0]);
    for (uint64_t i = 1; i < n_docs; ++i) {
        id_bytes += vbyte_len(ids[i] - ids[i - 1]);
    }
    stats->c_docids_vbyte += id_bytes;

    // Offsets: could a u16 offset table address this block's payload?
    constexpr uint64_t kU16Max = 0xFFFF;
    if (layout.end <= kU16Max) {
        stats->blocks_off_fits_u16 += 1;
        stats->c_off_u16 += (n_docs + 1) * sizeof(uint16_t);
    } else {
        stats->c_off_u16 += (n_docs + 1) * sizeof(uint32_t);
    }

    // Value distribution, 8-bit codes only.
    if (element_size == 1) {
        constexpr uint8_t kFourBitMax = 15;
        for (uint64_t j = 0; j < total_nnz; ++j) {
            const uint8_t value = view.vals[j];
            stats->vals_nonzero += value != 0 ? 1 : 0;
            stats->vals_le15 += value <= kFourBitMax ? 1 : 0;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <index.dat>\n", argv[0]);
        return 1;
    }
    const std::string path = argv[1];

    MmapFile file(path, MmapFile::AccessPattern::kScan);
    MmapCursor cursor(file.data(), file.size());
    const uint64_t file_bytes = file.size();

    const auto type_id = cursor.read_scalar<uint32_t>();
    const auto version = cursor.read_scalar<uint32_t>();
    const auto dimension = cursor.read_scalar<int>();
    std::array<char, 5> fourcc{};
    std::memcpy(fourcc.data(), &type_id, 4);
    std::printf("index: %s  type=%s version=%u dim=%d  %llu B (%.3f GiB)\n",
                path.c_str(), fourcc.data(), version, dimension,
                static_cast<unsigned long long>(file_bytes), gib(file_bytes));

    size_t element_size = nsparse::U32;
    if (std::strcmp(fourcc.data(), "DSSQ") == 0) {
        const auto quantizer = cursor.read_scalar<QuantizerType>();
        const auto vmin = cursor.read_scalar<float>();
        const auto vmax = cursor.read_scalar<float>();
        element_size = quantizer == QuantizerType::QT_8bit ? 1 : 2;
        std::printf("quantizer: %s vmin=%g vmax=%g -> element_size=%zu\n",
                    quantizer == QuantizerType::QT_8bit ? "8bit" : "16bit",
                    vmin, vmax, element_size);
    } else if (std::strcmp(fourcc.data(), "DSEI") != 0) {
        std::fprintf(stderr, "not a disk seismic index (got %s)\n",
                     fourcc.data());
        return 1;
    }

    const uint64_t header_end = cursor.pos();
    const auto num_vectors = cursor.read_scalar<uint64_t>();

    SeismicInvertedListsWriter inv_lists;
    inv_lists.mmap_deserialize(&cursor);
    const uint64_t summaries_end = cursor.pos();

    InlineForwardIndex fwd;
    fwd.mmap_deserialize(&cursor);
    const uint64_t fwd_end = cursor.pos();

    std::printf("docs: %llu   posting lists: %llu   blocks: %llu\n",
                static_cast<unsigned long long>(num_vectors),
                static_cast<unsigned long long>(fwd.num_lists()),
                static_cast<unsigned long long>(fwd.num_blocks()));
    std::printf("\n== sections ==\n");
    line("index + quantizer header", header_end, file_bytes);
    // The u64 doc count sits between that header and the summaries.
    line("cluster summaries", summaries_end - header_end - sizeof(uint64_t),
         file_bytes);
    line("inline forward index", fwd_end - summaries_end, file_bytes);
    line("doc locators + remainder", file_bytes - fwd_end, file_bytes);

    const uint64_t align = fwd.page_size();
    Stats stats;
    for (uint32_t pl = 0; pl < fwd.num_lists(); ++pl) {
        const uint64_t n_blocks = fwd.num_blocks_in_list(pl);
        for (uint32_t block = 0; block < n_blocks; ++block) {
            const BlockView view = fwd.block(pl, block);
            if (view.absent()) {
                continue;
            }
            accumulate(view, element_size, align, &stats);
        }
    }
    stats.b_dir = stats.n_blocks * sizeof(nsparse::detail::InlineDirEntry);

    const uint64_t inline_total =
        stats.b_payload + stats.b_inter_pad + stats.b_dir;
    std::printf(
        "\n== inline forward index: %llu blocks, %llu doc slots, %llu nnz "
        "(block alignment %llu) ==\n",
        static_cast<unsigned long long>(stats.n_blocks),
        static_cast<unsigned long long>(stats.n_docs),
        static_cast<unsigned long long>(stats.nnz),
        static_cast<unsigned long long>(align));
    std::printf(
        "  docs/block %.1f   nnz/doc %.1f   copies per doc %.2f\n",
        static_cast<double>(stats.n_docs) / static_cast<double>(stats.n_blocks),
        static_cast<double>(stats.nnz) / static_cast<double>(stats.n_docs),
        static_cast<double>(stats.n_docs) / static_cast<double>(num_vectors));
    line("block prefix", stats.b_prefix, inline_total);
    line("doc_id[] (u32)", stats.b_doc_ids, inline_total);
    line("off[] (u32)", stats.b_off, inline_total);
    line("comps[] (u16)", stats.b_comps, inline_total);
    line("vals[]", stats.b_vals, inline_total);
    line("intra-block pad", stats.b_intra_pad, inline_total);
    line("inter-block pad", stats.b_inter_pad, inline_total);
    line("directory", stats.b_dir, inline_total);
    line("TOTAL", inline_total, inline_total);
    std::printf(
        "  bytes per nnz: %.3f\n",
        static_cast<double>(inline_total) / static_cast<double>(stats.nnz));

    std::printf("\n== encodings not taken ==\n");
    const auto saving = [&](const char* label, uint64_t now, uint64_t then) {
        const int64_t delta =
            static_cast<int64_t>(now) - static_cast<int64_t>(then);
        std::printf(
            "  %-38s %8.3f -> %8.3f GiB   save %8.3f GiB (%5.2f%% of file)\n",
            label, gib(now), gib(then), gib(delta > 0 ? delta : 0),
            100.0 * static_cast<double>(delta) /
                static_cast<double>(file_bytes));
    };
    // Packing leaves each block padded to kMinBlockAlign, so on average half of
    // it -- which is what a page-aligned file would come down to.
    saving("pack blocks (align 8, no page pad)", stats.b_inter_pad,
           stats.n_blocks * (nsparse::detail::kMinBlockAlign / 2));
    saving("comps: delta + VByte", stats.b_comps, stats.c_comps_vbyte);
    saving("comps: delta + width flags", stats.b_comps, stats.c_comps_flagged);
    saving("doc_id: sorted delta + VByte", stats.b_doc_ids,
           stats.c_docids_vbyte);
    saving("off: per-doc nnz as VByte", stats.b_off, stats.c_off_vbyte);
    saving("off: u16 where the block fits", stats.b_off, stats.c_off_u16);
    if (element_size == 1) {
        saving("vals: 4-bit codes", stats.b_vals, stats.nnz / 2);
    }

    std::printf("\n== diagnostics ==\n");
    std::printf("  docs with non-ascending comps: %llu / %llu\n",
                static_cast<unsigned long long>(stats.docs_comps_unsorted),
                static_cast<unsigned long long>(stats.n_docs));
    std::printf("  blocks with non-ascending doc ids: %llu / %llu\n",
                static_cast<unsigned long long>(stats.blocks_docids_unsorted),
                static_cast<unsigned long long>(stats.n_blocks));
    std::printf("  blocks whose payload fits a u16 offset: %llu / %llu\n",
                static_cast<unsigned long long>(stats.blocks_off_fits_u16),
                static_cast<unsigned long long>(stats.n_blocks));
    if (element_size == 1) {
        std::printf("  8-bit codes: %.2f%% nonzero, %.2f%% <= 15\n",
                    100.0 * static_cast<double>(stats.vals_nonzero) /
                        static_cast<double>(stats.nnz),
                    100.0 * static_cast<double>(stats.vals_le15) /
                        static_cast<double>(stats.nnz));
    }
    return 0;
}
