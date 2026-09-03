/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/seismic_batched_build.h"

#include <cstddef>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/io/file_io.h"
#include "nsparse/io/io.h"
#include "nsparse/io/seismic_invlists_writer.h"
#include "nsparse/seismic_common.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/utils/mmap_cursor.h"
#include "nsparse/utils/mmap_file.h"

namespace nsparse::detail {
namespace {

// A spill file of this build's own inside `dir`. Named uniquely rather than
// fixed, so concurrent builds sharing a scratch directory cannot overwrite each
// other's windows.
std::string scratch_file_path(const std::string& dir) {
    const auto token = static_cast<uint64_t>(std::random_device{}()) << 32U |
                       std::random_device{}();
    return (std::filesystem::path(dir) /
            ("nsparse-clustered-lists-" + std::to_string(token) + ".tmp"))
        .string();
}

// The posting-list section -- [count][list...], the layout
// SeismicInvertedListsWriter produces -- streamed into `writer` one window at a
// time.
//
// One writer for the whole section, windows serialized straight into it rather
// than written separately and concatenated: serialize() pads each array
// relative to the writer's current offset (see io/align.h), so bytes produced
// by a writer that started at 0 carry the wrong padding once appended at some
// other offset.
void stream_clustered_lists(const SparseVectors* vectors,
                            const SparseVectorsConfig& config,
                            const SeismicClusterParameters& params,
                            IOWriter* writer) {
    // The list count, exactly where SeismicInvertedListsWriter::serialize puts
    // it. It is the whole dimension, known before any window is built, which is
    // what lets the lists be streamed after it rather than counted first.
    size_t n_lists = config.dimension;
    writer->write(&n_lists, sizeof(size_t), 1);

    // Windows arrive in ascending term order, so appending each in turn
    // produces the same byte sequence as writing every list at once.
    size_t next_term = 0;
    for_each_clustered_window(
        vectors, config, params,
        [&](size_t term_begin, std::vector<InvertedListClusters>&& clusters) {
            if (term_begin != next_term) {
                // The layout carries no per-list offsets, so a gap or a repeat
                // would silently shift every list after it.
                throw std::runtime_error(
                    "spill_clustered_lists: windows arrived out of order");
            }
            for (const auto& list : clusters) {
                list.serialize(writer);
            }
            next_term = term_begin + clusters.size();
            // clusters freed on return, before the next window is built.
        });
    if (next_term != config.dimension) {
        throw std::runtime_error(
            "spill_clustered_lists: spilled " + std::to_string(next_term) +
            " of " + std::to_string(config.dimension) + " posting lists");
    }
}

}  // namespace

SpilledLists spill_clustered_lists(const SparseVectors* vectors,
                                   const SparseVectorsConfig& config,
                                   const SeismicClusterParameters& params,
                                   const std::string& scratch_dir,
                                   MmapFile* into) {
    if (!std::filesystem::is_directory(scratch_dir)) {
        throw std::invalid_argument(
            "spill_clustered_lists: batch_file_output_path must be an existing "
            "directory to spill into, got '" +
            scratch_dir + "'");
    }
    if (vectors == nullptr || vectors->num_vectors() == 0) {
        throw std::invalid_argument(
            "spill_clustered_lists: corpus is empty; there is nothing to "
            "spill");
    }

    const std::string path = scratch_file_path(scratch_dir);
    {
        // Closed before the mapping is taken: the writer buffers, and what is
        // not flushed is not in the file to map.
        FileIOWriter writer(const_cast<char*>(path.c_str()));
        stream_clustered_lists(vectors, config, params, &writer);
        writer.close();
    }

    MmapFile mapped(path);
    // The cursor starts at the section, which is the whole file: a spill has no
    // header for it to sit behind. Absolute offsets are what serialize() padded
    // against, and here they are the section's own.
    MmapCursor cursor(mapped.data(), mapped.size());
    SeismicInvertedListsWriter lists;
    lists.mmap_deserialize(&cursor);

    // Unlinked now rather than when the index is done with it: the mapping
    // keeps the bytes alive wherever unlinking an open file is allowed, so
    // scratch cannot outlive the process even if it dies mid-build. Where it is
    // not allowed, the path goes back to the caller to remove after the
    // mapping.
    std::error_code failed;
    std::filesystem::remove(path, failed);

    SpilledLists spilled;
    spilled.lists = std::move(lists.release());
    spilled.scratch_path = failed ? path : std::string();
    // Committed once the walk succeeded, so a truncated spill cannot leave the
    // caller holding lists that point into a mapping it never took.
    *into = std::move(mapped);
    return spilled;
}

}  // namespace nsparse::detail
