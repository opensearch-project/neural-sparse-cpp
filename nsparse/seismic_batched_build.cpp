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
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/io/file_io.h"
#include "nsparse/io/index_io.h"
#include "nsparse/io/io.h"
#include "nsparse/io/seismic_invlists_writer.h"
#include "nsparse/seismic_common.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/utils/mmap_cursor.h"
#include "nsparse/utils/mmap_file.h"

namespace nsparse::detail {
namespace {

// Both entry points below refuse the same two inputs: nowhere to write to, and
// a corpus with no postings to write.
void throw_if_not_streamable(const SparseVectors* vectors,
                             const std::string& path, const char* who) {
    if (path.empty()) {
        throw std::invalid_argument(std::string(who) +
                                    ": output path must not be empty");
    }
    if (vectors == nullptr || vectors->num_vectors() == 0) {
        throw std::invalid_argument(
            std::string(who) + ": corpus is empty; there is nothing to stream");
    }
}

// The posting-list section -- [count][list...], the layout
// SeismicInvertedListsWriter produces -- streamed into `writer` one window at a
// time, starting wherever the writer has reached.
//
// The writer is the one the whole file is being written through, rather than a
// per-window one whose output is concatenated: serialize() pads each array
// relative to the writer's current offset (see io/align.h), so bytes produced
// by a writer that started at 0 carry the wrong padding once appended at some
// other offset. Streaming through a single writer keeps pos() the true absolute
// offset.
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
                    "stream_clustered_lists: windows arrived out of order");
            }
            for (const auto& list : clusters) {
                list.serialize(writer);
            }
            next_term = term_begin + clusters.size();
            // clusters freed on return, before the next window is built.
        });
    if (next_term != config.dimension) {
        throw std::runtime_error(
            "stream_clustered_lists: wrote " + std::to_string(next_term) +
            " of " + std::to_string(config.dimension) + " posting lists");
    }
}

}  // namespace

size_t write_seismic_index_batched(
    const SparseVectors* vectors, const SparseVectorsConfig& config,
    const SeismicClusterParameters& params, const IndexHeader& header,
    const std::function<void(IOWriter*)>& write_prefix,
    const std::string& out_path) {
    throw_if_not_streamable(vectors, out_path, "write_seismic_index_batched");

    FileIOWriter writer(const_cast<char*>(out_path.c_str()));
    write_header(header, &writer);
    write_prefix(&writer);

    const size_t lists_offset = writer.pos();
    stream_clustered_lists(vectors, config, params, &writer);
    writer.close();
    return lists_offset;
}

std::vector<InvertedListClusters> spill_clustered_lists(
    const SparseVectors* vectors, const SparseVectorsConfig& config,
    const SeismicClusterParameters& params, const std::string& path,
    MmapFile* into) {
    throw_if_not_streamable(vectors, path, "spill_clustered_lists");
    {
        // Closed before the mapping is taken: the writer buffers, and what is
        // not flushed is not in the file to map.
        FileIOWriter writer(const_cast<char*>(path.c_str()));
        stream_clustered_lists(vectors, config, params, &writer);
        writer.close();
    }
    // Offset 0: a spill is the section and nothing else, with no header for it
    // to sit behind.
    return map_streamed_lists(path, /*lists_offset=*/0, into);
}

std::vector<InvertedListClusters> map_streamed_lists(const std::string& path,
                                                     size_t lists_offset,
                                                     MmapFile* into) {
    MmapFile mapped(path);
    // The cursor starts at 0 and skips, rather than mapping from lists_offset:
    // absolute file offsets are what serialize() padded against, so a cursor
    // that began part-way through would compute different padding and misread
    // every array. Same reason mmap_index skips rather than offsets.
    MmapCursor cursor(mapped.data(), mapped.size());
    cursor.skip(lists_offset);
    SeismicInvertedListsWriter lists;
    lists.mmap_deserialize(&cursor);

    // Committed only once the walk succeeded, so a truncated file cannot leave
    // the index holding lists that point into a mapping it never took.
    *into = std::move(mapped);
    return std::move(lists.release());
}

}  // namespace nsparse::detail
