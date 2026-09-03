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
#include <cstdint>
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

// How a spill is named. Checked again before removal, so this code can only
// ever delete a file it wrote.
constexpr const char* kSpillPrefix = "nsparse-clustered-lists-";
constexpr const char* kSpillSuffix = ".tmp";

// Unique per build, so builds sharing a scratch directory cannot overwrite each
// other's windows.
std::string spill_path(const std::string& dir) {
    std::random_device entropy;
    const auto token = static_cast<uint64_t>(entropy()) << 32U | entropy();
    return (std::filesystem::path(dir) /
            (kSpillPrefix + std::to_string(token) + kSpillSuffix))
        .string();
}

bool is_spill_path(const std::string& path) {
    const std::string name = std::filesystem::path(path).filename().string();
    return name.starts_with(kSpillPrefix) && name.ends_with(kSpillSuffix);
}

// The posting-list section -- [count][list...], the layout
// SeismicInvertedListsWriter produces -- streamed into `writer` a window at a
// time. One writer for the whole section: serialize() pads each array relative
// to the writer's offset (io/align.h), so separately written windows would
// carry the wrong padding once concatenated.
void stream_clustered_lists(const SparseVectors* vectors, size_t dimension,
                            const SeismicClusterParameters& params,
                            IOWriter* writer) {
    // The whole dimension, known before any window is built, which is what lets
    // the lists follow it rather than be counted first.
    size_t n_lists = dimension;
    writer->write(&n_lists, sizeof(size_t), 1);

    size_t next_term = 0;
    for_each_clustered_window(
        vectors, dimension, params,
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
    if (next_term != dimension) {
        throw std::runtime_error("spill_clustered_lists: spilled " +
                                 std::to_string(next_term) + " of " +
                                 std::to_string(dimension) + " posting lists");
    }
}

}  // namespace

void ClusteredListsSpill::adopt(const std::string& path) {
    MmapFile mapped(path);
    std::error_code failed;
    std::filesystem::remove(path, failed);
    // Committed after the mapping succeeded, so a failed open leaves nothing
    // half-owned.
    release();
    mapping_ = std::move(mapped);
    path_ = failed ? path : std::string();
}

void ClusteredListsSpill::release() {
    if (path_.empty()) {
        return;
    }
    const std::string path = std::move(path_);
    path_.clear();
    mapping_ = MmapFile{};
    if (is_spill_path(path)) {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
}

std::vector<InvertedListClusters> build_clustered_lists(
    const SparseVectors* vectors, size_t dimension,
    const SeismicClusterParameters& params, ClusteredListsSpill* spill) {
    const auto& batch = params.batch_clustering;
    if (batch.effective_batch_size() <= 1) {
        return build_inverted_lists_clusters(vectors, dimension, params);
    }
    return spill_clustered_lists(vectors, dimension, params,
                                 batch.batch_file_output_path, spill);
}

std::vector<InvertedListClusters> spill_clustered_lists(
    const SparseVectors* vectors, size_t dimension,
    const SeismicClusterParameters& params, const std::string& scratch_dir,
    ClusteredListsSpill* into) {
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

    const std::string path = spill_path(scratch_dir);
    {
        // Closed before the file is mapped: the writer buffers.
        FileIOWriter writer(const_cast<char*>(path.c_str()));
        stream_clustered_lists(vectors, dimension, params, &writer);
        writer.close();
    }
    into->adopt(path);

    // The section is the whole file, and absolute offsets are the ones
    // serialize() padded against, so the cursor starts where the writer did.
    MmapCursor cursor(into->mapping().data(), into->mapping().size());
    SeismicInvertedListsWriter lists;
    lists.mmap_deserialize(&cursor);
    return std::move(lists.release());
}

}  // namespace nsparse::detail
