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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/io/file_io.h"
#include "nsparse/io/index_io.h"
#include "nsparse/io/io.h"
#include "nsparse/seismic_common.h"
#include "nsparse/sparse_vectors.h"

namespace nsparse::detail {

void write_seismic_index_batched(
    const SparseVectors* vectors, const SparseVectorsConfig& config,
    const SeismicClusterParameters& params, const IndexHeader& header,
    const std::function<void(IOWriter*)>& write_prefix,
    const std::string& out_path) {
    if (out_path.empty()) {
        throw std::invalid_argument(
            "write_seismic_index_batched: output path must not be empty");
    }
    if (vectors == nullptr || vectors->num_vectors() == 0) {
        throw std::invalid_argument(
            "write_seismic_index_batched: corpus is empty; there is nothing to "
            "stream");
    }

    // One writer for the whole file, windows serialized straight into it rather
    // than spilled and concatenated: serialize() pads each array relative to the
    // writer's current offset (see io/align.h), so bytes produced by a writer
    // that started at 0 carry the wrong padding once appended at some other
    // offset. Streaming through a single writer keeps pos() the true absolute
    // offset.
    FileIOWriter writer(const_cast<char*>(out_path.c_str()));
    write_header(header, &writer);
    write_prefix(&writer);

    // The list count, exactly where SeismicInvertedListsWriter::serialize puts
    // it. It is the whole dimension, known before any window is built, which is
    // what lets the lists be streamed after it rather than counted first.
    size_t n_lists = config.dimension;
    writer.write(&n_lists, sizeof(size_t), 1);

    // Windows arrive in ascending term order, so appending each in turn produces
    // the same byte sequence as writing every list at once.
    size_t next_term = 0;
    for_each_clustered_window(
        vectors, config, params,
        [&](size_t term_begin, std::vector<InvertedListClusters>&& clusters) {
            if (term_begin != next_term) {
                // The layout carries no per-list offsets, so a gap or a repeat
                // would silently shift every list after it.
                throw std::runtime_error(
                    "write_seismic_index_batched: windows arrived out of order");
            }
            for (const auto& list : clusters) {
                list.serialize(&writer);
            }
            next_term = term_begin + clusters.size();
            // clusters freed on return, before the next window is built.
        });
    if (next_term != config.dimension) {
        throw std::runtime_error(
            "write_seismic_index_batched: wrote " + std::to_string(next_term) +
            " of " + std::to_string(config.dimension) + " posting lists");
    }
    writer.close();
}

}  // namespace nsparse::detail
