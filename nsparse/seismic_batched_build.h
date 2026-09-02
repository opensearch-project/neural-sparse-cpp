/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef SEISMIC_BATCHED_BUILD_H
#define SEISMIC_BATCHED_BUILD_H

#include <functional>
#include <string>

#include "nsparse/io/io.h"
#include "nsparse/seismic_common.h"
#include "nsparse/sparse_vectors.h"

namespace nsparse::detail {

// Builds a seismic-family index and writes it straight to `out_path`, one term
// window at a time, without ever holding the whole index in memory.
//
// The usual build holds two whole-corpus intermediates -- the inverted lists and
// then the clustered posting lists -- so its peak memory scales with the
// corpus's non-zeros, and a corpus whose posting lists do not fit in RAM cannot
// be indexed at all. for_each_clustered_window bounds the first to one window;
// serializing each window and dropping it, which is what this does, bounds the
// second. What is left resident is the forward corpus (which the caller already
// holds, at whatever residency SparseVectors was given) plus one window.
//
// Reached through an index's build(), by setting
// SeismicClusterParameters::batch_clustering.batch_file_output_path. The index is
// then the file, not the object: nothing is retained to serve or to write_index
// afterwards.
//
// `header` and `write_prefix` are what make this work for every type in the
// family rather than just SEIS. `write_prefix` writes whatever the type puts
// between the header and its posting lists -- the forward vectors, and for a
// quantizing index its quantization header first. The lists then follow in the
// byte-for-byte layout SeismicInvertedListsWriter produces, so the file is an
// ordinary index of that type: read it back with read_index, mapped or copying,
// exactly as if it had been built in memory and written with write_index.
//
// Identical to the unbatched build for a fixed `params.seed`, and identical
// whatever batch_size is, because every list's k-means seed comes from its own
// global term id -- see for_each_clustered_window.
//
// Throws if the corpus is empty: there would be no windows to stream, and a
// header-only file is not a readable index.
void write_seismic_index_batched(
    const SparseVectors* vectors, const SparseVectorsConfig& config,
    const SeismicClusterParameters& params, const IndexHeader& header,
    const std::function<void(IOWriter*)>& write_prefix,
    const std::string& out_path);

}  // namespace nsparse::detail

#endif  // SEISMIC_BATCHED_BUILD_H
