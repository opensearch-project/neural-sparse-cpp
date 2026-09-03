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
#include <memory>
#include <string>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/index.h"
#include "nsparse/io/io.h"
#include "nsparse/seismic_common.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/utils/mmap_file.h"

namespace nsparse::detail {

// Builds a seismic-family index and writes it straight to `out_path`, one term
// window at a time, without ever holding the whole index in memory.
//
// The usual build holds two whole-corpus intermediates -- the inverted lists
// and then the clustered posting lists -- so its peak memory scales with the
// corpus's non-zeros, and a corpus whose posting lists do not fit in RAM cannot
// be indexed at all. for_each_clustered_window bounds the first to one window;
// serializing each window and dropping it, which is what this does, bounds the
// second. What is left resident is the forward corpus (which the caller already
// holds, at whatever residency SparseVectors was given) plus one window.
//
// Reached through an index's build(), by setting
// SeismicClusterParameters::batch_clustering.batch_file_output_path. The build
// then maps the file back (see map_streamed_lists) rather than dropping it, so
// it ends holding the index it wrote.
//
// `header` and `write_prefix` are what make this work for every type whose
// payload ends with its posting lists, rather than just SEIS. `write_prefix`
// writes whatever the type puts
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
// Returns the absolute byte offset of the posting-list section, so the lists
// can be mapped back in without re-parsing everything before them -- see
// map_streamed_lists.
//
// Throws if the corpus is empty: there would be no windows to stream, and a
// header-only file is not a readable index.
size_t write_seismic_index_batched(
    const SparseVectors* vectors, const SparseVectorsConfig& config,
    const SeismicClusterParameters& params, const IndexHeader& header,
    const std::function<void(IOWriter*)>& write_prefix,
    const std::string& out_path);

// Streams every window's clustered posting lists to `path` and then maps them
// back, so the caller ends up holding all of them without two windows ever
// having been resident at once.
//
// For the index types whose payload ends with its posting lists,
// write_seismic_index_batched writes the index itself and there is nothing to
// spill. A DiskSeismic payload is not one of those: its summaries precede an
// inline forward index whose blocks are laid out from the doc-id membership of
// every list, so no window's lists can be dropped before the last window is
// clustered. What can be dropped is their *residency* -- which is what this is
// for. `path` gets the lists in the same [count][list...] layout
// SeismicInvertedListsWriter produces, doc ids included (an index's own section
// writes them empty; the forward index is what needs them here), and the
// returned lists borrow from the mapping handed to `into` rather than the heap.
//
// The spill is scratch, not an index: it carries no header, nothing else reads
// it, and deleting it is the caller's job. It must outlive the returned lists.
//
// Throws if the corpus is empty, for the same reason as
// write_seismic_index_batched: there would be no windows to stream.
std::vector<InvertedListClusters> spill_clustered_lists(
    const SparseVectors* vectors, const SparseVectorsConfig& config,
    const SeismicClusterParameters& params, const std::string& path,
    MmapFile* into);

// Maps the file a streamed build just wrote and borrows its posting lists out
// of it, so the build ends holding a usable index without ever having held all
// of the lists at once.
//
// Only the lists. The forward vectors in the file are a copy of ones the index
// already has, at whatever residency the caller chose for them, so re-reading
// them would be work for nothing -- and it is why the corpus mapping can be
// left alone rather than swapped out. `lists_offset` is what
// write_seismic_index_batched returned, which saves parsing past the vectors to
// find where the lists start.
//
// The mapping is handed to `into`, which must outlive the returned lists: they
// point into it.
std::vector<InvertedListClusters> map_streamed_lists(const std::string& path,
                                                     size_t lists_offset,
                                                     MmapFile* into);

}  // namespace nsparse::detail

#endif  // SEISMIC_BATCHED_BUILD_H
