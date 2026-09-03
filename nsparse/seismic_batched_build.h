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

#include <string>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/seismic_common.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/utils/mmap_file.h"

namespace nsparse::detail {

// What a spilled build hands back.
struct SpilledLists {
    // Every term's clustered posting list, in term order -- the same thing
    // build_inverted_lists_clusters returns, borrowed from the spill's mapping
    // rather than allocated.
    std::vector<InvertedListClusters> lists;
    // The spill file, when it is still on disk to be removed. Empty when it was
    // unlinked as soon as it was mapped, which is what happens wherever a
    // mapped file can be unlinked; where it cannot (Windows), whoever holds the
    // mapping has to remove this after releasing it.
    std::string scratch_path;
};

// Clusters the corpus one contiguous term window at a time, spilling each
// window's lists to a temporary file in `scratch_dir` and mapping them back, so
// the caller ends up holding every list without two windows ever having been
// resident at once.
//
// The usual build holds two whole-corpus intermediates -- the inverted lists
// (every posting) and then the clustered posting lists -- so its peak memory
// scales with the corpus's non-zeros, and a corpus whose posting lists do not
// fit in RAM cannot be indexed at all. for_each_clustered_window bounds the
// first to one window; spilling each window's clusters and dropping them, which
// is what this does, bounds the second. What is left resident is the forward
// corpus (which the caller already holds, at whatever residency SparseVectors
// was given) plus one window.
//
// The spill is scratch, not an index: it carries the posting-list section and
// nothing else -- no index header, no forward vectors -- and nothing outside
// this build ever reads it. Writing an index file remains write_index's job,
// from the lists this returns; a build that borrows its lists from scratch
// serializes byte-for-byte what a whole-corpus build would have.
//
// Identical to the unbatched build for a fixed `params.seed`, and identical
// whatever the window count is, because every list's k-means seed comes from
// its own global term id -- see for_each_clustered_window.
//
// `into` takes the mapping the returned lists borrow from, and so must outlive
// them. Throws if `scratch_dir` is not a directory, or if the corpus is empty:
// there would be no windows to spill, and an empty spill maps back to no lists
// at all.
SpilledLists spill_clustered_lists(const SparseVectors* vectors,
                                   const SparseVectorsConfig& config,
                                   const SeismicClusterParameters& params,
                                   const std::string& scratch_dir,
                                   MmapFile* into);

}  // namespace nsparse::detail

#endif  // SEISMIC_BATCHED_BUILD_H
