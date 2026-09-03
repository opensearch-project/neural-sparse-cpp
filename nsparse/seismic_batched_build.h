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

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/seismic_common.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/utils/mmap_file.h"

namespace nsparse::detail {

// The temporary file a batched build spilled its clustered lists to, and the
// mapping those lists borrow from. Owns both, so an index that holds one keeps
// its lists valid and cleans up after itself.
class ClusteredListsSpill {
public:
    ClusteredListsSpill() = default;
    ~ClusteredListsSpill() { release(); }

    ClusteredListsSpill(const ClusteredListsSpill&) = delete;
    ClusteredListsSpill& operator=(const ClusteredListsSpill&) = delete;
    ClusteredListsSpill(ClusteredListsSpill&& other) noexcept
        : path_(std::move(other.path_)), mapping_(std::move(other.mapping_)) {
        other.path_.clear();
    }
    ClusteredListsSpill& operator=(ClusteredListsSpill&& other) noexcept {
        if (this != &other) {
            release();
            path_ = std::move(other.path_);
            mapping_ = std::move(other.mapping_);
            other.path_.clear();
        }
        return *this;
    }

    // Takes a spill this build just wrote and maps it. `path` is unlinked here
    // when the platform allows it while mapped, so a crash cannot strand
    // scratch; otherwise it is remembered and removed on release().
    void adopt(const std::string& path);

    [[nodiscard]] const MmapFile& mapping() const { return mapping_; }

private:
    // Unmaps, then removes the file -- in that order, since Windows cannot
    // unlink a mapped file. Only ever a path adopt() created, and re-checked
    // against the spill naming, because this deletes.
    void release();

    std::string path_;  // empty unless a removal is still owed
    MmapFile mapping_;
};

// The build every seismic-family index runs, and the one place the batching
// decision is made. Clusters one contiguous term window at a time into `spill`
// when both batch knobs are set, whole-corpus otherwise; either way the lists
// come back complete and in term order.
//
// `dimension` is the index's, not the corpus's: it is the list count, and an
// empty corpus still has one. The element width comes from `vectors`.
std::vector<InvertedListClusters> build_clustered_lists(
    const SparseVectors* vectors, size_t dimension,
    const SeismicClusterParameters& params, ClusteredListsSpill* spill);

// Clusters one term window at a time, spilling each window's lists into
// `scratch_dir` and freeing them, then maps the finished lists back out.
//
// A whole-corpus build holds two intermediates that scale with the corpus's
// non-zeros -- the inverted lists and then the clustered lists -- which is what
// puts a ceiling on the corpus an index can be built from.
// for_each_clustered_window bounds the first to a window; spilling bounds the
// second. What stays resident is the corpus plus one window.
//
// The spill is scratch, not an index: posting lists only, no header, read by
// nothing else. Serializing an index remains write_index's job, and at a fixed
// `params.seed` what it then writes is byte-for-byte what a whole-corpus build
// would have produced -- every list's k-means seed comes from its own global
// term id, so the window count cannot leak into the output.
//
// Throws if `scratch_dir` is not a directory, or if the corpus is empty: there
// would be no windows to spill.
std::vector<InvertedListClusters> spill_clustered_lists(
    const SparseVectors* vectors, size_t dimension,
    const SeismicClusterParameters& params, const std::string& scratch_dir,
    ClusteredListsSpill* into);

}  // namespace nsparse::detail

#endif  // SEISMIC_BATCHED_BUILD_H
