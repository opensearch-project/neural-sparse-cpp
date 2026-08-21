/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef DISK_SEISMIC_INDEX_H
#define DISK_SEISMIC_INDEX_H
#include <array>
#include <cstddef>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/io/inline_forward_index_io.h"
#include "nsparse/io/io.h"
#include "nsparse/mmap_index.h"
#include "nsparse/seismic_common.h"
#include "nsparse/seismic_index.h"  // SeismicSearchParameters
#include "nsparse/types.h"

namespace nsparse {

// Default block budget K': the primary recall/latency knob.
inline constexpr int kDefaultBlockBudget = 50;

// `cut` (inherited) bounds which posting lists' summaries are scored; `k_prime`
// caps how many blocks are read (the k_prime highest-scoring ones). Replaces
// the inherited heap_factor, which DiskSeismicIndex ignores.
struct DiskSeismicSearchParameters : public SeismicSearchParameters {
    int k_prime = kDefaultBlockBudget;
    DiskSeismicSearchParameters() = default;
    DiskSeismicSearchParameters(int cut, int k_prime)
        : SeismicSearchParameters(cut, /*heap_factor=*/1.0F),
          k_prime(k_prime) {}
};

// A seismic index whose per-document forward vectors live on disk in the
// block-contiguous (inline) layout of io/inline_forward_index_io.h, borrowed
// via mmap at search time; the cluster summaries stay in RAM. Search scores
// every candidate block's summary across the `cut` posting lists, selects the
// global top-k_prime blocks, and reads and scores only those. Pass a
// DiskSeismicSearchParameters to set k_prime, else kDefaultBlockBudget is used.
//
// A block's vectors come from fwd_.block() (the mapping) once loaded, else from
// the in-RAM SparseVectors of a fresh build.
//
// mmap-only: load with read_index(file, kUseMmap); the copying read throws.
class DiskSeismicIndex : public MmapIndex, public IndexIO {
public:
    static constexpr std::array<char, 4> name = {'D', 'S', 'E', 'I'};

    explicit DiskSeismicIndex(int dim);
    DiskSeismicIndex(int dim, SeismicClusterParameters parameter);
    ~DiskSeismicIndex() override = default;
    std::array<char, 4> id() const override { return name; }

    DiskSeismicIndex(const DiskSeismicIndex&) = delete;
    DiskSeismicIndex& operator=(const DiskSeismicIndex&) = delete;

    void add(idx_t n, const idx_t* indptr, const term_t* indices,
             const float* values) override;
    void build() override;

    // Persisted, since a mapped index has no in-RAM vectors_ to derive it from.
    size_t num_vectors() const override { return num_vectors_; }

    // Borrows a serialized index from a file mapping. `pos` is where the
    // payload begins.
    static DiskSeismicIndex* mmap_index(int dimension, const char* index_file,
                                        size_t pos);

protected:
    std::vector<InvertedListClusters> clustered_inverted_lists;

private:
    void write_index(IOWriter* io_writer) override;
    // Unsupported: the inline forward index is mmap-only. Throws.
    void read_index(IOReader* io_reader, int io_flags = 0) override;

    auto search(idx_t n, const idx_t* indptr, const term_t* indices,
                const float* values, int k,
                SearchParameters* search_parameters = nullptr)
        -> pair_of_score_id_vectors_t override;

    // Selects the global top-k_prime blocks across `cuts` by summary score and
    // scores their docs. `dense` is per-thread scratch, zero on entry and
    // restored on exit; `visited` is cleared on entry.
    auto single_query(std::vector<float>& dense,
                      absl::flat_hash_set<idx_t>& visited,
                      const term_t* q_indices, const float* q_values,
                      size_t q_len, const std::vector<term_t>& cuts, int k,
                      int k_prime, SearchParameters* search_parameters)
        -> pair_of_score_id_vector_t;

    SeismicClusterParameters cluster_parameter_;
    // Borrows from the base's mapped_file_, so it is declared here (destroyed
    // before the mapping). Empty after a fresh build (search uses vectors_
    // then).
    detail::InlineForwardIndex fwd_;
    size_t num_vectors_ = 0;
};
}  // namespace nsparse

#endif  // DISK_SEISMIC_INDEX_H
