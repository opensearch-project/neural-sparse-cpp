/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef DISK_SEISMIC_SEARCH_H
#define DISK_SEISMIC_SEARCH_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/disk_seismic_index.h"  // DiskSeismicSearchParameters
#include "nsparse/id_selector.h"
#include "nsparse/index.h"  // SearchParameters, pair_of_score_id_vector(s)_t
#include "nsparse/io/inline_forward_index_io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

// Search machinery shared by the two disk-resident SEISMIC indexes,
// DiskSeismicIndex (float) and DiskSeismicScalarQuantizedIndex (8-/16-bit
// codes). They differ only in the value width, so the block-budget search is
// one element_size-parameterized routine here rather than a copy in each.
namespace nsparse::detail {

// A candidate block: its summary score and its (posting list, cluster) address.
struct BlockCandidate {
    float score;
    uint32_t pl;
    uint32_t cid;
};

// (cut, k_prime) resolved from search parameters, defaulting to
// DiskSeismicSearchParameters. Each index still validates k_prime with its own
// error message.
struct DiskSeismicCutBudget {
    int cut;
    int k_prime;
};
DiskSeismicCutBudget resolve_cut_and_budget(
    const SearchParameters* search_parameters);

// The `cut` highest-weighted query terms, read from `codes` at `element_size`
// bytes per value (4 = float, 2 = uint16, 1 = uint8).
std::vector<term_t> top_cut_tokens(const term_t* indices, const uint8_t* codes,
                                   size_t len, int cut, size_t element_size);

// The all-padding result grid a search returns when there is nothing to score.
// Distances -1.0, labels INVALID_IDX.
pair_of_score_id_vectors_t initialize_padded_results(idx_t n, int k);

// Block-budget query core. The query must already be encoded at `element_size`
// bytes per component in `query_values` -- raw float (width 4) for the plain
// index, quantized codes (width 1 or 2) for the quantized one; the width also
// selects the dot-product kernel. `dense` is a dimension*element_size byte
// buffer, all-zero on entry and restored to all-zero on exit; `visited`,
// `candidates`, and `score_scratch` are per-thread scratch reused across
// queries (cleared/overwritten here). Blocks come from `fwd` (the mapping) when
// it is non-null, else from the in-RAM `vectors` of a fresh build. Returns the
// raw (undecoded) top-k scores + ids, unpadded: the caller decodes if it must
// and pads to k.
pair_of_score_id_vector_t block_budget_query(
    uint8_t* dense, size_t element_size, absl::flat_hash_set<idx_t>& visited,
    std::vector<BlockCandidate>& candidates, std::vector<float>& score_scratch,
    const term_t* query_indices, const uint8_t* query_values, size_t query_len,
    const std::vector<term_t>& cuts, int k, int k_prime,
    const std::vector<InvertedListClusters>& clusters,
    const InlineForwardIndex* fwd, const SparseVectors* vectors,
    const IDSelector* id_selector);

}  // namespace nsparse::detail

#endif  // DISK_SEISMIC_SEARCH_H
