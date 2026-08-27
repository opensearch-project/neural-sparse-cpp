/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/disk_seismic_search.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/id_selector.h"
#include "nsparse/index.h"
#include "nsparse/io/inline_forward_index_io.h"
#include "nsparse/seismic_index.h"  // SeismicSearchParameters
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/distance_simd.h"
#include "nsparse/utils/ranker.h"
#include "nsparse/utils/vector_process.h"

namespace nsparse::detail {
namespace {

// Score one doc's vector against the dense query and push it, honoring
// visited-dedup and the id selector. `dense` and `vals` are element_size bytes
// per component; the width selects the kernel (4 = float, 2 = uint16, else
// uint8). The score is the raw dot product; a quantized caller decodes later.
inline void score_doc(idx_t doc_id, const term_t* comps, const uint8_t* vals,
                      size_t nnz, const uint8_t* dense, size_t element_size,
                      const IDSelector* id_selector, TopKHolder<idx_t>& heap,
                      absl::flat_hash_set<idx_t>& visited) {
    if (!visited.insert(doc_id).second) {
        return;
    }
    if (id_selector != nullptr && !id_selector->is_member(doc_id)) {
        return;
    }
    float score = 0.0F;
    if (element_size == U32) {
        score = dot_product_float_dense(
            comps, reinterpret_cast<const float*>(vals), nnz,
            reinterpret_cast<const float*>(dense));
    } else if (element_size == U16) {
        score = dot_product_uint16_dense(
            comps, reinterpret_cast<const uint16_t*>(vals), nnz,
            reinterpret_cast<const uint16_t*>(dense));
    } else {
        score = dot_product_uint8_dense(comps, vals, nnz, dense);
    }
    heap.add(score, doc_id);
}

// Score every document of one block against the dense query, dedup via
// `visited` and honor the id selector. Vectors come from the inline forward
// index (fwd) when loaded, else from the in-RAM CSR of a fresh build.
void score_block(const InlineForwardIndex* fwd, const SparseVectors* vectors,
                 const std::vector<InvertedListClusters>& clusters, uint32_t pl,
                 uint32_t cid, const uint8_t* dense, size_t element_size,
                 const IDSelector* id_selector, TopKHolder<idx_t>& heap,
                 absl::flat_hash_set<idx_t>& visited) {
    if (fwd != nullptr) {
        const BlockView bv = fwd->block(pl, cid);
        if (bv.absent()) {
            return;
        }
        for (uint32_t i = 0; i < bv.n_docs; ++i) {
            score_doc(static_cast<idx_t>(bv.doc_ids[i]), bv.doc_comps(i),
                      bv.doc_vals(i, element_size), bv.nnz(i), dense,
                      element_size, id_selector, heap, visited);
        }
    } else if (vectors != nullptr) {
        const idx_t* const indptr = vectors->indptr_data();
        const term_t* const indices = vectors->indices_data();
        const uint8_t* const values = vectors->values_data();
        for (const idx_t doc_id : clusters[pl].get_docs(cid)) {
            const idx_t start = indptr[doc_id];
            const size_t len = indptr[doc_id + 1] - start;
            score_doc(doc_id, indices + start,
                      values + static_cast<size_t>(start) * element_size, len,
                      dense, element_size, id_selector, heap, visited);
        }
    }
}

}  // namespace

DiskSeismicCutBudget resolve_cut_and_budget(
    const SearchParameters* search_parameters) {
    // A DiskSeismicSearchParameters (including a quantized subclass) carries
    // k_prime; a plain SeismicSearchParameters (or null) uses the default
    // budget.
    const DiskSeismicSearchParameters defaults;
    int cut = defaults.cut;
    int k_prime = defaults.k_prime;
    if (const auto* disk_parameters =
            dynamic_cast<const DiskSeismicSearchParameters*>(
                search_parameters)) {
        cut = disk_parameters->cut;
        k_prime = disk_parameters->k_prime;
    } else if (const auto* seismic_parameters =
                   dynamic_cast<const SeismicSearchParameters*>(
                       search_parameters)) {
        cut = seismic_parameters->cut;
    }
    return {cut, k_prime};
}

std::vector<term_t> top_cut_tokens(const term_t* indices, const uint8_t* codes,
                                   size_t len, int cut, size_t element_size) {
    if (element_size == U32) {
        return top_k_tokens<float>(
            indices, reinterpret_cast<const float*>(codes), len, cut);
    }
    if (element_size == U16) {
        return top_k_tokens<uint16_t>(
            indices, reinterpret_cast<const uint16_t*>(codes), len, cut);
    }
    return top_k_tokens<uint8_t>(indices, codes, len, cut);
}

pair_of_score_id_vectors_t initialize_padded_results(idx_t n, int k) {
    return {std::vector<std::vector<float>>(n, std::vector<float>(k, -1.0F)),
            std::vector<std::vector<idx_t>>(
                n, std::vector<idx_t>(k, INVALID_IDX))};
}

pair_of_score_id_vector_t block_budget_query(
    uint8_t* dense, size_t element_size, absl::flat_hash_set<idx_t>& visited,
    std::vector<BlockCandidate>& candidates, std::vector<float>& score_scratch,
    const term_t* query_indices, const uint8_t* query_values, size_t query_len,
    const std::vector<term_t>& cuts, int k, int k_prime,
    const std::vector<InvertedListClusters>& clusters,
    const InlineForwardIndex* fwd, const SparseVectors* vectors,
    const IDSelector* id_selector) {
    // Scatter the query into the dense lookup table: element_size contiguous
    // bytes per non-zero dim.
    for (size_t i = 0; i < query_len; ++i) {
        std::copy_n(
            query_values + i * element_size, element_size,
            dense + static_cast<size_t>(query_indices[i]) * element_size);
    }
    visited.clear();

    // Collect every candidate block across the cut lists with its summary
    // score. score_summaries_transposed dots the query with each cluster
    // summary (stored at the same width), so scores are comparable across
    // posting lists for the global ranking.
    candidates.clear();
    for (const term_t term : cuts) {
        if (term >= clusters.size()) [[unlikely]] {
            continue;
        }
        const InvertedListClusters& cluster_invlist = clusters[term];
        const size_t n_clusters = cluster_invlist.cluster_size();
        if (n_clusters == 0) {
            continue;
        }
        cluster_invlist.score_summaries_transposed(query_indices, query_values,
                                                    query_len, score_scratch);
        for (size_t cid = 0; cid < n_clusters; ++cid) {
            candidates.push_back(
                {score_scratch[cid], term, static_cast<uint32_t>(cid)});
        }
    }

    // Select the global top-k_prime blocks by summary score. The order among
    // them does not affect the result (dedup + exact scoring are
    // order-independent).
    const size_t budget =
        std::min(static_cast<size_t>(k_prime), candidates.size());
    if (budget < candidates.size()) {
        std::nth_element(candidates.begin(), candidates.begin() + budget,
                         candidates.end(),
                         [](const BlockCandidate& a, const BlockCandidate& b) {
                             return a.score > b.score;
                         });
        candidates.resize(budget);
    }

    // Score the selected blocks; visited dedups docs shared across them.
    TopKHolder<idx_t> holder(k);
    for (const BlockCandidate& candidate : candidates) {
        score_block(fwd, vectors, clusters, candidate.pl, candidate.cid, dense,
                    element_size, id_selector, holder, visited);
    }

    // Restore only the query's own positions to zero (mirrors the scatter at
    // entry): element_size bytes per touched dim. The scattered writes are
    // bounded by query_len and negligible next to the block scoring.
    for (size_t i = 0; i < query_len; ++i) {
        std::fill_n(
            dense + static_cast<size_t>(query_indices[i]) * element_size,
            element_size, uint8_t{0});
    }

    return holder.top_k_items_descending();
}

}  // namespace nsparse::detail
