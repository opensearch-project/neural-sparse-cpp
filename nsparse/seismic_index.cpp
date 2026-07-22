/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/seismic_index.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/cluster/random_kmeans.h"
#include "nsparse/exact_matcher.h"
#include "nsparse/id_selector.h"
#include "nsparse/index.h"
#include "nsparse/invlists/inverted_lists.h"
#include "nsparse/io/seismic_invlists_writer.h"
#include "nsparse/seismic_common.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"
#include "nsparse/utils/distance_simd.h"
#include "nsparse/utils/prefetch.h"
#include "nsparse/utils/ranker.h"
#include "nsparse/utils/vector_process.h"
#include "nsparse/utils/visited_set.h"

namespace nsparse {
namespace {

constexpr int kElementSize = U32;

void query_single_inverted_list(
    const SparseVectors* vectors, const InvertedListClusters& cluster_invlist,
    const std::vector<float>& dense, const term_t* q_idx, const float* q_val,
    size_t q_len, std::vector<float>& score_scratch,
    std::vector<uint32_t>& cluster_order, const float heap_factor,
    const bool first_list, const SearchParameters* search_parameters,
    detail::TopKHolder<idx_t>& heap, detail::VisitedSet& visited) {
    // Skip empty clusters
    size_t csize = cluster_invlist.cluster_size();
    if (csize == 0) {
        return;
    }
    const IDSelector* id_selector = search_parameters == nullptr
                                        ? nullptr
                                        : search_parameters->get_id_selector();
    // Query-driven summary scoring via the term-major transpose, avoiding the
    // per-summary gather into the dimension-sized dense buffer. The summaries
    // hold float values, so the query values are passed as their raw bytes.
    cluster_invlist.score_summaries_transposed(
        q_idx, reinterpret_cast<const uint8_t*>(q_val), q_len, score_scratch);
    const std::vector<float>& summary_scores = score_scratch;
    const size_t n_clusters = summary_scores.size();

    const auto& [indptr, indices, values] = vectors->get_all_data();

    // Process one cluster: prune by summary score, then score its docs.
    // Returns false when the early-out fires on the first (sorted) list, which
    // means no later cluster can qualify either — the caller stops iterating.
    auto process_cluster = [&](size_t cluster_id) -> bool {
        const float cluster_score = summary_scores[cluster_id];
        if (heap.full() && (cluster_score * heap_factor < heap.peek_score())) {
            // On the first list clusters are visited in descending score order,
            // so once one falls below the threshold every later one does too.
            return !first_list;
        }
        const auto& docs = cluster_invlist.get_docs(cluster_id);
        const size_t n_docs = docs.size();
        for (size_t i = 0; i < n_docs; ++i) {
            const idx_t doc_id = docs[i];
            // Prefetch one doc ahead, only the leading lines of the upcoming
            // row; the row is contiguous so the hardware streamer pulls the
            // tail, while bounding outstanding software prefetches keeps the
            // line-fill buffers from saturating (measured optimum ~4 lines).
            static constexpr size_t kPrefetchDist1 = 1;
            if (i + kPrefetchDist1 < n_docs) {
                const idx_t nd = docs[i + kPrefetchDist1];
                const idx_t next_start = indptr[nd];
                const size_t next_len = indptr[nd + 1] - next_start;
                static constexpr size_t kPrefetchHeadLines = 4;
                detail::prefetch_vector_head(indices + next_start,
                                             values + next_start, next_len,
                                             kPrefetchHeadLines);
            }
            if (!visited.insert(static_cast<size_t>(doc_id))) {
                continue;
            }
            if (id_selector != nullptr && !id_selector->is_member(doc_id)) {
                continue;
            }
            const idx_t start = indptr[doc_id];
            const size_t len = indptr[doc_id + 1] - start;
            auto score = detail::dot_product_float_dense(
                indices + start, values + start, len, dense.data());
            heap.add(score, doc_id);
        }
        return true;
    };

    if (first_list) {
        // Only the first list is score-ordered; sort a per-thread scratch of
        // narrow (u32) cluster ids instead of allocating a size_t vector per
        // list. Clusters per list stay well under 2^32.
        cluster_order.resize(n_clusters);
        std::iota(cluster_order.begin(), cluster_order.end(), 0U);
        std::ranges::sort(cluster_order, [&](uint32_t a, uint32_t b) {
            return summary_scores[a] > summary_scores[b];
        });
        for (const uint32_t cluster_id : cluster_order) {
            if (!process_cluster(cluster_id)) {
                break;
            }
        }
    } else {
        // Later lists are visited in natural cluster order, so iterate directly
        // — no order array, iota, or sort needed.
        for (size_t cluster_id = 0; cluster_id < n_clusters; ++cluster_id) {
            process_cluster(cluster_id);
        }
    }
}
}  // namespace

SeismicIndex::SeismicIndex(int dim)
    : Index(dim), cluster_parameter_(detail::kDefaultSeismicClusterParams) {}
SeismicIndex::SeismicIndex(int dim, SeismicClusterParameters parameter)
    : Index(dim), cluster_parameter_(parameter) {}

void SeismicIndex::add(idx_t n, const idx_t* indptr, const term_t* indices,
                       const float* values) {
    throw_if_not_positive(n);
    throw_if_any_null(indptr, indices, values);

    size_t indptr_size = n + 1;
    size_t nnz = indptr[n];  // Total non-zeros
    if (vectors_ == nullptr) {
        vectors_ = std::unique_ptr<SparseVectors>(
            new SparseVectors({.element_size = kElementSize,
                               .dimension = static_cast<size_t>(dimension_)}));
    }
    vectors_->add_vectors(indptr, indptr_size, indices, nnz,
                          reinterpret_cast<const uint8_t*>(values),
                          nnz * kElementSize);
}

void SeismicIndex::build() {
    clustered_inverted_lists = std::move(detail::build_inverted_lists_clusters(
        vectors_.get(),
        {.element_size = kElementSize,
         .dimension = static_cast<size_t>(get_dimension())},
        cluster_parameter_));
}

auto SeismicIndex::search(idx_t n, const idx_t* indptr, const term_t* indices,
                          const float* values, int k,
                          SearchParameters* search_parameters)
    -> pair_of_score_id_vectors_t {
    if (vectors_ == nullptr || n == 0) {
        return {std::vector<std::vector<float>>(n),
                std::vector<std::vector<idx_t>>(n)};
    }

    SeismicSearchParameters default_params;
    const auto* parameters =
        search_parameters != nullptr
            ? dynamic_cast<const SeismicSearchParameters*>(search_parameters)
            : &default_params;

    // The input batch CSR (indptr/indices/values) is already absolute-indexed,
    // so we index it directly in the common (no-filter) path — no need to
    // re-materialize a SparseVectors or copy its arrays. Only the exact-match
    // id-selector fast path still needs a SparseVectors view, so build one
    // lazily just for that case.
    if (search_parameters != nullptr) {
        const IDSelector* sel = search_parameters->get_id_selector();
        // should_run_exact_match ignores its queries arg (only inspects the
        // selector + k), so nullptr is fine here.
        if (sel != nullptr && detail::should_run_exact_match(sel, k, nullptr)) {
            size_t indptr_size = n + 1;
            size_t nnz = indptr[n];
            SparseVectors query_vectors(
                {.element_size = kElementSize,
                 .dimension = static_cast<size_t>(dimension_)});
            query_vectors.add_vectors(indptr, indptr_size, indices, nnz,
                                      reinterpret_cast<const uint8_t*>(values),
                                      nnz * kElementSize);
            return detail::ExactMatcher::search(
                vectors_.get(), dynamic_cast<const IDSelectorEnumerable*>(sel),
                &query_vectors, kElementSize, k);
        }
    }

    std::vector<std::vector<float>> result_distances(n);
    std::vector<std::vector<idx_t>> result_labels(n);
    const size_t dim = static_cast<size_t>(dimension_);

    // Per-thread scratch reused across all queries a thread handles: a
    // dimension-sized dense query buffer (kept all-zero between queries via a
    // sparse clear inside single_query) and the visited-doc set. This replaces
    // the previous per-query allocation of both. schedule(dynamic, 64) matches
    // the coarse-chunk scheduling used elsewhere in the codebase.
#pragma omp parallel
    {
        std::vector<float> dense(dim, 0.0F);
        // Generation-stamped visited set over the doc-id domain: O(1) reset per
        // query and a single indexed load per candidate instead of a hashed
        // random probe (the doc loop is memory-bound on random gathers).
        detail::VisitedSet visited(vectors_->num_vectors());
        // Per-thread scratch reused across queries: the per-cluster summary
        // score buffer and the sorted cluster-order buffer (first list only).
        std::vector<float> score_scratch;
        std::vector<uint32_t> cluster_order;

#pragma omp for schedule(dynamic, 64)
        for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
            const idx_t start = indptr[query_idx];
            const size_t len = indptr[query_idx + 1] - start;
            const term_t* q_indices = indices + start;
            const float* q_values = values + start;
            const auto& cuts =
                detail::top_k_tokens(q_indices, q_values, len, parameters->cut);
            auto [distances, labels] =
                single_query(dense, visited, q_indices, q_values, len, cuts, k,
                             parameters->heap_factor, search_parameters,
                             score_scratch, cluster_order);
            result_distances[query_idx] = std::move(distances);
            result_labels[query_idx] = std::move(labels);
        }
    }

    return {result_distances, result_labels};
}

/**
 * @brief query logic per single query, could be run multi-threaded
 *
 * @param dense
 * @param cuts
 * @param k
 * @param heap_factor
 * @return std::pair<std::vector<float>, std::vector<idx_t>>
 */
auto SeismicIndex::single_query(std::vector<float>& dense,
                                detail::VisitedSet& visited,
                                const term_t* q_indices, const float* q_values,
                                size_t q_len, const std::vector<term_t>& cuts,
                                int k, float heap_factor,
                                SearchParameters* search_parameters,
                                std::vector<float>& score_scratch,
                                std::vector<uint32_t>& cluster_order)
    -> pair_of_score_id_vector_t {
    size_t num_docs = vectors_->num_vectors();
    if (num_docs == 0) {
        return {{}, {}};
    }

    // Scatter the query into the reused dense buffer (all-zero on entry).
    for (size_t i = 0; i < q_len; ++i) {
        dense[q_indices[i]] = q_values[i];
    }
    visited.new_query();

    detail::TopKHolder<idx_t> holder(k);
    bool first_list = true;
    for (const auto& term : cuts) {
        if (term >= clustered_inverted_lists.size()) [[unlikely]] {
            continue;
        }
        const auto& cluster_invlist = clustered_inverted_lists[term];
        query_single_inverted_list(vectors_.get(), cluster_invlist, dense,
                                   q_indices, q_values, q_len, score_scratch,
                                   cluster_order, heap_factor, first_list,
                                   search_parameters, holder, visited);
        first_list = false;
    }

    // Restore the dense buffer to all-zero for the next query on this thread
    // (sparse clear over only the dims this query touched).
    for (size_t i = 0; i < q_len; ++i) {
        dense[q_indices[i]] = 0.0F;
    }

    auto [scores, ids] = holder.top_k_items_descending();
    scores.resize(k, -1.0F);
    ids.resize(k, INVALID_IDX);
    return {scores, ids};
}

const SparseVectors* SeismicIndex::get_vectors() const {
    return vectors_.get();
}

void SeismicIndex::write_index(IOWriter* io_writer) {
    // write vectors
    if (vectors_ == nullptr) {
        empty_sparse_vectors.serialize(io_writer);
    } else {
        vectors_->serialize(io_writer);
    }
    SeismicInvertedListsWriter inv_list_writer(clustered_inverted_lists);
    inv_list_writer.serialize(io_writer);
}

void SeismicIndex::read_index(IOReader* io_reader) {
    SparseVectors tmp_vectors;
    tmp_vectors.deserialize(io_reader);
    if (tmp_vectors.num_vectors() > 0) {
        vectors_ = std::make_unique<SparseVectors>(std::move(tmp_vectors));
    }
    SeismicInvertedListsWriter inv_list_writer({});
    inv_list_writer.deserialize(io_reader);
    clustered_inverted_lists = std::move(inv_list_writer.release());
    // The search path indexes a VisitedSet by doc id without a per-candidate
    // bounds check; validate the loaded ids fall in the corpus domain once here
    // so a corrupt/mismatched index fails loudly instead of writing OOB.
    if (vectors_ != nullptr) {
        const size_t num_docs = vectors_->num_vectors();
        for (const auto& cluster_invlist : clustered_inverted_lists) {
            cluster_invlist.validate_doc_ids(num_docs);
        }
    }
}
}  // namespace nsparse