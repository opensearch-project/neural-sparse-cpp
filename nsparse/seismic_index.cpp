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

#include "absl/container/flat_hash_set.h"
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

namespace nsparse {
namespace {

constexpr int kElementSize = U32;

void query_single_inverted_list(const SparseVectors* vectors,
                                const InvertedListClusters& cluster_invlist,
                                const std::vector<float>& dense,
                                const float heap_factor, const bool first_list,
                                const SearchParameters* search_parameters,
                                detail::TopKHolder<idx_t>& heap,
                                absl::flat_hash_set<idx_t>& visited) {
    // Skip empty clusters
    size_t csize = cluster_invlist.cluster_size();
    if (csize == 0) {
        return;
    }
    const IDSelector* id_selector = search_parameters == nullptr
                                        ? nullptr
                                        : search_parameters->get_id_selector();
    const auto& summaries = cluster_invlist.summaries();
    // compute dp with all summaries
    auto summary_scores =
        detail::dot_product_float_vectors_dense(&summaries, dense.data());
    size_t num_vectors = vectors->num_vectors();

    std::vector<size_t> cluster_order =
        detail::reorder_clusters(summary_scores, first_list);

    const auto& [indptr, indices, values] = vectors->get_all_data();

    for (const size_t& cluster_id : cluster_order) {
        const auto& cluster_score = summary_scores[cluster_id];
        if (heap.full() && (cluster_score * heap_factor < heap.peek_score())) {
            if (first_list) {
                break;
            }
            continue;
        }
        const auto& docs = cluster_invlist.get_docs(cluster_id);
        const size_t n_docs = docs.size();
        static constexpr size_t kPrefetchDist1 = 2;  // vector data prefetch
        static constexpr size_t kPrefetchDist2 = 4;  // indptr prefetch
        for (size_t i = 0; i < n_docs; ++i) {
            const auto& doc_id = docs[i];
            if (i + kPrefetchDist2 < n_docs) {
                detail::prefetch_indptr(indptr, docs[i + kPrefetchDist2]);
            }
            if (i + kPrefetchDist1 < n_docs) {
                const idx_t next_doc = docs[i + kPrefetchDist1];
                const idx_t next_start = indptr[next_doc];
                const size_t next_len = indptr[next_doc + 1] - next_start;
                detail::prefetch_vector(indices + next_start,
                                        values + next_start, next_len);
            }
            auto [_, inserted] = visited.insert(doc_id);
            if (!inserted) {
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
    size_t indptr_size = n + 1;
    size_t nnz = indptr[n];  // Total non-zeros
    // Create query vectors from input
    SparseVectors query_vectors({.element_size = kElementSize,
                                 .dimension = static_cast<size_t>(dimension_)});
    query_vectors.add_vectors(indptr, indptr_size, indices, nnz,
                              reinterpret_cast<const uint8_t*>(values),
                              nnz * kElementSize);

    SeismicSearchParameters default_params;
    const auto* parameters =
        search_parameters != nullptr
            ? dynamic_cast<const SeismicSearchParameters*>(search_parameters)
            : &default_params;

    // if filter ids size is <= k, just run exact match
    if (search_parameters != nullptr &&
        detail::should_run_exact_match(search_parameters->get_id_selector(), k,
                                       &query_vectors)) {
        return detail::ExactMatcher::search(
            vectors_.get(),
            dynamic_cast<const IDSelectorEnumerable*>(
                search_parameters->get_id_selector()),
            &query_vectors, kElementSize, k);
    }

    std::vector<std::vector<float>> result_distances(n);
    std::vector<std::vector<idx_t>> result_labels(n);
    // For each query vector
    const auto* query_indptr = query_vectors.indptr_data();
    const auto* query_indices = query_vectors.indices_data();
    const auto* query_values = query_vectors.values_data_float();

#pragma omp parallel for
    for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
        const auto& dense = query_vectors.get_dense_vector_float(query_idx);
        const idx_t start = query_indptr[query_idx];
        const size_t len = query_indptr[query_idx + 1] - start;
        const auto& cuts = detail::top_k_tokens(
            query_indices + start, query_values + start, len, parameters->cut);
        auto [distances, labels] = single_query(
            dense, cuts, k, parameters->heap_factor, search_parameters);
        result_distances[query_idx] = std::move(distances);
        result_labels[query_idx] = std::move(labels);
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
auto SeismicIndex::single_query(const std::vector<float>& dense,
                                const std::vector<term_t>& cuts, int k,
                                float heap_factor,
                                SearchParameters* search_parameters)
    -> pair_of_score_id_vector_t {
    size_t num_docs = vectors_->num_vectors();
    if (num_docs == 0) {
        return {{}, {}};
    }
    absl::flat_hash_set<idx_t> visited;
    visited.reserve(cuts.size() * 5000);
    detail::TopKHolder<idx_t> holder(k);
    bool first_list = true;
    for (const auto& term : cuts) {
        if (term >= clustered_inverted_lists.size()) [[unlikely]] {
            continue;
        }
        const auto& cluster_invlist = clustered_inverted_lists[term];
        query_single_inverted_list(vectors_.get(), cluster_invlist, dense,
                                   heap_factor, first_list, search_parameters,
                                   holder, visited);
        first_list = false;
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
}
}  // namespace nsparse