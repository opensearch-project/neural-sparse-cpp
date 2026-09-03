/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/disk_seismic_index_base.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/disk_seismic_search.h"
#include "nsparse/id_selector.h"
#include "nsparse/index.h"
#include "nsparse/io/inline_forward_index_io.h"
#include "nsparse/io/seismic_invlists_writer.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"
#include "nsparse/utils/mmap_cursor.h"
#include "nsparse/utils/mmap_file.h"

namespace nsparse {

DiskSeismicIndexBase::DiskSeismicIndexBase(int dim,
                                           SeismicClusterParameters parameter)
    : MmapIndex(dim), cluster_parameter_(parameter) {}

void DiskSeismicIndexBase::add(idx_t n, const idx_t* indptr,
                               const term_t* indices, const float* values) {
    throw_if_not_positive(n);
    throw_if_any_null(indptr, indices, values);
    const size_t indptr_size = n + 1;
    const size_t nnz = indptr[n];
    const size_t element_size = code_element_size();
    if (vectors_ == nullptr) {
        // Fresh container: start the count at 0 so a stale num_vectors_ (e.g.
        // left by a prior mmap load, which has no vectors_) cannot accumulate.
        num_vectors_ = 0;
        vectors_ = std::make_unique<SparseVectors>(SparseVectorsConfig{
            .element_size = element_size,
            .dimension = static_cast<size_t>(dimension_)});
    }
    std::vector<uint8_t> scratch;
    const uint8_t* codes = encode_values(values, nnz, scratch);
    vectors_->add_vectors(indptr, indptr_size, indices, nnz, codes,
                          nnz * element_size);
    num_vectors_ += n;
}

void DiskSeismicIndexBase::build() {
    clustered_inverted_lists = build_clustered_lists(
        {.element_size = code_element_size(),
         .dimension = static_cast<size_t>(get_dimension())},
        cluster_parameter_);
}

auto DiskSeismicIndexBase::search(idx_t n, const idx_t* indptr,
                                  const term_t* indices, const float* values,
                                  int k, SearchParameters* search_parameters)
    -> pair_of_score_id_vectors_t {
    // Quit early when there is nothing to score: no vectors, no queries, or no
    // forward-vector source (fwd_ empty and vectors_ null — a corrupt or
    // uninitialized index).
    if (num_vectors_ == 0 || n == 0 ||
        (fwd_.num_blocks() == 0 && vectors_ == nullptr)) {
        return detail::initialize_padded_results(n, k);
    }

    // The id-selector exact-match fast path is omitted (it needs an in-RAM
    // SparseVectors a mapped index lacks); the selector is still honored per
    // candidate doc inside the shared search core.
    const detail::DiskSeismicCutBudget budget =
        detail::resolve_cut_and_budget(search_parameters);
    const int cut = budget.cut;
    const int k_prime = budget.k_prime;
    // k_prime is a block budget, not a document count: a block holds many docs,
    // so k_prime < k is valid (a few blocks can still fill k, and a short-fall
    // is padded like any under-budget search). Only a non-positive budget is
    // rejected.
    if (k_prime <= 0) {
        throw std::invalid_argument(
            "DiskSeismic index: k_prime (block budget) must be positive");
    }

    // Encode the whole query batch once at the stored width; a query's codes
    // start at query_batch + start * element_size.
    const size_t element_size = code_element_size();
    const size_t nnz = indptr[n];
    std::vector<uint8_t> query_scratch;
    const uint8_t* query_batch =
        encode_query(values, nnz, search_parameters, query_scratch);

    // Rows are filled below, so start them empty rather than paying an n*k
    // padding fill only to overwrite it.
    std::vector<std::vector<float>> result_distances(n);
    std::vector<std::vector<idx_t>> result_labels(n);

    const detail::InlineForwardIndex* fwd =
        fwd_.num_blocks() > 0 ? &fwd_ : nullptr;
    const SparseVectors* vectors = fwd == nullptr ? vectors_.get() : nullptr;
    const IDSelector* id_selector = search_parameters == nullptr
                                        ? nullptr
                                        : search_parameters->get_id_selector();
    const size_t dense_bytes = static_cast<size_t>(dimension_) * element_size;

#pragma omp parallel
    {
        // Per-thread scratch reused across the queries a thread handles: the
        // dense lookup table, the visited-doc set, and the block-candidate /
        // summary-score buffers, so no query allocates on the hot path.
        std::vector<uint8_t> dense(dense_bytes, 0);
        absl::flat_hash_set<idx_t> visited;
        visited.reserve(static_cast<size_t>(std::max(k, 1)) * 4096);
        std::vector<detail::BlockCandidate> candidates;
        std::vector<float> score_scratch;

#pragma omp for schedule(dynamic, 64)
        for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
            const idx_t start = indptr[query_idx];
            const size_t len = indptr[query_idx + 1] - start;
            const term_t* query_indices = indices + start;
            const uint8_t* query_codes =
                query_batch + static_cast<size_t>(start) * element_size;
            const std::vector<term_t> cuts = detail::top_cut_tokens(
                query_indices, query_codes, len, cut, element_size);
            auto [scores, ids] = detail::block_budget_query(
                dense.data(), element_size, visited, candidates, score_scratch,
                query_indices, query_codes, len, cuts, k, k_prime,
                clustered_inverted_lists, fwd, vectors, id_selector);
            decode_scores(scores, search_parameters);
            scores.resize(k, -1.0F);
            ids.resize(k, detail::INVALID_IDX);
            result_distances[query_idx] = std::move(scores);
            result_labels[query_idx] = std::move(ids);
        }
    }
    return {result_distances, result_labels};
}

void DiskSeismicIndexBase::write_index(IOWriter* io_writer) {
    write_payload_header(io_writer);
    uint64_t nv = num_vectors_;
    io_writer->write(&nv, sizeof(uint64_t), 1);
    // Summaries only: the doc-id membership is already in the inline forward
    // index below, so writing it in the posting lists too would duplicate it.
    SeismicInvertedListsWriter inv_list_writer(clustered_inverted_lists,
                                               /*summaries_only=*/true);
    inv_list_writer.serialize(io_writer);
    // Inline forward index, built from the same clusters + vectors. An empty
    // corpus uses a correctly-typed empty SparseVectors (element_size must be a
    // valid width even with zero vectors) so the section still round-trips.
    SparseVectors empty_vectors(
        {.element_size = code_element_size(),
         .dimension = static_cast<size_t>(dimension_)});
    const SparseVectors& v = vectors_ != nullptr ? *vectors_ : empty_vectors;
    detail::InlineForwardIndex forward(clustered_inverted_lists, v);
    forward.serialize(io_writer);
}

void DiskSeismicIndexBase::read_index(IOReader* /*io_reader*/,
                                      const IndexHeader& /*header*/,
                                      int /*io_flags*/) {
    // The inline forward index is borrowed from a mapping, never copied onto
    // the heap, so this index has no copying read path.
    throw std::runtime_error(
        "DiskSeismic index is mmap-only; load with read_index(file, "
        "IndexIoFlag::kUseMmap)");
}

void DiskSeismicIndexBase::load_mapped_payload(MmapCursor* cursor,
                                               MmapFile&& mapped) {
    // Same order write_index wrote them (past any extra header the caller
    // already consumed): doc count, summaries, inline forward.
    num_vectors_ = cursor->read_scalar<uint64_t>();
    SeismicInvertedListsWriter inv_list_writer;
    inv_list_writer.mmap_deserialize(cursor);
    detail::InlineForwardIndex forward;
    forward.mmap_deserialize(cursor);

    clustered_inverted_lists = std::move(inv_list_writer.release());
    fwd_ = std::move(forward);
    // Now that the summaries and forward index are populated (still borrowing
    // from `mapped`, which is alive here), let the concrete index reject a width
    // mismatch before we commit.
    validate_mapped_payload();

    // mapped_file_ last: the summaries and the forward index borrow from it, and
    // moving it does not move the mapping.
    mapped_file_ = std::move(mapped);
}

}  // namespace nsparse
