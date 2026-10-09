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
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/disk_seismic_search.h"
#include "nsparse/exact_matcher.h"
#include "nsparse/id_selector.h"
#include "nsparse/index.h"
#include "nsparse/io/align.h"
#include "nsparse/io/inline_forward_index_io.h"
#include "nsparse/io/seismic_invlists_writer.h"
#include "nsparse/seismic_common.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"
#include "nsparse/utils/ranker.h"
#include "nsparse/utils/mmap_cursor.h"
#include "nsparse/utils/mmap_file.h"

namespace nsparse {

DiskSeismicIndexBase::DiskSeismicIndexBase(int dim,
                                           SeismicClusterParameters parameter,
                                           uint32_t inline_max_nnz)
    : MmapIndex(dim),
      cluster_parameter_(parameter),
      inline_max_nnz_(inline_max_nnz) {}

void DiskSeismicIndexBase::add(idx_t n, const offset_t* indptr,
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
        vectors_ = std::make_unique<SparseVectors>(
            SparseVectorsConfig{.element_size = element_size,
                                .dimension = static_cast<size_t>(dimension_)});
    }
    std::vector<uint8_t> scratch;
    const uint8_t* codes = encode_values(values, nnz, scratch);
    vectors_->add_vectors(indptr, indptr_size, indices, nnz, codes,
                          nnz * element_size);
    num_vectors_ += n;
}

void DiskSeismicIndexBase::build() {
    clustered_inverted_lists = detail::build_clustered_lists(
        get_vectors(), static_cast<size_t>(get_dimension()), cluster_parameter_,
        &batch_spill_);
}

auto DiskSeismicIndexBase::search(idx_t n, const offset_t* indptr,
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

    // An enumerable selector of size <= k must return every member, so score
    // exactly the selected docs rather than only those in the top-k' blocks.
    // get_vectors() is doc-id-addressable for a fresh or mmap-CSR build; a
    // mapped serialized index resolves docs through doc_locators_ instead.
    if (search_parameters != nullptr &&
        detail::should_run_exact_match(search_parameters->get_id_selector(), k,
                                       nullptr)) {
        const auto& selector = *dynamic_cast<const IDSelectorEnumerable*>(
            search_parameters->get_id_selector());
        const SparseVectors* vectors = get_vectors();
        if (vectors != nullptr) {
            const size_t element_size = code_element_size();
            const size_t nnz = indptr[n];
            std::vector<uint8_t> query_scratch;
            const uint8_t* query_codes =
                encode_query(values, nnz, search_parameters, query_scratch);
            SparseVectors query_vectors(
                {.element_size = element_size,
                 .dimension = static_cast<size_t>(get_dimension())});
            query_vectors.add_vectors(indptr, static_cast<size_t>(n) + 1,
                                      indices, nnz, query_codes,
                                      nnz * element_size);
            auto [distances, labels] = detail::ExactMatcher::search(
                vectors, &selector, &query_vectors, element_size, k);
            // Decode only the filled prefix; ExactMatcher pads the tail with
            // (-1.0F, INVALID_IDX), which must not be decoded.
            for (size_t q = 0; q < distances.size(); ++q) {
                size_t filled = 0;
                while (filled < labels[q].size() &&
                       labels[q][filled] != detail::INVALID_IDX) {
                    ++filled;
                }
                std::vector<float> head(distances[q].begin(),
                                        distances[q].begin() + filled);
                decode_scores(head, search_parameters);
                std::copy(head.begin(), head.end(), distances[q].begin());
            }
            return {distances, labels};
        }
        if (doc_locators_ != nullptr) {
            return exact_match_directory(n, indptr, indices, values, k,
                                         selector, search_parameters);
        }
    }

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
    // Checked even when the index is not truncated and the depth is unused, so
    // a bad value fails the same way whichever index it is pointed at.
    if (budget.rescore < 0) {
        throw std::invalid_argument(
            "DiskSeismic index: rescore (re-scoring depth) must be "
            "non-negative");
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
    // A truncated forward index makes block scores lower bounds, so the best
    // candidates have to be scored again against the whole copies. Nothing to do
    // when max_doc_nnz is 0 (including every pre-truncation file, which records
    // 0) or for a fresh in-RAM build, whose vectors are the corpus itself.
    const bool rescore_needed =
        fwd != nullptr && fwd->max_doc_nnz() > 0 && doc_locators_ != nullptr;
    // Re-scoring fewer than k candidates could not fill the result, so the depth
    // is at least k; beyond the candidate pool it simply re-scores all of it.
    const detail::FullVectorStore store =
        rescore_needed ? full_vector_store() : detail::FullVectorStore{};
    const int rescore_depth = rescore_needed ? std::max(budget.rescore, k) : 0;
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
        // Per-thread, so the batched whole-vector resolve reuses its buffers
        // across the queries this thread handles.
        detail::RescorePass rescore{.store = store, .depth = rescore_depth};

#pragma omp for schedule(dynamic, 64)
        for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
            const offset_t start = indptr[query_idx];
            const size_t len = indptr[query_idx + 1] - start;
            const term_t* query_indices = indices + start;
            const uint8_t* query_codes =
                query_batch + static_cast<size_t>(start) * element_size;
            const std::vector<term_t> cuts = detail::top_cut_tokens(
                query_indices, query_codes, len, cut, element_size);
            auto [scores, ids] = detail::block_budget_query(
                dense.data(), element_size, visited, candidates, score_scratch,
                query_indices, query_codes, len, cuts, k, k_prime,
                clustered_inverted_lists, fwd, vectors, id_selector,
                rescore_needed ? &rescore : nullptr);
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
    SparseVectors empty_vectors({.element_size = code_element_size(),
                                 .dimension = static_cast<size_t>(dimension_)});
    const SparseVectors& v = vectors_ != nullptr ? *vectors_ : empty_vectors;
    // One table, two readers: the inline writer needs it to leave exactly one
    // copy of each doc whole, and the directory below persists it so search can
    // find that copy again.
    const std::vector<detail::DocLocator> locators = build_doc_locators(v);
    detail::InlineForwardIndex forward(
        clustered_inverted_lists, v, detail::InlineForwardIndex::kDefaultPageSize,
        detail::InlineLayout::kPacked, inline_max_nnz_,
        inline_max_nnz_ == 0 ? nullptr : &locators);
    forward.serialize(io_writer);
    write_doc_directory(io_writer, v, locators);
}

auto DiskSeismicIndexBase::build_doc_locators(
    const SparseVectors& vectors) const -> std::vector<detail::DocLocator> {
    const size_t num_docs = vectors.num_vectors();

    // Default to remainder; the loop below overrides docs it finds in a block.
    std::vector<detail::DocLocator> locators(
        num_docs, {detail::DocLocator::kRemainder, 0, 0});
    std::vector<bool> covered(num_docs, false);
    // Must mirror InlineForwardIndex::write_body's block/slot iteration so the
    // recorded (posting_list, block, slot) match the blocks it writes.
    // First occurrence wins; when nothing is truncated every copy of a doc's
    // vector is identical, and when something is, this is the one left whole.
    for (size_t pl = 0; pl < clustered_inverted_lists.size(); ++pl) {
        const InvertedListClusters& list = clustered_inverted_lists[pl];
        const size_t n_clusters = list.cluster_size();
        for (size_t block = 0; block < n_clusters; ++block) {
            const std::span<const idx_t> docs = list.get_docs(block);
            for (size_t slot = 0; slot < docs.size(); ++slot) {
                const idx_t doc_id = docs[slot];
                if (doc_id < 0 || static_cast<size_t>(doc_id) >= num_docs ||
                    covered[doc_id]) {
                    continue;
                }
                covered[doc_id] = true;
                locators[doc_id] = {static_cast<uint32_t>(pl),
                                    static_cast<uint32_t>(block),
                                    static_cast<uint32_t>(slot)};
            }
        }
    }
    // Docs in no block get a remainder row, numbered in doc-id order --  the
    // order write_doc_directory emits them in.
    uint32_t remainder_row = 0;
    for (size_t doc_id = 0; doc_id < num_docs; ++doc_id) {
        if (!covered[doc_id]) {
            locators[doc_id] = {detail::DocLocator::kRemainder, remainder_row,
                                0};
            ++remainder_row;
        }
    }
    return locators;
}

void DiskSeismicIndexBase::write_doc_directory(
    IOWriter* io_writer, const SparseVectors& vectors,
    const std::vector<detail::DocLocator>& locators) const {
    const size_t num_docs = vectors.num_vectors();
    const size_t element_size = vectors.get_element_size();

    // The docs no block holds, in the doc-id order their rows were numbered in.
    SparseVectors remainder(
        {.element_size = element_size,
         .dimension = static_cast<size_t>(get_dimension())});
    const offset_t* indptr = vectors.indptr_data();
    const term_t* indices = vectors.indices_data();
    const uint8_t* values = vectors.values_data();
    for (size_t doc_id = 0; doc_id < num_docs; ++doc_id) {
        if (locators[doc_id].posting_list != detail::DocLocator::kRemainder) {
            continue;
        }
        const offset_t start = indptr[doc_id];
        const size_t nnz = static_cast<size_t>(indptr[doc_id + 1] - start);
        const offset_t row_indptr[2] = {0, static_cast<offset_t>(nnz)};
        remainder.add_vectors(
            row_indptr, 2, indices + start, nnz,
            values + static_cast<size_t>(start) * element_size,
            nnz * element_size);
    }

    // Aligned u64 doc count, then the locator array, then the remainder
    // vectors; load_mapped_payload reads them back in this order.
    io_align::pad_to(io_writer, detail::kMinBlockAlign);
    uint64_t count = num_docs;
    io_writer->write(&count, sizeof(uint64_t), 1);
    io_align::write_padded(io_writer, locators.data(), locators.size(),
                           alignof(detail::DocLocator));
    remainder.serialize(io_writer);
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

    // Doc-locator directory (borrowed in place) then remainder vectors, in the
    // order write_doc_directory wrote them.
    io_align::skip_padding(cursor, detail::kMinBlockAlign);
    const uint64_t num_locators = cursor->read_scalar<uint64_t>();
    if (num_locators != num_vectors_) {
        throw std::runtime_error(
            "DiskSeismic index: doc-locator count disagrees with vector count");
    }
    io_align::skip_padding(cursor, alignof(detail::DocLocator));
    doc_locators_ = cursor->read_array<detail::DocLocator>(num_locators);
    num_locators_ = num_locators;
    remainder_.mmap_deserialize(cursor);

    // Now that the summaries and forward index are populated (still borrowing
    // from `mapped`, which is alive here), let the concrete index reject a
    // width mismatch before we commit.
    validate_mapped_payload();

    // index_mapping_ last: the summaries and the forward index borrow from it,
    // and moving it does not move the mapping.
    index_mapping_ = std::move(mapped);
}

auto DiskSeismicIndexBase::full_vector_store() const
    -> detail::FullVectorStore {
    return {.fwd = &fwd_,
            .locators = doc_locators_,
            .num_locators = num_locators_,
            .remainder = &remainder_};
}

auto DiskSeismicIndexBase::exact_match_directory(
    idx_t n, const offset_t* indptr, const term_t* indices, const float* values,
    int k, const IDSelectorEnumerable& selector,
    const SearchParameters* search_parameters) const
    -> pair_of_score_id_vectors_t {
    const size_t element_size = code_element_size();
    const size_t total_nnz = indptr[n];
    std::vector<uint8_t> query_scratch;
    const uint8_t* query_codes =
        encode_query(values, total_nnz, search_parameters, query_scratch);
    const std::vector<idx_t> ids = selector.ordered_ids();
    const detail::FullVectorStore store = full_vector_store();

    std::vector<std::vector<float>> result_distances(n);
    std::vector<std::vector<idx_t>> result_labels(n);
    const size_t dense_bytes =
        static_cast<size_t>(get_dimension()) * element_size;

#pragma omp parallel
    {
        // Per-thread dense query buffer, cleared per query below.
        std::vector<uint8_t> dense(dense_bytes, 0);
#pragma omp for schedule(dynamic, 64)
        for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
            const offset_t start = indptr[query_idx];
            const size_t len =
                static_cast<size_t>(indptr[query_idx + 1] - start);
            const term_t* q_indices = indices + start;
            const uint8_t* q_codes =
                query_codes + static_cast<size_t>(start) * element_size;
            for (size_t i = 0; i < len; ++i) {
                std::copy_n(
                    q_codes + i * element_size, element_size,
                    dense.data() +
                        static_cast<size_t>(q_indices[i]) * element_size);
            }

            detail::TopKHolder<idx_t> holder(k);
            for (const idx_t doc_id : ids) {
                if (doc_id < 0 ||
                    static_cast<uint64_t>(doc_id) >= num_locators_) {
                    continue;  // out-of-range member: nothing to score
                }
                const detail::DocSlice doc =
                    detail::resolve_doc(store, doc_id, element_size);
                // Dot the doc's slice against the dense query via a 2-entry
                // indptr.
                const offset_t slice_indptr[2] = {
                    0, static_cast<offset_t>(doc.nnz)};
                const float score = detail::compute_similarity(
                    0, slice_indptr, doc.comps, doc.vals, dense.data(),
                    element_size);
                holder.add(score, doc_id);
            }

            // Decode before padding so the -1 pad is not decoded.
            auto [scores, labels] = holder.top_k_items_descending();
            decode_scores(scores, search_parameters);
            scores.resize(k, -1.0F);
            labels.resize(k, detail::INVALID_IDX);

            for (size_t i = 0; i < len; ++i) {
                std::fill_n(dense.data() + static_cast<size_t>(q_indices[i]) *
                                               element_size,
                            element_size, uint8_t{0});
            }
            result_distances[query_idx] = std::move(scores);
            result_labels[query_idx] = std::move(labels);
        }
    }
    return {result_distances, result_labels};
}

}  // namespace nsparse
