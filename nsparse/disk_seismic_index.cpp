/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/disk_seismic_index.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/id_selector.h"
#include "nsparse/index.h"
#include "nsparse/io/inline_forward_index_io.h"
#include "nsparse/io/seismic_invlists_writer.h"
#include "nsparse/seismic_common.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"
#include "nsparse/utils/distance_simd.h"
#include "nsparse/utils/mmap_cursor.h"
#include "nsparse/utils/mmap_file.h"
#include "nsparse/utils/ranker.h"
#include "nsparse/utils/vector_process.h"

namespace nsparse {
namespace {

constexpr int kElementSize = U32;

// Score one doc's vector against the dense query and push it, honoring
// visited-dedup and the id selector. Shared by both vector sources.
inline void score_doc(idx_t doc_id, const term_t* comps, const float* vals,
                      size_t nnz, const std::vector<float>& dense,
                      const IDSelector* id_selector,
                      detail::TopKHolder<idx_t>& heap,
                      absl::flat_hash_set<idx_t>& visited) {
    if (!visited.insert(doc_id).second) {
        return;
    }
    if (id_selector != nullptr && !id_selector->is_member(doc_id)) {
        return;
    }
    const float score =
        detail::dot_product_float_dense(comps, vals, nnz, dense.data());
    heap.add(score, doc_id);
}

// A candidate block: its summary score and its (posting list, cluster) address.
struct BlockCandidate {
    float score;
    uint32_t pl;
    uint32_t cid;
};

// Score every document of one block against the dense query, dedup via
// `visited` and honor the id selector. Vectors come from the inline forward
// index (fwd) when loaded, else from the in-RAM CSR of a fresh build.
void score_block(const detail::InlineForwardIndex* fwd,
                 const SparseVectors* vectors,
                 const std::vector<InvertedListClusters>& clusters, uint32_t pl,
                 uint32_t cid, const std::vector<float>& dense,
                 const IDSelector* id_selector, size_t element_size,
                 detail::TopKHolder<idx_t>& heap,
                 absl::flat_hash_set<idx_t>& visited) {
    if (fwd != nullptr) {
        const detail::BlockView bv = fwd->block(pl, cid);
        if (bv.absent()) {
            return;
        }
        for (uint32_t i = 0; i < bv.n_docs; ++i) {
            score_doc(
                static_cast<idx_t>(bv.doc_ids[i]), bv.doc_comps(i),
                reinterpret_cast<const float*>(bv.doc_vals(i, element_size)),
                bv.nnz(i), dense, id_selector, heap, visited);
        }
    } else if (vectors != nullptr) {
        const auto data = vectors->get_all_data();
        const idx_t* const indptr = data.indptr_data;
        const term_t* const indices = data.indices_data;
        const float* const values = data.values_data;
        for (const idx_t doc_id : clusters[pl].get_docs(cid)) {
            const idx_t start = indptr[doc_id];
            const size_t len = indptr[doc_id + 1] - start;
            score_doc(doc_id, indices + start, values + start, len, dense,
                      id_selector, heap, visited);
        }
    }
}
}  // namespace

DiskSeismicIndex::DiskSeismicIndex(int dim)
    : MmapIndex(dim),
      cluster_parameter_(detail::kDefaultSeismicClusterParams) {}

DiskSeismicIndex::DiskSeismicIndex(int dim, SeismicClusterParameters parameter)
    : MmapIndex(dim), cluster_parameter_(parameter) {}

void DiskSeismicIndex::add(idx_t n, const idx_t* indptr, const term_t* indices,
                           const float* values) {
    throw_if_not_positive(n);
    throw_if_any_null(indptr, indices, values);
    const size_t indptr_size = n + 1;
    const size_t nnz = indptr[n];
    if (vectors_ == nullptr) {
        // Fresh container: start the count at 0 so a stale num_vectors_ (e.g.
        // left by a prior mmap load, which has no vectors_) cannot accumulate.
        num_vectors_ = 0;
        vectors_ = std::make_unique<SparseVectors>(
            SparseVectorsConfig{.element_size = kElementSize,
                                .dimension = static_cast<size_t>(dimension_)});
    }
    vectors_->add_vectors(indptr, indptr_size, indices, nnz,
                          reinterpret_cast<const uint8_t*>(values),
                          nnz * kElementSize);
    num_vectors_ += n;
}

void DiskSeismicIndex::build() {
    clustered_inverted_lists = detail::build_inverted_lists_clusters(
        get_vectors(),
        {.element_size = kElementSize,
         .dimension = static_cast<size_t>(get_dimension())},
        cluster_parameter_);
}

auto DiskSeismicIndex::search(idx_t n, const idx_t* indptr,
                              const term_t* indices, const float* values, int k,
                              SearchParameters* search_parameters)
    -> pair_of_score_id_vectors_t {
    // Quit early when there is nothing to score: no vectors, no queries, or no
    // forward-vector source (fwd_ empty and vectors_ null — a corrupt or
    // uninitialized index).
    if (num_vectors_ == 0 || n == 0 ||
        (fwd_.num_blocks() == 0 && vectors_ == nullptr)) {
        return {
            std::vector<std::vector<float>>(n, std::vector<float>(k, -1.0F)),
            std::vector<std::vector<idx_t>>(
                n, std::vector<idx_t>(k, detail::INVALID_IDX))};
    }

    // Resolve `cut` and `k_prime`. A DiskSeismicSearchParameters carries
    // k_prime; a plain SeismicSearchParameters (or null) uses the default
    // budget. The id-selector exact-match fast path is omitted (it needs an
    // in-RAM SparseVectors a mapped index lacks); the selector is still honored
    // in score_block.
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
    // else: search_parameters is null or an unrelated SearchParameters subtype,
    // so cut and k_prime keep the defaults initialized above.

    // k_prime is a block budget, not a document count: a block holds many docs,
    // so k_prime < k is valid (a few blocks can still fill k, and a short-fall
    // is padded like any under-budget search). Only a non-positive budget is
    // rejected.
    if (k_prime <= 0) {
        throw std::invalid_argument(
            "DiskSeismicIndex: k_prime (block budget) must be positive");
    }

    std::vector<std::vector<float>> result_distances(n, std::vector<float>(k, -1.0F));
    std::vector<std::vector<idx_t>> result_labels(
        n, std::vector<idx_t>(k, detail::INVALID_IDX));
    const size_t dim = static_cast<size_t>(dimension_);

#pragma omp parallel
    {
        std::vector<float> dense(dim, 0.0F);
        absl::flat_hash_set<idx_t> visited;
        visited.reserve(static_cast<size_t>(std::max(k, 1)) * 4096);

#pragma omp for schedule(dynamic, 64)
        for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
            const idx_t start = indptr[query_idx];
            const size_t len = indptr[query_idx + 1] - start;
            const term_t* q_indices = indices + start;
            const float* q_values = values + start;
            const auto& cuts =
                detail::top_k_tokens(q_indices, q_values, len, cut);
            auto [distances, labels] =
                single_query(dense, visited, q_indices, q_values, len, cuts, k,
                             k_prime, search_parameters);
            result_distances[query_idx] = std::move(distances);
            result_labels[query_idx] = std::move(labels);
        }
    }
    return {result_distances, result_labels};
}

auto DiskSeismicIndex::single_query(
    std::vector<float>& dense, absl::flat_hash_set<idx_t>& visited,
    const term_t* q_indices, const float* q_values, size_t q_len,
    const std::vector<term_t>& cuts, int k, int k_prime,
    SearchParameters* search_parameters) -> pair_of_score_id_vector_t {
    if (num_vectors_ == 0) {
        return {{}, {}};
    }
    // Scatter the query into the dense lookup table.
    for (size_t i = 0; i < q_len; ++i) {
        dense[q_indices[i]] = q_values[i];
    }
    visited.clear();

    // Prefer the mapped inline forward index; fall back to the in-RAM vectors
    // of a fresh build.
    const detail::InlineForwardIndex* fwd =
        fwd_.num_blocks() > 0 ? &fwd_ : nullptr;
    const SparseVectors* vectors = fwd == nullptr ? vectors_.get() : nullptr;
    const size_t element_size = fwd != nullptr
                                    ? fwd->element_size()
                                    : static_cast<size_t>(kElementSize);
    const IDSelector* id_selector = search_parameters == nullptr
                                        ? nullptr
                                        : search_parameters->get_id_selector();

    // Collect every candidate block across the cut lists with its summary
    // score. score_summaries_transposed dots the full query with each cluster
    // summary, so scores are comparable across posting lists for the global
    // ranking.
    std::vector<BlockCandidate> candidates;
    std::vector<float> score_scratch;
    for (const term_t term : cuts) {
        if (term >= clustered_inverted_lists.size()) [[unlikely]] {
            continue;
        }
        const InvertedListClusters& cluster_invlist =
            clustered_inverted_lists[term];
        const size_t n_clusters = cluster_invlist.cluster_size();
        if (n_clusters == 0) {
            continue;
        }
        cluster_invlist.score_summaries_transposed(
            q_indices, reinterpret_cast<const uint8_t*>(q_values), q_len,
            score_scratch);
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
    detail::TopKHolder<idx_t> holder(k);
    for (const BlockCandidate& candidate : candidates) {
        score_block(fwd, vectors, clustered_inverted_lists, candidate.pl,
                    candidate.cid, dense, id_selector, element_size, holder,
                    visited);
    }

    // Reset only the query's own positions (q_len is small) instead of
    // memset-ing the whole dim-sized table; mirrors the scatter at entry. The
    // scattered writes are bounded by q_len and negligible next to the block
    // reads/scoring that dominate the query.
    for (size_t i = 0; i < q_len; ++i) {
        dense[q_indices[i]] = 0.0F;
    }
    auto [scores, ids] = holder.top_k_items_descending();
    scores.resize(k, -1.0F);
    ids.resize(k, detail::INVALID_IDX);
    return {scores, ids};
}

void DiskSeismicIndex::write_index(IOWriter* io_writer) {
    const uint64_t nv = num_vectors_;
    io_writer->write(const_cast<uint64_t*>(&nv), sizeof(uint64_t), 1);
    SeismicInvertedListsWriter inv_list_writer(clustered_inverted_lists);
    inv_list_writer.serialize(io_writer);
    // Inline forward index, built from the same clusters + vectors. An empty
    // corpus uses a correctly-typed empty SparseVectors (element_size must be a
    // valid width even with zero vectors) so the section still round-trips.
    SparseVectors empty_vectors({.element_size = kElementSize,
                                 .dimension = static_cast<size_t>(dimension_)});
    const SparseVectors& v = vectors_ != nullptr ? *vectors_ : empty_vectors;
    detail::InlineForwardIndex forward(clustered_inverted_lists, v);
    forward.serialize(io_writer);
}

void DiskSeismicIndex::read_index(IOReader* /*io_reader*/, int /*io_flags*/) {
    // The inline forward index is borrowed from a mapping, never copied onto
    // the heap, so this index has no copying read path.
    throw std::runtime_error(
        "DiskSeismicIndex is mmap-only; load with read_index(file, "
        "IndexIoFlag::kUseMmap)");
}

DiskSeismicIndex* DiskSeismicIndex::mmap_index(int dimension,
                                               const char* index_file,
                                               size_t pos) {
    throw_if_null(index_file, "index_file must not be null");
    auto index = std::make_unique<DiskSeismicIndex>(dimension);

    MmapFile mmap_file(std::string{index_file});
    MmapCursor cursor(mmap_file.data(), mmap_file.size());
    cursor.skip(pos);

    // Same order write_index wrote them: doc count, summaries, inline forward.
    const uint64_t nv = cursor.read_scalar<uint64_t>();
    SeismicInvertedListsWriter inv_list_writer;
    inv_list_writer.mmap_deserialize(&cursor);
    detail::InlineForwardIndex forward;
    forward.mmap_deserialize(&cursor);

    // Committed only once everything parsed. mapped_file_ last: the summaries
    // and the forward index borrow from it, and moving it does not move the
    // mapping.
    index->clustered_inverted_lists = std::move(inv_list_writer.release());
    index->fwd_ = std::move(forward);
    index->num_vectors_ = nv;
    index->mapped_file_ = std::move(mmap_file);
    return index.release();
}
}  // namespace nsparse
