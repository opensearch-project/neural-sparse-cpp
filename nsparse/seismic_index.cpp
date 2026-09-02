/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/seismic_index.h"

#include "nsparse/seismic_batched_build.h"

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
#include "nsparse/utils/mmap_file.h"
#include "nsparse/utils/prefetch.h"
#include "nsparse/utils/ranker.h"
#include "nsparse/utils/vector_process.h"

namespace nsparse {
namespace {

constexpr int kElementSize = U32;

// Values are read back as float, at the width add() encoded them with. A file
// declaring anything else would be strided at the wrong width, past the end of
// the array for a narrower one, so reject it at load rather than at search.
void throw_if_element_size_mismatch(const SparseVectors& vectors) {
    if (vectors.num_vectors() > 0 &&
        vectors.get_element_size() != kElementSize) {
        throw std::runtime_error(
            "index file's element size is not the seismic index's");
    }
}

void query_single_inverted_list(
    const SparseVectors* vectors, const InvertedListClusters& cluster_invlist,
    const std::vector<float>& dense, const term_t* q_idx, const float* q_val,
    size_t q_len, std::vector<float>& score_scratch, const float heap_factor,
    const bool first_list, const SearchParameters* search_parameters,
    detail::TopKHolder<idx_t>& heap, absl::flat_hash_set<idx_t>& visited) {
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
        // Prefetch one doc ahead, only the leading lines of the upcoming row;
        // the row is contiguous so the hardware streamer pulls the tail, while
        // bounding outstanding software prefetches keeps the line-fill buffers
        // from saturating (measured optimum ~4 lines).
        static constexpr size_t kPrefetchDist = 1;
        static constexpr size_t kPrefetchHeadLines = 4;
        for (size_t i = 0; i < n_docs; ++i) {
            const auto& doc_id = docs[i];
            if (i + kPrefetchDist < n_docs) {
                const idx_t next_doc = docs[i + kPrefetchDist];
                const idx_t next_start = indptr[next_doc];
                const size_t next_len = indptr[next_doc + 1] - next_start;
                detail::prefetch_vector_head(indices + next_start,
                                             values + next_start, next_len,
                                             kPrefetchHeadLines);
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
    : MmapIndex(dim),
      cluster_parameter_(detail::kDefaultSeismicClusterParams) {}
SeismicIndex::SeismicIndex(int dim, SeismicClusterParameters parameter)
    : MmapIndex(dim), cluster_parameter_(parameter) {}

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
    const SparseVectorsConfig config = {
        .element_size = kElementSize,
        .dimension = static_cast<size_t>(get_dimension())};
    const std::string& out_path =
        cluster_parameter_.batch_clustering.batch_file_output_path;
    if (!out_path.empty()) {
        // Streamed straight to a file and not retained: see
        // BatchClusteringOption. write_index afterwards would write an index with
        // no posting lists, so this index is deliberately left empty.
        detail::write_seismic_index_batched(
            get_vectors(), config, cluster_parameter_,
            {.id = fourcc(name),
             .version = kFormatVersion,
             .dimension = get_dimension()},
            [this](IOWriter* io_writer) { vectors_->serialize(io_writer); },
            out_path);
        return;
    }
    clustered_inverted_lists =
        detail::build_inverted_lists_clusters(get_vectors(), config,
                                              cluster_parameter_);
}

auto SeismicIndex::search(idx_t n, const idx_t* indptr, const term_t* indices,
                          const float* values, int k,
                          SearchParameters* search_parameters)
    -> pair_of_score_id_vectors_t {
    if (vectors_ == nullptr || n == 0) {
        return {std::vector<std::vector<float>>(n),
                std::vector<std::vector<idx_t>>(n)};
    }

    // Parameters that carry no cut / heap factor (none at all, or a plain
    // SearchParameters holding just an id selector) fall back to the defaults
    // rather than being dereferenced as null.
    SeismicSearchParameters default_params;
    const auto* seismic_parameters =
        dynamic_cast<const SeismicSearchParameters*>(search_parameters);
    const auto* parameters =
        seismic_parameters != nullptr ? seismic_parameters : &default_params;
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
                get_vectors(), dynamic_cast<const IDSelectorEnumerable*>(sel),
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
        absl::flat_hash_set<idx_t> visited;
        visited.reserve(static_cast<size_t>(std::max(k, 1)) * 4096);

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
                             parameters->heap_factor, search_parameters);
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
                                absl::flat_hash_set<idx_t>& visited,
                                const term_t* q_indices, const float* q_values,
                                size_t q_len, const std::vector<term_t>& cuts,
                                int k, float heap_factor,
                                SearchParameters* search_parameters)
    -> pair_of_score_id_vector_t {
    size_t num_docs = vectors_->num_vectors();
    if (num_docs == 0) {
        return {{}, {}};
    }

    // Scatter the query into the reused dense buffer (all-zero on entry).
    for (size_t i = 0; i < q_len; ++i) {
        dense[q_indices[i]] = q_values[i];
    }
    visited.clear();

    detail::TopKHolder<idx_t> holder(k);
    std::vector<float> score_scratch;
    bool first_list = true;
    for (const auto& term : cuts) {
        if (term >= clustered_inverted_lists.size()) [[unlikely]] {
            continue;
        }
        const auto& cluster_invlist = clustered_inverted_lists[term];
        query_single_inverted_list(vectors_.get(), cluster_invlist, dense,
                                   q_indices, q_values, q_len, score_scratch,
                                   heap_factor, first_list, search_parameters,
                                   holder, visited);
        first_list = false;
    }

    // Restore the dense buffer to all-zero for the next query on this thread
    // (sparse clear over only the dims this query touched).
    for (size_t i = 0; i < q_len; ++i) {
        dense[q_indices[i]] = 0.0F;
    }

    auto [scores, ids] = holder.top_k_items_descending();
    scores.resize(k, -1.0F);
    ids.resize(k, detail::INVALID_IDX);
    return {scores, ids};
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

void SeismicIndex::read_index(IOReader* io_reader, const IndexHeader& header,
                              int io_flags) {
    // read vectors
    SparseVectors tmp_vectors;
    tmp_vectors.deserialize(io_reader);
    throw_if_element_size_mismatch(tmp_vectors);
    if (tmp_vectors.num_vectors() > 0) {
        vectors_ = std::make_unique<SparseVectors>(std::move(tmp_vectors));
    }
    SeismicInvertedListsWriter inv_list_writer;
    inv_list_writer.deserialize(io_reader);
    clustered_inverted_lists = std::move(inv_list_writer.release());
}

SeismicIndex* SeismicIndex::mmap_index(const IndexHeader& header,
                                       const char* index_file, size_t pos) {
    throw_if_null(index_file, "index_file must not be null");
    auto index = std::make_unique<SeismicIndex>(header.dimension);

    MmapFile mmap_file(std::string{index_file});
    // `pos` is where write_index's payload begins, past the header read_header
    // consumed. Absolute file offsets are what serialize() padded against, so
    // the cursor has to start at 0 and skip rather than map from `pos`.
    MmapCursor cursor(mmap_file.data(), mmap_file.size());
    cursor.skip(pos);

    // Same order write_index wrote them.
    auto vectors = std::make_unique<SparseVectors>();
    vectors->mmap_deserialize(&cursor);
    throw_if_element_size_mismatch(*vectors);

    SeismicInvertedListsWriter inv_list_writer;
    inv_list_writer.mmap_deserialize(&cursor);

    // Committed only once everything parsed, so a corrupt file cannot leave a
    // half-mapped index behind. mapped_file_ last: the arrays above borrow from
    // it, and moving it does not move the mapping itself.
    index->clustered_inverted_lists = std::move(inv_list_writer.release());
    if (vectors->num_vectors() > 0) {
        index->vectors_ = std::move(vectors);
    }
    index->mapped_file_ = std::move(mmap_file);
    return index.release();
}
}  // namespace nsparse