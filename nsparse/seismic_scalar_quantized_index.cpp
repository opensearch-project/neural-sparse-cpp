/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/seismic_scalar_quantized_index.h"

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <typeinfo>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/cluster/random_kmeans.h"
#include "nsparse/exact_matcher.h"
#include "nsparse/id_selector.h"
#include "nsparse/index.h"
#include "nsparse/invlists/inverted_lists.h"
#include "nsparse/io/io.h"
#include "nsparse/io/seismic_invlists_writer.h"
#include "nsparse/seismic_common.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"
#include "nsparse/utils/distance_simd.h"
#include "nsparse/utils/prefetch.h"
#include "nsparse/utils/scalar_quantizer.h"
#include "nsparse/utils/vector_process.h"

namespace nsparse {
namespace {

void query_single_inverted_list(const SparseVectors* vectors,
                                const InvertedListClusters& cluster_invlist,
                                const std::vector<uint8_t>& dense,
                                float heap_factor, bool first_list,
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
    const auto element_size = vectors->get_element_size();
    const auto& summaries = cluster_invlist.summaries();
    // compute dp with all summaries
    std::vector<float> summary_scores =
        detail::calculate_summary_scores(element_size, &summaries, dense);
    size_t num_vectors = vectors->num_vectors();

    std::vector<size_t> cluster_order =
        detail::reorder_clusters(summary_scores, first_list);

    const auto* indptr = vectors->indptr_data();
    const auto* indices = vectors->indices_data();
    const auto* values = vectors->values_data();

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
        // Two-stage prefetch pipeline:
        // Stage 1 (distance 2): prefetch indptr[docs[i+2]] so the indptr
        //   lookup is cached by the time we need it next iteration.
        // Stage 2 (distance 1): read indptr[docs[i+1]] (now cached from
        //   stage 1 issued last iteration), prefetch the actual vector data.
        static constexpr size_t kPrefetchDist1 = 2;  // vector data prefetch
        static constexpr size_t kPrefetchDist2 = 4;  // indptr prefetch
        for (size_t i = 0; i < n_docs; ++i) {
            const auto& doc_id = docs[i];
            // Stage 1: prefetch indptr entry for doc at distance 2
            if (i + kPrefetchDist2 < n_docs) {
                detail::prefetch_indptr(indptr, docs[i + kPrefetchDist2]);
            }
            // Stage 2: prefetch vector data for next doc (indptr should
            // already be cached from stage 1 issued kPrefetchDist2 -
            // kPrefetchDist1 iterations ago)
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
            float score = detail::compute_similarity(
                doc_id, indptr, indices, values, dense.data(), element_size);
            heap.add(score, doc_id);
        }
    }
}
}  // namespace

SeismicScalarQuantizedIndex::SeismicScalarQuantizedIndex(int dim)
    : Index(dim), cluster_parameter_(detail::kDefaultSeismicClusterParams) {}

SeismicScalarQuantizedIndex::SeismicScalarQuantizedIndex(
    QuantizerType quantizer_type, float vmin, float vmax,
    SeismicClusterParameters parameter, int dim)
    : Index(dim),
      sq_(quantizer_type, vmin, vmax),
      cluster_parameter_(parameter) {}

void SeismicScalarQuantizedIndex::add(idx_t n, const idx_t* indptr,
                                      const term_t* indices,
                                      const float* values) {
    throw_if_not_positive(n);
    throw_if_any_null(indptr, indices, values);

    size_t indptr_size = n + 1;
    size_t nnz = indptr[n];  // Total non-zeros
    const size_t element_size = sq_.bytes_per_value();
    if (vectors_ == nullptr) {
        vectors_ = std::unique_ptr<SparseVectors>(
            new SparseVectors({.element_size = element_size,
                               .dimension = static_cast<size_t>(dimension_)}));
    }
    std::vector<uint8_t> codes(nnz * element_size);
    sq_.encode(values, codes.data(), nnz);
    vectors_->add_vectors(indptr, indptr_size, indices, nnz, codes.data(),
                          nnz * element_size);
}

// encode based on search_parameters type, if it's SeismicSearchParameters,
// use Index's quantizer, if it's SeismicSQSearchParameters, construct
// quantizer using SeismicSQSearchParameters's parameters
std::vector<uint8_t> SeismicScalarQuantizedIndex::encode(
    const float* values, size_t nnz, SearchParameters* search_parameters) {
    const size_t element_size = sq_.bytes_per_value();
    std::vector<uint8_t> codes(nnz * element_size);
    if (typeid(*search_parameters) == typeid(SeismicSearchParameters)) {
        sq_.encode(values, codes.data(), nnz);
    } else if (typeid(*search_parameters) ==
               typeid(SeismicSQSearchParameters)) {
        const auto* seismic_sq_search_parameters =
            static_cast<const SeismicSQSearchParameters*>(search_parameters);
        ScalarQuantizer search_sq(sq_.get_quantizer_type(),
                                  seismic_sq_search_parameters->vmin,
                                  seismic_sq_search_parameters->vmax);
        search_sq.encode(values, codes.data(), nnz);
    } else {
        throw std::runtime_error("Unsupported search parameters type!");
    }
    return codes;
}

void SeismicScalarQuantizedIndex::build() {
    clustered_inverted_lists = std::move(detail::build_inverted_lists_clusters(
        vectors_.get(),
        {.element_size = sq_.bytes_per_value(),
         .dimension = static_cast<size_t>(get_dimension())},
        cluster_parameter_));
}

auto SeismicScalarQuantizedIndex::search(idx_t n, const idx_t* indptr,
                                         const term_t* indices,
                                         const float* values, int k,
                                         SearchParameters* search_parameters)
    -> pair_of_score_id_vectors_t {
    throw_if_null(search_parameters, "search parameters cannot be null!");
    if (vectors_ == nullptr || n == 0) {
        return {std::vector<std::vector<float>>(n),
                std::vector<std::vector<idx_t>>(n)};
    }
    size_t indptr_size = n + 1;
    size_t nnz = indptr[n];  // Total non-zeros

    // Determine query quantizer based on search parameters
    ScalarQuantizer query_sq = sq_;  // default to same as ingest
    if (typeid(*search_parameters) == typeid(SeismicSQSearchParameters)) {
        const auto* sq_params =
            static_cast<const SeismicSQSearchParameters*>(search_parameters);
        query_sq = ScalarQuantizer(sq_.get_quantizer_type(), sq_params->vmin,
                                   sq_params->vmax);
    }

    // construct query vector
    const size_t element_size = sq_.bytes_per_value();
    SparseVectors query_vectors({.element_size = element_size,
                                 .dimension = static_cast<size_t>(dimension_)});
    std::vector<uint8_t> codes = encode(values, nnz, search_parameters);
    query_vectors.add_vectors(indptr, indptr_size, indices, nnz, codes.data(),
                              nnz * element_size);

    // if filter ids size is <= k, just run exact match
    if (detail::should_run_exact_match(search_parameters->get_id_selector(), k,
                                       &query_vectors)) {
        auto [distances, labels] = detail::ExactMatcher::search(
            vectors_.get(),
            dynamic_cast<const IDSelectorEnumerable*>(
                search_parameters->get_id_selector()),
            &query_vectors, element_size, k);
        // Decode quantized dot product scores
        for (auto& query_distances : distances) {
            for (auto& dist : query_distances) {
                dist = sq_.decode_dot_product(dist, query_sq);
            }
        }
        return {distances, labels};
    }

    std::vector<std::vector<float>> result_distances(n);
    std::vector<std::vector<idx_t>> result_labels(n);

    // query
    const auto* parameters =
        dynamic_cast<const SeismicSearchParameters*>(search_parameters);
    const auto* query_indptr = query_vectors.indptr_data();
    const auto* query_indices = query_vectors.indices_data();
    const auto* query_values = query_vectors.values_data();

#pragma omp parallel for
    for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
        const auto& dense = query_vectors.get_dense_vector(query_idx);
        const idx_t start = query_indptr[query_idx];
        const size_t len = query_indptr[query_idx + 1] - start;
        std::vector<term_t> cuts;
        if (element_size == U16) {
            // start is element index, need byte offset for uint16_t access
            cuts = detail::top_k_tokens<uint16_t>(
                query_indices + start,
                reinterpret_cast<const uint16_t*>(query_values +
                                                  start * sizeof(uint16_t)),
                len, parameters->cut);
        } else {
            cuts = detail::top_k_tokens<uint8_t>(query_indices + start,
                                                 query_values + start, len,
                                                 parameters->cut);
        }

        auto [distances, labels] =
            single_query(dense, cuts, k, parameters->heap_factor, query_sq,
                         search_parameters);
        result_distances[query_idx] = std::move(distances);
        result_labels[query_idx] = std::move(labels);
    }

    return {result_distances, result_labels};
}

auto SeismicScalarQuantizedIndex::single_query(
    const std::vector<uint8_t>& dense, const std::vector<term_t>& cuts, int k,
    float heap_factor, const ScalarQuantizer& query_sq,
    SearchParameters* search_parameters) -> pair_of_score_id_vector_t {
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

    auto [distances, labels] = holder.top_k_items_descending();

    // Decode quantized dot product scores
    for (auto& dist : distances) {
        dist = sq_.decode_dot_product(dist, query_sq);
    }
    distances.resize(k, INVALID_IDX);
    labels.resize(k, -1.0F);
    return {distances, labels};
}

void SeismicScalarQuantizedIndex::write_index(IOWriter* io_writer) {
    write_header(io_writer);
    // write vectors
    if (vectors_ == nullptr) {
        empty_sparse_vectors.serialize(io_writer);
    } else {
        vectors_->serialize(io_writer);
    }
    SeismicInvertedListsWriter inv_list_writer(clustered_inverted_lists);
    inv_list_writer.serialize(io_writer);
}

void SeismicScalarQuantizedIndex::read_index(IOReader* io_reader) {
    read_header(io_reader);
    SparseVectors tmp_vectors;
    tmp_vectors.deserialize(io_reader);
    if (tmp_vectors.num_vectors() > 0) {
        vectors_ = std::make_unique<SparseVectors>(std::move(tmp_vectors));
    }
    SeismicInvertedListsWriter inv_list_writer({});
    inv_list_writer.deserialize(io_reader);
    clustered_inverted_lists = std::move(inv_list_writer.release());
}

void SeismicScalarQuantizedIndex::write_header(IOWriter* io_writer) {
    auto sq_type = sq_.get_quantizer_type();
    io_writer->write(&sq_type, sizeof(QuantizerType), 1);
    auto vmin = sq_.get_min();
    io_writer->write(&vmin, sizeof(float), 1);
    auto vmax = sq_.get_max();
    io_writer->write(&vmax, sizeof(float), 1);
}

void SeismicScalarQuantizedIndex::read_header(IOReader* io_reader) {
    QuantizerType sq_type = QuantizerType::QT_8bit;
    float vmin = 0.0F;
    float vmax = 1.0F;
    io_reader->read(&sq_type, sizeof(QuantizerType), 1);
    io_reader->read(&vmin, sizeof(float), 1);
    io_reader->read(&vmax, sizeof(float), 1);
    sq_ = ScalarQuantizer(sq_type, vmin, vmax);
}
}  // namespace nsparse
