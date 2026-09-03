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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
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
#include "nsparse/utils/mmap_file.h"
#include "nsparse/utils/prefetch.h"
#include "nsparse/utils/scalar_quantizer.h"
#include "nsparse/utils/vector_process.h"

namespace nsparse {
namespace {

// A quantizer described by an index file. bytes_per_value() treats anything but
// QT_8bit as 16-bit, so an undefined type would silently pick an element width
// rather than be rejected; the constructor covers a non-finite or empty range.
ScalarQuantizer quantizer_from_file(QuantizerType type, float vmin,
                                    float vmax) {
    if (type != QuantizerType::QT_8bit && type != QuantizerType::QT_16bit) {
        throw std::runtime_error(
            "index file declares an unknown quantizer type");
    }
    return ScalarQuantizer(type, vmin, vmax);
}

// The stored values were encoded at the ingest quantizer's width, and search
// strides them at the width the quantizer reports. A file where the two
// disagree would be read at the wrong stride, so reject it at load.
void throw_if_element_size_mismatch(const SparseVectors& vectors,
                                    const ScalarQuantizer& sq) {
    if (vectors.num_vectors() > 0 &&
        vectors.get_element_size() != sq.bytes_per_value()) {
        throw std::runtime_error(
            "index file's element size disagrees with its quantizer type");
    }
}

void query_single_inverted_list(const SparseVectors* vectors,
                                const InvertedListClusters& cluster_invlist,
                                const std::vector<uint8_t>& dense,
                                const term_t* q_idx, const uint8_t* q_val_bytes,
                                size_t q_len, std::vector<float>& score_scratch,
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
    // Query-driven summary scoring via the term-major transpose. The query
    // values are the quantized codes in the summaries' element width.
    cluster_invlist.score_summaries_transposed(q_idx, q_val_bytes, q_len,
                                               score_scratch);
    const std::vector<float>& summary_scores = score_scratch;
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
            float score = detail::compute_similarity(
                doc_id, indptr, indices, values, dense.data(), element_size);
            heap.add(score, doc_id);
        }
    }
}
}  // namespace

SeismicScalarQuantizedIndex::SeismicScalarQuantizedIndex(int dim)
    : MmapIndex(dim),
      cluster_parameter_(detail::kDefaultSeismicClusterParams) {}

SeismicScalarQuantizedIndex::SeismicScalarQuantizedIndex(
    QuantizerType quantizer_type, float vmin, float vmax,
    SeismicClusterParameters parameter, int dim)
    : MmapIndex(dim),
      sq_(quantizer_type, vmin, vmax),
      cluster_parameter_(parameter) {}

void SeismicScalarQuantizedIndex::read_csr(const char* file_path,
                                           Residency residency) {
    if (residency == Residency::kMmap) {
        throw std::invalid_argument(
            "mmap residency is not available for a quantized index: a mapped "
            "CSR is borrowed as float, and this index searches over codes");
    }
    MmapIndex::read_csr(file_path, residency);
}

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

// The quantizer a query is encoded with: SeismicSQSearchParameters overrides
// the range the index was built with, anything else reuses it. The type is
// always the index's, since the codes are compared against stored ones.
ScalarQuantizer SeismicScalarQuantizedIndex::query_quantizer(
    const SearchParameters* search_parameters) const {
    const auto* sq_params =
        dynamic_cast<const SeismicSQSearchParameters*>(search_parameters);
    if (sq_params == nullptr) {
        return sq_;
    }
    return ScalarQuantizer(sq_.get_quantizer_type(), sq_params->vmin,
                           sq_params->vmax);
}

void SeismicScalarQuantizedIndex::build() {
    // add() has already quantized the values, so the shared build reads their
    // width off the corpus and needs to know nothing of the quantizer.
    clustered_inverted_lists = detail::build_clustered_lists(
        get_vectors(), static_cast<size_t>(get_dimension()), cluster_parameter_,
        &batch_spill_);
}

auto SeismicScalarQuantizedIndex::search(idx_t n, const idx_t* indptr,
                                         const term_t* indices,
                                         const float* values, int k,
                                         SearchParameters* search_parameters)
    -> pair_of_score_id_vectors_t {
    if (vectors_ == nullptr || n == 0) {
        return {std::vector<std::vector<float>>(n),
                std::vector<std::vector<idx_t>>(n)};
    }
    size_t indptr_size = n + 1;
    size_t nnz = indptr[n];  // Total non-zeros

    const ScalarQuantizer query_sq = query_quantizer(search_parameters);

    // Quantize the whole query batch once. `codes` holds the quantized values
    // in the same CSR order as `indices`, so the batch (indptr/indices/codes)
    // can be indexed directly below — no per-query SparseVectors needed.
    const size_t element_size = sq_.bytes_per_value();
    std::vector<uint8_t> codes(nnz * element_size);
    query_sq.encode(values, codes.data(), nnz);
    const uint8_t* query_values = codes.data();

    const IDSelector* id_selector = search_parameters == nullptr
                                        ? nullptr
                                        : search_parameters->get_id_selector();
    // if filter ids size is <= k, just run exact match. Only this path needs a
    // SparseVectors view of the query, so build one lazily just for it.
    if (detail::should_run_exact_match(id_selector, k, nullptr)) {
        SparseVectors query_vectors(
            {.element_size = element_size,
             .dimension = static_cast<size_t>(dimension_)});
        query_vectors.add_vectors(indptr, indptr_size, indices, nnz,
                                  codes.data(), nnz * element_size);
        auto [distances, labels] = detail::ExactMatcher::search(
            vectors_.get(),
            dynamic_cast<const IDSelectorEnumerable*>(id_selector),
            &query_vectors, element_size, k);
        // Decode quantized dot product scores, leaving the -1 padding
        // ExactMatcher appended for the slots it could not fill: scaling that
        // sentinel would turn it into an ordinary small negative score.
        for (size_t query_idx = 0; query_idx < distances.size(); ++query_idx) {
            auto& query_distances = distances[query_idx];
            const auto& query_labels = labels[query_idx];
            for (size_t i = 0; i < query_distances.size(); ++i) {
                if (query_labels[i] != detail::INVALID_IDX) {
                    query_distances[i] =
                        sq_.decode_dot_product(query_distances[i], query_sq);
                }
            }
        }
        return {distances, labels};
    }

    std::vector<std::vector<float>> result_distances(n);
    std::vector<std::vector<idx_t>> result_labels(n);

    // query. Parameters that carry no cut / heap factor (none at all, or a
    // plain SearchParameters holding just an id selector) fall back to the
    // defaults rather than being dereferenced as null.
    SeismicSearchParameters default_params;
    const auto* seismic_parameters =
        dynamic_cast<const SeismicSearchParameters*>(search_parameters);
    const auto* parameters =
        seismic_parameters != nullptr ? seismic_parameters : &default_params;
    const size_t dense_bytes = static_cast<size_t>(dimension_) * element_size;

    // Per-thread scratch reused across all queries a thread handles: a
    // dimension-sized quantized-code dense buffer (kept all-zero between
    // queries via a sparse clear inside single_query) and the visited-doc set.
    // This replaces the previous per-query allocation of both.
    // schedule(dynamic, 64) matches the coarse-chunk scheduling used by
    // SeismicIndex::search.
#pragma omp parallel
    {
        std::vector<uint8_t> dense(dense_bytes, 0);
        absl::flat_hash_set<idx_t> visited;
        visited.reserve(static_cast<size_t>(std::max(k, 1)) * 4096);

#pragma omp for schedule(dynamic, 64)
        for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
            const idx_t start = indptr[query_idx];
            const size_t len = indptr[query_idx + 1] - start;
            const term_t* q_indices = indices + start;
            const uint8_t* q_val_bytes = query_values + start * element_size;
            std::vector<term_t> cuts;
            if (element_size == U16) {
                cuts = detail::top_k_tokens<uint16_t>(
                    q_indices, reinterpret_cast<const uint16_t*>(q_val_bytes),
                    len, parameters->cut);
            } else {
                cuts = detail::top_k_tokens<uint8_t>(q_indices, q_val_bytes,
                                                     len, parameters->cut);
            }

            auto [distances, labels] = single_query(
                dense, visited, q_indices, q_val_bytes, len, element_size, cuts,
                k, parameters->heap_factor, query_sq, search_parameters);
            result_distances[query_idx] = std::move(distances);
            result_labels[query_idx] = std::move(labels);
        }
    }

    return {result_distances, result_labels};
}

auto SeismicScalarQuantizedIndex::single_query(
    std::vector<uint8_t>& dense, absl::flat_hash_set<idx_t>& visited,
    const term_t* q_idx, const uint8_t* q_val_bytes, size_t q_len,
    size_t element_size, const std::vector<term_t>& cuts, int k,
    float heap_factor, const ScalarQuantizer& query_sq,
    SearchParameters* search_parameters) -> pair_of_score_id_vector_t {
    size_t num_docs = vectors_->num_vectors();
    if (num_docs == 0) {
        return {{}, {}};
    }

    // Scatter the query's quantized codes into the reused dense buffer
    // (all-zero on entry): element_size contiguous bytes per non-zero dim.
    for (size_t i = 0; i < q_len; ++i) {
        std::copy_n(
            q_val_bytes + i * element_size, element_size,
            dense.data() + static_cast<size_t>(q_idx[i]) * element_size);
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
                                   q_idx, q_val_bytes, q_len, score_scratch,
                                   heap_factor, first_list, search_parameters,
                                   holder, visited);
        first_list = false;
    }

    // Restore the dense buffer to all-zero for the next query on this thread
    // (sparse clear over only the dims this query touched).
    for (size_t i = 0; i < q_len; ++i) {
        std::fill_n(dense.data() + static_cast<size_t>(q_idx[i]) * element_size,
                    element_size, uint8_t{0});
    }

    auto [distances, labels] = holder.top_k_items_descending();

    // Decode quantized dot product scores
    for (auto& dist : distances) {
        dist = sq_.decode_dot_product(dist, query_sq);
    }
    distances.resize(k, -1.0F);
    labels.resize(k, detail::INVALID_IDX);
    return {distances, labels};
}

void SeismicScalarQuantizedIndex::write_index(IOWriter* io_writer) {
    write_quantization_header(io_writer);
    // write vectors
    if (vectors_ == nullptr) {
        empty_sparse_vectors.serialize(io_writer);
    } else {
        vectors_->serialize(io_writer);
    }
    SeismicInvertedListsWriter inv_list_writer(clustered_inverted_lists);
    inv_list_writer.serialize(io_writer);
}

void SeismicScalarQuantizedIndex::read_index(IOReader* io_reader,
                                             const IndexHeader& header,
                                             int io_flags) {
    read_quantization_header(io_reader);
    SparseVectors tmp_vectors;
    tmp_vectors.deserialize(io_reader);
    throw_if_element_size_mismatch(tmp_vectors, sq_);
    if (tmp_vectors.num_vectors() > 0) {
        vectors_ = std::make_unique<SparseVectors>(std::move(tmp_vectors));
    }
    SeismicInvertedListsWriter inv_list_writer;
    inv_list_writer.deserialize(io_reader);
    clustered_inverted_lists = std::move(inv_list_writer.release());
}

SeismicScalarQuantizedIndex* SeismicScalarQuantizedIndex::mmap_index(
    const IndexHeader& header, const char* index_file, size_t pos) {
    throw_if_null(index_file, "index_file must not be null");
    auto index =
        std::make_unique<SeismicScalarQuantizedIndex>(header.dimension);

    MmapFile mmap_file(std::string{index_file});
    // `pos` is where write_index's payload begins, past the header read_header
    // consumed. Absolute file offsets are what serialize() padded against, so
    // the cursor has to start at 0 and skip rather than map from `pos`.
    MmapCursor cursor(mmap_file.data(), mmap_file.size());
    cursor.skip(pos);

    // Same order write_index wrote them, starting with the quantizer header.
    const auto sq_type = cursor.read_scalar<QuantizerType>();
    const auto vmin = cursor.read_scalar<float>();
    const auto vmax = cursor.read_scalar<float>();
    const ScalarQuantizer sq = quantizer_from_file(sq_type, vmin, vmax);

    auto vectors = std::make_unique<SparseVectors>();
    vectors->mmap_deserialize(&cursor);
    throw_if_element_size_mismatch(*vectors, sq);

    SeismicInvertedListsWriter inv_list_writer;
    inv_list_writer.mmap_deserialize(&cursor);

    // Committed only once everything parsed, so a corrupt file cannot leave a
    // half-mapped index behind. mapped_file_ last: the arrays above borrow from
    // it, and moving it does not move the mapping itself.
    index->sq_ = sq;
    index->clustered_inverted_lists = std::move(inv_list_writer.release());
    if (vectors->num_vectors() > 0) {
        index->vectors_ = std::move(vectors);
    }
    index->mapped_file_ = std::move(mmap_file);
    return index.release();
}

void SeismicScalarQuantizedIndex::write_quantization_header(
    IOWriter* io_writer) {
    auto sq_type = sq_.get_quantizer_type();
    io_writer->write(&sq_type, sizeof(QuantizerType), 1);
    auto vmin = sq_.get_min();
    io_writer->write(&vmin, sizeof(float), 1);
    auto vmax = sq_.get_max();
    io_writer->write(&vmax, sizeof(float), 1);
}

void SeismicScalarQuantizedIndex::read_quantization_header(
    IOReader* io_reader) {
    QuantizerType sq_type = QuantizerType::QT_8bit;
    float vmin = 0.0F;
    float vmax = 1.0F;
    io_reader->read(&sq_type, sizeof(QuantizerType), 1);
    io_reader->read(&vmin, sizeof(float), 1);
    io_reader->read(&vmax, sizeof(float), 1);
    sq_ = quantizer_from_file(sq_type, vmin, vmax);
}
}  // namespace nsparse
