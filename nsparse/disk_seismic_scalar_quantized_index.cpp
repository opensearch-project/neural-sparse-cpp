/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/disk_seismic_scalar_quantized_index.h"

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
#include "nsparse/utils/scalar_quantizer.h"
#include "nsparse/utils/vector_process.h"

namespace nsparse {
namespace {

// A quantizer described by an index file. bytes_per_value() treats anything but
// QT_8bit as 16-bit, so an undefined type would silently pick an element width
// rather than be rejected.
ScalarQuantizer quantizer_from_file(QuantizerType type, float vmin,
                                    float vmax) {
    if (type != QuantizerType::QT_8bit && type != QuantizerType::QT_16bit) {
        throw std::runtime_error(
            "index file declares an unknown quantizer type");
    }
    return ScalarQuantizer(type, vmin, vmax);
}

// The forward vectors were encoded at the ingest quantizer's width, and search
// strides them at the width the quantizer reports. A file where the two
// disagree would be read at the wrong stride, so reject it at load.
void throw_if_element_size_mismatch(const detail::InlineForwardIndex& fwd,
                                    const ScalarQuantizer& sq) {
    if (fwd.num_blocks() > 0 && fwd.element_size() != sq.bytes_per_value()) {
        throw std::runtime_error(
            "index file's element size disagrees with its quantizer type");
    }
}

// Score one doc's quantized vector against the dense query codes and push the
// integer dot product (decoded later), honoring visited-dedup and the id
// selector. `dense` and `vals` are element_size bytes per component.
inline void score_doc(idx_t doc_id, const term_t* comps, const uint8_t* vals,
                      size_t nnz, const std::vector<uint8_t>& dense,
                      size_t element_size, const IDSelector* id_selector,
                      detail::TopKHolder<idx_t>& heap,
                      absl::flat_hash_set<idx_t>& visited) {
    if (!visited.insert(doc_id).second) {
        return;
    }
    if (id_selector != nullptr && !id_selector->is_member(doc_id)) {
        return;
    }
    float score = 0.0F;
    if (element_size == U16) {
        score = detail::dot_product_uint16_dense(
            comps, reinterpret_cast<const uint16_t*>(vals), nnz,
            reinterpret_cast<const uint16_t*>(dense.data()));
    } else {
        score = detail::dot_product_uint8_dense(comps, vals, nnz, dense.data());
    }
    heap.add(score, doc_id);
}

// A candidate block: its summary score and its (posting list, cluster) address.
struct BlockCandidate {
    float score;
    uint32_t pl;
    uint32_t cid;
};

// Score every document of one block against the dense query codes, dedup via
// `visited` and honor the id selector. Vectors come from the inline forward
// index (fwd) when loaded, else from the in-RAM CSR of a fresh build.
void score_block(const detail::InlineForwardIndex* fwd,
                 const SparseVectors* vectors,
                 const std::vector<InvertedListClusters>& clusters, uint32_t pl,
                 uint32_t cid, const std::vector<uint8_t>& dense,
                 const IDSelector* id_selector, size_t element_size,
                 detail::TopKHolder<idx_t>& heap,
                 absl::flat_hash_set<idx_t>& visited) {
    if (fwd != nullptr) {
        const detail::BlockView bv = fwd->block(pl, cid);
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

DiskSeismicScalarQuantizedIndex::DiskSeismicScalarQuantizedIndex(int dim)
    : MmapIndex(dim),
      cluster_parameter_(detail::kDefaultSeismicClusterParams) {}

DiskSeismicScalarQuantizedIndex::DiskSeismicScalarQuantizedIndex(
    QuantizerType quantizer_type, float vmin, float vmax,
    SeismicClusterParameters parameter, int dim)
    : MmapIndex(dim),
      sq_(quantizer_type, vmin, vmax),
      cluster_parameter_(parameter) {}

void DiskSeismicScalarQuantizedIndex::read_csr(const char* file_path,
                                               Residency residency) {
    if (residency == Residency::kMmap) {
        throw std::invalid_argument(
            "mmap residency is not available for a quantized index: a mapped "
            "CSR is borrowed as float, and this index searches over codes");
    }
    MmapIndex::read_csr(file_path, residency);
}

void DiskSeismicScalarQuantizedIndex::add(idx_t n, const idx_t* indptr,
                                          const term_t* indices,
                                          const float* values) {
    throw_if_not_positive(n);
    throw_if_any_null(indptr, indices, values);
    const size_t indptr_size = n + 1;
    const size_t nnz = indptr[n];
    const size_t element_size = sq_.bytes_per_value();
    if (vectors_ == nullptr) {
        // Fresh container: start the count at 0 so a stale num_vectors_ (e.g.
        // left by a prior mmap load, which has no vectors_) cannot accumulate.
        num_vectors_ = 0;
        vectors_ = std::make_unique<SparseVectors>(SparseVectorsConfig{
            .element_size = element_size,
            .dimension = static_cast<size_t>(dimension_)});
    }
    std::vector<uint8_t> codes(nnz * element_size);
    sq_.encode(values, codes.data(), nnz);
    vectors_->add_vectors(indptr, indptr_size, indices, nnz, codes.data(),
                          nnz * element_size);
    num_vectors_ += n;
}

void DiskSeismicScalarQuantizedIndex::build() {
    clustered_inverted_lists = detail::build_inverted_lists_clusters(
        get_vectors(),
        {.element_size = sq_.bytes_per_value(),
         .dimension = static_cast<size_t>(get_dimension())},
        cluster_parameter_);
}

// The quantizer a query is encoded with: DiskSeismicSQSearchParameters
// overrides the range the index was built with, anything else reuses it. The
// type is always the index's, since the codes are compared against stored ones.
ScalarQuantizer DiskSeismicScalarQuantizedIndex::query_quantizer(
    const SearchParameters* search_parameters) const {
    const auto* sq_params =
        dynamic_cast<const DiskSeismicSQSearchParameters*>(search_parameters);
    if (sq_params == nullptr) {
        return sq_;
    }
    return ScalarQuantizer(sq_.get_quantizer_type(), sq_params->vmin,
                           sq_params->vmax);
}

auto DiskSeismicScalarQuantizedIndex::search(
    idx_t n, const idx_t* indptr, const term_t* indices, const float* values,
    int k, SearchParameters* search_parameters) -> pair_of_score_id_vectors_t {
    // Quit early when there is nothing to score: no vectors, no queries, or no
    // forward-vector source (fwd_ empty and vectors_ null).
    if (num_vectors_ == 0 || n == 0 ||
        (fwd_.num_blocks() == 0 && vectors_ == nullptr)) {
        return {
            std::vector<std::vector<float>>(n, std::vector<float>(k, -1.0F)),
            std::vector<std::vector<idx_t>>(
                n, std::vector<idx_t>(k, detail::INVALID_IDX))};
    }

    // Resolve `cut` and `k_prime`. A DiskSeismicSearchParameters (including a
    // DiskSeismicSQSearchParameters) carries k_prime; a plain
    // SeismicSearchParameters (or null) uses the default budget.
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

    if (k_prime <= 0) {
        throw std::invalid_argument(
            "DiskSeismicScalarQuantizedIndex: k_prime (block budget) must be "
            "positive");
    }

    // Quantize the whole query batch once into `codes`, in the same CSR order
    // as `indices`, so a query's codes are query_values + start * element_size.
    const ScalarQuantizer query_sq = query_quantizer(search_parameters);
    const size_t element_size = sq_.bytes_per_value();
    const size_t nnz = indptr[n];
    std::vector<uint8_t> codes(nnz * element_size);
    query_sq.encode(values, codes.data(), nnz);
    const uint8_t* query_values = codes.data();

    std::vector<std::vector<float>> result_distances(
        n, std::vector<float>(k, -1.0F));
    std::vector<std::vector<idx_t>> result_labels(
        n, std::vector<idx_t>(k, detail::INVALID_IDX));
    const size_t dense_bytes = static_cast<size_t>(dimension_) * element_size;

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
            const uint8_t* q_val_bytes =
                query_values + static_cast<size_t>(start) * element_size;
            std::vector<term_t> cuts;
            if (element_size == U16) {
                cuts = detail::top_k_tokens<uint16_t>(
                    q_indices, reinterpret_cast<const uint16_t*>(q_val_bytes),
                    len, cut);
            } else {
                cuts = detail::top_k_tokens<uint8_t>(q_indices, q_val_bytes,
                                                     len, cut);
            }
            auto [distances, labels] =
                single_query(dense, visited, q_indices, q_val_bytes, len,
                             element_size, cuts, k, k_prime, query_sq,
                             search_parameters);
            result_distances[query_idx] = std::move(distances);
            result_labels[query_idx] = std::move(labels);
        }
    }
    return {result_distances, result_labels};
}

auto DiskSeismicScalarQuantizedIndex::single_query(
    std::vector<uint8_t>& dense, absl::flat_hash_set<idx_t>& visited,
    const term_t* q_idx, const uint8_t* q_val_bytes, size_t q_len,
    size_t element_size, const std::vector<term_t>& cuts, int k, int k_prime,
    const ScalarQuantizer& query_sq, SearchParameters* search_parameters)
    -> pair_of_score_id_vector_t {
    if (num_vectors_ == 0) {
        return {{}, {}};
    }
    // Scatter the query's quantized codes into the dense lookup table:
    // element_size contiguous bytes per non-zero dim.
    for (size_t i = 0; i < q_len; ++i) {
        std::copy_n(q_val_bytes + i * element_size, element_size,
                    dense.data() + static_cast<size_t>(q_idx[i]) * element_size);
    }
    visited.clear();

    // Prefer the mapped inline forward index; fall back to the in-RAM vectors
    // of a fresh build.
    const detail::InlineForwardIndex* fwd =
        fwd_.num_blocks() > 0 ? &fwd_ : nullptr;
    const SparseVectors* vectors = fwd == nullptr ? vectors_.get() : nullptr;
    const IDSelector* id_selector = search_parameters == nullptr
                                        ? nullptr
                                        : search_parameters->get_id_selector();

    // Collect every candidate block across the cut lists with its summary
    // score. score_summaries_transposed dots the query codes with each cluster
    // summary (also stored as codes), so scores are comparable across posting
    // lists for the global ranking.
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
        cluster_invlist.score_summaries_transposed(q_idx, q_val_bytes, q_len,
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

    // Restore only the query's own positions to zero (mirrors the scatter at
    // entry): element_size bytes per touched dim.
    for (size_t i = 0; i < q_len; ++i) {
        std::fill_n(dense.data() + static_cast<size_t>(q_idx[i]) * element_size,
                    element_size, uint8_t{0});
    }

    auto [scores, ids] = holder.top_k_items_descending();
    // Decode the integer dot products back to float. The -1.0 padding is added
    // after, so the sentinel is never scaled.
    for (auto& score : scores) {
        score = sq_.decode_dot_product(score, query_sq);
    }
    scores.resize(k, -1.0F);
    ids.resize(k, detail::INVALID_IDX);
    return {scores, ids};
}

void DiskSeismicScalarQuantizedIndex::write_index(IOWriter* io_writer) {
    write_quantizer_header(io_writer);
    const uint64_t nv = num_vectors_;
    io_writer->write(const_cast<uint64_t*>(&nv), sizeof(uint64_t), 1);
    // Summaries only: the doc-id membership is already in the inline forward
    // index below, so writing it in the posting lists too would duplicate it.
    SeismicInvertedListsWriter inv_list_writer(clustered_inverted_lists,
                                               /*summaries_only=*/true);
    inv_list_writer.serialize(io_writer);
    // Inline forward index, built from the same clusters + vectors. An empty
    // corpus uses a correctly-typed (code-width) empty SparseVectors so the
    // section still round-trips.
    SparseVectors empty_vectors(
        {.element_size = sq_.bytes_per_value(),
         .dimension = static_cast<size_t>(dimension_)});
    const SparseVectors& v = vectors_ != nullptr ? *vectors_ : empty_vectors;
    detail::InlineForwardIndex forward(clustered_inverted_lists, v);
    forward.serialize(io_writer);
}

void DiskSeismicScalarQuantizedIndex::read_index(
    IOReader* /*io_reader*/, const IndexHeader& /*header*/, int /*io_flags*/) {
    // The inline forward index is borrowed from a mapping, never copied onto
    // the heap, so this index has no copying read path.
    throw std::runtime_error(
        "DiskSeismicScalarQuantizedIndex is mmap-only; load with "
        "read_index(file, IndexIoFlag::kUseMmap)");
}

void DiskSeismicScalarQuantizedIndex::write_quantizer_header(
    IOWriter* io_writer) {
    auto sq_type = sq_.get_quantizer_type();
    io_writer->write(&sq_type, sizeof(QuantizerType), 1);
    auto vmin = sq_.get_min();
    io_writer->write(&vmin, sizeof(float), 1);
    auto vmax = sq_.get_max();
    io_writer->write(&vmax, sizeof(float), 1);
}

DiskSeismicScalarQuantizedIndex* DiskSeismicScalarQuantizedIndex::mmap_index(
    const IndexHeader& header, const char* index_file, size_t pos) {
    throw_if_null(index_file, "index_file must not be null");
    auto index =
        std::make_unique<DiskSeismicScalarQuantizedIndex>(header.dimension);

    MmapFile mmap_file(std::string{index_file});
    MmapCursor cursor(mmap_file.data(), mmap_file.size());
    cursor.skip(pos);

    // Same order write_index wrote them: quantizer header, doc count,
    // summaries, inline forward.
    const auto sq_type = cursor.read_scalar<QuantizerType>();
    const auto vmin = cursor.read_scalar<float>();
    const auto vmax = cursor.read_scalar<float>();
    const ScalarQuantizer sq = quantizer_from_file(sq_type, vmin, vmax);

    const uint64_t nv = cursor.read_scalar<uint64_t>();
    SeismicInvertedListsWriter inv_list_writer;
    inv_list_writer.mmap_deserialize(&cursor);
    detail::InlineForwardIndex forward;
    forward.mmap_deserialize(&cursor);
    throw_if_element_size_mismatch(forward, sq);

    // Committed only once everything parsed. mapped_file_ last: the summaries
    // and the forward index borrow from it, and moving it does not move the
    // mapping.
    index->sq_ = sq;
    index->clustered_inverted_lists = std::move(inv_list_writer.release());
    index->fwd_ = std::move(forward);
    index->num_vectors_ = nv;
    index->mapped_file_ = std::move(mmap_file);
    return index.release();
}
}  // namespace nsparse
