/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/cluster/inverted_list_clusters.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "nsparse/io/align.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"
#include "nsparse/utils/mmap_file.h"
#ifdef NSPARSE_WITH_GPU
#include "nsparse/gpu/gpu_summarizer.h"
#endif

namespace nsparse {
namespace {

// Prune a cluster's (term, max-value) pairs at the alpha mass cutoff and append
// the survivors (in ascending term order) as a summary vector. Shared by the
// CPU and GPU max-pool paths so their output is identical.
template <class T>
void summarize_emit_cluster_(std::vector<std::pair<term_t, T>>& pairs,
                             float sum, float alpha, SparseVectors& out) {
    std::ranges::sort(pairs, [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    float addup = 0.0F;
    for (size_t j = 0; j < pairs.size(); ++j) {
        addup += pairs[j].second;
        if (addup / sum >= alpha) {
            pairs.erase(pairs.begin() + j + 1, pairs.end());
            break;
        }
    }
    std::ranges::sort(pairs, [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    std::vector<term_t> terms;
    std::vector<T> values;
    terms.reserve(pairs.size());
    values.reserve(pairs.size());
    for (const auto& [term, value] : pairs) {
        terms.push_back(term);
        values.push_back(value);
    }
    out.add_vector(terms.data(), terms.size(),
                   reinterpret_cast<const uint8_t*>(values.data()),
                   values.size() * sizeof(T));
}

// CPU per-term max-pool over a flat, dim-sized accumulator reused across all
// clusters (replaces a per-cluster std::unordered_map, ~2.8x cheaper for the
// many small clusters). A per-cluster epoch marks live slots, so the buffers
// need no per-cluster reset; first touch is detected via the epoch (not
// acc==0), robust to zero-valued weights.
template <class T>
SparseVectors summarize_with_cpu_(const SparseVectors* vectors,
                                  std::span<const idx_t> group_of_doc_ids,
                                  std::span<const idx_t> offsets, float alpha) {
    SparseVectors out({.element_size = vectors->get_element_size(),
                       .dimension = vectors->get_dimension()});
    const size_t n_clusters = offsets.size() - 1;
    const auto& indptr_data = vectors->indptr_data();
    const auto& indices_data = vectors->indices_data();
    const auto& values_data = vectors->values_data();
    const size_t dim = vectors->get_dimension();

    std::vector<T> acc(dim, T(0));
    std::vector<uint32_t> epoch(dim, 0);
    std::vector<term_t> touched;
    uint32_t cur_epoch = 0;

    for (size_t i = 0; i < n_clusters; ++i) {
        ++cur_epoch;
        touched.clear();
        float sum = 0.0F;
        auto doc_ids = std::span<const idx_t>(
            group_of_doc_ids.data() + offsets[i], offsets[i + 1] - offsets[i]);
        for (const auto& doc_id : doc_ids) {
            int start = indptr_data[doc_id];
            int end = indptr_data[doc_id + 1];
            for (size_t j = start; j < end; ++j) {
                const term_t term = indices_data[j];
                // j is element index, need byte offset for T access
                const T v =
                    *reinterpret_cast<const T*>(values_data + j * sizeof(T));
                if (epoch[term] != cur_epoch) {
                    // First occurrence in this cluster; value = max(0, v).
                    epoch[term] = cur_epoch;
                    const T value = std::max(T(0), v);
                    acc[term] = value;
                    touched.push_back(term);
                    sum += value;
                } else if (v > acc[term]) {
                    sum += v - acc[term];
                    acc[term] = v;
                }
            }
        }
        std::vector<std::pair<term_t, T>> pairs;
        pairs.reserve(touched.size());
        for (const term_t term : touched) {
            pairs.emplace_back(term, acc[term]);
        }
        summarize_emit_cluster_<T>(pairs, sum, alpha, out);
    }
    return out;
}

#ifdef NSPARSE_WITH_GPU
// GPU per-term max-pool: one kernel launch for the whole list, then the shared
// CPU sort/truncate. float (U32) only. Returns std::nullopt when the GPU
// declines (unavailable / empty list) so the caller falls back to the CPU path.
template <class T>
std::optional<SparseVectors> summarize_with_gpu_(
    const SparseVectors* vectors, std::span<const idx_t> group_of_doc_ids,
    std::span<const idx_t> offsets, float alpha) {
    if constexpr (!std::is_same_v<T, float>) {
        return std::nullopt;  // GPU max-pool is float-only
    } else {
        const size_t n_clusters = offsets.size() - 1;
        std::vector<detail::GpuSummarizer::ClusterSummary> gpu_clusters;
        if (!detail::GpuSummarizer::instance().summarize_list(
                vectors, group_of_doc_ids.data(), offsets.data(), n_clusters,
                gpu_clusters)) {
            return std::nullopt;
        }
        SparseVectors out({.element_size = vectors->get_element_size(),
                           .dimension = vectors->get_dimension()});
        for (size_t i = 0; i < n_clusters; ++i) {
            const auto& gc = gpu_clusters[i];
            std::vector<std::pair<term_t, T>> pairs;
            pairs.reserve(gc.terms.size());
            for (size_t t = 0; t < gc.terms.size(); ++t) {
                pairs.emplace_back(gc.terms[t], gc.values[t]);
            }
            summarize_emit_cluster_<T>(pairs, gc.sum, alpha, out);
        }
        return out;
    }
}
#endif

/**
 * @brief Generate the per-cluster summary sparse vector (CSR) for a posting
 *        list's clusters.
 *
 * @param vectors inverted index
 * @param group_of_doc_ids flattened doc ids of all clusters
 * @param offsets cluster boundaries into group_of_doc_ids
 * @param alpha prune ratio
 * @return SparseVectors  one summary vector per cluster
 */
template <class T>
SparseVectors summarize_(const SparseVectors* vectors,
                         std::span<const idx_t> group_of_doc_ids,
                         std::span<const idx_t> offsets, float alpha) {
    if (offsets.size() <= 1) {
        return SparseVectors({.element_size = vectors->get_element_size(),
                              .dimension = vectors->get_dimension()});
    }
#ifdef NSPARSE_WITH_GPU
    // GPU max-pool is opt-in (NSPARSE_GPU_SUMMARIZE=1); fall back to CPU when
    // disabled, unsupported for T, or the GPU declines.
    if (detail::should_offload_summarize_to_gpu()) {
        if (auto gpu_result = summarize_with_gpu_<T>(vectors, group_of_doc_ids,
                                                     offsets, alpha)) {
            return std::move(*gpu_result);
        }
    }
#endif
    return summarize_with_cpu_<T>(vectors, group_of_doc_ids, offsets, alpha);
}

}  // namespace

InvertedListClusters::InvertedListClusters(
    const std::vector<std::vector<idx_t>>& docs) {
    if (docs.empty()) return;
    // Flattened into locals and handed over at the end: a Buf is fixed-length,
    // so it cannot be appended to in place.
    std::vector<idx_t> flat_docs;
    std::vector<idx_t> offsets;
    offsets.reserve(docs.size() + 1);
    offsets.push_back(0);
    for (const auto& doc_ids : docs) {
        flat_docs.insert(flat_docs.end(), doc_ids.begin(), doc_ids.end());
        offsets.push_back(static_cast<idx_t>(flat_docs.size()));
    }
    docs_ = Buf<idx_t>::own(std::move(flat_docs));
    offsets_ = Buf<idx_t>::own(std::move(offsets));
}

auto InvertedListClusters::get_docs(idx_t idx) const -> std::span<const idx_t> {
    // A summaries-only load leaves offsets_ empty; return an empty span rather
    // than indexing it out of bounds. Populated lists are unaffected.
    if (offsets_.size() < static_cast<size_t>(idx) + 2) {
        return {};
    }
    return {docs_.data() + offsets_[idx],
            static_cast<size_t>(offsets_[idx + 1] - offsets_[idx])};
}

void InvertedListClusters::summarize(const SparseVectors* vectors, float alpha) {
    const auto element_size = vectors->get_element_size();
    const std::span<const idx_t> docs = docs_.span();
    const std::span<const idx_t> offsets = offsets_.span();
    SparseVectors summaries;
    if (element_size == U32) {
        summaries = summarize_<float>(vectors, docs, offsets, alpha);
    } else if (element_size == U16) {
        summaries = summarize_<uint16_t>(vectors, docs, offsets, alpha);
    } else {
        summaries = summarize_<uint8_t>(vectors, docs, offsets, alpha);
    }
    build_transpose(summaries);
}

void InvertedListClusters::build_transpose(const SparseVectors& summaries) {
    n_clusters_ = summaries.num_vectors();
    element_size_ = summaries.get_element_size();
    // Reassigned wholesale below, so an early return leaves them empty rather
    // than holding a previous build's transpose.
    term_ids_ = {};
    term_ptr_ = {};
    csc_cluster_ = {};
    csc_value_ = {};
    if (n_clusters_ == 0) {
        return;
    }
    // Every cluster index must fit in cluster_id_t; see its declaration.
    assert(n_clusters_ <= std::numeric_limits<cluster_id_t>::max() + size_t{1});

    const auto* indptr = summaries.indptr_data();
    const auto* indices = summaries.indices_data();
    const auto* values = summaries.values_data();  // raw bytes
    const size_t nnz = static_cast<size_t>(indptr[n_clusters_]);
    const size_t esz = element_size_;

    // Built in locals and handed to Buf::own at the end: a Buf is read-only and
    // fixed-length, so the counting sort cannot fill one in place.

    // Distinct summary terms, ascending. sort+unique over the summary indices
    // keeps the working set proportional to nnz, not to the dimension.
    std::vector<term_t> term_ids(indices, indices + nnz);
    std::ranges::sort(term_ids);
    term_ids.erase(std::ranges::unique(term_ids).begin(), term_ids.end());
    const size_t n_terms = term_ids.size();

    // Maps a term to its index in term_ids (its CSC column).
    auto term_column = [&term_ids](term_t term) -> size_t {
        return static_cast<size_t>(std::ranges::lower_bound(term_ids, term) -
                                   term_ids.begin());
    };

    // Counting sort: entries per term, prefix-summed into the CSC offsets.
    std::vector<idx_t> term_ptr(n_terms + 1, 0);
    for (size_t j = 0; j < nnz; ++j) {
        term_ptr[term_column(indices[j]) + 1]++;
    }
    for (size_t t = 0; t < n_terms; ++t) term_ptr[t + 1] += term_ptr[t];

    std::vector<cluster_id_t> csc_cluster(nnz);
    std::vector<uint8_t> csc_value(nnz * esz);
    std::vector<idx_t> cursor(term_ptr.begin(), term_ptr.end() - 1);
    for (size_t cluster = 0; cluster < n_clusters_; ++cluster) {
        const idx_t start = indptr[cluster];
        const idx_t end = indptr[cluster + 1];
        for (idx_t j = start; j < end; ++j) {
            const size_t col = term_column(indices[j]);
            const idx_t pos = cursor[col]++;
            csc_cluster[pos] = static_cast<cluster_id_t>(cluster);
            std::copy_n(values + static_cast<size_t>(j) * esz, esz,
                        csc_value.data() + static_cast<size_t>(pos) * esz);
        }
    }

    term_ids_ = Buf<term_t>::own(std::move(term_ids));
    term_ptr_ = Buf<idx_t>::own(std::move(term_ptr));
    csc_cluster_ = Buf<cluster_id_t>::own(std::move(csc_cluster));
    csc_value_ = Buf<uint8_t>::own(std::move(csc_value));
}

template <class T>
void InvertedListClusters::score_summaries_typed(
    const term_t* q_idx, const T* q_val, size_t q_len,
    std::vector<float>& out) const {
    const T* csc_values = reinterpret_cast<const T*>(csc_value_.data());
    for (size_t i = 0; i < q_len; ++i) {
        const term_t term = q_idx[i];
        // Terms absent from the summaries contribute nothing.
        auto it = std::ranges::lower_bound(term_ids_, term);
        if (it == term_ids_.end() || *it != term) {
            continue;
        }
        const size_t col = static_cast<size_t>(it - term_ids_.begin());
        const float qv = static_cast<float>(q_val[i]);
        const idx_t start = term_ptr_[col];
        const idx_t end = term_ptr_[col + 1];
        for (idx_t j = start; j < end; ++j) {
            out[csc_cluster_[j]] += qv * static_cast<float>(csc_values[j]);
        }
    }
}

void InvertedListClusters::score_summaries_transposed(
    const term_t* q_idx, const uint8_t* q_val_bytes, size_t q_len,
    std::vector<float>& out) const {
    out.assign(n_clusters_, 0.0F);
    if (n_clusters_ == 0 || term_ids_.empty()) return;
    if (element_size_ == U32) {
        score_summaries_typed<float>(
            q_idx, reinterpret_cast<const float*>(q_val_bytes), q_len, out);
    } else if (element_size_ == U16) {
        score_summaries_typed<uint16_t>(
            q_idx, reinterpret_cast<const uint16_t*>(q_val_bytes), q_len, out);
    } else {
        score_summaries_typed<uint8_t>(q_idx, q_val_bytes, q_len, out);
    }
}

void InvertedListClusters::serialize(IOWriter* writer) const {
    // Each array is preceded by padding to its element's alignment so a mapped
    // reader can borrow it in place; see io/align.h.
    size_t n_docs = docs_.size();
    writer->write(&n_docs, sizeof(size_t), 1);
    io_align::write_padded(writer, docs_.data(), n_docs);
    size_t n_offsets = offsets_.size();
    writer->write(&n_offsets, sizeof(size_t), 1);
    io_align::write_padded(writer, offsets_.data(), n_offsets);

    serialize_summary_store(writer);
}

void InvertedListClusters::serialize_summaries_only(IOWriter* writer) const {
    // docs_/offsets_ emitted as count-0 arrays: the membership is redundant
    // with the inline forward index and unread on the mmap path. Count-0 keeps
    // the byte layout identical (io/align.h), so the reader is unchanged.
    size_t zero = 0;
    writer->write(&zero, sizeof(size_t), 1);
    io_align::write_padded<idx_t>(writer, nullptr, 0);
    writer->write(&zero, sizeof(size_t), 1);
    io_align::write_padded<idx_t>(writer, nullptr, 0);

    serialize_summary_store(writer);
}

void InvertedListClusters::serialize_summary_store(IOWriter* writer) const {
    // Transposed (CSC) summary store.
    size_t n_clusters = n_clusters_;
    writer->write(&n_clusters, sizeof(size_t), 1);
    size_t element_size = element_size_;
    writer->write(&element_size, sizeof(size_t), 1);
    size_t n_terms = term_ids_.size();
    writer->write(&n_terms, sizeof(size_t), 1);
    if (n_terms > 0) {
        io_align::write_padded(writer, term_ids_.data(), n_terms);
        // term_ids_ is 2 bytes wide, so an odd term count leaves this 4-byte
        // array off its boundary without the pad.
        io_align::write_padded(writer, term_ptr_.data(), n_terms + 1);
    }
    size_t nnz = csc_cluster_.size();
    writer->write(&nnz, sizeof(size_t), 1);
    io_align::write_padded(writer, csc_cluster_.data(), nnz);
    // Values are bytes on the wire but reinterpreted as element_size-wide
    // words on read, so they are padded to that width, not to 1.
    io_align::write_padded(writer, csc_value_.data(), csc_value_.size(),
                           element_size_);
}

void InvertedListClusters::deserialize(IOReader* reader) {
    size_t n_docs = 0;
    reader->read(&n_docs, sizeof(size_t), 1);
    docs_ = io_align::read_padded<idx_t>(reader, n_docs);
    size_t n_offsets = 0;
    reader->read(&n_offsets, sizeof(size_t), 1);
    offsets_ = io_align::read_padded<idx_t>(reader, n_offsets);

    reader->read(&n_clusters_, sizeof(size_t), 1);
    reader->read(&element_size_, sizeof(size_t), 1);
    size_t n_terms = 0;
    reader->read(&n_terms, sizeof(size_t), 1);
    if (n_terms > 0) {
        term_ids_ = io_align::read_padded<term_t>(reader, n_terms);
        term_ptr_ = io_align::read_padded<idx_t>(reader, n_terms + 1);
    } else {
        term_ids_ = {};
        term_ptr_ = {};
    }
    size_t nnz = 0;
    reader->read(&nnz, sizeof(size_t), 1);
    csc_cluster_ = io_align::read_padded<cluster_id_t>(reader, nnz);
    csc_value_ = io_align::read_padded<uint8_t>(
        reader, checked_mul(nnz, element_size_), element_size_);
}

void InvertedListClusters::mmap_deserialize(MmapCursor* cursor) {
    throw_if_null(cursor, "cursor must not be null");

    // Borrows each array where deserialize() copies it. read_array rejects a
    // misaligned start, so the padding serialize() wrote is what makes this
    // possible; borrow_padded pairs the skip with the read.
    const auto n_docs = cursor->read_scalar<size_t>();
    docs_ = io_align::borrow_padded<idx_t>(cursor, n_docs);
    const auto n_offsets = cursor->read_scalar<size_t>();
    offsets_ = io_align::borrow_padded<idx_t>(cursor, n_offsets);

    n_clusters_ = cursor->read_scalar<size_t>();
    element_size_ = cursor->read_scalar<size_t>();
    const auto n_terms = cursor->read_scalar<size_t>();
    if (n_terms > 0) {
        term_ids_ = io_align::borrow_padded<term_t>(cursor, n_terms);
        term_ptr_ = io_align::borrow_padded<idx_t>(cursor, n_terms + 1);
    } else {
        term_ids_ = {};
        term_ptr_ = {};
    }

    const auto nnz = cursor->read_scalar<size_t>();
    csc_cluster_ = io_align::borrow_padded<cluster_id_t>(cursor, nnz);
    // Bytes on the wire, reinterpreted at element_size on read, so the borrow
    // must start on that boundary rather than on 1.
    csc_value_ = io_align::borrow_padded<uint8_t>(
        cursor, checked_mul(nnz, element_size_), element_size_);
    if (nnz > 0 && reinterpret_cast<uintptr_t>(csc_value_.data()) %
                           element_size_ != 0) {
        throw std::runtime_error(
            "mmap: cluster summary values are misaligned for the element size");
    }
}

}  // namespace nsparse
