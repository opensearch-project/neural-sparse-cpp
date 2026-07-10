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
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
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
                                  const std::vector<idx_t>& group_of_doc_ids,
                                  const std::vector<idx_t>& offsets,
                                  float alpha) {
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
    const SparseVectors* vectors, const std::vector<idx_t>& group_of_doc_ids,
    const std::vector<idx_t>& offsets, float alpha) {
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
 * @brief Generate the summary sparse vectors for a list's posting-list clusters.
 *
 * @param vectors inverted index
 * @param group_of_doc_ids flattened doc ids of all clusters
 * @param offsets cluster boundaries into group_of_doc_ids
 * @param alpha prune ratio
 */
template <class T>
SparseVectors summarize_(const SparseVectors* vectors,
                         const std::vector<idx_t>& group_of_doc_ids,
                         const std::vector<idx_t>& offsets, float alpha) {
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
    offsets_.reserve(docs.size() + 1);
    offsets_.push_back(0);
    for (const auto& doc_ids : docs) {
        docs_.insert(docs_.end(), doc_ids.begin(), doc_ids.end());
        offsets_.push_back(docs_.size());
    }
}

InvertedListClusters::InvertedListClusters(const InvertedListClusters& other) {
    docs_ = other.docs_;
    offsets_ = other.offsets_;
    if (other.summaries_ != nullptr) {
        summaries_ = std::make_unique<SparseVectors>(*other.summaries_);
    } else {
        summaries_.reset();
    }
}
InvertedListClusters& InvertedListClusters::operator=(
    const InvertedListClusters& other) {
    if (this != &other) {
        docs_ = other.docs_;
        offsets_ = other.offsets_;
        if (other.summaries_ != nullptr) {
            summaries_ = std::make_unique<SparseVectors>(*other.summaries_);
        } else {
            summaries_.reset();
        }
    }
    return *this;
}

auto InvertedListClusters::get_docs(idx_t idx) const -> std::span<const idx_t> {
    return {docs_.data() + offsets_[idx],
            static_cast<size_t>(offsets_[idx + 1] - offsets_[idx])};
}

void InvertedListClusters::summarize(const SparseVectors* vectors,
                                     float alpha) {
    if (summaries_ != nullptr) {
        summaries_.reset();
    }
    const auto element_size = vectors->get_element_size();
    if (element_size == U32) {
        summaries_ = std::make_unique<SparseVectors>(
            std::move(summarize_<float>(vectors, docs_, offsets_, alpha)));
    } else if (element_size == U16) {
        summaries_ = std::make_unique<SparseVectors>(
            std::move(summarize_<uint16_t>(vectors, docs_, offsets_, alpha)));
    } else {
        summaries_ = std::make_unique<SparseVectors>(
            std::move(summarize_<uint8_t>(vectors, docs_, offsets_, alpha)));
    }
}

void InvertedListClusters::serialize(IOWriter* writer) const {
    size_t n_docs = docs_.size();
    writer->write(&n_docs, sizeof(size_t), 1);
    if (n_docs > 0) {
        writer->write(const_cast<idx_t*>(docs_.data()), sizeof(idx_t), n_docs);
    }
    size_t n_offsets = offsets_.size();
    writer->write(&n_offsets, sizeof(size_t), 1);
    if (n_offsets > 0) {
        writer->write(const_cast<idx_t*>(offsets_.data()), sizeof(idx_t),
                      n_offsets);
    }
    if (summaries_ == nullptr) {
        empty_sparse_vectors.serialize(writer);
    } else {
        summaries_->serialize(writer);
    }
}

void InvertedListClusters::deserialize(IOReader* reader) {
    size_t n_docs = 0;
    reader->read(&n_docs, sizeof(size_t), 1);
    if (n_docs > 0) {
        docs_.resize(n_docs);
        reader->read(docs_.data(), sizeof(idx_t), n_docs);
    }
    size_t n_offsets = 0;
    reader->read(&n_offsets, sizeof(size_t), 1);
    if (n_offsets > 0) {
        offsets_.resize(n_offsets);
        reader->read(offsets_.data(), sizeof(idx_t), n_offsets);
    }
    summaries_ = std::make_unique<SparseVectors>();
    summaries_->deserialize(reader);
    if (summaries_->num_vectors() == 0) {
        summaries_.reset();
    }
}

}  // namespace nsparse