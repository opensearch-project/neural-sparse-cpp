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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#ifdef NSPARSE_WITH_CUDA
#include "nsparse/gpu/gpu_cluster_assigner.h"
#endif

namespace nsparse {
namespace {

// Opt-in split timing for summarize (NSPARSE_SUMMARIZE_PROFILE=1): how much of
// summarize is the memory-bound max-pool vs the per-cluster sort/truncate.
struct SummarizeProfile {
    std::atomic<int64_t> maxpool_ns{0};
    std::atomic<int64_t> sort_ns{0};
    bool enabled = false;
    SummarizeProfile() {
        const char* v = std::getenv("NSPARSE_SUMMARIZE_PROFILE");
        enabled = (v != nullptr && v[0] == '1');
    }
    ~SummarizeProfile() {
        if (!enabled) return;
        std::fprintf(stderr,
                     "[nsparse summarize] maxpool=%.1fs sort_truncate=%.1fs\n",
                     maxpool_ns.load() / 1e9, sort_ns.load() / 1e9);
    }
};
SummarizeProfile g_summ_profile;

inline int64_t summ_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/**
 * @brief Generage summary sparse vector for posting lists
 *
 * @param vectors inverted index
 * @param group_of_doc_ids a list of posting list
 * @param alpha prune ratio
 * @return SparseVectors
 */
template <class T>
SparseVectors summarize_(const SparseVectors* vectors,
                         const std::vector<idx_t>& group_of_doc_ids,
                         const std::vector<idx_t>& offsets, float alpha) {
    SparseVectors summarized_vectors(
        {.element_size = vectors->get_element_size(),
         .dimension = vectors->get_dimension()});
    if (offsets.size() <= 1) {
        return summarized_vectors;
    }
    const auto element_size = vectors->get_element_size();
    const auto& indptr_data = vectors->indptr_data();
    const auto& indices_data = vectors->indices_data();
    const auto& values_data = vectors->values_data();
    const bool prof = g_summ_profile.enabled;
    int64_t maxpool_acc = 0;
    int64_t sort_acc = 0;
    // The per-term max-pool (the memory-bound majority of summarize) can be
    // offloaded to the GPU for float (U32) weights, batched as ONE launch for
    // the whole list's clusters (per-cluster launches are dominated by GPU
    // round-trip overhead). The tie-order-sensitive sort/truncate below stays
    // on the CPU so output ordering is preserved.
    bool gpu_ok = false;
#ifdef NSPARSE_WITH_CUDA
    std::vector<detail::GpuClusterAssigner::ClusterSummary> gpu_clusters;
    if constexpr (std::is_same_v<T, float>) {
        if (detail::should_offload_summarize_to_gpu()) {
            gpu_ok = detail::GpuClusterAssigner::instance()
                         .summarize_list_maxpool(vectors,
                                                 group_of_doc_ids.data(),
                                                 offsets.data(),
                                                 offsets.size() - 1,
                                                 gpu_clusters);
        }
    }
#endif

    for (size_t i = 0; i < offsets.size() - 1; ++i) {
        const int64_t ts0 = prof ? summ_now_ns() : 0;
        size_t n_docs = offsets[i + 1] - offsets[i];
        std::vector<std::pair<term_t, T>> summary_vec;
        float sum = 0.0F;

        bool used_gpu = false;
#ifdef NSPARSE_WITH_CUDA
        if constexpr (std::is_same_v<T, float>) {
            if (gpu_ok) {
                const auto& gc = gpu_clusters[i];
                summary_vec.reserve(gc.terms.size());
                for (size_t t = 0; t < gc.terms.size(); ++t) {
                    summary_vec.emplace_back(gc.terms[t], gc.values[t]);
                }
                sum = gc.sum;
                used_gpu = true;
            }
        }
#endif
        if (!used_gpu) {
            std::unordered_map<term_t, T> summary_map;
            auto doc_ids = std::span<const idx_t>(
                group_of_doc_ids.data() + offsets[i], n_docs);
            for (const auto& doc_id : doc_ids) {
                int start = indptr_data[doc_id];
                int end = indptr_data[doc_id + 1];
                for (size_t j = start; j < end; ++j) {
                    const auto old = summary_map[indices_data[j]];
                    auto& value = summary_map[indices_data[j]];
                    // j is element index, need byte offset for T access
                    value = std::max(value, *reinterpret_cast<const T*>(
                                                values_data + j * sizeof(T)));
                    sum += value - old;
                }
            }
            summary_vec.assign(summary_map.begin(), summary_map.end());
        }

        const int64_t ts1 = prof ? summ_now_ns() : 0;

        // Sort by value in descending order
        std::ranges::sort(summary_vec, [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        float addup = 0.0F;
        for (int j = 0; j < summary_vec.size(); ++j) {
            addup += summary_vec[j].second;
            if (addup / sum >= alpha) {
                summary_vec.erase(summary_vec.begin() + j + 1,
                                  summary_vec.end());
                break;
            }
        }

        // Sort by term_t order
        std::ranges::sort(summary_vec, [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        // Break into separate terms and values vectors
        std::vector<term_t> terms;
        std::vector<T> values;
        terms.reserve(summary_vec.size());
        values.reserve(summary_vec.size());
        for (const auto& [term, value] : summary_vec) {
            terms.push_back(term);
            values.push_back(value);
        }

        summarized_vectors.add_vector(
            terms.data(), terms.size(),
            reinterpret_cast<const uint8_t*>(values.data()),
            values.size() * sizeof(T));

        if (prof) {
            const int64_t ts2 = summ_now_ns();
            maxpool_acc += ts1 - ts0;
            sort_acc += ts2 - ts1;
        }
    }
    if (prof) {
        g_summ_profile.maxpool_ns.fetch_add(maxpool_acc,
                                            std::memory_order_relaxed);
        g_summ_profile.sort_ns.fetch_add(sort_acc, std::memory_order_relaxed);
    }
    return summarized_vectors;
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