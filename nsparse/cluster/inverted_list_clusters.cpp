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
#include <span>
#include <unordered_map>
#include <vector>

#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse {
namespace {

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
    for (size_t i = 0; i < offsets.size() - 1; ++i) {
        size_t n_docs = offsets[i + 1] - offsets[i];
        std::unordered_map<term_t, T> summary_map;
        float sum = 0.0F;
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

        // Convert summary_map to vector of pairs
        std::vector<std::pair<term_t, T>> summary_vec(summary_map.begin(),
                                                      summary_map.end());

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
    build_transpose();
}

void InvertedListClusters::build_transpose() const {
    if (transpose_built_) return;
    transpose_built_ = true;
    if (summaries_ == nullptr || summaries_->num_vectors() == 0) return;
    const size_t n_clusters = summaries_->num_vectors();
    const auto* indptr = summaries_->indptr_data();
    const auto* indices = summaries_->indices_data();
    const auto* values = summaries_->values_data_float();
    const size_t nnz = static_cast<size_t>(indptr[n_clusters]);

    // Distinct terms present across all summaries, ascending. Collected by
    // sort+unique over the summary indices so no dimension-sized scratch is
    // needed (the working set stays proportional to nnz).
    term_ids_.assign(indices, indices + nnz);
    std::ranges::sort(term_ids_);
    term_ids_.erase(std::ranges::unique(term_ids_).begin(), term_ids_.end());
    const size_t n_terms = term_ids_.size();

    // Maps a term to its index in term_ids_ (its CSC column).
    auto term_column = [this](term_t term) -> size_t {
        return static_cast<size_t>(std::ranges::lower_bound(term_ids_, term) -
                                   term_ids_.begin());
    };

    // Count entries per term, then prefix-sum into CSC offsets term_ptr_.
    term_ptr_.assign(n_terms + 1, 0);
    for (size_t j = 0; j < nnz; ++j) {
        term_ptr_[term_column(indices[j]) + 1]++;
    }
    for (size_t t = 0; t < n_terms; ++t) term_ptr_[t + 1] += term_ptr_[t];

    csc_cluster_.resize(nnz);
    csc_value_.resize(nnz);
    std::vector<idx_t> cursor(term_ptr_.begin(), term_ptr_.end() - 1);
    for (size_t cluster = 0; cluster < n_clusters; ++cluster) {
        const idx_t start = indptr[cluster];
        const idx_t end = indptr[cluster + 1];
        for (idx_t j = start; j < end; ++j) {
            const size_t col = term_column(indices[j]);
            const idx_t pos = cursor[col]++;
            csc_cluster_[pos] = static_cast<int32_t>(cluster);
            csc_value_[pos] = values[j];
        }
    }
}

void InvertedListClusters::score_summaries_transposed(
    const term_t* q_idx, const float* q_val, size_t q_len,
    std::vector<float>& out) const {
    const size_t n_clusters = cluster_size();
    out.assign(n_clusters, 0.0F);
    if (!transpose_built_ || term_ids_.empty()) return;
    for (size_t i = 0; i < q_len; ++i) {
        const term_t term = q_idx[i];
        // Locate the query term among the summaries' distinct terms. Absent
        // terms (including any out of the summaries' range) contribute nothing.
        auto it = std::ranges::lower_bound(term_ids_, term);
        if (it == term_ids_.end() || *it != term) {
            continue;
        }
        const size_t col = static_cast<size_t>(it - term_ids_.begin());
        const float qv = q_val[i];
        const idx_t start = term_ptr_[col];
        const idx_t end = term_ptr_[col + 1];
        for (idx_t j = start; j < end; ++j) {
            out[csc_cluster_[j]] += qv * csc_value_[j];
        }
    }
}

}  // namespace nsparse