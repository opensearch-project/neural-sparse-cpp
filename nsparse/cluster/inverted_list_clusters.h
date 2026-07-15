/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef INVERTED_LIST_CLUSTERS_H
#define INVERTED_LIST_CLUSTERS_H
#include <memory>
#include <span>
#include <vector>

#include "nsparse/io/io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/dense_vector_matrix.h"

namespace nsparse {

class InvertedListClusters : public Serializable {
public:
    InvertedListClusters() = default;
    InvertedListClusters(const std::vector<std::vector<idx_t>>& docs);
    // copy constructor
    InvertedListClusters(const InvertedListClusters& other);
    InvertedListClusters& operator=(const InvertedListClusters& other);
    // move constructor
    InvertedListClusters(InvertedListClusters&& other) noexcept = default;
    InvertedListClusters& operator=(InvertedListClusters&& other) noexcept =
        default;
    virtual ~InvertedListClusters() = default;

    auto get_docs(idx_t idx) const -> std::span<const idx_t>;

    auto summaries() const -> const SparseVectors& { return *summaries_; }

    void summarize(const SparseVectors* vectors, float alpha);

    size_t cluster_size() const {
        return summaries_ == nullptr ? 0 : summaries_->num_vectors();
    }

    void serialize(IOWriter* writer) const override;
    void deserialize(IOReader* reader) override;

    // Build (once) a term-major transpose of the summaries so per-query summary
    // scoring can be driven by the sparse query's active terms instead of
    // gathering dense[term] for every summary term. For each distinct summary
    // term t, it stores the list of (cluster_id, value) pairs. Scoring: for
    // each query term (t, qv), add qv*value to scores[cluster_id]. Avoids the
    // dense-buffer gather entirely.
    void build_transpose() const;
    // Accumulate per-cluster summary scores for a query into `out` (sized to
    // the cluster count) using the transpose built by build_transpose().
    void score_summaries_transposed(const term_t* q_idx, const float* q_val,
                                    size_t q_len,
                                    std::vector<float>& out) const;

private:
    std::vector<idx_t> docs_;
    std::vector<idx_t> offsets_;
    std::unique_ptr<SparseVectors> summaries_;

    // Transposed (term-major) summary storage, built lazily. A query term is
    // located by binary-searching term_ids_ (the distinct summary terms, in
    // ascending order); no dimension-sized lookup table is retained, so the
    // per-list footprint is proportional to the summaries' nnz, not to the
    // vocabulary dimension.
    mutable bool transpose_built_ = false;
    // Distinct terms present across all summaries, ascending. term_ids_[i] owns
    // csc entries [term_ptr_[i], term_ptr_[i + 1]).
    mutable std::vector<term_t> term_ids_;
    mutable std::vector<idx_t> term_ptr_;
    mutable std::vector<int32_t> csc_cluster_;  // cluster id per entry
    mutable std::vector<float> csc_value_;      // summary value per entry
};

}  // namespace nsparse

#endif  // INVERTED_LIST_CLUSTERS_H