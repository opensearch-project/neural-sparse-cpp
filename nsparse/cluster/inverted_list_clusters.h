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

    void summarize(const SparseVectors* vectors, float alpha);

    size_t cluster_size() const { return n_clusters_; }

    void serialize(IOWriter* writer) const override;
    void deserialize(IOReader* reader) override;

    // Accumulate per-cluster summary scores for a query into `out` (resized to
    // the cluster count) using the term-major transpose. The query is given as
    // its sparse (term, value) pairs; `q_val_bytes` points at the query values
    // in the same element width as the stored summaries (float / uint16 /
    // uint8), matching how the dense path reinterprets the query buffer. A
    // query term is located by binary-searching term_ids_ (the distinct summary
    // terms, ascending), so no dimension-sized lookup table is retained and the
    // per-list footprint is proportional to the summaries' nnz.
    void score_summaries_transposed(const term_t* q_idx,
                                    const uint8_t* q_val_bytes, size_t q_len,
                                    std::vector<float>& out) const;

private:
    // Build the term-major (CSC) transpose from a per-cluster CSR summary.
    void build_transpose(const SparseVectors& summaries);
    template <class T>
    void score_summaries_typed(const term_t* q_idx, const T* q_val,
                               size_t q_len, std::vector<float>& out) const;

    std::vector<idx_t> docs_;
    std::vector<idx_t> offsets_;

    // Term-major transpose of the cluster summaries (replaces a CSR store). For
    // each distinct summary term term_ids_[i], entries
    // [term_ptr_[i], term_ptr_[i + 1]) in csc_cluster_/csc_value_ hold the
    // (cluster id, summary value) pairs for that term. Scoring iterates the
    // query's terms and scatter-adds q_val * summary_value into out[cluster].
    size_t n_clusters_ = 0;
    size_t element_size_ = U32;         // width of each csc_value_ entry
    std::vector<term_t> term_ids_;      // distinct summary terms, ascending
    std::vector<idx_t> term_ptr_;       // CSC offsets, size term_ids_+1
    std::vector<int32_t> csc_cluster_;  // cluster id per entry
    std::vector<uint8_t> csc_value_;    // summary value per entry (bytes)
};

}  // namespace nsparse

#endif  // INVERTED_LIST_CLUSTERS_H