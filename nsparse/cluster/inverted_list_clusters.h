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

#include "nsparse/io/mmap_io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/buf.h"

namespace nsparse {

class InvertedListClusters : public MmapSerializable {
public:
    InvertedListClusters() = default;
    InvertedListClusters(const std::vector<std::vector<idx_t>>& docs);
    // Move-only, because Buf is: a Buf may borrow memory it does not own, and
    // copying one would either alias the lender or silently deep-copy.
    InvertedListClusters(const InvertedListClusters& other) = delete;
    InvertedListClusters& operator=(const InvertedListClusters& other) = delete;
    InvertedListClusters(InvertedListClusters&& other) noexcept = default;
    InvertedListClusters& operator=(InvertedListClusters&& other) noexcept =
        default;
    virtual ~InvertedListClusters() = default;

    auto get_docs(idx_t idx) const -> std::span<const idx_t>;

    void summarize(const SparseVectors* vectors, float alpha);

    size_t cluster_size() const { return n_clusters_; }

    // Width (bytes) of each stored summary value: the dispatch
    // score_summaries_transposed uses, so a reader can check it against the
    // width its query is encoded at.
    [[nodiscard]] size_t element_size() const { return element_size_; }

    void serialize(IOWriter* writer) const override;
    // Like serialize() but emits the doc-id membership empty, for indexes that
    // keep it elsewhere (DiskSeismic's inline forward index). Read back by the
    // regular deserialize()/mmap_deserialize().
    void serialize_summaries_only(IOWriter* writer) const;
    void deserialize(IOReader* reader) override;
    void mmap_deserialize(MmapCursor* cursor) override;

    // Accumulate per-cluster summary scores for the query's sparse (term,
    // value) pairs into `out`, resized to the cluster count. `q_val_bytes` must
    // use the same element width as the stored summaries (float / uint16 /
    // uint8), matching how the dense path reinterprets the query buffer. Terms
    // are located by binary search over term_ids_, so the per-list footprint
    // stays proportional to the summaries' nnz rather than the dimension.
    void score_summaries_transposed(const term_t* q_idx,
                                    const uint8_t* q_val_bytes, size_t q_len,
                                    std::vector<float>& out) const;

private:
    // Writes the transposed (CSC) summary store — the tail shared by serialize()
    // and serialize_summaries_only().
    void serialize_summary_store(IOWriter* writer) const;

    // Build the term-major (CSC) transpose from a per-cluster CSR summary. The
    // CSR summary is transient; only the transpose is retained.
    void build_transpose(const SparseVectors& summaries);
    template <class T>
    void score_summaries_typed(const term_t* q_idx, const T* q_val,
                               size_t q_len, std::vector<float>& out) const;

    // Cluster ids within a posting list are bounded by beta (clusters per
    // list), which stays far below 2^16 for any workable configuration — the
    // canonical Rust seismic likewise caps the summary count at 2^16. 16 bits
    // therefore halve csc_cluster_ at no recall cost (build_transpose asserts
    // the bound holds).
    using cluster_id_t = uint16_t;

    // Term-major transpose of the cluster summaries. For each distinct summary
    // term term_ids_[i], entries [term_ptr_[i], term_ptr_[i + 1]) of
    // csc_cluster_/csc_value_ hold that term's (cluster id, summary value)
    // pairs.
    size_t n_clusters_ = 0;
    size_t element_size_ = U32;        // width of each csc_value_ entry

    Buf<idx_t> docs_;
    Buf<idx_t> offsets_;
    Buf<term_t> term_ids_;             // distinct summary terms, ascending
    Buf<idx_t> term_ptr_;              // CSC offsets, size term_ids_+1
    Buf<cluster_id_t> csc_cluster_;    // cluster id per entry
    Buf<uint8_t> csc_value_;           // summary value per entry (bytes)
};

}  // namespace nsparse

#endif  // INVERTED_LIST_CLUSTERS_H