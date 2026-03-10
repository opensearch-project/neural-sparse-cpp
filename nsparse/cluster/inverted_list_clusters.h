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

private:
    std::vector<idx_t> docs_;
    std::vector<idx_t> offsets_;
    std::unique_ptr<SparseVectors> summaries_;
};

}  // namespace nsparse

#endif  // INVERTED_LIST_CLUSTERS_H