/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef INLINE_FORWARD_INDEX_WRITER_H
#define INLINE_FORWARD_INDEX_WRITER_H

#include <cstdint>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/inline_forward_index.h"
#include "nsparse/io/io.h"
#include "nsparse/sparse_vectors.h"

namespace nsparse {

// Build-time writer for the inline forward-index file
class InlineForwardIndexWriter {
public:
    static constexpr uint64_t kDefaultPageSize = 4096;  // common OS page size

    // page_size: block alignment when page-aligned (power of two, >= header
    // size); ignored when packed.
    explicit InlineForwardIndexWriter(
        uint64_t page_size = kDefaultPageSize,
        InlineLayout layout = InlineLayout::kPageAligned);

    void write(const std::vector<InvertedListClusters>& lists,
               const SparseVectors& vectors, IOWriter* writer) const;

    uint64_t page_size() const { return page_size_; }
    InlineLayout layout() const { return layout_; }

    // page_size when page-aligned, else 1 (packed).
    uint64_t alignment() const {
        return layout_ == InlineLayout::kPageAligned ? page_size_ : 1;
    }

private:
    uint64_t page_size_;
    InlineLayout layout_;
};

}  // namespace nsparse

#endif  // INLINE_FORWARD_INDEX_WRITER_H
