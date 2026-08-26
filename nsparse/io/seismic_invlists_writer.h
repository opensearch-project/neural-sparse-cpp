/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef SEISMIC_INVLISTS_WRITER_H
#define SEISMIC_INVLISTS_WRITER_H
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/io/index_io.h"
#include "nsparse/io/mmap_io.h"

namespace nsparse {

// Serializes an index's posting lists, and deserializes them into its own
// storage for the caller to release().
//
// Writing borrows the caller's lists rather than copying them: InvertedListClusters
// is move-only now that its arrays are Buf, and a copy of every posting list was
// the largest allocation in write_index.
class SeismicInvertedListsWriter : public MmapSerializable {
public:
    // For writing. `clustered_inverted_lists` must outlive this writer.
    // `summaries_only` writes each list's doc-id membership empty (see
    // InvertedListClusters::serialize_summaries_only).
    explicit SeismicInvertedListsWriter(
        const std::vector<InvertedListClusters>& clustered_inverted_lists,
        bool summaries_only = false)
        : borrowed_(&clustered_inverted_lists),
          summaries_only_(summaries_only) {}

    // For reading; deserialize() fills the internal store.
    SeismicInvertedListsWriter() = default;

    void serialize(IOWriter* writer) const override {
        const auto& lists = borrowed_ != nullptr ? *borrowed_ : owned_;
        size_t size = lists.size();
        writer->write(&size, sizeof(size), 1);
        for (const auto& clusters : lists) {
            if (summaries_only_) {
                clusters.serialize_summaries_only(writer);
            } else {
                clusters.serialize(writer);
            }
        }
    }
    void deserialize(IOReader* reader) override {
        size_t size = 0;
        reader->read(&size, sizeof(size), 1);
        owned_ = std::vector<InvertedListClusters>(size);
        for (auto& clusters : owned_) {
            clusters.deserialize(reader);
        }
    }

    // Same walk as deserialize(), with each list borrowing from the mapping
    // instead of copying. The mapping must outlive whatever release() hands out.
    void mmap_deserialize(MmapCursor* cursor) override {
        const auto size = cursor->read_scalar<size_t>();
        owned_ = std::vector<InvertedListClusters>(size);
        for (auto& clusters : owned_) {
            clusters.mmap_deserialize(cursor);
        }
    }

    std::vector<InvertedListClusters>&& release() { return std::move(owned_); }

private:
    // Exactly one is in play: borrowed_ when constructed for writing, owned_
    // when default-constructed and filled by deserialize().
    const std::vector<InvertedListClusters>* borrowed_ = nullptr;
    std::vector<InvertedListClusters> owned_;
    // Write-side only: omit the doc-id membership (reader is agnostic).
    bool summaries_only_ = false;
};
}  // namespace nsparse

#endif  // SEISMIC_INVLISTS_WRITER_H