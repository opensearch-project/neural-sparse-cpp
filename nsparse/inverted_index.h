/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef INVERTED_INDEX_H
#define INVERTED_INDEX_H

#include <array>
#include <memory>
#include <vector>

#include "nsparse/invlists/inverted_lists.h"
#include "nsparse/io/io.h"
#include "nsparse/mmap_index.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/buf.h"

namespace nsparse {

class InvertedIndex : public MmapIndex, public IndexIO {
public:
    explicit InvertedIndex(int dim);

    InvertedIndex(const InvertedIndex&) = delete;
    InvertedIndex& operator=(const InvertedIndex&) = delete;
    void add(idx_t n, const idx_t* indptr, const term_t* indices,
             const float* values) override;
    void build() override;
    size_t num_vectors() const override { return num_vectors_; }
    std::array<char, 4> id() const override { return name; }
    static constexpr std::array<char, 4> name = {'I', 'N', 'V', 'T'};

    // Reads what write_index wrote, with the posting lists borrowing from a
    // mapping of `index_file` instead of being copied onto the heap. `pos` is
    // where the payload begins, past the header read_header consumed.
    static InvertedIndex* mmap_index(int dimension, const char* index_file,
                                     size_t pos);

protected:
    auto search(idx_t n, const idx_t* indptr, const term_t* indices,
                const float* values, int k,
                SearchParameters* search_parameters = nullptr)
        -> pair_of_score_id_vectors_t override;

private:
    // IndexIO overrides
    void write_index(IOWriter* io_writer) override;
    void read_index(IOReader* io_reader, int io_flags) override;

    // `id_selector` may be null; when set, only member docs are returned.
    auto single_query(const term_t* indices, const float* values, int size,
                      int k, const IDSelector* id_selector)
        -> pair_of_score_id_vector_t;
    std::unique_ptr<ArrayInvertedLists> inverted_lists_;
    // Tracked separately, since build() releases the vectors the base holds,
    // and serialised explicitly, since the posting lists hold no entry for a
    // document whose terms were all pruned.
    size_t num_vectors_ = 0;
    // Per-term max posting value, computed at build() time.
    // max_term_scores_[term_id] = max value in that term's posting list.
    // A Buf rather than a vector so a mapped index can borrow it in place.
    Buf<float> max_term_scores_;
};

}  // namespace nsparse
#endif  // INVERTED_INDEX_H
