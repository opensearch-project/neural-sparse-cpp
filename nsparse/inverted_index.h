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

#include "nsparse/index.h"
#include "nsparse/invlists/inverted_lists.h"
#include "nsparse/io/io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse {

class InvertedIndex : public Index, public IndexIO {
public:
    explicit InvertedIndex(int dim);

    InvertedIndex(const InvertedIndex&) = delete;
    InvertedIndex& operator=(const InvertedIndex&) = delete;
    void add(idx_t n, const idx_t* indptr, const term_t* indices,
             const float* values) override;
    void build() override;
    std::array<char, 4> id() const override { return name; }
    static constexpr std::array<char, 4> name = {'I', 'N', 'V', 'T'};

protected:
    auto search(idx_t n, const idx_t* indptr, const term_t* indices,
                const float* values, int k,
                SearchParameters* search_parameters = nullptr)
        -> pair_of_score_id_vectors_t override;

private:
    // IndexIO overrides
    void write_index(IOWriter* io_writer) override;
    void read_index(IOReader* io_reader) override;

    auto single_query(const term_t* indices, const float* values, int size,
                      int k) -> pair_of_score_id_vector_t;
    std::unique_ptr<ArrayInvertedLists> inverted_lists_;
    std::unique_ptr<SparseVectors> vectors_;
    // Per-term max posting value, computed at build() time.
    // max_term_scores_[term_id] = max value in that term's posting list.
    std::vector<float> max_term_scores_;
};

}  // namespace nsparse
#endif  // INVERTED_INDEX_H