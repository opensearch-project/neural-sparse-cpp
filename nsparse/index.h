/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef INDEX_H
#define INDEX_H

#include <array>

#include "nsparse/id_selector.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse {

struct SearchParameters {
    virtual ~SearchParameters() = default;
    const IDSelector* get_id_selector() const { return id_selector; }
    void set_id_selector(IDSelector* selector) { id_selector = selector; }

private:
    IDSelector* id_selector = nullptr;
};

class Index {
public:
    explicit Index(int dim = 0);
    virtual ~Index() = default;
    virtual std::array<char, 4> id() const = 0;
    virtual void add(idx_t n, const idx_t* indptr, const term_t* indices,
                     const float* values) = 0;
    virtual void build();
    virtual void search(
        idx_t n, const idx_t* indptr, const term_t* indices,
        const float* values, int k, float* distances, idx_t* labels,
        SearchParameters* search_parameters = nullptr);  // Pre-allocated: n * k
    virtual const SparseVectors* get_vectors() const { return nullptr; };

    int get_dimension() const { return dimension_; }
    size_t num_vectors() const {
        const auto* vectors = get_vectors();
        return vectors == nullptr ? 0 : vectors->num_vectors();
    }
    virtual void add_with_ids(idx_t n, const idx_t* indptr,
                              const term_t* indices, const float* values,
                              const idx_t* ids);

protected:
    virtual auto search(idx_t n, const idx_t* indptr, const term_t* indices,
                        const float* values, int k,
                        SearchParameters* search_parameters = nullptr)
        -> pair_of_score_id_vectors_t;

    int dimension_;
};

}  // namespace nsparse

#endif  // INDEX_H