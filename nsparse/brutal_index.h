/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef BRUTAL_INDEX_H
#define BRUTAL_INDEX_H

#include <array>
#include <memory>
#include <vector>

#include "nsparse/index.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
namespace nsparse {

class BrutalIndex : public Index {
public:
    explicit BrutalIndex(int dim = 0);

    BrutalIndex(const BrutalIndex&) = delete;
    BrutalIndex& operator=(const BrutalIndex&) = delete;
    void add(idx_t n, const idx_t* indptr, const term_t* indices,
             const float* values) override;

    std::array<char, 4> id() const override { return name; }
    static constexpr std::array<char, 4> name = {'B', 'R', 'U', 'T'};

protected:
    auto search(idx_t n, const idx_t* indptr, const term_t* indices,
                const float* values, int k,
                SearchParameters* search_parameters = nullptr)
        -> pair_of_score_id_vectors_t override;
    const SparseVectors* get_vectors() const override;

private:
    auto single_query(const std::vector<float>& dense, int k)
        -> pair_of_score_id_vector_t;
    std::unique_ptr<SparseVectors> vectors_;
};

}  // namespace nsparse
#endif  // BRUTAL_INDEX_H