/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef SEISMIC_INDEX_H
#define SEISMIC_INDEX_H
#include <array>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/index.h"
#include "nsparse/io/io.h"
#include "nsparse/seismic_common.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace nsparse {

struct SeismicSearchParameters : public SearchParameters {
    int cut = 10;
    float heap_factor = 1.0F;
    SeismicSearchParameters(int cut, float heap_factor)
        : cut(cut), heap_factor(heap_factor) {}
    SeismicSearchParameters() = default;
};

class SeismicIndex : public Index, public IndexIO {
    friend void write_index(Index* index, char* filename);
    friend Index* read_index(char* filename);

public:
    static constexpr std::array<char, 4> name = {'S', 'E', 'I', 'S'};

    explicit SeismicIndex(int dim);
    SeismicIndex(int dim, SeismicClusterParameters parameter);
    ~SeismicIndex() override = default;
    std::array<char, 4> id() const override { return name; }

    SeismicIndex(const SeismicIndex&) = delete;
    SeismicIndex& operator=(const SeismicIndex&) = delete;

    const SparseVectors* get_vectors() const override;
    void build() override;

    void add(idx_t n, const idx_t* indptr, const term_t* indices,
             const float* values) override;

protected:
    std::vector<InvertedListClusters> clustered_inverted_lists;

private:
    // override of IndexIO
    void write_index(IOWriter* io_writer) override;
    void read_index(IOReader* io_reader) override;

    auto search(idx_t n, const idx_t* indptr, const term_t* indices,
                const float* values, int k,
                SearchParameters* search_parameters = nullptr)
        -> pair_of_score_id_vectors_t override;

    auto single_query(const std::vector<float>& dense,
                      const std::vector<term_t>& cuts, int k, float heap_factor,
                      SearchParameters* search_parameters)
        -> pair_of_score_id_vector_t;
    std::unique_ptr<SparseVectors> vectors_;
    SeismicClusterParameters cluster_parameter_;
};
}  // namespace nsparse

#endif  // SEISMIC_INDEX_H