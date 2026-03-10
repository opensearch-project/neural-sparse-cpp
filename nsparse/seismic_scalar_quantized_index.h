/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef SEISMIC_SCALAR_QUANTIZED_INDEX_H
#define SEISMIC_SCALAR_QUANTIZED_INDEX_H

#include <array>
#include <memory>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/index.h"
#include "nsparse/seismic_index.h"
#include "nsparse/utils/scalar_quantizer.h"

namespace nsparse {

struct SeismicSQSearchParameters : public SeismicSearchParameters {
    float vmin;
    float vmax;
    SeismicSQSearchParameters(float vmin, float vmax, int cut,
                              float heap_factor)
        : SeismicSearchParameters(cut, heap_factor), vmax(vmax), vmin(vmin) {}
};

class SeismicScalarQuantizedIndex : public Index, public IndexIO {
    friend void write_index(Index* index, char* filename);
    friend Index* read_index(char* filename);

public:
    static constexpr std::array<char, 4> name = {'S', 'E', 'S', 'Q'};
    explicit SeismicScalarQuantizedIndex(int dim);
    SeismicScalarQuantizedIndex(QuantizerType quantizer_type, float vmin,
                                float vmax, SeismicClusterParameters parameter,
                                int dim);
    ~SeismicScalarQuantizedIndex() override = default;

    SeismicScalarQuantizedIndex(const SeismicScalarQuantizedIndex&) = delete;
    SeismicScalarQuantizedIndex& operator=(const SeismicScalarQuantizedIndex&) =
        delete;
    std::array<char, 4> id() const override { return name; }
    void add(idx_t n, const idx_t* indptr, const term_t* indices,
             const float* values) override;
    void build() override;
    const SparseVectors* get_vectors() const override { return vectors_.get(); }

    const ScalarQuantizer& get_scalar_quantizer() const { return sq_; }

private:
    // interfaces of IndexIO
    void write_index(IOWriter* io_writer) override;
    void read_index(IOReader* io_reader) override;
    void write_header(IOWriter* io_writer);
    void read_header(IOReader* io_reader);

    auto search(idx_t n, const idx_t* indptr, const term_t* indices,
                const float* values, int k,
                SearchParameters* search_parameters = nullptr)
        -> pair_of_score_id_vectors_t override;
    auto encode(const float* values, size_t nnz,
                SearchParameters* search_parameters) -> std::vector<uint8_t>;
    auto single_query(const std::vector<uint8_t>& dense,
                      const std::vector<term_t>& cuts, int k, float heap_factor,
                      const ScalarQuantizer& query_sq,
                      SearchParameters* search_parameters)
        -> pair_of_score_id_vector_t;
    ScalarQuantizer sq_;
    std::unique_ptr<SparseVectors> vectors_;
    SeismicClusterParameters cluster_parameter_;

protected:
    std::vector<InvertedListClusters> clustered_inverted_lists;
};
}  // namespace nsparse

#endif  // SEISMIC_SCALAR_QUANTIZED_INDEX_H