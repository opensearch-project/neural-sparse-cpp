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
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/index.h"
#include "nsparse/io/io.h"
#include "nsparse/mmap_index.h"
#include "nsparse/seismic_index.h"
#include "nsparse/types.h"
#include "nsparse/utils/scalar_quantizer.h"

namespace nsparse {

struct SeismicSQSearchParameters : public SeismicSearchParameters {
    float vmin;
    float vmax;
    SeismicSQSearchParameters(float vmin, float vmax, int cut,
                              float heap_factor)
        : SeismicSearchParameters(cut, heap_factor), vmax(vmax), vmin(vmin) {}
};

class SeismicScalarQuantizedIndex : public MmapIndex, public IndexIO {
public:
    static constexpr std::array<char, 4> name = {'S', 'E', 'S', 'Q'};
    // Bump whenever write_index's payload layout changes.
    static constexpr uint32_t kFormatVersion = 1;

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

    const ScalarQuantizer& get_scalar_quantizer() const { return sq_; }

    // Borrows a serialized index from a file mapping instead of copying it onto
    // the heap; see SeismicIndex::mmap_index, which this mirrors past the
    // quantizer header. `pos` is where write_index's payload begins.
    static SeismicScalarQuantizedIndex* mmap_index(const IndexHeader& header,
                                                   const char* index_file,
                                                   size_t pos);

    // Only the copying residency. A mapped CSR is borrowed at the width it was
    // written in, which is float, whereas this index searches over codes: the
    // values have to pass through add() to be quantized.
    void read_csr(const char* file_path,
                  Residency residency = Residency::kInMemory) override;

private:
    // interfaces of IndexIO
    [[nodiscard]] uint32_t format_version() const override {
        return kFormatVersion;
    }
    void write_index(IOWriter* io_writer) override;
    void read_index(IOReader* io_reader, const IndexHeader& header,
                    int io_flags = 0) override;
    // The quantizer parameters that open this index's payload -- distinct from
    // the IndexHeader the file itself starts with.
    void write_quantization_header(IOWriter* io_writer);
    void read_quantization_header(IOReader* io_reader);

    // Null `search_parameters` searches with the defaults, as it does for
    // SeismicIndex and as the base signature's default argument implies. This
    // index used to reject it, which made that default argument -- and the
    // bindings' params=None -- unusable here alone.
    auto search(idx_t n, const idx_t* indptr, const term_t* indices,
                const float* values, int k,
                SearchParameters* search_parameters = nullptr)
        -> pair_of_score_id_vectors_t override;
    ScalarQuantizer query_quantizer(
        const SearchParameters* search_parameters) const;
    // `dense` and `visited` are per-thread scratch reused across the queries a
    // thread handles (see search()). `dense` (a dimension-sized quantized-code
    // buffer, element_size bytes per dim) must be all-zero on entry and is
    // restored to all-zero on exit via a sparse clear over the query's own dims
    // (q_idx/q_len); `visited` is cleared on entry.
    auto single_query(std::vector<uint8_t>& dense,
                      absl::flat_hash_set<idx_t>& visited, const term_t* q_idx,
                      const uint8_t* q_val_bytes, size_t q_len,
                      size_t element_size, const std::vector<term_t>& cuts,
                      int k, float heap_factor, const ScalarQuantizer& query_sq,
                      SearchParameters* search_parameters)
        -> pair_of_score_id_vector_t;
    ScalarQuantizer sq_;
    SeismicClusterParameters cluster_parameter_;

protected:
    std::vector<InvertedListClusters> clustered_inverted_lists;
};
}  // namespace nsparse

#endif  // SEISMIC_SCALAR_QUANTIZED_INDEX_H