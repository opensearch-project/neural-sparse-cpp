/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef DISK_SEISMIC_SCALAR_QUANTIZED_INDEX_H
#define DISK_SEISMIC_SCALAR_QUANTIZED_INDEX_H
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "nsparse/disk_seismic_index.h"  // DiskSeismicSearchParameters
#include "nsparse/disk_seismic_index_base.h"
#include "nsparse/io/io.h"
#include "nsparse/mmap_index.h"  // Residency
#include "nsparse/types.h"
#include "nsparse/utils/scalar_quantizer.h"

namespace nsparse {

// Overrides the quantizer range a query is encoded with (the type is always the
// index's); everything else is DiskSeismicSearchParameters (cut + k_prime).
struct DiskSeismicSQSearchParameters : public DiskSeismicSearchParameters {
    float vmin;
    float vmax;
    DiskSeismicSQSearchParameters(float vmin, float vmax, int cut, int k_prime)
        : DiskSeismicSearchParameters(cut, k_prime), vmax(vmax), vmin(vmin) {}
};

// A DiskSeismicIndex over scalar-quantized codes: the forward vectors and the
// cluster summaries are stored as 8- or 16-bit codes instead of float. The
// disk-resident search and serialization are shared with DiskSeismicIndex via
// DiskSeismicIndexBase; this type supplies the code width, the query
// quantization + score decoding, and a leading quantization header. Pass a
// DiskSeismicSQSearchParameters to override the query range, else the index's
// build-time range is reused.
//
// mmap-only: load with read_index(file, kUseMmap); the copying read throws.
class DiskSeismicScalarQuantizedIndex : public DiskSeismicIndexBase {
public:
    static constexpr std::array<char, 4> name = {'D', 'S', 'S', 'Q'};
    // Bump whenever write_index's payload layout changes.
    static constexpr uint32_t kFormatVersion = 1;

    explicit DiskSeismicScalarQuantizedIndex(int dim);
    DiskSeismicScalarQuantizedIndex(QuantizerType quantizer_type, float vmin,
                                    float vmax,
                                    SeismicClusterParameters parameter,
                                    int dim);
    ~DiskSeismicScalarQuantizedIndex() override = default;
    std::array<char, 4> id() const override { return name; }

    DiskSeismicScalarQuantizedIndex(const DiskSeismicScalarQuantizedIndex&) =
        delete;
    DiskSeismicScalarQuantizedIndex& operator=(
        const DiskSeismicScalarQuantizedIndex&) = delete;

    const ScalarQuantizer& get_scalar_quantizer() const { return sq_; }

    // Borrows a serialized index from a file mapping. `pos` is where the
    // payload begins, past the header read_header consumed.
    static DiskSeismicScalarQuantizedIndex* mmap_index(const IndexHeader& header,
                                                       const char* index_file,
                                                       size_t pos);

private:
    [[nodiscard]] uint32_t format_version() const override {
        return kFormatVersion;
    }
    [[nodiscard]] size_t code_element_size() const override;
    const uint8_t* encode_values(const float* values, size_t nnz,
                                 std::vector<uint8_t>& scratch) const override;
    const uint8_t* encode_query(
        const float* values, size_t nnz,
        const SearchParameters* search_parameters,
        std::vector<uint8_t>& scratch) const override;
    void decode_scores(std::vector<float>& scores,
                       const SearchParameters* search_parameters) const override;
    // The quantization parameters that open this index's payload -- distinct
    // from the IndexHeader the file itself starts with.
    void write_payload_header(IOWriter* io_writer) const override;
    void validate_mapped_payload() const override;

    // The quantizer a query is encoded with: DiskSeismicSQSearchParameters
    // overrides the range the index was built with, anything else reuses it.
    ScalarQuantizer query_quantizer(
        const SearchParameters* search_parameters) const;

    ScalarQuantizer sq_;
};
}  // namespace nsparse

#endif  // DISK_SEISMIC_SCALAR_QUANTIZED_INDEX_H
