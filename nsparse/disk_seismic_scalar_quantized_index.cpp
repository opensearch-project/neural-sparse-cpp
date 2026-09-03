/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/disk_seismic_scalar_quantized_index.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/disk_seismic_index_base.h"
#include "nsparse/index.h"
#include "nsparse/io/inline_forward_index_io.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"
#include "nsparse/utils/mmap_cursor.h"
#include "nsparse/utils/mmap_file.h"
#include "nsparse/utils/scalar_quantizer.h"

namespace nsparse {
namespace {

// Validate a quantizer type declared by a stored index and construct it.
// bytes_per_value() treats anything but QT_8bit as 16-bit, so an undefined type
// would silently pick an element width rather than be rejected.
ScalarQuantizer make_scalar_quantizer(QuantizerType type, float vmin,
                                      float vmax) {
    if (type != QuantizerType::QT_8bit && type != QuantizerType::QT_16bit) {
        throw std::runtime_error(
            "index file declares an unknown quantizer type");
    }
    return ScalarQuantizer(type, vmin, vmax);
}

// The stored codes are strided at the width the quantizer reports, both in the
// forward index and in the cluster summaries (score_summaries_transposed
// dispatches on the summaries' own width). A file where either disagrees would
// be read at the wrong stride, so reject it at load.
void throw_if_forward_width_mismatch(const detail::InlineForwardIndex& fwd,
                                     const ScalarQuantizer& sq) {
    if (fwd.num_blocks() > 0 && fwd.element_size() != sq.bytes_per_value()) {
        throw std::runtime_error(
            "index file's forward element size disagrees with its quantizer "
            "type");
    }
}

void throw_if_summary_width_mismatch(
    const std::vector<InvertedListClusters>& clusters,
    const ScalarQuantizer& sq) {
    for (const InvertedListClusters& list : clusters) {
        if (list.cluster_size() > 0 &&
            list.element_size() != sq.bytes_per_value()) {
            throw std::runtime_error(
                "index file's summary element size disagrees with its "
                "quantizer type");
        }
    }
}

}  // namespace

DiskSeismicScalarQuantizedIndex::DiskSeismicScalarQuantizedIndex(int dim)
    : DiskSeismicIndexBase(dim, detail::kDefaultSeismicClusterParams) {}

DiskSeismicScalarQuantizedIndex::DiskSeismicScalarQuantizedIndex(
    QuantizerType quantizer_type, float vmin, float vmax,
    SeismicClusterParameters parameter, int dim)
    : DiskSeismicIndexBase(dim, parameter), sq_(quantizer_type, vmin, vmax) {}

void DiskSeismicScalarQuantizedIndex::read_csr(const char* file_path,
                                               Residency residency) {
    if (residency == Residency::kMmap) {
        throw std::invalid_argument(
            "mmap residency is not available for a quantized index: a mapped "
            "CSR is borrowed as float, and this index searches over codes");
    }
    MmapIndex::read_csr(file_path, residency);
}

size_t DiskSeismicScalarQuantizedIndex::code_element_size() const {
    return sq_.bytes_per_value();
}

const uint8_t* DiskSeismicScalarQuantizedIndex::encode_values(
    const float* values, size_t nnz, std::vector<uint8_t>& scratch) const {
    scratch.resize(nnz * sq_.bytes_per_value());
    sq_.encode(values, scratch.data(), nnz);
    return scratch.data();
}

const uint8_t* DiskSeismicScalarQuantizedIndex::encode_query(
    const float* values, size_t nnz,
    const SearchParameters* search_parameters,
    std::vector<uint8_t>& scratch) const {
    const ScalarQuantizer query_sq = query_quantizer(search_parameters);
    scratch.resize(nnz * sq_.bytes_per_value());
    query_sq.encode(values, scratch.data(), nnz);
    return scratch.data();
}

void DiskSeismicScalarQuantizedIndex::decode_scores(
    std::vector<float>& scores,
    const SearchParameters* search_parameters) const {
    // Decode the integer dot products back to float. Called before the -1.0
    // padding is added, so the sentinel is never scaled.
    const ScalarQuantizer query_sq = query_quantizer(search_parameters);
    for (float& score : scores) {
        score = sq_.decode_dot_product(score, query_sq);
    }
}

// The type is always the index's, since the codes are compared against stored
// ones; only the range can be overridden per search.
ScalarQuantizer DiskSeismicScalarQuantizedIndex::query_quantizer(
    const SearchParameters* search_parameters) const {
    const auto* sq_params =
        dynamic_cast<const DiskSeismicSQSearchParameters*>(search_parameters);
    if (sq_params == nullptr) {
        return sq_;
    }
    return ScalarQuantizer(sq_.get_quantizer_type(), sq_params->vmin,
                           sq_params->vmax);
}

void DiskSeismicScalarQuantizedIndex::write_payload_header(
    IOWriter* io_writer) const {
    auto sq_type = sq_.get_quantizer_type();
    io_writer->write(&sq_type, sizeof(QuantizerType), 1);
    auto vmin = sq_.get_min();
    io_writer->write(&vmin, sizeof(float), 1);
    auto vmax = sq_.get_max();
    io_writer->write(&vmax, sizeof(float), 1);
}

void DiskSeismicScalarQuantizedIndex::read_mapped_payload_header(
    MmapCursor* cursor) {
    const auto sq_type = cursor->read_scalar<QuantizerType>();
    const auto vmin = cursor->read_scalar<float>();
    const auto vmax = cursor->read_scalar<float>();
    sq_ = make_scalar_quantizer(sq_type, vmin, vmax);
}

void DiskSeismicScalarQuantizedIndex::validate_mapped_payload() const {
    throw_if_forward_width_mismatch(fwd_, sq_);
    throw_if_summary_width_mismatch(clustered_inverted_lists, sq_);
}

DiskSeismicScalarQuantizedIndex* DiskSeismicScalarQuantizedIndex::mmap_index(
    const IndexHeader& header, const char* index_file, size_t pos) {
    throw_if_null(index_file, "index_file must not be null");
    auto index =
        std::make_unique<DiskSeismicScalarQuantizedIndex>(header.dimension);

    MmapFile mmap_file(std::string{index_file});
    MmapCursor cursor(mmap_file.data(), mmap_file.size());
    cursor.skip(pos);

    // The quantization header opens the payload; the shared loader reads the
    // rest and calls validate_mapped_payload() against this quantizer.
    index->read_mapped_payload_header(&cursor);
    index->load_mapped_payload(&cursor, std::move(mmap_file));
    return index.release();
}
}  // namespace nsparse
