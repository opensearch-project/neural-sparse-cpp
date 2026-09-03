/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/disk_seismic_index.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nsparse/disk_seismic_index_base.h"
#include "nsparse/seismic_common.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"
#include "nsparse/utils/mmap_cursor.h"
#include "nsparse/utils/mmap_file.h"

namespace nsparse {

DiskSeismicIndex::DiskSeismicIndex(int dim)
    : DiskSeismicIndexBase(dim, detail::kDefaultSeismicClusterParams) {}

DiskSeismicIndex::DiskSeismicIndex(int dim, SeismicClusterParameters parameter)
    : DiskSeismicIndexBase(dim, parameter) {}

size_t DiskSeismicIndex::code_element_size() const { return U32; }

// Float values are stored verbatim, so their bytes are the codes -- no copy,
// `scratch` unused.
const uint8_t* DiskSeismicIndex::encode_values(
    const float* values, size_t /*nnz*/,
    std::vector<uint8_t>& /*scratch*/) const {
    return reinterpret_cast<const uint8_t*>(values);
}

const uint8_t* DiskSeismicIndex::encode_query(
    const float* values, size_t /*nnz*/,
    const SearchParameters* /*search_parameters*/,
    std::vector<uint8_t>& /*scratch*/) const {
    return reinterpret_cast<const uint8_t*>(values);
}

DiskSeismicIndex* DiskSeismicIndex::mmap_index(const IndexHeader& header,
                                               const char* index_file,
                                               size_t pos) {
    throw_if_null(index_file, "index_file must not be null");
    auto index = std::make_unique<DiskSeismicIndex>(header.dimension);

    MmapFile mmap_file(std::string{index_file});
    MmapCursor cursor(mmap_file.data(), mmap_file.size());
    cursor.skip(pos);

    // No extra header for the float index, so the hook is a no-op and the
    // shared payload follows directly; called anyway, so the two disk types
    // read their payload through the same two steps.
    index->read_mapped_payload_header(&cursor);
    index->load_mapped_payload(&cursor, std::move(mmap_file));
    return index.release();
}
}  // namespace nsparse
