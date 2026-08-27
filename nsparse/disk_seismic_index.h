/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef DISK_SEISMIC_INDEX_H
#define DISK_SEISMIC_INDEX_H
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "nsparse/disk_seismic_index_base.h"
#include "nsparse/io/io.h"
#include "nsparse/seismic_index.h"  // SeismicSearchParameters
#include "nsparse/types.h"

namespace nsparse {

// Default block budget K': the primary recall/latency knob.
inline constexpr int kDefaultBlockBudget = 50;

// `cut` (inherited) bounds which posting lists' summaries are scored; `k_prime`
// caps how many blocks are read (the k_prime highest-scoring ones). Replaces
// the inherited heap_factor, which the disk-resident indexes ignore.
struct DiskSeismicSearchParameters : public SeismicSearchParameters {
    int k_prime = kDefaultBlockBudget;
    DiskSeismicSearchParameters() = default;
    DiskSeismicSearchParameters(int cut, int k_prime)
        : SeismicSearchParameters(cut, /*heap_factor=*/1.0F),
          k_prime(k_prime) {}
};

// A SEISMIC index whose per-document forward vectors live on disk as float, in
// the block-contiguous (inline) layout, borrowed via mmap at search time; the
// cluster summaries stay in RAM. The disk-resident search and serialization
// live in DiskSeismicIndexBase; this type only pins the value width to float.
//
// mmap-only: load with read_index(file, kUseMmap); the copying read throws.
class DiskSeismicIndex : public DiskSeismicIndexBase {
public:
    static constexpr std::array<char, 4> name = {'D', 'S', 'E', 'I'};
    // Bump whenever write_index's payload layout changes.
    static constexpr uint32_t kFormatVersion = 1;

    explicit DiskSeismicIndex(int dim);
    DiskSeismicIndex(int dim, SeismicClusterParameters parameter);
    ~DiskSeismicIndex() override = default;
    std::array<char, 4> id() const override { return name; }

    DiskSeismicIndex(const DiskSeismicIndex&) = delete;
    DiskSeismicIndex& operator=(const DiskSeismicIndex&) = delete;

    // Borrows a serialized index from a file mapping. `pos` is where the
    // payload begins.
    static DiskSeismicIndex* mmap_index(const IndexHeader& header,
                                        const char* index_file, size_t pos);

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
};
}  // namespace nsparse

#endif  // DISK_SEISMIC_INDEX_H
