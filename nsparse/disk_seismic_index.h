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

// How many of the best block-scored candidates get re-scored against their
// untruncated vectors, for an index built with a truncated inline forward index
// (see inline_forward_index.h). Ignored when nothing is truncated.
//
// This is the recall/latency knob, and it is not free: reaching a whole copy is
// a scattered read, so each extra candidate costs far more than its dot product
// (benchmarks/DISK_SEISMIC_BENCH.md has the curve). 200 is the smallest depth at
// which recall@10 over MS MARCO comes back within the noise a k-means seed
// already causes, measured at inline_max_nnz=48; raising it buys hundredths of a
// point for a third more latency, and at the candidate-pool size it is simply
// the unpruned search. Lower it to trade recall for speed.
inline constexpr int kDefaultRescoreDepth = 200;

// `cut` (inherited) bounds which posting lists' summaries are scored; `k_prime`
// caps how many blocks are read (the k_prime highest-scoring ones). Replaces
// the inherited heap_factor, which the disk-resident indexes ignore. `rescore`
// matters only for an index built with inline_max_nnz > 0: values below k are
// raised to k, and a negative one is rejected at search time.
struct DiskSeismicSearchParameters : public SeismicSearchParameters {
    int k_prime = kDefaultBlockBudget;
    int rescore = kDefaultRescoreDepth;
    DiskSeismicSearchParameters() = default;
    DiskSeismicSearchParameters(int cut, int k_prime)
        : SeismicSearchParameters(cut, /*heap_factor=*/1.0F),
          k_prime(k_prime) {}
    DiskSeismicSearchParameters(int cut, int k_prime, int rescore)
        : SeismicSearchParameters(cut, /*heap_factor=*/1.0F),
          k_prime(k_prime),
          rescore(rescore) {}
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
    // Bump whenever write_index's payload layout changes. 2 added the
    // truncated inline forward index: a v2 file may hold partial copies that a
    // v1 reader would score as if whole. Reading v1 stays supported (nothing is
    // truncated there), so only new files need the newer build.
    static constexpr uint32_t kFormatVersion = 2;

    explicit DiskSeismicIndex(int dim);
    DiskSeismicIndex(int dim, SeismicClusterParameters parameter,
                     uint32_t inline_max_nnz = 0);
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
