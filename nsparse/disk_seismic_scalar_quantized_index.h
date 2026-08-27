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

#include "absl/container/flat_hash_set.h"
#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/disk_seismic_index.h"  // DiskSeismicSearchParameters
#include "nsparse/io/inline_forward_index_io.h"
#include "nsparse/io/io.h"
#include "nsparse/mmap_index.h"
#include "nsparse/seismic_common.h"
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

// A DiskSeismicIndex over scalar-quantized codes: the per-document forward
// vectors and the cluster summaries are stored as 8- or 16-bit codes instead of
// float. The forward vectors live on disk in the block-contiguous (inline)
// layout, borrowed via mmap at search time; the summaries stay in RAM. Search
// scores every candidate block's summary across the `cut` posting lists in code
// space, selects the global top-k_prime blocks, reads and scores only those,
// then decodes the integer dot products back to float. Pass a
// DiskSeismicSQSearchParameters to override the query range, else the index's
// build-time range is reused.
//
// This mirrors how SeismicScalarQuantizedIndex relates to SeismicIndex: a
// parallel class, not a subclass, cloning DiskSeismicIndex's block-budget
// search with the value width threaded through and a quantizer header prepended.
//
// mmap-only: load with read_index(file, kUseMmap); the copying read throws.
class DiskSeismicScalarQuantizedIndex : public MmapIndex, public IndexIO {
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

    void add(idx_t n, const idx_t* indptr, const term_t* indices,
             const float* values) override;
    void build() override;

    // Persisted, since a mapped index has no in-RAM vectors_ to derive it from.
    size_t num_vectors() const override { return num_vectors_; }

    const ScalarQuantizer& get_scalar_quantizer() const { return sq_; }

    // Borrows a serialized index from a file mapping. `pos` is where the
    // payload begins, past the header read_header consumed.
    static DiskSeismicScalarQuantizedIndex* mmap_index(const IndexHeader& header,
                                                       const char* index_file,
                                                       size_t pos);

    // Only the copying residency. A mapped CSR is borrowed at the width it was
    // written in, which is float, whereas this index searches over codes: the
    // values have to pass through add() to be quantized.
    void read_csr(const char* file_path,
                  Residency residency = Residency::kInMemory) override;

protected:
    std::vector<InvertedListClusters> clustered_inverted_lists;

private:
    [[nodiscard]] uint32_t format_version() const override {
        return kFormatVersion;
    }
    void write_index(IOWriter* io_writer) override;
    // Unsupported: the inline forward index is mmap-only. Throws.
    void read_index(IOReader* io_reader, const IndexHeader& header,
                    int io_flags = 0) override;
    // The quantizer parameters that open this index's payload -- distinct from
    // the IndexHeader the file itself starts with.
    void write_quantizer_header(IOWriter* io_writer);

    auto search(idx_t n, const idx_t* indptr, const term_t* indices,
                const float* values, int k,
                SearchParameters* search_parameters = nullptr)
        -> pair_of_score_id_vectors_t override;

    ScalarQuantizer query_quantizer(
        const SearchParameters* search_parameters) const;

    // Selects the global top-k_prime blocks across `cuts` by summary score and
    // scores their docs in code space, decoding the survivors. `dense` is
    // per-thread scratch (a dimension-sized quantized-code buffer, element_size
    // bytes per dim), zero on entry and restored on exit; `visited` is cleared
    // on entry.
    auto single_query(std::vector<uint8_t>& dense,
                      absl::flat_hash_set<idx_t>& visited, const term_t* q_idx,
                      const uint8_t* q_val_bytes, size_t q_len,
                      size_t element_size, const std::vector<term_t>& cuts,
                      int k, int k_prime, const ScalarQuantizer& query_sq,
                      SearchParameters* search_parameters)
        -> pair_of_score_id_vector_t;

    ScalarQuantizer sq_;
    SeismicClusterParameters cluster_parameter_;
    // Borrows from the base's mapped_file_, so it is declared here (destroyed
    // before the mapping). Empty after a fresh build (search uses vectors_
    // then).
    detail::InlineForwardIndex fwd_;
    size_t num_vectors_ = 0;
};
}  // namespace nsparse

#endif  // DISK_SEISMIC_SCALAR_QUANTIZED_INDEX_H
