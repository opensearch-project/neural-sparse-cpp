/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef DISK_SEISMIC_INDEX_BASE_H
#define DISK_SEISMIC_INDEX_BASE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/index.h"
#include "nsparse/io/inline_forward_index_io.h"
#include "nsparse/io/io.h"
#include "nsparse/mmap_index.h"
#include "nsparse/types.h"
#include "nsparse/utils/mmap_cursor.h"
#include "nsparse/utils/mmap_file.h"

namespace nsparse {

// Shared implementation of the two disk-resident SEISMIC indexes: the cluster
// summaries live in RAM, the per-document forward vectors live on disk in the
// block-contiguous (inline) layout and are borrowed via mmap at search time,
// and search scores the global top-k_prime blocks (a
// DiskSeismicSearchParameters sets k_prime). add / build / search /
// serialization / mmap loading are all here; the concrete indexes differ only
// in the stored value width and, for the quantized one, a leading quantization
// header and a score-decoding step, which they supply through the virtual hooks
// below.
//
// mmap-only: load with read_index(file, kUseMmap); the copying read throws.
class DiskSeismicIndexBase : public MmapIndex, public IndexIO {
public:
    // Persisted, since a mapped index has no in-RAM vectors_ to derive it from.
    size_t num_vectors() const override { return num_vectors_; }

    void add(idx_t n, const idx_t* indptr, const term_t* indices,
             const float* values) override;
    void build() override;

    // read_mcsr (the kMmap path) borrows vectors_ from the mapping but never
    // touches num_vectors_, which otherwise only add() maintains; write_index
    // and search read the member directly, so an out-of-sync count would
    // persist nv=0 and gate search to empty. Sync it here after the base read.
    // (kInMemory routes through add(), which already set it, so re-reading
    // get_vectors() is then a harmless no-op.)
    void read_csr(const char* file_path,
                  Residency residency = Residency::kInMemory) override {
        MmapIndex::read_csr(file_path, residency);
        const auto* v = get_vectors();
        if (v != nullptr) {
            num_vectors_ = v->num_vectors();
        }
    }

protected:
    DiskSeismicIndexBase(int dim, SeismicClusterParameters parameter);

    // --- Hooks the concrete indexes implement. ---

    // Stored value width in bytes: 4 (float) or 1/2 (quantized codes).
    [[nodiscard]] virtual size_t code_element_size() const = 0;

    // Encode nnz float values to the stored width, returning a pointer to
    // code_element_size()-byte-per-value data. `scratch` backs the result when
    // encoding must allocate; the float index returns its input reinterpreted,
    // with no copy.
    virtual const uint8_t* encode_values(
        const float* values, size_t nnz,
        std::vector<uint8_t>& scratch) const = 0;

    // Encode a query batch the same way, honoring any per-search range
    // override.
    virtual const uint8_t* encode_query(
        const float* values, size_t nnz,
        const SearchParameters* search_parameters,
        std::vector<uint8_t>& scratch) const = 0;

    // Decode raw integer dot products back to float. No-op for the float index.
    virtual void decode_scores(
        std::vector<float>& /*scores*/,
        const SearchParameters* /*search_parameters*/) const {}

    // Extra header this index writes before the shared payload (the
    // quantization header for the quantized index; nothing for the float one).
    virtual void write_payload_header(IOWriter* /*io_writer*/) const {}

    // The mirror of write_payload_header: consume that header off a mapping and
    // adopt what it declares. Used by the mapped read and by a batched build
    // reopening the file it just wrote, so the two cannot read the payload from
    // different offsets.
    virtual void read_mapped_payload_header(MmapCursor* /*cursor*/) {}

    // Reject a just-mapped payload whose stored width disagrees with this
    // index. No-op for the float index, which stores a fixed width. Runs after
    // the summaries and forward index are populated but before the mapping
    // commits.
    virtual void validate_mapped_payload() const {}

    // Reads the shared payload (doc count, summaries, inline forward) from the
    // cursor, validates it, and commits the mapping into `slot`. Each concrete
    // mmap_index reads its own extra header first, then calls this.
    //
    // `slot` is mapped_file_ for a read_index load, which owns nothing else,
    // but batch_mapped_file_ for a batched build: there the corpus may still be
    // borrowing from mapped_file_, and giving that up would leave the index
    // unable to score anything.
    void load_mapped_payload(MmapCursor* cursor, MmapFile&& mapped,
                             MmapFile* slot);
    void load_mapped_payload(MmapCursor* cursor, MmapFile&& mapped) {
        load_mapped_payload(cursor, std::move(mapped), &mapped_file_);
    }

    // Borrowed from by score_summaries_transposed / the inline forward index,
    // so the concrete validate_mapped_payload can inspect their widths.
    std::vector<InvertedListClusters> clustered_inverted_lists;
    detail::InlineForwardIndex fwd_;

private:
    // build() with batch_file_output_path set: clusters one term window at a
    // time into a spill file, writes the index out of that mapping, and ends
    // borrowing its own output. See the definition for why this payload cannot
    // be streamed section by section the way the in-memory ones are.
    void build_streamed(const SparseVectorsConfig& config,
                        const std::string& out_path);

    auto search(idx_t n, const idx_t* indptr, const term_t* indices,
                const float* values, int k,
                SearchParameters* search_parameters = nullptr)
        -> pair_of_score_id_vectors_t override;
    void write_index(IOWriter* io_writer) override;
    // Unsupported: the inline forward index is mmap-only. Throws.
    void read_index(IOReader* io_reader, const IndexHeader& header,
                    int io_flags = 0) override;

    SeismicClusterParameters cluster_parameter_;
    size_t num_vectors_ = 0;
};

}  // namespace nsparse

#endif  // DISK_SEISMIC_INDEX_BASE_H
