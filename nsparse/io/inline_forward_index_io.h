/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef INLINE_FORWARD_INDEX_IO_H
#define INLINE_FORWARD_INDEX_IO_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "nsparse/cluster/inverted_list_clusters.h"
#include "nsparse/inline_forward_index.h"
#include "nsparse/io/io.h"
#include "nsparse/io/mmap_io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/buf.h"
#include "nsparse/utils/mmap_cursor.h"

// Internal serialization machinery; not exposed to Python/JNI, hence detail.
namespace nsparse::detail {

// A read-only, in-place view of one block's structure-of-arrays (a within-block
// CSR). Every array points into the mapping and is aligned for its element
// type, so it can be read directly. `doc_ids == nullptr` means the (pl, block)
// does not exist; a present block with n_docs == 0 has non-null (empty) arrays.
struct BlockView {
    uint32_t n_docs = 0;
    const uint32_t* doc_ids = nullptr;  // [n_docs] global doc ids
    // [n_docs + 1] within-block CSR offsets; u32 but capped at INT32_MAX by the
    // writer, so they convert to idx_t losslessly for compute_similarity.
    const uint32_t* offsets = nullptr;
    const term_t* comps = nullptr;  // [offsets[n_docs]] component ids
    const uint8_t* vals = nullptr;  // element_size bytes * offsets[n_docs]

    bool absent() const { return doc_ids == nullptr; }

    // Component count / component ids / value bytes of the i-th doc (i <
    // n_docs).
    uint32_t nnz(uint32_t i) const { return offsets[i + 1] - offsets[i]; }
    const term_t* doc_comps(uint32_t i) const { return comps + offsets[i]; }
    const uint8_t* doc_vals(uint32_t i, size_t element_size) const {
        return vals + static_cast<size_t>(offsets[i]) * element_size;
    }
};

// One doc's within-doc slice: component ids, element_size-wide codes, and the
// count. Borrowed from a live mapping (or an in-RAM store), so valid only for as
// long as its source is.
struct DocSlice {
    const term_t* comps = nullptr;
    const uint8_t* vals = nullptr;
    size_t nnz = 0;
};

class InlineForwardIndex;

// Where to find the untruncated copy of any doc when the blocks hold truncated
// ones (see inline_forward_index.h). `locators` is the per-doc directory; a doc
// that no block holds lives in `remainder` instead. One copy of the corpus, not
// one per block, which is what makes re-scoring cheap enough to pay for the
// truncation.
struct FullVectorStore {
    const InlineForwardIndex* fwd = nullptr;
    const DocLocator* locators = nullptr;
    uint64_t num_locators = 0;
    const SparseVectors* remainder = nullptr;
};

// The untruncated vector of `doc_id`. Throws on a locator that does not resolve
// to its doc (a corrupt directory), so a caller never scores another doc's
// bytes.
DocSlice resolve_doc(const FullVectorStore& store, idx_t doc_id,
                     size_t element_size);

// Scratch for the batched resolve below, owned by the caller and reused across
// queries so the walk allocates nothing.
struct DocResolveScratch {
    std::vector<DocLocator> locators;
    std::vector<const InlineDirEntry*> entries;
};

// resolve_doc for many docs at once, and much faster per doc: reaching one whole
// vector is a four-deep pointer chase (locator -> directory entry -> block
// header -> the doc's slice), each step a likely cache miss on a table far too
// big to cache, and done one doc at a time the misses serialize. This walks all
// the docs through each step together, prefetching the next step's input for
// every doc before any of them needs it, so the misses overlap instead. Same
// results as calling resolve_doc in a loop, including the throws.
void resolve_docs(const FullVectorStore& store, const idx_t* doc_ids,
                  size_t n_docs, size_t element_size,
                  DocResolveScratch* scratch, std::vector<DocSlice>* out);

// The inline forward-index as a serialization component (format in
// inline_forward_index.h). Construct with the source lists + vectors to write
// it (serialize); default-construct to read it. On read the directory is copied
// into RAM and block payloads are borrowed in place, giving O(1) (pl, block)
// lookup. Move-only.
//
// The serialized form is self-delimiting: serialize() pads to kMinBlockAlign
// (so the section base is readable wherever the enclosing stream places it) and
// prefixes the section with its byte length; mmap_deserialize() skips the same
// padding, reads the length, and advances a shared cursor exactly past itself
// -- like its sibling components, no caller-scoped subcursor required. The
// length is precomputed by a sizing pass, so serialize() streams the body
// straight to the writer without buffering the whole (corpus-sized) section in
// RAM. The mapping must outlive this object. mmap_deserialize throws
// std::runtime_error on a malformed section (fail-closed: block() never returns
// an out-of-bounds view).
//
// deserialize(IOReader*) is unsupported and throws: the forward index is
// disk-resident, read only by borrowing from a mapping. This deliberately
// narrows Serializable (a Liskov break), so any index that embeds this
// component is mmap-only -- it can never be loaded in kInMemory residency.
// Intended for a DiskSeismicIndex; state the constraint wherever such an index
// is composed.
class InlineForwardIndex : public MmapSerializable {
public:
    // Default block alignment. 4096 is the common OS page size, but this is
    // just the alignment granularity, not a claim about the host page (a 16
    // KB-page host gets sub-page block starts from this) -- caller-overridable.
    static constexpr uint64_t kDefaultPageSize = 4096;

    InlineForwardIndex() = default;  // read mode; fill via mmap_deserialize

    // Write mode. `lists`/`vectors` must outlive any serialize() call.
    // page_size: block alignment when page-aligned (power of two, >= header
    // size); ignored when packed, which is the default (see InlineLayout).
    //
    // max_doc_nnz > 0 truncates a doc's *extra* copies to its largest
    // max_doc_nnz components (see inline_forward_index.h); `full_copies` then
    // names, per doc id, the one block whose copy stays whole, and must outlive
    // serialize() too. The two arguments only make sense together, so passing
    // one without the other throws.
    InlineForwardIndex(const std::vector<InvertedListClusters>& lists,
                       const SparseVectors& vectors,
                       uint64_t page_size = kDefaultPageSize,
                       InlineLayout layout = InlineLayout::kPacked,
                       uint32_t max_doc_nnz = 0,
                       const std::vector<DocLocator>* full_copies = nullptr);

    // Explicit, not defaulted: Buf's move keeps the source's size()/data()
    // (only its owner moves), so the moved-from members are reset to stay
    // inert.
    InlineForwardIndex(InlineForwardIndex&& other) noexcept;
    InlineForwardIndex& operator=(InlineForwardIndex&& other) noexcept;
    InlineForwardIndex(const InlineForwardIndex&) = delete;
    InlineForwardIndex& operator=(const InlineForwardIndex&) = delete;

    void serialize(IOWriter* writer) const override;
    void deserialize(IOReader* reader) override;  // unsupported (throws)
    void mmap_deserialize(MmapCursor* cursor) override;

    size_t element_size() const { return element_size_; }
    uint64_t num_blocks() const { return entries_.size(); }
    // list_offset_ holds num_lists + 1 prefix-sum entries; empty when
    // moved-from.
    uint64_t num_lists() const {
        return list_offset_.empty() ? 0 : list_offset_.size() - 1;
    }
    uint64_t page_size() const { return page_size_; }
    // 0 = every copy of every doc is whole; otherwise block scores are lower
    // bounds and the caller must re-score its best candidates against the whole
    // copies. See inline_forward_index.h.
    uint32_t max_doc_nnz() const { return max_doc_nnz_; }
    uint64_t num_blocks_in_list(uint32_t pl) const;
    // Section start, i.e. where a directory entry's byte_off is measured from.
    // Exposed so a batched walk can prefetch a block before reading it.
    const uint8_t* block_base() const { return block_base_; }

    BlockView block(uint32_t pl, uint32_t block) const;

    // The directory entry for (pl, block), or null when the block is absent.
    // Exposed so a batched walk can prefetch the entry before dereferencing it.
    const InlineDirEntry* dir_entry(uint32_t pl, uint32_t block) const;
    // One doc's slice out of an already-located block. Reads only off[slot],
    // off[slot + 1] and off[n_docs], where block() validates the whole block --
    // worth separating because the re-scoring pass wants one doc out of a block
    // it will not otherwise touch. Still fail-closed: the slot and the block
    // length are checked, so a corrupt entry cannot yield an out-of-range slice.
    DocSlice doc_slice(const InlineDirEntry& entry, uint32_t slot) const;

private:
    // n_docs and total_nnz of a block, from a validated doc-id span.
    struct BlockCounts {
        uint32_t n_docs;
        uint64_t total_nnz;
    };
    // page_size when page-aligned, else kMinBlockAlign (packed).
    uint64_t write_alignment() const {
        return write_layout_ == InlineLayout::kPageAligned ? write_page_size_
                                                           : kMinBlockAlign;
    }
    // Validate a block's doc ids and return its counts; total_nnz is capped at
    // INT32_MAX so off[] stays idx_t-convertible. Shared by section_length()
    // (sizing) and write_body() (emitting) so their per-block math can't drift,
    // which is also why it must know which block it is: a doc contributes its
    // whole nnz to one block and a truncated nnz to the rest.
    BlockCounts count_block(std::span<const idx_t> docs, const offset_t* indptr,
                            size_t num_vectors, uint32_t pl,
                            uint32_t block) const;
    // Components this block stores for `doc_id`: all of them when truncation is
    // off, when the doc is short enough anyway, or when this is the block
    // holding the doc's whole copy; else write_max_doc_nnz_.
    uint64_t doc_stored_nnz(idx_t doc_id, uint64_t full_nnz, uint32_t pl,
                            uint32_t block) const;

    // Which of a doc's components a truncated copy keeps: everything whose code
    // is above `boundary`, plus the first `ties` of those equal to it. `whole`
    // short-circuits a copy that keeps everything.
    struct CodeCut {
        double boundary = 0.0;
        uint32_t ties = 0;
        bool whole = true;
    };
    // The cut that keeps the `keep` largest of `nnz` codes. `scratch` is reused
    // across calls so the per-doc selection does not allocate.
    static CodeCut select_top_codes(const uint8_t* values, uint64_t nnz,
                                    uint64_t keep, size_t element_size,
                                    std::vector<double>* scratch);
    // Append the components (and/or the value bytes) the cut keeps, in stored
    // order. Either output may be null; both read the same cut, which is what
    // keeps the comps[] and vals[] passes in step.
    static void gather_selected(const term_t* comps, const uint8_t* values,
                                uint64_t nnz, size_t element_size, CodeCut cut,
                                std::vector<term_t>* comps_out,
                                std::vector<uint8_t>* vals_out);
    // Byte length of the section body, from a payload-free sizing pass over
    // lists + indptr. Lets serialize() write the length prefix without first
    // rendering the body into a buffer.
    uint64_t section_length() const;
    // Stream the section body (header, blocks, directory, trailer) straight to
    // the writer; serialize() precedes it with padding + the length prefix.
    void write_body(IOWriter* writer) const;
    // Validate the section at [base, base + section_len) and load its directory
    // into RAM; sets element_size_/page_size_/entries_/list_offset_.
    void load_directory(const uint8_t* base, size_t section_len);
    // Validate a block's interior and build an in-place SoA view.
    BlockView view_block(const InlineDirEntry& entry) const;

    // Write mode (borrowed source; null in read mode).
    const std::vector<InvertedListClusters>* lists_ = nullptr;
    const SparseVectors* vectors_ = nullptr;
    uint64_t write_page_size_ = kDefaultPageSize;
    InlineLayout write_layout_ = InlineLayout::kPageAligned;
    uint32_t write_max_doc_nnz_ = 0;
    const std::vector<DocLocator>* write_full_copies_ = nullptr;

    // Read mode.
    const uint8_t* block_base_ =
        nullptr;  // section start (blocks at +byte_off)
    size_t element_size_ = 0;
    uint64_t page_size_ = 0;
    uint32_t max_doc_nnz_ = 0;
    Buf<InlineDirEntry> entries_;  // directory copied into RAM
    Buf<uint64_t> list_offset_;    // start of each list in entries_
};

}  // namespace nsparse::detail

#endif  // INLINE_FORWARD_INDEX_IO_H
