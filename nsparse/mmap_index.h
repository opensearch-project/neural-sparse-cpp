/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef MMAP_INDEX_H
#define MMAP_INDEX_H

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "nsparse/index.h"
#include "nsparse/seismic_common.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/utils/checks.h"
#include "nsparse/utils/csr_layout.h"
#include "nsparse/utils/mmap_file.h"

namespace nsparse {
class MmapIndex : public Index {
public:
    explicit MmapIndex(int dim = 0) : Index(dim) {}

    void read_csr(const char* file_path,
                  Residency residency = Residency::kInMemory) override {
        switch (residency) {
            case Residency::kInMemory:
                Index::read_csr(file_path);
                break;
            case Residency::kMmap:
                read_mcsr(file_path);
                break;
            default:
                throw std::invalid_argument("invalid residency");
        }
    }

    [[nodiscard]] const SparseVectors* get_vectors() const override {
        return vectors_.get();
    }

protected:
    // The mapping borrowed buffers point into: a native CSR file via read_csr,
    // or a serialized index file. Those sources are mutually exclusive, so one
    // member serves both.
    //
    // Borrowers do not reference it, so it must outlive them. Members are
    // destroyed in reverse declaration order, hence this one first: do not
    // reorder it past anything that borrows from it.
    MmapFile mapped_file_;

    // Either residency: buffers owned when built or deserialized, borrowed from
    // mapped_file_ when mapped. get_vectors() cannot tell the two apart.
    std::unique_ptr<SparseVectors> vectors_;

    // A second mapping, for the file a batched build streams itself to and then
    // borrows its posting lists back from. Separate from mapped_file_ rather
    // than replacing it, because the two coexist: the corpus may itself be a
    // mapping that vectors_ is still borrowing from, and giving that up would
    // leave the index unable to score anything.
    //
    // Whatever borrows from this lives in the derived class, and derived
    // members are destroyed before base ones, so the borrowers are always gone
    // first.
    MmapFile batch_mapped_file_;

private:
    // Values are borrowed at their stored width, so a quantizing index cannot
    // use this path.
    static constexpr size_t kMmapElementSize = U32;
    bool is_mmap_index_ = false;

    // Points at the native layout (csr_layout.h) rather than copying it. Terms
    // are not range-checked, unlike Index::read_csr: that scan would fault in
    // the whole indices array at open.
    void read_mcsr(const char* file_path) {
        throw_if_null(file_path, "file_path must not be null");
        if (vectors_ != nullptr) {
            // Remapping would unmap data a built index points into, and
            // borrowed buffers cannot absorb a second batch.
            throw std::runtime_error("mmap index already has vectors");
        }
        if (!std::filesystem::exists(file_path)) {
            throw std::invalid_argument(
                std::string("CSR file does not exist: ") + file_path);
        }

        MmapFile file(file_path);
        MmapCursor cursor(file.data(), file.size());
        const auto num_rows = cursor.read_scalar<int64_t>();
        const auto num_cols = cursor.read_scalar<int64_t>();
        const auto nnz = cursor.read_scalar<int64_t>();

        if (num_rows <= 0 || num_cols <= 0 || nnz < 0) {
            throw std::invalid_argument(std::string("Invalid CSR header in: ") +
                                        file_path);
        }
        if (num_rows > std::numeric_limits<idx_t>::max() ||
            nnz > std::numeric_limits<idx_t>::max()) {
            throw std::invalid_argument(std::string("CSR file too large for ") +
                                        "32-bit offsets: " + file_path);
        }
        if (num_cols > dimension_) {
            throw std::invalid_argument(
                std::string("CSR column count exceeds index dimension: ") +
                file_path);
        }

        const size_t indptr_size = static_cast<size_t>(num_rows) + 1;
        const auto nnz_size = static_cast<size_t>(nnz);
        if (file.size() !=
            csr_layout::native_file_size(indptr_size, nnz_size)) {
            throw std::invalid_argument(
                std::string("CSR file is not in the native layout (convert it "
                            "with csr_layout::convert): ") +
                file_path);
        }

        const auto* indptr = cursor.read_array<idx_t>(indptr_size);
        const auto* indices = cursor.read_array<term_t>(nnz_size);
        cursor.skip(csr_layout::native_values_offset(indptr_size, nnz_size) -
                    cursor.pos());
        const auto* values = cursor.read_array<float>(nnz_size);

        // Validates before borrowing, so a corrupt file throws here rather
        // than faulting during search.
        auto vectors =
            std::make_unique<SparseVectors>(SparseVectors::map_vectors(
                {.element_size = kMmapElementSize,
                 .dimension = static_cast<size_t>(dimension_)},
                indptr, indptr_size, indices, nnz_size,
                reinterpret_cast<const uint8_t*>(values),
                nnz_size * kMmapElementSize));

        // Committed last, so a rejected file leaves the index untouched.
        mapped_file_ = std::move(file);
        vectors_ = std::move(vectors);
    }
};
}  // namespace nsparse

#endif  // MMAP_INDEX_H
