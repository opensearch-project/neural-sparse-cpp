/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef CSR_LAYOUT_H
#define CSR_LAYOUT_H

#include <cstddef>
#include <cstdint>
#include <string>

#include "nsparse/types.h"

// Two on-disk CSR layouts, both after an int64 header of (rows, cols, nnz):
//
//   interchange           native
//   int64  indptr[r + 1]  idx_t  indptr[r + 1]
//   int32  indices[nnz]   term_t indices[nnz]
//                         <padding to 4 bytes>
//   float  values[nnz]    value  values[nnz]
//
// Interchange is what scipy writes and what Index::read_csr narrows while
// copying. Native stores in-memory widths so MmapIndex can borrow the arrays in
// place; values is padded because term_t is 2 bytes, and an odd nnz would leave
// the floats misaligned for the mapped reader's in-place reinterpret. The
// native value width is the index's own: 4-byte float for the unquantized
// types, or 1-/2-byte scalar codes for a quantizing one (see native_file_size's
// element_size). The pad is to alignof(float), which also satisfies the smaller
// code widths.
//
// File size cannot tell the layouts apart (rows=1, nnz=4 is 64 bytes either
// way), so conversion is explicit and the results are kept apart by suffix.
namespace nsparse::csr_layout {

constexpr size_t kHeaderBytes = 3 * sizeof(int64_t);

inline constexpr const char* kNativeSuffix = ".mcsr";

// Bytes of padding before values, given everything preceding it.
constexpr size_t padding(size_t pos) {
    return (alignof(float) - pos % alignof(float)) % alignof(float);
}

constexpr size_t interchange_file_size(size_t indptr_size, size_t nnz) {
    return kHeaderBytes + indptr_size * sizeof(int64_t) +
           nnz * sizeof(int32_t) + nnz * sizeof(float);
}

constexpr size_t native_values_offset(size_t indptr_size, size_t nnz) {
    const size_t unaligned =
        kHeaderBytes + indptr_size * sizeof(idx_t) + nnz * sizeof(term_t);
    return unaligned + padding(unaligned);
}

// `element_size` is the native value width: sizeof(float) for the unquantized
// types (the default), or the quantizer's byte width (1 or 2) for a quantizing
// index whose CSR holds codes rather than floats.
constexpr size_t native_file_size(size_t indptr_size, size_t nnz,
                                  size_t element_size = sizeof(float)) {
    return native_values_offset(indptr_size, nnz) + nnz * element_size;
}

std::string native_path(const std::string& path);

// Rewrites an interchange file as a native one. Throws if the source is
// malformed or an element does not fit its narrower type. Streams the arrays, so
// peak memory is a fixed buffer rather than the file.
void convert(const std::string& interchange_path,
             const std::string& native_path);

}  // namespace nsparse::csr_layout

#endif  // CSR_LAYOUT_H
