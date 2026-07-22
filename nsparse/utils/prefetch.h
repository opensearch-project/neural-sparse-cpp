/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef PREFETCH_H
#define PREFETCH_H
#include <algorithm>
#include <cstddef>

#include "nsparse/types.h"

#ifndef NSPARSE_PREFETCH
#ifdef _MSC_VER
#include <intrin.h>
#define NSPARSE_PREFETCH(addr, rw, locality) \
    _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#else
#define NSPARSE_PREFETCH(addr, rw, locality) \
    __builtin_prefetch(addr, rw, locality)
#endif
#endif

namespace nsparse::detail {

template <class T>
inline void prefetch_vector(const term_t* indices, const T* values,
                            size_t len) {
    static constexpr size_t kCacheLineSize = 64;  // bytes

    const char* indices_ptr = reinterpret_cast<const char*>(indices);
    const char* values_ptr = reinterpret_cast<const char*>(values);

    const size_t indices_bytes = len * sizeof(term_t);
    const size_t values_bytes = len * sizeof(T);

    for (size_t offset = 0; offset < indices_bytes; offset += kCacheLineSize) {
        NSPARSE_PREFETCH(indices_ptr + offset, 0, 0);
    }
    for (size_t offset = 0; offset < values_bytes; offset += kCacheLineSize) {
        NSPARSE_PREFETCH(values_ptr + offset, 0, 0);
    }
}

// Prefetch only the leading `max_lines` cache lines of each stream. A doc row
// spans ~13 lines here; prefetching all of them for several docs ahead
// overruns the core's ~10-12 line-fill buffers (l1d_pend_miss.fb_full), which
// stalls demand loads. Because the row is contiguous, touching just the first
// line or two lets the hardware stream prefetcher pull the rest while keeping
// the number of outstanding software prefetches bounded.
template <class T>
inline void prefetch_vector_head(const term_t* indices, const T* values,
                                 size_t len, size_t max_lines) {
    static constexpr size_t kCacheLineSize = 64;  // bytes
    const char* indices_ptr = reinterpret_cast<const char*>(indices);
    const char* values_ptr = reinterpret_cast<const char*>(values);
    const size_t indices_bytes =
        std::min(len * sizeof(term_t), max_lines * kCacheLineSize);
    const size_t values_bytes =
        std::min(len * sizeof(T), max_lines * kCacheLineSize);
    for (size_t offset = 0; offset < indices_bytes; offset += kCacheLineSize) {
        NSPARSE_PREFETCH(indices_ptr + offset, 0, 0);
    }
    for (size_t offset = 0; offset < values_bytes; offset += kCacheLineSize) {
        NSPARSE_PREFETCH(values_ptr + offset, 0, 0);
    }
}

inline void prefetch_indptr(const idx_t* indptr, idx_t doc_id) {
    NSPARSE_PREFETCH(&indptr[doc_id], 0, 0);
}

}  // namespace nsparse::detail

#endif  // PREFETCH_H