/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef NSPARSE_HUGEPAGE_H
#define NSPARSE_HUGEPAGE_H

#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace nsparse::detail {

// Hint the kernel to back a large, mostly-read-only array with transparent
// huge pages. Random gathers into the multi-GB posting/summary arrays are
// dominated by TLB misses + page-table walks (2 MB pages cover 512x the span
// of 4 KB pages), so this cuts the DTLB/page-walk cost with no change to the
// data layout or results. No-op on non-Linux or for small buffers.
template <class T>
inline void advise_hugepage(const std::vector<T>& v) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    const size_t bytes = v.size() * sizeof(T);
    // 2 MB is the x86-64 huge-page size; only bother once the region spans at
    // least one huge page so the alignment rounding can't consume the whole
    // buffer.
    static constexpr size_t kHugePage = size_t{2} << 20;
    if (bytes < kHugePage) {
        return;
    }
    const auto base = reinterpret_cast<uintptr_t>(v.data());
    // madvise() requires a page-aligned start; round the start up and the end
    // down to whole huge pages so only fully-covered pages are advised.
    const uintptr_t start = (base + kHugePage - 1) & ~(kHugePage - 1);
    const uintptr_t end = (base + bytes) & ~(kHugePage - 1);
    if (end > start) {
        void* p = reinterpret_cast<void*>(start);
        const size_t len = static_cast<size_t>(end - start);
        // Mark the region so future faults and khugepaged prefer huge pages.
        ::madvise(p, len, MADV_HUGEPAGE);
#if defined(MADV_COLLAPSE)
        // The arrays are already faulted in (resize + read) as 4 KB pages, so a
        // plain MADV_HUGEPAGE only takes effect lazily via khugepaged.
        // MADV_COLLAPSE (Linux 6.1+) collapses the existing pages into huge
        // pages synchronously, giving the TLB win immediately. Best-effort:
        // ignore failure (unsupported kernel, fragmentation).
        ::madvise(p, len, MADV_COLLAPSE);
#endif
    }
#else
    (void)v;
#endif
}

}  // namespace nsparse::detail

#endif  // NSPARSE_HUGEPAGE_H
