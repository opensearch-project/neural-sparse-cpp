/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef MMAP_FILE_H
#define MMAP_FILE_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "nsparse/utils/mmap_cursor.h"

#if defined(_WIN32)
// windows.h defines min/max as macros, which turns every later
// std::numeric_limits<T>::max() into a syntax error. This header reaches most of
// the library through mmap_index.h -> seismic_index.h, so without NOMINMAX the
// breakage lands in unrelated files. WIN32_LEAN_AND_MEAN drops the socket and
// RPC headers nothing here uses. Guarded because a translation unit may have
// defined either already.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace nsparse {

// RAII wrapper around a read-only memory mapping of a whole file. The mapping
// is established once when the file is opened and torn down on destruction; the
// OS page cache keeps hot pages resident across queries, so after warmup reads
// through data() hit RAM rather than disk. The mapping is zero-copy: consumers
// point their data structures directly into the returned bytes.
class MmapFile {
public:
    // How the region is read, which picks the default madvise hint below.
    // kScan expects broad reads over one forward index (MADV_HUGEPAGE, fewer
    // TLB entries); kPointLookup expects scattered small reads (MADV_RANDOM, no
    // readahead) -- what DiskSeismic's ~k' inline blocks per query need.
    // NSPARSE_MMAP_ADVISE overrides this.
    enum class AccessPattern { kScan, kPointLookup };

    // The madvise mode for `access`, or `env` verbatim when it is non-null
    // (NSPARSE_MMAP_ADVISE overrides the per-type default). Pure so the choice
    // is unit-testable without a real mapping.
    static std::string resolve_advise(AccessPattern access, const char* env) {
        // An empty NSPARSE_MMAP_ADVISE (set but "") is treated as unset, so it
        // does not silently override the per-type default with the "hugepage"
        // fallback below.
        if (env != nullptr && *env != '\0') {
            return std::string(env);
        }
        return access == AccessPattern::kPointLookup ? "random" : "hugepage";
    }

    MmapFile() = default;
    explicit MmapFile(const std::string& path,
                      AccessPattern access = AccessPattern::kScan) {
        open(path, access);
    }
    ~MmapFile() { close(); }

    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;
    MmapFile(MmapFile&& other) noexcept { move_from(other); }
    MmapFile& operator=(MmapFile&& other) noexcept {
        if (this != &other) {
            close();
            move_from(other);
        }
        return *this;
    }

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

    void open(const std::string& path,
              AccessPattern access = AccessPattern::kScan) {
        close();
#if defined(_WIN32)
        file_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("mmap: failed to open " + path);
        }
        LARGE_INTEGER file_size;
        if (GetFileSizeEx(file_, &file_size) == 0) {
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error("mmap: failed to stat " + path);
        }
        size_ = static_cast<size_t>(file_size.QuadPart);
        if (size_ == 0) {
            return;  // empty file: leave data_ null
        }
        mapping_ = CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_ == nullptr) {
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error("mmap: CreateFileMapping failed for " + path);
        }
        data_ = static_cast<const uint8_t*>(
            MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        if (data_ == nullptr) {
            CloseHandle(mapping_);
            CloseHandle(file_);
            mapping_ = nullptr;
            file_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error("mmap: MapViewOfFile failed for " + path);
        }
#else
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("mmap: failed to open " + path);
        }
        struct stat st{};
        if (fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error("mmap: failed to stat " + path);
        }
        size_ = static_cast<size_t>(st.st_size);
        if (size_ == 0) {
            ::close(fd);
            return;  // empty file: leave data_ null
        }

        // NSPARSE_MMAP_ADVISE overrides everything; otherwise the access
        // pattern picks the default: point lookups (DiskSeismic) suppress
        // readahead (MADV_RANDOM), scans collapse to huge pages.
        const std::string mode =
            resolve_advise(access, std::getenv("NSPARSE_MMAP_ADVISE"));

        // "hugetlb": copy the file into an anonymous MAP_HUGETLB region so the
        // index data is backed by real, pre-reserved 2 MiB pages (vm.nr_hugepages
        // must be set). Unlike MADV_HUGEPAGE this is NOT advisory and NOT
        // zero-copy: it costs one memcpy of the whole file and one physical copy
        // in RAM, but it is the only way to get guaranteed huge pages for a
        // file-backed index. Diagnostic path — measures the TLB-walk ceiling.
#if defined(MAP_HUGETLB)
        if (mode == "hugetlb") {
            void* huge = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
            if (huge == MAP_FAILED) {
                ::close(fd);
                size_ = 0;
                throw std::runtime_error(
                    "mmap: MAP_HUGETLB failed (reserve vm.nr_hugepages?) for " +
                    path);
            }
            // Read the whole file into the huge-page region.
            size_t off = 0;
            auto* dst = static_cast<uint8_t*>(huge);
            while (off < size_) {
                ssize_t r = ::read(fd, dst + off, size_ - off);
                if (r <= 0) {
                    ::munmap(huge, size_);
                    ::close(fd);
                    size_ = 0;
                    throw std::runtime_error("mmap: read into hugetlb failed");
                }
                off += static_cast<size_t>(r);
            }
            ::close(fd);
            hugetlb_ = true;
            data_ = static_cast<const uint8_t*>(huge);
            return;
        }
#endif

        void* addr = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        // The fd can be closed immediately; the mapping keeps its own reference.
        ::close(fd);
        if (addr == MAP_FAILED) {
            size_ = 0;
            throw std::runtime_error("mmap: mmap failed for " + path);
        }
        data_ = static_cast<const uint8_t*>(addr);
        // Access is effectively random across the posting lists. Two competing
        // hints matter for a large index:
        //   - MADV_RANDOM  suppresses wasteful readahead, but also makes the
        //     mapping ineligible for transparent-huge-page collapse, leaving it
        //     on 4 KiB pages. For a multi-GB working set that thrashes the TLB.
        //   - MADV_HUGEPAGE lets the kernel back the region with 2 MiB pages,
        //     cutting TLB entries ~512x on the random-access dot-product path,
        //     at the cost of readahead the huge pages imply.
        // Which wins depends on the access pattern (see AccessPattern), so the
        // default is per index type; NSPARSE_MMAP_ADVISE overrides it with
        // "hugepage" | "random" | "normal" | "hugetlb".
        if (mode == "random") {
            ::madvise(addr, size_, MADV_RANDOM);
        } else if (mode == "normal") {
            // leave kernel defaults
        } else {
#if defined(MADV_HUGEPAGE)
            ::madvise(addr, size_, MADV_HUGEPAGE);
#endif
        }
#endif
    }

    void close() {
#if defined(_WIN32)
        if (data_ != nullptr) {
            UnmapViewOfFile(data_);
        }
        if (mapping_ != nullptr) {
            CloseHandle(mapping_);
        }
        if (file_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_);
        }
        mapping_ = nullptr;
        file_ = INVALID_HANDLE_VALUE;
#else
        if (data_ != nullptr && size_ > 0) {
            ::munmap(const_cast<uint8_t*>(data_), size_);
        }
#endif
        data_ = nullptr;
        size_ = 0;
    }

private:
    void move_from(MmapFile& other) {
        data_ = other.data_;
        size_ = other.size_;
        hugetlb_ = other.hugetlb_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.hugetlb_ = false;
#if defined(_WIN32)
        file_ = other.file_;
        mapping_ = other.mapping_;
        other.file_ = INVALID_HANDLE_VALUE;
        other.mapping_ = nullptr;
#endif
    }

    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    bool hugetlb_ = false;  // data_ is an anonymous MAP_HUGETLB copy, not the file
#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#endif
};

}  // namespace nsparse

#endif  // MMAP_FILE_H
