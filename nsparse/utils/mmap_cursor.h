/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef MMAP_CURSOR_H
#define MMAP_CURSOR_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

// Split from mmap_file.h, which needs <sys/mman.h> or <windows.h>: a cursor is
// just bounds-checked arithmetic over bytes, so it carries no platform
// dependency. Keeping it separate lets io.h declare MmapSerializable against it
// without pulling the OS mapping headers into every translation unit.
namespace nsparse {

// Sequential cursor over a memory-mapped buffer. Scalars are copied out (small,
// alignment-safe via memcpy) while arrays are returned as pointers directly
// into the mapping, so the bulk index data is never copied. Every access is
// bounds-checked against the mapped size to reject a truncated/corrupt file.
class MmapCursor {
public:
    MmapCursor(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    template <class T>
    T read_scalar() {
        ensure(sizeof(T));
        T value;
        std::memcpy(&value, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return value;
    }

    // Returns a pointer to `n` contiguous T within the mapping and advances the
    // cursor past them. The returned pointer is only valid for the lifetime of
    // the underlying MmapFile.
    //
    // Rejects a start the T cannot be loaded from -- misaligned loads are UB on
    // x86 and fault on ARM -- so a file whose arrays were written without
    // alignment padding fails here rather than at the first dereference.
    // Compared against the address, not the offset: a file mapping starts page
    // aligned, but the (data, size) constructor accepts any buffer.
    template <class T>
    const T* read_array(size_t n) {
        if (n == 0) {
            return nullptr;
        }
        // Divided rather than ensure(n * sizeof(T)): `n` comes from the file,
        // and a wrapped product passes the bounds check.
        if (n > remaining() / sizeof(T)) {
            throw std::runtime_error("mmap: unexpected end of index file");
        }
        const uint8_t* start = data_ + pos_;
        if (reinterpret_cast<uintptr_t>(start) % alignof(T) != 0) {
            throw std::runtime_error(
                "mmap: array is misaligned for its element type");
        }
        pos_ += n * sizeof(T);
        return reinterpret_cast<const T*>(start);
    }

    // Advance the cursor by `bytes` without returning them (e.g. to skip
    // alignment padding). Bounds-checked like the reads.
    void skip(size_t bytes) {
        ensure(bytes);
        pos_ += bytes;
    }

    // Pointer at the current position, non-mutating: lets a caller record where
    // a borrowed section begins before consuming it (e.g. a length-prefixed
    // component borrows [current(), current() + section_len) then skip()s it).
    [[nodiscard]] const uint8_t* current() const { return data_ + pos_; }

    [[nodiscard]] size_t pos() const { return pos_; }
    [[nodiscard]] size_t remaining() const { return size_ - pos_; }

private:
    // Subtraction, not `pos_ + bytes > size_`: `bytes` can come from the file
    // and the sum would wrap. pos_ <= size_ always, so this cannot underflow.
    void ensure(size_t bytes) const {
        if (bytes > size_ - pos_) {
            throw std::runtime_error("mmap: unexpected end of index file");
        }
    }

    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;
};

}  // namespace nsparse

#endif  // MMAP_CURSOR_H
