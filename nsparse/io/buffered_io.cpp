/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/io/buffered_io.h"

#include <algorithm>
#include <cstring>

namespace nsparse {

void BufferedIOWriter::write(void* ptr, size_t size, size_t nitems) {
    size_t total_bytes = size * nitems;
    const uint8_t* src = static_cast<const uint8_t*>(ptr);
    buffer_.insert(buffer_.end(), src, src + total_bytes);
}

BufferedIOReader::BufferedIOReader(const std::vector<uint8_t>& data)
    : data_(data.data()), size_(data.size()), pos_(0) {}

BufferedIOReader::BufferedIOReader(const uint8_t* data, size_t size)
    : data_(data), size_(size), pos_(0) {}

size_t BufferedIOReader::read(void* ptr, size_t size, size_t nitems) {
    size_t total_bytes = size * nitems;
    size_t available = size_ - pos_;
    size_t to_read = std::min(total_bytes, available);

    if (to_read > 0) {
        std::memcpy(ptr, data_ + pos_, to_read);
        pos_ += to_read;
    }

    return to_read / size;  // Return number of items read
}

}  // namespace nsparse
