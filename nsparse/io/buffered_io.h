/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef BUFFER_IO_H
#define BUFFER_IO_H

#include <cstdint>
#include <vector>

#include "nsparse/io/io.h"

namespace nsparse {

class BufferedIOWriter : public IOWriter {
public:
    BufferedIOWriter() = default;

    void write(void* ptr, size_t size, size_t nitems) override;

    // Get the written data
    const std::vector<uint8_t>& data() const { return buffer_; }

    // Move the buffer out (transfers ownership)
    std::vector<uint8_t> release() { return std::move(buffer_); }

    size_t size() const { return buffer_.size(); }

    void clear() { buffer_.clear(); }

private:
    std::vector<uint8_t> buffer_;
};

class BufferedIOReader : public IOReader {
public:
    // Read from existing buffer (does not copy)
    explicit BufferedIOReader(const std::vector<uint8_t>& data);

    // Read from raw pointer
    BufferedIOReader(const uint8_t* data, size_t size);

    size_t read(void* ptr, size_t size, size_t nitems) override;

    size_t remaining() const { return size_ - pos_; }

    void reset() { pos_ = 0; }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;
};

}  // namespace nsparse

#endif  // BUFFER_IO_H