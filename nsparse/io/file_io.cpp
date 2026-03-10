/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/io/file_io.h"

#include <stdexcept>

#include "nsparse/io/index_io.h"

namespace nsparse {

FileIOReader::FileIOReader(char* filename) : file_(nullptr) {
    file_ = fopen(filename, "rb");
    if (file_ == nullptr) {
        throw std::runtime_error("Failed to open file for reading");
    }
}

FileIOReader::FileIOReader(FILE* file) : file_(file) {}

FileIOReader::~FileIOReader() {
    if (file_ != nullptr) {
        fclose(file_);
    }
}

void FileIOReader::close() {
    if (file_ != nullptr) {
        if (fclose(file_) != 0) {
            file_ = nullptr;
            throw std::runtime_error("Failed to close file");
        }
        file_ = nullptr;
    }
}

size_t FileIOReader::read(void* ptr, size_t size, size_t nitems) {
    return fread(ptr, size, nitems, file_);
}

FileIOWriter::FileIOWriter(char* filename) : file_(nullptr) {
    file_ = fopen(filename, "wb");
    if (file_ == nullptr) {
        throw std::runtime_error("Failed to open file for writing");
    }
}

FileIOWriter::FileIOWriter(FILE* file) : file_(file) {}

FileIOWriter::~FileIOWriter() {
    if (file_ != nullptr) {
        fclose(file_);  // Ignore errors in destructor
    }
}

void FileIOWriter::close() {
    if (file_ != nullptr) {
        if (fclose(file_) != 0) {
            file_ = nullptr;
            throw std::runtime_error("Failed to close file");
        }
        file_ = nullptr;
    }
}

void FileIOWriter::write(void* ptr, size_t size, size_t nitems) {
    size_t written = fwrite(ptr, size, nitems, file_);
    if (written != nitems) {
        throw std::runtime_error("Failed to write to file");
    }
}
}  // namespace nsparse