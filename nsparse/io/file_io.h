/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef FILE_IO_H
#define FILE_IO_H

#include <cstdio>

#include "nsparse/io/index_io.h"

namespace nsparse {
class FileIOReader : public IOReader {
public:
    explicit FileIOReader(char* filename);
    explicit FileIOReader(FILE* file);
    ~FileIOReader();

    size_t read(void* ptr, size_t size, size_t nitems) override;
    void close() override;

private:
    FILE* file_;
};

class FileIOWriter : public IOWriter {
public:
    explicit FileIOWriter(char* filename);
    explicit FileIOWriter(FILE* file);
    ~FileIOWriter();

    void write(void* ptr, size_t size, size_t nitems) override;
    void close()
        override;  // Call explicitly if you need error handling on close

private:
    FILE* file_;
};
}  // namespace nsparse

#endif  // FILE_IO_H