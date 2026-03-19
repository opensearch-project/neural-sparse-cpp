/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef INDEX_IO_H
#define INDEX_IO_H

#include "nsparse/index.h"
#include "nsparse/io/io.h"
namespace nsparse {

namespace detail {
void write_index(Index* index, IOWriter* io_writer, bool keep_open);
Index* read_index(IOReader* io_reader, bool keep_open);
}  // namespace detail

void write_index(Index* index, char* filename);
void write_index(Index* index, IOWriter* io_writer);
Index* read_index(char* filename);
Index* read_index(IOReader* io_reader);

}  // namespace nsparse

#endif  // INDEX_IO_H