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

enum IndexIoFlag {
    kUseMmap = 0x0001,
};

namespace detail {
// The fixed prefix every serialized index starts with. Exposed because a writer
// that streams a payload out itself, rather than through an Index, still has to
// lay the header out exactly the way read_header expects — see
// build_seismic_index_batched.
void write_header(const IndexHeader& header, IOWriter* io_writer);
void write_index(Index* index, IOWriter* io_writer, bool keep_open);
// `filename`, when given, lets an index that was written for mmap borrow from
// the file instead of copying; without one the copying path is used, since a
// stream has no file to map.
Index* read_index(IOReader* io_reader, bool keep_open, int io_flags = 0);
}  // namespace detail

void write_index(Index* index, char* file_name);
void write_index(Index* index, IOWriter* io_writer);
Index* read_index(char* file_name, int io_flags = 0);
Index* read_index(IOReader* io_reader, int io_flags = 0);

}  // namespace nsparse

#endif  // INDEX_IO_H
