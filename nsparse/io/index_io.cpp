/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/io/index_io.h"

#include <stdexcept>

#include "nsparse/brutal_index.h"
#include "nsparse/id_map_index.h"
#include "nsparse/inverted_index.h"
#include "nsparse/io/file_io.h"
#include "nsparse/seismic_index.h"
#include "nsparse/seismic_scalar_quantized_index.h"

namespace nsparse {

namespace {
constexpr uint32_t BRUT = fourcc(BrutalIndex::name);
constexpr uint32_t SEIS = fourcc(SeismicIndex::name);
constexpr uint32_t SESQ = fourcc(SeismicScalarQuantizedIndex::name);
constexpr uint32_t IDMP = fourcc(IDMapIndex::name);
constexpr uint32_t INVT = fourcc(InvertedIndex::name);

void write_header(Index* index, IOWriter* io_writer) {
    // write index type
    auto id_val = fourcc(index->id());
    io_writer->write(&id_val, sizeof(uint32_t), 1);
    // write dimension
    auto dimension = index->get_dimension();
    io_writer->write(&dimension, sizeof(int), 1);
}

Index* read_header(IOReader* io_reader) {
    uint32_t id_val = 0;
    io_reader->read(&id_val, sizeof(uint32_t), 1);
    int dimension = 0;
    io_reader->read(&dimension, sizeof(int), 1);
    switch (id_val) {
        case BRUT:
            return new BrutalIndex(dimension);
        case SEIS:
            return new SeismicIndex(dimension);
        case SESQ:
            return new SeismicScalarQuantizedIndex(dimension);
        case IDMP:
            return new IDMapIndex();
        case INVT:
            return new InvertedIndex(dimension);
        default:
            throw std::runtime_error("Unknown index type");
    }
}
}  // namespace

namespace detail {
void write_index(Index* index, IOWriter* io_writer, bool keep_open) {
    auto* index_io = dynamic_cast<IndexIO*>(index);
    if (index_io == nullptr) {
        throw std::runtime_error("Index does not support serialization");
    }
    // write header
    write_header(index, io_writer);
    // write index customized payload
    index_io->write_index(io_writer);
    if (!keep_open) {
        io_writer->close();
    }
}

Index* read_index(IOReader* io_reader, bool keep_open) {
    Index* index = read_header(io_reader);
    auto* index_io = dynamic_cast<IndexIO*>(index);
    if (index_io == nullptr) {
        throw std::runtime_error("Index does not support serialization");
    }
    index_io->read_index(io_reader);
    if (!keep_open) {
        io_reader->close();
    }
    return index;
}
}  // namespace detail

void write_index(Index* index, IOWriter* io_writer) {
    detail::write_index(index, io_writer, false);
}

void write_index(Index* index, char* filename) {
    FileIOWriter writer(filename);
    write_index(index, &writer);
}

Index* read_index(IOReader* io_reader) {
    return detail::read_index(io_reader, false);
}

Index* read_index(char* filename) {
    FileIOReader reader(filename);
    return read_index(&reader);
}
}  // namespace nsparse
