/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/io/index_io.h"

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "nsparse/brutal_index.h"
#include "nsparse/disk_seismic_index.h"
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
constexpr uint32_t DSEI = fourcc(DiskSeismicIndex::name);

// Closes a stream once, on whichever path leaves the scope.
//
// close() reports a failed flush/fclose by throwing, so it cannot live in a
// destructor alone. The explicit close() forwards that failure; the destructor
// covers unwinding and swallows it, the in-flight exception being the one the
// caller needs. `keep_open` leaves a nested index's stream to its enclosing
// writer.
template <class T>
class StreamCloser {
public:
    StreamCloser(T* stream, bool keep_open)
        : stream_(keep_open ? nullptr : stream) {}

    ~StreamCloser() {
        if (stream_ == nullptr) {
            return;
        }
        try {
            stream_->close();
        } catch (...) {  // NOLINT(bugprone-empty-catch)
        }
    }

    StreamCloser(const StreamCloser&) = delete;
    StreamCloser& operator=(const StreamCloser&) = delete;
    StreamCloser(StreamCloser&&) = delete;
    StreamCloser& operator=(StreamCloser&&) = delete;

    void close() {
        T* stream = stream_;
        stream_ = nullptr;
        if (stream != nullptr) {
            stream->close();
        }
    }

private:
    T* stream_;
};

// Reads `id`'s payload by borrowing it from the file rather than copying it, or
// returns nullptr for an index type without a mapped reader. `pos` is where the
// payload begins, past the header read_header consumed.
Index* mmap_index_payload(uint32_t id, int dimension, const char* file_name,
                          size_t pos) {
    switch (id) {
        case SEIS:
            return SeismicIndex::mmap_index(dimension, file_name, pos);
        case SESQ:
            return SeismicScalarQuantizedIndex::mmap_index(dimension, file_name,
                                                           pos);
        case INVT:
            return InvertedIndex::mmap_index(dimension, file_name, pos);
        case DSEI:
            return DiskSeismicIndex::mmap_index(dimension, file_name, pos);
        default:
            return nullptr;
    }
}

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
        case DSEI:
            return new DiskSeismicIndex(dimension);
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
    StreamCloser closer(io_writer, keep_open);
    if (index_io == nullptr) {
        throw std::runtime_error("Index does not support serialization");
    }
    // write header
    write_header(index, io_writer);
    // write index customized payload
    index_io->write_index(io_writer);
    closer.close();
}

Index* read_index(IOReader* io_reader, bool keep_open, int io_flags) {
    StreamCloser closer(io_reader, keep_open);
    // Held so it does not leak if anything below throws, close() included.
    std::unique_ptr<Index> index(read_header(io_reader));
    auto* index_io = dynamic_cast<IndexIO*>(index.get());
    if (index_io == nullptr) {
        throw std::runtime_error("Index does not support serialization");
    }

    // handle mmap
    if ((io_flags & IndexIoFlag::kUseMmap) == IndexIoFlag::kUseMmap) {
        if (auto* file_io_reader = dynamic_cast<FileIOReader*>(io_reader)) {
            // The mapping takes its own handle, so the reader stays open while
            // it is built. `pos` is where the payload starts, which is what
            // serialize() padded against. An index type without a mapped reader
            // returns null and falls through to the copying read below.
            std::unique_ptr<Index> mapped(mmap_index_payload(
                fourcc(index->id()), index->get_dimension(),
                file_io_reader->file_name().c_str(), io_reader->pos()));
            if (mapped != nullptr) {
                index.reset();
                closer.close();
                return mapped.release();
            }
        }
    }

    index_io->read_index(io_reader, io_flags);
    closer.close();
    return index.release();
}
}  // namespace detail

void write_index(Index* index, IOWriter* io_writer) {
    detail::write_index(index, io_writer, false);
}

void write_index(Index* index, char* file_name) {
    FileIOWriter writer(file_name);
    write_index(index, &writer);
}

Index* read_index(IOReader* io_reader, int io_flags) {
    return detail::read_index(io_reader, false, io_flags);
}

Index* read_index(char* file_name, int io_flags) {
    FileIOReader reader(file_name);
    return read_index(&reader, io_flags);
}
}  // namespace nsparse
