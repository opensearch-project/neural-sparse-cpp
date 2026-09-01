/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/io/index_io.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "nsparse/brutal_index.h"
#include "nsparse/disk_seismic_index.h"
#include "nsparse/disk_seismic_scalar_quantized_index.h"
#include "nsparse/id_map_index.h"
#include "nsparse/inverted_index.h"
#include "nsparse/io/file_io.h"
#include "nsparse/seismic_index.h"
#include "nsparse/seismic_scalar_quantized_index.h"

namespace nsparse {

namespace {
constexpr uint32_t kBitsPerByte = 8;

constexpr uint32_t BRUT = fourcc(BrutalIndex::name);
constexpr uint32_t SEIS = fourcc(SeismicIndex::name);
constexpr uint32_t SESQ = fourcc(SeismicScalarQuantizedIndex::name);
constexpr uint32_t IDMP = fourcc(IDMapIndex::name);
constexpr uint32_t INVT = fourcc(InvertedIndex::name);
constexpr uint32_t DSEI = fourcc(DiskSeismicIndex::name);
constexpr uint32_t DSSQ = fourcc(DiskSeismicScalarQuantizedIndex::name);

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

// Reads `header`'s payload by borrowing it from the file rather than copying
// it, or returns nullptr for an index type without a mapped reader. `pos` is
// where the payload begins, past the header read_header consumed.
Index* mmap_index_payload(const IndexHeader& header, const char* file_name,
                          size_t pos) {
    switch (header.id) {
        case SEIS:
            return SeismicIndex::mmap_index(header, file_name, pos);
        case SESQ:
            return SeismicScalarQuantizedIndex::mmap_index(header, file_name,
                                                           pos);
        case INVT:
            return InvertedIndex::mmap_index(header, file_name, pos);
        case DSEI:
            return DiskSeismicIndex::mmap_index(header, file_name, pos);
        case DSSQ:
            return DiskSeismicScalarQuantizedIndex::mmap_index(header,
                                                               file_name, pos);
        default:
            return nullptr;
    }
}

// The id as it reads in the file, for error messages: a fourcc is four
// printable characters, and its numeric value is not what a reader would
// recognise.
std::string id_to_string(uint32_t id_val) {
    std::string chars(4, '\0');
    for (size_t i = 0; i < chars.size(); ++i) {
        chars[i] = static_cast<char>((id_val >> (kBitsPerByte * i)) & 0xFFU);
    }
    return chars;
}

IndexHeader read_header(IOReader* io_reader) {
    IndexHeader header;
    io_reader->read(&header.id, sizeof(uint32_t), 1);
    io_reader->read(&header.version, sizeof(uint32_t), 1);
    io_reader->read(&header.dimension, sizeof(int), 1);
    return header;
}

// Constructs the index the id names, still empty: the payload is what
// read_index/mmap_index fill in.
Index* make_index(const IndexHeader& header) {
    switch (header.id) {
        case BRUT:
            return new BrutalIndex(header.dimension);
        case SEIS:
            return new SeismicIndex(header.dimension);
        case SESQ:
            return new SeismicScalarQuantizedIndex(header.dimension);
        case DSEI:
            return new DiskSeismicIndex(header.dimension);
        case DSSQ:
            return new DiskSeismicScalarQuantizedIndex(header.dimension);
        case IDMP:
            return new IDMapIndex();
        case INVT:
            return new InvertedIndex(header.dimension);
        default:
            throw std::runtime_error("Unknown index type");
    }
}

// A version outside 1..supported is one this build cannot lay out: either it
// postdates this binary, or no writer ever produced it. Reading the payload
// anyway would consume whatever the fields happen to align with, so the file is
// rejected here instead.
void throw_if_version_unsupported(const IndexHeader& header,
                                  uint32_t supported) {
    if (header.version == 0 || header.version > supported) {
        throw std::runtime_error("Unsupported " + id_to_string(header.id) +
                                 " index format version " +
                                 std::to_string(header.version) +
                                 "; this build reads versions 1 through " +
                                 std::to_string(supported));
    }
}
}  // namespace

namespace detail {
void write_header(const IndexHeader& header, IOWriter* io_writer) {
    // write index type
    uint32_t id_val = header.id;
    io_writer->write(&id_val, sizeof(uint32_t), 1);
    // write payload layout version
    uint32_t version = header.version;
    io_writer->write(&version, sizeof(uint32_t), 1);
    // write dimension
    int dimension = header.dimension;
    io_writer->write(&dimension, sizeof(int), 1);
}

void write_index(Index* index, IOWriter* io_writer, bool keep_open) {
    auto* index_io = dynamic_cast<IndexIO*>(index);
    StreamCloser closer(io_writer, keep_open);
    if (index_io == nullptr) {
        throw std::runtime_error("Index does not support serialization");
    }
    // write header
    write_header({.id = fourcc(index->id()),
                  .version = index_io->format_version(),
                  .dimension = index->get_dimension()},
                 io_writer);
    // write index customized payload
    index_io->write_index(io_writer);
    closer.close();
}

Index* read_index(IOReader* io_reader, bool keep_open, int io_flags) {
    StreamCloser closer(io_reader, keep_open);
    const IndexHeader header = read_header(io_reader);
    // Held so it does not leak if anything below throws, close() included.
    std::unique_ptr<Index> index(make_index(header));
    auto* index_io = dynamic_cast<IndexIO*>(index.get());
    if (index_io == nullptr) {
        throw std::runtime_error("Index does not support serialization");
    }
    // Ahead of either read below, so a payload this build cannot lay out is
    // never parsed.
    throw_if_version_unsupported(header, index_io->format_version());

    // handle mmap
    if ((io_flags & IndexIoFlag::kUseMmap) == IndexIoFlag::kUseMmap) {
        if (auto* file_io_reader = dynamic_cast<FileIOReader*>(io_reader)) {
            // The mapping takes its own handle, so the reader stays open while
            // it is built. `pos` is where the payload starts, which is what
            // serialize() padded against. An index type without a mapped reader
            // returns null and falls through to the copying read below.
            std::unique_ptr<Index> mapped(mmap_index_payload(
                header, file_io_reader->file_name().c_str(), io_reader->pos()));
            if (mapped != nullptr) {
                index.reset();
                closer.close();
                return mapped.release();
            }
        }
    }

    index_io->read_index(io_reader, header, io_flags);
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
