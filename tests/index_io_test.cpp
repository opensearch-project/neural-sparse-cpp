/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/io/index_io.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "nsparse/brutal_index.h"
#include "nsparse/id_map_index.h"
#include "nsparse/index.h"
#include "nsparse/inverted_index.h"
#include "nsparse/io/buffered_io.h"
#include "nsparse/io/io.h"
#include "nsparse/seismic_index.h"
#include "nsparse/seismic_scalar_quantized_index.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"

namespace {

// IOWriter that throws if write() is called after close().
// Simulates real file I/O where writing to a closed stream is invalid.
class StrictBufferedIOWriter : public nsparse::IOWriter {
public:
    void write(void* ptr, size_t size, size_t nitems) override {
        if (closed_) {
            throw std::runtime_error(
                "Write after close: stream already closed");
        }
        delegate_.write(ptr, size, nitems);
    }

    size_t pos() const override { return delegate_.pos(); }

    void close() override { closed_ = true; }

    const std::vector<uint8_t>& data() const { return delegate_.data(); }
    size_t size() const { return delegate_.size(); }

private:
    nsparse::BufferedIOWriter delegate_;
    bool closed_ = false;
};

// IOReader that throws if read() is called after close().
// Simulates real file I/O where reading from a closed stream is invalid.
class StrictBufferedIOReader : public nsparse::IOReader {
public:
    explicit StrictBufferedIOReader(const std::vector<uint8_t>& data)
        : delegate_(data) {}

    size_t read(void* ptr, size_t size, size_t nitems) override {
        if (closed_) {
            throw std::runtime_error("Read after close: stream already closed");
        }
        return delegate_.read(ptr, size, nitems);
    }

    size_t pos() const override { return delegate_.pos(); }

    void close() override { closed_ = true; }

private:
    nsparse::BufferedIOReader delegate_;
    bool closed_ = false;
};

// Mock Index that implements both Index and IndexIO for testing
class MockIndex : public nsparse::Index, public nsparse::IndexIO {
public:
    static constexpr std::array<char, 4> name = {'M', 'O', 'C', 'K'};

    explicit MockIndex(int dim = 0) : Index(dim) {}

    std::array<char, 4> id() const override { return name; }

    void add(nsparse::idx_t /*n*/, const nsparse::idx_t* /*indptr*/,
             const nsparse::term_t* /*indices*/,
             const float* /*values*/) override {}

    const nsparse::SparseVectors* get_vectors() const override {
        return nullptr;
    }

    // IndexIO implementation
    void write_index(nsparse::IOWriter* io_writer) override {
        io_writer->write(&test_data_, sizeof(int), 1);
        size_t size = test_string_.size();
        io_writer->write(&size, sizeof(size_t), 1);
        io_writer->write(test_string_.data(), sizeof(char), size);
    }

    void read_index(nsparse::IOReader* io_reader, int /*io_flags*/) override {
        io_reader->read(&test_data_, sizeof(int), 1);
        size_t size = 0;
        io_reader->read(&size, sizeof(size_t), 1);
        test_string_.resize(size);
        io_reader->read(test_string_.data(), sizeof(char), size);
    }

    void set_test_data(int data) { test_data_ = data; }
    int get_test_data() const { return test_data_; }

    void set_test_string(const std::string& str) { test_string_ = str; }
    const std::string& get_test_string() const { return test_string_; }

private:
    int test_data_ = 0;
    std::string test_string_;
};

}  // namespace

// Test write_index with BufferedIOWriter
TEST(IndexIO, WriteIndexBasic) {
    MockIndex index(128);
    index.set_test_data(42);
    index.set_test_string("hello");

    nsparse::BufferedIOWriter writer;
    nsparse::write_index(&index, &writer);

    // Verify something was written
    ASSERT_GT(writer.size(), 0);

    // Verify header: first 4 bytes should be the fourcc
    const auto& data = writer.data();
    uint32_t written_fourcc = 0;
    std::memcpy(&written_fourcc, data.data(), sizeof(uint32_t));
    ASSERT_EQ(written_fourcc, nsparse::fourcc(MockIndex::name));

    // Verify dimension is written after fourcc
    int written_dim = 0;
    std::memcpy(&written_dim, data.data() + sizeof(uint32_t), sizeof(int));
    ASSERT_EQ(written_dim, 128);
}

// Test write_index throws for non-IndexIO index
TEST(IndexIO, WriteIndexThrowsForNonIndexIO) {
    // Create a minimal Index that doesn't implement IndexIO
    class NonSerializableIndex : public nsparse::Index {
    public:
        NonSerializableIndex() : Index(10) {}
        std::array<char, 4> id() const override { return {'N', 'O', 'I', 'O'}; }
        void add(nsparse::idx_t, const nsparse::idx_t*, const nsparse::term_t*,
                 const float*) override {}
        const nsparse::SparseVectors* get_vectors() const override {
            return nullptr;
        }
    };

    NonSerializableIndex index;
    nsparse::BufferedIOWriter writer;

    ASSERT_THROW(nsparse::write_index(&index, &writer), std::runtime_error);
}

// Test read_index throws for unknown index type
TEST(IndexIO, ReadIndexThrowsForUnknownType) {
    // Create a buffer with an unknown fourcc
    std::vector<uint8_t> buffer(sizeof(uint32_t) + sizeof(int));
    uint32_t unknown_fourcc = 0xDEADBEEF;
    int dimension = 64;
    std::memcpy(buffer.data(), &unknown_fourcc, sizeof(uint32_t));
    std::memcpy(buffer.data() + sizeof(uint32_t), &dimension, sizeof(int));

    nsparse::BufferedIOReader reader(buffer);
    ASSERT_THROW(nsparse::read_index(&reader), std::runtime_error);
}

// Test roundtrip with SeismicIndex (real index that supports serialization)
TEST(IndexIO, RoundtripSeismicIndex) {
    // Create and write a SeismicIndex
    auto* original = new nsparse::SeismicIndex(256);

    nsparse::BufferedIOWriter writer;
    nsparse::write_index(original, &writer);

    // Read it back
    nsparse::BufferedIOReader reader(writer.data());
    nsparse::Index* loaded = nsparse::read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->get_dimension(), 256);
    ASSERT_EQ(loaded->id(), original->id());

    delete original;
    delete loaded;
}

// Test write_index throws for BrutalIndex (doesn't implement IndexIO)
TEST(IndexIO, WriteIndexThrowsForBrutalIndex) {
    auto* index = new nsparse::BrutalIndex(512);

    nsparse::BufferedIOWriter writer;
    ASSERT_THROW(nsparse::write_index(index, &writer), std::runtime_error);

    delete index;
}

// Test roundtrip with SeismicScalarQuantizedIndex
TEST(IndexIO, RoundtripSeismicScalarQuantizedIndex) {
    auto* original = new nsparse::SeismicScalarQuantizedIndex(1024);

    nsparse::BufferedIOWriter writer;
    nsparse::write_index(original, &writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::Index* loaded = nsparse::read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->get_dimension(), 1024);
    ASSERT_EQ(loaded->id(), original->id());

    delete original;
    delete loaded;
}

// Test write_index with empty writer
TEST(IndexIO, WriteIndexEmptyWriter) {
    auto* index = new nsparse::SeismicIndex(64);

    nsparse::BufferedIOWriter writer;
    ASSERT_EQ(writer.size(), 0);

    nsparse::write_index(index, &writer);
    ASSERT_GT(writer.size(), 0);

    delete index;
}

// Test read_index with small dimension
TEST(IndexIO, ReadIndexSmallDimension) {
    // Use a real SeismicIndex to create a valid buffer
    auto* original = new nsparse::SeismicIndex(32);

    nsparse::BufferedIOWriter writer;
    nsparse::write_index(original, &writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::Index* loaded = nsparse::read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->get_dimension(), 32);

    delete original;
    delete loaded;
}

// Test BufferedIOWriter and BufferedIOReader work correctly together
TEST(IndexIO, BufferedIOWriterReaderIntegration) {
    nsparse::BufferedIOWriter writer;

    // Write various data types
    int int_val = 12345;
    float float_val = 3.14159F;
    std::vector<uint8_t> bytes = {1, 2, 3, 4, 5};

    writer.write(&int_val, sizeof(int), 1);
    writer.write(&float_val, sizeof(float), 1);
    writer.write(bytes.data(), sizeof(uint8_t), bytes.size());

    // Read back
    nsparse::BufferedIOReader reader(writer.data());

    int read_int = 0;
    float read_float = 0.0F;
    std::vector<uint8_t> read_bytes(5);

    reader.read(&read_int, sizeof(int), 1);
    reader.read(&read_float, sizeof(float), 1);
    reader.read(read_bytes.data(), sizeof(uint8_t), 5);

    ASSERT_EQ(read_int, int_val);
    ASSERT_FLOAT_EQ(read_float, float_val);
    ASSERT_EQ(read_bytes, bytes);
}

// Test multiple write/read cycles
TEST(IndexIO, MultipleWriteReadCycles) {
    for (int i = 0; i < 3; ++i) {
        auto* index = new nsparse::SeismicIndex(64 * (i + 1));

        nsparse::BufferedIOWriter writer;
        nsparse::write_index(index, &writer);

        nsparse::BufferedIOReader reader(writer.data());
        nsparse::Index* loaded = nsparse::read_index(&reader);

        ASSERT_EQ(loaded->get_dimension(), 64 * (i + 1));

        delete index;
        delete loaded;
    }
}

// Test roundtrip with IDMapIndex wrapping SeismicIndex
TEST(IndexIO, RoundtripIDMapIndex) {
    auto* seismic = new nsparse::SeismicIndex(128);
    auto* original = new nsparse::IDMapIndex(seismic);

    nsparse::BufferedIOWriter writer;
    nsparse::write_index(original, &writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::Index* loaded = nsparse::read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->id(), original->id());
    ASSERT_EQ(loaded->num_vectors(), 0);

    delete original;
    delete loaded;
}

// Test roundtrip with IDMapIndex with data
TEST(IndexIO, RoundtripIDMapIndexWithData) {
    auto* seismic = new nsparse::SeismicIndex(128);
    auto* original = new nsparse::IDMapIndex(seismic);

    // Add some vectors with custom IDs
    std::vector<nsparse::idx_t> indptr = {0, 2, 4};
    std::vector<nsparse::term_t> indices = {0, 1, 2, 3};
    std::vector<float> values = {1.0F, 0.5F, 0.8F, 0.3F};
    std::vector<nsparse::idx_t> ids = {100, 200};
    original->add_with_ids(2, indptr.data(), indices.data(), values.data(),
                           ids.data());

    nsparse::BufferedIOWriter writer;
    nsparse::write_index(original, &writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::Index* loaded = nsparse::read_index(&reader);

    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->id(), original->id());
    ASSERT_EQ(loaded->num_vectors(), 2);

    delete original;
    delete loaded;
}

// Verify StrictBufferedIOWriter throws on write after close
TEST(IndexIO, StrictWriterThrowsAfterClose) {
    StrictBufferedIOWriter writer;
    int val = 42;
    writer.write(&val, sizeof(int), 1);
    writer.close();
    ASSERT_THROW(writer.write(&val, sizeof(int), 1), std::runtime_error);
}

// FileIO*::close() reports a failed flush/fclose by throwing. Closing from a
// destructor would turn that into a terminate, so the close has to sit on the
// success path where the exception can propagate.
TEST(IndexIO, WriteIndexPropagatesACloseFailure) {
    class ThrowingCloseWriter : public StrictBufferedIOWriter {
    public:
        void close() override {
            StrictBufferedIOWriter::close();
            throw std::runtime_error("Failed to close file");
        }
    };

    auto* index = new nsparse::SeismicIndex(8);
    ThrowingCloseWriter writer;
    ASSERT_THROW(nsparse::write_index(index, &writer), std::runtime_error);
    delete index;
}

TEST(IndexIO, ReadIndexPropagatesACloseFailure) {
    class ThrowingCloseReader : public StrictBufferedIOReader {
    public:
        explicit ThrowingCloseReader(const std::vector<uint8_t>& data)
            : StrictBufferedIOReader(data) {}

        void close() override {
            StrictBufferedIOReader::close();
            throw std::runtime_error("Failed to close file");
        }
    };

    auto* original = new nsparse::SeismicIndex(8);
    StrictBufferedIOWriter writer;
    nsparse::write_index(original, &writer);
    delete original;

    ThrowingCloseReader reader(writer.data());
    ASSERT_THROW(nsparse::read_index(&reader), std::runtime_error);
}

// Verify StrictBufferedIOReader throws on read after close
TEST(IndexIO, StrictReaderThrowsAfterClose) {
    std::vector<uint8_t> buf(sizeof(int), 0);
    StrictBufferedIOReader reader(buf);
    int val = 0;
    reader.read(&val, sizeof(int), 1);
    reader.close();
    ASSERT_THROW(reader.read(&val, sizeof(int), 1), std::runtime_error);
}

// IDMapIndex wrapping SeismicIndex: write then read with strict IO.
// Before the keep_open fix, the delegate's write_index closed the stream out
// from under IDMapIndex, causing a write-after-close crash.
TEST(IndexIO, StrictIO_RoundtripIDMapSeismicIndex) {
    auto* seismic = new nsparse::SeismicIndex(128);
    auto* original = new nsparse::IDMapIndex(seismic);

    std::vector<nsparse::idx_t> indptr = {0, 2, 4};
    std::vector<nsparse::term_t> indices = {0, 1, 2, 3};
    std::vector<float> values = {1.0F, 0.5F, 0.8F, 0.3F};
    std::vector<nsparse::idx_t> ids = {100, 200};
    original->add_with_ids(2, indptr.data(), indices.data(), values.data(),
                           ids.data());

    StrictBufferedIOWriter writer;
    ASSERT_NO_THROW(nsparse::write_index(original, &writer));

    StrictBufferedIOReader reader(writer.data());
    nsparse::Index* loaded = nullptr;
    ASSERT_NO_THROW(loaded = nsparse::read_index(&reader));

    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->id(), original->id());
    ASSERT_EQ(loaded->num_vectors(), 2);

    delete original;
    delete loaded;
}

// IDMapIndex wrapping InvertedIndex: the original segfault scenario.
// InvertedIndex has a non-trivial write_index/read_index, so the stream must
// stay open for the delegate to be serialized after IDMapIndex's id map.
TEST(IndexIO, StrictIO_RoundtripIDMapInvertedIndex) {
    auto* inverted = new nsparse::InvertedIndex(128);
    auto* original = new nsparse::IDMapIndex(inverted);

    std::vector<nsparse::idx_t> indptr = {0, 2, 4};
    std::vector<nsparse::term_t> indices = {0, 1, 2, 3};
    std::vector<float> values = {1.0F, 0.5F, 0.8F, 0.3F};
    std::vector<nsparse::idx_t> ids = {100, 200};
    original->add_with_ids(2, indptr.data(), indices.data(), values.data(),
                           ids.data());
    original->build();

    StrictBufferedIOWriter writer;
    ASSERT_NO_THROW(nsparse::write_index(original, &writer));

    StrictBufferedIOReader reader(writer.data());
    nsparse::Index* loaded = nullptr;
    ASSERT_NO_THROW(loaded = nsparse::read_index(&reader));

    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->id(), original->id());
    ASSERT_EQ(loaded->num_vectors(), 2);

    delete original;
    delete loaded;
}

// A document whose terms were all pruned -- here, one with none to begin with
// -- leaves no posting entry behind, so the count cannot be recovered from the
// lists and has to be serialized. Trailing, because an interior gap is still
// bracketed by the ids around it.
TEST(IndexIO, RoundtripInvertedIndexCountsATrailingEmptyDocument) {
    nsparse::InvertedIndex original(128);

    std::vector<nsparse::idx_t> indptr = {0, 2, 2};
    std::vector<nsparse::term_t> indices = {0, 1};
    std::vector<float> values = {1.0F, 0.5F};
    original.add(2, indptr.data(), indices.data(), values.data());
    original.build();
    ASSERT_EQ(original.num_vectors(), 2);

    nsparse::BufferedIOWriter writer;
    nsparse::write_index(&original, &writer);
    nsparse::BufferedIOReader reader(writer.data());
    std::unique_ptr<nsparse::Index> loaded(nsparse::read_index(&reader));

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->num_vectors(), 2);
}

// The INVT payload has no version to check, so a file written before it carried
// the document count reads with every field one slot early. The element width
// is what gives that away: write_index only ever writes U32, and the value
// landing there instead is the first posting list's size.
TEST(IndexIO, ReadIndexRejectsAnInvertedIndexFileWithoutTheDocumentCount) {
    nsparse::InvertedIndex original(128);

    std::vector<nsparse::idx_t> indptr = {0, 2, 4};
    std::vector<nsparse::term_t> indices = {0, 1, 2, 3};
    std::vector<float> values = {1.0F, 0.5F, 0.8F, 0.3F};
    original.add(2, indptr.data(), indices.data(), values.data());
    original.build();

    nsparse::BufferedIOWriter writer;
    nsparse::write_index(&original, &writer);

    // The header is a fourcc and a dimension; the count is the payload's first
    // field, so dropping it is exactly the old layout.
    std::vector<uint8_t> stale = writer.data();
    const size_t header = sizeof(uint32_t) + sizeof(int);
    stale.erase(stale.begin() + header,
                stale.begin() + header + sizeof(size_t));

    nsparse::BufferedIOReader reader(stale);
    EXPECT_THROW(nsparse::read_index(&reader), std::runtime_error);
}

// IDMapIndex wrapping SeismicScalarQuantizedIndex with strict IO
TEST(IndexIO, StrictIO_RoundtripIDMapSeismicSQIndex) {
    auto* sq_index = new nsparse::SeismicScalarQuantizedIndex(128);
    auto* original = new nsparse::IDMapIndex(sq_index);

    std::vector<nsparse::idx_t> indptr = {0, 2, 4};
    std::vector<nsparse::term_t> indices = {0, 1, 2, 3};
    std::vector<float> values = {1.0F, 0.5F, 0.8F, 0.3F};
    std::vector<nsparse::idx_t> ids = {100, 200};
    original->add_with_ids(2, indptr.data(), indices.data(), values.data(),
                           ids.data());

    StrictBufferedIOWriter writer;
    ASSERT_NO_THROW(nsparse::write_index(original, &writer));

    StrictBufferedIOReader reader(writer.data());
    nsparse::Index* loaded = nullptr;
    ASSERT_NO_THROW(loaded = nsparse::read_index(&reader));

    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->id(), original->id());
    ASSERT_EQ(loaded->num_vectors(), 2);

    delete original;
    delete loaded;
}

namespace {
// Index file removed on destruction. read_index/write_index take char*, not
// const char*, hence the owned string.
class TempIndexFile {
public:
    explicit TempIndexFile(const std::string& name)
        : path_(std::filesystem::temp_directory_path() / name) {
        std::filesystem::remove(path_);
    }
    ~TempIndexFile() { std::filesystem::remove(path_); }

    TempIndexFile(const TempIndexFile&) = delete;
    TempIndexFile& operator=(const TempIndexFile&) = delete;

    char* c_str() { return path_str_.data(); }

private:
    std::filesystem::path path_{};
    std::string path_str_ = path_.string();
};
}  // namespace

// IDMapIndex has no mapped reader of its own, so kUseMmap has to fall back to
// the copying read for the id map while still reaching the delegate -- here an
// InvertedIndex, whose posting lists then borrow from the file.
TEST(IndexIO, UseMmapFlagMapsTheInvertedIndexDelegateOfAnIDMap) {
    TempIndexFile file("nsparse_index_io_inverted_mmap_flag.idx");
    auto* inverted = new nsparse::InvertedIndex(128);
    nsparse::IDMapIndex original(inverted);

    std::vector<nsparse::idx_t> indptr = {0, 2, 4};
    std::vector<nsparse::term_t> indices = {0, 1, 2, 3};
    std::vector<float> values = {1.0F, 0.5F, 0.8F, 0.3F};
    std::vector<nsparse::idx_t> ids = {100, 200};
    original.add_with_ids(2, indptr.data(), indices.data(), values.data(),
                          ids.data());
    original.build();
    nsparse::write_index(&original, file.c_str());

    std::unique_ptr<nsparse::Index> loaded(
        nsparse::read_index(file.c_str(), nsparse::IndexIoFlag::kUseMmap));

    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->id(), original.id());
    ASSERT_EQ(loaded->num_vectors(), 2);

    // The id map still round-trips: search returns external ids, not internal.
    std::vector<nsparse::idx_t> q_indptr = {0, 1};
    std::vector<nsparse::term_t> q_indices = {0};
    std::vector<float> q_values = {1.0F};
    std::vector<nsparse::idx_t> labels(1, nsparse::detail::INVALID_IDX);
    std::vector<float> distances(1, -1.0F);
    loaded->search(1, q_indptr.data(), q_indices.data(), q_values.data(), 1,
                   distances.data(), labels.data());
    EXPECT_EQ(labels[0], 100);
}

// The id map is serialized ahead of the delegate so that a mapped delegate read
// -- which consumes the rest of the file and hands back a fresh index -- has
// nothing left behind it to parse. Mapping is only reachable if read_index
// forwards its io_flags down to the wrapped index.
TEST(IndexIO, UseMmapFlagReachesTheIDMapDelegate) {
    TempIndexFile file("nsparse_index_io_idmap_mmap.idx");
    auto* seismic =
        new nsparse::SeismicIndex(5, {.lambda = 10, .beta = 2, .alpha = 0.5F});
    nsparse::IDMapIndex original(seismic);

    std::vector<nsparse::idx_t> indptr = {0, 2, 4};
    std::vector<nsparse::term_t> indices = {0, 1, 2, 3};
    std::vector<float> values = {1.0F, 0.5F, 0.8F, 0.3F};
    std::vector<nsparse::idx_t> ids = {100, 200};
    original.add_with_ids(2, indptr.data(), indices.data(), values.data(),
                          ids.data());
    original.build();
    nsparse::write_index(&original, file.c_str());

    std::unique_ptr<nsparse::Index> loaded(
        nsparse::read_index(file.c_str(), nsparse::IndexIoFlag::kUseMmap));

    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->id(), original.id());
    ASSERT_EQ(loaded->num_vectors(), 2);

    // A borrowed CSR keeps the file's contiguity: indices sit right after
    // indptr, which holds for the mapping but not for separate allocations.
    const auto* vectors = loaded->get_vectors();
    ASSERT_NE(vectors, nullptr);
    const auto* indptr_bytes =
        reinterpret_cast<const uint8_t*>(vectors->indptr_data());
    const auto* indices_bytes =
        reinterpret_cast<const uint8_t*>(vectors->indices_data());
    EXPECT_EQ(indices_bytes - indptr_bytes,
              static_cast<ptrdiff_t>((vectors->num_vectors() + 1) *
                                     sizeof(nsparse::idx_t)));

    // The id map still round-trips: search returns external ids, not internal.
    std::vector<nsparse::idx_t> q_indptr = {0, 1};
    std::vector<nsparse::term_t> q_indices = {0};
    std::vector<float> q_values = {1.0F};
    std::vector<nsparse::idx_t> labels(1, nsparse::detail::INVALID_IDX);
    std::vector<float> distances(1, -1.0F);
    loaded->search(1, q_indptr.data(), q_indices.data(), q_values.data(), 1,
                   distances.data(), labels.data());
    EXPECT_EQ(labels[0], 100);
}

// Same forwarding, with a quantized delegate: its payload opens with the
// quantizer header, so the mapped reader has to consume that before the
// vectors.
TEST(IndexIO, UseMmapFlagReachesTheIDMapQuantizedDelegate) {
    TempIndexFile file("nsparse_index_io_idmap_sq_mmap.idx");
    auto* seismic_sq = new nsparse::SeismicScalarQuantizedIndex(
        nsparse::QuantizerType::QT_8bit, 0.0F, 1.0F,
        {.lambda = 10, .beta = 2, .alpha = 0.5F}, 5);
    nsparse::IDMapIndex original(seismic_sq);

    std::vector<nsparse::idx_t> indptr = {0, 2, 4};
    std::vector<nsparse::term_t> indices = {0, 1, 2, 3};
    std::vector<float> values = {1.0F, 0.5F, 0.8F, 0.3F};
    std::vector<nsparse::idx_t> ids = {100, 200};
    original.add_with_ids(2, indptr.data(), indices.data(), values.data(),
                          ids.data());
    original.build();
    nsparse::write_index(&original, file.c_str());

    std::unique_ptr<nsparse::Index> loaded(
        nsparse::read_index(file.c_str(), nsparse::IndexIoFlag::kUseMmap));

    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->id(), original.id());
    ASSERT_EQ(loaded->num_vectors(), 2);

    const auto* vectors = loaded->get_vectors();
    ASSERT_NE(vectors, nullptr);
    // Borrowed, not copied: indices sit right after indptr in the file.
    const auto* indptr_bytes =
        reinterpret_cast<const uint8_t*>(vectors->indptr_data());
    const auto* indices_bytes =
        reinterpret_cast<const uint8_t*>(vectors->indices_data());
    EXPECT_EQ(indices_bytes - indptr_bytes,
              static_cast<ptrdiff_t>((vectors->num_vectors() + 1) *
                                     sizeof(nsparse::idx_t)));
    // The quantizer header survived the mapped read, so the codes are 1 byte
    // wide rather than being strided as floats.
    EXPECT_EQ(vectors->get_element_size(), 1);

    std::vector<nsparse::idx_t> q_indptr = {0, 1};
    std::vector<nsparse::term_t> q_indices = {0};
    std::vector<float> q_values = {1.0F};
    std::vector<nsparse::idx_t> labels(1, nsparse::detail::INVALID_IDX);
    std::vector<float> distances(1, -1.0F);
    nsparse::SeismicSearchParameters params(5, 1.0F);
    loaded->search(1, q_indptr.data(), q_indices.data(), q_values.data(), 1,
                   distances.data(), labels.data(), &params);
    EXPECT_EQ(labels[0], 100);
}
