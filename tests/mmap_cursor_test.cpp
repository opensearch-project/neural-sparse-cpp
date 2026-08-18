/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/utils/mmap_cursor.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

// 4-byte aligned, so read_array<uint32_t> reaches the bounds check under test
// rather than failing on its start address first.
std::vector<uint32_t> make_buffer(size_t words) {
    std::vector<uint32_t> buffer(words);
    for (size_t i = 0; i < words; ++i) {
        buffer[i] = static_cast<uint32_t>(i);
    }
    return buffer;
}

nsparse::MmapCursor cursor_over(const std::vector<uint32_t>& buffer) {
    return nsparse::MmapCursor(reinterpret_cast<const uint8_t*>(buffer.data()),
                               buffer.size() * sizeof(uint32_t));
}

TEST(MmapCursor, ReadsScalarsAndArraysInSequence) {
    const auto buffer = make_buffer(8);
    auto cursor = cursor_over(buffer);

    EXPECT_EQ(cursor.read_scalar<uint32_t>(), 0U);
    const uint32_t* array = cursor.read_array<uint32_t>(3);
    ASSERT_NE(array, nullptr);
    EXPECT_EQ(array[0], 1U);
    EXPECT_EQ(array[2], 3U);
    EXPECT_EQ(cursor.pos(), 4 * sizeof(uint32_t));
    EXPECT_EQ(cursor.remaining(), 4 * sizeof(uint32_t));
}

TEST(MmapCursor, RejectsAnArrayLongerThanTheMapping) {
    const auto buffer = make_buffer(4);
    auto cursor = cursor_over(buffer);
    EXPECT_THROW(cursor.read_array<uint32_t>(5), std::runtime_error);
}

// A count whose product with sizeof(T) wraps to a handful of bytes: a bounds
// check on the product passes, and the caller gets a view over ~2^62 elements
// of a 16-byte mapping.
TEST(MmapCursor, RejectsACountThatOverflowsTheByteSize) {
    const auto buffer = make_buffer(4);
    auto cursor = cursor_over(buffer);

    const size_t wraps_to_four =
        (std::numeric_limits<size_t>::max() / sizeof(uint32_t)) + 2;
    ASSERT_EQ(wraps_to_four * sizeof(uint32_t), 4U)
        << "the count under test must still wrap to a passing byte size";

    EXPECT_THROW(cursor.read_array<uint32_t>(wraps_to_four),
                 std::runtime_error);
    EXPECT_EQ(cursor.pos(), 0U) << "a rejected read must not advance";
}

TEST(MmapCursor, RejectsASkipThatOverflowsThePosition) {
    const auto buffer = make_buffer(4);
    auto cursor = cursor_over(buffer);
    cursor.read_scalar<uint32_t>();

    // pos_ + bytes wraps to 0, which is not greater than size_.
    EXPECT_THROW(cursor.skip(std::numeric_limits<size_t>::max() -
                             (sizeof(uint32_t) - 1)),
                 std::runtime_error);
}

TEST(MmapCursor, RejectsAScalarPastTheEnd) {
    const auto buffer = make_buffer(1);
    auto cursor = cursor_over(buffer);
    cursor.read_scalar<uint32_t>();
    EXPECT_THROW(cursor.read_scalar<uint32_t>(), std::runtime_error);
}

TEST(MmapCursor, RejectsAMisalignedArrayStart) {
    const auto buffer = make_buffer(4);
    auto cursor = cursor_over(buffer);
    cursor.skip(1);
    EXPECT_THROW(cursor.read_array<uint32_t>(1), std::runtime_error);
}

TEST(MmapCursor, AnEmptyArrayIsNullAndConsumesNothing) {
    const auto buffer = make_buffer(4);
    auto cursor = cursor_over(buffer);
    EXPECT_EQ(cursor.read_array<uint32_t>(0), nullptr);
    EXPECT_EQ(cursor.pos(), 0U);
}

TEST(MmapCursor, SeekMovesForwardAndBackward) {
    const auto buffer = make_buffer(8);  // 32 bytes, words 0..7
    auto cursor = cursor_over(buffer);
    cursor.read_scalar<uint32_t>();
    EXPECT_EQ(cursor.pos(), 4U);

    cursor.seek(16);  // word 4
    EXPECT_EQ(cursor.read_scalar<uint32_t>(), 4U);
    cursor.seek(0);  // back to the start
    EXPECT_EQ(cursor.read_scalar<uint32_t>(), 0U);

    cursor.seek(buffer.size() * sizeof(uint32_t));  // one-past-end is legal
    EXPECT_EQ(cursor.remaining(), 0U);
}

TEST(MmapCursor, SeekPastEndThrows) {
    const auto buffer = make_buffer(4);
    auto cursor = cursor_over(buffer);
    EXPECT_THROW(cursor.seek(buffer.size() * sizeof(uint32_t) + 1),
                 std::runtime_error);
}

TEST(MmapCursor, SubcursorReadsWithinItsWindow) {
    const auto buffer = make_buffer(8);  // 32 bytes, words 0..7
    auto cursor = cursor_over(buffer);

    auto sub = cursor.subcursor(16, 8);  // words 4 and 5
    EXPECT_EQ(sub.read_scalar<uint32_t>(), 4U);
    EXPECT_EQ(sub.read_scalar<uint32_t>(), 5U);
    EXPECT_THROW(sub.read_scalar<uint32_t>(),
                 std::runtime_error);  // past window
    EXPECT_EQ(cursor.pos(), 0U);       // parent cursor untouched
}

TEST(MmapCursor, SubcursorRejectsAnOutOfRangeWindow) {
    const auto buffer = make_buffer(4);  // 16 bytes
    auto cursor = cursor_over(buffer);
    EXPECT_THROW(cursor.subcursor(12, 8), std::runtime_error);  // 12 + 8 > 16
    EXPECT_THROW(cursor.subcursor(17, 0), std::runtime_error);  // offset > size
    // Overflow-safe: len / offset near SIZE_MAX must not wrap past the guard.
    EXPECT_THROW(cursor.subcursor(8, std::numeric_limits<size_t>::max()),
                 std::runtime_error);
    EXPECT_THROW(cursor.subcursor(std::numeric_limits<size_t>::max(), 8),
                 std::runtime_error);
    EXPECT_NO_THROW(cursor.subcursor(16, 0));  // empty window at the end
}

TEST(MmapCursor, CurrentPointsAtThePosition) {
    const auto buffer = make_buffer(4);
    auto cursor = cursor_over(buffer);
    const auto* base = reinterpret_cast<const uint8_t*>(buffer.data());
    const size_t size = buffer.size() * sizeof(uint32_t);
    EXPECT_EQ(cursor.current(), base);
    cursor.read_scalar<uint32_t>();
    EXPECT_EQ(cursor.current(), base + sizeof(uint32_t));
    cursor.seek(size);  // one-past-end pointer is formed, never dereferenced
    EXPECT_EQ(cursor.current(), base + size);
}

}  // namespace
