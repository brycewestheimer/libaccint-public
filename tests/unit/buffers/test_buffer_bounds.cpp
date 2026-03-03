// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_buffer_bounds.cpp
/// @brief Bounds-checking tests for buffer accessors

#include <gtest/gtest.h>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>

namespace libaccint {

// ============================================================================
// Valid Access Tests
// ============================================================================

TEST(BufferBoundsTest, OneElectronValidAccess) {
    OneElectronBuffer<0> buf(3, 4);
    buf.clear();

    // All valid boundary indices
    buf(0, 0) = 1.0;
    buf(0, 3) = 2.0;
    buf(2, 0) = 3.0;
    buf(2, 3) = 4.0;

    EXPECT_DOUBLE_EQ(buf(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(buf(0, 3), 2.0);
    EXPECT_DOUBLE_EQ(buf(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(buf(2, 3), 4.0);
}

TEST(BufferBoundsTest, TwoElectronValidAccess) {
    TwoElectronBuffer<0> buf(2, 3, 4, 5);
    buf.clear();

    // Boundary valid indices
    buf(0, 0, 0, 0) = 1.0;
    buf(1, 2, 3, 4) = 2.0;

    EXPECT_DOUBLE_EQ(buf(0, 0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(buf(1, 2, 3, 4), 2.0);
}

TEST(BufferBoundsTest, TwoElectronGradientValidAccess) {
    TwoElectronBuffer<1> buf(2, 2, 2, 2);
    buf.clear();

    // Valid derivative index boundaries
    buf(0, 0, 0, 0, 0) = 1.0;
    buf(1, 1, 1, 1, 11) = 2.0;

    EXPECT_DOUBLE_EQ(buf(0, 0, 0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(buf(1, 1, 1, 1, 11), 2.0);
}

TEST(BufferBoundsTest, TwoElectronHessianValidAccess) {
    TwoElectronBuffer<2> buf(2, 2, 2, 2);
    buf.clear();

    buf(0, 0, 0, 0, 0) = 1.0;
    buf(1, 1, 1, 1, 77) = 2.0;

    EXPECT_DOUBLE_EQ(buf(0, 0, 0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(buf(1, 1, 1, 1, 77), 2.0);
}

TEST(BufferBoundsTest, OneElectronGradientValidAccess) {
    OneElectronBuffer<1> buf(3, 3);
    buf.clear();

    buf(0, 0, 0) = 1.0;
    buf(2, 2, 5) = 2.0;

    EXPECT_DOUBLE_EQ(buf(0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(buf(2, 2, 5), 2.0);
}

// ============================================================================
// Death Tests (Debug Mode Only)
// ============================================================================

#ifndef NDEBUG

TEST(BufferBoundsDeathTest, OneElectronOutOfBoundsA) {
    OneElectronBuffer<0> buf(2, 2);
    EXPECT_DEATH(buf(5, 0), ".*");
}

TEST(BufferBoundsDeathTest, OneElectronOutOfBoundsB) {
    OneElectronBuffer<0> buf(2, 2);
    EXPECT_DEATH(buf(0, 5), ".*");
}

TEST(BufferBoundsDeathTest, OneElectronNegativeIndex) {
    OneElectronBuffer<0> buf(2, 2);
    EXPECT_DEATH(buf(-1, 0), ".*");
}

TEST(BufferBoundsDeathTest, TwoElectronOutOfBoundsA) {
    TwoElectronBuffer<0> buf(2, 2, 2, 2);
    EXPECT_DEATH(buf(5, 0, 0, 0), ".*");
}

TEST(BufferBoundsDeathTest, TwoElectronOutOfBoundsB) {
    TwoElectronBuffer<0> buf(2, 2, 2, 2);
    EXPECT_DEATH(buf(0, 5, 0, 0), ".*");
}

TEST(BufferBoundsDeathTest, TwoElectronOutOfBoundsC) {
    TwoElectronBuffer<0> buf(2, 2, 2, 2);
    EXPECT_DEATH(buf(0, 0, 5, 0), ".*");
}

TEST(BufferBoundsDeathTest, TwoElectronOutOfBoundsD) {
    TwoElectronBuffer<0> buf(2, 2, 2, 2);
    EXPECT_DEATH(buf(0, 0, 0, 5), ".*");
}

TEST(BufferBoundsDeathTest, TwoElectronGradientOutOfBoundsDeriv) {
    TwoElectronBuffer<1> buf(2, 2, 2, 2);
    EXPECT_DEATH(buf(0, 0, 0, 0, 20), ".*");
}

TEST(BufferBoundsDeathTest, TwoElectronNegativeIndex) {
    TwoElectronBuffer<0> buf(2, 2, 2, 2);
    EXPECT_DEATH(buf(-1, 0, 0, 0), ".*");
}

#endif  // !NDEBUG

}  // namespace libaccint
