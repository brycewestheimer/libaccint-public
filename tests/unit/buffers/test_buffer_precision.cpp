// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_buffer_precision.cpp
/// @brief Precision conversion tests for OneElectronBuffer and TwoElectronBuffer

#include <gtest/gtest.h>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <cmath>

namespace libaccint {

// ============================================================================
// OneElectronBuffer Precision Conversion
// ============================================================================

TEST(BufferPrecisionTest, OneElectronDoubleTruncation) {
    OneElectronBuffer<0> buf(2, 2);
    buf.clear();
    // Use a value with more precision than float can hold
    buf(0, 0) = 1.0 + 1e-15;
    buf(0, 1) = 1234567.890123456;
    buf(1, 0) = -0.000123456789012345;
    buf(1, 1) = 3.141592653589793;

    auto fbuf = buf.to_precision<float>();

    EXPECT_EQ(fbuf.na(), 2);
    EXPECT_EQ(fbuf.nb(), 2);
    EXPECT_TRUE(fbuf.is_single_precision);

    // Float can't represent the 1e-15 offset from 1.0
    EXPECT_NEAR(fbuf(0, 0), 1.0f, 1e-6f);
    // Float truncates precision
    EXPECT_NEAR(fbuf(1, 1), 3.14159265f, 1e-6f);
}

TEST(BufferPrecisionTest, OneElectronRoundTrip) {
    // float -> double -> float should give same result
    OneElectronBuffer<0, float> fbuf(2, 2);
    fbuf.clear();
    fbuf(0, 0) = 1.5f;
    fbuf(0, 1) = 2.25f;
    fbuf(1, 0) = -3.75f;
    fbuf(1, 1) = 0.125f;

    auto dbuf = fbuf.to_precision<double>();
    auto fbuf2 = dbuf.to_precision<float>();

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            EXPECT_FLOAT_EQ(fbuf(a, b), fbuf2(a, b))
                << "Round-trip mismatch at (" << a << "," << b << ")";
        }
    }
}

TEST(BufferPrecisionTest, OneElectronGradientConversion) {
    OneElectronBuffer<1> dbuf(2, 2);
    dbuf.clear();
    dbuf(0, 0, 0) = 1.0;
    dbuf(0, 1, 2) = 2.5;
    dbuf(1, 0, 4) = -3.0;
    dbuf(1, 1, 5) = 0.001;

    auto fbuf = dbuf.to_precision<float>();

    EXPECT_EQ(fbuf.na(), 2);
    EXPECT_EQ(fbuf.nb(), 2);
    EXPECT_EQ(fbuf.N_DERIV, 6);
    EXPECT_TRUE(fbuf.is_single_precision);

    EXPECT_NEAR(fbuf(0, 0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(fbuf(0, 1, 2), 2.5f, 1e-6f);
    EXPECT_NEAR(fbuf(1, 0, 4), -3.0f, 1e-6f);
    EXPECT_NEAR(fbuf(1, 1, 5), 0.001f, 1e-6f);
}

TEST(BufferPrecisionTest, OneElectronHessianConversion) {
    OneElectronBuffer<2> dbuf(2, 2);
    dbuf.clear();
    dbuf(0, 0, 0) = 1.0;
    dbuf(1, 1, 20) = 99.0;

    auto fbuf = dbuf.to_precision<float>();

    EXPECT_EQ(fbuf.N_DERIV, 21);
    EXPECT_NEAR(fbuf(0, 0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(fbuf(1, 1, 20), 99.0f, 1e-6f);
}

// ============================================================================
// TwoElectronBuffer Precision Conversion
// ============================================================================

TEST(BufferPrecisionTest, TwoElectronDoubleTruncation) {
    TwoElectronBuffer<0> buf(2, 2, 2, 2);
    buf.clear();
    buf(0, 0, 0, 0) = 1.0 + 1e-15;
    buf(1, 1, 1, 1) = 3.141592653589793;

    auto fbuf = buf.to_precision<float>();

    EXPECT_EQ(fbuf.na(), 2);
    EXPECT_EQ(fbuf.nb(), 2);
    EXPECT_EQ(fbuf.nc(), 2);
    EXPECT_EQ(fbuf.nd(), 2);
    EXPECT_NEAR(fbuf(0, 0, 0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(fbuf(1, 1, 1, 1), 3.14159265f, 1e-6f);
}

TEST(BufferPrecisionTest, TwoElectronRoundTrip) {
    TwoElectronBuffer<0, float> fbuf(2, 2, 2, 2);
    fbuf.clear();
    fbuf(0, 0, 0, 0) = 1.5f;
    fbuf(1, 0, 1, 0) = -2.25f;
    fbuf(0, 1, 0, 1) = 0.125f;
    fbuf(1, 1, 1, 1) = 7.75f;

    auto dbuf = fbuf.to_precision<double>();
    auto fbuf2 = dbuf.to_precision<float>();

    EXPECT_FLOAT_EQ(fbuf(0, 0, 0, 0), fbuf2(0, 0, 0, 0));
    EXPECT_FLOAT_EQ(fbuf(1, 0, 1, 0), fbuf2(1, 0, 1, 0));
    EXPECT_FLOAT_EQ(fbuf(0, 1, 0, 1), fbuf2(0, 1, 0, 1));
    EXPECT_FLOAT_EQ(fbuf(1, 1, 1, 1), fbuf2(1, 1, 1, 1));
}

TEST(BufferPrecisionTest, TwoElectronCopyFrom) {
    TwoElectronBuffer<0> dbuf(2, 2, 2, 2);
    dbuf.clear();
    dbuf(0, 0, 0, 0) = 1.0;
    dbuf(1, 1, 1, 1) = 2.0;

    TwoElectronBuffer<0, float> fbuf;
    fbuf.copy_from(dbuf);

    EXPECT_EQ(fbuf.na(), 2);
    EXPECT_EQ(fbuf.nb(), 2);
    EXPECT_EQ(fbuf.nc(), 2);
    EXPECT_EQ(fbuf.nd(), 2);
    EXPECT_FLOAT_EQ(fbuf(0, 0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(fbuf(1, 1, 1, 1), 2.0f);
}

TEST(BufferPrecisionTest, OneElectronCopyFrom) {
    OneElectronBuffer<0, float> fbuf(3, 3);
    fbuf.clear();
    fbuf(0, 0) = 1.5f;
    fbuf(2, 2) = 9.0f;

    OneElectronBuffer<0> dbuf;
    dbuf.copy_from(fbuf);

    EXPECT_EQ(dbuf.na(), 3);
    EXPECT_EQ(dbuf.nb(), 3);
    EXPECT_DOUBLE_EQ(dbuf(0, 0), 1.5);
    EXPECT_DOUBLE_EQ(dbuf(2, 2), 9.0);
}

TEST(BufferPrecisionTest, TwoElectronGradientConversion) {
    TwoElectronBuffer<1> dbuf(2, 2, 2, 2);
    dbuf.clear();
    dbuf(0, 0, 0, 0, 0) = 1.0;
    dbuf(1, 1, 1, 1, 11) = -5.0;

    auto fbuf = dbuf.to_precision<float>();

    EXPECT_EQ(fbuf.N_DERIV, 12);
    EXPECT_NEAR(fbuf(0, 0, 0, 0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(fbuf(1, 1, 1, 1, 11), -5.0f, 1e-6f);
}

}  // namespace libaccint
