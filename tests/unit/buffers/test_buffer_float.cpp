// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_buffer_float.cpp
/// @brief Float-precision tests for OneElectronBuffer and TwoElectronBuffer

#include <gtest/gtest.h>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>

namespace libaccint {

// ============================================================================
// OneElectronBuffer<0, float> Tests
// ============================================================================

TEST(OneElectronBufferFloatTest, Construction) {
    OneElectronBuffer<0, float> buf(3, 3);
    EXPECT_FALSE(buf.empty());
    EXPECT_EQ(buf.na(), 3);
    EXPECT_EQ(buf.nb(), 3);
    EXPECT_EQ(buf.size(), 9);
    EXPECT_TRUE(buf.is_single_precision);
    EXPECT_FALSE(buf.is_double_precision);
}

TEST(OneElectronBufferFloatTest, DefaultConstruction) {
    OneElectronBuffer<0, float> buf;
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.size(), 0);
    EXPECT_EQ(buf.na(), 0);
    EXPECT_EQ(buf.nb(), 0);
}

TEST(OneElectronBufferFloatTest, Accessors) {
    OneElectronBuffer<0, float> buf(3, 3);
    buf.clear();

    buf(0, 0) = 1.0f;
    buf(1, 1) = 2.5f;
    buf(2, 2) = 3.14f;

    EXPECT_FLOAT_EQ(buf(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(buf(1, 1), 2.5f);
    EXPECT_FLOAT_EQ(buf(2, 2), 3.14f);
    EXPECT_FLOAT_EQ(buf(0, 1), 0.0f);
}

TEST(OneElectronBufferFloatTest, ConstAccessor) {
    OneElectronBuffer<0, float> buf(2, 2);
    buf.clear();
    buf(0, 1) = 7.5f;

    const auto& const_buf = buf;
    EXPECT_FLOAT_EQ(const_buf(0, 1), 7.5f);
}

TEST(OneElectronBufferFloatTest, Clear) {
    OneElectronBuffer<0, float> buf(2, 2);
    buf(0, 0) = 1.0f;
    buf(1, 1) = 2.0f;
    buf.clear();

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            EXPECT_FLOAT_EQ(buf(a, b), 0.0f);
        }
    }
}

TEST(OneElectronBufferFloatTest, DataSpan) {
    OneElectronBuffer<0, float> buf(2, 3);
    buf.clear();
    buf(1, 2) = 5.0f;

    auto span = buf.data();
    EXPECT_EQ(span.size(), 6);
    EXPECT_FLOAT_EQ(span[5], 5.0f);  // (1, 2) => index 1*3+2=5
}

TEST(OneElectronBufferFloatTest, SizeBytes) {
    OneElectronBuffer<0, float> buf(3, 3);
    EXPECT_EQ(buf.size_bytes(), 9 * sizeof(float));
}

TEST(OneElectronBufferFloatTest, GradientFloat) {
    OneElectronBuffer<1, float> buf(3, 3);
    EXPECT_EQ(buf.N_DERIV, 6);
    EXPECT_EQ(buf.size(), 54);  // 6 * 3 * 3
    EXPECT_TRUE(buf.is_single_precision);

    buf.clear();
    buf(0, 0, 0) = 1.5f;
    buf(2, 2, 5) = 9.9f;

    EXPECT_FLOAT_EQ(buf(0, 0, 0), 1.5f);
    EXPECT_FLOAT_EQ(buf(2, 2, 5), 9.9f);
    EXPECT_FLOAT_EQ(buf(1, 1, 3), 0.0f);
}

TEST(OneElectronBufferFloatTest, TypeAliases) {
    OverlapBufferFloat obuf(2, 2);
    KineticBufferFloat kbuf(2, 2);
    NuclearBufferFloat nbuf(2, 2);

    EXPECT_TRUE(obuf.is_single_precision);
    EXPECT_TRUE(kbuf.is_single_precision);
    EXPECT_TRUE(nbuf.is_single_precision);
    EXPECT_EQ(obuf.size(), 4);
}

// ============================================================================
// TwoElectronBuffer<0, float> Tests
// ============================================================================

TEST(TwoElectronBufferFloatTest, Construction) {
    TwoElectronBuffer<0, float> buf(2, 2, 2, 2);
    EXPECT_FALSE(buf.empty());
    EXPECT_EQ(buf.na(), 2);
    EXPECT_EQ(buf.nb(), 2);
    EXPECT_EQ(buf.nc(), 2);
    EXPECT_EQ(buf.nd(), 2);
    EXPECT_EQ(buf.n_integrals(), 16);
    EXPECT_EQ(buf.size(), 16);
    EXPECT_TRUE(buf.is_single_precision);
}

TEST(TwoElectronBufferFloatTest, DefaultConstruction) {
    TwoElectronBuffer<0, float> buf;
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.size(), 0);
}

TEST(TwoElectronBufferFloatTest, Accessors) {
    TwoElectronBuffer<0, float> buf(2, 2, 2, 2);
    buf.clear();

    buf(0, 0, 0, 0) = 1.0f;
    buf(1, 1, 1, 1) = 2.5f;
    buf(0, 1, 0, 1) = 3.14f;

    EXPECT_FLOAT_EQ(buf(0, 0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(buf(1, 1, 1, 1), 2.5f);
    EXPECT_FLOAT_EQ(buf(0, 1, 0, 1), 3.14f);
    EXPECT_FLOAT_EQ(buf(0, 0, 0, 1), 0.0f);
}

TEST(TwoElectronBufferFloatTest, ConstAccessor) {
    TwoElectronBuffer<0, float> buf(2, 2, 2, 2);
    buf.clear();
    buf(1, 0, 1, 0) = 4.5f;

    const auto& const_buf = buf;
    EXPECT_FLOAT_EQ(const_buf(1, 0, 1, 0), 4.5f);
}

TEST(TwoElectronBufferFloatTest, Clear) {
    TwoElectronBuffer<0, float> buf(2, 2, 2, 2);
    buf(0, 0, 0, 0) = 1.0f;
    buf(1, 1, 1, 1) = 2.0f;
    buf.clear();

    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
            for (int c = 0; c < 2; ++c)
                for (int d = 0; d < 2; ++d)
                    EXPECT_FLOAT_EQ(buf(a, b, c, d), 0.0f);
}

TEST(TwoElectronBufferFloatTest, SizeBytes) {
    TwoElectronBuffer<0, float> buf(2, 2, 2, 2);
    EXPECT_EQ(buf.size_bytes(), 16 * sizeof(float));
}

TEST(TwoElectronBufferFloatTest, GradientFloat) {
    TwoElectronBuffer<1, float> buf(2, 2, 2, 2);
    EXPECT_EQ(buf.N_DERIV, 12);
    EXPECT_EQ(buf.size(), 192);  // 12 * 2*2*2*2
    EXPECT_TRUE(buf.is_single_precision);

    buf.clear();
    buf(0, 0, 0, 0, 0) = 1.5f;
    buf(1, 1, 1, 1, 11) = 9.9f;

    EXPECT_FLOAT_EQ(buf(0, 0, 0, 0, 0), 1.5f);
    EXPECT_FLOAT_EQ(buf(1, 1, 1, 1, 11), 9.9f);
}

TEST(TwoElectronBufferFloatTest, TypeAliases) {
    ERIBufferFloat ebuf(2, 2, 2, 2);
    ERIGradientBufferFloat gbuf(2, 2, 2, 2);

    EXPECT_TRUE(ebuf.is_single_precision);
    EXPECT_TRUE(gbuf.is_single_precision);
    EXPECT_EQ(ebuf.size(), 16);
    EXPECT_EQ(gbuf.size(), 192);
}

}  // namespace libaccint
