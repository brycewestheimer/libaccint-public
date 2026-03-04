// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_three_center_buffer.cpp
/// @brief Unit tests for ThreeCenterBuffer

#include <gtest/gtest.h>
#include <libaccint/buffers/three_center_buffer.hpp>
#include <libaccint/core/types.hpp>

namespace libaccint {

// ============================================================================
// Default Construction
// ============================================================================

TEST(ThreeCenterBufferTest, DefaultConstruction) {
    ThreeCenterBuffer<0> buf;
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.size(), 0);
    EXPECT_EQ(buf.np(), 0);
    EXPECT_EQ(buf.na(), 0);
    EXPECT_EQ(buf.nb(), 0);
    EXPECT_EQ(buf.n_integrals(), 0);
    EXPECT_EQ(buf.data().size(), 0);
}

// ============================================================================
// Sized Construction
// ============================================================================

TEST(ThreeCenterBufferTest, SizedConstruction) {
    ThreeCenterBuffer<0> buf(5, 3, 3);
    EXPECT_FALSE(buf.empty());
    EXPECT_EQ(buf.np(), 5);
    EXPECT_EQ(buf.na(), 3);
    EXPECT_EQ(buf.nb(), 3);
    EXPECT_EQ(buf.n_integrals(), 45);  // 5 * 3 * 3
    EXPECT_EQ(buf.size(), 45);         // N_DERIV=1 for DerivOrder=0
}

TEST(ThreeCenterBufferTest, NonUniformConstruction) {
    ThreeCenterBuffer<0> buf(10, 6, 3);
    EXPECT_EQ(buf.np(), 10);
    EXPECT_EQ(buf.na(), 6);
    EXPECT_EQ(buf.nb(), 3);
    EXPECT_EQ(buf.n_integrals(), 180);  // 10 * 6 * 3
    EXPECT_EQ(buf.size(), 180);
}

// ============================================================================
// Resize
// ============================================================================

TEST(ThreeCenterBufferTest, Resize) {
    ThreeCenterBuffer<0> buf;
    buf.resize(4, 3, 3);

    EXPECT_FALSE(buf.empty());
    EXPECT_EQ(buf.np(), 4);
    EXPECT_EQ(buf.na(), 3);
    EXPECT_EQ(buf.nb(), 3);
    EXPECT_EQ(buf.size(), 36);
}

TEST(ThreeCenterBufferTest, ResizeChangeDimensions) {
    ThreeCenterBuffer<0> buf(2, 2, 2);
    EXPECT_EQ(buf.size(), 8);

    buf.resize(5, 3, 3);
    EXPECT_EQ(buf.np(), 5);
    EXPECT_EQ(buf.na(), 3);
    EXPECT_EQ(buf.nb(), 3);
    EXPECT_EQ(buf.size(), 45);
}

// ============================================================================
// Clear
// ============================================================================

TEST(ThreeCenterBufferTest, Clear) {
    ThreeCenterBuffer<0> buf(3, 2, 2);
    buf(0, 0, 0) = 1.0;
    buf(1, 1, 1) = 2.0;
    buf(2, 0, 1) = 3.0;

    buf.clear();

    for (int p = 0; p < 3; ++p)
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
                EXPECT_DOUBLE_EQ(buf(p, a, b), 0.0);
}

// ============================================================================
// Energy Accessor (DerivOrder=0)
// ============================================================================

TEST(ThreeCenterBufferTest, EnergyAccessor) {
    ThreeCenterBuffer<0> buf(3, 2, 2);
    buf.clear();

    buf(0, 0, 0) = 1.0;
    buf(0, 1, 0) = 2.0;
    buf(1, 0, 1) = 3.0;
    buf(2, 1, 1) = 4.0;

    EXPECT_DOUBLE_EQ(buf(0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(buf(0, 1, 0), 2.0);
    EXPECT_DOUBLE_EQ(buf(1, 0, 1), 3.0);
    EXPECT_DOUBLE_EQ(buf(2, 1, 1), 4.0);
    EXPECT_DOUBLE_EQ(buf(1, 1, 0), 0.0);
}

TEST(ThreeCenterBufferTest, EnergyAccessorConst) {
    ThreeCenterBuffer<0> buf(2, 2, 2);
    buf.clear();
    buf(1, 0, 1) = 3.14;

    const auto& const_buf = buf;
    EXPECT_DOUBLE_EQ(const_buf(1, 0, 1), 3.14);
}

TEST(ThreeCenterBufferTest, LinearIndexComputation) {
    // Verify layout: p * (na*nb) + a * nb + b
    ThreeCenterBuffer<0> buf(4, 3, 5);
    buf.clear();

    buf(2, 1, 3) = 42.0;

    // Expected index = 2 * (3*5) + 1 * 5 + 3 = 30 + 5 + 3 = 38
    EXPECT_DOUBLE_EQ(buf.data()[38], 42.0);
}

// ============================================================================
// Gradient Accessor (DerivOrder=1)
// ============================================================================

TEST(ThreeCenterBufferTest, GradientAccessor) {
    ThreeCenterBuffer<1> buf(3, 2, 2);
    EXPECT_EQ(buf.N_DERIV, 9);  // 3 coords * 3 centers
    EXPECT_EQ(buf.size(), 108);  // 9 * 3 * 2 * 2

    buf.clear();
    buf(0, 0, 0, 0) = 1.0;
    buf(0, 0, 0, 4) = 2.0;
    buf(2, 1, 1, 8) = 3.0;

    EXPECT_DOUBLE_EQ(buf(0, 0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(buf(0, 0, 0, 4), 2.0);
    EXPECT_DOUBLE_EQ(buf(2, 1, 1, 8), 3.0);
    EXPECT_DOUBLE_EQ(buf(1, 0, 0, 0), 0.0);
}

TEST(ThreeCenterBufferTest, GradientAccessorConst) {
    ThreeCenterBuffer<1> buf(2, 2, 2);
    buf.clear();
    buf(1, 0, 1, 5) = 2.71;

    const auto& const_buf = buf;
    EXPECT_DOUBLE_EQ(const_buf(1, 0, 1, 5), 2.71);
}

// ============================================================================
// Hessian (DerivOrder=2)
// ============================================================================

TEST(ThreeCenterBufferTest, HessianNDerivComponents) {
    ThreeCenterBuffer<2> buf;
    EXPECT_EQ(buf.N_DERIV, 45);  // 9 * 10 / 2 = 45
}

TEST(ThreeCenterBufferTest, HessianConstruction) {
    ThreeCenterBuffer<2> buf(2, 2, 2);
    EXPECT_EQ(buf.n_integrals(), 8);
    EXPECT_EQ(buf.size(), 360);  // 45 * 8

    buf.clear();
    buf(0, 0, 0, 0) = 1.0;
    buf(1, 1, 1, 44) = 2.0;

    EXPECT_DOUBLE_EQ(buf(0, 0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(buf(1, 1, 1, 44), 2.0);
}

// ============================================================================
// Data Access
// ============================================================================

TEST(ThreeCenterBufferTest, DataSpan) {
    ThreeCenterBuffer<0> buf(3, 2, 2);
    buf.clear();

    auto span = buf.data();
    EXPECT_EQ(span.size(), 12);

    // Modify through span
    span[0] = 10.0;
    span[11] = 20.0;

    EXPECT_DOUBLE_EQ(buf(0, 0, 0), 10.0);
    EXPECT_DOUBLE_EQ(buf(2, 1, 1), 20.0);
}

TEST(ThreeCenterBufferTest, DataPointer) {
    ThreeCenterBuffer<0> buf(2, 2, 2);
    buf.clear();

    Real* ptr = buf.data_ptr();
    EXPECT_NE(ptr, nullptr);

    ptr[0] = 1.0;
    EXPECT_DOUBLE_EQ(buf(0, 0, 0), 1.0);

    const auto& const_buf = buf;
    const Real* cptr = const_buf.data_ptr();
    EXPECT_EQ(ptr, cptr);
}

TEST(ThreeCenterBufferTest, SizeBytes) {
    ThreeCenterBuffer<0> buf(3, 2, 2);
    EXPECT_EQ(buf.size_bytes(), 12 * sizeof(double));
}

// ============================================================================
// Type Aliases
// ============================================================================

TEST(ThreeCenterBufferTest, TypeAliases) {
    ThreeCenterERIBuffer eri_buf(5, 3, 3);
    EXPECT_TRUE(eri_buf.is_double_precision);
    EXPECT_FALSE(eri_buf.is_single_precision);
    EXPECT_EQ(eri_buf.size(), 45);

    ThreeCenterERIBufferFloat eri_buf_f(5, 3, 3);
    EXPECT_TRUE(eri_buf_f.is_single_precision);
    EXPECT_FALSE(eri_buf_f.is_double_precision);
    EXPECT_EQ(eri_buf_f.size(), 45);

    ThreeCenterERIGradientBuffer eri_grad(5, 3, 3);
    EXPECT_EQ(eri_grad.N_DERIV, 9);
    EXPECT_EQ(eri_grad.size(), 405);  // 9 * 45ints

    ThreeCenterERIGradientBufferFloat eri_grad_f(5, 3, 3);
    EXPECT_TRUE(eri_grad_f.is_single_precision);
}

// ============================================================================
// Precision Conversion
// ============================================================================

TEST(ThreeCenterBufferTest, ToPrecision) {
    ThreeCenterBuffer<0> dbuf(3, 2, 2);
    dbuf.clear();
    dbuf(0, 0, 0) = 1.234567890123456;
    dbuf(2, 1, 1) = -9.87654321;

    auto fbuf = dbuf.to_precision<float>();

    EXPECT_EQ(fbuf.np(), 3);
    EXPECT_EQ(fbuf.na(), 2);
    EXPECT_EQ(fbuf.nb(), 2);
    EXPECT_TRUE(fbuf.is_single_precision);
    EXPECT_NEAR(fbuf(0, 0, 0), 1.234567890123456f, 1e-6f);
    EXPECT_NEAR(fbuf(2, 1, 1), -9.87654321f, 1e-5f);
}

TEST(ThreeCenterBufferTest, CopyFrom) {
    ThreeCenterBuffer<0, float> fbuf(3, 2, 2);
    fbuf.clear();
    fbuf(0, 0, 0) = 1.5f;
    fbuf(2, 1, 1) = -3.0f;

    ThreeCenterBuffer<0> dbuf;
    dbuf.copy_from(fbuf);

    EXPECT_EQ(dbuf.np(), 3);
    EXPECT_EQ(dbuf.na(), 2);
    EXPECT_EQ(dbuf.nb(), 2);
    EXPECT_DOUBLE_EQ(dbuf(0, 0, 0), 1.5);
    EXPECT_DOUBLE_EQ(dbuf(2, 1, 1), -3.0);
}

TEST(ThreeCenterBufferTest, RoundTrip) {
    ThreeCenterBuffer<0, float> fbuf(2, 2, 2);
    fbuf.clear();
    fbuf(0, 0, 0) = 1.5f;
    fbuf(1, 1, 1) = 0.25f;

    auto dbuf = fbuf.to_precision<double>();
    auto fbuf2 = dbuf.to_precision<float>();

    EXPECT_FLOAT_EQ(fbuf(0, 0, 0), fbuf2(0, 0, 0));
    EXPECT_FLOAT_EQ(fbuf(1, 1, 1), fbuf2(1, 1, 1));
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(ThreeCenterBufferTest, SingleElementDimension) {
    ThreeCenterBuffer<0> buf(1, 1, 1);
    EXPECT_EQ(buf.n_integrals(), 1);
    EXPECT_EQ(buf.size(), 1);

    buf.clear();
    buf(0, 0, 0) = 42.0;
    EXPECT_DOUBLE_EQ(buf(0, 0, 0), 42.0);
}

TEST(ThreeCenterBufferTest, DataContiguity) {
    ThreeCenterBuffer<0> buf(2, 2, 3);
    buf.clear();

    Real value = 1.0;
    for (int p = 0; p < 2; ++p)
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 3; ++b)
                buf(p, a, b) = value++;

    // Verify contiguous layout
    auto span = buf.data();
    value = 1.0;
    int idx = 0;
    for (int p = 0; p < 2; ++p)
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 3; ++b) {
                EXPECT_DOUBLE_EQ(span[idx], value)
                    << "Mismatch at index " << idx;
                value++;
                idx++;
            }
}

}  // namespace libaccint
