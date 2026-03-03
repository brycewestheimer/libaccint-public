// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_one_electron_buffer.cpp
/// @brief Unit tests for OneElectronBuffer

#include <gtest/gtest.h>
#include <libaccint/buffers/one_electron_buffer.hpp>

namespace libaccint {

// ============================================================================
// Test Default Construction
// ============================================================================

TEST(OneElectronBufferTest, DefaultConstruction) {
    OneElectronBuffer<0> buf;
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.size(), 0);
    EXPECT_EQ(buf.na(), 0);
    EXPECT_EQ(buf.nb(), 0);
    EXPECT_EQ(buf.data().size(), 0);
}

TEST(OneElectronBufferTest, DefaultConstructionGradient) {
    OneElectronBuffer<1> buf;
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.size(), 0);
    EXPECT_EQ(buf.na(), 0);
    EXPECT_EQ(buf.nb(), 0);
}

// ============================================================================
// Test Resize
// ============================================================================

TEST(OneElectronBufferTest, ResizeEnergy) {
    OneElectronBuffer<0> buf;
    buf.resize(3, 3);

    EXPECT_FALSE(buf.empty());
    EXPECT_EQ(buf.na(), 3);
    EXPECT_EQ(buf.nb(), 3);
    EXPECT_EQ(buf.size(), 9);  // 3 * 3 = 9
}

TEST(OneElectronBufferTest, ResizeGradient) {
    OneElectronBuffer<1> buf;
    buf.resize(3, 3);

    EXPECT_FALSE(buf.empty());
    EXPECT_EQ(buf.na(), 3);
    EXPECT_EQ(buf.nb(), 3);
    EXPECT_EQ(buf.size(), 54);  // 6 * 3 * 3 = 54 (6 derivative components)
    EXPECT_EQ(OneElectronBuffer<1>::N_DERIV, 6);
}

TEST(OneElectronBufferTest, ResizeHessian) {
    OneElectronBuffer<2> buf;
    buf.resize(2, 2);

    EXPECT_FALSE(buf.empty());
    EXPECT_EQ(buf.na(), 2);
    EXPECT_EQ(buf.nb(), 2);
    EXPECT_EQ(buf.size(), 84);  // 21 * 2 * 2 = 84 (21 derivative components)
    EXPECT_EQ(OneElectronBuffer<2>::N_DERIV, 21);
}

TEST(OneElectronBufferTest, ResizeNonSquare) {
    OneElectronBuffer<0> buf;
    buf.resize(2, 5);

    EXPECT_EQ(buf.na(), 2);
    EXPECT_EQ(buf.nb(), 5);
    EXPECT_EQ(buf.size(), 10);  // 2 * 5 = 10
}

TEST(OneElectronBufferTest, ResizeChangeDimensions) {
    OneElectronBuffer<0> buf;
    buf.resize(3, 3);
    EXPECT_EQ(buf.size(), 9);

    buf.resize(2, 4);
    EXPECT_EQ(buf.na(), 2);
    EXPECT_EQ(buf.nb(), 4);
    EXPECT_EQ(buf.size(), 8);  // 2 * 4 = 8
}

// ============================================================================
// Test Clear
// ============================================================================

TEST(OneElectronBufferTest, ClearEnergy) {
    OneElectronBuffer<0> buf;
    buf.resize(3, 3);

    // Set some values
    buf(0, 0) = 1.0;
    buf(1, 1) = 2.0;
    buf(2, 2) = 3.0;

    // Clear
    buf.clear();

    // Verify all zeros
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            EXPECT_DOUBLE_EQ(buf(a, b), 0.0);
        }
    }
}

TEST(OneElectronBufferTest, ClearGradient) {
    OneElectronBuffer<1> buf;
    buf.resize(2, 2);

    // Set some derivative values
    buf(0, 0, 0) = 1.0;
    buf(1, 1, 3) = 2.0;

    // Clear
    buf.clear();

    // Verify all zeros
    for (int deriv = 0; deriv < 6; ++deriv) {
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                EXPECT_DOUBLE_EQ(buf(a, b, deriv), 0.0);
            }
        }
    }
}

// ============================================================================
// Test Energy Accessors (DerivOrder = 0)
// ============================================================================

TEST(OneElectronBufferTest, EnergyAccessor) {
    OneElectronBuffer<0> buf;
    buf.resize(3, 3);
    buf.clear();

    // Test setting and getting values
    buf(0, 0) = 1.0;
    buf(0, 1) = 2.0;
    buf(1, 0) = 3.0;
    buf(2, 2) = 4.0;

    EXPECT_DOUBLE_EQ(buf(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(buf(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(buf(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(buf(2, 2), 4.0);
    EXPECT_DOUBLE_EQ(buf(1, 1), 0.0);
}

TEST(OneElectronBufferTest, EnergyAccessorNonSquare) {
    OneElectronBuffer<0> buf;
    buf.resize(2, 4);
    buf.clear();

    // Test all positions
    buf(0, 0) = 1.0;
    buf(0, 3) = 2.0;
    buf(1, 0) = 3.0;
    buf(1, 3) = 4.0;

    EXPECT_DOUBLE_EQ(buf(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(buf(0, 3), 2.0);
    EXPECT_DOUBLE_EQ(buf(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(buf(1, 3), 4.0);
}

TEST(OneElectronBufferTest, EnergyAccessorConst) {
    OneElectronBuffer<0> buf;
    buf.resize(2, 2);
    buf.clear();
    buf(0, 1) = 3.14;

    const auto& const_buf = buf;
    EXPECT_DOUBLE_EQ(const_buf(0, 1), 3.14);
}

// ============================================================================
// Test Gradient Accessors (DerivOrder = 1)
// ============================================================================

TEST(OneElectronBufferTest, GradientAccessor) {
    OneElectronBuffer<1> buf;
    buf.resize(2, 2);
    buf.clear();

    // Set derivative components
    // deriv 0-2: dAx, dAy, dAz
    // deriv 3-5: dBx, dBy, dBz
    buf(0, 0, 0) = 1.0;  // dAx
    buf(0, 0, 1) = 2.0;  // dAy
    buf(0, 0, 2) = 3.0;  // dAz
    buf(1, 1, 3) = 4.0;  // dBx
    buf(1, 1, 4) = 5.0;  // dBy
    buf(1, 1, 5) = 6.0;  // dBz

    EXPECT_DOUBLE_EQ(buf(0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(buf(0, 0, 1), 2.0);
    EXPECT_DOUBLE_EQ(buf(0, 0, 2), 3.0);
    EXPECT_DOUBLE_EQ(buf(1, 1, 3), 4.0);
    EXPECT_DOUBLE_EQ(buf(1, 1, 4), 5.0);
    EXPECT_DOUBLE_EQ(buf(1, 1, 5), 6.0);
}

TEST(OneElectronBufferTest, GradientAccessorConst) {
    OneElectronBuffer<1> buf;
    buf.resize(2, 2);
    buf.clear();
    buf(1, 0, 2) = 2.71;

    const auto& const_buf = buf;
    EXPECT_DOUBLE_EQ(const_buf(1, 0, 2), 2.71);
}

// ============================================================================
// Test Hessian Accessors (DerivOrder = 2)
// ============================================================================

TEST(OneElectronBufferTest, HessianAccessor) {
    OneElectronBuffer<2> buf;
    buf.resize(2, 2);
    buf.clear();

    // Set some hessian components
    buf(0, 0, 0) = 1.0;
    buf(0, 1, 5) = 2.0;
    buf(1, 0, 10) = 3.0;
    buf(1, 1, 20) = 4.0;

    EXPECT_DOUBLE_EQ(buf(0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(buf(0, 1, 5), 2.0);
    EXPECT_DOUBLE_EQ(buf(1, 0, 10), 3.0);
    EXPECT_DOUBLE_EQ(buf(1, 1, 20), 4.0);
    EXPECT_DOUBLE_EQ(buf(0, 0, 1), 0.0);
}

// ============================================================================
// Test Data Access
// ============================================================================

TEST(OneElectronBufferTest, DataSpan) {
    OneElectronBuffer<0> buf;
    buf.resize(3, 3);
    buf.clear();

    auto span = buf.data();
    EXPECT_EQ(span.size(), 9);

    // Modify through span
    span[0] = 1.0;
    span[4] = 2.0;  // Middle element (1, 1)
    span[8] = 3.0;  // Last element (2, 2)

    EXPECT_DOUBLE_EQ(buf(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(buf(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(buf(2, 2), 3.0);
}

TEST(OneElectronBufferTest, DataSpanConst) {
    OneElectronBuffer<0> buf;
    buf.resize(2, 2);
    buf.clear();
    buf(0, 1) = 5.0;

    const auto& const_buf = buf;
    auto span = const_buf.data();
    EXPECT_EQ(span.size(), 4);
    EXPECT_DOUBLE_EQ(span[1], 5.0);  // (0, 1) maps to index 1
}

TEST(OneElectronBufferTest, DataPointer) {
    OneElectronBuffer<0> buf;
    buf.resize(2, 2);
    buf.clear();

    Real* ptr = buf.data_ptr();
    EXPECT_NE(ptr, nullptr);

    // Modify through pointer
    ptr[0] = 1.0;
    ptr[3] = 2.0;

    EXPECT_DOUBLE_EQ(buf(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(buf(1, 1), 2.0);
}

TEST(OneElectronBufferTest, DataPointerConst) {
    OneElectronBuffer<0> buf;
    buf.resize(2, 2);
    buf.clear();
    buf(1, 0) = 7.0;

    const auto& const_buf = buf;
    const Real* ptr = const_buf.data_ptr();
    EXPECT_NE(ptr, nullptr);
    EXPECT_DOUBLE_EQ(ptr[2], 7.0);  // (1, 0) maps to index 2
}

// ============================================================================
// Test Memory Layout
// ============================================================================

TEST(OneElectronBufferTest, MemoryLayoutEnergy) {
    OneElectronBuffer<0> buf;
    buf.resize(3, 2);
    buf.clear();

    // Verify row-major layout: index = a * nb + b
    buf(0, 0) = 0.0;
    buf(0, 1) = 1.0;
    buf(1, 0) = 2.0;
    buf(1, 1) = 3.0;
    buf(2, 0) = 4.0;
    buf(2, 1) = 5.0;

    auto span = buf.data();
    EXPECT_DOUBLE_EQ(span[0], 0.0);  // (0, 0)
    EXPECT_DOUBLE_EQ(span[1], 1.0);  // (0, 1)
    EXPECT_DOUBLE_EQ(span[2], 2.0);  // (1, 0)
    EXPECT_DOUBLE_EQ(span[3], 3.0);  // (1, 1)
    EXPECT_DOUBLE_EQ(span[4], 4.0);  // (2, 0)
    EXPECT_DOUBLE_EQ(span[5], 5.0);  // (2, 1)
}

TEST(OneElectronBufferTest, MemoryLayoutGradient) {
    OneElectronBuffer<1> buf;
    buf.resize(2, 2);
    buf.clear();

    // Verify layout: deriv * (na * nb) + a * nb + b
    buf(0, 0, 0) = 0.0;
    buf(0, 1, 0) = 1.0;
    buf(1, 0, 0) = 2.0;
    buf(1, 1, 0) = 3.0;
    buf(0, 0, 1) = 4.0;

    auto span = buf.data();
    EXPECT_DOUBLE_EQ(span[0], 0.0);  // deriv=0, (0, 0)
    EXPECT_DOUBLE_EQ(span[1], 1.0);  // deriv=0, (0, 1)
    EXPECT_DOUBLE_EQ(span[2], 2.0);  // deriv=0, (1, 0)
    EXPECT_DOUBLE_EQ(span[3], 3.0);  // deriv=0, (1, 1)
    EXPECT_DOUBLE_EQ(span[4], 4.0);  // deriv=1, (0, 0)
}

// ============================================================================
// Test Type Aliases
// ============================================================================

TEST(OneElectronBufferTest, TypeAliases) {
    // Test that type aliases compile and work correctly
    OverlapBuffer overlap_buf;
    overlap_buf.resize(2, 2);
    overlap_buf.clear();
    overlap_buf(0, 0) = 1.0;
    EXPECT_DOUBLE_EQ(overlap_buf(0, 0), 1.0);

    KineticBuffer kinetic_buf;
    kinetic_buf.resize(2, 2);
    kinetic_buf.clear();
    kinetic_buf(1, 1) = 2.0;
    EXPECT_DOUBLE_EQ(kinetic_buf(1, 1), 2.0);

    NuclearBuffer nuclear_buf;
    nuclear_buf.resize(2, 2);
    nuclear_buf.clear();
    nuclear_buf(0, 1) = 3.0;
    EXPECT_DOUBLE_EQ(nuclear_buf(0, 1), 3.0);
}

// ============================================================================
// Test Size and Empty Methods
// ============================================================================

TEST(OneElectronBufferTest, SizeMethod) {
    OneElectronBuffer<0> buf;
    EXPECT_EQ(buf.size(), 0);

    buf.resize(3, 4);
    EXPECT_EQ(buf.size(), 12);  // 3 * 4 = 12

    buf.resize(2, 2);
    EXPECT_EQ(buf.size(), 4);   // 2 * 2 = 4
}

TEST(OneElectronBufferTest, EmptyMethod) {
    OneElectronBuffer<0> buf;
    EXPECT_TRUE(buf.empty());

    buf.resize(1, 1);
    EXPECT_FALSE(buf.empty());

    buf.resize(0, 0);
    EXPECT_TRUE(buf.empty());
}

// ============================================================================
// Test Derivative Component Count
// ============================================================================

TEST(OneElectronBufferTest, DerivativeComponentCount) {
    // Verify N_DERIV is correctly computed
    EXPECT_EQ(OneElectronBuffer<0>::N_DERIV, 1);   // Energy: 1 component
    EXPECT_EQ(OneElectronBuffer<1>::N_DERIV, 6);   // Gradient: 3 * 2 = 6
    EXPECT_EQ(OneElectronBuffer<2>::N_DERIV, 21);  // Hessian: 6 * 7 / 2 = 21
}

// ============================================================================
// Edge Cases: Resize and Clear
// ============================================================================

TEST(OneElectronBufferEdgeTest, ResizeToZero) {
    OneElectronBuffer<0> buf(3, 3);
    EXPECT_FALSE(buf.empty());

    buf.resize(0, 0);
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.size(), 0);
    EXPECT_EQ(buf.na(), 0);
    EXPECT_EQ(buf.nb(), 0);
}

TEST(OneElectronBufferEdgeTest, ResizeMultipleTimes) {
    OneElectronBuffer<0> buf;

    buf.resize(2, 2);
    EXPECT_EQ(buf.size(), 4);

    buf.resize(5, 5);
    EXPECT_EQ(buf.size(), 25);

    buf.resize(1, 1);
    EXPECT_EQ(buf.size(), 1);

    buf.resize(10, 3);
    EXPECT_EQ(buf.size(), 30);
}

TEST(OneElectronBufferEdgeTest, ClearEmptyBuffer) {
    OneElectronBuffer<0> buf;
    EXPECT_TRUE(buf.empty());
    // Should not crash
    buf.clear();
    EXPECT_TRUE(buf.empty());
}

// ============================================================================
// New Type Alias Tests
// ============================================================================

TEST(OneElectronBufferTest, GradientTypeAliases) {
    KineticGradientBuffer kgrad(2, 2);
    NuclearGradientBuffer ngrad(2, 2);
    KineticGradientBufferFloat kgradf(2, 2);
    NuclearGradientBufferFloat ngradf(2, 2);

    EXPECT_EQ(kgrad.N_DERIV, 6);
    EXPECT_EQ(ngrad.N_DERIV, 6);
    EXPECT_TRUE(kgrad.is_double_precision);
    EXPECT_TRUE(ngrad.is_double_precision);
    EXPECT_TRUE(kgradf.is_single_precision);
    EXPECT_TRUE(ngradf.is_single_precision);
}

TEST(OneElectronBufferTest, HessianTypeAliases) {
    OverlapHessianBuffer ohess(2, 2);
    KineticHessianBuffer khess(2, 2);
    NuclearHessianBuffer nhess(2, 2);
    OverlapHessianBufferFloat ohessf(2, 2);
    KineticHessianBufferFloat khessf(2, 2);
    NuclearHessianBufferFloat nhessf(2, 2);

    EXPECT_EQ(ohess.N_DERIV, 21);
    EXPECT_EQ(khess.N_DERIV, 21);
    EXPECT_EQ(nhess.N_DERIV, 21);
    EXPECT_TRUE(ohess.is_double_precision);
    EXPECT_TRUE(ohessf.is_single_precision);
    EXPECT_TRUE(khessf.is_single_precision);
    EXPECT_TRUE(nhessf.is_single_precision);
}

}  // namespace libaccint
