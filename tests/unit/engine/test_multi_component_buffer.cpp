// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_multi_component_buffer.cpp
/// @brief Unit tests for MultiComponentBuffer

#include <libaccint/engine/multi_component_buffer.hpp>
#include <gtest/gtest.h>

using namespace libaccint;

// ============================================================================
// Construction Tests
// ============================================================================

TEST(MultiComponentBufferTest, ConstructFromOperatorKind) {
    MultiComponentBuffer buf(OperatorKind::ElectricDipole);
    EXPECT_EQ(buf.n_components(), 3);
    EXPECT_EQ(buf.operator_kind(), OperatorKind::ElectricDipole);
}

TEST(MultiComponentBufferTest, ConstructQuadrupole) {
    MultiComponentBuffer buf(OperatorKind::ElectricQuadrupole);
    EXPECT_EQ(buf.n_components(), 6);
}

TEST(MultiComponentBufferTest, ConstructOctupole) {
    MultiComponentBuffer buf(OperatorKind::ElectricOctupole);
    EXPECT_EQ(buf.n_components(), 10);
}

TEST(MultiComponentBufferTest, ConstructLinearMomentum) {
    MultiComponentBuffer buf(OperatorKind::LinearMomentum);
    EXPECT_EQ(buf.n_components(), 3);
}

TEST(MultiComponentBufferTest, ConstructAngularMomentum) {
    MultiComponentBuffer buf(OperatorKind::AngularMomentum);
    EXPECT_EQ(buf.n_components(), 3);
}

TEST(MultiComponentBufferTest, ConstructExplicitComponentCount) {
    MultiComponentBuffer buf(static_cast<Size>(5));
    EXPECT_EQ(buf.n_components(), 5);
}

// ============================================================================
// Resize and Access Tests
// ============================================================================

TEST(MultiComponentBufferTest, ResizeAndAccess) {
    MultiComponentBuffer buf(OperatorKind::ElectricDipole);
    buf.resize(3, 6);  // p × d

    EXPECT_EQ(buf.na(), 3);
    EXPECT_EQ(buf.nb(), 6);
    EXPECT_EQ(buf.total_size(), 3 * 3 * 6);  // 3 components × 3 × 6

    // All elements should be zero after resize
    for (Size c = 0; c < 3; ++c) {
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 6; ++b) {
                EXPECT_DOUBLE_EQ(buf(c, a, b), 0.0);
            }
        }
    }
}

TEST(MultiComponentBufferTest, WriteAndRead) {
    MultiComponentBuffer buf(OperatorKind::ElectricDipole);
    buf.resize(2, 3);

    buf(0, 0, 0) = 1.0;
    buf(1, 0, 1) = 2.0;
    buf(2, 1, 2) = 3.0;

    EXPECT_DOUBLE_EQ(buf(0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(buf(1, 0, 1), 2.0);
    EXPECT_DOUBLE_EQ(buf(2, 1, 2), 3.0);
    // Other elements remain zero
    EXPECT_DOUBLE_EQ(buf(0, 1, 0), 0.0);
}

TEST(MultiComponentBufferTest, ComponentSpan) {
    MultiComponentBuffer buf(OperatorKind::ElectricDipole);
    buf.resize(2, 3);

    buf(1, 0, 2) = 42.0;
    auto span = buf.component(1);
    EXPECT_EQ(span.size(), 6);  // 2 × 3
    EXPECT_DOUBLE_EQ(span[0 * 3 + 2], 42.0);
}

TEST(MultiComponentBufferTest, Clear) {
    MultiComponentBuffer buf(OperatorKind::ElectricDipole);
    buf.resize(2, 2);
    buf(0, 0, 0) = 99.0;
    buf.clear();
    EXPECT_DOUBLE_EQ(buf(0, 0, 0), 0.0);
}

// ============================================================================
// Origin Tests
// ============================================================================

TEST(MultiComponentBufferTest, Origin) {
    MultiComponentBuffer buf(OperatorKind::ElectricDipole);
    EXPECT_DOUBLE_EQ(buf.origin()[0], 0.0);
    EXPECT_DOUBLE_EQ(buf.origin()[1], 0.0);
    EXPECT_DOUBLE_EQ(buf.origin()[2], 0.0);

    buf.set_origin({1.0, 2.0, 3.0});
    EXPECT_DOUBLE_EQ(buf.origin()[0], 1.0);
    EXPECT_DOUBLE_EQ(buf.origin()[1], 2.0);
    EXPECT_DOUBLE_EQ(buf.origin()[2], 3.0);
}

// ============================================================================
// Symmetry Type Tests
// ============================================================================

TEST(MultiComponentBufferTest, SymmetryType) {
    MultiComponentBuffer dipole(OperatorKind::ElectricDipole);
    EXPECT_EQ(dipole.symmetry_type(), MatrixSymmetry::Symmetric);

    MultiComponentBuffer linmom(OperatorKind::LinearMomentum);
    EXPECT_EQ(linmom.symmetry_type(), MatrixSymmetry::AntiSymmetric);

    MultiComponentBuffer angmom(OperatorKind::AngularMomentum);
    EXPECT_EQ(angmom.symmetry_type(), MatrixSymmetry::AntiSymmetric);
}

// ============================================================================
// Component Label Tests
// ============================================================================

TEST(MultiComponentBufferTest, DipoleLabels) {
    MultiComponentBuffer buf(OperatorKind::ElectricDipole);
    EXPECT_EQ(buf.component_label(0), "x");
    EXPECT_EQ(buf.component_label(1), "y");
    EXPECT_EQ(buf.component_label(2), "z");
}

TEST(MultiComponentBufferTest, QuadrupoleLabels) {
    MultiComponentBuffer buf(OperatorKind::ElectricQuadrupole);
    EXPECT_EQ(buf.component_label(0), "xx");
    EXPECT_EQ(buf.component_label(1), "xy");
    EXPECT_EQ(buf.component_label(5), "zz");
}

TEST(MultiComponentBufferTest, OctupoleLabels) {
    MultiComponentBuffer buf(OperatorKind::ElectricOctupole);
    EXPECT_EQ(buf.component_label(0), "xxx");
    EXPECT_EQ(buf.component_label(9), "zzz");
}
