// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_interaction_tensors.cpp
/// @brief Validation tests for interaction tensors and solid harmonics

#include <libaccint/math/interaction_tensors.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

using namespace libaccint;
using namespace libaccint::math;

namespace {
constexpr double TOL = 1e-12;
constexpr double LOOSE_TOL = 1e-10;
}

// ============================================================================
// Regular Solid Harmonics
// ============================================================================

TEST(SolidHarmonicsTest, RegularL0M0) {
    // R_0^0 = 1
    EXPECT_NEAR(regular_solid_harmonic(0, 0, 1.0, 2.0, 3.0), 1.0, TOL);
    EXPECT_NEAR(regular_solid_harmonic(0, 0, 0.0, 0.0, 0.0), 1.0, TOL);
}

TEST(SolidHarmonicsTest, RegularL1) {
    // R_1^0 = x, R_1^1 = y, R_1^2 = z  (Cartesian index convention)
    double x = 1.5, y = 2.3, z = -0.7;

    EXPECT_NEAR(regular_solid_harmonic(1, 0, x, y, z), x, TOL);
    EXPECT_NEAR(regular_solid_harmonic(1, 1, x, y, z), y, TOL);
    EXPECT_NEAR(regular_solid_harmonic(1, 2, x, y, z), z, TOL);
}

TEST(SolidHarmonicsTest, RegularL2) {
    // Rank-2 Cartesian solid harmonics: m=0→xx, m=1→xy, m=2→xz, m=3→yy, m=4→yz, m=5→zz
    double x = 1.0, y = 2.0, z = 3.0;

    Real r20 = regular_solid_harmonic(2, 0, x, y, z);
    EXPECT_NEAR(r20, x * x, TOL) << "R_2^0 = x² (Cartesian)";
}

// ============================================================================
// Irregular Solid Harmonics
// ============================================================================

TEST(SolidHarmonicsTest, IrregularL0) {
    // I_0^0 = 1/r
    double x = 0.0, y = 0.0, z = 2.0;
    double r = 2.0;
    EXPECT_NEAR(irregular_solid_harmonic(0, 0, x, y, z), 1.0 / r, TOL);
}

TEST(SolidHarmonicsTest, IrregularL1) {
    // I_1^m = R_1^m / r³, Cartesian indices: m=0→x, m=1→y, m=2→z
    double x = 1.0, y = 0.0, z = 0.0;
    double r = 1.0;
    double r3 = r * r * r;

    EXPECT_NEAR(irregular_solid_harmonic(1, 0, x, y, z), x / r3, TOL);
    EXPECT_NEAR(irregular_solid_harmonic(1, 2, x, y, z), z / r3, TOL);
}

// ============================================================================
// Interaction Tensors
// ============================================================================

TEST(InteractionTensorTest, Rank0) {
    // T^(0) = 1/R
    std::array<Real, 3> R = {0.0, 0.0, 2.0};
    auto T = interaction_tensor(0, R);
    ASSERT_EQ(T.size(), 1);
    EXPECT_NEAR(T[0], 0.5, TOL) << "T^(0) = 1/|R| for R = (0,0,2)";
}

TEST(InteractionTensorTest, Rank1) {
    // T^(1)_i = -R_i / R³
    std::array<Real, 3> R = {0.0, 0.0, 2.0};
    auto T = interaction_tensor(1, R);
    ASSERT_EQ(T.size(), 3);

    double r3 = 8.0;  // |R|³ = 2³
    EXPECT_NEAR(T[0], -0.0 / r3, TOL) << "Tx = -Rx/R³";
    EXPECT_NEAR(T[1], -0.0 / r3, TOL) << "Ty = -Ry/R³";
    EXPECT_NEAR(T[2], -2.0 / r3, TOL) << "Tz = -Rz/R³";
}

TEST(InteractionTensorTest, Rank2) {
    // T^(2)_ij = (3*R_i*R_j - R²*delta_ij) / R⁵
    std::array<Real, 3> R = {0.0, 0.0, 2.0};
    auto T = interaction_tensor(2, R);
    ASSERT_EQ(T.size(), 6);  // xx, xy, xz, yy, yz, zz

    double r2 = 4.0;
    double r5 = 32.0;

    // xx: (3*0*0 - 4) / 32 = -4/32 = -0.125
    EXPECT_NEAR(T[0], -r2 / r5, LOOSE_TOL) << "T_xx";
    // xy: (3*0*0) / 32 = 0
    EXPECT_NEAR(T[1], 0.0, TOL) << "T_xy";
    // xz: (3*0*2) / 32 = 0
    EXPECT_NEAR(T[2], 0.0, TOL) << "T_xz";
    // yy: (3*0*0 - 4) / 32 = -0.125
    EXPECT_NEAR(T[3], -r2 / r5, LOOSE_TOL) << "T_yy";
    // yz: (3*0*2) / 32 = 0
    EXPECT_NEAR(T[4], 0.0, TOL) << "T_yz";
    // zz: (3*2*2 - 4) / 32 = (12-4)/32 = 0.25
    EXPECT_NEAR(T[5], (3.0 * 4.0 - r2) / r5, LOOSE_TOL) << "T_zz";
}

TEST(InteractionTensorTest, Rank2Traceless) {
    // The rank-2 tensor should be traceless: T_xx + T_yy + T_zz = 0
    std::array<Real, 3> R = {1.0, 2.0, 3.0};
    auto T = interaction_tensor(2, R);

    Real trace = T[0] + T[3] + T[5];  // xx + yy + zz
    EXPECT_NEAR(trace, 0.0, LOOSE_TOL) << "Rank-2 interaction tensor should be traceless";
}

TEST(InteractionTensorTest, ComponentCounts) {
    EXPECT_EQ(n_tensor_components(0), 1);
    EXPECT_EQ(n_tensor_components(1), 3);
    EXPECT_EQ(n_tensor_components(2), 6);
}

// ============================================================================
// Task 3.3.5: Exception Tests for Unsupported Ranks
// ============================================================================

TEST(InteractionTensorTest, Rank3ThrowsException) {
    std::array<Real, 3> R = {1.0, 2.0, 3.0};
    EXPECT_THROW(interaction_tensor(3, R), std::invalid_argument);
}

TEST(InteractionTensorTest, Rank4ThrowsException) {
    std::array<Real, 3> R = {1.0, 0.0, 0.0};
    EXPECT_THROW(interaction_tensor(4, R), std::invalid_argument);
}

TEST(InteractionTensorTest, Rank5ThrowsException) {
    std::array<Real, 3> R = {0.0, 0.0, 1.0};
    EXPECT_THROW(interaction_tensor(5, R), std::invalid_argument);
}

TEST(InteractionTensorTest, HighRankExceptionMessage) {
    std::array<Real, 3> R = {1.0, 2.0, 3.0};
    try {
        interaction_tensor(3, R);
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("rank"), std::string::npos)
            << "Exception message should mention rank: " << msg;
        EXPECT_NE(msg.find("3"), std::string::npos)
            << "Exception message should mention rank value: " << msg;
    }
}

TEST(InteractionTensorTest, ComponentCountsHigherRanks) {
    // n_tensor_components(rank) = (rank+1)*(rank+2)/2 for symmetric tensor
    EXPECT_EQ(n_tensor_components(3), 10);   // (3+1)*(3+2)/2 = 10
    EXPECT_EQ(n_tensor_components(4), 15);   // (4+1)*(4+2)/2 = 15
    EXPECT_EQ(n_tensor_components(5), 21);   // (5+1)*(5+2)/2 = 21
}
