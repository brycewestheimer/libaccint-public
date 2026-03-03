// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_norm_correction.cpp
/// @brief Unit tests for the shared norm_correction utility
///
/// Tests the norm_correction(lx, ly, lz) function that computes
/// 1/sqrt((2lx-1)!! * (2ly-1)!! * (2lz-1)!!).

#include <libaccint/kernels/norm_correction.hpp>
#include <gtest/gtest.h>
#include <cmath>

using namespace libaccint;

constexpr Real TOL = 1e-14;

// ============================================================================
// S-type (L=0): all components are (0,0,0) → correction = 1.0
// ============================================================================

TEST(NormCorrectionTest, SType) {
    EXPECT_NEAR(norm_correction(0, 0, 0), 1.0, TOL);
}

// ============================================================================
// P-type (L=1): components (1,0,0), (0,1,0), (0,0,1) → correction = 1.0
// since (2*1-1)!! = 1!! = 1
// ============================================================================

TEST(NormCorrectionTest, PType_100) {
    EXPECT_NEAR(norm_correction(1, 0, 0), 1.0, TOL);
}

TEST(NormCorrectionTest, PType_010) {
    EXPECT_NEAR(norm_correction(0, 1, 0), 1.0, TOL);
}

TEST(NormCorrectionTest, PType_001) {
    EXPECT_NEAR(norm_correction(0, 0, 1), 1.0, TOL);
}

// ============================================================================
// D-type (L=2): (2lx-1)!! for lx=2 is 3!! = 3
// ============================================================================

TEST(NormCorrectionTest, DType_200) {
    // 1/sqrt(3!! * 1 * 1) = 1/sqrt(3)
    EXPECT_NEAR(norm_correction(2, 0, 0), 1.0 / std::sqrt(3.0), TOL);
}

TEST(NormCorrectionTest, DType_020) {
    EXPECT_NEAR(norm_correction(0, 2, 0), 1.0 / std::sqrt(3.0), TOL);
}

TEST(NormCorrectionTest, DType_002) {
    EXPECT_NEAR(norm_correction(0, 0, 2), 1.0 / std::sqrt(3.0), TOL);
}

TEST(NormCorrectionTest, DType_110) {
    // 1/sqrt(1 * 1 * 1) = 1.0
    EXPECT_NEAR(norm_correction(1, 1, 0), 1.0, TOL);
}

TEST(NormCorrectionTest, DType_101) {
    EXPECT_NEAR(norm_correction(1, 0, 1), 1.0, TOL);
}

TEST(NormCorrectionTest, DType_011) {
    EXPECT_NEAR(norm_correction(0, 1, 1), 1.0, TOL);
}

// ============================================================================
// F-type (L=3): (2*3-1)!! = 5!! = 15
// ============================================================================

TEST(NormCorrectionTest, FType_300) {
    // 1/sqrt(15)
    EXPECT_NEAR(norm_correction(3, 0, 0), 1.0 / std::sqrt(15.0), TOL);
}

TEST(NormCorrectionTest, FType_210) {
    // 1/sqrt(3 * 1 * 1) = 1/sqrt(3)
    EXPECT_NEAR(norm_correction(2, 1, 0), 1.0 / std::sqrt(3.0), TOL);
}

TEST(NormCorrectionTest, FType_111) {
    // 1/sqrt(1 * 1 * 1) = 1
    EXPECT_NEAR(norm_correction(1, 1, 1), 1.0, TOL);
}

// ============================================================================
// G-type (L=4): (2*4-1)!! = 7!! = 105
// ============================================================================

TEST(NormCorrectionTest, GType_400) {
    // 1/sqrt(105)
    EXPECT_NEAR(norm_correction(4, 0, 0), 1.0 / std::sqrt(105.0), TOL);
}

TEST(NormCorrectionTest, GType_220) {
    // 1/sqrt(3 * 3 * 1) = 1/3
    EXPECT_NEAR(norm_correction(2, 2, 0), 1.0 / 3.0, TOL);
}

// ============================================================================
// Symmetry: norm_correction is symmetric in its arguments
// ============================================================================

TEST(NormCorrectionTest, ArgumentSymmetry) {
    EXPECT_NEAR(norm_correction(2, 1, 0), norm_correction(0, 2, 1), TOL);
    EXPECT_NEAR(norm_correction(2, 1, 0), norm_correction(1, 0, 2), TOL);
    EXPECT_NEAR(norm_correction(3, 1, 0), norm_correction(0, 3, 1), TOL);
    EXPECT_NEAR(norm_correction(2, 2, 1), norm_correction(1, 2, 2), TOL);
}

// ============================================================================
// All values should be positive and <= 1
// ============================================================================

TEST(NormCorrectionTest, BoundsCheck) {
    for (int lx = 0; lx <= 4; ++lx) {
        for (int ly = 0; ly <= 4 - lx; ++ly) {
            for (int lz = 0; lz <= 4 - lx - ly; ++lz) {
                Real val = norm_correction(lx, ly, lz);
                EXPECT_GT(val, 0.0) << "norm_correction must be positive at ("
                    << lx << "," << ly << "," << lz << ")";
                EXPECT_LE(val, 1.0 + TOL) << "norm_correction must be <= 1 at ("
                    << lx << "," << ly << "," << lz << ")";
            }
        }
    }
}
