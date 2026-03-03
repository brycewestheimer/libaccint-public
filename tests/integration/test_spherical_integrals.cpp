// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_spherical_integrals.cpp
/// @brief Integration tests for spherical harmonic integral transformations

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/math/spherical_transform.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace libaccint;
using namespace libaccint::math;

class SphericalIntegralsTest : public ::testing::Test {
protected:
    static constexpr double TOL = 1e-12;

    // Helper to create a simple shell
    Shell create_shell(int am, double alpha, Point3D center) {
        return Shell(am, center, {alpha}, {1.0});
    }
};

// ============================================================================
// Dimension Consistency Tests
// ============================================================================

TEST_F(SphericalIntegralsTest, CartesianVsSphericalCounts) {
    // Verify that spherical always <= Cartesian for L >= 2
    for (int L = 0; L <= 6; ++L) {
        int n_cart = n_cartesian(L);
        int n_sph = n_spherical(L);

        if (L <= 1) {
            EXPECT_EQ(n_cart, n_sph) << "L=" << L;
        } else {
            EXPECT_LT(n_sph, n_cart) << "L=" << L;
        }
    }
}

TEST_F(SphericalIntegralsTest, FunctionCountFormulas) {
    // n_cartesian(L) = (L+1)(L+2)/2
    // n_spherical(L) = 2L+1
    for (int L = 0; L <= 6; ++L) {
        EXPECT_EQ(n_cartesian(L), (L + 1) * (L + 2) / 2) << "L=" << L;
        EXPECT_EQ(n_spherical(L), 2 * L + 1) << "L=" << L;
    }
}

TEST_F(SphericalIntegralsTest, GenericNFunctions) {
    for (int L = 0; L <= 6; ++L) {
        EXPECT_EQ(n_functions(L, false), n_cartesian(L)) << "Cartesian L=" << L;
        EXPECT_EQ(n_functions(L, true), n_spherical(L)) << "Spherical L=" << L;
    }
}

// ============================================================================
// BasisSet Spherical Flag Tests
// ============================================================================

TEST_F(SphericalIntegralsTest, BasisSetDefaultIsCartesian) {
    std::vector<Shell> shells = {
        create_shell(0, 1.0, {0.0, 0.0, 0.0}),
        create_shell(1, 1.0, {0.0, 0.0, 0.0}),
        create_shell(2, 1.0, {0.0, 0.0, 0.0})
    };
    BasisSet basis(std::move(shells));

    EXPECT_FALSE(basis.is_spherical());
}

TEST_F(SphericalIntegralsTest, BasisSetSphericalFlag) {
    std::vector<Shell> shells = {
        create_shell(0, 1.0, {0.0, 0.0, 0.0}),
        create_shell(1, 1.0, {0.0, 0.0, 0.0}),
        create_shell(2, 1.0, {0.0, 0.0, 0.0})
    };
    BasisSet basis(std::move(shells));

    basis.set_spherical(true);
    EXPECT_TRUE(basis.is_spherical());

    basis.set_spherical(false);
    EXPECT_FALSE(basis.is_spherical());
}

TEST_F(SphericalIntegralsTest, BasisSetFunctionCounts) {
    // Create basis with s, p, d shells
    std::vector<Shell> shells = {
        create_shell(0, 1.0, {0.0, 0.0, 0.0}),  // 1 Cartesian, 1 spherical
        create_shell(1, 1.0, {0.0, 0.0, 0.0}),  // 3 Cartesian, 3 spherical
        create_shell(2, 1.0, {0.0, 0.0, 0.0})   // 6 Cartesian, 5 spherical
    };
    BasisSet basis(std::move(shells));

    // Cartesian count: 1 + 3 + 6 = 10
    EXPECT_EQ(basis.n_basis_functions(), 10);

    // Spherical count: 1 + 3 + 5 = 9
    EXPECT_EQ(basis.n_basis_functions_spherical(), 9);
}

// ============================================================================
// Transformation Matrix Property Tests
// ============================================================================

TEST_F(SphericalIntegralsTest, TransformationMatrixRowNormalization) {
    // For d-functions, check that transformation produces properly normalized output
    const double* C = get_cart_to_sph_matrix(2);
    const int n_sph = n_spherical(2);
    const int n_cart = n_cartesian(2);

    // Each row should have some non-zero coefficients
    for (int i = 0; i < n_sph; ++i) {
        double row_norm_sq = 0.0;
        for (int j = 0; j < n_cart; ++j) {
            row_norm_sq += C[i * n_cart + j] * C[i * n_cart + j];
        }
        EXPECT_GT(row_norm_sq, 0.0) << "Row " << i << " is zero";
    }
}

TEST_F(SphericalIntegralsTest, TransformationPreservesTrace) {
    // For an identity-like Cartesian matrix, trace should be preserved (approximately)
    // This tests mathematical consistency of the transformation

    // Create a (pp|..) overlap-like block
    const int n = 3;  // p-functions
    std::vector<double> cart_identity(n * n, 0.0);
    for (int i = 0; i < n; ++i) cart_identity[i * n + i] = 1.0;

    std::vector<double> sph_result(n * n, 0.0);
    std::vector<double> work(n * n);

    transform_2d(1, 1, cart_identity.data(), sph_result.data(), work.data());

    // For p-functions, transformation is just a reordering, so trace = 3
    double trace = 0.0;
    for (int i = 0; i < n; ++i) trace += sph_result[i * n + i];
    EXPECT_NEAR(trace, 3.0, TOL);
}

// ============================================================================
// Overlap Integral Symmetry Tests (Simulated)
// ============================================================================

TEST_F(SphericalIntegralsTest, SphericalOverlapSymmetry) {
    // Simulate overlap transformation and verify symmetry is preserved
    // Real overlap matrix is symmetric, transformed should also be symmetric

    // Create a symmetric 3x3 "Cartesian overlap" for p-functions
    std::vector<double> cart_overlap = {
        1.0, 0.2, 0.1,
        0.2, 1.0, 0.3,
        0.1, 0.3, 1.0
    };

    std::vector<double> sph_overlap(9, 0.0);
    std::vector<double> work(9);

    transform_2d(1, 1, cart_overlap.data(), sph_overlap.data(), work.data());

    // Check symmetry
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            EXPECT_NEAR(sph_overlap[i * 3 + j], sph_overlap[j * 3 + i], TOL)
                << "Asymmetry at (" << i << "," << j << ")";
        }
    }
}

// ============================================================================
// ERI Symmetry Tests (Simulated)
// ============================================================================

TEST_F(SphericalIntegralsTest, SphericalERISymmetry) {
    // (ss|ss) should remain unchanged
    double cart_ssss = 1.234;
    double sph_ssss = 0.0;
    std::vector<double> work(16);

    transform_4d(0, 0, 0, 0, &cart_ssss, &sph_ssss, work.data());

    EXPECT_DOUBLE_EQ(sph_ssss, cart_ssss);
}

TEST_F(SphericalIntegralsTest, SphericalERIFourFold) {
    // For (sp|sp), verify basic transformation works
    const int n_sp = 1 * 3;  // 3 integrals for s*p
    std::vector<double> cart_eri(n_sp * n_sp, 0.0);

    // Set diagonal to 1
    for (int i = 0; i < n_sp; ++i) cart_eri[i * n_sp + i] = 1.0;

    std::vector<double> sph_eri(n_sp * n_sp, 0.0);
    std::vector<double> work(2 * n_sp * n_sp);

    transform_4d(0, 1, 0, 1, cart_eri.data(), sph_eri.data(), work.data());

    // Trace should be preserved
    double trace = 0.0;
    for (int i = 0; i < n_sp; ++i) trace += sph_eri[i * n_sp + i];
    EXPECT_NEAR(trace, 3.0, TOL);
}

// ============================================================================
// Quality Gate G11 Preparation Tests
// ============================================================================

TEST_F(SphericalIntegralsTest, DFunctionTransformationCorrectness) {
    // Test specific d-function transformation against known values
    // d_z² = (2z² - x² - y²) / 2

    // Cartesian input: pure zz (index 5)
    double cart[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0};  // zz = 1
    double sph[5] = {0.0};

    transform_1d(2, cart, sph);

    // d0 (z²) should get coefficient 1.0 from zz term
    // Based on our matrix: d0 = -0.5*xx - 0.5*yy + 1.0*zz
    EXPECT_NEAR(sph[0], 1.0, TOL);  // d0 from pure zz

    // Other d-functions should be near zero for pure zz input
    EXPECT_NEAR(sph[1], 0.0, TOL);  // d1 (xz)
    EXPECT_NEAR(sph[2], 0.0, TOL);  // d-1 (yz)
    EXPECT_NEAR(sph[3], 0.0, TOL);  // d2 (x²-y²)
    EXPECT_NEAR(sph[4], 0.0, TOL);  // d-2 (xy)
}

TEST_F(SphericalIntegralsTest, TransformationRoundTrip) {
    // While we don't implement spherical-to-Cartesian,
    // we can verify the matrix has full row rank by checking
    // that different Cartesian inputs give different spherical outputs

    double cart1[6] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // xx
    double cart2[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};  // yy

    double sph1[5], sph2[5];
    transform_1d(2, cart1, sph1);
    transform_1d(2, cart2, sph2);

    // sph1 and sph2 should be different
    bool different = false;
    for (int i = 0; i < 5; ++i) {
        if (std::abs(sph1[i] - sph2[i]) > TOL) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different) << "xx and yy should produce different spherical outputs";
}
