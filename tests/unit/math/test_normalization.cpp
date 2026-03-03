// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/math/normalization.hpp>
#include <libaccint/utils/constants.hpp>
#include <gtest/gtest.h>
#include <cmath>

using namespace libaccint;
using namespace libaccint::math;

namespace {

// Test tolerance for floating-point comparisons
constexpr Real TOLERANCE = 1e-14;

// Helper function to compare Real values with tolerance
bool are_close(Real a, Real b, Real tol = TOLERANCE) {
    return std::abs(a - b) <= tol;
}

}  // anonymous namespace

// =============================================================================
// Double Factorial Tests
// =============================================================================

TEST(DoublFactorialOddTest, ZeroCase) {
    // By convention, (-1)!! = 1
    EXPECT_EQ(double_factorial_odd(0), 1);
}

TEST(DoublFactorialOddTest, NegativeCase) {
    // For n < 0, should return 1 by convention
    EXPECT_EQ(double_factorial_odd(-1), 1);
    EXPECT_EQ(double_factorial_odd(-5), 1);
}

TEST(DoublFactorialOddTest, CorrectValues) {
    // Test known values for n=1 to n=6
    EXPECT_EQ(double_factorial_odd(1), 1);      // 1!! = 1
    EXPECT_EQ(double_factorial_odd(2), 3);      // 3!! = 3
    EXPECT_EQ(double_factorial_odd(3), 15);     // 5!! = 15
    EXPECT_EQ(double_factorial_odd(4), 105);    // 7!! = 105
    EXPECT_EQ(double_factorial_odd(5), 945);    // 9!! = 945
    EXPECT_EQ(double_factorial_odd(6), 10395);  // 11!! = 10395
}

// =============================================================================
// Normalization Factor Tests - Self-Overlap Integral Verification
// =============================================================================

/**
 * @brief Analytical computation of 1D Gaussian integral
 *
 * Computes: ∫_{-∞}^{∞} x^(2n) * exp(-2α*x²) dx
 *
 * The result is: (2n-1)!! * sqrt(π) / (2^(n+0.5) * (2α)^(n+0.5))
 *
 * For n=0: ∫ exp(-2α*x²) dx = sqrt(π/(2α))
 */
inline Real compute_1d_gaussian_integral(int n, Real alpha) {
    // For a normalized Gaussian with exponent α, we compute the self-overlap integral:
    // ∫ x^(2n) * exp(-2α*x²) dx
    //
    // Using the formula (a = 2α):
    // ∫ x^(2n) * exp(-a*x²) dx = (2n-1)!! * sqrt(π) / (2^n * a^(n+0.5))
    // = (2n-1)!! * sqrt(π) / (2^n * (2α)^(n+0.5))

    const Real two_alpha = 2.0 * alpha;
    const Real df_n = static_cast<Real>(double_factorial_odd(n));
    const Real sqrt_pi = constants::SQRT_PI;

    // Compute 2^n
    const Real two_to_n = std::pow(2.0, static_cast<Real>(n));

    // Compute (2α)^(n+0.5) = (2α)^n * sqrt(2α)
    const Real two_alpha_to_n = std::pow(two_alpha, static_cast<Real>(n));
    const Real sqrt_two_alpha = std::sqrt(two_alpha);
    const Real two_alpha_to_n_plus_half = two_alpha_to_n * sqrt_two_alpha;

    const Real result = df_n * sqrt_pi / (two_to_n * two_alpha_to_n_plus_half);
    return result;
}

/**
 * @brief Verify self-overlap integral for normalized Cartesian Gaussian
 *
 * For a normalized Cartesian Gaussian:
 *   μ(r) = N * x^i * y^j * z^k * exp(-α*r²)
 *
 * where N is the normalization factor from normalization_factor(alpha, i, j, k),
 * the self-overlap integral should be:
 *   ∫ μ(r)² dr = 1.0
 *
 * This integral factors as:
 *   ∫∫∫ N² * x^(2i) * y^(2j) * z^(2k) * exp(-2α*r²) dx dy dz
 *   = N² * I_x(i) * I_y(j) * I_z(k)
 *
 * where each I_c is a 1D Gaussian integral.
 */
inline Real compute_self_overlap_integral(Real alpha, int i, int j, int k) {
    const Real N = normalization_factor(alpha, i, j, k);

    // Compute 1D integrals for each dimension
    const Real I_x = compute_1d_gaussian_integral(i, alpha);
    const Real I_y = compute_1d_gaussian_integral(j, alpha);
    const Real I_z = compute_1d_gaussian_integral(k, alpha);

    // Self-overlap: N² * I_x * I_y * I_z
    const Real self_overlap = N * N * I_x * I_y * I_z;
    return self_overlap;
}

// =============================================================================
// S-Type (L=0) Tests
// =============================================================================

TEST(NormalizationTest, STypeAlpha05) {
    const Real alpha = 0.5;
    const Real self_overlap = compute_self_overlap_integral(alpha, 0, 0, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "S-type self-overlap for α=0.5";
}

TEST(NormalizationTest, STypeAlpha10) {
    const Real alpha = 1.0;
    const Real self_overlap = compute_self_overlap_integral(alpha, 0, 0, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "S-type self-overlap for α=1.0";
}

TEST(NormalizationTest, STypeAlpha25) {
    const Real alpha = 2.5;
    const Real self_overlap = compute_self_overlap_integral(alpha, 0, 0, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "S-type self-overlap for α=2.5";
}

TEST(NormalizationTest, STypeAlpha100) {
    const Real alpha = 10.0;
    const Real self_overlap = compute_self_overlap_integral(alpha, 0, 0, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "S-type self-overlap for α=10.0";
}

// =============================================================================
// P-Type (L=1) Tests - All 3 Components
// =============================================================================

// Px: i=1, j=0, k=0
TEST(NormalizationTest, PxTypeAlpha05) {
    const Real alpha = 0.5;
    const Real self_overlap = compute_self_overlap_integral(alpha, 1, 0, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Px-type self-overlap for α=0.5";
}

TEST(NormalizationTest, PxTypeAlpha10) {
    const Real alpha = 1.0;
    const Real self_overlap = compute_self_overlap_integral(alpha, 1, 0, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Px-type self-overlap for α=1.0";
}

TEST(NormalizationTest, PxTypeAlpha25) {
    const Real alpha = 2.5;
    const Real self_overlap = compute_self_overlap_integral(alpha, 1, 0, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Px-type self-overlap for α=2.5";
}

TEST(NormalizationTest, PxTypeAlpha100) {
    const Real alpha = 10.0;
    const Real self_overlap = compute_self_overlap_integral(alpha, 1, 0, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Px-type self-overlap for α=10.0";
}

// Py: i=0, j=1, k=0
TEST(NormalizationTest, PyTypeAlpha05) {
    const Real alpha = 0.5;
    const Real self_overlap = compute_self_overlap_integral(alpha, 0, 1, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Py-type self-overlap for α=0.5";
}

TEST(NormalizationTest, PyTypeAlpha10) {
    const Real alpha = 1.0;
    const Real self_overlap = compute_self_overlap_integral(alpha, 0, 1, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Py-type self-overlap for α=1.0";
}

TEST(NormalizationTest, PyTypeAlpha25) {
    const Real alpha = 2.5;
    const Real self_overlap = compute_self_overlap_integral(alpha, 0, 1, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Py-type self-overlap for α=2.5";
}

TEST(NormalizationTest, PyTypeAlpha100) {
    const Real alpha = 10.0;
    const Real self_overlap = compute_self_overlap_integral(alpha, 0, 1, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Py-type self-overlap for α=10.0";
}

// Pz: i=0, j=0, k=1
TEST(NormalizationTest, PzTypeAlpha05) {
    const Real alpha = 0.5;
    const Real self_overlap = compute_self_overlap_integral(alpha, 0, 0, 1);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Pz-type self-overlap for α=0.5";
}

TEST(NormalizationTest, PzTypeAlpha10) {
    const Real alpha = 1.0;
    const Real self_overlap = compute_self_overlap_integral(alpha, 0, 0, 1);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Pz-type self-overlap for α=1.0";
}

TEST(NormalizationTest, PzTypeAlpha25) {
    const Real alpha = 2.5;
    const Real self_overlap = compute_self_overlap_integral(alpha, 0, 0, 1);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Pz-type self-overlap for α=2.5";
}

TEST(NormalizationTest, PzTypeAlpha100) {
    const Real alpha = 10.0;
    const Real self_overlap = compute_self_overlap_integral(alpha, 0, 0, 1);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Pz-type self-overlap for α=10.0";
}

// =============================================================================
// D-Type (L=2) Tests - All 6 Cartesian Components
// =============================================================================

// xx: i=2, j=0, k=0
TEST(NormalizationTest, DxxTypeAlpha05) {
    const Real alpha = 0.5;
    const Real self_overlap = compute_self_overlap_integral(alpha, 2, 0, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Dxx-type self-overlap for α=0.5";
}

TEST(NormalizationTest, DxxTypeAlpha10) {
    const Real alpha = 1.0;
    const Real self_overlap = compute_self_overlap_integral(alpha, 2, 0, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Dxx-type self-overlap for α=1.0";
}

// yy: i=0, j=2, k=0
TEST(NormalizationTest, DyyTypeAlpha25) {
    const Real alpha = 2.5;
    const Real self_overlap = compute_self_overlap_integral(alpha, 0, 2, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Dyy-type self-overlap for α=2.5";
}

// zz: i=0, j=0, k=2
TEST(NormalizationTest, DzzTypeAlpha100) {
    const Real alpha = 10.0;
    const Real self_overlap = compute_self_overlap_integral(alpha, 0, 0, 2);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Dzz-type self-overlap for α=10.0";
}

// xy: i=1, j=1, k=0
TEST(NormalizationTest, DxyTypeAlpha10) {
    const Real alpha = 1.0;
    const Real self_overlap = compute_self_overlap_integral(alpha, 1, 1, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Dxy-type self-overlap for α=1.0";
}

// xz: i=1, j=0, k=1
TEST(NormalizationTest, DxzTypeAlpha25) {
    const Real alpha = 2.5;
    const Real self_overlap = compute_self_overlap_integral(alpha, 1, 0, 1);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Dxz-type self-overlap for α=2.5";
}

// yz: i=0, j=1, k=1
TEST(NormalizationTest, DyzTypeAlpha05) {
    const Real alpha = 0.5;
    const Real self_overlap = compute_self_overlap_integral(alpha, 0, 1, 1);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Dyz-type self-overlap for α=0.5";
}

// =============================================================================
// F-Type (L=3) Tests - Selected Components
// =============================================================================

// xxx: i=3, j=0, k=0
TEST(NormalizationTest, FxxxTypeAlpha10) {
    const Real alpha = 1.0;
    const Real self_overlap = compute_self_overlap_integral(alpha, 3, 0, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Fxxx-type self-overlap for α=1.0";
}

// yyy: i=0, j=3, k=0
TEST(NormalizationTest, FyyyTypeAlpha25) {
    const Real alpha = 2.5;
    const Real self_overlap = compute_self_overlap_integral(alpha, 0, 3, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Fyyy-type self-overlap for α=2.5";
}

// zzz: i=0, j=0, k=3
TEST(NormalizationTest, FzzzTypeAlpha05) {
    const Real alpha = 0.5;
    const Real self_overlap = compute_self_overlap_integral(alpha, 0, 0, 3);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Fzzz-type self-overlap for α=0.5";
}

// xxy: i=2, j=1, k=0
TEST(NormalizationTest, FxxyTypeAlpha10) {
    const Real alpha = 1.0;
    const Real self_overlap = compute_self_overlap_integral(alpha, 2, 1, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Fxxy-type self-overlap for α=1.0";
}

// xxz: i=2, j=0, k=1
TEST(NormalizationTest, FxxzTypeAlpha100) {
    const Real alpha = 10.0;
    const Real self_overlap = compute_self_overlap_integral(alpha, 2, 0, 1);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Fxxz-type self-overlap for α=10.0";
}

// =============================================================================
// Convenience Function Tests (normalization_factor with single L parameter)
// =============================================================================

TEST(NormalizationTest, ConvenienceFunctionL0) {
    const Real alpha = 1.0;
    const Real norm_ijk = normalization_factor(alpha, 0, 0, 0);
    const Real norm_L = normalization_factor(alpha, 0);
    EXPECT_NEAR(norm_ijk, norm_L, TOLERANCE) << "Convenience function should match (0,0,0)";
}

TEST(NormalizationTest, ConvenienceFunctionL1) {
    const Real alpha = 2.5;
    const Real norm_ijk = normalization_factor(alpha, 1, 0, 0);
    const Real norm_L = normalization_factor(alpha, 1);
    EXPECT_NEAR(norm_ijk, norm_L, TOLERANCE) << "Convenience function should match (1,0,0)";
}

TEST(NormalizationTest, ConvenienceFunctionL2) {
    const Real alpha = 0.5;
    const Real norm_ijk = normalization_factor(alpha, 2, 0, 0);
    const Real norm_L = normalization_factor(alpha, 2);
    EXPECT_NEAR(norm_ijk, norm_L, TOLERANCE) << "Convenience function should match (2,0,0)";
}

TEST(NormalizationTest, ConvenienceFunctionL3) {
    const Real alpha = 1.0;
    const Real norm_ijk = normalization_factor(alpha, 3, 0, 0);
    const Real norm_L = normalization_factor(alpha, 3);
    EXPECT_NEAR(norm_ijk, norm_L, TOLERANCE) << "Convenience function should match (3,0,0)";
}

// =============================================================================
// Normalization Factor Properties
// =============================================================================

TEST(NormalizationTest, FactorIncreaseWithAlpha) {
    // Normalization factor should increase with alpha (tighter Gaussian)
    const Real N_small_alpha = normalization_factor(0.5, 0, 0, 0);
    const Real N_large_alpha = normalization_factor(10.0, 0, 0, 0);
    EXPECT_LT(N_small_alpha, N_large_alpha) << "N should increase with alpha";
}

TEST(NormalizationTest, FactorIncreaseWithAngularMomentum) {
    // For fixed alpha=1, normalization factor increases with L for (L,0,0)
    // because N_L = sqrt((2α/π)^{3/2} * (4α)^L / (2L-1)!!) and (4α)^L
    // grows faster than (2L-1)!! for moderate L
    const Real alpha = 1.0;
    const Real N_s = normalization_factor(alpha, 0, 0, 0);
    const Real N_p = normalization_factor(alpha, 1, 0, 0);
    const Real N_d = normalization_factor(alpha, 2, 0, 0);
    EXPECT_LT(N_s, N_p) << "S should have smaller factor than P for alpha=1";
    EXPECT_LT(N_p, N_d) << "P should have smaller factor than D for alpha=1";
}

TEST(NormalizationTest, PositiveFactors) {
    // All normalization factors should be positive
    for (Real alpha : {0.1, 0.5, 1.0, 2.5, 5.0, 10.0}) {
        for (int l = 0; l <= 3; ++l) {
            const Real norm = normalization_factor(alpha, l);
            EXPECT_GT(norm, 0.0) << "Normalization factor must be positive";
        }
    }
}

// =============================================================================
// Edge Cases and Robustness
// =============================================================================

TEST(NormalizationTest, SmallAlpha) {
    // Very small alpha (diffuse function)
    const Real alpha = 0.01;
    const Real self_overlap = compute_self_overlap_integral(alpha, 1, 0, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Self-overlap should be 1 for small alpha";
}

TEST(NormalizationTest, LargeAlpha) {
    // Very large alpha (tight function)
    const Real alpha = 100.0;
    const Real self_overlap = compute_self_overlap_integral(alpha, 1, 0, 0);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Self-overlap should be 1 for large alpha";
}

TEST(NormalizationTest, HighAngularMomentum) {
    // Test with maximum angular momentum (L=6)
    const Real alpha = 1.0;
    const Real self_overlap = compute_self_overlap_integral(alpha, 2, 2, 2);
    EXPECT_NEAR(self_overlap, 1.0, TOLERANCE) << "Self-overlap should be 1 for L=6";
}
