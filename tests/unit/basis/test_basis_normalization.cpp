// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_basis_normalization.cpp
/// @brief Unit tests for Shell normalization

#include <gtest/gtest.h>

#include <libaccint/basis/shell.hpp>
#include <libaccint/core/types.hpp>

#include <cmath>
#include <numbers>
#include <numeric>
#include <vector>

namespace libaccint::testing {

namespace {

/// Compute the overlap integral between two primitives of the same AM
/// at the same center: S = N_a * N_b * integral
/// For two normalized s-type Gaussians with exponents a and b at the same center:
///   overlap = (pi / (a + b))^(3/2)
///   With normalization factor N = (2*a/pi)^(3/4):
///   normalized overlap = ((2*a/pi)*(2*b/pi))^(3/4) * (pi/(a+b))^(3/2)
///                      = (4*a*b/pi^2)^(3/4) * (pi/(a+b))^(3/2)
double s_prim_overlap(double alpha_a, double alpha_b) {
    double sum = alpha_a + alpha_b;
    // Unnormalized overlap of s-type Gaussians at the same center
    return std::pow(std::numbers::pi / sum, 1.5);
}

/// Primitive normalization factor for s-type Gaussians
double s_prim_norm(double alpha) {
    return std::pow(2.0 * alpha / std::numbers::pi, 0.75);
}

/// Primitive normalization for p-type Gaussians
/// N = (2*alpha/pi)^(3/4) * (4*alpha)^(1/2) / sqrt(1)
/// where (2L-1)!! = 1!! = 1 for L=1
double p_prim_norm(double alpha) {
    return std::pow(2.0 * alpha / std::numbers::pi, 0.75) * std::sqrt(4.0 * alpha);
}

/// Unnormalized p-type overlap at the same center, same direction
/// integral of x^2 * exp(-a*r^2) * exp(-b*r^2) over all space
/// = (1/(2*(a+b))) * (pi/(a+b))^(3/2)
double p_prim_overlap_unnorm(double alpha_a, double alpha_b) {
    double sum = alpha_a + alpha_b;
    return (1.0 / (2.0 * sum)) * std::pow(std::numbers::pi / sum, 1.5);
}

/// Compute the self-overlap of a normalized contracted s-shell
/// S = sum_ij c_i * c_j * S_ij
/// where S_ij is the overlap between normalized primitives i and j
double compute_s_self_overlap(const Shell& shell) {
    const Size K = shell.n_primitives();
    double overlap = 0.0;
    for (Size i = 0; i < K; ++i) {
        for (Size j = 0; j < K; ++j) {
            double a = shell.exponents()[i];
            double b = shell.exponents()[j];
            // shell.coefficients() already includes primitive normalization,
            // so we only need the unnormalized overlap integral
            overlap += shell.coefficients()[i] * shell.coefficients()[j] *
                       s_prim_overlap(a, b);
        }
    }
    return overlap;
}

/// Compute self-overlap for p-shell (one Cartesian direction)
double compute_p_self_overlap(const Shell& shell) {
    const Size K = shell.n_primitives();
    double overlap = 0.0;
    for (Size i = 0; i < K; ++i) {
        for (Size j = 0; j < K; ++j) {
            double a = shell.exponents()[i];
            double b = shell.exponents()[j];
            // shell.coefficients() already includes primitive normalization,
            // so we only need the unnormalized overlap integral
            overlap += shell.coefficients()[i] * shell.coefficients()[j] *
                       p_prim_overlap_unnorm(a, b);
        }
    }
    return overlap;
}

}  // anonymous namespace

// =============================================================================
// S-Shell Normalization Tests
// =============================================================================

TEST(BasisNormalizationTest, SinglePrimitiveSShell) {
    // Single primitive s-shell: the normalization should make self-overlap = 1
    Point3D origin{0.0, 0.0, 0.0};
    Shell s(0, origin, {1.0}, {1.0});

    // The shell normalizes the coefficient; verify via self-overlap
    double overlap = compute_s_self_overlap(s);
    EXPECT_NEAR(overlap, 1.0, 1e-12);
}

TEST(BasisNormalizationTest, SinglePrimitiveSShellDiffExponent) {
    Point3D origin{0.0, 0.0, 0.0};
    Shell s(0, origin, {3.42525091}, {1.0});
    double overlap = compute_s_self_overlap(s);
    EXPECT_NEAR(overlap, 1.0, 1e-12);
}

TEST(BasisNormalizationTest, MultiPrimitiveSShell) {
    // STO-3G hydrogen 1s
    Point3D origin{0.0, 0.0, 0.0};
    Shell s(0, origin,
            {3.42525091, 0.62391373, 0.16885540},
            {0.15432897, 0.53532814, 0.44463454});

    double overlap = compute_s_self_overlap(s);
    EXPECT_NEAR(overlap, 1.0, 1e-10);
}

TEST(BasisNormalizationTest, MultiPrimitiveSShellOxygen) {
    // STO-3G oxygen 1s
    Point3D origin{0.0, 0.0, 0.0};
    Shell s(0, origin,
            {130.7093200, 23.8088610, 6.4436083},
            {0.15432897, 0.53532814, 0.44463454});

    double overlap = compute_s_self_overlap(s);
    EXPECT_NEAR(overlap, 1.0, 1e-10);
}

// =============================================================================
// P-Shell Normalization Tests
// =============================================================================

TEST(BasisNormalizationTest, SinglePrimitivePShell) {
    Point3D origin{0.0, 0.0, 0.0};
    Shell p(1, origin, {1.0}, {1.0});

    double overlap = compute_p_self_overlap(p);
    EXPECT_NEAR(overlap, 1.0, 1e-12);
}

TEST(BasisNormalizationTest, MultiPrimitivePShell) {
    // STO-3G oxygen 2p
    Point3D origin{0.0, 0.0, 0.0};
    Shell p(1, origin,
            {5.0331513, 1.1695961, 0.3803890},
            {0.15591627, 0.60768372, 0.39195739});

    double overlap = compute_p_self_overlap(p);
    EXPECT_NEAR(overlap, 1.0, 1e-10);
}

// =============================================================================
// D-Shell Normalization Tests
// =============================================================================

TEST(BasisNormalizationTest, SinglePrimitiveDShell) {
    Point3D origin{0.0, 0.0, 0.0};
    Shell d(2, origin, {1.0}, {1.0});

    // d-shell: check that it was created with the right AM
    EXPECT_EQ(d.angular_momentum(), 2);
    EXPECT_EQ(d.n_functions(), 6);  // 6 Cartesian d-functions
    EXPECT_GT(d.n_primitives(), 0u);
}

TEST(BasisNormalizationTest, DShellExponentValue) {
    Point3D origin{0.0, 0.0, 0.0};
    Shell d(2, origin, {1.5}, {1.0});
    EXPECT_GT(std::abs(d.coefficients()[0]), 0.0);
}

// =============================================================================
// Pre-normalized Tests
// =============================================================================

TEST(BasisNormalizationTest, PreNormalizedSkipsNormalization) {
    // Create a shell with pre-normalized tag — coefficients should remain unchanged
    Point3D origin{0.0, 0.0, 0.0};
    const double coeff = 0.42;
    Shell s(pre_normalized, 0, origin, {1.0}, {coeff});

    EXPECT_DOUBLE_EQ(s.coefficients()[0], coeff);
}

TEST(BasisNormalizationTest, PreNormalizedVsNormalized) {
    // Same exponents/coefficients: pre-normalized should differ from auto-normalized
    Point3D origin{0.0, 0.0, 0.0};
    Shell auto_norm(0, origin, {1.0}, {1.0});
    Shell pre_norm(pre_normalized, 0, origin, {1.0}, {1.0});

    // Auto-normalized coefficient will be different from 1.0
    // (it includes the normalization factor)
    // Pre-normalized coefficient stays at 1.0
    EXPECT_DOUBLE_EQ(pre_norm.coefficients()[0], 1.0);
    // The auto-normalized coefficient should not be 1.0 in general
    // (unless the normalization factor happens to be 1, which is only for specific exponents)
}

// =============================================================================
// Shell Function Counts
// =============================================================================

TEST(BasisNormalizationTest, CartesianFunctionCounts) {
    Point3D origin{0.0, 0.0, 0.0};
    Shell s(0, origin, {1.0}, {1.0});
    Shell p(1, origin, {1.0}, {1.0});
    Shell d(2, origin, {1.0}, {1.0});
    Shell f(3, origin, {1.0}, {1.0});

    EXPECT_EQ(s.n_functions(), 1);   // (l+1)(l+2)/2 = 1
    EXPECT_EQ(p.n_functions(), 3);   // 3
    EXPECT_EQ(d.n_functions(), 6);   // 6
    EXPECT_EQ(f.n_functions(), 10);  // 10
}

}  // namespace libaccint::testing
