// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/basis/shell.hpp>
#include <libaccint/utils/constants.hpp>
#include <libaccint/utils/error_handling.hpp>
#include <gtest/gtest.h>
#include <cmath>

using namespace libaccint;

namespace {

// Test tolerance for floating-point comparisons
constexpr Real TOLERANCE = 1e-12;

// Helper function to compare Real values with tolerance
bool are_close(Real a, Real b, Real tol = TOLERANCE) {
    return std::abs(a - b) <= tol;
}

/**
 * @brief Compute self-overlap integral for a contracted shell
 *
 * For validation purposes, we compute the self-overlap of the (l,0,0) component:
 *   <φ|φ> = Σ_i Σ_j c_i c_j <exp(-α_i*r²) x^l | exp(-α_j*r²) x^l>
 *
 * The overlap between two primitives is:
 *   <exp(-α_i*r²) x^l | exp(-α_j*r²) x^l> = (π/p)^(3/2) * (1/2p)^l * (2l-1)!!
 * where p = α_i + α_j
 *
 * However, if the coefficients include the primitive normalization, the (2l-1)!!
 * factor is absorbed into the coefficients, so we just compute:
 *   overlap = (π/p)^(3/2) * (0.5/p)^l
 */
Real compute_self_overlap(const Shell& shell) {
    const int am = shell.angular_momentum();
    const auto exponents = shell.exponents();
    const auto coefficients = shell.coefficients();
    const Size n_prim = shell.n_primitives();

    Real self_overlap = 0.0;
    for (Size i = 0; i < n_prim; ++i) {
        for (Size j = 0; j < n_prim; ++j) {
            const Real p = exponents[i] + exponents[j];

            // Compute (π/p)^(3/2)
            Real overlap = std::pow(constants::PI / p, 1.5);

            // Apply angular momentum factor: (0.5/p)^l
            for (int k = 0; k < am; ++k) {
                overlap *= 0.5 / p;
            }

            self_overlap += coefficients[i] * coefficients[j] * overlap;
        }
    }

    return self_overlap;
}

}  // anonymous namespace

// =============================================================================
// Valid Construction Tests
// =============================================================================

TEST(ShellTest, ConstructSShell) {
    // S-shell: 3 primitives (STO-3G hydrogen)
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {3.425250914, 0.6239137298, 0.168855404};
    std::vector<Real> coefficients = {0.1543289673, 0.5353281423, 0.4446345422};

    Shell shell(AngularMomentum::S, center, exponents, coefficients);

    EXPECT_EQ(shell.angular_momentum(), 0);
    EXPECT_EQ(shell.angular_momentum_enum(), AngularMomentum::S);
    EXPECT_EQ(shell.n_primitives(), 3);
    EXPECT_EQ(shell.n_functions(), 1);
    EXPECT_TRUE(shell.valid());

    // Check center
    EXPECT_DOUBLE_EQ(shell.center().x, 0.0);
    EXPECT_DOUBLE_EQ(shell.center().y, 0.0);
    EXPECT_DOUBLE_EQ(shell.center().z, 0.0);

    // Check exponents (should be unchanged)
    EXPECT_DOUBLE_EQ(shell.exponent(0), 3.425250914);
    EXPECT_DOUBLE_EQ(shell.exponent(1), 0.6239137298);
    EXPECT_DOUBLE_EQ(shell.exponent(2), 0.168855404);

    // Check that coefficients are normalized (self-overlap should be 1)
    const Real overlap = compute_self_overlap(shell);
    EXPECT_TRUE(are_close(overlap, 1.0))
        << "Self-overlap = " << overlap << ", expected 1.0";
}

TEST(ShellTest, ConstructPShell) {
    // P-shell: 3 primitives (STO-3G carbon)
    Point3D center(1.0, 2.0, 3.0);
    std::vector<Real> exponents = {2.941249355, 0.6834830964, 0.2222899159};
    std::vector<Real> coefficients = {0.1559162750, 0.6076837186, 0.3919573931};

    Shell shell(AngularMomentum::P, center, exponents, coefficients);

    EXPECT_EQ(shell.angular_momentum(), 1);
    EXPECT_EQ(shell.angular_momentum_enum(), AngularMomentum::P);
    EXPECT_EQ(shell.n_primitives(), 3);
    EXPECT_EQ(shell.n_functions(), 3);  // Px, Py, Pz
    EXPECT_TRUE(shell.valid());

    // Check center
    EXPECT_DOUBLE_EQ(shell.center().x, 1.0);
    EXPECT_DOUBLE_EQ(shell.center().y, 2.0);
    EXPECT_DOUBLE_EQ(shell.center().z, 3.0);

    // Check normalization
    const Real overlap = compute_self_overlap(shell);
    EXPECT_TRUE(are_close(overlap, 1.0))
        << "Self-overlap = " << overlap << ", expected 1.0";
}

TEST(ShellTest, ConstructDShell) {
    // D-shell: 1 primitive (single Gaussian)
    Point3D center(-1.0, 0.5, -2.0);
    std::vector<Real> exponents = {0.8};
    std::vector<Real> coefficients = {1.0};

    Shell shell(AngularMomentum::D, center, exponents, coefficients);

    EXPECT_EQ(shell.angular_momentum(), 2);
    EXPECT_EQ(shell.angular_momentum_enum(), AngularMomentum::D);
    EXPECT_EQ(shell.n_primitives(), 1);
    EXPECT_EQ(shell.n_functions(), 6);  // xx, yy, zz, xy, xz, yz
    EXPECT_TRUE(shell.valid());

    // Check normalization
    const Real overlap = compute_self_overlap(shell);
    EXPECT_TRUE(are_close(overlap, 1.0))
        << "Self-overlap = " << overlap << ", expected 1.0";
}

TEST(ShellTest, ConstructWithIntAM) {
    // Test construction with int angular momentum
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {1.0};
    std::vector<Real> coefficients = {1.0};

    Shell shell(2, center, exponents, coefficients);  // D-shell

    EXPECT_EQ(shell.angular_momentum(), 2);
    EXPECT_EQ(shell.angular_momentum_enum(), AngularMomentum::D);
}

TEST(ShellTest, ConstructHighAngularMomentum) {
    // Test F and G shells (alpha contract: max L=4)
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {1.0};
    std::vector<Real> coefficients = {1.0};

    Shell f_shell(AngularMomentum::F, center, exponents, coefficients);
    EXPECT_EQ(f_shell.angular_momentum(), 3);
    EXPECT_EQ(f_shell.n_functions(), 10);

    Shell g_shell(AngularMomentum::G, center, exponents, coefficients);
    EXPECT_EQ(g_shell.angular_momentum(), 4);
    EXPECT_EQ(g_shell.n_functions(), 15);
}

// =============================================================================
// Pre-Normalized Constructor Tests
// =============================================================================

TEST(ShellTest, ConstructPreNormalized) {
    // Create a shell with manually normalized coefficients
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {1.0, 0.5};

    // Manually compute normalized coefficients for these exponents
    // (This is a simplified example; in practice these would come from basis set data)
    std::vector<Real> pre_normalized_coeffs = {0.7, 0.3};

    // Construct with pre-normalized tag
    Shell shell(pre_normalized, AngularMomentum::S, center, exponents, pre_normalized_coeffs);

    EXPECT_EQ(shell.angular_momentum(), 0);
    EXPECT_EQ(shell.n_primitives(), 2);

    // Coefficients should be exactly as provided (no additional normalization)
    EXPECT_DOUBLE_EQ(shell.coefficient(0), 0.7);
    EXPECT_DOUBLE_EQ(shell.coefficient(1), 0.3);
}

TEST(ShellTest, ConstructPreNormalizedWithIntAM) {
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {1.0};
    std::vector<Real> pre_normalized_coeffs = {1.0};

    Shell shell(pre_normalized, 1, center, exponents, pre_normalized_coeffs);

    EXPECT_EQ(shell.angular_momentum(), 1);
    EXPECT_DOUBLE_EQ(shell.coefficient(0), 1.0);
}

// =============================================================================
// Invalid Construction Tests
// =============================================================================

TEST(ShellTest, InvalidAngularMomentumNegative) {
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {1.0};
    std::vector<Real> coefficients = {1.0};

    EXPECT_THROW({
        Shell shell(-1, center, exponents, coefficients);
    }, InvalidArgumentException);
}

TEST(ShellTest, InvalidAngularMomentumTooLarge) {
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {1.0};
    std::vector<Real> coefficients = {1.0};

    EXPECT_THROW({
        Shell shell(MAX_ANGULAR_MOMENTUM + 1, center, exponents, coefficients);
    }, InvalidArgumentException);
}

TEST(ShellTest, EmptyPrimitives) {
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {};
    std::vector<Real> coefficients = {};

    EXPECT_THROW({
        Shell shell(AngularMomentum::S, center, exponents, coefficients);
    }, InvalidArgumentException);
}

TEST(ShellTest, MismatchedLengths) {
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {1.0, 2.0};
    std::vector<Real> coefficients = {1.0};  // Mismatch: 2 vs 1

    EXPECT_THROW({
        Shell shell(AngularMomentum::S, center, exponents, coefficients);
    }, InvalidArgumentException);
}

TEST(ShellTest, NonPositiveExponent) {
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {1.0, 0.0};  // Zero exponent
    std::vector<Real> coefficients = {1.0, 1.0};

    EXPECT_THROW({
        Shell shell(AngularMomentum::S, center, exponents, coefficients);
    }, InvalidArgumentException);
}

TEST(ShellTest, NegativeExponent) {
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {1.0, -0.5};  // Negative exponent
    std::vector<Real> coefficients = {1.0, 1.0};

    EXPECT_THROW({
        Shell shell(AngularMomentum::S, center, exponents, coefficients);
    }, InvalidArgumentException);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST(ShellTest, ExponentAccessor) {
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {3.0, 2.0, 1.0};
    std::vector<Real> coefficients = {0.3, 0.4, 0.3};

    Shell shell(AngularMomentum::S, center, exponents, coefficients);

    EXPECT_DOUBLE_EQ(shell.exponent(0), 3.0);
    EXPECT_DOUBLE_EQ(shell.exponent(1), 2.0);
    EXPECT_DOUBLE_EQ(shell.exponent(2), 1.0);

    // Test out of bounds
    EXPECT_THROW(shell.exponent(3), InvalidArgumentException);
}

TEST(ShellTest, CoefficientAccessor) {
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {1.0, 2.0};
    std::vector<Real> coefficients = {0.5, 0.5};

    Shell shell(AngularMomentum::S, center, exponents, coefficients);

    // Coefficients are normalized, so they won't be exactly 0.5
    EXPECT_GT(shell.coefficient(0), 0.0);
    EXPECT_GT(shell.coefficient(1), 0.0);

    // Test out of bounds
    EXPECT_THROW(shell.coefficient(2), InvalidArgumentException);
}

TEST(ShellTest, SpanAccessors) {
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {3.0, 2.0, 1.0};
    std::vector<Real> coefficients = {0.3, 0.4, 0.3};

    Shell shell(AngularMomentum::S, center, exponents, coefficients);

    auto exp_span = shell.exponents();
    EXPECT_EQ(exp_span.size(), 3);
    EXPECT_DOUBLE_EQ(exp_span[0], 3.0);
    EXPECT_DOUBLE_EQ(exp_span[1], 2.0);
    EXPECT_DOUBLE_EQ(exp_span[2], 1.0);

    auto coeff_span = shell.coefficients();
    EXPECT_EQ(coeff_span.size(), 3);
}

TEST(ShellTest, DefaultConstructor) {
    Shell shell;

    EXPECT_FALSE(shell.valid());
    EXPECT_EQ(shell.angular_momentum(), 0);
    EXPECT_EQ(shell.n_primitives(), 0);
}

// =============================================================================
// Tracking Indices Tests
// =============================================================================

TEST(ShellTest, TrackingIndices) {
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {1.0};
    std::vector<Real> coefficients = {1.0};

    Shell shell(AngularMomentum::S, center, exponents, coefficients);

    // Default values should be -1
    EXPECT_EQ(shell.atom_index(), -1);
    EXPECT_EQ(shell.shell_index(), -1);
    EXPECT_EQ(shell.function_index(), -1);

    // Set and get
    shell.set_atom_index(5);
    shell.set_shell_index(10);
    shell.set_function_index(42);

    EXPECT_EQ(shell.atom_index(), 5);
    EXPECT_EQ(shell.shell_index(), 10);
    EXPECT_EQ(shell.function_index(), 42);
}

// =============================================================================
// Normalization Validation Tests
// =============================================================================

TEST(ShellTest, NormalizationSShell) {
    // Test that S-shells are properly normalized
    Point3D center(0.0, 0.0, 0.0);

    // Single primitive
    {
        Shell shell(AngularMomentum::S, center, {1.0}, {1.0});
        const Real overlap = compute_self_overlap(shell);
        EXPECT_TRUE(are_close(overlap, 1.0, 1e-10))
            << "1-prim S-shell self-overlap = " << overlap << ", expected 1.0";
    }
    // Two primitives
    {
        Shell shell(AngularMomentum::S, center, {2.0, 0.5}, {0.6, 0.4});
        const Real overlap = compute_self_overlap(shell);
        EXPECT_TRUE(are_close(overlap, 1.0, 1e-10))
            << "2-prim S-shell self-overlap = " << overlap << ", expected 1.0";
    }
    // Three primitives
    {
        Shell shell(AngularMomentum::S, center, {3.0, 1.5, 0.5}, {0.5, 0.3, 0.2});
        const Real overlap = compute_self_overlap(shell);
        EXPECT_TRUE(are_close(overlap, 1.0, 1e-10))
            << "3-prim S-shell self-overlap = " << overlap << ", expected 1.0";
    }
}

TEST(ShellTest, NormalizationPShell) {
    // Test that P-shells are properly normalized
    Point3D center(0.0, 0.0, 0.0);

    std::vector<Real> exponents = {2.0, 0.8};
    std::vector<Real> coefficients = {0.7, 0.3};

    Shell shell(AngularMomentum::P, center, exponents, coefficients);

    const Real overlap = compute_self_overlap(shell);
    EXPECT_TRUE(are_close(overlap, 1.0, 1e-10))
        << "P-shell self-overlap = " << overlap << ", expected 1.0";
}

TEST(ShellTest, NormalizationDShell) {
    // Test that D-shells are properly normalized
    Point3D center(0.0, 0.0, 0.0);

    std::vector<Real> exponents = {1.5, 0.6, 0.2};
    std::vector<Real> coefficients = {0.4, 0.4, 0.2};

    Shell shell(AngularMomentum::D, center, exponents, coefficients);

    const Real overlap = compute_self_overlap(shell);
    EXPECT_TRUE(are_close(overlap, 1.0, 1e-10))
        << "D-shell self-overlap = " << overlap << ", expected 1.0";
}

TEST(ShellTest, NormalizationHigherAM) {
    // Test F and G shells (alpha contract: max L=4)
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {1.0};
    std::vector<Real> coefficients = {1.0};

    for (int am = 3; am <= 4; ++am) {
        Shell shell(am, center, exponents, coefficients);
        const Real overlap = compute_self_overlap(shell);
        EXPECT_TRUE(are_close(overlap, 1.0, 1e-10))
            << "AM=" << am << " shell self-overlap = " << overlap << ", expected 1.0";
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(ShellTest, SinglePrimitiveNormalization) {
    // A shell with a single primitive should still be normalized
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {0.5};
    std::vector<Real> coefficients = {2.0};  // Un-normalized

    Shell shell(AngularMomentum::S, center, exponents, coefficients);

    const Real overlap = compute_self_overlap(shell);
    EXPECT_TRUE(are_close(overlap, 1.0))
        << "Single-primitive self-overlap = " << overlap << ", expected 1.0";
}

TEST(ShellTest, VeryLargeExponents) {
    // Test with very large exponents
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {100.0, 50.0};
    std::vector<Real> coefficients = {0.5, 0.5};

    Shell shell(AngularMomentum::S, center, exponents, coefficients);

    const Real overlap = compute_self_overlap(shell);
    EXPECT_TRUE(are_close(overlap, 1.0, 1e-10))
        << "Large-exponent self-overlap = " << overlap << ", expected 1.0";
}

TEST(ShellTest, VerySmallExponents) {
    // Test with very small exponents
    Point3D center(0.0, 0.0, 0.0);
    std::vector<Real> exponents = {0.01, 0.005};
    std::vector<Real> coefficients = {0.5, 0.5};

    Shell shell(AngularMomentum::S, center, exponents, coefficients);

    const Real overlap = compute_self_overlap(shell);
    EXPECT_TRUE(are_close(overlap, 1.0, 1e-10))
        << "Small-exponent self-overlap = " << overlap << ", expected 1.0";
}
