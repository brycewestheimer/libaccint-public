// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/math/gaussian_product.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace libaccint;
using namespace libaccint::math;

namespace {

// Test tolerance for floating-point comparisons
constexpr Real TOLERANCE = 1e-12;

// Helper function to compare Real values with tolerance
bool are_close(Real a, Real b, Real tol = TOLERANCE) {
    return std::abs(a - b) <= tol;
}

// Helper function to compare Point3D values with tolerance
bool are_close(const Point3D& a, const Point3D& b, Real tol = TOLERANCE) {
    return are_close(a.x, b.x, tol) &&
           are_close(a.y, b.y, tol) &&
           are_close(a.z, b.z, tol);
}

}  // anonymous namespace

// =============================================================================
// Single Product Tests
// =============================================================================

TEST(GaussianProductTest, SameCenterIdentity) {
    // When A == B, the product center should equal A (and B)
    // and K_AB should be 1.0 (no displacement)
    const Real alpha = 1.5;
    const Real beta = 2.3;
    const Point3D A{1.0, 2.0, 3.0};
    const Point3D B = A;  // Same center

    const auto result = compute_gaussian_product(alpha, A, beta, B);

    EXPECT_TRUE(are_close(result.P, A)) << "Product center should equal input center";
    EXPECT_NEAR(result.zeta, alpha + beta, TOLERANCE) << "zeta = alpha + beta";
    EXPECT_NEAR(result.mu, alpha * beta / (alpha + beta), TOLERANCE) << "mu = alpha*beta/(alpha+beta)";
    EXPECT_NEAR(result.K_AB, 1.0, TOLERANCE) << "K_AB should be 1.0 for same center";
}

TEST(GaussianProductTest, KnownValues) {
    // Test case: alpha = 1.0, beta = 1.0, A = (0,0,0), B = (1,0,0)
    // Expected:
    //   zeta = 2.0
    //   mu = 0.5
    //   P = (0.5, 0, 0)
    //   |A-B|² = 1.0
    //   K_AB = exp(-0.5 * 1.0) = exp(-0.5)
    const Real alpha = 1.0;
    const Real beta = 1.0;
    const Point3D A{0.0, 0.0, 0.0};
    const Point3D B{1.0, 0.0, 0.0};

    const auto result = compute_gaussian_product(alpha, A, beta, B);

    EXPECT_NEAR(result.zeta, 2.0, TOLERANCE);
    EXPECT_NEAR(result.mu, 0.5, TOLERANCE);
    EXPECT_TRUE(are_close(result.P, Point3D{0.5, 0.0, 0.0}));
    EXPECT_NEAR(result.K_AB, std::exp(-0.5), TOLERANCE);
}

TEST(GaussianProductTest, Symmetry) {
    // compute_gaussian_product(alpha, A, beta, B) should give
    // same zeta, mu, and K_AB as compute_gaussian_product(beta, B, alpha, A)
    // Product center will differ but be consistent
    const Real alpha = 1.2;
    const Real beta = 2.5;
    const Point3D A{1.0, 2.0, 3.0};
    const Point3D B{-1.0, 0.5, 2.0};

    const auto result_AB = compute_gaussian_product(alpha, A, beta, B);
    const auto result_BA = compute_gaussian_product(beta, B, alpha, A);

    EXPECT_NEAR(result_AB.zeta, result_BA.zeta, TOLERANCE) << "zeta should be symmetric";
    EXPECT_NEAR(result_AB.mu, result_BA.mu, TOLERANCE) << "mu should be symmetric";
    EXPECT_NEAR(result_AB.K_AB, result_BA.K_AB, TOLERANCE) << "K_AB should be symmetric";
    EXPECT_TRUE(are_close(result_AB.P, result_BA.P)) << "Product centers should match";
}

TEST(GaussianProductTest, ExponentRatioAlphaMuchLarger) {
    // When alpha >> beta, product center should be close to A
    const Real alpha = 100.0;
    const Real beta = 1.0;
    const Point3D A{0.0, 0.0, 0.0};
    const Point3D B{10.0, 0.0, 0.0};

    const auto result = compute_gaussian_product(alpha, A, beta, B);

    // P ≈ (100*0 + 1*10) / 101 ≈ 0.099
    // Should be much closer to A than to B
    const Real distance_to_A = result.P.distance_squared(A);
    const Real distance_to_B = result.P.distance_squared(B);
    EXPECT_LT(distance_to_A, distance_to_B) << "Product center should be closer to A";
    EXPECT_LT(distance_to_A, 1.0) << "Product center should be very close to A";
}

TEST(GaussianProductTest, ExponentRatioBetaMuchLarger) {
    // When beta >> alpha, product center should be close to B
    const Real alpha = 1.0;
    const Real beta = 100.0;
    const Point3D A{0.0, 0.0, 0.0};
    const Point3D B{10.0, 0.0, 0.0};

    const auto result = compute_gaussian_product(alpha, A, beta, B);

    // P ≈ (1*0 + 100*10) / 101 ≈ 9.901
    // Should be much closer to B than to A
    const Real distance_to_A = result.P.distance_squared(A);
    const Real distance_to_B = result.P.distance_squared(B);
    EXPECT_LT(distance_to_B, distance_to_A) << "Product center should be closer to B";
    EXPECT_LT(distance_to_B, 1.0) << "Product center should be very close to B";
}

TEST(GaussianProductTest, EqualExponents) {
    // When alpha == beta, product center should be midpoint of A and B
    const Real alpha = 2.5;
    const Real beta = 2.5;
    const Point3D A{1.0, 2.0, 3.0};
    const Point3D B{5.0, 6.0, 7.0};

    const auto result = compute_gaussian_product(alpha, A, beta, B);

    const Point3D midpoint{(A.x + B.x) / 2.0, (A.y + B.y) / 2.0, (A.z + B.z) / 2.0};
    EXPECT_TRUE(are_close(result.P, midpoint)) << "Product center should be midpoint";
}

TEST(GaussianProductTest, ThreeDimensionalCase) {
    // General 3D case with all coordinates non-zero
    const Real alpha = 1.5;
    const Real beta = 2.0;
    const Point3D A{1.0, -2.0, 3.5};
    const Point3D B{-0.5, 1.0, -1.0};

    const auto result = compute_gaussian_product(alpha, A, beta, B);

    // Manual calculation
    const Real expected_zeta = 3.5;
    const Real expected_mu = 1.5 * 2.0 / 3.5;
    const Point3D expected_P{
        (1.5 * 1.0 + 2.0 * (-0.5)) / 3.5,
        (1.5 * (-2.0) + 2.0 * 1.0) / 3.5,
        (1.5 * 3.5 + 2.0 * (-1.0)) / 3.5
    };
    const Real AB_squared = A.distance_squared(B);
    const Real expected_K_AB = std::exp(-expected_mu * AB_squared);

    EXPECT_NEAR(result.zeta, expected_zeta, TOLERANCE);
    EXPECT_NEAR(result.mu, expected_mu, TOLERANCE);
    EXPECT_TRUE(are_close(result.P, expected_P));
    EXPECT_NEAR(result.K_AB, expected_K_AB, TOLERANCE);
}

// =============================================================================
// Batch Computation Tests
// =============================================================================

TEST(GaussianProductBatchTest, SingleProductMatchesSingle) {
    // Batch computation with n=1 should match single computation
    const Real alpha = 1.5;
    const Real beta = 2.0;
    const Point3D A{1.0, 2.0, 3.0};
    const Point3D B{4.0, 5.0, 6.0};

    // Single computation
    const auto single_result = compute_gaussian_product(alpha, A, beta, B);

    // Batch computation
    std::vector<Real> alphas{alpha};
    std::vector<Real> A_x{A.x}, A_y{A.y}, A_z{A.z};
    std::vector<Real> betas{beta};
    std::vector<Real> B_x{B.x}, B_y{B.y}, B_z{B.z};
    std::vector<Real> zetas(1), mus(1);
    std::vector<Real> P_x(1), P_y(1), P_z(1);
    std::vector<Real> K_AB(1);

    compute_gaussian_products_batch(
        1,
        alphas.data(), A_x.data(), A_y.data(), A_z.data(),
        betas.data(), B_x.data(), B_y.data(), B_z.data(),
        zetas.data(), mus.data(),
        P_x.data(), P_y.data(), P_z.data(),
        K_AB.data());

    EXPECT_NEAR(zetas[0], single_result.zeta, TOLERANCE);
    EXPECT_NEAR(mus[0], single_result.mu, TOLERANCE);
    EXPECT_NEAR(P_x[0], single_result.P.x, TOLERANCE);
    EXPECT_NEAR(P_y[0], single_result.P.y, TOLERANCE);
    EXPECT_NEAR(P_z[0], single_result.P.z, TOLERANCE);
    EXPECT_NEAR(K_AB[0], single_result.K_AB, TOLERANCE);
}

TEST(GaussianProductBatchTest, MultipleProducts) {
    // Test batch computation with multiple products
    const Size n = 5;
    std::vector<Real> alphas{1.0, 2.0, 1.5, 3.0, 0.5};
    std::vector<Real> A_x{0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<Real> A_y{0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<Real> A_z{0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<Real> betas{1.0, 1.0, 2.5, 1.5, 2.0};
    std::vector<Real> B_x{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<Real> B_y{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<Real> B_z{1.0, 2.0, 3.0, 4.0, 5.0};

    std::vector<Real> zetas(n), mus(n);
    std::vector<Real> P_x(n), P_y(n), P_z(n);
    std::vector<Real> K_AB(n);

    compute_gaussian_products_batch(
        n,
        alphas.data(), A_x.data(), A_y.data(), A_z.data(),
        betas.data(), B_x.data(), B_y.data(), B_z.data(),
        zetas.data(), mus.data(),
        P_x.data(), P_y.data(), P_z.data(),
        K_AB.data());

    // Verify each product against single computation
    for (Size i = 0; i < n; ++i) {
        const Point3D A{A_x[i], A_y[i], A_z[i]};
        const Point3D B{B_x[i], B_y[i], B_z[i]};
        const auto single_result = compute_gaussian_product(alphas[i], A, betas[i], B);

        EXPECT_NEAR(zetas[i], single_result.zeta, TOLERANCE) << "Mismatch at index " << i;
        EXPECT_NEAR(mus[i], single_result.mu, TOLERANCE) << "Mismatch at index " << i;
        EXPECT_NEAR(P_x[i], single_result.P.x, TOLERANCE) << "Mismatch at index " << i;
        EXPECT_NEAR(P_y[i], single_result.P.y, TOLERANCE) << "Mismatch at index " << i;
        EXPECT_NEAR(P_z[i], single_result.P.z, TOLERANCE) << "Mismatch at index " << i;
        EXPECT_NEAR(K_AB[i], single_result.K_AB, TOLERANCE) << "Mismatch at index " << i;
    }
}

TEST(GaussianProductBatchTest, ZeroProducts) {
    // Edge case: n_products = 0 should not crash
    std::vector<Real> dummy(1, 0.0);

    compute_gaussian_products_batch(
        0,
        dummy.data(), dummy.data(), dummy.data(), dummy.data(),
        dummy.data(), dummy.data(), dummy.data(), dummy.data(),
        dummy.data(), dummy.data(),
        dummy.data(), dummy.data(), dummy.data(),
        dummy.data());

    // If we get here without crashing, test passes
    SUCCEED();
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(GaussianProductTest, SmallExponents) {
    // Test with very small exponents (diffuse functions)
    const Real alpha = 0.01;
    const Real beta = 0.02;
    const Point3D A{0.0, 0.0, 0.0};
    const Point3D B{10.0, 0.0, 0.0};

    const auto result = compute_gaussian_product(alpha, A, beta, B);

    EXPECT_NEAR(result.zeta, 0.03, TOLERANCE);
    EXPECT_NEAR(result.mu, alpha * beta / (alpha + beta), TOLERANCE);
    EXPECT_GT(result.K_AB, 0.0) << "K_AB should be positive";
    EXPECT_LE(result.K_AB, 1.0) << "K_AB should not exceed 1.0";
}

TEST(GaussianProductTest, LargeExponents) {
    // Test with very large exponents (tight functions)
    const Real alpha = 100.0;
    const Real beta = 200.0;
    const Point3D A{0.0, 0.0, 0.0};
    const Point3D B{1.0, 0.0, 0.0};

    const auto result = compute_gaussian_product(alpha, A, beta, B);

    EXPECT_NEAR(result.zeta, 300.0, TOLERANCE);
    EXPECT_NEAR(result.mu, alpha * beta / (alpha + beta), TOLERANCE);
    EXPECT_GT(result.K_AB, 0.0) << "K_AB should be positive";
    EXPECT_LE(result.K_AB, 1.0) << "K_AB should not exceed 1.0";
    // For large exponents and non-zero displacement, K_AB should be small
    EXPECT_LT(result.K_AB, 1e-10) << "K_AB should be very small for large exponents and displacement";
}

TEST(GaussianProductTest, LargeDisplacement) {
    // Test with large separation between centers
    const Real alpha = 1.0;
    const Real beta = 1.0;
    const Point3D A{0.0, 0.0, 0.0};
    const Point3D B{100.0, 0.0, 0.0};

    const auto result = compute_gaussian_product(alpha, A, beta, B);

    EXPECT_NEAR(result.zeta, 2.0, TOLERANCE);
    EXPECT_GE(result.K_AB, 0.0) << "K_AB should be non-negative";
    EXPECT_LE(result.K_AB, 1.0) << "K_AB should not exceed 1.0";
    // For large displacement, K_AB underflows to 0 (exp(-5000) ≈ 0 in double precision)
    EXPECT_LT(result.K_AB, 1e-20) << "K_AB should be extremely small for large displacement";
}

// =============================================================================
// Task 3.3.7: SIMD Batch Verification Tests
// =============================================================================

TEST(GaussianProductBatchTest, Batch4ExactlyMatchesSingle) {
    // 4 is the AVX2 SIMD width - tests full SIMD path with no remainder
    const Size n = 4;
    std::vector<Real> alphas{1.0, 2.5, 0.5, 3.0};
    std::vector<Real> A_x{0.0, 1.0, -1.0, 2.0};
    std::vector<Real> A_y{0.0, 1.0,  0.5, -1.0};
    std::vector<Real> A_z{0.0, 0.0,  2.0,  1.0};
    std::vector<Real> betas{1.0, 1.5, 2.0, 0.5};
    std::vector<Real> B_x{1.0, -1.0, 3.0, 0.0};
    std::vector<Real> B_y{1.0,  2.0, -1.0, 1.0};
    std::vector<Real> B_z{1.0,  3.0,  0.0, -2.0};

    std::vector<Real> zetas(n), mus(n);
    std::vector<Real> P_x(n), P_y(n), P_z(n);
    std::vector<Real> K_AB(n);

    compute_gaussian_products_batch(
        n,
        alphas.data(), A_x.data(), A_y.data(), A_z.data(),
        betas.data(), B_x.data(), B_y.data(), B_z.data(),
        zetas.data(), mus.data(),
        P_x.data(), P_y.data(), P_z.data(),
        K_AB.data());

    for (Size i = 0; i < n; ++i) {
        const Point3D A{A_x[i], A_y[i], A_z[i]};
        const Point3D B{B_x[i], B_y[i], B_z[i]};
        const auto ref = compute_gaussian_product(alphas[i], A, betas[i], B);

        EXPECT_NEAR(zetas[i], ref.zeta, TOLERANCE) << "zeta mismatch at " << i;
        EXPECT_NEAR(mus[i], ref.mu, TOLERANCE) << "mu mismatch at " << i;
        EXPECT_NEAR(P_x[i], ref.P.x, TOLERANCE) << "P_x mismatch at " << i;
        EXPECT_NEAR(P_y[i], ref.P.y, TOLERANCE) << "P_y mismatch at " << i;
        EXPECT_NEAR(P_z[i], ref.P.z, TOLERANCE) << "P_z mismatch at " << i;
        EXPECT_NEAR(K_AB[i], ref.K_AB, TOLERANCE) << "K_AB mismatch at " << i;
    }
}

TEST(GaussianProductBatchTest, Batch8MatchesSingle) {
    // 8 = 2 * SIMD width: tests multiple full SIMD iterations
    const Size n = 8;
    std::vector<Real> alphas(n), betas(n);
    std::vector<Real> A_x(n), A_y(n), A_z(n);
    std::vector<Real> B_x(n), B_y(n), B_z(n);

    for (Size i = 0; i < n; ++i) {
        alphas[i] = 0.5 + i * 0.3;
        betas[i] = 1.0 + i * 0.2;
        A_x[i] = static_cast<Real>(i) * 0.5;
        A_y[i] = static_cast<Real>(i) * 0.3 - 1.0;
        A_z[i] = static_cast<Real>(i) * 0.1;
        B_x[i] = -static_cast<Real>(i) * 0.4 + 1.0;
        B_y[i] = static_cast<Real>(i) * 0.2;
        B_z[i] = static_cast<Real>(i) * 0.6 - 0.5;
    }

    std::vector<Real> zetas(n), mus(n);
    std::vector<Real> P_x(n), P_y(n), P_z(n);
    std::vector<Real> K_AB(n);

    compute_gaussian_products_batch(
        n,
        alphas.data(), A_x.data(), A_y.data(), A_z.data(),
        betas.data(), B_x.data(), B_y.data(), B_z.data(),
        zetas.data(), mus.data(),
        P_x.data(), P_y.data(), P_z.data(),
        K_AB.data());

    for (Size i = 0; i < n; ++i) {
        const Point3D A{A_x[i], A_y[i], A_z[i]};
        const Point3D B{B_x[i], B_y[i], B_z[i]};
        const auto ref = compute_gaussian_product(alphas[i], A, betas[i], B);

        EXPECT_NEAR(zetas[i], ref.zeta, TOLERANCE) << "i=" << i;
        EXPECT_NEAR(mus[i], ref.mu, TOLERANCE) << "i=" << i;
        EXPECT_NEAR(P_x[i], ref.P.x, TOLERANCE) << "i=" << i;
        EXPECT_NEAR(P_y[i], ref.P.y, TOLERANCE) << "i=" << i;
        EXPECT_NEAR(P_z[i], ref.P.z, TOLERANCE) << "i=" << i;
        EXPECT_NEAR(K_AB[i], ref.K_AB, TOLERANCE) << "i=" << i;
    }
}

TEST(GaussianProductBatchTest, Batch7NonMultipleOfSIMDWidth) {
    // 7 = SIMD_width(4) + 3 remainder: tests SIMD path + scalar remainder
    const Size n = 7;
    std::vector<Real> alphas(n), betas(n);
    std::vector<Real> A_x(n), A_y(n), A_z(n);
    std::vector<Real> B_x(n), B_y(n), B_z(n);

    for (Size i = 0; i < n; ++i) {
        alphas[i] = 1.0 + i * 0.5;
        betas[i] = 0.5 + i * 0.3;
        A_x[i] = std::sin(static_cast<double>(i));
        A_y[i] = std::cos(static_cast<double>(i));
        A_z[i] = static_cast<double>(i) * 0.7;
        B_x[i] = -std::cos(static_cast<double>(i));
        B_y[i] = std::sin(static_cast<double>(i));
        B_z[i] = -static_cast<double>(i) * 0.3 + 1.0;
    }

    std::vector<Real> zetas(n), mus(n);
    std::vector<Real> P_x(n), P_y(n), P_z(n);
    std::vector<Real> K_AB(n);

    compute_gaussian_products_batch(
        n,
        alphas.data(), A_x.data(), A_y.data(), A_z.data(),
        betas.data(), B_x.data(), B_y.data(), B_z.data(),
        zetas.data(), mus.data(),
        P_x.data(), P_y.data(), P_z.data(),
        K_AB.data());

    for (Size i = 0; i < n; ++i) {
        const Point3D A{A_x[i], A_y[i], A_z[i]};
        const Point3D B{B_x[i], B_y[i], B_z[i]};
        const auto ref = compute_gaussian_product(alphas[i], A, betas[i], B);

        EXPECT_NEAR(zetas[i], ref.zeta, TOLERANCE) << "i=" << i;
        EXPECT_NEAR(mus[i], ref.mu, TOLERANCE) << "i=" << i;
        EXPECT_NEAR(P_x[i], ref.P.x, TOLERANCE) << "i=" << i;
        EXPECT_NEAR(P_y[i], ref.P.y, TOLERANCE) << "i=" << i;
        EXPECT_NEAR(P_z[i], ref.P.z, TOLERANCE) << "i=" << i;
        EXPECT_NEAR(K_AB[i], ref.K_AB, TOLERANCE) << "i=" << i;
    }
}

TEST(GaussianProductBatchTest, Batch1MatchesSingle) {
    // Single-element batch (scalar remainder only, no SIMD iterations)
    const Size n = 1;
    std::vector<Real> alphas{2.0}, betas{3.0};
    std::vector<Real> A_x{1.0}, A_y{2.0}, A_z{3.0};
    std::vector<Real> B_x{-1.0}, B_y{0.0}, B_z{1.0};

    std::vector<Real> zetas(n), mus(n);
    std::vector<Real> P_x(n), P_y(n), P_z(n);
    std::vector<Real> K_AB(n);

    compute_gaussian_products_batch(
        n,
        alphas.data(), A_x.data(), A_y.data(), A_z.data(),
        betas.data(), B_x.data(), B_y.data(), B_z.data(),
        zetas.data(), mus.data(),
        P_x.data(), P_y.data(), P_z.data(),
        K_AB.data());

    const auto ref = compute_gaussian_product(2.0, Point3D{1.0, 2.0, 3.0},
                                              3.0, Point3D{-1.0, 0.0, 1.0});

    EXPECT_NEAR(zetas[0], ref.zeta, TOLERANCE);
    EXPECT_NEAR(mus[0], ref.mu, TOLERANCE);
    EXPECT_NEAR(P_x[0], ref.P.x, TOLERANCE);
    EXPECT_NEAR(P_y[0], ref.P.y, TOLERANCE);
    EXPECT_NEAR(P_z[0], ref.P.z, TOLERANCE);
    EXPECT_NEAR(K_AB[0], ref.K_AB, TOLERANCE);
}

TEST(GaussianProductBatchTest, LargeBatchMatchesSingle) {
    // Test with a large batch to exercise multiple SIMD iterations
    const Size n = 100;
    std::vector<Real> alphas(n), betas(n);
    std::vector<Real> A_x(n), A_y(n), A_z(n);
    std::vector<Real> B_x(n), B_y(n), B_z(n);

    for (Size i = 0; i < n; ++i) {
        double d = static_cast<double>(i);
        alphas[i] = 0.1 + d * 0.05;
        betas[i] = 0.2 + d * 0.03;
        A_x[i] = std::sin(d * 0.7);
        A_y[i] = std::cos(d * 0.3);
        A_z[i] = std::sin(d * 1.1) * 2.0;
        B_x[i] = std::cos(d * 0.5) * 3.0;
        B_y[i] = std::sin(d * 0.9);
        B_z[i] = std::cos(d * 1.3) * 1.5;
    }

    std::vector<Real> zetas(n), mus(n);
    std::vector<Real> P_x(n), P_y(n), P_z(n);
    std::vector<Real> K_AB(n);

    compute_gaussian_products_batch(
        n,
        alphas.data(), A_x.data(), A_y.data(), A_z.data(),
        betas.data(), B_x.data(), B_y.data(), B_z.data(),
        zetas.data(), mus.data(),
        P_x.data(), P_y.data(), P_z.data(),
        K_AB.data());

    for (Size i = 0; i < n; ++i) {
        const Point3D A{A_x[i], A_y[i], A_z[i]};
        const Point3D B{B_x[i], B_y[i], B_z[i]};
        const auto ref = compute_gaussian_product(alphas[i], A, betas[i], B);

        EXPECT_NEAR(zetas[i], ref.zeta, TOLERANCE) << "i=" << i;
        EXPECT_NEAR(mus[i], ref.mu, TOLERANCE) << "i=" << i;
        EXPECT_NEAR(P_x[i], ref.P.x, TOLERANCE) << "i=" << i;
        EXPECT_NEAR(P_y[i], ref.P.y, TOLERANCE) << "i=" << i;
        EXPECT_NEAR(P_z[i], ref.P.z, TOLERANCE) << "i=" << i;
        EXPECT_NEAR(K_AB[i], ref.K_AB, TOLERANCE) << "i=" << i;
    }
}
