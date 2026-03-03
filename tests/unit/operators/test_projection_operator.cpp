// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_projection_operator.cpp
/// @brief Validation tests for projection operator construction

#include <libaccint/operators/projection_operator.hpp>
#include <libaccint/operators/operator_types.hpp>
#include <gtest/gtest.h>
#include <cmath>

using namespace libaccint;

namespace {
constexpr double TOL = 1e-12;
}

// ============================================================================
// Test 1: Identity-like projection (single projector, unit coefficient)
// ============================================================================
TEST(ProjectionOperatorTest, SingleProjectorSingleBasis) {
    // 1×1 case: P = c * w * c = w
    ProjectionOperatorParams params;
    params.n_basis = 1;
    params.n_projectors = 1;
    params.coefficients = {1.0};
    params.weights = {1.0};

    EXPECT_TRUE(params.is_valid());

    auto P = build_projection_matrix(params);
    ASSERT_EQ(P.size(), 1);
    EXPECT_NEAR(P[0], 1.0, TOL);
}

// ============================================================================
// Test 2: 2×2 projection from 2 projectors
// ============================================================================
TEST(ProjectionOperatorTest, TwoProjectorsTwoBasis) {
    // C = [[1, 0], [0, 1]], w = [1, 1]
    // P = C * diag(w) * C^T = I
    ProjectionOperatorParams params;
    params.n_basis = 2;
    params.n_projectors = 2;
    params.coefficients = {1.0, 0.0, 0.0, 1.0};  // column-major: C(0,0)=1, C(1,0)=0, C(0,1)=0, C(1,1)=1
    params.weights = {1.0, 1.0};

    auto P = build_projection_matrix(params);
    ASSERT_EQ(P.size(), 4);

    EXPECT_NEAR(P[0], 1.0, TOL);  // P(0,0)
    EXPECT_NEAR(P[1], 0.0, TOL);  // P(0,1)
    EXPECT_NEAR(P[2], 0.0, TOL);  // P(1,0)
    EXPECT_NEAR(P[3], 1.0, TOL);  // P(1,1)
}

// ============================================================================
// Test 3: General projection — trace equals sum of weights
// ============================================================================
TEST(ProjectionOperatorTest, TraceEqualsWeightSum) {
    // P = C * diag(w) * C^T → tr(P) = Σ_k w_k * Σ_μ C(μ,k)² = Σ_k w_k * ||C_k||²
    // For orthonormal columns (||C_k||² = 1): tr(P) = Σ w_k
    ProjectionOperatorParams params;
    params.n_basis = 3;
    params.n_projectors = 2;
    // Two orthonormal columns in 3D: col0 = (1,0,0), col1 = (0,1,0)
    // Column-major: C(0,0), C(1,0), C(2,0), C(0,1), C(1,1), C(2,1)
    params.coefficients = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    params.weights = {2.0, 3.0};

    auto P = build_projection_matrix(params);
    ASSERT_EQ(P.size(), 9);

    Real trace = P[0] + P[4] + P[8];  // P(0,0) + P(1,1) + P(2,2)
    EXPECT_NEAR(trace, 5.0, TOL) << "trace should equal sum of weights for orthonormal columns";
}

// ============================================================================
// Test 4: Symmetry verification
// ============================================================================
TEST(ProjectionOperatorTest, Symmetry) {
    ProjectionOperatorParams params;
    params.n_basis = 3;
    params.n_projectors = 2;
    // Non-trivial coefficients (still column-major)
    params.coefficients = {0.6, 0.8, 0.0, 0.0, 0.0, 1.0};
    params.weights = {1.0, 2.0};

    auto P = build_projection_matrix(params);
    ASSERT_EQ(P.size(), 9);

    EXPECT_TRUE(verify_projection_matrix(P, 3, TOL));

    // Check that off-diagonals are symmetric
    EXPECT_NEAR(P[0 * 3 + 1], P[1 * 3 + 0], TOL);
    EXPECT_NEAR(P[0 * 3 + 2], P[2 * 3 + 0], TOL);
    EXPECT_NEAR(P[1 * 3 + 2], P[2 * 3 + 1], TOL);
}

// ============================================================================
// Test 5: Idempotency for orthonormal projectors with unit weights
// ============================================================================
TEST(ProjectionOperatorTest, Idempotency) {
    // P = C * I * C^T = C * C^T with orthonormal C
    // Then P² = P (projector is idempotent)
    ProjectionOperatorParams params;
    params.n_basis = 3;
    params.n_projectors = 2;
    // Orthonormal columns
    params.coefficients = {1.0, 0.0, 0.0,   0.0, 1.0, 0.0};
    params.weights = {1.0, 1.0};

    auto P = build_projection_matrix(params);

    // Compute P * P
    Size n = 3;
    std::vector<Real> P2(n * n, 0.0);
    for (Size i = 0; i < n; ++i) {
        for (Size j = 0; j < n; ++j) {
            for (Size k = 0; k < n; ++k) {
                P2[i * n + j] += P[i * n + k] * P[k * n + j];
            }
        }
    }

    for (Size i = 0; i < n * n; ++i) {
        EXPECT_NEAR(P[i], P2[i], TOL) << "P² should equal P for orthonormal projectors, element " << i;
    }
}

// ============================================================================
// Test 6: Params validation
// ============================================================================
TEST(ProjectionOperatorTest, ParamsValidation) {
    ProjectionOperatorParams good;
    good.n_basis = 2;
    good.n_projectors = 1;
    good.coefficients = {0.7, 0.7};
    good.weights = {1.0};
    EXPECT_TRUE(good.is_valid());

    ProjectionOperatorParams bad_coefficients;
    bad_coefficients.n_basis = 2;
    bad_coefficients.n_projectors = 1;
    bad_coefficients.coefficients = {0.7};  // wrong size
    bad_coefficients.weights = {1.0};
    EXPECT_FALSE(bad_coefficients.is_valid());

    ProjectionOperatorParams bad_weights;
    bad_weights.n_basis = 2;
    bad_weights.n_projectors = 2;
    bad_weights.coefficients = {1.0, 0.0, 0.0, 1.0};
    bad_weights.weights = {1.0};  // wrong size
    EXPECT_FALSE(bad_weights.is_valid());
}

// ============================================================================
// Test 7: Coefficient accessor
// ============================================================================
TEST(ProjectionOperatorTest, CoefficientAccessor) {
    ProjectionOperatorParams params;
    params.n_basis = 3;
    params.n_projectors = 2;
    // Column-major: C(0,0)=1, C(1,0)=2, C(2,0)=3, C(0,1)=4, C(1,1)=5, C(2,1)=6
    params.coefficients = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    params.weights = {1.0, 1.0};

    EXPECT_NEAR(params.coefficient(0, 0), 1.0, TOL);
    EXPECT_NEAR(params.coefficient(1, 0), 2.0, TOL);
    EXPECT_NEAR(params.coefficient(2, 0), 3.0, TOL);
    EXPECT_NEAR(params.coefficient(0, 1), 4.0, TOL);
    EXPECT_NEAR(params.coefficient(1, 1), 5.0, TOL);
    EXPECT_NEAR(params.coefficient(2, 1), 6.0, TOL);
}

// ============================================================================
// Test 8: verify_projection_matrix with valid symmetric PSD matrix
// ============================================================================
TEST(ProjectionOperatorTest, VerifyValidSymmetricPSD) {
    // Build a valid projection matrix from known params
    ProjectionOperatorParams params;
    params.n_basis = 3;
    params.n_projectors = 2;
    params.coefficients = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0};  // orthonormal columns
    params.weights = {1.0, 1.0};

    auto P = build_projection_matrix(params);
    EXPECT_TRUE(verify_projection_matrix(P, 3, TOL));
}

// ============================================================================
// Test 9: verify_projection_matrix with non-symmetric matrix
// ============================================================================
TEST(ProjectionOperatorTest, VerifyNonSymmetricMatrix) {
    // 2×2 non-symmetric matrix
    std::vector<Real> P = {1.0, 0.5, 0.0, 1.0};  // P(0,1)=0.5, P(1,0)=0.0
    EXPECT_FALSE(verify_projection_matrix(P, 2, TOL));
}

// ============================================================================
// Test 10: verify_projection_matrix with negative-definite matrix
// ============================================================================
TEST(ProjectionOperatorTest, VerifyNegativeDefiniteMatrix) {
    // 2×2 negative-definite: [[-1, 0], [0, -1]]
    std::vector<Real> P = {-1.0, 0.0, 0.0, -1.0};
    EXPECT_FALSE(verify_projection_matrix(P, 2, TOL));
}

// ============================================================================
// Test 11: verify_projection_matrix with identity matrix
// ============================================================================
TEST(ProjectionOperatorTest, VerifyIdentityMatrix) {
    // 3×3 identity matrix — symmetric, PSD, trace=3
    std::vector<Real> P = {1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0};
    EXPECT_TRUE(verify_projection_matrix(P, 3, TOL));
}
