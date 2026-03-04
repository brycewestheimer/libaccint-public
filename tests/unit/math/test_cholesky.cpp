// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_cholesky.cpp
/// @brief Unit tests for Cholesky decomposition and triangular matrix utilities

#include <libaccint/math/cholesky.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>

using namespace libaccint;
using namespace libaccint::math;

namespace {
constexpr double TOL = 1e-12;
constexpr double LOOSE_TOL = 1e-10;

/// @brief Helper: multiply L * L^T and store in result (n x n, row-major)
void multiply_LLT(const Real* L, Real* result, Size n) {
    for (Size i = 0; i < n; ++i) {
        for (Size j = 0; j < n; ++j) {
            Real sum = 0.0;
            for (Size k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];  // L[i,k] * L^T[k,j] = L[i,k] * L[j,k]
            }
            result[i * n + j] = sum;
        }
    }
}

/// @brief Helper: multiply A * B and store in result (n x n, row-major)
void mat_mul(const Real* A, const Real* B, Real* result, Size n) {
    for (Size i = 0; i < n; ++i) {
        for (Size j = 0; j < n; ++j) {
            Real sum = 0.0;
            for (Size k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            result[i * n + j] = sum;
        }
    }
}

}  // anonymous namespace

// ============================================================================
// Cholesky Decompose Tests
// ============================================================================

TEST(CholeskyTest, Decompose1x1) {
    Real A[] = {4.0};
    cholesky_decompose(A, 1);
    EXPECT_NEAR(A[0], 2.0, TOL);
}

TEST(CholeskyTest, Decompose2x2) {
    // A = [[4, 2], [2, 10]]
    // L = [[2, 0], [1, 3]]
    Real A[] = {4.0, 2.0,
                2.0, 10.0};
    Real A_orig[] = {4.0, 2.0,
                     2.0, 10.0};

    cholesky_decompose(A, 2);

    EXPECT_NEAR(A[0], 2.0, TOL);   // L[0,0]
    EXPECT_NEAR(A[1], 0.0, TOL);   // L[0,1] (upper zeroed)
    EXPECT_NEAR(A[2], 1.0, TOL);   // L[1,0]
    EXPECT_NEAR(A[3], 3.0, TOL);   // L[1,1]

    // Verify L * L^T = A_orig
    Real LLT[4];
    multiply_LLT(A, LLT, 2);
    for (Size i = 0; i < 4; ++i) {
        EXPECT_NEAR(LLT[i], A_orig[i], TOL) << "Mismatch at index " << i;
    }
}

TEST(CholeskyTest, Decompose3x3) {
    // A = [[25, 15, -5],
    //      [15, 18,  0],
    //      [-5,  0, 11]]
    Real A[] = {25.0, 15.0, -5.0,
                15.0, 18.0,  0.0,
                -5.0,  0.0, 11.0};
    Real A_orig[9];
    std::copy(A, A + 9, A_orig);

    cholesky_decompose(A, 3);

    // Verify L is lower triangular
    EXPECT_NEAR(A[1], 0.0, TOL);  // L[0,1]
    EXPECT_NEAR(A[2], 0.0, TOL);  // L[0,2]
    EXPECT_NEAR(A[5], 0.0, TOL);  // L[1,2]

    // Verify L * L^T = A_orig
    Real LLT[9];
    multiply_LLT(A, LLT, 3);
    for (Size i = 0; i < 9; ++i) {
        EXPECT_NEAR(LLT[i], A_orig[i], TOL) << "Mismatch at index " << i;
    }
}

TEST(CholeskyTest, Decompose4x4) {
    // 4x4 positive definite matrix: A = B^T * B where B is random
    // A = [[4, 2, 1, 0],
    //      [2, 5, 3, 1],
    //      [1, 3, 6, 2],
    //      [0, 1, 2, 4]]
    Real A[] = {4.0, 2.0, 1.0, 0.0,
                2.0, 5.0, 3.0, 1.0,
                1.0, 3.0, 6.0, 2.0,
                0.0, 1.0, 2.0, 4.0};
    Real A_orig[16];
    std::copy(A, A + 16, A_orig);

    cholesky_decompose(A, 4);

    // Verify lower triangular
    for (Size i = 0; i < 4; ++i) {
        for (Size j = i + 1; j < 4; ++j) {
            EXPECT_NEAR(A[i * 4 + j], 0.0, TOL) << "i=" << i << " j=" << j;
        }
    }

    // Verify L * L^T = A_orig
    Real LLT[16];
    multiply_LLT(A, LLT, 4);
    for (Size i = 0; i < 16; ++i) {
        EXPECT_NEAR(LLT[i], A_orig[i], TOL) << "Mismatch at index " << i;
    }
}

TEST(CholeskyTest, DecomposeNotPositiveDefinite) {
    // Non-positive-definite matrix should throw
    Real A[] = {1.0, 2.0,
                2.0, 1.0};  // Eigenvalues: 3 and -1
    EXPECT_THROW(cholesky_decompose(A, 2), std::runtime_error);
}

TEST(CholeskyTest, DecomposeNearSingular) {
    // Near-singular but still positive definite
    Real A[] = {1.0, 0.9999,
                0.9999, 1.0};
    Real A_orig[4];
    std::copy(A, A + 4, A_orig);

    EXPECT_NO_THROW(cholesky_decompose(A, 2));

    // Verify reconstruction
    Real LLT[4];
    multiply_LLT(A, LLT, 2);
    for (Size i = 0; i < 4; ++i) {
        EXPECT_NEAR(LLT[i], A_orig[i], 1e-10) << "Mismatch at index " << i;
    }
}

TEST(CholeskyTest, DecomposeIdentity) {
    // Identity matrix: L = I
    Real A[] = {1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0};

    cholesky_decompose(A, 3);

    for (Size i = 0; i < 3; ++i) {
        for (Size j = 0; j < 3; ++j) {
            Real expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(A[i * 3 + j], expected, TOL);
        }
    }
}

// ============================================================================
// Triangular Inverse Tests
// ============================================================================

TEST(TriangularInverseTest, LowerTriangular2x2) {
    Real L[] = {2.0, 0.0,
                1.0, 3.0};
    Real L_inv[4];

    triangular_inverse(L, L_inv, 2, true);

    // L * L_inv should be identity
    Real product[4];
    mat_mul(L, L_inv, product, 2);
    for (Size i = 0; i < 2; ++i) {
        for (Size j = 0; j < 2; ++j) {
            Real expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(product[i * 2 + j], expected, TOL);
        }
    }
}

TEST(TriangularInverseTest, LowerTriangular3x3) {
    Real L[] = {2.0, 0.0, 0.0,
                1.0, 3.0, 0.0,
                0.5, 2.0, 4.0};
    Real L_inv[9];

    triangular_inverse(L, L_inv, 3, true);

    Real product[9];
    mat_mul(L, L_inv, product, 3);
    for (Size i = 0; i < 3; ++i) {
        for (Size j = 0; j < 3; ++j) {
            Real expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(product[i * 3 + j], expected, TOL)
                << "i=" << i << " j=" << j;
        }
    }
}

TEST(TriangularInverseTest, UpperTriangular2x2) {
    Real U[] = {2.0, 3.0,
                0.0, 4.0};
    Real U_inv[4];

    triangular_inverse(U, U_inv, 2, false);

    Real product[4];
    mat_mul(U, U_inv, product, 2);
    for (Size i = 0; i < 2; ++i) {
        for (Size j = 0; j < 2; ++j) {
            Real expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(product[i * 2 + j], expected, TOL);
        }
    }
}

TEST(TriangularInverseTest, Identity) {
    Real I[] = {1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0};
    Real I_inv[9];

    triangular_inverse(I, I_inv, 3, true);

    for (Size i = 0; i < 9; ++i) {
        EXPECT_NEAR(I_inv[i], I[i], TOL);
    }
}

// ============================================================================
// Cholesky Inverse Tests
// ============================================================================

TEST(CholeskyInverseTest, Inverse2x2) {
    Real A[] = {4.0, 2.0,
                2.0, 10.0};
    Real L[4];
    std::copy(A, A + 4, L);
    cholesky_decompose(L, 2);

    Real A_inv[4];
    cholesky_inverse(L, A_inv, 2);

    // A * A_inv should be identity
    Real product[4];
    mat_mul(A, A_inv, product, 2);
    for (Size i = 0; i < 2; ++i) {
        for (Size j = 0; j < 2; ++j) {
            Real expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(product[i * 2 + j], expected, TOL);
        }
    }
}

TEST(CholeskyInverseTest, Inverse3x3) {
    Real A[] = {25.0, 15.0, -5.0,
                15.0, 18.0,  0.0,
                -5.0,  0.0, 11.0};
    Real L[9];
    std::copy(A, A + 9, L);
    cholesky_decompose(L, 3);

    Real A_inv[9];
    cholesky_inverse(L, A_inv, 3);

    Real product[9];
    mat_mul(A, A_inv, product, 3);
    for (Size i = 0; i < 3; ++i) {
        for (Size j = 0; j < 3; ++j) {
            Real expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(product[i * 3 + j], expected, LOOSE_TOL)
                << "i=" << i << " j=" << j;
        }
    }
}

TEST(CholeskyInverseTest, InverseSymmetry) {
    // A^{-1} should be symmetric
    Real A[] = {4.0, 2.0, 1.0,
                2.0, 5.0, 3.0,
                1.0, 3.0, 6.0};
    Real L[9];
    std::copy(A, A + 9, L);
    cholesky_decompose(L, 3);

    Real A_inv[9];
    cholesky_inverse(L, A_inv, 3);

    for (Size i = 0; i < 3; ++i) {
        for (Size j = i + 1; j < 3; ++j) {
            EXPECT_NEAR(A_inv[i * 3 + j], A_inv[j * 3 + i], TOL)
                << "A_inv not symmetric at (" << i << "," << j << ")";
        }
    }
}

// ============================================================================
// Forward Substitution Tests
// ============================================================================

TEST(ForwardSubstitutionTest, Simple2x2) {
    // L = [[2, 0], [1, 3]], b = [4, 7]
    // Lx = b => x = [2, 5/3]
    Real L[] = {2.0, 0.0,
                1.0, 3.0};
    Real b[] = {4.0, 7.0};
    Real x[2];

    forward_substitution(L, b, x, 2);

    EXPECT_NEAR(x[0], 2.0, TOL);
    EXPECT_NEAR(x[1], 5.0 / 3.0, TOL);

    // Verify L * x = b
    for (Size i = 0; i < 2; ++i) {
        Real sum = 0.0;
        for (Size j = 0; j < 2; ++j) {
            sum += L[i * 2 + j] * x[j];
        }
        EXPECT_NEAR(sum, b[i], TOL) << "i=" << i;
    }
}

TEST(ForwardSubstitutionTest, Simple3x3) {
    Real L[] = {1.0, 0.0, 0.0,
                2.0, 1.0, 0.0,
                3.0, 4.0, 1.0};
    Real b[] = {1.0, 4.0, 15.0};
    Real x[3];

    forward_substitution(L, b, x, 3);

    // Verify L * x = b
    for (Size i = 0; i < 3; ++i) {
        Real sum = 0.0;
        for (Size j = 0; j < 3; ++j) {
            sum += L[i * 3 + j] * x[j];
        }
        EXPECT_NEAR(sum, b[i], TOL) << "i=" << i;
    }
}

// ============================================================================
// Backward Substitution Transpose Tests
// ============================================================================

TEST(BackwardSubstitutionTransposeTest, Simple2x2) {
    // L^T * x = b, where L = [[2, 0], [1, 3]]
    // L^T = [[2, 1], [0, 3]]
    Real L[] = {2.0, 0.0,
                1.0, 3.0};
    Real b[] = {5.0, 9.0};
    Real x[2];

    backward_substitution_transpose(L, b, x, 2);

    // Verify L^T * x = b
    for (Size i = 0; i < 2; ++i) {
        Real sum = 0.0;
        for (Size j = 0; j < 2; ++j) {
            sum += L[j * 2 + i] * x[j];  // L^T[i,j] = L[j,i]
        }
        EXPECT_NEAR(sum, b[i], TOL) << "i=" << i;
    }
}

TEST(BackwardSubstitutionTransposeTest, Simple3x3) {
    Real L[] = {3.0, 0.0, 0.0,
                1.0, 2.0, 0.0,
                0.0, 1.0, 4.0};
    Real b[] = {7.0, 8.0, 12.0};
    Real x[3];

    backward_substitution_transpose(L, b, x, 3);

    // Verify L^T * x = b
    for (Size i = 0; i < 3; ++i) {
        Real sum = 0.0;
        for (Size j = 0; j < 3; ++j) {
            sum += L[j * 3 + i] * x[j];
        }
        EXPECT_NEAR(sum, b[i], TOL) << "i=" << i;
    }
}

// ============================================================================
// Large Condition Number Test
// ============================================================================

TEST(CholeskyTest, LargeConditionNumber) {
    // Matrix with large condition number
    Real A[] = {1.0,    0.999,
                0.999,  1.0};
    Real A_orig[4];
    std::copy(A, A + 4, A_orig);

    Real L[4];
    std::copy(A, A + 4, L);
    cholesky_decompose(L, 2);

    Real A_inv[4];
    cholesky_inverse(L, A_inv, 2);

    Real product[4];
    mat_mul(A_orig, A_inv, product, 2);

    // May have larger error due to conditioning, but should still be close
    for (Size i = 0; i < 2; ++i) {
        for (Size j = 0; j < 2; ++j) {
            Real expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(product[i * 2 + j], expected, 1e-8)
                << "i=" << i << " j=" << j;
        }
    }
}

// ============================================================================
// Round-Trip Test: Decompose -> Solve -> Verify
// ============================================================================

TEST(CholeskyTest, SolveLinearSystem) {
    // Solve A * x = b using Cholesky: L * L^T * x = b
    // 1. Forward solve: L * y = b
    // 2. Backward solve: L^T * x = y
    Real A[] = {4.0, 2.0,
                2.0, 10.0};
    Real b[] = {14.0, 32.0};

    Real L[4];
    std::copy(A, A + 4, L);
    cholesky_decompose(L, 2);

    Real y[2], x[2];
    forward_substitution(L, b, y, 2);
    backward_substitution_transpose(L, y, x, 2);

    // Verify A * x = b
    for (Size i = 0; i < 2; ++i) {
        Real sum = 0.0;
        for (Size j = 0; j < 2; ++j) {
            sum += A[i * 2 + j] * x[j];
        }
        EXPECT_NEAR(sum, b[i], TOL) << "i=" << i;
    }
}
