// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_df_blas_contractions.cpp
/// @brief Tests for BLAS-accelerated J/K contractions in DFFockBuilder
///
/// Validates that the BLAS wrapper-based J and K contraction implementations
/// produce correct results for various matrix sizes and density types.

#include <libaccint/math/blas_wrappers.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

using namespace libaccint;

namespace {

/// Reference raw-loop GEMV: y = alpha * A * x + beta * y  (no transpose)
void ref_gemv(Size m, Size n, Real alpha, const Real* A,
              const Real* x, Real beta, Real* y) {
    for (Size i = 0; i < m; ++i) {
        Real sum = 0.0;
        for (Size j = 0; j < n; ++j) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}

/// Reference raw-loop GEMV transposed: y = alpha * A^T * x + beta * y
void ref_gemv_t(Size m, Size n, Real alpha, const Real* A,
                const Real* x, Real beta, Real* y) {
    for (Size j = 0; j < n; ++j) {
        y[j] *= beta;
    }
    for (Size i = 0; i < m; ++i) {
        const Real ax = alpha * x[i];
        for (Size j = 0; j < n; ++j) {
            y[j] += ax * A[i * n + j];
        }
    }
}

/// Reference raw-loop GEMM: C = alpha * A * B + beta * C
void ref_gemm(Size m, Size n, Size k, Real alpha, const Real* A,
              const Real* B, Real beta, Real* C) {
    for (Size i = 0; i < m; ++i) {
        for (Size j = 0; j < n; ++j) {
            Real sum = 0.0;
            for (Size p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
}

/// Reference raw-loop GEMM_BT: C = alpha * A * B^T + beta * C
void ref_gemm_bt(Size m, Size n, Size k, Real alpha, const Real* A,
                 const Real* B, Real beta, Real* C) {
    for (Size i = 0; i < m; ++i) {
        for (Size j = 0; j < n; ++j) {
            Real sum = 0.0;
            for (Size p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[j * k + p];
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
}

/// Fill vector with random values
void fill_random(std::vector<Real>& v, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);
    for (auto& x : v) x = dist(rng);
}

constexpr Real TOL = 1e-12;

}  // namespace

// =============================================================================
// GEMV Tests
// =============================================================================

TEST(BLASContractions, GemvNoTransposeTiny) {
    // 2x3 matrix
    Size m = 2, n = 3;
    std::vector<Real> A = {1.0, 2.0, 3.0,
                           4.0, 5.0, 6.0};
    std::vector<Real> x = {1.0, 0.5, 0.25};
    std::vector<Real> y_blas(m, 0.0), y_ref(m, 0.0);

    math::gemv(false, m, n, 1.0, A.data(), x.data(), 0.0, y_blas.data());
    ref_gemv(m, n, 1.0, A.data(), x.data(), 0.0, y_ref.data());

    for (Size i = 0; i < m; ++i) {
        EXPECT_NEAR(y_blas[i], y_ref[i], TOL) << "Mismatch at i=" << i;
    }
}

TEST(BLASContractions, GemvTransposeTiny) {
    Size m = 2, n = 3;
    std::vector<Real> A = {1.0, 2.0, 3.0,
                           4.0, 5.0, 6.0};
    std::vector<Real> x = {2.0, 0.5};
    std::vector<Real> y_blas(n, 0.0), y_ref(n, 0.0);

    math::gemv(true, m, n, 1.0, A.data(), x.data(), 0.0, y_blas.data());
    ref_gemv_t(m, n, 1.0, A.data(), x.data(), 0.0, y_ref.data());

    for (Size i = 0; i < n; ++i) {
        EXPECT_NEAR(y_blas[i], y_ref[i], TOL) << "Mismatch at i=" << i;
    }
}

TEST(BLASContractions, GemvSmallRandom) {
    Size m = 10, n = 30;
    std::vector<Real> A(m * n), x(n), y_blas(m, 0.0), y_ref(m, 0.0);
    fill_random(A, 1); fill_random(x, 2);

    math::gemv(false, m, n, 1.0, A.data(), x.data(), 0.0, y_blas.data());
    ref_gemv(m, n, 1.0, A.data(), x.data(), 0.0, y_ref.data());

    for (Size i = 0; i < m; ++i) {
        EXPECT_NEAR(y_blas[i], y_ref[i], TOL) << "i=" << i;
    }
}

TEST(BLASContractions, GemvTransposeMedium) {
    Size m = 30, n = 100;
    std::vector<Real> A(m * n), x(m), y_blas(n, 0.0), y_ref(n, 0.0);
    fill_random(A, 3); fill_random(x, 4);

    math::gemv(true, m, n, 1.0, A.data(), x.data(), 0.0, y_blas.data());
    ref_gemv_t(m, n, 1.0, A.data(), x.data(), 0.0, y_ref.data());

    for (Size i = 0; i < n; ++i) {
        EXPECT_NEAR(y_blas[i], y_ref[i], TOL) << "i=" << i;
    }
}

// =============================================================================
// GEMM Tests
// =============================================================================

TEST(BLASContractions, GemmTiny) {
    Size m = 2, n = 2, k = 3;
    std::vector<Real> A = {1.0, 2.0, 3.0,
                           4.0, 5.0, 6.0};
    std::vector<Real> B = {7.0, 8.0,
                           9.0, 10.0,
                           11.0, 12.0};
    std::vector<Real> C_blas(m * n, 0.0), C_ref(m * n, 0.0);

    math::gemm(m, n, k, 1.0, A.data(), B.data(), 0.0, C_blas.data());
    ref_gemm(m, n, k, 1.0, A.data(), B.data(), 0.0, C_ref.data());

    for (Size i = 0; i < m * n; ++i) {
        EXPECT_NEAR(C_blas[i], C_ref[i], TOL) << "i=" << i;
    }
}

TEST(BLASContractions, GemmSmallRandom) {
    Size m = 10, n = 10, k = 30;
    std::vector<Real> A(m * k), B(k * n), C_blas(m * n, 0.0), C_ref(m * n, 0.0);
    fill_random(A, 5); fill_random(B, 6);

    math::gemm(m, n, k, 1.0, A.data(), B.data(), 0.0, C_blas.data());
    ref_gemm(m, n, k, 1.0, A.data(), B.data(), 0.0, C_ref.data());

    for (Size i = 0; i < m * n; ++i) {
        EXPECT_NEAR(C_blas[i], C_ref[i], TOL) << "i=" << i;
    }
}

TEST(BLASContractions, GemmBTSmallRandom) {
    Size m = 10, n = 10, k = 15;
    std::vector<Real> A(m * k), B(n * k), C_blas(m * n, 0.0), C_ref(m * n, 0.0);
    fill_random(A, 7); fill_random(B, 8);

    math::gemm_bt(m, n, k, 1.0, A.data(), B.data(), 0.0, C_blas.data());
    ref_gemm_bt(m, n, k, 1.0, A.data(), B.data(), 0.0, C_ref.data());

    for (Size i = 0; i < m * n; ++i) {
        EXPECT_NEAR(C_blas[i], C_ref[i], TOL) << "i=" << i;
    }
}

TEST(BLASContractions, GemmWithBetaAccumulation) {
    // C = 2.0 * A * B + 0.5 * C  (verify beta != 0 path)
    Size m = 5, n = 5, k = 5;
    std::vector<Real> A(m * k), B(k * n), C_blas(m * n), C_ref(m * n);
    fill_random(A, 10); fill_random(B, 11);

    // Initialize C with some values
    fill_random(C_blas, 12);
    C_ref = C_blas;

    math::gemm(m, n, k, 2.0, A.data(), B.data(), 0.5, C_blas.data());
    ref_gemm(m, n, k, 2.0, A.data(), B.data(), 0.5, C_ref.data());

    for (Size i = 0; i < m * n; ++i) {
        EXPECT_NEAR(C_blas[i], C_ref[i], TOL) << "i=" << i;
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(BLASContractions, SingleElement) {
    // 1x1 * 1x1 = 1x1
    std::vector<Real> A = {3.0}, B = {4.0}, C(1, 0.0);
    math::gemm(1, 1, 1, 1.0, A.data(), B.data(), 0.0, C.data());
    EXPECT_NEAR(C[0], 12.0, TOL);
}

TEST(BLASContractions, GemvSingleElement) {
    std::vector<Real> A = {5.0}, x = {2.0}, y(1, 0.0);
    math::gemv(false, 1, 1, 1.0, A.data(), x.data(), 0.0, y.data());
    EXPECT_NEAR(y[0], 10.0, TOL);
}

TEST(BLASContractions, GemmATSmallRandom) {
    // C = alpha * A^T * B + beta * C
    Size m = 8, n = 6, k = 10;
    std::vector<Real> A(k * m), B(k * n), C_blas(m * n, 0.0), C_ref(m * n, 0.0);
    fill_random(A, 20); fill_random(B, 21);

    math::gemm_at(m, n, k, 1.0, A.data(), B.data(), 0.0, C_blas.data());

    // Reference: A^T[i,p] = A[p*m+i]
    for (Size i = 0; i < m; ++i) {
        for (Size j = 0; j < n; ++j) {
            Real sum = 0.0;
            for (Size p = 0; p < k; ++p) {
                sum += A[p * m + i] * B[p * n + j];
            }
            C_ref[i * n + j] = sum;
        }
    }

    for (Size i = 0; i < m * n; ++i) {
        EXPECT_NEAR(C_blas[i], C_ref[i], TOL) << "i=" << i;
    }
}
