// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file blas_wrappers.hpp
/// @brief BLAS-like operation wrappers with raw-loop fallback
///
/// Provides GEMM and GEMV operations that use BLAS when available
/// (LIBACCINT_HAS_BLAS is defined), otherwise fall back to equivalent
/// raw-loop implementations.
///
/// These are designed for the DF module's J/K contractions where matrix
/// sizes are O(N_orb x N_orb) and O(N_aux), which benefit from BLAS-3
/// acceleration for larger bases.

#include <libaccint/core/types.hpp>

#include <cstring>

#if LIBACCINT_HAS_BLAS
extern "C" {
    void dgemm_(const char* transa, const char* transb,
                const int* m, const int* n, const int* k,
                const double* alpha, const double* a, const int* lda,
                const double* b, const int* ldb,
                const double* beta, double* c, const int* ldc);

    void dgemv_(const char* trans, const int* m, const int* n,
                const double* alpha, const double* a, const int* lda,
                const double* x, const int* incx,
                const double* beta, double* y, const int* incy);
}
#endif

namespace libaccint::math {

/// @brief Matrix-vector multiply: y = alpha * A * x + beta * y
///
/// A is (m x n), x is (n x 1), y is (m x 1).
/// If transpose=true: y = alpha * A^T * x + beta * y (A^T is n x m, x is m, y is n)
///
/// @param transpose If true, use A^T instead of A
/// @param m Number of rows of A
/// @param n Number of columns of A
/// @param alpha Scalar multiplier
/// @param A Matrix (row-major, m x n)
/// @param x Input vector
/// @param beta Scalar for y
/// @param y Output vector (modified in place)
inline void gemv(bool transpose, Size m, Size n,
                 Real alpha, const Real* A,
                 const Real* x, Real beta, Real* y) {
#if LIBACCINT_HAS_BLAS
    // BLAS uses column-major, so row-major A with 'N' becomes column-major A^T
    // For row-major: gemv('T', n, m, ...) for no-transpose, gemv('N', n, m, ...) for transpose
    char trans = transpose ? 'N' : 'T';
    int bm = static_cast<int>(n);  // leading dimension of column-major = n_cols
    int bn = static_cast<int>(m);  // other dimension
    int lda = static_cast<int>(n);
    int inc = 1;
    dgemv_(&trans, &bm, &bn, &alpha, A, &lda, x, &inc, &beta, y, &inc);
#else
    if (!transpose) {
        // y = alpha * A * x + beta * y
        for (Size i = 0; i < m; ++i) {
            Real sum = 0.0;
            for (Size j = 0; j < n; ++j) {
                sum += A[i * n + j] * x[j];
            }
            y[i] = alpha * sum + beta * y[i];
        }
    } else {
        // y = alpha * A^T * x + beta * y
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
#endif
}

/// @brief Matrix-matrix multiply: C = alpha * A * B + beta * C
///
/// A is (m x k), B is (k x n), C is (m x n). All row-major.
///
/// @param m Rows of A and C
/// @param n Columns of B and C
/// @param k Columns of A / rows of B
/// @param alpha Scalar multiplier
/// @param A Left matrix (m x k, row-major)
/// @param B Right matrix (k x n, row-major)
/// @param beta Scalar for C
/// @param C Output matrix (m x n, row-major, modified in place)
inline void gemm(Size m, Size n, Size k,
                 Real alpha, const Real* A, const Real* B,
                 Real beta, Real* C) {
#if LIBACCINT_HAS_BLAS
    // Row-major C = A * B → Column-major C^T = B^T * A^T
    char transa = 'N';
    char transb = 'N';
    int bm = static_cast<int>(n);
    int bn = static_cast<int>(m);
    int bk = static_cast<int>(k);
    int lda = static_cast<int>(n);   // ld of B^T in col-major = n
    int ldb = static_cast<int>(k);   // ld of A^T in col-major = k
    int ldc = static_cast<int>(n);
    dgemm_(&transa, &transb, &bm, &bn, &bk, &alpha, B, &lda, A, &ldb, &beta, C, &ldc);
#else
    for (Size i = 0; i < m; ++i) {
        for (Size j = 0; j < n; ++j) {
            Real sum = 0.0;
            for (Size p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
#endif
}

/// @brief Matrix-matrix multiply with A transposed: C = alpha * A^T * B + beta * C
///
/// A is (k x m), A^T is (m x k), B is (k x n), C is (m x n). All row-major.
inline void gemm_at(Size m, Size n, Size k,
                    Real alpha, const Real* A, const Real* B,
                    Real beta, Real* C) {
#if LIBACCINT_HAS_BLAS
    // Row-major: C = A^T * B → Col-major: C^T = B^T * A
    char transa = 'N';  // B^T
    char transb = 'T';  // A (transposed back = plain A in col-major)
    int bm = static_cast<int>(n);
    int bn = static_cast<int>(m);
    int bk = static_cast<int>(k);
    int lda = static_cast<int>(n);   // B col-major leading dim
    int ldb = static_cast<int>(m);   // A col-major leading dim
    int ldc = static_cast<int>(n);
    dgemm_(&transa, &transb, &bm, &bn, &bk, &alpha, B, &lda, A, &ldb, &beta, C, &ldc);
#else
    for (Size i = 0; i < m; ++i) {
        for (Size j = 0; j < n; ++j) {
            Real sum = 0.0;
            for (Size p = 0; p < k; ++p) {
                sum += A[p * m + i] * B[p * n + j];
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
#endif
}

/// @brief Matrix-matrix multiply with B transposed: C = alpha * A * B^T + beta * C
///
/// A is (m x k), B is (n x k), B^T is (k x n), C is (m x n). All row-major.
inline void gemm_bt(Size m, Size n, Size k,
                    Real alpha, const Real* A, const Real* B,
                    Real beta, Real* C) {
#if LIBACCINT_HAS_BLAS
    // Row-major: C = A * B^T → Col-major: C^T = B * A^T
    char transa = 'N';  // B (col-major = B^T in row-major, but we want B)
    char transb = 'T';  // A^T
    int bm = static_cast<int>(n);
    int bn = static_cast<int>(m);
    int bk = static_cast<int>(k);
    int lda = static_cast<int>(k);   // B row-major leading dim
    int ldb = static_cast<int>(k);   // A row-major leading dim
    int ldc = static_cast<int>(n);
    dgemm_(&transa, &transb, &bm, &bn, &bk, &alpha, B, &lda, A, &ldb, &beta, C, &ldc);
#else
    for (Size i = 0; i < m; ++i) {
        for (Size j = 0; j < n; ++j) {
            Real sum = 0.0;
            for (Size p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[j * k + p];
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
#endif
}

}  // namespace libaccint::math
