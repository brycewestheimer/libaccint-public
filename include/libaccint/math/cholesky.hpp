// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file cholesky.hpp
/// @brief Cholesky decomposition and triangular matrix utilities
///
/// Provides Cholesky decomposition for positive-definite matrices
/// and triangular matrix inversion for density fitting metric handling.

#include <libaccint/core/types.hpp>

#include <cmath>
#include <stdexcept>

namespace libaccint::math {

/// @brief Perform Cholesky decomposition in-place
///
/// Decomposes a symmetric positive-definite matrix A into L * L^T
/// where L is lower triangular. The result overwrites the lower
/// triangle of A.
///
/// @param[in,out] A Matrix to decompose (n x n, row-major)
/// @param n Matrix dimension
/// @throws std::runtime_error if matrix is not positive definite
inline void cholesky_decompose(Real* A, Size n) {
    for (Size i = 0; i < n; ++i) {
        // Diagonal element
        Real sum = A[i * n + i];
        for (Size k = 0; k < i; ++k) {
            sum -= A[i * n + k] * A[i * n + k];
        }

        if (sum <= 0.0) {
            throw std::runtime_error(
                "Cholesky decomposition failed: matrix is not positive definite");
        }

        A[i * n + i] = std::sqrt(sum);

        // Off-diagonal elements
        for (Size j = i + 1; j < n; ++j) {
            sum = A[j * n + i];
            for (Size k = 0; k < i; ++k) {
                sum -= A[j * n + k] * A[i * n + k];
            }
            A[j * n + i] = sum / A[i * n + i];
        }
    }

    // Zero out upper triangle
    for (Size i = 0; i < n; ++i) {
        for (Size j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
}

/// @brief Invert a triangular matrix
///
/// Computes the inverse of a lower or upper triangular matrix.
///
/// @param L Input triangular matrix (n x n, row-major)
/// @param[out] L_inv Output inverse matrix (n x n, row-major)
/// @param n Matrix dimension
/// @param lower True for lower triangular, false for upper
inline void triangular_inverse(const Real* L, Real* L_inv, Size n, bool lower) {
    // Initialize L_inv to identity
    for (Size i = 0; i < n * n; ++i) {
        L_inv[i] = 0.0;
    }
    for (Size i = 0; i < n; ++i) {
        L_inv[i * n + i] = 1.0;
    }

    if (lower) {
        // Forward substitution for lower triangular
        for (Size j = 0; j < n; ++j) {
            for (Size i = j; i < n; ++i) {
                Real sum = (i == j) ? 1.0 : 0.0;
                for (Size k = j; k < i; ++k) {
                    sum -= L[i * n + k] * L_inv[k * n + j];
                }
                L_inv[i * n + j] = sum / L[i * n + i];
            }
        }
    } else {
        // Backward substitution for upper triangular
        for (Size j = n; j > 0; --j) {
            const Size jj = j - 1;
            for (Size ii = jj + 1; ii > 0; --ii) {
                const Size i = ii - 1;
                Real sum = (i == jj) ? 1.0 : 0.0;
                for (Size k = i + 1; k <= jj; ++k) {
                    sum -= L[i * n + k] * L_inv[k * n + jj];
                }
                L_inv[i * n + jj] = sum / L[i * n + i];
            }
        }
    }
}

/// @brief Compute L^{-T} * L^{-1} = (L * L^T)^{-1}
///
/// Given the Cholesky factor L where A = L * L^T, computes A^{-1}.
///
/// @param L Lower triangular Cholesky factor (n x n, row-major)
/// @param[out] A_inv Output inverse matrix (n x n, row-major)
/// @param n Matrix dimension
inline void cholesky_inverse(const Real* L, Real* A_inv, Size n) {
    // Compute L^{-1}
    std::vector<Real> L_inv(n * n);
    triangular_inverse(L, L_inv.data(), n, true);

    // Compute A^{-1} = L^{-T} * L^{-1}
    for (Size i = 0; i < n; ++i) {
        for (Size j = 0; j < n; ++j) {
            Real sum = 0.0;
            for (Size k = 0; k < n; ++k) {
                sum += L_inv[k * n + i] * L_inv[k * n + j];
            }
            A_inv[i * n + j] = sum;
        }
    }
}

/// @brief Solve L * x = b for x (forward substitution)
///
/// @param L Lower triangular matrix (n x n, row-major)
/// @param b Right-hand side vector (length n)
/// @param[out] x Solution vector (length n)
/// @param n System dimension
inline void forward_substitution(const Real* L, const Real* b, Real* x, Size n) {
    for (Size i = 0; i < n; ++i) {
        Real sum = b[i];
        for (Size j = 0; j < i; ++j) {
            sum -= L[i * n + j] * x[j];
        }
        x[i] = sum / L[i * n + i];
    }
}

/// @brief Solve L^T * x = b for x (backward substitution)
///
/// @param L Lower triangular matrix (its transpose is upper)
/// @param b Right-hand side vector (length n)
/// @param[out] x Solution vector (length n)
/// @param n System dimension
inline void backward_substitution_transpose(const Real* L, const Real* b, Real* x, Size n) {
    for (Size ii = n; ii > 0; --ii) {
        const Size i = ii - 1;
        Real sum = b[i];
        for (Size j = i + 1; j < n; ++j) {
            sum -= L[j * n + i] * x[j];  // L^T[i,j] = L[j,i]
        }
        x[i] = sum / L[i * n + i];
    }
}

}  // namespace libaccint::math
