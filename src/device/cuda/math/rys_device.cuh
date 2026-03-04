// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file rys_device.cuh
/// @brief CUDA device-side Rys quadrature implementation
///
/// Provides __device__ functions for computing Rys quadrature roots and weights
/// on the GPU. Uses the modified Chebyshev algorithm to compute three-term
/// recurrence coefficients, then the QL algorithm to find eigenvalues.
///
/// Target precision: relative error < 1e-12 for n_roots <= 5

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include "boys_device.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace libaccint::device::math {

// ============================================================================
// Constants
// ============================================================================

/// Maximum Rys roots supported for ERI integrals
/// For (gg|gg): n_roots = (4+4+4+4)/2 + 1 = 9
inline constexpr int MAX_RYS_ROOTS_DEVICE = 9;

// ============================================================================
// Device Functions
// ============================================================================

/**
 * @brief Modified Chebyshev algorithm to compute three-term recurrence coefficients
 *
 * Given moments μ_k = F_k(T) for k = 0..2n-1, computes the three-term
 * recurrence coefficients {α_k, β_k} for the monic orthogonal polynomials.
 *
 * @param n Number of roots
 * @param moments Boys function moments F_k(T), k = 0..2n-1
 * @param alpha Output: diagonal coefficients [n]
 * @param beta Output: off-diagonal coefficients [n]
 */
__device__ __forceinline__ void rys_chebyshev_impl(int n, const double* moments,
                                                    double* alpha, double* beta) {
    const int n2 = 2 * n;

    // Three rows of the sigma table (circular buffer)
    double sigma[3][2 * MAX_RYS_ROOTS_DEVICE];

    // Initialize
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < n2; ++j)
            sigma[i][j] = 0.0;

    int r_m2 = 0;  // sigma[l-2]
    int r_m1 = 1;  // sigma[l-1]
    int r_cur = 2;

    for (int k = 0; k < n2; ++k) {
        sigma[r_m1][k] = moments[k];
    }

    alpha[0] = moments[1] / moments[0];
    beta[0] = moments[0];

    for (int l = 1; l < n; ++l) {
        for (int k = l; k < n2 - l; ++k) {
            sigma[r_cur][k] = sigma[r_m1][k + 1]
                            - alpha[l - 1] * sigma[r_m1][k]
                            - beta[l - 1] * sigma[r_m2][k];
        }

        alpha[l] = sigma[r_cur][l + 1] / sigma[r_cur][l]
                 - sigma[r_m1][l] / sigma[r_m1][l - 1];
        beta[l] = sigma[r_cur][l] / sigma[r_m1][l - 1];

        int tmp = r_m2;
        r_m2 = r_m1;
        r_m1 = r_cur;
        r_cur = tmp;
    }
}

/**
 * @brief Implicit QL algorithm for symmetric tridiagonal eigenvalue problem
 *
 * Finds eigenvalues and first-row eigenvector components of the Jacobi matrix.
 *
 * @param n Matrix size
 * @param diag Input: diagonal elements, Output: eigenvalues (sorted)
 * @param offdiag Input: off-diagonal elements, Output: destroyed
 * @param z Input: z[0]=1, rest=0, Output: first row of eigenvectors
 */
__device__ __forceinline__ void tridiag_ql_impl(int n, double* diag, double* offdiag, double* z) {
    if (n == 1) return;

    offdiag[n - 1] = 0.0;

    for (int l = 0; l < n; ++l) {
        int iter = 0;

        while (true) {
            // Find smallest m >= l such that offdiag[m] is negligible
            int m = l;
            while (m < n - 1) {
                double tst = fabs(diag[m]) + fabs(diag[m + 1]);
                if (tst + fabs(offdiag[m]) == tst) break;
                ++m;
            }

            if (m == l) break;

            if (++iter > 50) break;  // Convergence protection

            // Compute implicit shift
            double g = (diag[l + 1] - diag[l]) / (2.0 * offdiag[l]);
            double r = sqrt(g * g + 1.0);
            g = diag[m] - diag[l] + offdiag[l] / (g + copysign(r, g));

            double s = 1.0, c = 1.0, p = 0.0;

            // Chase the bulge from bottom to top using conditional Givens
            // rotation (avoids overflow from f*f + g*g for large values)
            for (int i = m - 1; i >= l; --i) {
                double f = s * offdiag[i];
                double b = c * offdiag[i];

                if (fabs(f) >= fabs(g)) {
                    c = g / f;
                    r = sqrt(c * c + 1.0);
                    offdiag[i + 1] = f * r;
                    s = 1.0 / r;
                    c *= s;
                } else {
                    s = f / g;
                    r = sqrt(s * s + 1.0);
                    offdiag[i + 1] = g * r;
                    c = 1.0 / r;
                    s *= c;
                }

                g = diag[i + 1] - p;
                r = (diag[i] - g) * s + 2.0 * c * b;
                p = s * r;
                diag[i + 1] = g + p;
                g = c * r - b;

                // Accumulate first row of eigenvector matrix
                f = z[i + 1];
                z[i + 1] = s * z[i] + c * f;
                z[i] = c * z[i] - s * f;
            }

            diag[l] -= p;
            offdiag[l] = g;
            offdiag[m] = 0.0;
        }
    }

    // Sort eigenvalues (and z) in ascending order
    for (int i = 0; i < n - 1; ++i) {
        int k = i;
        double pk = diag[i];
        for (int j = i + 1; j < n; ++j) {
            if (diag[j] < pk) {
                k = j;
                pk = diag[j];
            }
        }
        if (k != i) {
            diag[k] = diag[i];
            diag[i] = pk;
            double tmp = z[k];
            z[k] = z[i];
            z[i] = tmp;
        }
    }
}

/**
 * @brief Compute Rys quadrature roots and weights
 *
 * @param n_roots Number of Rys roots (1 to MAX_RYS_ROOTS_DEVICE)
 * @param T Boys function argument
 * @param boys_coeffs Device Boys function Chebyshev coefficients
 * @param roots Output: Rys roots [n_roots]
 * @param weights Output: Rys weights [n_roots]
 */
__device__ __forceinline__ void rys_quadrature_impl(int n_roots, double T,
                                                     const double* boys_coeffs,
                                                     double* roots, double* weights) {
    // Compute Boys function moments F_k(T), k = 0..2n-1
    double moments[2 * MAX_RYS_ROOTS_DEVICE];
    boys_evaluate_array_device(2 * n_roots - 1, T, moments, boys_coeffs);

    // Special case: single root
    if (n_roots == 1) {
        roots[0] = moments[1] / moments[0];
        weights[0] = moments[0];
        return;
    }

    // Compute recurrence coefficients via modified Chebyshev
    double alpha[MAX_RYS_ROOTS_DEVICE], beta[MAX_RYS_ROOTS_DEVICE];
    rys_chebyshev_impl(n_roots, moments, alpha, beta);

    // Build tridiagonal matrix and compute eigenvalues
    double diag[MAX_RYS_ROOTS_DEVICE], offdiag[MAX_RYS_ROOTS_DEVICE], z[MAX_RYS_ROOTS_DEVICE];

    for (int i = 0; i < n_roots; ++i) {
        diag[i] = alpha[i];
        z[i] = (i == 0) ? 1.0 : 0.0;
    }
    for (int i = 0; i < n_roots - 1; ++i) {
        offdiag[i] = sqrt(fabs(beta[i + 1]));
    }
    offdiag[n_roots - 1] = 0.0;

    tridiag_ql_impl(n_roots, diag, offdiag, z);

    // Extract roots and weights, clamping to physically valid range.
    // Rys roots are u = t^2 ∈ [0, 1); weights must be non-negative.
    // Without clamping, QL convergence artifacts at higher AM (3+ roots)
    // can produce out-of-range values that corrupt the 2D recursion.
    for (int i = 0; i < n_roots; ++i) {
        roots[i] = fmax(1e-14, fmin(diag[i], 1.0 - 1e-14));
        weights[i] = fmax(0.0, beta[0] * z[i] * z[i]);
    }
}

}  // namespace libaccint::device::math

#endif  // LIBACCINT_USE_CUDA
