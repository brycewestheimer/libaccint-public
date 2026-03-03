// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file boys_device.cuh
/// @brief CUDA device-side Boys function evaluation
///
/// Provides __device__ functions for computing the Boys function F_n(T)
/// on the GPU. Uses the same algorithms as the CPU implementation:
/// - Asymptotic expansion for large T (T >= 36)
/// - Chebyshev interpolation for small/medium T (T < 36)
/// - Downward recursion for computing multiple orders efficiently
///
/// Target precision: relative error < 1e-14

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <cuda_runtime.h>
#include <cmath>

namespace libaccint::device::math {

// ============================================================================
// Constants
// ============================================================================

/// @brief Maximum angular momentum supported for Boys function on device
/// Covers up to L_total = 28 (e.g., four i-shells): n_max = 14 + 1 = 15 for Rys
/// Full support up to n = 30 for flexibility
inline constexpr int BOYS_MAX_N_DEVICE = 30;

/// @brief Threshold for switching to asymptotic expansion
inline constexpr double BOYS_ASYMPTOTIC_THRESHOLD_DEVICE = 36.0;

/// @brief Threshold for Taylor expansion (very small T)
inline constexpr double BOYS_TAYLOR_THRESHOLD_DEVICE = 1e-14;

// Chebyshev table constants
inline constexpr int BOYS_CHEB_N_INTERVALS_DEVICE = 36;
inline constexpr int BOYS_CHEB_N_ORDERS_DEVICE = 31;
inline constexpr int BOYS_CHEB_N_TERMS_DEVICE = 12;
inline constexpr double BOYS_CHEB_INTERVAL_WIDTH_DEVICE = 1.0;

// ============================================================================
// Device Double Factorial Helper
// ============================================================================

/// @brief Double factorial (2n-1)!! lookup table (inlined for device use)
///
/// This is a constexpr array that can be used directly in device code.
/// Uses static for internal linkage as required by CUDA without -rdc.
__device__ static constexpr double BOYS_DOUBLE_FACTORIAL[16] = {
    1.0,              // (2×0-1)!! = (-1)!! = 1 (by convention)
    1.0,              // (2×1-1)!! = 1!! = 1
    3.0,              // (2×2-1)!! = 3!! = 3
    15.0,             // (2×3-1)!! = 5!! = 5 × 3 × 1 = 15
    105.0,            // (2×4-1)!! = 7!! = 7 × 5 × 3 × 1 = 105
    945.0,            // (2×5-1)!! = 9!! = 945
    10395.0,          // (2×6-1)!! = 11!! = 10395
    135135.0,         // (2×7-1)!! = 13!! = 135135
    2027025.0,        // (2×8-1)!! = 15!! = 2027025
    34459425.0,       // (2×9-1)!! = 17!! = 34459425
    654729075.0,      // (2×10-1)!! = 19!! = 654729075
    13749310575.0,    // (2×11-1)!! = 21!! = 13749310575
    316234143225.0,   // (2×12-1)!! = 23!! = 316234143225
    7905853580625.0,  // (2×13-1)!! = 25!! = 7905853580625
    213458046676875.0,     // (2×14-1)!! = 27!! = 213458046676875
    6190283353629375.0,    // (2×15-1)!! = 29!! = 6190283353629375
};

// ============================================================================
// Device Functions
// ============================================================================

/**
 * @brief Compute double factorial (2n-1)!! on device
 * @param n Order (0 <= n <= 30)
 * @return Value of (2n-1)!!
 */
__device__ __forceinline__ double boys_double_factorial(int n) {
    if (n < 16) {
        return BOYS_DOUBLE_FACTORIAL[n];
    }
    // For larger n, compute iteratively
    double result = BOYS_DOUBLE_FACTORIAL[15];
    for (int k = 16; k <= n; ++k) {
        result *= (2 * k - 1);
    }
    return result;
}

/**
 * @brief Evaluate F_n(T) using asymptotic expansion for large T
 * @param n Order of the Boys function (0 <= n <= BOYS_MAX_N_DEVICE)
 * @param T Argument value (T > 0, typically T >= 36)
 * @return Value of F_n(T)
 *
 * Uses the leading term: F_n(T) = (2n-1)!! / 2^(n+1) × √π / T^(n+0.5)
 */
__device__ __forceinline__ double boys_asymptotic_device(int n, double T) {
    // sqrt(M_PI) = 1.7724538509055159
    constexpr double SQRT_PI = 1.7724538509055159;

    const double df = boys_double_factorial(n);
    const double power_of_2 = __double2ll_rn(1LL << (n + 1));  // 2^(n+1) exactly for n <= 62
    const double T_power = pow(T, n + 0.5);
    return df / power_of_2 * SQRT_PI / T_power;
}

/**
 * @brief Evaluate F_n(T) using Taylor expansion for very small T
 * @param n Order
 * @param T Argument (T << 1)
 * @return Value of F_n(T)
 */
__device__ __forceinline__ double boys_taylor_device(int n, double T) {
    double result = 1.0 / (2 * n + 1);
    double term = 1.0;

    #pragma unroll 8
    for (int k = 1; k <= 20; ++k) {
        term *= -T / k;
        double contribution = term / (2 * n + 2 * k + 1);
        result += contribution;
        if (fabs(contribution) < 1e-16 * fabs(result)) {
            break;
        }
    }

    return result;
}

/**
 * @brief Evaluate Chebyshev expansion via Clenshaw recurrence
 * @param coeffs Pointer to 12 Chebyshev coefficients
 * @param x Evaluation point in [-1, 1]
 * @return Value of the Chebyshev expansion at x
 */
__device__ __forceinline__ double clenshaw_eval_device(const double* coeffs, double x) {
    double d_next1 = 0.0;  // d_{k+1}
    double d_next2 = 0.0;  // d_{k+2}

    // Unroll loop for better performance
    #pragma unroll
    for (int j = BOYS_CHEB_N_TERMS_DEVICE - 1; j >= 1; --j) {
        double d_k = 2.0 * x * d_next1 - d_next2 + coeffs[j];
        d_next2 = d_next1;
        d_next1 = d_k;
    }

    return coeffs[0] + x * d_next1 - d_next2;
}

/**
 * @brief Evaluate F_n(T) using Chebyshev interpolation
 * @param n Order (0 <= n <= 30)
 * @param T Argument (0 <= T < 36)
 * @param cheb_coeffs Pointer to Chebyshev coefficient table in device memory
 * @return Value of F_n(T)
 */
__device__ __forceinline__ double boys_chebyshev_device(int n, double T, const double* cheb_coeffs) {
    // Determine which interval T falls in
    int interval = static_cast<int>(T);  // T / 1.0
    if (interval >= BOYS_CHEB_N_INTERVALS_DEVICE) {
        interval = BOYS_CHEB_N_INTERVALS_DEVICE - 1;
    }

    // Map T to x in [-1, 1] within the interval
    // For unit-width intervals: x = 2 * (T - interval) - 1 = 2*T - 2*interval - 1
    const double x = 2.0 * T - 2.0 * interval - 1.0;

    // Index into coefficient table: [interval][n][term]
    // Stride: interval * (31 * 12) + n * 12
    const int coeff_offset = interval * (BOYS_CHEB_N_ORDERS_DEVICE * BOYS_CHEB_N_TERMS_DEVICE)
                           + n * BOYS_CHEB_N_TERMS_DEVICE;

    return clenshaw_eval_device(cheb_coeffs + coeff_offset, x);
}

/**
 * @brief Evaluate a single Boys function value F_n(T)
 * @param n Order of the Boys function (0 <= n <= BOYS_MAX_N_DEVICE)
 * @param T Argument value (T >= 0)
 * @param cheb_coeffs Pointer to Chebyshev coefficient table in device memory
 * @return Value of F_n(T)
 *
 * This is the primary single-value interface for device code.
 * Automatically selects the optimal evaluation strategy.
 */
__device__ __forceinline__ double boys_evaluate_device(int n, double T, const double* cheb_coeffs) {
    // T = 0 exactly
    if (T == 0.0) {
        return 1.0 / (2 * n + 1);
    }

    // Very small T - Taylor expansion
    if (T < BOYS_TAYLOR_THRESHOLD_DEVICE) {
        return boys_taylor_device(n, T);
    }

    // Large T - asymptotic expansion
    if (T >= BOYS_ASYMPTOTIC_THRESHOLD_DEVICE) {
        return boys_asymptotic_device(n, T);
    }

    // Small/medium T - Chebyshev interpolation
    return boys_chebyshev_device(n, T, cheb_coeffs);
}

/**
 * @brief Evaluate F_0(T), F_1(T), ..., F_{n_max}(T)
 * @param n_max Maximum order to compute (0 <= n_max <= BOYS_MAX_N_DEVICE)
 * @param T Argument value (T >= 0)
 * @param result Output array of size at least n_max + 1
 * @param cheb_coeffs Pointer to Chebyshev coefficient table in device memory
 *
 * This is the primary batch interface for device code.
 * Uses downward recursion for efficiency and numerical stability.
 */
__device__ __forceinline__ void boys_evaluate_array_device(
    int n_max, double T, double* result, const double* cheb_coeffs) {

    // T = 0 exactly
    if (T == 0.0) {
        for (int n = 0; n <= n_max; ++n) {
            result[n] = 1.0 / (2 * n + 1);
        }
        return;
    }

    // Very small T - Taylor expansion for each
    if (T < BOYS_TAYLOR_THRESHOLD_DEVICE) {
        for (int n = 0; n <= n_max; ++n) {
            result[n] = boys_taylor_device(n, T);
        }
        return;
    }

    // Large T - asymptotic F_0 + upward recursion
    if (T >= BOYS_ASYMPTOTIC_THRESHOLD_DEVICE) {
        result[0] = boys_asymptotic_device(0, T);
        if (n_max == 0) return;

        const double exp_neg_T = exp(-T);
        const double inv_2T = 0.5 / T;

        for (int n = 1; n <= n_max; ++n) {
            result[n] = ((2 * n - 1) * result[n - 1] - exp_neg_T) * inv_2T;
        }
        return;
    }

    // Small/medium T - Chebyshev start + downward recursion
    result[n_max] = boys_chebyshev_device(n_max, T, cheb_coeffs);

    if (n_max == 0) return;

    const double exp_neg_T = exp(-T);
    const double two_T = 2.0 * T;

    for (int n = n_max; n >= 1; --n) {
        result[n - 1] = (two_T * result[n] + exp_neg_T) / (2 * n - 1);
    }
}

// ============================================================================
// Initialization Functions (called from host)
// ============================================================================

/**
 * @brief Initialize device Boys function tables
 *
 * Uploads Chebyshev coefficients to device memory. Must be called before
 * using any device Boys functions.
 *
 * @param stream CUDA stream for async upload (nullptr = default stream)
 * @return Pointer to device Chebyshev coefficient memory
 */
double* boys_device_init(cudaStream_t stream = nullptr);

/**
 * @brief Free device Boys function tables
 *
 * Releases device memory allocated by boys_device_init().
 */
void boys_device_cleanup();

/**
 * @brief Get pointer to device Chebyshev coefficients
 *
 * Returns the pointer that should be passed to device functions.
 */
double* boys_device_get_coeffs();

/**
 * @brief Check if device Boys tables have been initialized
 */
bool boys_device_is_initialized();

}  // namespace libaccint::device::math

#endif  // LIBACCINT_USE_CUDA
