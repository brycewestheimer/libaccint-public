// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file boys_function_float.hpp
/// @brief Float32 (single-precision) Boys function implementation
///
/// This file provides single-precision versions of the Boys function
/// optimized for GPU performance. The float32 implementation uses:
/// - Smaller interpolation tables (reduced memory bandwidth)
/// - Float32-specific polynomial coefficients
/// - Adjusted asymptotic thresholds
/// - Target accuracy: relative error < 1e-6

#include <libaccint/core/precision.hpp>
#include <libaccint/math/boys_function.hpp>

#include <cmath>
#include <array>

namespace libaccint::math {

// ============================================================================
// Float32 Boys Function Constants
// ============================================================================

/// @brief Threshold for switching to asymptotic expansion (float32)
///
/// Lower than double precision due to reduced accuracy requirements.
inline constexpr float BOYS_ASYMPTOTIC_THRESHOLD_FLOAT = 25.0f;

/// @brief Maximum T value for Chebyshev interpolation (float32)
inline constexpr float BOYS_CHEBYSHEV_T_MAX_FLOAT = 30.0f;

/// @brief Maximum angular momentum supported for Boys function (float32)
inline constexpr int BOYS_MAX_N_FLOAT = 30;

/// @brief Target relative accuracy for float32 Boys function
inline constexpr float BOYS_TARGET_ACCURACY_FLOAT = 1.0e-6f;

// ============================================================================
// Float32 Boys Function - Asymptotic Regime
// ============================================================================

/// @brief Evaluate F_n(T) using asymptotic expansion for large T (float32)
/// @param n Order of the Boys function (0 ≤ n ≤ BOYS_MAX_N_FLOAT)
/// @param T Argument value (T > BOYS_ASYMPTOTIC_THRESHOLD_FLOAT)
/// @return Value of F_n(T) in single precision
///
/// Uses the asymptotic formula:
/// F_n(T) = (2n-1)!! / (2^(n+1)) × √(π) / T^(n+0.5)
///
/// The omitted correction is O(erfc(√T)), which for T ≥ 25 is < 1e-6,
/// giving accuracy appropriate for float32.
inline float boys_asymptotic_float(int n, float T) {
    // Precomputed double factorials (2n-1)!! for n = 0..20
    static constexpr float DOUBLE_FACTORIAL[21] = {
        1.0f, 1.0f, 3.0f, 15.0f, 105.0f, 945.0f, 10395.0f, 135135.0f,
        2027025.0f, 34459425.0f, 654729075.0f, 1.3749310575e10f,
        3.1623414323e11f, 7.9058535806e12f, 2.1345804668e14f,
        6.1902833537e15f, 1.9189878397e17f, 6.3326598707e18f,
        2.2164309548e20f, 8.2007945069e21f, 3.1983098677e23f
    };

    // sqrt(pi) in float
    constexpr float SQRT_PI_F = 1.7724538509f;

    float inv_T = 1.0f / T;
    float sqrt_inv_T = sqrtf(inv_T);

    // F_0(T) asymptotic: sqrt(pi/T) / 2
    float F0 = 0.5f * SQRT_PI_F * sqrt_inv_T;

    if (n == 0) {
        return F0;
    }

    // For n > 0: F_n(T) = (2n-1)!! / 2^(n+1) * sqrt(pi) / T^(n+0.5)
    // = (2n-1)!! / 2^(n+1) * sqrt(pi/T) / T^n
    // Use recurrence to avoid large intermediate values

    float result = F0;
    float half_inv_T = 0.5f * inv_T;

    for (int i = 0; i < n; ++i) {
        result *= static_cast<float>(2*i + 1) * half_inv_T;
    }

    return result;
}

/// @brief Evaluate F_0(T), F_1(T), ..., F_{n_max}(T) using asymptotic expansion (float32)
/// @param n_max Maximum order to compute (0 ≤ n_max ≤ BOYS_MAX_N_FLOAT)
/// @param T Argument value (T > BOYS_ASYMPTOTIC_THRESHOLD_FLOAT)
/// @param result Output array, must have size at least n_max + 1
inline void boys_asymptotic_array_float(int n_max, float T, float* result) {
    constexpr float SQRT_PI_F = 1.7724538509f;

    float inv_T = 1.0f / T;
    float exp_neg_T = expf(-T);

    // F_0(T) asymptotic
    result[0] = 0.5f * SQRT_PI_F * sqrtf(inv_T);

    // Upward recursion: F_n(T) = ((2n-1) * F_{n-1}(T) - exp(-T)) / (2T)
    float two_T = 2.0f * T;
    for (int n = 1; n <= n_max; ++n) {
        result[n] = (static_cast<float>(2*n - 1) * result[n - 1] - exp_neg_T) / two_T;
    }
}

// ============================================================================
// Float32 Boys Function - Taylor Series for Small T
// ============================================================================

/// @brief Evaluate F_n(T) using Taylor series for small T (float32)
/// @param n Order of the Boys function
/// @param T Argument value (T close to 0)
/// @return Value of F_n(T)
///
/// For small T, F_n(T) ≈ 1/(2n+1) - T/(2n+3) + T^2/(2*(2n+5)) - ...
inline float boys_taylor_float(int n, float T) {
    // F_n(0) = 1/(2n+1)
    float two_n_plus_1 = static_cast<float>(2*n + 1);
    float result = 1.0f / two_n_plus_1;

    if (T < 1.0e-10f) {
        return result;
    }

    // Taylor expansion: sum_{k=0}^{inf} (-T)^k / (k! * (2n+2k+1))
    float term = result;
    float neg_T = -T;

    for (int k = 1; k <= 15; ++k) {
        term *= neg_T / static_cast<float>(k);
        float denominator = static_cast<float>(2*n + 2*k + 1);
        float delta = term * two_n_plus_1 / denominator;
        result += delta;

        if (fabsf(delta) < 1.0e-7f * fabsf(result)) {
            break;
        }
    }

    return result;
}

// ============================================================================
// Float32 Boys Function - Chebyshev Interpolation
// ============================================================================

/// @brief Chebyshev coefficients for float32 Boys function interpolation
///
/// The interval [0, 30) is divided into 30 unit-width subintervals.
/// Each is approximated by a degree-7 Chebyshev polynomial (8 terms).
/// This gives relative error < 1e-6 across the range.
struct BoysFloatChebyshevTable {
    /// Number of interpolation intervals
    static constexpr int N_INTERVALS = 30;

    /// Number of Chebyshev coefficients per interval
    static constexpr int N_COEFFS = 8;

    /// Chebyshev coefficients for F_0(T)
    /// coeffs[interval][coeff_index]
    /// Generated from high-precision reference values
    alignas(32) float coeffs_f0[N_INTERVALS][N_COEFFS];

    /// Chebyshev coefficients for F_1(T) through F_20(T)
    /// For higher orders, we use downward recursion from the table value
};

/// @brief Evaluate F_n(T) using Chebyshev interpolation (float32)
/// @param n Order of the Boys function (0 ≤ n ≤ BOYS_MAX_N_FLOAT)
/// @param T Argument value (0 ≤ T < BOYS_CHEBYSHEV_T_MAX_FLOAT)
/// @return Value of F_n(T)
///
/// Uses precomputed Chebyshev coefficients with Clenshaw recurrence.
float boys_chebyshev_float(int n, float T);

/// @brief Evaluate F_0(T), ..., F_{n_max}(T) using Chebyshev + downward recursion (float32)
/// @param n_max Maximum order to compute
/// @param T Argument value
/// @param result Output array
void boys_chebyshev_array_float(int n_max, float T, float* result);

// ============================================================================
// Float32 Boys Function - Unified Interface
// ============================================================================

/// @brief Evaluate F_0(T), F_1(T), ..., F_{n_max}(T) for any T ≥ 0 (float32)
/// @param n_max Maximum order to compute (0 ≤ n_max ≤ BOYS_MAX_N_FLOAT)
/// @param T Argument value (T ≥ 0)
/// @param result Output array, must have size at least n_max + 1
///
/// Automatically selects the optimal evaluation strategy:
/// - T ≈ 0: Direct formula F_n(0) = 1/(2n+1)
/// - T < 1e-6: Taylor expansion
/// - T < BOYS_ASYMPTOTIC_THRESHOLD_FLOAT (25): Chebyshev start + downward recursion
/// - T ≥ BOYS_ASYMPTOTIC_THRESHOLD_FLOAT: Asymptotic F_0 + upward recursion
inline void boys_evaluate_array_float(int n_max, float T, float* result) {
    // Handle T = 0 exactly
    if (T == 0.0f) {
        for (int n = 0; n <= n_max; ++n) {
            result[n] = 1.0f / static_cast<float>(2*n + 1);
        }
        return;
    }

    // Very small T: use Taylor expansion
    if (T < 1.0e-6f) {
        for (int n = 0; n <= n_max; ++n) {
            result[n] = boys_taylor_float(n, T);
        }
        return;
    }

    // Large T: asymptotic expansion with upward recursion
    if (T >= BOYS_ASYMPTOTIC_THRESHOLD_FLOAT) {
        boys_asymptotic_array_float(n_max, T, result);
        return;
    }

    // Medium T: Chebyshev for highest order, then downward recursion
    // Downward recursion is stable: F_{n-1}(T) = (2T * F_n(T) + exp(-T)) / (2n-1)
    result[n_max] = boys_chebyshev_float(n_max, T);

    if (n_max == 0) return;

    const float exp_neg_T = expf(-T);
    const float two_T = 2.0f * T;

    for (int n = n_max; n >= 1; --n) {
        result[n - 1] = (two_T * result[n] + exp_neg_T) / static_cast<float>(2 * n - 1);
    }
}

/// @brief Evaluate a single Boys function value F_n(T) for any T ≥ 0 (float32)
/// @param n Order of the Boys function (0 ≤ n ≤ BOYS_MAX_N_FLOAT)
/// @param T Argument value (T ≥ 0)
/// @return Value of F_n(T)
inline float boys_evaluate_float(int n, float T) {
    float result[BOYS_MAX_N_FLOAT + 1];
    boys_evaluate_array_float(n, T, result);
    return result[n];
}

/// @brief Evaluate Boys function for multiple T values (float32)
/// @param n_max Maximum order to compute
/// @param T_array Array of T values
/// @param n_values Number of T values
/// @param result Output array, size n_values * (n_max + 1)
inline void boys_evaluate_batch_float(int n_max, const float* T_array, int n_values,
                                       float* result) {
    for (int i = 0; i < n_values; ++i) {
        boys_evaluate_array_float(n_max, T_array[i], result + i * (n_max + 1));
    }
}

// ============================================================================
// Templated Boys Function Interface
// ============================================================================

/// @brief Float overload of boys_evaluate_array
/// Dispatches to optimized float implementation
inline void boys_evaluate_array(int n_max, float T, float* result) {
    boys_evaluate_array_float(n_max, T, result);
}

/// @brief Templated Boys function array evaluation
/// @tparam Real Precision type (float or double)
template<typename Real>
    requires ValidPrecision<Real>
inline void boys_evaluate_array_dispatch(int n_max, Real T, Real* result) {
    if constexpr (std::is_same_v<Real, float>) {
        boys_evaluate_array_float(n_max, T, result);
    } else {
        // Call the double-precision free function from boys_function.hpp
        boys_evaluate_array(n_max, T, result);
    }
}

/// @brief Float overload of boys_function
/// Dispatches to optimized float implementation
inline float boys_function(int n, float T) {
    return boys_evaluate_float(n, T);
}

/// @brief Templated single Boys function value
template<typename Real>
    requires ValidPrecision<Real>
inline Real boys_function(int n, Real T) {
    if constexpr (std::is_same_v<Real, float>) {
        return boys_evaluate_float(n, T);
    } else {
        return boys_evaluate(n, T);
    }
}

}  // namespace libaccint::math
