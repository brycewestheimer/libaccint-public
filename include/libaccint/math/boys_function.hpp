// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file boys_function.hpp
/// @brief Boys function evaluation F_n(T) = ∫₀¹ t^(2n) exp(-T t²) dt
///
/// The Boys function is fundamental to electron repulsion integral evaluation
/// in Gaussian basis sets. This implementation provides multiple evaluation
/// strategies optimized for different regimes:
/// - Asymptotic expansion for large T (T > 30)
/// - Chebyshev interpolation for small/medium T (T in [0, 36))
/// - Downward recursion for computing multiple orders efficiently
///
/// Target precision: relative error < 1e-14

#include <cmath>
#include <array>

namespace libaccint::math {

// ============================================================================
// Constants
// ============================================================================

/// @brief Threshold for switching to asymptotic expansion
///
/// For T > BOYS_ASYMPTOTIC_THRESHOLD, the asymptotic expansion provides
/// accurate and efficient evaluation. Below this threshold, Chebyshev
/// interpolation (to be implemented) is preferred.
inline constexpr double BOYS_ASYMPTOTIC_THRESHOLD = 30.0;

/// @brief Maximum angular momentum supported for Boys function
///
/// This determines the maximum n value for F_n(T). For electron repulsion
/// integrals with angular momentum l_a, l_b, l_c, l_d, we need n up to
/// l_a + l_b + l_c + l_d.
inline constexpr int BOYS_MAX_N = 30;

// ============================================================================
// Helper Functions
// ============================================================================

/// @brief Compute double factorial (2n-1)!! = (2n-1) × (2n-3) × ... × 3 × 1
/// @param n Order (must be >= 0)
/// @return Value of (2n-1)!!
///
/// Uses a lookup table for small n (up to 15), which covers most practical
/// cases. For larger n, computes directly. Note: (-1)!! = 1 by convention.
///
/// Examples:
/// - double_factorial(0) = 1  (by convention, (-1)!! = 1)
/// - double_factorial(1) = 1  (1!! = 1)
/// - double_factorial(2) = 3  (3!! = 3)
/// - double_factorial(3) = 15 (5!! = 5 × 3 × 1)
double double_factorial(int n);

// ============================================================================
// Boys Function Evaluation (Asymptotic Regime)
// ============================================================================

/// @brief Evaluate F_n(T) using asymptotic expansion for large T
/// @param n Order of the Boys function (0 ≤ n ≤ BOYS_MAX_N)
/// @param T Argument value (T > 0, preferably T > BOYS_ASYMPTOTIC_THRESHOLD)
/// @return Value of F_n(T)
///
/// For large T (T > 30), the asymptotic leading term is:
///
/// F_n(T) = (2n-1)!! / (2^(n+1)) × √(π) / T^(n+0.5)
///
/// The omitted correction is O(erfc(√T)), which for T ≥ 30 is < 1e-14,
/// giving relative accuracy better than double-precision epsilon.
///
/// @note This function is accurate for T >= 30. For smaller T, use Chebyshev
///       interpolation (to be implemented in Task 0.3.2).
double boys_asymptotic(int n, double T);

/// @brief Evaluate F_0(T), F_1(T), ..., F_{n_max}(T) using asymptotic expansion
/// @param n_max Maximum order to compute (0 ≤ n_max ≤ BOYS_MAX_N)
/// @param T Argument value (T > 0, preferably T > BOYS_ASYMPTOTIC_THRESHOLD)
/// @param result Output array, must have size at least n_max + 1
///
/// Computes all orders from 0 to n_max using the asymptotic formula for F_0(T)
/// and upward recursion:
///
/// F_n(T) = ((2n-1) × F_{n-1}(T) - exp(-T)) / (2T)
///
/// This is more efficient than calling boys_asymptotic repeatedly when multiple
/// orders are needed.
///
/// @note For maximum efficiency, use downward recursion (to be implemented in
///       Task 0.3.3), which is numerically stable and faster. This upward
///       recursion is provided for the asymptotic regime only.
void boys_asymptotic_array(int n_max, double T, double* result);

// ============================================================================
// Boys Function Evaluation (Chebyshev Interpolation Regime)
// ============================================================================

/// @brief Evaluate F_n(T) using Chebyshev interpolation for small/medium T
/// @param n Order of the Boys function (0 ≤ n ≤ BOYS_MAX_N)
/// @param T Argument value (0 ≤ T < BOYS_CHEBYSHEV_T_MAX = 36)
/// @return Value of F_n(T)
///
/// Uses precomputed Chebyshev coefficients evaluated via Clenshaw recurrence.
/// The interval [0, 36) is divided into 36 unit-width subintervals, each
/// approximated by a degree-11 Chebyshev polynomial (12 terms). Coefficients
/// were generated offline using mpmath at 60-digit precision.
///
/// Target accuracy: relative error < 1e-15 across the entire range.
///
/// @note For T ≥ 36, use boys_asymptotic() instead. The unified dispatch
///       function (Task 0.3.4) will select the appropriate method automatically.
double boys_chebyshev(int n, double T);

/// @brief Evaluate F_0(T), F_1(T), ..., F_{n_max}(T) using Chebyshev interpolation
/// @param n_max Maximum order to compute (0 ≤ n_max ≤ BOYS_MAX_N)
/// @param T Argument value (0 ≤ T < BOYS_CHEBYSHEV_T_MAX = 36)
/// @param result Output array, must have size at least n_max + 1
///
/// Evaluates all orders from 0 to n_max using Chebyshev interpolation.
/// Each order uses its own set of precomputed coefficients, so this is
/// equivalent to calling boys_chebyshev() for each order but shares the
/// interval lookup and coordinate mapping.
void boys_chebyshev_array(int n_max, double T, double* result);

/// @brief Maximum T value for Chebyshev interpolation
///
/// For T >= this value, use the asymptotic expansion instead.
inline constexpr double BOYS_CHEBYSHEV_T_MAX = 36.0;

// ============================================================================
// Boys Function Unified Evaluation (Primary Interface)
// ============================================================================

/// @brief Evaluate F_0(T), F_1(T), ..., F_{n_max}(T) for any T ≥ 0
/// @param n_max Maximum order to compute (0 ≤ n_max ≤ BOYS_MAX_N)
/// @param T Argument value (T ≥ 0)
/// @param result Output array, must have size at least n_max + 1
///
/// This is the primary interface for computing the Boys function. It
/// automatically selects the optimal evaluation strategy:
///
/// - T = 0: Direct formula F_n(0) = 1/(2n+1)
/// - T < 1e-14: Taylor expansion (avoids numerical issues near T = 0)
/// - T < BOYS_CHEBYSHEV_T_MAX (36): Chebyshev start + downward recursion
/// - T ≥ BOYS_CHEBYSHEV_T_MAX: Asymptotic F_0 + upward recursion
///
/// The downward recursion is numerically stable for the Boys function:
///   F_{n-1}(T) = (2T × F_n(T) + exp(-T)) / (2n - 1)
///
/// Starting from an accurate Chebyshev value for F_{n_max}(T), this
/// recursion fills all lower orders to machine precision.
///
/// @note All errors < 1e-14 compared to arbitrary-precision reference.
void boys_evaluate_array(int n_max, double T, double* result);

/// @brief Evaluate a single Boys function value F_n(T) for any T ≥ 0
/// @param n Order of the Boys function (0 ≤ n ≤ BOYS_MAX_N)
/// @param T Argument value (T ≥ 0)
/// @return Value of F_n(T)
///
/// Convenience function for single-value evaluation. For computing multiple
/// orders, boys_evaluate_array() is more efficient.
double boys_evaluate(int n, double T);

// ============================================================================
// Boys Function Batch Evaluation
// ============================================================================

/// @brief Evaluate Boys function for multiple T values simultaneously
/// @param n_max Maximum order to compute (0 ≤ n_max ≤ BOYS_MAX_N)
/// @param T_array Array of T values, each ≥ 0
/// @param n_values Number of T values in T_array
/// @param result Output array, must have size at least n_values * (n_max + 1)
///
/// Computes F_0(T_i), F_1(T_i), ..., F_{n_max}(T_i) for each T_i in T_array.
///
/// Memory layout: result[i * (n_max + 1) + n] = F_n(T_i)
///
/// This groups all n values for a given T_i contiguously, matching the
/// access pattern in integral kernels (one primitive pair at a time,
/// accessing all n values). T values are processed in order for
/// cache-friendly access.
///
/// This is the primary entry point used by integral kernels.
void boys_evaluate_batch(int n_max, const double* T_array, int n_values,
                         double* result);

// ============================================================================
// SIMD-Accelerated Boys Function Evaluation
// ============================================================================

/// @brief Evaluate Boys function for multiple T values using SIMD
/// @param n_max Maximum order to compute (0 ≤ n_max ≤ BOYS_MAX_N)
/// @param T_array Array of T values (preferably aligned for SIMD), each ≥ 0
/// @param n_values Number of T values in T_array
/// @param result Output array, must have size at least n_values * (n_max + 1)
///
/// SIMD-optimized version that processes 4 T values simultaneously using
/// AVX2 instructions. Falls back to scalar evaluation for:
/// - Remainder values when n_values is not divisible by SIMD width
/// - T values in different regimes (asymptotic vs Chebyshev)
///
/// Memory layout: Same as boys_evaluate_batch
/// Performance: Target 2x+ speedup over scalar batch evaluation
void boys_evaluate_batch_simd(int n_max, const double* T_array, int n_values,
                               double* result);

/// @brief SIMD-vectorized Chebyshev evaluation for same-regime T values
/// @param n Order to compute
/// @param T Pointer to 4 T values (must all be in Chebyshev regime)
/// @param result Pointer to 4 output values
///
/// Low-level SIMD kernel that evaluates F_n(T) for 4 T values simultaneously.
/// All T values must be in the same interval for optimal performance.
/// Used internally by boys_evaluate_batch_simd.
void boys_chebyshev_simd4(int n, const double* T, double* result);

} // namespace libaccint::math
