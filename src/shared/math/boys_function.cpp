// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include "libaccint/math/boys_function.hpp"
#include "libaccint/math/boys_tables.hpp"
#include "libaccint/utils/constants.hpp"
#include "libaccint/utils/simd.hpp"
#include <cmath>
#include <array>
#include <cassert>
#include <cstring>

namespace libaccint::math {

// ============================================================================
// Double Factorial Lookup Table
// ============================================================================

namespace {

/// @brief Precomputed double factorial values (2n-1)!! for n = 0..15
///
/// These are exact double values (no rounding errors) for small n.
/// For n > 15, we compute on-the-fly.
constexpr std::array<double, 16> DOUBLE_FACTORIAL_TABLE = {
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

} // anonymous namespace

// ============================================================================
// Double Factorial Implementation
// ============================================================================

double double_factorial(int n) {
    assert(n >= 0 && "double_factorial: n must be non-negative");

    // Use lookup table for small n
    if (n < static_cast<int>(DOUBLE_FACTORIAL_TABLE.size())) {
        return DOUBLE_FACTORIAL_TABLE[n];
    }

    // For larger n, compute iteratively
    // (2n-1)!! = (2n-1) × (2n-3) × ... × 3 × 1
    double result = DOUBLE_FACTORIAL_TABLE.back();
    for (int k = DOUBLE_FACTORIAL_TABLE.size(); k <= n; ++k) {
        result *= (2 * k - 1);
    }

    return result;
}

// ============================================================================
// Boys Function Asymptotic Expansion
// ============================================================================

double boys_asymptotic(int n, double T) {
    assert(n >= 0 && n <= BOYS_MAX_N && "boys_asymptotic: n out of range");
    assert(T > 0.0 && "boys_asymptotic: T must be positive");

    // Asymptotic leading term:
    //   F_n(T) = (2n-1)!! / (2^(n+1)) × √π / T^(n+0.5)
    //
    // This is the dominant contribution for large T. The omitted correction
    // is O(exp(-T) × √T / √π), which equals erfc(√T) for n=0 and decreases
    // with increasing n. For T >= 30, this correction is < 1e-14, giving
    // relative accuracy better than double-precision epsilon.
    //
    // For higher precision near the threshold, the array function
    // boys_asymptotic_array() naturally includes the exp(-T) correction
    // through the upward recursion formula.
    const double df = double_factorial(n);
    const double power_of_2 = std::pow(2.0, n + 1);
    const double T_power = std::pow(T, n + 0.5);
    return df / power_of_2 * constants::SQRT_PI / T_power;
}

// ============================================================================
// Boys Function Array Evaluation (Asymptotic)
// ============================================================================

void boys_asymptotic_array(int n_max, double T, double* result) {
    assert(n_max >= 0 && n_max <= BOYS_MAX_N &&
           "boys_asymptotic_array: n_max out of range");
    assert(T > 0.0 && "boys_asymptotic_array: T must be positive");
    assert(result != nullptr && "boys_asymptotic_array: result is null");

    // Compute F_0(T) using asymptotic formula
    result[0] = boys_asymptotic(0, T);

    // For n_max = 0, we're done
    if (n_max == 0) {
        return;
    }

    // Compute exp(-T) once (needed for recursion)
    const double exp_neg_T = std::exp(-T);

    // Upward recursion: F_n(T) = ((2n-1) × F_{n-1}(T) - exp(-T)) / (2T)
    //
    // This recursion is derived from integration by parts:
    // F_n(T) = ∫₀¹ t^(2n) exp(-T t²) dt
    //
    // Note: For general use, downward recursion is more stable and will be
    // implemented in Task 0.3.3. For the asymptotic regime (large T), upward
    // recursion is acceptable because the exponential term is very small.

    const double inv_2T = 1.0 / (2.0 * T);

    for (int n = 1; n <= n_max; ++n) {
        result[n] = ((2 * n - 1) * result[n - 1] - exp_neg_T) * inv_2T;
    }
}

// ============================================================================
// Boys Function Chebyshev Interpolation
// ============================================================================

namespace {

/// @brief Evaluate Chebyshev expansion via Clenshaw recurrence
///
/// Computes S = c_0 + c_1*T_1(x) + c_2*T_2(x) + ... + c_{N-1}*T_{N-1}(x)
/// where c_0 is the (already halved) constant term and T_j are Chebyshev
/// polynomials of the first kind.
///
/// @param coeffs Pointer to N Chebyshev coefficients
/// @param N Number of terms
/// @param x Evaluation point in [-1, 1]
/// @return Value of the Chebyshev expansion at x
inline double clenshaw_eval(const double* coeffs, int N, double x) {
    assert(N > 0 && "clenshaw_eval: N must be positive");
    assert(coeffs != nullptr && "clenshaw_eval: coeffs must not be null");

    // Clenshaw recurrence:
    //   d_{N+1} = d_N = 0
    //   d_k = 2*x*d_{k+1} - d_{k+2} + c_k   for k = N-1, N-2, ..., 1
    //   S = c_0 + x*d_1 - d_2
    double d_next1 = 0.0;  // d_{k+1}
    double d_next2 = 0.0;  // d_{k+2}

    for (int j = N - 1; j >= 1; --j) {
        double d_k = 2.0 * x * d_next1 - d_next2 + coeffs[j];
        d_next2 = d_next1;
        d_next1 = d_k;
    }

    return coeffs[0] + x * d_next1 - d_next2;
}

} // anonymous namespace

double boys_chebyshev(int n, double T) {
    assert(n >= 0 && n <= BOYS_MAX_N && "boys_chebyshev: n out of range");
    assert(T >= 0.0 && T < BOYS_CHEBYSHEV_T_MAX &&
           "boys_chebyshev: T out of Chebyshev range [0, T_MAX)");

    // Determine which interval T falls in
    const int interval = static_cast<int>(T / detail::BOYS_CHEB_INTERVAL_WIDTH);

    // Clamp interval to valid range (handles T exactly at boundary due to
    // floating-point rounding)
    const int safe_interval = (interval >= detail::BOYS_CHEB_N_INTERVALS)
                                  ? detail::BOYS_CHEB_N_INTERVALS - 1
                                  : interval;

    // Map T to x in [-1, 1] within the interval
    // interval covers [a, b) where a = interval * width, b = a + width
    const double a = safe_interval * detail::BOYS_CHEB_INTERVAL_WIDTH;
    const double b = a + detail::BOYS_CHEB_INTERVAL_WIDTH;
    const double x = (2.0 * T - a - b) / (b - a);

    // Evaluate using Clenshaw recurrence
    return clenshaw_eval(
        detail::BOYS_CHEBYSHEV_COEFFICIENTS[safe_interval][n],
        detail::BOYS_CHEB_N_TERMS, x);
}

void boys_chebyshev_array(int n_max, double T, double* result) {
    assert(n_max >= 0 && n_max <= BOYS_MAX_N &&
           "boys_chebyshev_array: n_max out of range");
    assert(T >= 0.0 && T < BOYS_CHEBYSHEV_T_MAX &&
           "boys_chebyshev_array: T out of Chebyshev range [0, T_MAX)");
    assert(result != nullptr && "boys_chebyshev_array: result is null");

    // Determine which interval T falls in
    const int interval = static_cast<int>(T / detail::BOYS_CHEB_INTERVAL_WIDTH);
    const int safe_interval = (interval >= detail::BOYS_CHEB_N_INTERVALS)
                                  ? detail::BOYS_CHEB_N_INTERVALS - 1
                                  : interval;

    // Map T to x in [-1, 1] within the interval
    const double a = safe_interval * detail::BOYS_CHEB_INTERVAL_WIDTH;
    const double b = a + detail::BOYS_CHEB_INTERVAL_WIDTH;
    const double x = (2.0 * T - a - b) / (b - a);

    // Evaluate each order using Clenshaw recurrence
    for (int n = 0; n <= n_max; ++n) {
        result[n] = clenshaw_eval(
            detail::BOYS_CHEBYSHEV_COEFFICIENTS[safe_interval][n],
            detail::BOYS_CHEB_N_TERMS, x);
    }
}

// ============================================================================
// Boys Function Unified Evaluation
// ============================================================================

namespace {

/// @brief Taylor expansion for F_n(T) when T is very small
///
/// F_n(T) = Σ_{k=0}^{K} (-T)^k / (k! × (2n+2k+1))
///
/// For T < 1e-14, this series converges to machine precision in a few terms.
/// Avoids potential issues with Chebyshev interpolation near T = 0.
inline double boys_taylor(int n, double T) {
    double result = 1.0 / (2 * n + 1);
    double term = 1.0;

    for (int k = 1; k <= 20; ++k) {
        term *= -T / k;
        double contribution = term / (2 * n + 2 * k + 1);
        result += contribution;
        if (std::abs(contribution) < 1e-16 * std::abs(result)) {
            break;
        }
    }

    return result;
}

/// @brief Threshold below which Taylor expansion is used
constexpr double BOYS_TAYLOR_THRESHOLD = 1e-14;

} // anonymous namespace

void boys_evaluate_array(int n_max, double T, double* result) {
    assert(n_max >= 0 && n_max <= BOYS_MAX_N &&
           "boys_evaluate_array: n_max out of range");
    assert(T >= 0.0 && "boys_evaluate_array: T must be non-negative");
    assert(result != nullptr && "boys_evaluate_array: result is null");

    // Case 1: T = 0 exactly — F_n(0) = 1/(2n+1)
    if (T == 0.0) {
        for (int n = 0; n <= n_max; ++n) {
            result[n] = 1.0 / (2 * n + 1);
        }
        return;
    }

    // Case 2: Very small T — Taylor expansion
    if (T < BOYS_TAYLOR_THRESHOLD) {
        for (int n = 0; n <= n_max; ++n) {
            result[n] = boys_taylor(n, T);
        }
        return;
    }

    // Case 3: Large T — asymptotic F_0 + upward recursion
    if (T >= BOYS_CHEBYSHEV_T_MAX) {
        boys_asymptotic_array(n_max, T, result);
        return;
    }

    // Case 4: Small/medium T — Chebyshev start + downward recursion
    //
    // Get F_{n_max}(T) from Chebyshev interpolation (accurate to ~1e-16),
    // then use stable downward recursion to fill F_{n_max-1}..F_0.
    //
    // Downward recursion: F_{n-1}(T) = (2T × F_n(T) + exp(-T)) / (2n - 1)
    //
    // The Boys function is the minimal solution of the three-term recurrence,
    // so downward recursion preserves relative error regardless of the
    // amplification factor 2T/(2n-1) at individual steps.
    result[n_max] = boys_chebyshev(n_max, T);

    if (n_max == 0) {
        return;
    }

    const double exp_neg_T = std::exp(-T);
    const double two_T = 2.0 * T;

    for (int n = n_max; n >= 1; --n) {
        result[n - 1] = (two_T * result[n] + exp_neg_T) / (2 * n - 1);
    }
}

double boys_evaluate(int n, double T) {
    assert(n >= 0 && n <= BOYS_MAX_N && "boys_evaluate: n out of range");
    assert(T >= 0.0 && "boys_evaluate: T must be non-negative");

    if (T == 0.0) {
        return 1.0 / (2 * n + 1);
    }

    if (T < BOYS_TAYLOR_THRESHOLD) {
        return boys_taylor(n, T);
    }

    if (T >= BOYS_CHEBYSHEV_T_MAX) {
        return boys_asymptotic(n, T);
    }

    return boys_chebyshev(n, T);
}

// ============================================================================
// Boys Function Batch Evaluation
// ============================================================================

void boys_evaluate_batch(int n_max, const double* T_array, int n_values,
                         double* result) {
    assert(n_max >= 0 && n_max <= BOYS_MAX_N &&
           "boys_evaluate_batch: n_max out of range");
    assert(T_array != nullptr && "boys_evaluate_batch: T_array is null");
    assert(n_values >= 0 && "boys_evaluate_batch: n_values must be non-negative");
    assert(result != nullptr && "boys_evaluate_batch: result is null");

    const int stride = n_max + 1;

    for (int i = 0; i < n_values; ++i) {
        boys_evaluate_array(n_max, T_array[i], result + i * stride);
    }
}

// ============================================================================
// SIMD-Accelerated Boys Function Evaluation
// ============================================================================

#if defined(LIBACCINT_SIMD_AVX2)

namespace {

/// @brief Vectorized Clenshaw recurrence for 4 T values
///
/// Evaluates the Chebyshev polynomial expansion for 4 T values simultaneously.
/// All T values must be mapped to x coordinates in [-1, 1].
inline __m256d clenshaw_eval_simd4(const double* coeffs, int N,
                                    __m256d x) {
    __m256d d_next1 = _mm256_setzero_pd();  // d_{k+1}
    __m256d d_next2 = _mm256_setzero_pd();  // d_{k+2}
    __m256d two_x = _mm256_add_pd(x, x);

    for (int j = N - 1; j >= 1; --j) {
        __m256d coeff = _mm256_set1_pd(coeffs[j]);
        __m256d d_k = _mm256_fmadd_pd(two_x, d_next1, coeff);
        d_k = _mm256_sub_pd(d_k, d_next2);
        d_next2 = d_next1;
        d_next1 = d_k;
    }

    // Final step: S = c_0 + x*d_1 - d_2
    __m256d c0 = _mm256_set1_pd(coeffs[0]);
    __m256d result = _mm256_fmadd_pd(x, d_next1, c0);
    result = _mm256_sub_pd(result, d_next2);

    return result;
}

/// @brief Check if all 4 T values fall in the same Chebyshev interval
inline bool same_interval_simd4(const double* T, int& interval) {
    int i0 = static_cast<int>(T[0] / detail::BOYS_CHEB_INTERVAL_WIDTH);
    int i1 = static_cast<int>(T[1] / detail::BOYS_CHEB_INTERVAL_WIDTH);
    int i2 = static_cast<int>(T[2] / detail::BOYS_CHEB_INTERVAL_WIDTH);
    int i3 = static_cast<int>(T[3] / detail::BOYS_CHEB_INTERVAL_WIDTH);

    if (i0 == i1 && i1 == i2 && i2 == i3) {
        interval = (i0 >= detail::BOYS_CHEB_N_INTERVALS)
                       ? detail::BOYS_CHEB_N_INTERVALS - 1
                       : i0;
        return true;
    }
    return false;
}

/// @brief Threshold for Taylor expansion in SIMD context
constexpr double BOYS_SIMD_TAYLOR_THRESHOLD = 1e-14;

/// @brief Check if all 4 T values are in Chebyshev regime
inline bool all_chebyshev_regime_simd4(const double* T) {
    return T[0] < BOYS_CHEBYSHEV_T_MAX &&
           T[1] < BOYS_CHEBYSHEV_T_MAX &&
           T[2] < BOYS_CHEBYSHEV_T_MAX &&
           T[3] < BOYS_CHEBYSHEV_T_MAX &&
           T[0] >= BOYS_SIMD_TAYLOR_THRESHOLD &&
           T[1] >= BOYS_SIMD_TAYLOR_THRESHOLD &&
           T[2] >= BOYS_SIMD_TAYLOR_THRESHOLD &&
           T[3] >= BOYS_SIMD_TAYLOR_THRESHOLD;
}

}  // anonymous namespace

void boys_chebyshev_simd4(int n, const double* T, double* result) {
    assert(n >= 0 && n <= BOYS_MAX_N && "boys_chebyshev_simd4: n out of range");

    // Load T values
    __m256d T_v = _mm256_loadu_pd(T);

    // Compute intervals for each T
    __m256d inv_width = _mm256_set1_pd(1.0 / detail::BOYS_CHEB_INTERVAL_WIDTH);
    __m256d interval_d = _mm256_mul_pd(T_v, inv_width);

    // Floor to get interval indices
    __m256d interval_floor = _mm256_floor_pd(interval_d);

    // Check if all in same interval (fast path)
    int interval;
    if (same_interval_simd4(T, interval)) {
        // All T values in same interval - fully vectorized path
        const double a = interval * detail::BOYS_CHEB_INTERVAL_WIDTH;
        const double b = a + detail::BOYS_CHEB_INTERVAL_WIDTH;

        // Map T to x in [-1, 1]: x = (2*T - a - b) / (b - a)
        __m256d a_v = _mm256_set1_pd(a);
        __m256d b_v = _mm256_set1_pd(b);
        __m256d two = _mm256_set1_pd(2.0);
        __m256d x = _mm256_mul_pd(two, T_v);
        x = _mm256_sub_pd(x, a_v);
        x = _mm256_sub_pd(x, b_v);
        __m256d width = _mm256_sub_pd(b_v, a_v);
        x = _mm256_div_pd(x, width);

        // Evaluate Chebyshev polynomial
        __m256d val = clenshaw_eval_simd4(
            detail::BOYS_CHEBYSHEV_COEFFICIENTS[interval][n],
            detail::BOYS_CHEB_N_TERMS, x);

        _mm256_storeu_pd(result, val);
    } else {
        // Different intervals - fall back to scalar for each
        for (int i = 0; i < 4; ++i) {
            result[i] = boys_chebyshev(n, T[i]);
        }
    }
}

void boys_evaluate_batch_simd(int n_max, const double* T_array, int n_values,
                               double* result) {
    assert(n_max >= 0 && n_max <= BOYS_MAX_N &&
           "boys_evaluate_batch_simd: n_max out of range");
    assert(T_array != nullptr && "boys_evaluate_batch_simd: T_array is null");
    assert(n_values >= 0 && "boys_evaluate_batch_simd: n_values must be non-negative");
    assert(result != nullptr && "boys_evaluate_batch_simd: result is null");

    const int stride = n_max + 1;
    int i = 0;

    // Process 4 T values at a time
    for (; i + 4 <= n_values; i += 4) {
        const double* T_ptr = T_array + i;

        // Check if all 4 T values are in Chebyshev regime
        if (all_chebyshev_regime_simd4(T_ptr)) {
            // Check if all in same interval for best vectorization
            int interval;
            if (same_interval_simd4(T_ptr, interval)) {
                // Fully vectorized path: same interval, Chebyshev regime
                const double a = interval * detail::BOYS_CHEB_INTERVAL_WIDTH;
                const double b = a + detail::BOYS_CHEB_INTERVAL_WIDTH;

                // Load T values
                __m256d T_v = _mm256_loadu_pd(T_ptr);

                // Map to x in [-1, 1]
                __m256d a_v = _mm256_set1_pd(a);
                __m256d b_v = _mm256_set1_pd(b);
                __m256d two = _mm256_set1_pd(2.0);
                __m256d x = _mm256_mul_pd(two, T_v);
                x = _mm256_sub_pd(x, a_v);
                x = _mm256_sub_pd(x, b_v);
                __m256d width = _mm256_sub_pd(b_v, a_v);
                x = _mm256_div_pd(x, width);

                // Precompute exp(-T) for all 4 values
                __m256d neg_T = _mm256_sub_pd(_mm256_setzero_pd(), T_v);
                __m256d exp_neg_T = simd::exp(neg_T);
                __m256d two_T = _mm256_add_pd(T_v, T_v);

                // Temporary storage for F_n values
                alignas(32) double F_n[4];

                // Start with F_{n_max} from Chebyshev
                __m256d F_current = clenshaw_eval_simd4(
                    detail::BOYS_CHEBYSHEV_COEFFICIENTS[interval][n_max],
                    detail::BOYS_CHEB_N_TERMS, x);

                // Store F_{n_max} for each T
                _mm256_storeu_pd(F_n, F_current);
                result[(i + 0) * stride + n_max] = F_n[0];
                result[(i + 1) * stride + n_max] = F_n[1];
                result[(i + 2) * stride + n_max] = F_n[2];
                result[(i + 3) * stride + n_max] = F_n[3];

                // Downward recursion: F_{n-1}(T) = (2T * F_n(T) + exp(-T)) / (2n-1)
                for (int n = n_max; n >= 1; --n) {
                    __m256d denom = _mm256_set1_pd(2.0 * n - 1.0);
                    __m256d F_prev = _mm256_fmadd_pd(two_T, F_current, exp_neg_T);
                    F_prev = _mm256_div_pd(F_prev, denom);

                    _mm256_storeu_pd(F_n, F_prev);
                    result[(i + 0) * stride + (n - 1)] = F_n[0];
                    result[(i + 1) * stride + (n - 1)] = F_n[1];
                    result[(i + 2) * stride + (n - 1)] = F_n[2];
                    result[(i + 3) * stride + (n - 1)] = F_n[3];

                    F_current = F_prev;
                }
            } else {
                // Different intervals - process each individually but use SIMD where possible
                for (int j = 0; j < 4; ++j) {
                    boys_evaluate_array(n_max, T_ptr[j], result + (i + j) * stride);
                }
            }
        } else {
            // Mixed regimes - fall back to scalar
            for (int j = 0; j < 4; ++j) {
                boys_evaluate_array(n_max, T_ptr[j], result + (i + j) * stride);
            }
        }
    }

    // Handle remainder
    for (; i < n_values; ++i) {
        boys_evaluate_array(n_max, T_array[i], result + i * stride);
    }
}

#else  // Non-AVX2 fallback

void boys_chebyshev_simd4(int n, const double* T, double* result) {
    // Scalar fallback
    for (int i = 0; i < 4; ++i) {
        result[i] = boys_chebyshev(n, T[i]);
    }
}

void boys_evaluate_batch_simd(int n_max, const double* T_array, int n_values,
                               double* result) {
    // Fall back to scalar implementation
    boys_evaluate_batch(n_max, T_array, n_values, result);
}

#endif  // LIBACCINT_SIMD_AVX2

} // namespace libaccint::math
