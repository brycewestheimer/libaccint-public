// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file boys_range_separated.hpp
/// @brief Range-separated Boys function for erf/erfc Coulomb operators
///
/// Implements the modified Boys function F_n^{erf}(T, omega) and F_n^{erfc}(T, omega)
/// for range-separated Coulomb operators used in CAM-B3LYP, omega-B97X, LC-BLYP, etc.
///
/// Mathematical background:
/// The range-separated Coulomb operator decomposes as:
///   1/r = erf(omega * r)/r + erfc(omega * r)/r
///
/// The modified Boys function for erf-attenuated operator:
///   F_n^{erf}(T, omega) = (omega^2/(omega^2 + 1))^{n+1/2} * F_n(T_eff)
///
/// where T_eff = T * omega^2 / (omega^2 + 1)
///
/// Key properties:
/// - omega -> 0: F_n^{erf}(T, omega) -> 0 (short-range limit)
/// - omega -> infinity: F_n^{erf}(T, omega) -> F_n(T) (full Coulomb)
/// - F_n^{erf}(T, omega) + F_n^{erfc}(T, omega) = F_n(T)
///
/// Target precision: relative error < 1e-14

#include <libaccint/math/boys_function.hpp>
#include <cmath>

namespace libaccint::math {

// ============================================================================
// Range-Separation Parameter Handling
// ============================================================================

/// @brief Compute the effective exponent and scaling factor for range-separated Boys
/// @param omega Range-separation parameter (in atomic units)
/// @param[out] omega2_ratio omega^2 / (omega^2 + 1)
/// @return True if omega is in valid range (omega > 0)
[[nodiscard]] inline bool range_sep_params(double omega, double& omega2_ratio) noexcept {
    if (omega <= 0.0) {
        omega2_ratio = 0.0;
        return false;
    }
    const double omega2 = omega * omega;
    omega2_ratio = omega2 / (omega2 + 1.0);
    return true;
}

/// @brief Compute the scaling prefactor for F_n^{erf}
/// @param omega2_ratio omega^2 / (omega^2 + 1) (precomputed)
/// @param n Order of Boys function
/// @return (omega^2/(omega^2 + 1))^{n+1/2}
[[nodiscard]] inline double range_sep_scaling(double omega2_ratio, int n) noexcept {
    // (omega^2/(omega^2+1))^{n+1/2} = omega2_ratio^n * sqrt(omega2_ratio)
    return std::pow(omega2_ratio, n + 0.5);
}

// ============================================================================
// Range-Separated Boys Function (erf-attenuated)
// ============================================================================

/// @brief Evaluate erf-attenuated Boys function F_n^{erf}(T, omega)
/// @param n Order of the Boys function (0 <= n <= BOYS_MAX_N)
/// @param T Argument value (T >= 0)
/// @param omega Range-separation parameter (omega > 0)
/// @return Value of F_n^{erf}(T, omega)
///
/// Uses the transformation:
///   F_n^{erf}(T, omega) = scale * F_n(T_eff)
/// where:
///   scale = (omega^2 / (omega^2 + 1))^{n+1/2}
///   T_eff = T * omega^2 / (omega^2 + 1)
///
/// Limiting behavior:
/// - omega -> 0: Returns 0 (short-range, no contribution)
/// - omega -> infinity: Returns F_n(T) (full Coulomb)
[[nodiscard]] double boys_erf(int n, double T, double omega);

/// @brief Evaluate F_0^{erf}, F_1^{erf}, ..., F_{n_max}^{erf} for given T and omega
/// @param n_max Maximum order to compute (0 <= n_max <= BOYS_MAX_N)
/// @param T Argument value (T >= 0)
/// @param omega Range-separation parameter (omega > 0)
/// @param result Output array, must have size at least n_max + 1
///
/// More efficient than calling boys_erf() repeatedly when multiple orders needed.
void boys_erf_array(int n_max, double T, double omega, double* result);

// ============================================================================
// Range-Separated Boys Function (erfc-attenuated)
// ============================================================================

/// @brief Evaluate erfc-attenuated Boys function F_n^{erfc}(T, omega)
/// @param n Order of the Boys function (0 <= n <= BOYS_MAX_N)
/// @param T Argument value (T >= 0)
/// @param omega Range-separation parameter (omega > 0)
/// @return Value of F_n^{erfc}(T, omega)
///
/// Uses the identity:
///   F_n^{erfc}(T, omega) = F_n(T) - F_n^{erf}(T, omega)
///
/// Limiting behavior:
/// - omega -> 0: Returns F_n(T) (full Coulomb, since erf part -> 0)
/// - omega -> infinity: Returns 0 (short-range only)
[[nodiscard]] double boys_erfc(int n, double T, double omega);

/// @brief Evaluate F_0^{erfc}, F_1^{erfc}, ..., F_{n_max}^{erfc} for given T and omega
/// @param n_max Maximum order to compute (0 <= n_max <= BOYS_MAX_N)
/// @param T Argument value (T >= 0)
/// @param omega Range-separation parameter (omega > 0)
/// @param result Output array, must have size at least n_max + 1
///
/// More efficient than calling boys_erfc() repeatedly when multiple orders needed.
void boys_erfc_array(int n_max, double T, double omega, double* result);

// ============================================================================
// Batch Evaluation for Integral Kernels
// ============================================================================

/// @brief Batch evaluate erf-attenuated Boys function for multiple T values
/// @param n_max Maximum order to compute (0 <= n_max <= BOYS_MAX_N)
/// @param T_array Array of T values (each >= 0)
/// @param n_values Number of T values
/// @param omega Range-separation parameter (same for all T values)
/// @param result Output array, size >= n_values * (n_max + 1)
///
/// Memory layout: result[i * (n_max + 1) + n] = F_n^{erf}(T_array[i], omega)
void boys_erf_batch(int n_max, const double* T_array, int n_values,
                    double omega, double* result);

/// @brief Batch evaluate erfc-attenuated Boys function for multiple T values
/// @param n_max Maximum order to compute (0 <= n_max <= BOYS_MAX_N)
/// @param T_array Array of T values (each >= 0)
/// @param n_values Number of T values
/// @param omega Range-separation parameter (same for all T values)
/// @param result Output array, size >= n_values * (n_max + 1)
///
/// Memory layout: result[i * (n_max + 1) + n] = F_n^{erfc}(T_array[i], omega)
void boys_erfc_batch(int n_max, const double* T_array, int n_values,
                     double omega, double* result);

}  // namespace libaccint::math
