// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file boys_range_separated_device.cuh
/// @brief CUDA device-side range-separated Boys function for erf/erfc operators
///
/// Provides __device__ functions for computing the range-separated Boys function
/// on the GPU. Uses the same algorithms as the CPU implementation.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include "boys_device.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace libaccint::device::math {

// ============================================================================
// Range-Separated Boys Function (erf-attenuated)
// ============================================================================

/**
 * @brief Evaluate erf-attenuated Boys function F_n^{erf}(T, omega) on device
 * @param n Order of the Boys function (0 <= n <= BOYS_MAX_N_DEVICE)
 * @param T Argument value (T >= 0)
 * @param omega Range-separation parameter (omega > 0)
 * @param d_boys_coeffs Pointer to device Chebyshev coefficients table
 * @return Value of F_n^{erf}(T, omega)
 *
 * Uses: F_n^{erf}(T, omega) = scale * F_n(T_eff)
 * where: scale = (omega^2/(omega^2+1))^{n+1/2}, T_eff = T * omega^2/(omega^2+1)
 */
__device__ __forceinline__
double boys_erf_device(int n, double T, double omega, const double* d_boys_coeffs) {
    // Handle limiting cases
    if (omega <= 0.0) {
        return 0.0;
    }

    if (omega > 1000.0) {
        return boys_evaluate_device(n, T, d_boys_coeffs);
    }

    // General case
    const double omega2 = omega * omega;
    const double omega2_ratio = omega2 / (omega2 + 1.0);
    const double T_eff = T * omega2_ratio;
    const double scale = pow(omega2_ratio, n + 0.5);

    return scale * boys_evaluate_device(n, T_eff, d_boys_coeffs);
}

/**
 * @brief Evaluate erf-attenuated Boys function array on device
 * @param n_max Maximum order to compute
 * @param T Argument value (T >= 0)
 * @param omega Range-separation parameter
 * @param d_boys_coeffs Pointer to device Chebyshev coefficients table
 * @param result Output array (must have size at least n_max + 1)
 */
__device__ __forceinline__
void boys_erf_array_device(int n_max, double T, double omega,
                           const double* d_boys_coeffs, double* result) {
    // Handle limiting cases
    if (omega <= 0.0) {
        for (int n = 0; n <= n_max; ++n) {
            result[n] = 0.0;
        }
        return;
    }

    if (omega > 1000.0) {
        boys_evaluate_array_device(n_max, T, d_boys_coeffs, result);
        return;
    }

    // General case
    const double omega2 = omega * omega;
    const double omega2_ratio = omega2 / (omega2 + 1.0);
    const double T_eff = T * omega2_ratio;
    const double sqrt_ratio = sqrt(omega2_ratio);

    // Compute standard Boys at T_eff
    boys_evaluate_array_device(n_max, T_eff, d_boys_coeffs, result);

    // Apply scaling factors
    double ratio_power = 1.0;
    for (int n = 0; n <= n_max; ++n) {
        result[n] *= sqrt_ratio * ratio_power;
        ratio_power *= omega2_ratio;
    }
}

// ============================================================================
// Range-Separated Boys Function (erfc-attenuated)
// ============================================================================

/**
 * @brief Evaluate erfc-attenuated Boys function F_n^{erfc}(T, omega) on device
 * @param n Order of the Boys function
 * @param T Argument value (T >= 0)
 * @param omega Range-separation parameter (omega > 0)
 * @param d_boys_coeffs Pointer to device Chebyshev coefficients table
 * @return Value of F_n^{erfc}(T, omega)
 */
__device__ __forceinline__
double boys_erfc_device(int n, double T, double omega, const double* d_boys_coeffs) {
    if (omega <= 0.0) {
        return boys_evaluate_device(n, T, d_boys_coeffs);
    }

    if (omega > 1000.0) {
        return 0.0;
    }

    // erfc = full - erf
    return boys_evaluate_device(n, T, d_boys_coeffs) -
           boys_erf_device(n, T, omega, d_boys_coeffs);
}

/**
 * @brief Evaluate erfc-attenuated Boys function array on device
 * @param n_max Maximum order to compute
 * @param T Argument value (T >= 0)
 * @param omega Range-separation parameter
 * @param d_boys_coeffs Pointer to device Chebyshev coefficients table
 * @param result Output array (must have size at least n_max + 1)
 */
__device__ __forceinline__
void boys_erfc_array_device(int n_max, double T, double omega,
                            const double* d_boys_coeffs, double* result) {
    if (omega <= 0.0) {
        boys_evaluate_array_device(n_max, T, d_boys_coeffs, result);
        return;
    }

    if (omega > 1000.0) {
        for (int n = 0; n <= n_max; ++n) {
            result[n] = 0.0;
        }
        return;
    }

    // Compute full Boys
    boys_evaluate_array_device(n_max, T, d_boys_coeffs, result);

    // Compute erf part and subtract
    const double omega2 = omega * omega;
    const double omega2_ratio = omega2 / (omega2 + 1.0);
    const double T_eff = T * omega2_ratio;
    const double sqrt_ratio = sqrt(omega2_ratio);

    // Temporary array for erf values (using registers for small n_max)
    double erf_temp[BOYS_MAX_N_DEVICE + 1];
    boys_evaluate_array_device(n_max, T_eff, d_boys_coeffs, erf_temp);

    double ratio_power = 1.0;
    for (int n = 0; n <= n_max; ++n) {
        result[n] -= sqrt_ratio * ratio_power * erf_temp[n];
        ratio_power *= omega2_ratio;
    }
}

}  // namespace libaccint::device::math

#endif  // LIBACCINT_USE_CUDA
