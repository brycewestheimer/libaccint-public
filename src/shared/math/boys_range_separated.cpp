// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file boys_range_separated.cpp
/// @brief Implementation of range-separated Boys function for erf/erfc operators

#include <libaccint/math/boys_range_separated.hpp>
#include <libaccint/math/boys_function.hpp>
#include <cmath>
#include <cassert>

namespace libaccint::math {

// ============================================================================
// erf-Attenuated Boys Function Implementation
// ============================================================================

double boys_erf(int n, double T, double omega) {
    assert(n >= 0 && n <= BOYS_MAX_N && "boys_erf: n out of range");
    assert(T >= 0.0 && "boys_erf: T must be non-negative");

    // Handle limiting cases
    if (omega <= 0.0) {
        // omega -> 0: erf(omega * r) -> 0, so F_n^{erf} -> 0
        return 0.0;
    }

    if (omega > 1000.0) {
        // omega -> infinity: F_n^{erf} -> F_n(T)
        return boys_evaluate(n, T);
    }

    // General case: compute effective T and scaling
    const double omega2 = omega * omega;
    const double omega2_ratio = omega2 / (omega2 + 1.0);

    // T_eff = T * omega^2 / (omega^2 + 1)
    const double T_eff = T * omega2_ratio;

    // Scaling factor: (omega^2/(omega^2+1))^{n+1/2}
    const double scale = std::pow(omega2_ratio, n + 0.5);

    // Compute F_n(T_eff) and scale
    return scale * boys_evaluate(n, T_eff);
}

void boys_erf_array(int n_max, double T, double omega, double* result) {
    assert(n_max >= 0 && n_max <= BOYS_MAX_N && "boys_erf_array: n_max out of range");
    assert(T >= 0.0 && "boys_erf_array: T must be non-negative");
    assert(result != nullptr && "boys_erf_array: result is null");

    // Handle limiting cases
    if (omega <= 0.0) {
        // omega -> 0: all values are zero
        for (int n = 0; n <= n_max; ++n) {
            result[n] = 0.0;
        }
        return;
    }

    if (omega > 1000.0) {
        // omega -> infinity: F_n^{erf} = F_n(T)
        boys_evaluate_array(n_max, T, result);
        return;
    }

    // General case
    const double omega2 = omega * omega;
    const double omega2_ratio = omega2 / (omega2 + 1.0);
    const double T_eff = T * omega2_ratio;

    // Compute standard Boys function at T_eff
    boys_evaluate_array(n_max, T_eff, result);

    // Apply scaling factors: (omega2_ratio)^{n+1/2} = sqrt(omega2_ratio) * omega2_ratio^n
    const double sqrt_ratio = std::sqrt(omega2_ratio);
    double ratio_power = 1.0;  // omega2_ratio^n, starting at n=0

    for (int n = 0; n <= n_max; ++n) {
        result[n] *= sqrt_ratio * ratio_power;
        ratio_power *= omega2_ratio;
    }
}

// ============================================================================
// erfc-Attenuated Boys Function Implementation
// ============================================================================

double boys_erfc(int n, double T, double omega) {
    assert(n >= 0 && n <= BOYS_MAX_N && "boys_erfc: n out of range");
    assert(T >= 0.0 && "boys_erfc: T must be non-negative");

    // Handle limiting cases
    if (omega <= 0.0) {
        // omega -> 0: erfc(omega * r) -> 1, so F_n^{erfc} -> F_n(T)
        return boys_evaluate(n, T);
    }

    if (omega > 1000.0) {
        // omega -> infinity: erfc(omega * r) -> 0, so F_n^{erfc} -> 0
        return 0.0;
    }

    // Use identity: F_n^{erfc} = F_n(T) - F_n^{erf}(T, omega)
    const double F_n_full = boys_evaluate(n, T);
    const double F_n_erf = boys_erf(n, T, omega);

    return F_n_full - F_n_erf;
}

void boys_erfc_array(int n_max, double T, double omega, double* result) {
    assert(n_max >= 0 && n_max <= BOYS_MAX_N && "boys_erfc_array: n_max out of range");
    assert(T >= 0.0 && "boys_erfc_array: T must be non-negative");
    assert(result != nullptr && "boys_erfc_array: result is null");

    // Handle limiting cases
    if (omega <= 0.0) {
        // omega -> 0: F_n^{erfc} = F_n(T)
        boys_evaluate_array(n_max, T, result);
        return;
    }

    if (omega > 1000.0) {
        // omega -> infinity: all values are zero
        for (int n = 0; n <= n_max; ++n) {
            result[n] = 0.0;
        }
        return;
    }

    // Compute full Boys function
    boys_evaluate_array(n_max, T, result);

    // Compute erf-attenuated Boys function
    const double omega2 = omega * omega;
    const double omega2_ratio = omega2 / (omega2 + 1.0);
    const double T_eff = T * omega2_ratio;

    // Temporary array for erf values (use stack for small n_max)
    double erf_values[BOYS_MAX_N + 1];
    boys_evaluate_array(n_max, T_eff, erf_values);

    // Subtract scaled erf values: F_n^{erfc} = F_n - scale * F_n(T_eff)
    const double sqrt_ratio = std::sqrt(omega2_ratio);
    double ratio_power = 1.0;

    for (int n = 0; n <= n_max; ++n) {
        result[n] -= sqrt_ratio * ratio_power * erf_values[n];
        ratio_power *= omega2_ratio;
    }
}

// ============================================================================
// Batch Evaluation
// ============================================================================

void boys_erf_batch(int n_max, const double* T_array, int n_values,
                    double omega, double* result) {
    assert(n_max >= 0 && n_max <= BOYS_MAX_N && "boys_erf_batch: n_max out of range");
    assert(T_array != nullptr && "boys_erf_batch: T_array is null");
    assert(result != nullptr && "boys_erf_batch: result is null");
    assert(n_values >= 0 && "boys_erf_batch: n_values must be non-negative");

    const int stride = n_max + 1;

    // Handle limiting cases for all values at once
    if (omega <= 0.0) {
        for (int i = 0; i < n_values * stride; ++i) {
            result[i] = 0.0;
        }
        return;
    }

    if (omega > 1000.0) {
        boys_evaluate_batch(n_max, T_array, n_values, result);
        return;
    }

    // General case: precompute omega-dependent values
    const double omega2 = omega * omega;
    const double omega2_ratio = omega2 / (omega2 + 1.0);
    const double sqrt_ratio = std::sqrt(omega2_ratio);

    // Precompute scaling factors for each n
    double scale_factors[BOYS_MAX_N + 1];
    double ratio_power = 1.0;
    for (int n = 0; n <= n_max; ++n) {
        scale_factors[n] = sqrt_ratio * ratio_power;
        ratio_power *= omega2_ratio;
    }

    // Process each T value
    for (int i = 0; i < n_values; ++i) {
        const double T = T_array[i];
        const double T_eff = T * omega2_ratio;
        double* output = result + i * stride;

        // Compute Boys function at T_eff
        boys_evaluate_array(n_max, T_eff, output);

        // Apply scaling
        for (int n = 0; n <= n_max; ++n) {
            output[n] *= scale_factors[n];
        }
    }
}

void boys_erfc_batch(int n_max, const double* T_array, int n_values,
                     double omega, double* result) {
    assert(n_max >= 0 && n_max <= BOYS_MAX_N && "boys_erfc_batch: n_max out of range");
    assert(T_array != nullptr && "boys_erfc_batch: T_array is null");
    assert(result != nullptr && "boys_erfc_batch: result is null");
    assert(n_values >= 0 && "boys_erfc_batch: n_values must be non-negative");

    const int stride = n_max + 1;

    // Handle limiting cases
    if (omega <= 0.0) {
        boys_evaluate_batch(n_max, T_array, n_values, result);
        return;
    }

    if (omega > 1000.0) {
        for (int i = 0; i < n_values * stride; ++i) {
            result[i] = 0.0;
        }
        return;
    }

    // General case: compute full - erf
    const double omega2 = omega * omega;
    const double omega2_ratio = omega2 / (omega2 + 1.0);
    const double sqrt_ratio = std::sqrt(omega2_ratio);

    // Precompute scaling factors
    double scale_factors[BOYS_MAX_N + 1];
    double ratio_power = 1.0;
    for (int n = 0; n <= n_max; ++n) {
        scale_factors[n] = sqrt_ratio * ratio_power;
        ratio_power *= omega2_ratio;
    }

    // Temporary storage for erf values
    double erf_temp[BOYS_MAX_N + 1];

    for (int i = 0; i < n_values; ++i) {
        const double T = T_array[i];
        const double T_eff = T * omega2_ratio;
        double* output = result + i * stride;

        // Compute full Boys function
        boys_evaluate_array(n_max, T, output);

        // Compute erf-attenuated Boys at T_eff
        boys_evaluate_array(n_max, T_eff, erf_temp);

        // Subtract: F_n^{erfc} = F_n - scale * F_n(T_eff)
        for (int n = 0; n <= n_max; ++n) {
            output[n] -= scale_factors[n] * erf_temp[n];
        }
    }
}

}  // namespace libaccint::math
