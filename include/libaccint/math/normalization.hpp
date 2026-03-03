// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file normalization.hpp
/// @brief Normalization factors for Cartesian Gaussian basis functions

#include <libaccint/core/types.hpp>

namespace libaccint::math {

/**
 * @brief Compute double factorial (2n-1)!!
 *
 * The double factorial (2n-1)!! is the product of odd numbers from 1 to 2n-1:
 *   (2n-1)!! = (2n-1) * (2n-3) * ... * 3 * 1
 *
 * By convention, (-1)!! = 1 (when n <= 0).
 *
 * Examples:
 *   n=0: (-1)!! = 1
 *   n=1: (1)!! = 1
 *   n=2: (3)!! = 3
 *   n=3: (5)!! = 15
 *   n=4: (7)!! = 105
 *   n=5: (9)!! = 945
 *   n=6: (11)!! = 10395
 *
 * @param n Index for (2n-1)!!
 * @return The value (2n-1)!! computed using constexpr lookup table
 */
[[nodiscard]] constexpr int double_factorial_odd(int n) noexcept {
    if (n <= 0) return 1;
    // Lookup table for n=1..6 (covers MAX_ANGULAR_MOMENTUM=6)
    constexpr int table[] = {1, 1, 3, 15, 105, 945, 10395};
    if (n <= 6) return table[n];
    // Fallback for n > 6
    int result = 1;
    for (int i = 1; i <= n; ++i) result *= (2 * i - 1);
    return result;
}

/**
 * @brief Compute normalization factor for Cartesian Gaussian N_ijk(α)
 *
 * For a Cartesian Gaussian basis function:
 *   μ(r) = N_ijk(α) * (x-A_x)^i (y-A_y)^j (z-A_z)^k * exp(-α|r-A|²)
 *
 * The normalization constant ensures that the self-overlap integral equals 1:
 *   ∫ μ(r)² dr = 1
 *
 * The formula is:
 *   N_ijk(α) = sqrt( (2α/π)^(3/2) * (4α)^L / ((2i-1)!! * (2j-1)!! * (2k-1)!!) )
 *
 * where:
 *   L = i + j + k (total angular momentum)
 *   (2n-1)!! denotes the double factorial
 *
 * @param alpha Gaussian exponent α (must be positive)
 * @param i Power of (x-A_x) component
 * @param j Power of (y-A_y) component
 * @param k Power of (z-A_z) component
 * @return The normalization factor N_ijk(α)
 */
[[nodiscard]] double normalization_factor(Real alpha, int i, int j, int k) noexcept;

/**
 * @brief Compute normalization factor for (L,0,0) component
 *
 * Convenience function for computing the normalization factor for a Gaussian
 * of the form: x^L * exp(-α*r²) (i.e., j=0, k=0).
 *
 * Equivalent to: normalization_factor(alpha, L, 0, 0)
 *
 * @param alpha Gaussian exponent α (must be positive)
 * @param L Angular momentum (power of x component)
 * @return The normalization factor N_L(α)
 */
[[nodiscard]] double normalization_factor(Real alpha, int L) noexcept;

}  // namespace libaccint::math
