// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file derivative_utils.hpp
/// @brief Utility functions for derivative integral computations
///
/// Provides index mapping utilities for accessing derivative components
/// in OneElectronBuffer<1> and TwoElectronBuffer<1>.

#include <libaccint/core/types.hpp>
#include <array>
#include <cassert>
#include <cmath>

namespace libaccint {

// ============================================================================
// Derivative Component Indexing
// ============================================================================

/// @brief Cartesian direction enumeration
enum class CartDir : int {
    X = 0,
    Y = 1,
    Z = 2
};

/// @brief Convert CartDir enum to integer
[[nodiscard]] constexpr int to_int(CartDir dir) noexcept {
    return static_cast<int>(dir);
}

/// @brief Compute derivative component index for one-electron integrals
///
/// For OneElectronBuffer<1> with 2 centers (A, B):
///   - Components 0-2: dA_x, dA_y, dA_z
///   - Components 3-5: dB_x, dB_y, dB_z
///
/// @param center Center index (0 = A, 1 = B)
/// @param cart_dir Cartesian direction (0 = x, 1 = y, 2 = z)
/// @return Linear derivative component index
[[nodiscard]] constexpr int deriv_component_1e(int center, int cart_dir) noexcept {
    assert(center >= 0 && center < 2);
    assert(cart_dir >= 0 && cart_dir < 3);
    return center * 3 + cart_dir;
}

/// @brief Compute derivative component index for one-electron integrals (enum version)
[[nodiscard]] constexpr int deriv_component_1e(int center, CartDir cart_dir) noexcept {
    return deriv_component_1e(center, to_int(cart_dir));
}

/// @brief Compute derivative component index for two-electron integrals
///
/// For TwoElectronBuffer<1> with 4 centers (A, B, C, D):
///   - Components 0-2:  dA_x, dA_y, dA_z
///   - Components 3-5:  dB_x, dB_y, dB_z
///   - Components 6-8:  dC_x, dC_y, dC_z
///   - Components 9-11: dD_x, dD_y, dD_z
///
/// Note: Due to translational invariance, only 9 components are independent.
/// The 4th center derivatives can be computed as: dD = -(dA + dB + dC)
///
/// @param center Center index (0 = A, 1 = B, 2 = C, 3 = D)
/// @param cart_dir Cartesian direction (0 = x, 1 = y, 2 = z)
/// @return Linear derivative component index
[[nodiscard]] constexpr int deriv_component_2e(int center, int cart_dir) noexcept {
    assert(center >= 0 && center < 4);
    assert(cart_dir >= 0 && cart_dir < 3);
    return center * 3 + cart_dir;
}

/// @brief Compute derivative component index for two-electron integrals (enum version)
[[nodiscard]] constexpr int deriv_component_2e(int center, CartDir cart_dir) noexcept {
    return deriv_component_2e(center, to_int(cart_dir));
}

// ============================================================================
// Derivative Component Count Constants
// ============================================================================

/// @brief Number of derivative components for one-electron integrals (2 centers x 3 dirs)
inline constexpr int N_DERIV_1E = 6;

/// @brief Number of derivative components for two-electron integrals (4 centers x 3 dirs)
inline constexpr int N_DERIV_2E = 12;

/// @brief Number of independent derivative components for two-electron integrals
/// (3 centers x 3 dirs, due to translational invariance)
inline constexpr int N_DERIV_2E_INDEPENDENT = 9;

// ============================================================================
// Translational Invariance Utilities
// ============================================================================

/// @brief Compute the 4th center derivative from the first 3 using translational invariance
///
/// For any molecular integral, the sum of all nuclear derivatives is zero:
///   d/dR_A + d/dR_B + ... + d/dR_N = 0
///
/// This allows computing the last derivative as minus the sum of the others:
///   d/dR_D = -(d/dR_A + d/dR_B + d/dR_C)
///
/// @tparam T Numeric type (Real or Float)
/// @param dA Derivative with respect to center A (array of 3)
/// @param dB Derivative with respect to center B (array of 3)
/// @param dC Derivative with respect to center C (array of 3)
/// @return Derivative with respect to center D (computed via translational invariance)
template<typename T>
[[nodiscard]] constexpr std::array<T, 3> compute_d_by_translational_invariance(
    const std::array<T, 3>& dA,
    const std::array<T, 3>& dB,
    const std::array<T, 3>& dC) noexcept {
    return {
        -(dA[0] + dB[0] + dC[0]),
        -(dA[1] + dB[1] + dC[1]),
        -(dA[2] + dB[2] + dC[2])
    };
}

/// @brief Verify translational invariance for a derivative buffer
///
/// Checks that the sum of all center derivatives is approximately zero.
/// Useful for debugging and validation.
///
/// @tparam T Numeric type
/// @param derivs Array of derivative components (3*N_centers values)
/// @param n_centers Number of atomic centers
/// @param tolerance Tolerance for sum being zero
/// @return True if translational invariance is satisfied
template<typename T>
[[nodiscard]] bool verify_translational_invariance(
    const T* derivs, int n_centers, T tolerance = T{1e-10}) noexcept {
    T sum_x = T{0}, sum_y = T{0}, sum_z = T{0};
    for (int c = 0; c < n_centers; ++c) {
        sum_x += derivs[c * 3 + 0];
        sum_y += derivs[c * 3 + 1];
        sum_z += derivs[c * 3 + 2];
    }
    return (std::abs(sum_x) < tolerance &&
            std::abs(sum_y) < tolerance &&
            std::abs(sum_z) < tolerance);
}

// ============================================================================
// Derivative Recursion Helpers
// ============================================================================

/// @brief Compute the derivative recursion factor for incrementing angular momentum
///
/// For a Gaussian function with exponent alpha and angular momentum l:
///   d/dx [x^l * exp(-alpha*r^2)] = 2*alpha * x^(l+1) * exp(-alpha*r^2)
///                                  - l * x^(l-1) * exp(-alpha*r^2)
///
/// This means:
///   d/dA_x [a|b] = 2*alpha_a * [a+1_x|b] - a_x * [a-1_x|b]
///
/// where a+1_x means increment the x-component of angular momentum on center A.
///
/// @param alpha Primitive exponent
/// @param l Angular momentum component in the differentiation direction
/// @return Pair of (coefficient for +1, coefficient for -1) angular momentum terms
template<typename T>
[[nodiscard]] constexpr std::pair<T, T> derivative_recursion_coefficients(
    T alpha, int l) noexcept {
    return {T{2} * alpha, static_cast<T>(l)};
}

}  // namespace libaccint
