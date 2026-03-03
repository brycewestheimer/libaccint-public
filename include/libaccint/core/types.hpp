// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file types.hpp
/// @brief Fundamental numeric and index types for LibAccInt

#include <cstdint>
#include <cstddef>
#include <array>
#include <span>
#include <vector>
#include <optional>
#include <concepts>

namespace libaccint {

// ============================================================================
// Numeric Types
// ============================================================================

/// Default floating-point type for integral computation
using Real = double;

/// Single-precision floating-point (for certain GPU operations)
using Float = float;

/// Integer type for shell/function indices
using Index = std::int64_t;

/// Integer type for sizes and counts
using Size = std::size_t;

// ============================================================================
// Constants
// ============================================================================

/// Stable maximum supported angular momentum this cycle (G-functions, l=4)
constexpr int MAX_ANGULAR_MOMENTUM = 4;

/// Maximum number of Rys quadrature roots needed
constexpr int MAX_RYS_ROOTS = 15;

// ============================================================================
// Angular Momentum
// ============================================================================

/// Angular momentum quantum number
enum class AngularMomentum : int {
    S = 0,  ///< l = 0, 1 function
    P = 1,  ///< l = 1, 3 functions
    D = 2,  ///< l = 2, 6 functions (Cartesian) or 5 (spherical)
    F = 3,  ///< l = 3, 10 functions (Cartesian) or 7 (spherical)
    G = 4,  ///< l = 4, 15 functions (Cartesian) or 9 (spherical)
    H = 5,  ///< l = 5, reserved for future cycles (not in stable contract)
    I = 6,  ///< l = 6, reserved for future cycles (not in stable contract)
};

/// Convert AngularMomentum enum to integer
[[nodiscard]] constexpr int to_int(AngularMomentum am) noexcept {
    return static_cast<int>(am);
}

/// Number of Cartesian basis functions for given angular momentum
[[nodiscard]] constexpr int n_cartesian(int l) noexcept {
    return (l + 1) * (l + 2) / 2;
}

/// Number of spherical basis functions for given angular momentum
[[nodiscard]] constexpr int n_spherical(int l) noexcept {
    return 2 * l + 1;
}

/// Number of Cartesian basis functions for a pair of shells
[[nodiscard]] constexpr int n_cartesian_pair(int la, int lb) noexcept {
    return n_cartesian(la) * n_cartesian(lb);
}

/// Number of basis functions for given angular momentum and spherical flag
[[nodiscard]] constexpr int n_functions(int l, bool spherical) noexcept {
    return spherical ? n_spherical(l) : n_cartesian(l);
}

/// Number of spherical basis functions for a pair of shells
[[nodiscard]] constexpr int n_spherical_pair(int la, int lb) noexcept {
    return n_spherical(la) * n_spherical(lb);
}

// ============================================================================
// Derivative Orders
// ============================================================================

/// Derivative order for integral computation
enum class DerivativeOrder : std::int8_t {
    Energy = 0,    ///< Energy integrals (no derivatives)
    Gradient = 1,  ///< First derivatives (gradients)
    Hessian = 2,   ///< Second derivatives (Hessians)
};

/// Number of derivative components for N centers at given order
template<int DerivOrder, int NCenters>
[[nodiscard]] consteval int n_derivative_components() {
    if constexpr (DerivOrder == 0) {
        return 1;
    } else if constexpr (DerivOrder == 1) {
        return 3 * NCenters;  // x, y, z for each center
    } else if constexpr (DerivOrder == 2) {
        constexpr int n_first = 3 * NCenters;
        return n_first * (n_first + 1) / 2;  // Upper triangle
    }
    return 0;  // Should not reach
}

// ============================================================================
// Integral Shell Types
// ============================================================================

/// Angular momentum tuple for shell quartet (bra_a, bra_b | ket_c, ket_d)
using AMQuartet = std::array<int, 4>;

/// Angular momentum tuple for shell pair (a | b)
using AMPair = std::array<int, 2>;

/// Angular momentum tuple for shell triplet (a | b c) - density fitting
using AMTriplet = std::array<int, 3>;

// ============================================================================
// Concepts
// ============================================================================

/// Concept for numeric types usable in integral computation
template<typename T>
concept Numeric = std::is_floating_point_v<T>;

// ============================================================================
// 3D Point
// ============================================================================

/// 3D Cartesian point/vector
struct Point3D {
    Real x{0.0};
    Real y{0.0};
    Real z{0.0};

    constexpr Point3D() = default;
    constexpr Point3D(Real x_, Real y_, Real z_) : x(x_), y(y_), z(z_) {}

    [[nodiscard]] constexpr Real& operator[](int i) noexcept {
        return i == 0 ? x : (i == 1 ? y : z);
    }

    [[nodiscard]] constexpr Real operator[](int i) const noexcept {
        return i == 0 ? x : (i == 1 ? y : z);
    }

    [[nodiscard]] constexpr Real distance_squared(const Point3D& other) const noexcept {
        Real dx = x - other.x;
        Real dy = y - other.y;
        Real dz = z - other.z;
        return dx*dx + dy*dy + dz*dz;
    }
};

}  // namespace libaccint
