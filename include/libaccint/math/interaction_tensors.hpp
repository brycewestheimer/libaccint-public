// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file interaction_tensors.hpp
/// @brief Solid harmonic interaction tensors for distributed multipole analysis
///
/// Provides interaction tensor functions T^(l)_m(R) for computing the
/// electrostatic potential from distributed multipole expansions.
/// Uses real solid harmonics (regular and irregular) and their Cartesian
/// decomposition for efficient integral evaluation.

#include <libaccint/core/types.hpp>

#include <array>
#include <vector>

namespace libaccint::math {

/// @brief Compute regular solid harmonic R_lm(r) = r^l * Y_lm(θ,φ)
///
/// Real solid harmonics in Cartesian form for l = 0, 1, 2.
/// Used in the multipole expansion of the electrostatic potential.
///
/// @param l Rank (0=monopole, 1=dipole, 2=quadrupole)
/// @param m Component index within rank
/// @param x, y, z Cartesian coordinates
/// @return Value of R_lm(x, y, z)
[[nodiscard]] Real regular_solid_harmonic(int l, int m, Real x, Real y, Real z);

/// @brief Compute irregular solid harmonic I_lm(r) = R_lm(r) / r^(2l+1)
///
/// Used for the far-field contribution of multipole interactions.
///
/// @param l Rank
/// @param m Component index within rank
/// @param x, y, z Cartesian coordinates
/// @return Value of I_lm(x, y, z)
[[nodiscard]] Real irregular_solid_harmonic(int l, int m, Real x, Real y, Real z);

/// @brief Compute Cartesian interaction tensor T^(n)_{i1...in}(R)
///
/// The interaction tensor is the n-th derivative of 1/|R|:
///   T^(0) = 1/R
///   T^(1)_i = -R_i / R^3
///   T^(2)_{ij} = (3*R_i*R_j - R^2 * delta_ij) / R^5
///
/// @param rank Tensor rank (0, 1, or 2)
/// @param R Displacement vector (3 elements)
/// @return Flattened tensor components (1, 3, or 6 elements for ranks 0, 1, 2)
[[nodiscard]] std::vector<Real> interaction_tensor(int rank,
                                                   const std::array<Real, 3>& R);

/// @brief Number of unique symmetric tensor components for given rank
/// @param rank Tensor rank
/// @return Number of unique components: 1 (rank 0), 3 (rank 1), 6 (rank 2)
[[nodiscard]] constexpr int n_tensor_components(int rank) noexcept {
    return (rank + 1) * (rank + 2) / 2;
}

}  // namespace libaccint::math
