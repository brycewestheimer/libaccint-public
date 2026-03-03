// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file multipole_kernel.hpp
/// @brief CPU kernels for electric multipole moment integrals (dipole, quadrupole, octupole)
///
/// Implements multipole moment integrals <μ|(r-O)^n|ν> using extended Obara-Saika
/// recursion. Dipole (n=1, 3 components), quadrupole (n=2, 6 components),
/// and octupole (n=3, 10 components) are supported.

#include <libaccint/basis/shell.hpp>
#include <libaccint/engine/multi_component_buffer.hpp>

#include <array>

namespace libaccint::kernels {

/// @brief Compute electric dipole integrals <μ|(r-O)|ν> for a shell pair
///
/// Produces 3 component matrices (x, y, z) stored in a MultiComponentBuffer.
/// Each component matrix is symmetric: <μ|r_α|ν> = <ν|r_α|μ>.
///
/// @param shell_a First shell (bra)
/// @param shell_b Second shell (ket)
/// @param origin Gauge origin for origin-dependent integrals
/// @param buffer Output buffer (resized and filled with 3 component matrices)
void compute_dipole(const Shell& shell_a, const Shell& shell_b,
                    const std::array<Real, 3>& origin,
                    MultiComponentBuffer& buffer);

/// @brief Compute electric quadrupole integrals <μ|(r-O)_α(r-O)_β|ν> for a shell pair
///
/// Produces 6 component matrices (xx, xy, xz, yy, yz, zz).
/// Each component matrix is symmetric.
///
/// @param shell_a First shell (bra)
/// @param shell_b Second shell (ket)
/// @param origin Gauge origin
/// @param buffer Output buffer (resized and filled with 6 component matrices)
void compute_quadrupole(const Shell& shell_a, const Shell& shell_b,
                        const std::array<Real, 3>& origin,
                        MultiComponentBuffer& buffer);

/// @brief Compute electric octupole integrals <μ|(r-O)_α(r-O)_β(r-O)_γ|ν> for a shell pair
///
/// Produces 10 component matrices (xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz).
/// Each component matrix is symmetric.
///
/// @param shell_a First shell (bra)
/// @param shell_b Second shell (ket)
/// @param origin Gauge origin
/// @param buffer Output buffer (resized and filled with 10 component matrices)
void compute_octupole(const Shell& shell_a, const Shell& shell_b,
                      const std::array<Real, 3>& origin,
                      MultiComponentBuffer& buffer);

}  // namespace libaccint::kernels
