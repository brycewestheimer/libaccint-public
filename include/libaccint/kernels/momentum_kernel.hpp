// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file momentum_kernel.hpp
/// @brief CPU kernels for linear and angular momentum integrals
///
/// Linear momentum: <μ|-i∇|ν> (3 anti-Hermitian components, stored as real derivative integrals)
/// Angular momentum: <μ|r×(-i∇)|ν> (3 anti-Hermitian components)

#include <libaccint/basis/shell.hpp>
#include <libaccint/engine/multi_component_buffer.hpp>

#include <array>

namespace libaccint::kernels {

/// @brief Compute linear momentum integrals <μ|d/dr_α|ν> for a shell pair
///
/// Computes the real derivative integrals (the -i prefactor is factored out).
/// Each component matrix is anti-symmetric: <μ|d/dx|ν> = -<ν|d/dx|μ>.
///
/// The derivative integral is: <a|d/dx|b> = 2β<a|b+1_x> - b_x<a|b-1_x>
///
/// @param shell_a First shell (bra)
/// @param shell_b Second shell (ket)
/// @param buffer Output buffer (3 anti-symmetric component matrices)
void compute_linear_momentum(const Shell& shell_a, const Shell& shell_b,
                             MultiComponentBuffer& buffer);

/// @brief Compute angular momentum integrals <μ|(r-O)×(-i∇)|ν> for a shell pair
///
/// Computes the real part of angular momentum integrals (the -i prefactor is factored out).
/// L_x = y*d/dz - z*d/dy, L_y = z*d/dx - x*d/dz, L_z = x*d/dy - y*d/dx
///
/// Each component matrix is anti-symmetric.
///
/// @param shell_a First shell (bra)
/// @param shell_b Second shell (ket)
/// @param origin Gauge origin for angular momentum
/// @param buffer Output buffer (3 anti-symmetric component matrices: Lx, Ly, Lz)
void compute_angular_momentum(const Shell& shell_a, const Shell& shell_b,
                              const std::array<Real, 3>& origin,
                              MultiComponentBuffer& buffer);

}  // namespace libaccint::kernels
