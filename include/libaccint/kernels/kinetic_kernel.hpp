// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file kinetic_kernel.hpp
/// @brief Kinetic energy integral kernel (T) for Cartesian Gaussian basis functions
///
/// Implements the kinetic energy integral T_uv = -1/2 <u|nabla^2|v> using the
/// relationship between second derivatives of Gaussians and overlap integrals
/// with shifted angular momentum indices. The 1D overlap recursion tables are
/// built with extended range (Lb+2) to accommodate the shifted indices needed
/// for the kinetic energy formula.

#include <libaccint/basis/primitive_pair_data.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>

namespace libaccint::kernels {

/// @brief Compute kinetic energy integrals for a pair of shells
///
/// Computes the kinetic energy matrix elements T(a,b) for all Cartesian
/// components of two shells:
///
///   T(a,b) = -1/2 <a|nabla^2|b>
///
/// The kinetic energy integral is decomposed into directional contributions:
///
///   T(a,b) = T_x * S_y * S_z + S_x * T_y * S_z + S_x * S_y * T_z
///
/// where S_d is the 1D overlap integral and T_d is the 1D kinetic integral
/// computed from overlap integrals with shifted angular momentum:
///
///   T_d(i,j) = -1/2 * [4*beta^2 * S_d(i,j+2) - 2*beta*(2j+1) * S_d(i,j)
///                       + j*(j-1) * S_d(i,j-2)]
///
/// The Shell's stored coefficients are normalized for the (L,0,0) component.
/// A normalization correction factor is applied for other Cartesian components.
///
/// @param shell_a First shell (bra)
/// @param shell_b Second shell (ket)
/// @param buffer Output buffer (resized and filled with kinetic energy integrals)
void compute_kinetic(const Shell& shell_a, const Shell& shell_b,
                     KineticBuffer& buffer);

/// @brief Compute kinetic energy integrals using pre-computed primitive pair data
///
/// @param shell_a First shell (bra)
/// @param shell_b Second shell (ket)
/// @param pair_data Pre-computed Gaussian product data
/// @param shell_i Index of shell_a in its ShellSet
/// @param shell_j Index of shell_b in its ShellSet
/// @param buffer Output buffer (resized and filled with kinetic energy integrals)
void compute_kinetic(const Shell& shell_a, const Shell& shell_b,
                     const PrimitivePairData& pair_data,
                     Size shell_i, Size shell_j,
                     KineticBuffer& buffer);

}  // namespace libaccint::kernels
