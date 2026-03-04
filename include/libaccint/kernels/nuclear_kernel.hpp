// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file nuclear_kernel.hpp
/// @brief Nuclear attraction integral kernel (V) for Cartesian Gaussian basis functions
///
/// Implements the nuclear attraction integral V_uv = <u| sum_C -Z_C/|r-R_C| |v>
/// using Rys quadrature. For each nucleus C with charge Z_C at position R_C, the
/// integral is evaluated via Rys roots and weights, with the 1D recursion tables
/// built using modified Obara-Saika recursion with effective displacements and
/// a root-dependent half-inverse exponent.
///
/// The number of Rys quadrature points for angular momenta La, Lb is:
///   n_roots = (La + Lb) / 2 + 1
///
/// Contributions from all nuclei are summed to form the total nuclear attraction
/// matrix for the shell pair.

#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/operators/operator_types.hpp>

namespace libaccint::kernels {

/// @brief Compute nuclear attraction integrals for a pair of shells
///
/// Computes the nuclear attraction matrix elements V(a,b) for all Cartesian
/// components of two shells using Rys quadrature:
///
///   V(a,b) = sum_C -Z_C * sum_ij c_i c_j * (2*pi/zeta) * K_AB
///            * sum_r w_r * Ix(lx_a, lx_b; t_r) * Iy(ly_a, ly_b; t_r) * Iz(lz_a, lz_b; t_r)
///
/// where:
///   - c_i, c_j are normalized contraction coefficients
///   - K_AB = exp(-mu * |A-B|^2) is the Gaussian product prefactor
///   - w_r are Rys quadrature weights, t_r are Rys roots
///   - Ix, Iy, Iz are 1D recursion integrals with root-dependent effective displacements
///
/// The Shell's stored coefficients are normalized for the (L,0,0) component.
/// A normalization correction factor is applied for other Cartesian components.
///
/// @param shell_a First shell (bra)
/// @param shell_b Second shell (ket)
/// @param charges Point charges (nuclear positions and charges)
/// @param buffer Output buffer (resized and filled with nuclear attraction integrals)
void compute_nuclear(const Shell& shell_a, const Shell& shell_b,
                     const PointChargeParams& charges,
                     NuclearBuffer& buffer);

}  // namespace libaccint::kernels
