// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file eri_kernel.hpp
/// @brief Electron repulsion integral (ERI) kernel for Cartesian Gaussian basis functions
///
/// Implements the four-center electron repulsion integral (ab|cd) using Rys quadrature:
///
///   (ab|cd) = sum_ijkl c_i c_j c_k c_l * [ij|kl]
///
/// where [ij|kl] is the primitive integral:
///
///   [ij|kl] = prefactor * sum_r w_r * Ix(r) * Iy(r) * Iz(r)
///
/// The prefactor is:
///   2 * pi^(5/2) / (zeta * eta * sqrt(zeta + eta)) * K_AB * K_CD
///
/// Rys quadrature points:
///   n_roots = ceil((La + Lb + Lc + Ld + 1) / 2) = (La+Lb+Lc+Ld)/2 + 1
///
/// The 2D recursion builds I_x, I_y, I_z tables at each Rys root, then
/// contracts over Cartesian components to produce the full integral tensor.

#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/operators/operator_types.hpp>

namespace libaccint::kernels {

/// @brief Compute contracted electron repulsion integrals for a shell quartet
///
/// Computes all (na*nb*nc*nd) integrals for a quartet of shells using
/// Rys quadrature with 2D recursion. The buffer is resized and zeroed
/// before computation.
///
/// @param shell_a First bra shell
/// @param shell_b Second bra shell
/// @param shell_c First ket shell
/// @param shell_d Second ket shell
/// @param buffer Output buffer (resized and filled with ERIs)
void compute_eri(const Shell& shell_a, const Shell& shell_b,
                 const Shell& shell_c, const Shell& shell_d,
                 TwoElectronBuffer<0>& buffer);

}  // namespace libaccint::kernels
