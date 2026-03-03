// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file eri_erf_coulomb_kernel.hpp
/// @brief erf-attenuated Coulomb electron repulsion integral kernel
///
/// Implements the long-range erf-attenuated ERI:
///   (ab|erf(omega*r12)/r12|cd)
///
/// Used in range-separated hybrid functionals like CAM-B3LYP, omega-B97X, LC-BLYP.
/// The implementation modifies the standard Rys quadrature approach by:
/// 1. Computing T_eff = T * omega^2 / (omega^2 + 1) instead of T
/// 2. Applying scaling factor (omega^2/(omega^2+1))^{n+1/2} to Boys function
///
/// Key identity: erf + erfc = full Coulomb (1/r12)

#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/operators/operator_types.hpp>

namespace libaccint::kernels {

/// @brief Compute erf-attenuated Coulomb ERIs for a shell quartet
///
/// Computes (ab|erf(omega*r12)/r12|cd) using modified Rys quadrature.
/// The buffer is resized and zeroed before computation.
///
/// @param shell_a First bra shell
/// @param shell_b Second bra shell
/// @param shell_c First ket shell
/// @param shell_d Second ket shell
/// @param omega Range-separation parameter (in atomic units)
/// @param buffer Output buffer (resized and filled with ERIs)
void compute_eri_erf_coulomb(const Shell& shell_a, const Shell& shell_b,
                              const Shell& shell_c, const Shell& shell_d,
                              Real omega,
                              TwoElectronBuffer<0>& buffer);

/// @brief Compute erf-attenuated Coulomb ERIs using operator parameters
///
/// Convenience overload that extracts omega from RangeSeparatedParams.
///
/// @param shell_a First bra shell
/// @param shell_b Second bra shell
/// @param shell_c First ket shell
/// @param shell_d Second ket shell
/// @param params Range-separated operator parameters
/// @param buffer Output buffer
void compute_eri_erf_coulomb(const Shell& shell_a, const Shell& shell_b,
                              const Shell& shell_c, const Shell& shell_d,
                              const RangeSeparatedParams& params,
                              TwoElectronBuffer<0>& buffer);

}  // namespace libaccint::kernels
