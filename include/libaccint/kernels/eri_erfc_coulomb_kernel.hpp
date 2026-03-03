// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file eri_erfc_coulomb_kernel.hpp
/// @brief erfc-attenuated Coulomb electron repulsion integral kernel
///
/// Implements the short-range erfc-attenuated ERI:
///   (ab|erfc(omega*r12)/r12|cd)
///
/// Used in range-separated hybrid functionals for the short-range component.
/// Computed via the identity: erfc = full - erf
///   (ab|erfc(omega*r12)/r12|cd) = (ab|1/r12|cd) - (ab|erf(omega*r12)/r12|cd)
///
/// Key identity: erf + erfc = full Coulomb (1/r12)

#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/operators/operator_types.hpp>

namespace libaccint::kernels {

/// @brief Compute erfc-attenuated Coulomb ERIs for a shell quartet
///
/// Computes (ab|erfc(omega*r12)/r12|cd) using modified Rys quadrature.
/// The buffer is resized and zeroed before computation.
///
/// @param shell_a First bra shell
/// @param shell_b Second bra shell
/// @param shell_c First ket shell
/// @param shell_d Second ket shell
/// @param omega Range-separation parameter (in atomic units)
/// @param buffer Output buffer (resized and filled with ERIs)
void compute_eri_erfc_coulomb(const Shell& shell_a, const Shell& shell_b,
                               const Shell& shell_c, const Shell& shell_d,
                               Real omega,
                               TwoElectronBuffer<0>& buffer);

/// @brief Compute erfc-attenuated Coulomb ERIs using operator parameters
///
/// Convenience overload that extracts omega from RangeSeparatedParams.
///
/// @param shell_a First bra shell
/// @param shell_b Second bra shell
/// @param shell_c First ket shell
/// @param shell_d Second ket shell
/// @param params Range-separated operator parameters
/// @param buffer Output buffer
void compute_eri_erfc_coulomb(const Shell& shell_a, const Shell& shell_b,
                               const Shell& shell_c, const Shell& shell_d,
                               const RangeSeparatedParams& params,
                               TwoElectronBuffer<0>& buffer);

}  // namespace libaccint::kernels
