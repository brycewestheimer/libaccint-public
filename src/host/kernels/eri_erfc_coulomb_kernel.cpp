// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file eri_erfc_coulomb_kernel.cpp
/// @brief erfc-attenuated Coulomb ERI kernel
///
/// Implements erfc-Coulomb ERIs using the identity:
///   (ab|erfc(omega*r12)/r12|cd) = (ab|1/r12|cd) - (ab|erf(omega*r12)/r12|cd)
///
/// This approach is numerically stable and ensures the decomposition identity
/// erf + erfc = full Coulomb holds to machine precision.

#include <libaccint/kernels/eri_erfc_coulomb_kernel.hpp>
#include <libaccint/kernels/eri_kernel.hpp>
#include <libaccint/kernels/eri_erf_coulomb_kernel.hpp>

namespace libaccint::kernels {

void compute_eri_erfc_coulomb(const Shell& shell_a, const Shell& shell_b,
                               const Shell& shell_c, const Shell& shell_d,
                               Real omega,
                               TwoElectronBuffer<0>& buffer) {
    const int na = shell_a.n_functions();
    const int nb = shell_b.n_functions();
    const int nc = shell_c.n_functions();
    const int nd = shell_d.n_functions();

    // Handle limiting cases
    if (omega <= 0.0) {
        // omega -> 0: erfc(0) = 1, so erfc/r = 1/r (full Coulomb)
        compute_eri(shell_a, shell_b, shell_c, shell_d, buffer);
        return;
    }

    if (omega > 1000.0) {
        // omega -> infinity: erfc -> 0, so all integrals are zero
        buffer.resize(na, nb, nc, nd);
        buffer.clear();
        return;
    }

    // General case: erfc = full - erf
    // First compute full Coulomb ERI
    compute_eri(shell_a, shell_b, shell_c, shell_d, buffer);

    // Compute erf-Coulomb ERI in temporary buffer
    TwoElectronBuffer<0> erf_buffer;
    compute_eri_erf_coulomb(shell_a, shell_b, shell_c, shell_d, omega, erf_buffer);

    // Subtract: erfc = full - erf
    for (int ia = 0; ia < na; ++ia) {
        for (int ib = 0; ib < nb; ++ib) {
            for (int ic = 0; ic < nc; ++ic) {
                for (int id = 0; id < nd; ++id) {
                    buffer(ia, ib, ic, id) -= erf_buffer(ia, ib, ic, id);
                }
            }
        }
    }
}

void compute_eri_erfc_coulomb(const Shell& shell_a, const Shell& shell_b,
                               const Shell& shell_c, const Shell& shell_d,
                               const RangeSeparatedParams& params,
                               TwoElectronBuffer<0>& buffer) {
    compute_eri_erfc_coulomb(shell_a, shell_b, shell_c, shell_d,
                              params.omega, buffer);
}

}  // namespace libaccint::kernels
