// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file distributed_multipole_kernel.hpp
/// @brief CPU and CUDA kernel declarations for distributed multipole integrals
///
/// Computes the one-electron matrix V^DMA_μν representing the electrostatic
/// potential from a set of external multipole sites (charges, dipoles, quadrupoles).
///
/// V^DMA_μν = Σ_s [ q_s * V^nuc(R_s) + μ_s · ∇V(R_s) + Θ_s : ∇∇V(R_s) + ... ]
///
/// At charge-only level this reduces to nuclear attraction integrals.

#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/operators/operator_types.hpp>

namespace libaccint::kernels {

/// @brief Compute distributed multipole one-electron integrals for a shell pair
///
/// Computes contributions from all multipole sites to the matrix element V_ab.
/// Uses Boys function evaluation for the Coulomb-type integrals, matching
/// the nuclear attraction kernel pattern.
///
/// For charge-only sites, this reduces exactly to nuclear attraction integrals.
/// Higher multipole ranks add derivative contributions.
///
/// @param shell_a First shell (bra)
/// @param shell_b Second shell (ket)
/// @param params Distributed multipole parameters (sites, charges, dipoles, quadrupoles)
/// @param buffer Output buffer (resized and filled)
void compute_distributed_multipole(
    const Shell& shell_a, const Shell& shell_b,
    const DistributedMultipoleParams& params,
    OverlapBuffer& buffer);

}  // namespace libaccint::kernels
