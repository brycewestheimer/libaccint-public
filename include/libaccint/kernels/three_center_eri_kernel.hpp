// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file three_center_eri_kernel.hpp
/// @brief Three-center ERI kernel for density fitting
///
/// Computes (ab|P) integrals where a, b are orbital basis functions
/// and P is an auxiliary basis function:
///   (ab|P) = integral phi_a(r1) * phi_b(r1) * (1/r12) * chi_P(r2) dr1 dr2
///
/// These integrals are the computational heart of density fitting.

#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/core/types.hpp>

#include <span>

namespace libaccint::kernels {

// =============================================================================
// Three-Center ERI Kernel Interface
// =============================================================================

/// @brief Compute three-center ERI for a shell triple (a, b | P)
///
/// Computes all (n_a * n_b * n_P) integrals.
///
/// @param shell_a First orbital shell
/// @param shell_b Second orbital shell
/// @param shell_P Auxiliary shell
/// @param[out] buffer Output buffer resized to (n_a * n_b, n_P)
void compute_three_center_eri(const Shell& shell_a,
                               const Shell& shell_b,
                               const Shell& shell_P,
                               TwoElectronBuffer<0>& buffer);

/// @brief Compute three-center ERI block
///
/// Stores integrals in raw array with layout (ab, P).
///
/// @param shell_a First orbital shell
/// @param shell_b Second orbital shell
/// @param shell_P Auxiliary shell
/// @param[out] output Array of size >= n_a * n_b * n_P
/// @param stride_ab Stride between ab pairs (default = n_P)
/// @param stride_a Stride between a functions (default = n_b * stride_ab)
void compute_three_center_eri_block(const Shell& shell_a,
                                     const Shell& shell_b,
                                     const Shell& shell_P,
                                     Real* output,
                                     Size stride_ab = 0,
                                     Size stride_a = 0);

/// @brief Compute single three-center ERI (for testing)
///
/// For s-shell triples, returns the scalar integral.
///
/// @param shell_a First orbital shell (s-type)
/// @param shell_b Second orbital shell (s-type)
/// @param shell_P Auxiliary shell (s-type)
/// @return (ab|P) integral value
[[nodiscard]] Real compute_three_center_eri_scalar(const Shell& shell_a,
                                                    const Shell& shell_b,
                                                    const Shell& shell_P);

// =============================================================================
// Batched Interface
// =============================================================================

/// @brief Three-center integral tensor storage format
enum class ThreeCenterStorageFormat {
    abP,   ///< (n_orb, n_orb, n_aux) - full tensor
    uabP,  ///< (n_pairs, n_aux) - upper triangle a <= b
    Pab,   ///< (n_aux, n_orb, n_orb) - auxiliary first
    Puab   ///< (n_aux, n_pairs) - auxiliary first, upper triangle
};

/// @brief Compute three-center ERI tensor for full basis
///
/// @param orbital_shells Orbital basis shells
/// @param auxiliary_shells Auxiliary basis shells  
/// @param[out] tensor Output tensor
/// @param n_orb Number of orbital basis functions
/// @param n_aux Number of auxiliary basis functions
/// @param format Storage format for tensor
void compute_three_center_tensor(std::span<const Shell> orbital_shells,
                                  std::span<const Shell> auxiliary_shells,
                                  Real* tensor,
                                  Size n_orb,
                                  Size n_aux,
                                  ThreeCenterStorageFormat format = ThreeCenterStorageFormat::abP);

/// @brief Compute B tensor: B_ab^P = sum_Q (ab|Q) * L^{-1}_{QP}
///
/// The B tensor is the key intermediate for DF-Fock construction.
/// L is the Cholesky factor of the metric: L * L^T = (P|Q).
///
/// @param three_center Three-center integrals (ab|P) in abP format
/// @param L_inv Inverse of Cholesky factor L
/// @param[out] B_tensor Output B tensor in abP format
/// @param n_orb Number of orbital functions
/// @param n_aux Number of auxiliary functions
void compute_B_tensor(const Real* three_center,
                       const Real* L_inv,
                       Real* B_tensor,
                       Size n_orb,
                       Size n_aux);

}  // namespace libaccint::kernels
