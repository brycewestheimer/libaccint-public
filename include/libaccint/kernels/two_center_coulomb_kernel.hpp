// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file two_center_coulomb_kernel.hpp
/// @brief Two-center Coulomb integral kernel for density fitting metric
///
/// Computes (P|Q) integrals between auxiliary basis functions:
///   (P|Q) = integral chi_P(r1) * (1/r12) * chi_Q(r2) dr1 dr2
///
/// These integrals form the metric matrix in density fitting.

#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/core/types.hpp>

#include <span>

namespace libaccint::kernels {

// =============================================================================
// Two-Center Coulomb Kernel Interface
// =============================================================================

/// @brief Compute two-center Coulomb integral for a pair of shells
///
/// Computes all (n_P x n_Q) integrals for auxiliary shells P and Q.
///
/// @param shell_P First auxiliary shell
/// @param shell_Q Second auxiliary shell
/// @param[out] buffer Output buffer resized to (n_P, n_Q)
void compute_two_center_coulomb(const Shell& shell_P,
                                 const Shell& shell_Q,
                                 TwoElectronBuffer<0>& buffer);

/// @brief Compute two-center Coulomb integral block
///
/// Computes integrals and stores in a raw array.
/// Array must have size >= n_P * n_Q.
///
/// @param shell_P First auxiliary shell
/// @param shell_Q Second auxiliary shell
/// @param[out] output Pointer to output array (row-major, n_P rows x n_Q cols)
/// @param stride Row stride for output (default = n_Q)
void compute_two_center_coulomb_block(const Shell& shell_P,
                                       const Shell& shell_Q,
                                       Real* output,
                                       Size stride = 0);

/// @brief Compute single two-center Coulomb integral (for testing)
///
/// For s-shell pairs with single contraction, returns the scalar integral.
///
/// @param shell_P First auxiliary shell (must be s-type)
/// @param shell_Q Second auxiliary shell (must be s-type)
/// @return (P|Q) integral value
[[nodiscard]] Real compute_two_center_coulomb_scalar(const Shell& shell_P,
                                                      const Shell& shell_Q);

// =============================================================================
// Batched Interface
// =============================================================================

/// @brief Compute two-center Coulomb matrix for full auxiliary basis
///
/// Computes the full (P|Q) metric matrix for an auxiliary basis.
/// Exploits symmetry: (P|Q) = (Q|P).
///
/// @param shells All auxiliary shells
/// @param[out] metric Output matrix (n_aux x n_aux), row-major
/// @param n_aux Total number of auxiliary functions
void compute_two_center_metric(std::span<const Shell> shells,
                                Real* metric,
                                Size n_aux);

/// @brief Compute upper triangle of two-center Coulomb matrix
///
/// Only computes P <= Q entries for symmetric storage.
///
/// @param shells All auxiliary shells
/// @param[out] metric_upper Upper triangle in packed format
/// @param n_aux Total number of auxiliary functions
void compute_two_center_metric_upper(std::span<const Shell> shells,
                                      Real* metric_upper,
                                      Size n_aux);

}  // namespace libaccint::kernels
