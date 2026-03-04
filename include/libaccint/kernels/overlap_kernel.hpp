// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file overlap_kernel.hpp
/// @brief Overlap integral kernel (S) for Cartesian Gaussian basis functions
///
/// Implements the overlap integral S_uv = <u|v> using Obara-Saika recursion.
/// The kernel processes a pair of shells and fills the output buffer with
/// the overlap integrals for all Cartesian component pairs.

#include <libaccint/basis/primitive_pair_data.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>

namespace libaccint::kernels {

/// @brief Compute overlap integrals for a pair of shells
///
/// Computes the overlap matrix elements S(a,b) for all Cartesian components
/// of two shells using Obara-Saika recursion:
///
///   S(a,b) = sum_ij c_i * c_j * prod_d I_d[l_a_d][l_b_d]
///
/// where:
///   - c_i, c_j are normalized contraction coefficients
///   - I_d is the 1D Obara-Saika recursion table for direction d
///   - l_a_d, l_b_d are Cartesian angular momentum components
///
/// The Shell's stored coefficients are normalized for the (L,0,0) component.
/// A normalization correction factor is applied for other Cartesian components.
///
/// @param shell_a First shell (bra)
/// @param shell_b Second shell (ket)
/// @param buffer Output buffer (resized and filled with overlap integrals)
void compute_overlap(const Shell& shell_a, const Shell& shell_b,
                     OverlapBuffer& buffer);

/// @brief Compute overlap integrals using pre-computed primitive pair data
///
/// Uses cached Gaussian products from PrimitivePairData instead of computing
/// them on-the-fly. This avoids redundant exp() calls when the same shell pair
/// is used repeatedly (e.g., across operators in a composed OneElectronOperator).
///
/// @param shell_a First shell (bra)
/// @param shell_b Second shell (ket)
/// @param pair_data Pre-computed Gaussian product data
/// @param shell_i Index of shell_a in its ShellSet
/// @param shell_j Index of shell_b in its ShellSet
/// @param buffer Output buffer (resized and filled with overlap integrals)
void compute_overlap(const Shell& shell_a, const Shell& shell_b,
                     const PrimitivePairData& pair_data,
                     Size shell_i, Size shell_j,
                     OverlapBuffer& buffer);

}  // namespace libaccint::kernels
