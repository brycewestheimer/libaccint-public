// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file matrix_assembly.hpp
/// @brief Convenience utilities for assembling one-electron integral matrices

#include <libaccint/engine/engine.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/core/types.hpp>
#include <span>
#include <vector>

namespace libaccint {

/// @brief Assemble a one-electron integral matrix
///
/// Computes the full nbf × nbf matrix for the given one-electron operator
/// using the Engine's compute_1e method. The matrix is stored in row-major order.
///
/// @param engine The Engine to use for computation
/// @param op The one-electron operator (S, T, V, or composed)
/// @param matrix Output span of size nbf × nbf (pre-allocated by caller)
/// @throws InvalidArgumentException if matrix.size() != nbf * nbf
void assemble_one_electron_matrix(Engine& engine,
                                  const OneElectronOperator& op,
                                  std::span<Real> matrix);

/// @brief Assemble a one-electron integral matrix (convenience overload)
///
/// Returns a newly allocated vector containing the nbf × nbf matrix.
///
/// @param engine The Engine to use for computation
/// @param op The one-electron operator
/// @return Vector of size nbf × nbf containing the matrix in row-major order
[[nodiscard]] std::vector<Real> assemble_one_electron_matrix(
    Engine& engine,
    const OneElectronOperator& op);

}  // namespace libaccint
