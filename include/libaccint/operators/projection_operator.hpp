// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file projection_operator.hpp
/// @brief Projection operator engine pathway
///
/// Computes the projection matrix P = C * diag(w) * C^T algebraically
/// (no integral evaluation needed). The projection operator is used in
/// embedding and level-shifting contexts.

#include <libaccint/operators/operator_types.hpp>
#include <libaccint/core/types.hpp>

#include <vector>

namespace libaccint {

/// @brief Build the projection matrix P = C * diag(w) * C^T
///
/// Given a set of projector functions defined by their expansion coefficients C
/// and weights w, constructs the projection matrix:
///   P(μ,ν) = Σ_k w_k * C(μ,k) * C(ν,k)
///
/// This is a purely algebraic operation — no integral evaluation is needed.
///
/// @param params Projection operator parameters (coefficients, weights, dimensions)
/// @return Flattened projection matrix P of size n_basis × n_basis (row-major)
/// @throws std::invalid_argument if params.is_valid() returns false
[[nodiscard]] std::vector<Real> build_projection_matrix(
    const ProjectionOperatorParams& params);

/// @brief Verify projection matrix properties
///
/// Checks that P satisfies:
///   1. Symmetry: P(μ,ν) = P(ν,μ)
///   2. Trace: tr(P) = Σ_k w_k (for unit-weighted projectors)
///   3. Positive semi-definiteness (eigenvalues ≥ 0)
///
/// @param P Flattened projection matrix (n × n, row-major)
/// @param n Matrix dimension
/// @param tol Tolerance for symmetry check
/// @return true if all checks pass
[[nodiscard]] bool verify_projection_matrix(
    const std::vector<Real>& P, Size n, Real tol = 1e-12);

}  // namespace libaccint
