// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file projection_operator.cpp
/// @brief Implementation of projection operator matrix construction

#include <libaccint/operators/projection_operator.hpp>
#include <libaccint/math/cholesky.hpp>

#include <cmath>
#include <stdexcept>

namespace libaccint {

std::vector<Real> build_projection_matrix(const ProjectionOperatorParams& params) {
    if (!params.is_valid()) {
        throw std::invalid_argument(
            "ProjectionOperatorParams validation failed: coefficient/weight dimensions inconsistent");
    }

    const Size n = params.n_basis;
    const Size k = params.n_projectors;

    // P(μ,ν) = Σ_k w_k * C(μ,k) * C(ν,k)
    std::vector<Real> P(n * n, 0.0);

    for (Size proj = 0; proj < k; ++proj) {
        const Real w = params.weights[proj];
        for (Size mu = 0; mu < n; ++mu) {
            const Real c_mu = params.coefficient(mu, proj);
            for (Size nu = 0; nu < n; ++nu) {
                const Real c_nu = params.coefficient(nu, proj);
                P[mu * n + nu] += w * c_mu * c_nu;
            }
        }
    }

    return P;
}

bool verify_projection_matrix(const std::vector<Real>& P, Size n, Real tol) {
    if (P.size() != n * n) return false;

    // Check 1: Symmetry — P(μ,ν) = P(ν,μ)
    for (Size mu = 0; mu < n; ++mu) {
        for (Size nu = mu + 1; nu < n; ++nu) {
            if (std::abs(P[mu * n + nu] - P[nu * n + mu]) > tol) {
                return false;
            }
        }
    }

    // Check 2: Trace — tr(P) must be non-negative and finite
    Real trace = 0.0;
    for (Size i = 0; i < n; ++i) {
        trace += P[i * n + i];
    }
    if (!std::isfinite(trace) || trace < -tol) {
        return false;
    }

    // Check 3: Positive semi-definiteness via Cholesky decomposition
    // Add a small regularisation to the diagonal to handle the semi-definite
    // case (Cholesky requires strictly positive definite).
    std::vector<Real> P_shifted(P);
    const Real shift = tol > 0.0 ? tol : 1e-12;
    for (Size i = 0; i < n; ++i) {
        P_shifted[i * n + i] += shift;
    }
    try {
        math::cholesky_decompose(P_shifted.data(), n);
    } catch (const std::runtime_error&) {
        // Cholesky failed => matrix is not positive semi-definite
        return false;
    }

    return true;
}

}  // namespace libaccint
