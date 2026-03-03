// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file rys_quadrature.hpp
/// @brief Rys quadrature roots and weights for electron repulsion integrals
///
/// The Rys quadrature provides nodes {u_i} and weights {w_i} such that:
///   ∑_i w_i u_i^k = F_k(T)   for k = 0, 1, ..., 2n-1
///
/// where F_k(T) is the Boys function and u_i = t_i² are the squared Rys roots.
/// This n-point quadrature exactly integrates polynomials in t² of degree ≤ 2n-1
/// against the weight function exp(-Tt²) on [0, 1].
///
/// For an ERI with angular momenta L_a, L_b, L_c, L_d:
///   n_roots = ⌊(L_a + L_b + L_c + L_d) / 2⌋ + 1
///
/// Implementation uses:
/// - Modified Chebyshev algorithm to compute three-term recurrence from moments
/// - Implicit QL algorithm to find eigenvalues/eigenvectors of the Jacobi matrix
/// - Optional Newton-Raphson polishing for maximum accuracy

#include <array>

namespace libaccint::math {

// ============================================================================
// Constants
// ============================================================================

/// @brief Maximum number of Rys quadrature roots supported
///
/// Covers up to L_total = 28 (e.g., four i-shells): n_roots = 14 + 1 = 15
inline constexpr int MAX_RYS_ROOTS = 15;

// ============================================================================
// Primary Interface
// ============================================================================

/// @brief Compute Rys quadrature roots and weights
///
/// @param n_roots Number of quadrature points (1 ≤ n_roots ≤ MAX_RYS_ROOTS)
/// @param T Boys function argument (T ≥ 0)
/// @param roots Output: squared Rys roots u_i = t_i² in (0, 1), sorted ascending
/// @param weights Output: corresponding quadrature weights w_i > 0
///
/// After the call, the quadrature rule satisfies:
///   ∑_i w_i * roots[i]^k = F_k(T)   for k = 0, 1, ..., 2*n_roots - 1
///
/// The sum of weights equals F_0(T).
void rys_compute(int n_roots, double T, double* roots, double* weights);

/// @brief Compute Rys quadrature roots only (no weights)
///
/// @param n_roots Number of quadrature points (1 ≤ n_roots ≤ MAX_RYS_ROOTS)
/// @param T Boys function argument (T ≥ 0)
/// @param roots Output: squared Rys roots u_i = t_i² in (0, 1), sorted ascending
void rys_roots(int n_roots, double T, double* roots);

// ============================================================================
// Table-Based Lookup (Fast Path)
// ============================================================================

/// @brief Maximum T for table-based interpolation
inline constexpr double RYS_TABLE_T_MAX = 40.0;

/// @brief Compute Rys roots and weights via table interpolation
///
/// Uses precomputed tables with Neville interpolation for T in [0, 40].
/// Falls back to rys_compute() for T > 40.
///
/// @param n_roots Number of quadrature points (1 ≤ n_roots ≤ MAX_RYS_ROOTS)
/// @param T Boys function argument (T ≥ 0)
/// @param roots Output: squared Rys roots
/// @param weights Output: corresponding quadrature weights
void rys_lookup(int n_roots, double T, double* roots, double* weights);

// ============================================================================
// Weight Computation from Known Roots
// ============================================================================

/// @brief Compute quadrature weights from known roots
///
/// Given the squared roots u_i, computes weights by solving the
/// moment-matching linear system using the orthogonal polynomial approach.
///
/// @param n_roots Number of quadrature points
/// @param T Boys function argument
/// @param roots Input: squared Rys roots u_i
/// @param weights Output: corresponding quadrature weights
void rys_weights_from_roots(int n_roots, double T,
                            const double* roots, double* weights);

// ============================================================================
// Low-Level Functions (exposed for testing)
// ============================================================================

/// @brief Compute three-term recurrence coefficients via modified Chebyshev algorithm
///
/// Given moments μ_k = F_k(T) for k = 0..2n-1, computes the three-term
/// recurrence coefficients {α_k, β_k} for the monic orthogonal polynomials:
///   π_0(u) = 1
///   π_{k+1}(u) = (u - α_k) π_k(u) - β_k π_{k-1}(u)
///
/// @param n Number of roots (determines polynomial degree)
/// @param moments Array of 2n Boys function values [F_0(T), ..., F_{2n-1}(T)]
/// @param alpha Output: diagonal coefficients α_0..α_{n-1}
/// @param beta Output: off-diagonal coefficients β_0..β_{n-1}
///   Note: β_0 = μ_0 = F_0(T) is the norm of π_0, used only for weight computation.
///   The Jacobi matrix uses β_1..β_{n-1} for off-diagonal elements.
void rys_chebyshev(int n, const double* moments,
                   double* alpha, double* beta);

/// @brief Implicit QL algorithm for symmetric tridiagonal eigenvalue problem
///
/// Finds eigenvalues and first-row eigenvector components of the Jacobi matrix.
/// The eigenvalues are the Rys roots (u_i = t_i²), and the weights are
/// computed as w_i = β_0 × z_i².
///
/// @param n Matrix size
/// @param diag Diagonal elements (input); eigenvalues on output (sorted ascending)
/// @param offdiag Off-diagonal elements (destroyed on output)
/// @param z First row of eigenvector accumulator. Initialize to [1,0,...,0].
///   On output: z[i] = Q[0,i] where Q diagonalizes the matrix.
void tridiag_ql(int n, double* diag, double* offdiag, double* z);

/// @brief Evaluate Rys polynomial and its derivative at point u
///
/// Uses the three-term recurrence to evaluate π_n(u) and π_n'(u).
///
/// @param n Polynomial degree
/// @param u Evaluation point
/// @param alpha Recurrence coefficients α_0..α_{n-1}
/// @param beta Recurrence coefficients β_0..β_{n-1}
/// @param value Output: π_n(u)
/// @param derivative Output: π_n'(u)
void rys_poly_eval(int n, double u, const double* alpha, const double* beta,
                   double* value, double* derivative);

/// @brief Polish roots using Newton-Raphson iteration
///
/// Refines each root to machine precision using Newton's method on the
/// Rys polynomial π_n(u).
///
/// @param n Number of roots
/// @param alpha Recurrence coefficients
/// @param beta Recurrence coefficients
/// @param roots Roots to polish (modified in place)
/// @param max_iter Maximum Newton iterations per root
void rys_newton_polish(int n, const double* alpha, const double* beta,
                       double* roots, int max_iter = 10);

// ============================================================================
// Public API: RysRootsWeights and RysQuadrature
// ============================================================================

/// @brief Stack-allocated storage for Rys quadrature results
///
/// Provides compile-time sized arrays for roots and weights, avoiding
/// heap allocation in hot paths (integral evaluation kernels).
///
/// @tparam MaxRoots Maximum number of roots this container can hold
template <int MaxRoots>
struct RysRootsWeights {
    static_assert(MaxRoots >= 1 && MaxRoots <= MAX_RYS_ROOTS,
                  "MaxRoots must be in [1, MAX_RYS_ROOTS]");

    std::array<double, MaxRoots> roots{};
    std::array<double, MaxRoots> weights{};
    int n_roots = 0;

    /// @brief Compute roots and weights for given parameters
    ///
    /// @param n Number of quadrature points (1 ≤ n ≤ MaxRoots)
    /// @param T Boys function argument (T ≥ 0)
    void compute(int n, double T) {
        n_roots = n;
        rys_lookup(n, T, roots.data(), weights.data());
    }
};

/// @brief Singleton class providing Rys quadrature access
///
/// Thread-safe: all state is in read-only precomputed tables.
/// Provides a clean API hiding implementation details (table lookup
/// vs. direct computation, Chebyshev vs. Stieltjes, etc.).
class RysQuadrature {
public:
    /// @brief Access the singleton instance (Meyers singleton, thread-safe)
    static RysQuadrature& instance() {
        static RysQuadrature inst;
        return inst;
    }

    /// @brief Compute Rys quadrature roots and weights
    ///
    /// Uses table-accelerated lookup for T ∈ [0, 40] and n ≤ 12,
    /// with fallback to direct computation otherwise.
    ///
    /// @param n_roots Number of quadrature points (1 ≤ n_roots ≤ MAX_RYS_ROOTS)
    /// @param T Boys function argument (T ≥ 0)
    /// @param roots Output: squared Rys roots u_i = t_i² ∈ (0, 1)
    /// @param weights Output: corresponding quadrature weights w_i > 0
    void compute_roots_and_weights(int n_roots, double T,
                                   double* roots, double* weights) const {
        rys_lookup(n_roots, T, roots, weights);
    }

    /// @brief Compute into a stack-allocated result container
    ///
    /// @tparam MaxRoots Template parameter for stack storage size
    /// @param n_roots Number of quadrature points (1 ≤ n_roots ≤ MaxRoots)
    /// @param T Boys function argument (T ≥ 0)
    /// @return RysRootsWeights<MaxRoots> containing the computed results
    template <int MaxRoots>
    RysRootsWeights<MaxRoots> compute(int n_roots, double T) const {
        RysRootsWeights<MaxRoots> result;
        result.compute(n_roots, T);
        return result;
    }

    RysQuadrature(const RysQuadrature&) = delete;
    RysQuadrature& operator=(const RysQuadrature&) = delete;

private:
    RysQuadrature() = default;
};

/// @brief Convenience function to access the RysQuadrature singleton
inline RysQuadrature& get_rys_quadrature() {
    return RysQuadrature::instance();
}

// ============================================================================
// Template Wrapper for Generated Kernels
// ============================================================================

/// @brief Compile-time dispatched Rys roots/weights computation
///
/// Generated CPU kernels call this template wrapper which
/// forwards to the runtime rys_compute() implementation.
///
/// @tparam N Number of Rys quadrature roots (1 ≤ N ≤ MAX_RYS_ROOTS)
/// @param T Boys function argument (T ≥ 0)
/// @param roots Output: squared Rys roots u_i = t_i² in (0, 1)
/// @param weights Output: corresponding quadrature weights w_i > 0
template <int N>
inline void compute_rys_roots_weights(double T, double* roots, double* weights) {
    static_assert(N >= 1 && N <= MAX_RYS_ROOTS,
                  "N must be in [1, MAX_RYS_ROOTS]");
    rys_lookup(N, T, roots, weights);
}

} // namespace libaccint::math
