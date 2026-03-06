// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include "libaccint/math/rys_quadrature.hpp"
#include "libaccint/math/boys_function.hpp"
#include "libaccint/math/rys_tables.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace libaccint::math {

// ============================================================================
// Modified Chebyshev Algorithm
// ============================================================================

void rys_chebyshev(int n, const double* moments,
                   double* alpha, double* beta) {
    assert(n >= 1 && n <= MAX_RYS_ROOTS);

    // The modified Chebyshev algorithm constructs the three-term recurrence
    // coefficients {α_k, β_k} from the power moments μ_k = F_k(T).
    //
    // σ_{l,k} = <u^k, π_l> where π_l is the l-th monic orthogonal polynomial.
    //
    // Note: This approach is numerically stable for n ≤ ~12. For larger n,
    // use the Stieltjes procedure (rys_stieltjes) which avoids the
    // ill-conditioned moment matrix.

    const int n2 = 2 * n;

    // Three rows of the sigma table (circular buffer)
    double sigma[3][2 * MAX_RYS_ROOTS] = {};

    int r_m2 = 0;  // sigma[l-2] — initialized to zeros (represents l = -1)
    int r_m1 = 1;  // sigma[l-1] — initialized to moments (represents l = 0)
    int r_cur = 2;

    for (int k = 0; k < n2; ++k) {
        sigma[r_m1][k] = moments[k];
    }

    alpha[0] = moments[1] / moments[0];
    beta[0] = moments[0];

    for (int l = 1; l < n; ++l) {
        for (int k = l; k < n2 - l; ++k) {
            sigma[r_cur][k] = sigma[r_m1][k + 1]
                            - alpha[l - 1] * sigma[r_m1][k]
                            - beta[l - 1] * sigma[r_m2][k];
        }

        alpha[l] = sigma[r_cur][l + 1] / sigma[r_cur][l]
                 - sigma[r_m1][l] / sigma[r_m1][l - 1];
        beta[l] = sigma[r_cur][l] / sigma[r_m1][l - 1];

        int tmp = r_m2;
        r_m2 = r_m1;
        r_m1 = r_cur;
        r_cur = tmp;
    }
}

// ============================================================================
// Implicit QL Algorithm for Symmetric Tridiagonal Eigenvalue Problem
// ============================================================================

void tridiag_ql(int n, double* diag, double* offdiag, double* z) {
    assert(n >= 1);

    if (n == 1) return;

    // Ensure the last off-diagonal element is zero
    offdiag[n - 1] = 0.0;

    for (int l = 0; l < n; ++l) {
        int iter = 0;

        while (true) {
            // Find smallest m >= l such that offdiag[m] is negligible
            int m = l;
            while (m < n - 1) {
                double tst = std::abs(diag[m]) + std::abs(diag[m + 1]);
                if (tst + std::abs(offdiag[m]) == tst) break;
                ++m;
            }

            if (m == l) break;

            assert(++iter <= 50 && "QL algorithm failed to converge");

            // Compute implicit shift
            double g = (diag[l + 1] - diag[l]) / (2.0 * offdiag[l]);
            double r = std::hypot(g, 1.0);
            g = diag[m] - diag[l] + offdiag[l] / (g + std::copysign(r, g));

            double s = 1.0, c = 1.0, p = 0.0;

            // Chase the bulge from bottom to top
            int i;
            for (i = m - 1; i >= l; --i) {
                double f = s * offdiag[i];
                double b = c * offdiag[i];

                r = std::hypot(f, g);
                offdiag[i + 1] = r;

                if (r < 1e-300) {
                    diag[i + 1] -= p;
                    offdiag[m] = 0.0;
                    break;
                }

                s = f / r;
                c = g / r;
                g = diag[i + 1] - p;
                r = (diag[i] - g) * s + 2.0 * c * b;
                p = s * r;
                diag[i + 1] = g + p;
                g = c * r - b;

                // Accumulate first row of eigenvector matrix
                f = z[i + 1];
                z[i + 1] = s * z[i] + c * f;
                z[i] = c * z[i] - s * f;
            }

            if (r < 1e-300 && i >= l) {
                continue;
            }

            diag[l] -= p;
            offdiag[l] = g;
            offdiag[m] = 0.0;
        }
    }

    // Sort eigenvalues (and corresponding z) in ascending order
    for (int i = 0; i < n - 1; ++i) {
        int k = i;
        double pk = diag[i];
        for (int j = i + 1; j < n; ++j) {
            if (diag[j] < pk) {
                k = j;
                pk = diag[j];
            }
        }
        if (k != i) {
            diag[k] = diag[i];
            diag[i] = pk;
            double tmp = z[k];
            z[k] = z[i];
            z[i] = tmp;
        }
    }
}

// ============================================================================
// Rys Polynomial Evaluation
// ============================================================================

void rys_poly_eval(int n, double u, const double* alpha, const double* beta,
                   double* value, double* derivative) {
    assert(n >= 1);

    double p_prev = 1.0;
    double p_curr = u - alpha[0];
    double dp_prev = 0.0;
    double dp_curr = 1.0;

    if (n == 1) {
        *value = p_curr;
        *derivative = dp_curr;
        return;
    }

    for (int k = 1; k < n; ++k) {
        double u_minus_ak = u - alpha[k];
        double p_next = u_minus_ak * p_curr - beta[k] * p_prev;
        double dp_next = p_curr + u_minus_ak * dp_curr - beta[k] * dp_prev;

        p_prev = p_curr;
        p_curr = p_next;
        dp_prev = dp_curr;
        dp_curr = dp_next;
    }

    *value = p_curr;
    *derivative = dp_curr;
}

// ============================================================================
// Newton-Raphson Root Polishing
// ============================================================================

void rys_newton_polish(int n, const double* alpha, const double* beta,
                       double* roots, int max_iter) {
    for (int i = 0; i < n; ++i) {
        double u = roots[i];

        for (int iter = 0; iter < max_iter; ++iter) {
            double val, deriv;
            rys_poly_eval(n, u, alpha, beta, &val, &deriv);

            if (std::abs(deriv) < 1e-300) break;

            double delta = val / deriv;
            u -= delta;

            if (std::abs(delta) <= std::abs(u) * 2.0 *
                std::numeric_limits<double>::epsilon()) {
                break;
            }
        }

        roots[i] = u;
    }
}

// ============================================================================
// Discretized Stieltjes Procedure (robust for large n)
// ============================================================================

namespace {

/// @brief Compute M-point Gauss-Legendre quadrature on [0, 1]
///
/// Uses the Golub-Welsch approach: the Jacobi matrix for Legendre polynomials
/// has known coefficients (diagonal = 0, off-diagonal = k/sqrt(4k²-1)),
/// so we apply the tridiag_ql eigensolver directly.
void gauss_legendre_01(int M, double* nodes, double* wts) {
    assert(M >= 1);

    if (M == 1) {
        nodes[0] = 0.5;
        wts[0] = 1.0;
        return;
    }

    // Build tridiagonal Jacobi matrix for Legendre on [-1, 1]
    // Monic Legendre: a_k = 0, b_k = k²/(4k²-1)
    std::vector<double> diag(M, 0.0);
    std::vector<double> offdiag(M, 0.0);
    std::vector<double> z(M, 0.0);
    z[0] = 1.0;

    for (int i = 0; i < M - 1; ++i) {
        int k = i + 1;
        double bk = static_cast<double>(k * k) / (4.0 * k * k - 1.0);
        offdiag[i] = std::sqrt(bk);
    }

    tridiag_ql(M, diag.data(), offdiag.data(), z.data());

    // Transform from [-1,1] to [0,1]: x' = (x+1)/2
    // Weights on [-1,1]: w_i = 2 × z_i²  (since b_0 = μ_0 = 2 for Legendre)
    // On [0,1]: w_i = z_i²  (the factor of 2 cancels with the /2 Jacobian)
    for (int i = 0; i < M; ++i) {
        nodes[i] = (diag[i] + 1.0) / 2.0;
        wts[i] = z[i] * z[i];
    }
}

/// @brief Cached wrapper for gauss_legendre_01
///
/// Thread-local cache mapping grid size M to precomputed GL nodes/weights.
/// Avoids recomputing the QL eigensolver each time Stieltjes is invoked.
void gauss_legendre_01_cached(int M, const double*& nodes, const double*& wts) {
    using GLGrid = std::pair<std::vector<double>, std::vector<double>>;
    thread_local std::unordered_map<int, GLGrid> gl_cache;

    auto it = gl_cache.find(M);
    if (it == gl_cache.end()) {
        GLGrid grid;
        grid.first.resize(M);
        grid.second.resize(M);
        gauss_legendre_01(M, grid.first.data(), grid.second.data());
        auto [ins_it, _] = gl_cache.emplace(M, std::move(grid));
        it = ins_it;
    }
    nodes = it->second.first.data();
    wts = it->second.second.data();
}

} // anonymous namespace

/// @brief Compute recurrence coefficients via discretized Stieltjes procedure
///
/// More robust than the modified Chebyshev algorithm for n > ~12.
/// Uses Gauss-Legendre quadrature in the t-variable to evaluate inner products
/// directly, avoiding the ill-conditioned moment (Hankel) matrix.
static void rys_stieltjes(int n, double T, double* alpha, double* beta) {
    // Number of GL quadrature points: need enough to integrate degree ~8n
    // polynomials accurately. M >= 4n suffices; use at least 100.
    const int M = std::max(100, 6 * n);

    const double* gl_nodes = nullptr;
    const double* gl_weights = nullptr;
    gauss_legendre_01_cached(M, gl_nodes, gl_weights);

    // Precompute: u_j = t_j² and combined weights w_j = gl_w_j × exp(-T t_j²)
    std::vector<double> u(M), w(M);
    for (int j = 0; j < M; ++j) {
        double t = gl_nodes[j];
        u[j] = t * t;
        w[j] = gl_weights[j] * std::exp(-T * u[j]);
    }

    // π_k(u_j) values — store current and previous levels
    std::vector<double> pi_prev(M, 0.0);   // π_{k-2}
    std::vector<double> pi_curr(M, 1.0);   // π_{k-1} = π_0 = 1

    // <π_0, π_0>
    double norm_prev = 0.0;
    for (int j = 0; j < M; ++j) {
        norm_prev += w[j];
    }
    beta[0] = norm_prev;

    // <u π_0, π_0>
    double u_prod = 0.0;
    for (int j = 0; j < M; ++j) {
        u_prod += w[j] * u[j];
    }
    alpha[0] = u_prod / norm_prev;

    for (int k = 1; k < n; ++k) {
        // Build π_k(u_j) = (u_j - α_{k-1}) π_{k-1}(u_j) - β_{k-1} π_{k-2}(u_j)
        std::vector<double> pi_next(M);
        for (int j = 0; j < M; ++j) {
            pi_next[j] = (u[j] - alpha[k - 1]) * pi_curr[j];
            // Skip beta[k-1] * pi_prev for k=1: at this point pi_prev holds
            // pi_0 (all 1.0), not pi_{-1} (all 0.0), so the subtraction
            // would incorrectly remove beta[0] * pi_0 from the recurrence.
            if (k >= 2) {
                pi_next[j] -= beta[k - 1] * pi_prev[j];
            }
        }

        // <π_k, π_k>
        double norm_curr = 0.0;
        for (int j = 0; j < M; ++j) {
            norm_curr += w[j] * pi_next[j] * pi_next[j];
        }
        beta[k] = norm_curr / norm_prev;

        // <u π_k, π_k>
        u_prod = 0.0;
        for (int j = 0; j < M; ++j) {
            u_prod += w[j] * u[j] * pi_next[j] * pi_next[j];
        }
        alpha[k] = u_prod / norm_curr;

        norm_prev = norm_curr;
        pi_prev = pi_curr;
        pi_curr = pi_next;
    }
}

// ============================================================================
// Weight Computation from Known Roots
// ============================================================================

void rys_weights_from_roots(int n_roots, double T,
                            const double* roots, double* weights) {
    assert(n_roots >= 1 && n_roots <= MAX_RYS_ROOTS);
    assert(T >= 0.0);

    std::array<double, 2 * MAX_RYS_ROOTS> moments;
    boys_evaluate_array(2 * n_roots - 1, T, moments.data());

    if (n_roots == 1) {
        weights[0] = moments[0];
        return;
    }

    // Compute recurrence coefficients
    std::array<double, MAX_RYS_ROOTS> alpha, beta;
    rys_chebyshev(n_roots, moments.data(), alpha.data(), beta.data());

    // h_{n-1} = β_0 × β_1 × ... × β_{n-1}
    double h = 1.0;
    for (int k = 0; k < n_roots; ++k) {
        h *= beta[k];
    }

    for (int i = 0; i < n_roots; ++i) {
        double val, deriv;
        rys_poly_eval(n_roots, roots[i], alpha.data(), beta.data(),
                      &val, &deriv);

        // Evaluate π_{n-1}(u_i)
        double p_prev = 1.0;
        double p_curr = roots[i] - alpha[0];
        for (int k = 1; k < n_roots - 1; ++k) {
            double p_next = (roots[i] - alpha[k]) * p_curr - beta[k] * p_prev;
            p_prev = p_curr;
            p_curr = p_next;
        }

        weights[i] = h / (deriv * p_curr);
    }
}

// ============================================================================
// Core: Compute roots and weights from recurrence coefficients
// ============================================================================

namespace {

/// @brief Given alpha/beta recurrence coefficients, compute roots and weights
///        via QL eigensolver
void rys_from_coeffs(int n_roots, const double* alpha, const double* beta,
                     double* roots, double* weights) {
    std::array<double, MAX_RYS_ROOTS> diag, offdiag, z;

    for (int i = 0; i < n_roots; ++i) {
        diag[i] = alpha[i];
        z[i] = (i == 0) ? 1.0 : 0.0;
    }
    for (int i = 0; i < n_roots - 1; ++i) {
        offdiag[i] = std::sqrt(std::abs(beta[i + 1]));
    }
    offdiag[n_roots - 1] = 0.0;

    tridiag_ql(n_roots, diag.data(), offdiag.data(), z.data());

    for (int i = 0; i < n_roots; ++i) {
        roots[i] = diag[i];
        weights[i] = beta[0] * z[i] * z[i];
    }
}

/// @brief Check if all roots are within valid range (0, 1)
bool roots_valid(int n, const double* roots) {
    for (int i = 0; i < n; ++i) {
        if (roots[i] <= 0.0 || roots[i] >= 1.0) return false;
    }
    return true;
}

} // anonymous namespace

// ============================================================================
// Primary Interface: Compute Roots and Weights
// ============================================================================

void rys_compute(int n_roots, double T, double* roots, double* weights) {
    assert(n_roots >= 1 && n_roots <= MAX_RYS_ROOTS);
    assert(T >= 0.0);
    assert(roots != nullptr);
    assert(weights != nullptr);

    // Special case: single root (direct formula)
    if (n_roots == 1) {
        std::array<double, 2> moments;
        boys_evaluate_array(1, T, moments.data());
        roots[0] = moments[1] / moments[0];
        weights[0] = moments[0];
        return;
    }

    // Step 1: Try the moment-based modified Chebyshev algorithm (fast)
    std::array<double, 2 * MAX_RYS_ROOTS> moments;
    boys_evaluate_array(2 * n_roots - 1, T, moments.data());

    std::array<double, MAX_RYS_ROOTS> alpha, beta;
    rys_chebyshev(n_roots, moments.data(), alpha.data(), beta.data());

    // Check for invalid beta values (sign of instability)
    bool chebyshev_ok = true;
    for (int i = 1; i < n_roots; ++i) {
        if (beta[i] <= 0.0) {
            chebyshev_ok = false;
            break;
        }
    }

    if (chebyshev_ok) {
        rys_from_coeffs(n_roots, alpha.data(), beta.data(), roots, weights);

        if (roots_valid(n_roots, roots)) {
            // Polish roots with Newton-Raphson for maximum accuracy
            rys_newton_polish(n_roots, alpha.data(), beta.data(), roots, 10);
            // Clamp to valid range (matching GPU kernels) as safety net
            for (int i = 0; i < n_roots; ++i) {
                roots[i] = std::max(1e-14, std::min(roots[i], 1.0 - 1e-14));
                weights[i] = std::max(0.0, weights[i]);
            }
            return;
        }
    }

    // Step 2: Fallback to Stieltjes procedure (robust for large n)
    rys_stieltjes(n_roots, T, alpha.data(), beta.data());
    rys_from_coeffs(n_roots, alpha.data(), beta.data(), roots, weights);

    // Polish roots
    rys_newton_polish(n_roots, alpha.data(), beta.data(), roots, 10);

    // Clamp to valid range (matching GPU kernels) as safety net
    for (int i = 0; i < n_roots; ++i) {
        roots[i] = std::max(1e-14, std::min(roots[i], 1.0 - 1e-14));
        weights[i] = std::max(0.0, weights[i]);
    }
}

void rys_roots(int n_roots, double T, double* roots) {
    assert(n_roots >= 1 && n_roots <= MAX_RYS_ROOTS);
    assert(T >= 0.0);
    assert(roots != nullptr);

    std::array<double, MAX_RYS_ROOTS> weights;
    rys_compute(n_roots, T, roots, weights.data());
}

// ============================================================================
// Table-Based Lookup with Neville Interpolation
// ============================================================================

namespace {

/// @brief Neville interpolation of order P (P+1 points)
///
/// Interpolates a scalar function f(T) given P+1 equally-spaced samples
/// centered around the target T.
///
/// @param table Pointer to the flat table (stride = n_roots between T entries)
/// @param component Index of the root/weight within each T entry
/// @param n_roots Stride (number of values per T entry)
/// @param T Target argument
/// @return Interpolated value
/// @brief Neville interpolation of order P (P+1 points)
double neville_interpolate(const double* table, int component, int n_roots,
                           double T) {
    constexpr int P = 6;  // Interpolation order (7 points)
    const int N = detail::RYS_TABLE_N_ENTRIES;
    const double h = detail::RYS_TABLE_STEP;

    // Find the centered window of P+1 grid points
    int i0 = static_cast<int>(T / h) - P / 2;
    if (i0 < 0) i0 = 0;
    if (i0 + P >= N) i0 = N - P - 1;

    // Extract the P+1 function values
    double c[P + 1];
    double t[P + 1];
    for (int j = 0; j <= P; ++j) {
        t[j] = (i0 + j) * h;
        c[j] = table[(i0 + j) * n_roots + component];
    }

    // Neville's algorithm
    for (int k = 1; k <= P; ++k) {
        for (int j = P; j >= k; --j) {
            c[j] = ((T - t[j - k]) * c[j] - (T - t[j]) * c[j - 1]) /
                   (t[j] - t[j - k]);
        }
    }

    return c[P];
}

/// @brief Get table pointers for given n_roots
void get_table_ptrs(int n_roots, const double*& roots_table,
                    const double*& weights_table) {
    switch (n_roots) {
        case 1:  roots_table = &detail::RYS_ROOTS_1[0][0];
                 weights_table = &detail::RYS_WEIGHTS_1[0][0]; break;
        case 2:  roots_table = &detail::RYS_ROOTS_2[0][0];
                 weights_table = &detail::RYS_WEIGHTS_2[0][0]; break;
        case 3:  roots_table = &detail::RYS_ROOTS_3[0][0];
                 weights_table = &detail::RYS_WEIGHTS_3[0][0]; break;
        case 4:  roots_table = &detail::RYS_ROOTS_4[0][0];
                 weights_table = &detail::RYS_WEIGHTS_4[0][0]; break;
        case 5:  roots_table = &detail::RYS_ROOTS_5[0][0];
                 weights_table = &detail::RYS_WEIGHTS_5[0][0]; break;
        case 6:  roots_table = &detail::RYS_ROOTS_6[0][0];
                 weights_table = &detail::RYS_WEIGHTS_6[0][0]; break;
        case 7:  roots_table = &detail::RYS_ROOTS_7[0][0];
                 weights_table = &detail::RYS_WEIGHTS_7[0][0]; break;
        case 8:  roots_table = &detail::RYS_ROOTS_8[0][0];
                 weights_table = &detail::RYS_WEIGHTS_8[0][0]; break;
        case 9:  roots_table = &detail::RYS_ROOTS_9[0][0];
                 weights_table = &detail::RYS_WEIGHTS_9[0][0]; break;
        case 10: roots_table = &detail::RYS_ROOTS_10[0][0];
                 weights_table = &detail::RYS_WEIGHTS_10[0][0]; break;
        case 11: roots_table = &detail::RYS_ROOTS_11[0][0];
                 weights_table = &detail::RYS_WEIGHTS_11[0][0]; break;
        case 12: roots_table = &detail::RYS_ROOTS_12[0][0];
                 weights_table = &detail::RYS_WEIGHTS_12[0][0]; break;
        case 13: roots_table = &detail::RYS_ROOTS_13[0][0];
                 weights_table = &detail::RYS_WEIGHTS_13[0][0]; break;
        case 14: roots_table = &detail::RYS_ROOTS_14[0][0];
                 weights_table = &detail::RYS_WEIGHTS_14[0][0]; break;
        case 15: roots_table = &detail::RYS_ROOTS_15[0][0];
                 weights_table = &detail::RYS_WEIGHTS_15[0][0]; break;
        default:
            throw std::out_of_range("rys_quadrature: n_roots must be in [1, 15], got " +
                                    std::to_string(n_roots));
    }
}

/// @brief Compute weights from roots and recurrence coefficients (Christoffel)
///
/// w_i = h_{n-1} / (π_n'(u_i) × π_{n-1}(u_i))
/// where h_{n-1} = β_0 × β_1 × ... × β_{n-1}
void weights_from_recurrence(int n, const double* alpha, const double* beta,
                             const double* roots, double* weights) {
    if (n == 1) {
        weights[0] = beta[0];
        return;
    }

    double h = 1.0;
    for (int k = 0; k < n; ++k) {
        h *= beta[k];
    }

    for (int i = 0; i < n; ++i) {
        double val, deriv;
        rys_poly_eval(n, roots[i], alpha, beta, &val, &deriv);

        // Evaluate π_{n-1}(roots[i])
        double p_prev = 1.0;
        double p_curr = roots[i] - alpha[0];
        for (int k = 1; k < n - 1; ++k) {
            double p_next = (roots[i] - alpha[k]) * p_curr - beta[k] * p_prev;
            p_prev = p_curr;
            p_curr = p_next;
        }

        weights[i] = h / (deriv * p_curr);
    }
}

} // anonymous namespace

void rys_lookup(int n_roots, double T, double* roots, double* weights) {
    assert(n_roots >= 1 && n_roots <= MAX_RYS_ROOTS);
    assert(T >= 0.0);
    assert(roots != nullptr);
    assert(weights != nullptr);

    // Fallback to direct computation for T outside table range or large n_roots
    // (n > 12 requires Stieltjes procedure whose recurrence coefficients are
    // sensitive to the exact T value, making table-interpolated initial guesses
    // unreliable for Newton polishing)
    if (T > detail::RYS_TABLE_T_MAX || n_roots > 12) {
        rys_compute(n_roots, T, roots, weights);
        return;
    }

    // Single root: direct formula (exact, no interpolation needed)
    if (n_roots == 1) {
        std::array<double, 2> moments;
        boys_evaluate_array(1, T, moments.data());
        roots[0] = moments[1] / moments[0];
        weights[0] = moments[0];
        return;
    }

    // Step 1: Interpolate roots from table as initial guesses
    const double* roots_table = nullptr;
    const double* weights_table = nullptr;
    get_table_ptrs(n_roots, roots_table, weights_table);

    for (int i = 0; i < n_roots; ++i) {
        roots[i] = neville_interpolate(roots_table, i, n_roots, T);
    }

    // Step 2: Compute recurrence coefficients for Newton polishing
    // (n ≤ 12 guaranteed here, so Chebyshev is numerically stable)
    std::array<double, 2 * MAX_RYS_ROOTS> moments;
    boys_evaluate_array(2 * n_roots - 1, T, moments.data());

    std::array<double, MAX_RYS_ROOTS> alpha, beta;
    rys_chebyshev(n_roots, moments.data(), alpha.data(), beta.data());

    // Step 3: Polish interpolated roots to machine precision via Newton-Raphson
    rys_newton_polish(n_roots, alpha.data(), beta.data(), roots, 10);

    // Step 4: Compute weights from polished roots via Christoffel formula
    weights_from_recurrence(n_roots, alpha.data(), beta.data(),
                            roots, weights);
}

} // namespace libaccint::math
