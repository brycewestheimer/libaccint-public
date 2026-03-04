// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file nuclear_kernel.cpp
/// @brief Nuclear attraction integral kernel implementation using Rys quadrature
///
/// Computes V_uv = <u| sum_C -Z_C/|r-R_C| |v> for a pair of shells and a set
/// of point charges. The algorithm uses Rys quadrature to decompose the 1/r
/// Coulomb operator into a sum over quadrature points, where each point
/// contributes through 1D recursion integrals that share the same structure
/// as the Obara-Saika overlap recursion but with root-dependent parameters.

#include <libaccint/kernels/nuclear_kernel.hpp>
#include <libaccint/kernels/norm_correction.hpp>
#include <libaccint/math/gaussian_product.hpp>
#include <libaccint/math/normalization.hpp>
#include <libaccint/math/cartesian_indices.hpp>
#include <libaccint/math/rys_quadrature.hpp>
#include <libaccint/utils/constants.hpp>

#include <cmath>
#include <vector>

namespace libaccint::kernels {

namespace {

/// @brief Build 1D Rys recursion table for one Cartesian direction
///
/// Uses the modified Obara-Saika recursion with Rys-root-dependent parameters:
///   I(0,0) = 1.0  (the prefactor is applied separately)
///
/// Recursion to increment first index:
///   I(i+1, j) = PA_eff * I(i, j) + B00 * [i * I(i-1, j) + j * I(i, j-1)]
///
/// Recursion to increment second index:
///   I(i, j+1) = PB_eff * I(i, j) + B00 * [i * I(i-1, j) + j * I(i, j-1)]
///
/// where:
///   PA_eff = (P_d - A_d) - u * (P_d - C_d)     (effective PA displacement)
///   PB_eff = (P_d - B_d) - u * (P_d - C_d)     (effective PB displacement)
///   B00    = (1 - u) / (2 * zeta)                (root-dependent half-inverse exponent)
///   u      = t^2 is the squared Rys root
///
/// @param La Maximum angular momentum for first center
/// @param Lb Maximum angular momentum for second center
/// @param PA_eff Effective displacement from product center to center A
/// @param PB_eff Effective displacement from product center to center B
/// @param B00 Root-dependent half-inverse exponent: (1-u)/(2*zeta)
/// @param[out] I Recursion table of size (La+1) x (Lb+1)
void build_1d_rys(int La, int Lb,
                  Real PA_eff, Real PB_eff, Real B00,
                  std::vector<std::vector<Real>>& I) {
    I.assign(static_cast<Size>(La + 1),
             std::vector<Real>(static_cast<Size>(Lb + 1), 0.0));

    // Base case
    I[0][0] = 1.0;

    // Build up first index: I(i+1, 0) for j=0
    // I(i+1, 0) = PA_eff * I(i, 0) + i * B00 * I(i-1, 0)
    for (int i = 0; i < La; ++i) {
        I[i + 1][0] = PA_eff * I[i][0];
        if (i > 0) {
            I[i + 1][0] += static_cast<Real>(i) * B00 * I[i - 1][0];
        }
    }

    // Build up second index: I(i, j+1) for each j, all i
    // I(i, j+1) = PB_eff * I(i, j) + B00 * [i * I(i-1, j) + j * I(i, j-1)]
    for (int j = 0; j < Lb; ++j) {
        for (int i = 0; i <= La; ++i) {
            Real val = PB_eff * I[i][j];
            if (i > 0) {
                val += static_cast<Real>(i) * B00 * I[i - 1][j];
            }
            if (j > 0) {
                val += static_cast<Real>(j) * B00 * I[i][j - 1];
            }
            I[i][j + 1] = val;
        }
    }
}

}  // anonymous namespace

void compute_nuclear(const Shell& shell_a, const Shell& shell_b,
                     const PointChargeParams& charges,
                     NuclearBuffer& buffer) {
    const int La = shell_a.angular_momentum();
    const int Lb = shell_b.angular_momentum();
    const int na = shell_a.n_functions();
    const int nb = shell_b.n_functions();

    buffer.resize(na, nb);
    buffer.clear();

    // Early return if no charge centers
    const Size n_centers = charges.n_centers();
    if (n_centers == 0) {
        return;
    }

    const auto& A = shell_a.center();
    const auto& B = shell_b.center();

    // Generate Cartesian index tuples for each shell
    const auto indices_a = math::generate_cartesian_indices(La);
    const auto indices_b = math::generate_cartesian_indices(Lb);

    // Precompute normalization correction factors for each component
    std::vector<Real> corr_a(na);
    for (int idx = 0; idx < na; ++idx) {
        const auto& [lx, ly, lz] = indices_a[idx];
        corr_a[idx] = norm_correction(lx, ly, lz);
    }

    std::vector<Real> corr_b(nb);
    for (int idx = 0; idx < nb; ++idx) {
        const auto& [lx, ly, lz] = indices_b[idx];
        corr_b[idx] = norm_correction(lx, ly, lz);
    }

    // Number of Rys quadrature roots for nuclear attraction
    const int n_rys_roots = (La + Lb) / 2 + 1;

    // Temporary storage for Rys roots and weights
    std::vector<double> roots(n_rys_roots);
    std::vector<double> weights(n_rys_roots);

    // Temporary storage for 1D recursion tables
    std::vector<std::vector<Real>> Ix, Iy, Iz;

    // Loop over primitive pairs
    const auto exponents_a = shell_a.exponents();
    const auto exponents_b = shell_b.exponents();
    const auto coefficients_a = shell_a.coefficients();
    const auto coefficients_b = shell_b.coefficients();

    for (Size p = 0; p < shell_a.n_primitives(); ++p) {
        const Real alpha = exponents_a[p];
        const Real ca = coefficients_a[p];

        for (Size q = 0; q < shell_b.n_primitives(); ++q) {
            const Real beta = exponents_b[q];
            const Real cb = coefficients_b[q];

            // Compute Gaussian product
            const auto gp = math::compute_gaussian_product(alpha, A, beta, B);
            const Real zeta = gp.zeta;

            // Nuclear attraction prefactor: (2*pi/zeta) * K_AB
            // Note: NOT (pi/zeta)^(3/2) as in overlap; the 1/r Coulomb
            // operator replaces one dimension of the Gaussian integral
            // with the Rys quadrature.
            const Real prefactor = (2.0 * constants::PI / zeta) * gp.K_AB;

            // Base half-inverse exponent (will be scaled by (1-u) per root)
            const Real B00_base = 0.5 / zeta;

            // Distances from product center to shell centers
            const Real PA_x = gp.P.x - A.x;
            const Real PA_y = gp.P.y - A.y;
            const Real PA_z = gp.P.z - A.z;
            const Real PB_x = gp.P.x - B.x;
            const Real PB_y = gp.P.y - B.y;
            const Real PB_z = gp.P.z - B.z;

            // Loop over nuclear centers
            for (Size c = 0; c < n_centers; ++c) {
                const Real Z_C = charges.charge[c];

                // Skip zero charges
                if (Z_C == 0.0) {
                    continue;
                }

                // Distance from product center P to nuclear center C
                const Real PC_x = gp.P.x - charges.x[c];
                const Real PC_y = gp.P.y - charges.y[c];
                const Real PC_z = gp.P.z - charges.z[c];

                // Boys function argument: T = zeta * |P - C|^2
                const Real T = zeta * (PC_x * PC_x + PC_y * PC_y + PC_z * PC_z);

                // Get Rys quadrature roots and weights
                math::rys_compute(n_rys_roots, T, roots.data(), weights.data());

                // Loop over Rys quadrature points
                for (int r = 0; r < n_rys_roots; ++r) {
                    const Real u = roots[r];       // u = t^2 (squared Rys root)
                    const Real w = weights[r];     // Rys weight

                    // Root-dependent half-inverse exponent
                    const Real B00 = B00_base * (1.0 - u);

                    // Effective displacements (Rys-modified)
                    const Real PA_x_eff = PA_x - u * PC_x;
                    const Real PA_y_eff = PA_y - u * PC_y;
                    const Real PA_z_eff = PA_z - u * PC_z;
                    const Real PB_x_eff = PB_x - u * PC_x;
                    const Real PB_y_eff = PB_y - u * PC_y;
                    const Real PB_z_eff = PB_z - u * PC_z;

                    // Build 1D Rys recursion tables for each Cartesian direction
                    build_1d_rys(La, Lb, PA_x_eff, PB_x_eff, B00, Ix);
                    build_1d_rys(La, Lb, PA_y_eff, PB_y_eff, B00, Iy);
                    build_1d_rys(La, Lb, PA_z_eff, PB_z_eff, B00, Iz);

                    // Combined coefficient: -Z_C * ca * cb * prefactor * w
                    const Real coeff = -Z_C * ca * cb * prefactor * w;

                    // Accumulate contributions for all Cartesian component pairs
                    for (int a_idx = 0; a_idx < na; ++a_idx) {
                        const auto& [lx_a, ly_a, lz_a] = indices_a[a_idx];

                        for (int b_idx = 0; b_idx < nb; ++b_idx) {
                            const auto& [lx_b, ly_b, lz_b] = indices_b[b_idx];

                            const Real val = Ix[lx_a][lx_b] *
                                             Iy[ly_a][ly_b] *
                                             Iz[lz_a][lz_b];

                            buffer(a_idx, b_idx) += coeff * val;
                        }
                    }
                }
            }
        }
    }

    // Apply normalization correction factors for each Cartesian component pair
    for (int a_idx = 0; a_idx < na; ++a_idx) {
        for (int b_idx = 0; b_idx < nb; ++b_idx) {
            buffer(a_idx, b_idx) *= corr_a[a_idx] * corr_b[b_idx];
        }
    }
}

}  // namespace libaccint::kernels
