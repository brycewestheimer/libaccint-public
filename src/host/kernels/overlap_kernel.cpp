// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file overlap_kernel.cpp
/// @brief Overlap integral kernel implementation using Obara-Saika recursion

#include <libaccint/kernels/overlap_kernel.hpp>
#include <libaccint/kernels/norm_correction.hpp>
#include <libaccint/math/gaussian_product.hpp>
#include <libaccint/math/normalization.hpp>
#include <libaccint/math/cartesian_indices.hpp>
#include <libaccint/utils/constants.hpp>

#include <cmath>
#include <vector>

namespace libaccint::kernels {

namespace {

/// @brief Build 1D overlap recursion table for one Cartesian direction
///
/// Uses the Obara-Saika recursion with factored-out prefactor:
///   I(0,0) = 1.0  (the 3D prefactor (pi/zeta)^(3/2) * K_AB is applied separately)
///
/// Recursion to increment first index:
///   I(i+1, j) = XPA * I(i, j) + 1/(2*zeta) * [i * I(i-1, j) + j * I(i, j-1)]
///
/// Recursion to increment second index:
///   I(i, j+1) = XPB * I(i, j) + 1/(2*zeta) * [i * I(i-1, j) + j * I(i, j-1)]
///
/// @param La Maximum angular momentum for first center
/// @param Lb Maximum angular momentum for second center
/// @param XPA Distance from product center P to center A (P_d - A_d)
/// @param XPB Distance from product center P to center B (P_d - B_d)
/// @param one_over_2zeta 1 / (2 * zeta)
/// @param[out] I Recursion table of size (La+1) x (Lb+1)
void build_1d_overlap(int La, int Lb,
                      Real XPA, Real XPB, Real one_over_2zeta,
                      std::vector<std::vector<Real>>& I) {
    I.assign(static_cast<Size>(La + 1),
             std::vector<Real>(static_cast<Size>(Lb + 1), 0.0));

    // Base case
    I[0][0] = 1.0;

    // Build up first index: I(i+1, 0) for j=0
    // I(i+1, 0) = XPA * I(i, 0) + i/(2*zeta) * I(i-1, 0)
    for (int i = 0; i < La; ++i) {
        I[i + 1][0] = XPA * I[i][0];
        if (i > 0) {
            I[i + 1][0] += static_cast<Real>(i) * one_over_2zeta * I[i - 1][0];
        }
    }

    // Build up second index: I(i, j+1) for each j, all i
    // I(i, j+1) = XPB * I(i, j) + 1/(2*zeta) * [i * I(i-1, j) + j * I(i, j-1)]
    for (int j = 0; j < Lb; ++j) {
        for (int i = 0; i <= La; ++i) {
            Real val = XPB * I[i][j];
            if (i > 0) {
                val += static_cast<Real>(i) * one_over_2zeta * I[i - 1][j];
            }
            if (j > 0) {
                val += static_cast<Real>(j) * one_over_2zeta * I[i][j - 1];
            }
            I[i][j + 1] = val;
        }
    }
}

/// @brief Compute normalization correction factor for a Cartesian component
///
}  // anonymous namespace

void compute_overlap(const Shell& shell_a, const Shell& shell_b,
                     OverlapBuffer& buffer) {
    const int La = shell_a.angular_momentum();
    const int Lb = shell_b.angular_momentum();
    const int na = shell_a.n_functions();
    const int nb = shell_b.n_functions();

    buffer.resize(na, nb);
    buffer.clear();

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
            const Real one_over_2zeta = 0.5 / zeta;

            // 3D prefactor: (pi/zeta)^(3/2) * K_AB
            const Real prefactor = std::pow(constants::PI / zeta, 1.5) * gp.K_AB;

            // Distances from product center to shell centers
            const Real XPA_x = gp.P.x - A.x;
            const Real XPA_y = gp.P.y - A.y;
            const Real XPA_z = gp.P.z - A.z;
            const Real XPB_x = gp.P.x - B.x;
            const Real XPB_y = gp.P.y - B.y;
            const Real XPB_z = gp.P.z - B.z;

            // Build 1D overlap recursion tables for each Cartesian direction
            build_1d_overlap(La, Lb, XPA_x, XPB_x, one_over_2zeta, Ix);
            build_1d_overlap(La, Lb, XPA_y, XPB_y, one_over_2zeta, Iy);
            build_1d_overlap(La, Lb, XPA_z, XPB_z, one_over_2zeta, Iz);

            // Combined primitive coefficient with prefactor
            const Real prim_coeff = ca * cb * prefactor;

            // Accumulate contributions for all Cartesian component pairs
            for (int a_idx = 0; a_idx < na; ++a_idx) {
                const auto& [lx_a, ly_a, lz_a] = indices_a[a_idx];

                for (int b_idx = 0; b_idx < nb; ++b_idx) {
                    const auto& [lx_b, ly_b, lz_b] = indices_b[b_idx];

                    const Real val = Ix[lx_a][lx_b] *
                                     Iy[ly_a][ly_b] *
                                     Iz[lz_a][lz_b];

                    buffer(a_idx, b_idx) += prim_coeff * val;
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

void compute_overlap(const Shell& shell_a, const Shell& shell_b,
                     const PrimitivePairData& pair_data,
                     Size shell_i, Size shell_j,
                     OverlapBuffer& buffer) {
    const int La = shell_a.angular_momentum();
    const int Lb = shell_b.angular_momentum();
    const int na = shell_a.n_functions();
    const int nb = shell_b.n_functions();

    buffer.resize(na, nb);
    buffer.clear();

    // Generate Cartesian index tuples for each shell
    const auto indices_a = math::generate_cartesian_indices(La);
    const auto indices_b = math::generate_cartesian_indices(Lb);

    // Precompute normalization correction factors
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

    // Temporary storage for 1D recursion tables
    std::vector<std::vector<Real>> Ix, Iy, Iz;

    const auto Ka = pair_data.K_a;
    const auto Kb = pair_data.K_b;

    // Loop over primitive pairs using pre-computed data
    for (Size p = 0; p < Ka; ++p) {
        for (Size q = 0; q < Kb; ++q) {
            const Size idx = pair_data.pair_index(shell_i, shell_j, p, q);

            const Real zeta = pair_data.zeta[idx];
            const Real one_over_2zeta = pair_data.one_over_2zeta[idx];

            // 3D prefactor: (pi/zeta)^(3/2) * K_AB
            const Real prefactor = std::pow(constants::PI / zeta, 1.5) *
                                   pair_data.K_AB[idx];

            // Build 1D overlap recursion tables using pre-computed displacements
            build_1d_overlap(La, Lb,
                             pair_data.PA_x[idx], pair_data.PB_x[idx],
                             one_over_2zeta, Ix);
            build_1d_overlap(La, Lb,
                             pair_data.PA_y[idx], pair_data.PB_y[idx],
                             one_over_2zeta, Iy);
            build_1d_overlap(La, Lb,
                             pair_data.PA_z[idx], pair_data.PB_z[idx],
                             one_over_2zeta, Iz);

            // Combined primitive coefficient with prefactor
            const Real prim_coeff = pair_data.coeff_product[idx] * prefactor;

            // Accumulate contributions for all Cartesian component pairs
            for (int a_idx = 0; a_idx < na; ++a_idx) {
                const auto& [lx_a, ly_a, lz_a] = indices_a[a_idx];

                for (int b_idx = 0; b_idx < nb; ++b_idx) {
                    const auto& [lx_b, ly_b, lz_b] = indices_b[b_idx];

                    const Real val = Ix[lx_a][lx_b] *
                                     Iy[ly_a][ly_b] *
                                     Iz[lz_a][lz_b];

                    buffer(a_idx, b_idx) += prim_coeff * val;
                }
            }
        }
    }

    // Apply normalization correction factors
    for (int a_idx = 0; a_idx < na; ++a_idx) {
        for (int b_idx = 0; b_idx < nb; ++b_idx) {
            buffer(a_idx, b_idx) *= corr_a[a_idx] * corr_b[b_idx];
        }
    }
}

}  // namespace libaccint::kernels
