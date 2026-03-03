// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file multipole_kernel.cpp
/// @brief CPU implementation of electric multipole moment integrals
///
/// Uses extended Obara-Saika recursion to compute dipole, quadrupole, and octupole
/// integrals. The multipole integral <a|(r-O)^n|b> is computed by building 1D
/// tables T_d[i][j][e] where the e dimension represents the multipole power.

#include <libaccint/kernels/multipole_kernel.hpp>
#include <libaccint/kernels/norm_correction.hpp>
#include <libaccint/math/gaussian_product.hpp>
#include <libaccint/math/normalization.hpp>
#include <libaccint/math/cartesian_indices.hpp>
#include <libaccint/utils/constants.hpp>

#include <cmath>
#include <vector>

namespace libaccint::kernels {

namespace {

/// 3D table type: T[i][j][e]
using Table3D = std::vector<std::vector<std::vector<Real>>>;

/// @brief Build 1D multipole recursion table T_d[i][j][e]
///
/// T[i][j][0] = standard Obara-Saika overlap
/// T[i][j][e+1] = (P_d - O_d) * T[i][j][e]
///              + 1/(2z) * (i * T[i-1][j][e] + j * T[i][j-1][e] + e * T[i][j][e-1])
Table3D build_1d_multipole(int La_max, int Lb_max, int rank_max,
                           Real XPA, Real XPB, Real XPO, Real one_over_2zeta) {
    Table3D T(La_max + 1, std::vector<std::vector<Real>>(
        Lb_max + 1, std::vector<Real>(rank_max + 1, 0.0)));

    // e=0: standard overlap recursion
    T[0][0][0] = 1.0;
    for (int i = 0; i < La_max; ++i) {
        T[i + 1][0][0] = XPA * T[i][0][0];
        if (i > 0) T[i + 1][0][0] += static_cast<Real>(i) * one_over_2zeta * T[i - 1][0][0];
    }
    for (int j = 0; j < Lb_max; ++j) {
        for (int i = 0; i <= La_max; ++i) {
            Real val = XPB * T[i][j][0];
            if (i > 0) val += static_cast<Real>(i) * one_over_2zeta * T[i - 1][j][0];
            if (j > 0) val += static_cast<Real>(j) * one_over_2zeta * T[i][j - 1][0];
            T[i][j + 1][0] = val;
        }
    }

    // Higher e layers: multipole recursion
    for (int e = 0; e < rank_max; ++e) {
        for (int i = 0; i <= La_max; ++i) {
            for (int j = 0; j <= Lb_max; ++j) {
                Real val = XPO * T[i][j][e];
                if (i > 0) val += static_cast<Real>(i) * one_over_2zeta * T[i - 1][j][e];
                if (j > 0) val += static_cast<Real>(j) * one_over_2zeta * T[i][j - 1][e];
                if (e > 0) val += static_cast<Real>(e) * one_over_2zeta * T[i][j][e - 1];
                T[i][j][e + 1] = val;
            }
        }
    }

    return T;
}

/// @brief Generic multipole integral computation for rank 1/2/3
void compute_multipole_generic(
    const Shell& shell_a, const Shell& shell_b,
    const std::array<Real, 3>& origin,
    int rank,
    MultiComponentBuffer& buffer)
{
    const int La = shell_a.angular_momentum();
    const int Lb = shell_b.angular_momentum();
    const int na = shell_a.n_functions();
    const int nb = shell_b.n_functions();

    buffer.resize(na, nb);

    const auto& A = shell_a.center();
    const auto& B = shell_b.center();

    const auto indices_a = math::generate_cartesian_indices(La);
    const auto indices_b = math::generate_cartesian_indices(Lb);

    std::vector<Real> corr_a(na), corr_b(nb);
    for (int idx = 0; idx < na; ++idx) {
        const auto& [lx, ly, lz] = indices_a[idx];
        corr_a[idx] = norm_correction(lx, ly, lz);
    }
    for (int idx = 0; idx < nb; ++idx) {
        const auto& [lx, ly, lz] = indices_b[idx];
        corr_b[idx] = norm_correction(lx, ly, lz);
    }

    // Generate multipole component indices: all (ex,ey,ez) with ex+ey+ez = rank
    // Ordered: descending ex, then descending ey
    std::vector<std::array<int, 3>> multipole_indices;
    for (int ex = rank; ex >= 0; --ex) {
        for (int ey = rank - ex; ey >= 0; --ey) {
            int ez = rank - ex - ey;
            multipole_indices.push_back({ex, ey, ez});
        }
    }

    const Size n_comp = buffer.n_components();

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

            const auto gp = math::compute_gaussian_product(alpha, A, beta, B);
            const Real zeta = gp.zeta;
            const Real one_over_2zeta = 0.5 / zeta;
            const Real prefactor = std::pow(constants::PI / zeta, 1.5) * gp.K_AB;
            const Real prim_coeff = ca * cb * prefactor;

            const Real XPA[3] = {gp.P.x - A.x, gp.P.y - A.y, gp.P.z - A.z};
            const Real XPB[3] = {gp.P.x - B.x, gp.P.y - B.y, gp.P.z - B.z};
            const Real XPO[3] = {gp.P.x - origin[0], gp.P.y - origin[1], gp.P.z - origin[2]};

            auto Tx = build_1d_multipole(La, Lb, rank, XPA[0], XPB[0], XPO[0], one_over_2zeta);
            auto Ty = build_1d_multipole(La, Lb, rank, XPA[1], XPB[1], XPO[1], one_over_2zeta);
            auto Tz = build_1d_multipole(La, Lb, rank, XPA[2], XPB[2], XPO[2], one_over_2zeta);

            for (int a_idx = 0; a_idx < na; ++a_idx) {
                const auto& [lx_a, ly_a, lz_a] = indices_a[a_idx];

                for (int b_idx = 0; b_idx < nb; ++b_idx) {
                    const auto& [lx_b, ly_b, lz_b] = indices_b[b_idx];

                    for (Size comp = 0; comp < n_comp; ++comp) {
                        const auto& [ex, ey, ez] = multipole_indices[comp];
                        const Real val = Tx[lx_a][lx_b][ex] *
                                         Ty[ly_a][ly_b][ey] *
                                         Tz[lz_a][lz_b][ez];
                        buffer(comp, a_idx, b_idx) += prim_coeff * val *
                            corr_a[a_idx] * corr_b[b_idx];
                    }
                }
            }
        }
    }
}

}  // anonymous namespace

void compute_dipole(const Shell& shell_a, const Shell& shell_b,
                    const std::array<Real, 3>& origin,
                    MultiComponentBuffer& buffer) {
    compute_multipole_generic(shell_a, shell_b, origin, 1, buffer);
}

void compute_quadrupole(const Shell& shell_a, const Shell& shell_b,
                        const std::array<Real, 3>& origin,
                        MultiComponentBuffer& buffer) {
    compute_multipole_generic(shell_a, shell_b, origin, 2, buffer);
}

void compute_octupole(const Shell& shell_a, const Shell& shell_b,
                      const std::array<Real, 3>& origin,
                      MultiComponentBuffer& buffer) {
    compute_multipole_generic(shell_a, shell_b, origin, 3, buffer);
}

}  // namespace libaccint::kernels
