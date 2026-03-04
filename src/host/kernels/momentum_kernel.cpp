// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file momentum_kernel.cpp
/// @brief CPU implementation of linear and angular momentum integrals
///
/// Linear momentum: <a|d/dx|b> = 2β * S(a, b+1_x) - b_x * S(a, b-1_x)
///   where S is the unnormalized overlap table.
///
/// Angular momentum: L_z = (x-Ox)*d/dy - (y-Oy)*d/dx  (and cyclic)
///   combines position-weighted and derivative integrals.

#include <libaccint/kernels/momentum_kernel.hpp>
#include <libaccint/kernels/norm_correction.hpp>
#include <libaccint/math/gaussian_product.hpp>
#include <libaccint/math/normalization.hpp>
#include <libaccint/math/cartesian_indices.hpp>
#include <libaccint/utils/constants.hpp>

#include <cmath>
#include <vector>

namespace libaccint::kernels {

namespace {

/// Build 1D overlap recursion table (extended range)
void build_1d_overlap_ext(int La_max, int Lb_max,
                          Real XPA, Real XPB, Real one_over_2zeta,
                          std::vector<std::vector<Real>>& I) {
    I.assign(static_cast<Size>(La_max + 1),
             std::vector<Real>(static_cast<Size>(Lb_max + 1), 0.0));

    I[0][0] = 1.0;

    for (int i = 0; i < La_max; ++i) {
        I[i + 1][0] = XPA * I[i][0];
        if (i > 0) I[i + 1][0] += static_cast<Real>(i) * one_over_2zeta * I[i - 1][0];
    }

    for (int j = 0; j < Lb_max; ++j) {
        for (int i = 0; i <= La_max; ++i) {
            Real val = XPB * I[i][j];
            if (i > 0) val += static_cast<Real>(i) * one_over_2zeta * I[i - 1][j];
            if (j > 0) val += static_cast<Real>(j) * one_over_2zeta * I[i][j - 1];
            I[i][j + 1] = val;
        }
    }
}

/// 3D table type: T[i][j][e]
using Table3D = std::vector<std::vector<std::vector<Real>>>;

/// Build 1D multipole table T[i][j][e] for position-weighted integrals
Table3D build_1d_multipole(int La_max, int Lb_max, int rank_max,
                           Real XPA, Real XPB, Real XPO, Real one_over_2zeta) {
    Table3D T(La_max + 1, std::vector<std::vector<Real>>(
        Lb_max + 1, std::vector<Real>(rank_max + 1, 0.0)));

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

}  // anonymous namespace

void compute_linear_momentum(const Shell& shell_a, const Shell& shell_b,
                             MultiComponentBuffer& buffer) {
    const int La = shell_a.angular_momentum();
    const int Lb = shell_b.angular_momentum();
    const int na = shell_a.n_functions();
    const int nb = shell_b.n_functions();

    buffer.resize(na, nb);

    const auto& A = shell_a.center();
    const auto& B = shell_b.center();

    const auto indices_a = math::generate_cartesian_indices(La);
    const auto indices_b = math::generate_cartesian_indices(Lb);

    std::vector<Real> corr_a(na);
    for (int idx = 0; idx < na; ++idx) {
        const auto& [lx, ly, lz] = indices_a[idx];
        corr_a[idx] = norm_correction(lx, ly, lz);
    }

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

            // Overlap tables extended to Lb+1 for derivative
            std::vector<std::vector<Real>> Ix, Iy, Iz;
            build_1d_overlap_ext(La, Lb + 1, XPA[0], XPB[0], one_over_2zeta, Ix);
            build_1d_overlap_ext(La, Lb + 1, XPA[1], XPB[1], one_over_2zeta, Iy);
            build_1d_overlap_ext(La, Lb + 1, XPA[2], XPB[2], one_over_2zeta, Iz);

            for (int a_idx = 0; a_idx < na; ++a_idx) {
                const auto& [ax, ay, az] = indices_a[a_idx];

                for (int b_idx = 0; b_idx < nb; ++b_idx) {
                    const auto& [bx, by, bz] = indices_b[b_idx];

                    Real overlap_x = Ix[ax][bx];
                    Real overlap_y = Iy[ay][by];
                    Real overlap_z = Iz[az][bz];

                    // Use the ORIGINAL ket normalization correction
                    Real corr_b_orig = norm_correction(bx, by, bz);
                    Real ca_corr = corr_a[a_idx] * corr_b_orig * prim_coeff;

                    // x-derivative: <a|d/dx|b> = 2β*I_x[ax][bx+1] - bx*I_x[ax][bx-1]
                    Real px = 2.0 * beta * Ix[ax][bx + 1] * overlap_y * overlap_z;
                    if (bx > 0) px -= static_cast<Real>(bx) * Ix[ax][bx - 1] * overlap_y * overlap_z;
                    buffer(0, a_idx, b_idx) += px * ca_corr;

                    // y-derivative
                    Real py = 2.0 * beta * overlap_x * Iy[ay][by + 1] * overlap_z;
                    if (by > 0) py -= static_cast<Real>(by) * overlap_x * Iy[ay][by - 1] * overlap_z;
                    buffer(1, a_idx, b_idx) += py * ca_corr;

                    // z-derivative
                    Real pz = 2.0 * beta * overlap_x * overlap_y * Iz[az][bz + 1];
                    if (bz > 0) pz -= static_cast<Real>(bz) * overlap_x * overlap_y * Iz[az][bz - 1];
                    buffer(2, a_idx, b_idx) += pz * ca_corr;
                }
            }
        }
    }
}

void compute_angular_momentum(const Shell& shell_a, const Shell& shell_b,
                              const std::array<Real, 3>& origin,
                              MultiComponentBuffer& buffer) {
    const int La = shell_a.angular_momentum();
    const int Lb = shell_b.angular_momentum();
    const int na = shell_a.n_functions();
    const int nb = shell_b.n_functions();

    buffer.resize(na, nb);

    const auto& A = shell_a.center();
    const auto& B = shell_b.center();

    const auto indices_a = math::generate_cartesian_indices(La);
    const auto indices_b = math::generate_cartesian_indices(Lb);

    std::vector<Real> corr_a(na);
    for (int idx = 0; idx < na; ++idx) {
        const auto& [lx, ly, lz] = indices_a[idx];
        corr_a[idx] = norm_correction(lx, ly, lz);
    }

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

            // Extended overlap tables for derivatives
            std::vector<std::vector<Real>> Ix, Iy, Iz;
            build_1d_overlap_ext(La, Lb + 1, XPA[0], XPB[0], one_over_2zeta, Ix);
            build_1d_overlap_ext(La, Lb + 1, XPA[1], XPB[1], one_over_2zeta, Iy);
            build_1d_overlap_ext(La, Lb + 1, XPA[2], XPB[2], one_over_2zeta, Iz);

            // Multipole tables for position-weighted integrals (rank 1)
            auto Tx = build_1d_multipole(La, Lb + 1, 1, XPA[0], XPB[0], XPO[0], one_over_2zeta);
            auto Ty = build_1d_multipole(La, Lb + 1, 1, XPA[1], XPB[1], XPO[1], one_over_2zeta);
            auto Tz = build_1d_multipole(La, Lb + 1, 1, XPA[2], XPB[2], XPO[2], one_over_2zeta);

            for (int a_idx = 0; a_idx < na; ++a_idx) {
                const auto& [ax, ay, az] = indices_a[a_idx];

                for (int b_idx = 0; b_idx < nb; ++b_idx) {
                    const auto& [bx, by, bz] = indices_b[b_idx];

                    Real ca_corr = corr_a[a_idx] * prim_coeff;

                    // Use the ORIGINAL ket normalization correction for all components
                    Real corr_b_orig = norm_correction(bx, by, bz);

                    // Angular momentum: Lα = (r - O) × ∇ (real part, -i factored out)
                    //
                    // For L_x = (y-Oy)*d/dz - (z-Oz)*d/dy:
                    //   term1 = T_y[ay][by][1] * deriv_z * I_x[ax][bx]
                    //   term2 = -T_z[az][bz][1] * deriv_y * I_x[ax][bx]
                    //
                    // Note: T[i][j][1] is the position-weighted 1D integral <i|(r_d-O_d)|j>
                    // and deriv_d is the derivative integral in direction d

                    // L_x = (y-Oy)*d/dz - (z-Oz)*d/dy
                    {
                        Real deriv_z = 2.0 * beta * Iz[az][bz + 1] - ((bz > 0) ? static_cast<Real>(bz) * Iz[az][bz - 1] : 0.0);
                        Real t1 = Ix[ax][bx] * Ty[ay][by][1] * deriv_z;

                        Real deriv_y = 2.0 * beta * Iy[ay][by + 1] - ((by > 0) ? static_cast<Real>(by) * Iy[ay][by - 1] : 0.0);
                        Real t2 = Ix[ax][bx] * deriv_y * Tz[az][bz][1];

                        buffer(0, a_idx, b_idx) += (t1 - t2) * ca_corr * corr_b_orig;
                    }

                    // L_y = (z-Oz)*d/dx - (x-Ox)*d/dz
                    {
                        Real deriv_x = 2.0 * beta * Ix[ax][bx + 1] - ((bx > 0) ? static_cast<Real>(bx) * Ix[ax][bx - 1] : 0.0);
                        Real t1 = deriv_x * Iy[ay][by] * Tz[az][bz][1];

                        Real deriv_z = 2.0 * beta * Iz[az][bz + 1] - ((bz > 0) ? static_cast<Real>(bz) * Iz[az][bz - 1] : 0.0);
                        Real t2 = Tx[ax][bx][1] * Iy[ay][by] * deriv_z;

                        buffer(1, a_idx, b_idx) += (t1 - t2) * ca_corr * corr_b_orig;
                    }

                    // L_z = (x-Ox)*d/dy - (y-Oy)*d/dx
                    {
                        Real deriv_y = 2.0 * beta * Iy[ay][by + 1] - ((by > 0) ? static_cast<Real>(by) * Iy[ay][by - 1] : 0.0);
                        Real t1 = Tx[ax][bx][1] * deriv_y * Iz[az][bz];

                        Real deriv_x = 2.0 * beta * Ix[ax][bx + 1] - ((bx > 0) ? static_cast<Real>(bx) * Ix[ax][bx - 1] : 0.0);
                        Real t2 = deriv_x * Ty[ay][by][1] * Iz[az][bz];

                        buffer(2, a_idx, b_idx) += (t1 - t2) * ca_corr * corr_b_orig;
                    }
                }
            }
        }
    }
}

}  // namespace libaccint::kernels
