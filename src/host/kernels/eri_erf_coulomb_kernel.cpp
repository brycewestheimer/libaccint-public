// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file eri_erf_coulomb_kernel.cpp
/// @brief erf-attenuated Coulomb ERI kernel using modified Rys quadrature
///
/// This is structurally identical to the standard ERI kernel, but uses the
/// erf-modified Boys function with:
///   T_eff = T * omega^2 / (omega^2 + 1)
///   scale = (omega^2 / (omega^2 + 1))^{n+1/2}
///
/// The Rys roots and weights are computed for the modified T_eff, and the
/// resulting integrals are scaled appropriately.

#include <libaccint/kernels/eri_erf_coulomb_kernel.hpp>
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

/// @brief Build 2D Rys recursion table for one Cartesian direction
void build_2d_rys_erf(int La, int Lb, int Lc, int Ld,
                       Real PA_eff, Real QC_eff, Real AB, Real CD,
                       Real B10, Real B01, Real B00,
                       std::vector<std::vector<std::vector<std::vector<Real>>>>& I) {
    const int dim_a = La + Lb + 1;
    const int dim_b = Lb + 1;
    const int dim_c = Lc + Ld + 1;
    const int dim_d = Ld + 1;

    I.assign(dim_a, std::vector<std::vector<std::vector<Real>>>(
        dim_b, std::vector<std::vector<Real>>(
            dim_c, std::vector<Real>(dim_d, 0.0))));

    // Base case
    I[0][0][0][0] = 1.0;

    // Build (a, 0 | c, 0) via VRR
    for (int a = 0; a < La + Lb; ++a) {
        I[a + 1][0][0][0] = PA_eff * I[a][0][0][0];
        if (a > 0) {
            I[a + 1][0][0][0] += static_cast<Real>(a) * B10 * I[a - 1][0][0][0];
        }
    }

    for (int c = 0; c < Lc + Ld; ++c) {
        for (int a = 0; a <= La + Lb; ++a) {
            I[a][0][c + 1][0] = QC_eff * I[a][0][c][0];
            if (c > 0) {
                I[a][0][c + 1][0] += static_cast<Real>(c) * B01 * I[a][0][c - 1][0];
            }
            if (a > 0) {
                I[a][0][c + 1][0] += static_cast<Real>(a) * B00 * I[a - 1][0][c][0];
            }
        }
    }

    // HRR: transfer angular momentum to B
    for (int b = 0; b < Lb; ++b) {
        for (int a = 0; a <= La + Lb - b - 1; ++a) {
            for (int c = 0; c <= Lc + Ld; ++c) {
                for (int d = 0; d < dim_d; ++d) {
                    I[a][b + 1][c][d] = I[a + 1][b][c][d] + AB * I[a][b][c][d];
                }
            }
        }
    }

    // HRR: transfer angular momentum to D
    for (int d = 0; d < Ld; ++d) {
        for (int a = 0; a <= La; ++a) {
            for (int b = 0; b <= Lb; ++b) {
                for (int c = 0; c <= Lc + Ld - d - 1; ++c) {
                    I[a][b][c][d + 1] = I[a][b][c + 1][d] + CD * I[a][b][c][d];
                }
            }
        }
    }
}

}  // anonymous namespace

void compute_eri_erf_coulomb(const Shell& shell_a, const Shell& shell_b,
                              const Shell& shell_c, const Shell& shell_d,
                              Real omega,
                              TwoElectronBuffer<0>& buffer) {
    const int La = shell_a.angular_momentum();
    const int Lb = shell_b.angular_momentum();
    const int Lc = shell_c.angular_momentum();
    const int Ld = shell_d.angular_momentum();
    const int na = shell_a.n_functions();
    const int nb = shell_b.n_functions();
    const int nc = shell_c.n_functions();
    const int nd = shell_d.n_functions();

    buffer.resize(na, nb, nc, nd);
    buffer.clear();

    // Handle omega = 0 case: all integrals are zero
    if (omega <= 0.0) {
        return;
    }

    // Precompute omega-dependent factors
    const Real omega2 = omega * omega;
    const Real omega2_ratio = omega2 / (omega2 + 1.0);
    const Real sqrt_omega2_ratio = std::sqrt(omega2_ratio);

    const auto& A = shell_a.center();
    const auto& B = shell_b.center();
    const auto& C = shell_c.center();
    const auto& D = shell_d.center();

    const Real AB_x = A.x - B.x;
    const Real AB_y = A.y - B.y;
    const Real AB_z = A.z - B.z;
    const Real CD_x = C.x - D.x;
    const Real CD_y = C.y - D.y;
    const Real CD_z = C.z - D.z;

    const auto indices_a = math::generate_cartesian_indices(La);
    const auto indices_b = math::generate_cartesian_indices(Lb);
    const auto indices_c = math::generate_cartesian_indices(Lc);
    const auto indices_d = math::generate_cartesian_indices(Ld);

    std::vector<Real> corr_a(na), corr_b(nb), corr_c(nc), corr_d(nd);
    for (int i = 0; i < na; ++i) {
        const auto& [lx, ly, lz] = indices_a[i];
        corr_a[i] = norm_correction(lx, ly, lz);
    }
    for (int i = 0; i < nb; ++i) {
        const auto& [lx, ly, lz] = indices_b[i];
        corr_b[i] = norm_correction(lx, ly, lz);
    }
    for (int i = 0; i < nc; ++i) {
        const auto& [lx, ly, lz] = indices_c[i];
        corr_c[i] = norm_correction(lx, ly, lz);
    }
    for (int i = 0; i < nd; ++i) {
        const auto& [lx, ly, lz] = indices_d[i];
        corr_d[i] = norm_correction(lx, ly, lz);
    }

    const int n_rys_roots = (La + Lb + Lc + Ld) / 2 + 1;
    std::vector<double> roots(n_rys_roots);
    std::vector<double> weights(n_rys_roots);

    std::vector<std::vector<std::vector<std::vector<Real>>>> Ix, Iy, Iz;

    const auto exp_a = shell_a.exponents();
    const auto exp_b = shell_b.exponents();
    const auto exp_c = shell_c.exponents();
    const auto exp_d = shell_d.exponents();
    const auto coeff_a = shell_a.coefficients();
    const auto coeff_b = shell_b.coefficients();
    const auto coeff_c = shell_c.coefficients();
    const auto coeff_d = shell_d.coefficients();

    // Four-fold contraction loop
    for (Size p = 0; p < shell_a.n_primitives(); ++p) {
        const Real alpha = exp_a[p];
        const Real ca = coeff_a[p];

        for (Size q = 0; q < shell_b.n_primitives(); ++q) {
            const Real beta = exp_b[q];
            const Real cb = coeff_b[q];

            const auto gp_bra = math::compute_gaussian_product(alpha, A, beta, B);
            const Real zeta = gp_bra.zeta;
            const auto& P = gp_bra.P;

            for (Size r = 0; r < shell_c.n_primitives(); ++r) {
                const Real gamma = exp_c[r];
                const Real cc = coeff_c[r];

                for (Size s = 0; s < shell_d.n_primitives(); ++s) {
                    const Real delta = exp_d[s];
                    const Real cd = coeff_d[s];

                    const auto gp_ket = math::compute_gaussian_product(gamma, C, delta, D);
                    const Real eta = gp_ket.zeta;
                    const auto& Q = gp_ket.P;

                    const Real rho = zeta * eta / (zeta + eta);

                    const Real PQ_x = P.x - Q.x;
                    const Real PQ_y = P.y - Q.y;
                    const Real PQ_z = P.z - Q.z;
                    const Real PQ2 = PQ_x * PQ_x + PQ_y * PQ_y + PQ_z * PQ_z;

                    // Standard T
                    const Real T = rho * PQ2;

                    // Modified T for erf operator
                    const Real T_eff = T * omega2_ratio;

                    // Prefactor: 2 * pi^(5/2) / (zeta * eta * sqrt(zeta + eta)) * K_AB * K_CD
                    const Real pi_52 = constants::PI * constants::PI * std::sqrt(constants::PI);
                    const Real prefactor = 2.0 * pi_52 /
                        (zeta * eta * std::sqrt(zeta + eta)) *
                        gp_bra.K_AB * gp_ket.K_AB;

                    // Scaling for erf-attenuated operator
                    // The overall scaling is (omega^2/(omega^2+1))^{1/2} for Rys quadrature
                    // because the scaling is built into the modified weights
                    const Real erf_scale = sqrt_omega2_ratio;

                    const Real coeff = ca * cb * cc * cd * prefactor * erf_scale;

                    // Get Rys roots and weights for the MODIFIED T_eff
                    math::rys_compute(n_rys_roots, T_eff, roots.data(), weights.data());

                    // Modified recursion coefficients for range-separated operator
                    // The Rys quadrature with T_eff automatically handles the modification

                    for (int root = 0; root < n_rys_roots; ++root) {
                        const Real u = roots[root];
                        const Real w = weights[root];

                        // Recursion coefficients using modified u values
                        // The Rys roots for T_eff already account for the range separation
                        const Real rho_over_zeta = rho / zeta;
                        const Real rho_over_eta = rho / eta;

                        // For erf-attenuated operator, we use the effective u scaled by omega2_ratio
                        const Real u_eff = u * omega2_ratio;

                        const Real B10 = 0.5 / zeta * (1.0 - rho_over_zeta * u_eff);
                        const Real B01 = 0.5 / eta * (1.0 - rho_over_eta * u_eff);
                        const Real B00 = 0.5 / (zeta + eta) * u_eff;

                        const Real PA_x_eff = (P.x - A.x) - rho_over_zeta * u_eff * PQ_x;
                        const Real PA_y_eff = (P.y - A.y) - rho_over_zeta * u_eff * PQ_y;
                        const Real PA_z_eff = (P.z - A.z) - rho_over_zeta * u_eff * PQ_z;

                        const Real QC_x_eff = (Q.x - C.x) + rho_over_eta * u_eff * PQ_x;
                        const Real QC_y_eff = (Q.y - C.y) + rho_over_eta * u_eff * PQ_y;
                        const Real QC_z_eff = (Q.z - C.z) + rho_over_eta * u_eff * PQ_z;

                        build_2d_rys_erf(La, Lb, Lc, Ld,
                                         PA_x_eff, QC_x_eff, AB_x, CD_x,
                                         B10, B01, B00, Ix);
                        build_2d_rys_erf(La, Lb, Lc, Ld,
                                         PA_y_eff, QC_y_eff, AB_y, CD_y,
                                         B10, B01, B00, Iy);
                        build_2d_rys_erf(La, Lb, Lc, Ld,
                                         PA_z_eff, QC_z_eff, AB_z, CD_z,
                                         B10, B01, B00, Iz);

                        const Real wcoeff = coeff * w;

                        for (int ia = 0; ia < na; ++ia) {
                            const auto& [ax, ay, az] = indices_a[ia];
                            for (int ib = 0; ib < nb; ++ib) {
                                const auto& [bx, by, bz] = indices_b[ib];
                                for (int ic = 0; ic < nc; ++ic) {
                                    const auto& [cx, cy, cz] = indices_c[ic];
                                    for (int id = 0; id < nd; ++id) {
                                        const auto& [dx, dy, dz] = indices_d[id];

                                        const Real val =
                                            Ix[ax][bx][cx][dx] *
                                            Iy[ay][by][cy][dy] *
                                            Iz[az][bz][cz][dz];

                                        buffer(ia, ib, ic, id) += wcoeff * val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply normalization correction factors
    for (int ia = 0; ia < na; ++ia) {
        for (int ib = 0; ib < nb; ++ib) {
            for (int ic = 0; ic < nc; ++ic) {
                for (int id = 0; id < nd; ++id) {
                    buffer(ia, ib, ic, id) *=
                        corr_a[ia] * corr_b[ib] * corr_c[ic] * corr_d[id];
                }
            }
        }
    }
}

void compute_eri_erf_coulomb(const Shell& shell_a, const Shell& shell_b,
                              const Shell& shell_c, const Shell& shell_d,
                              const RangeSeparatedParams& params,
                              TwoElectronBuffer<0>& buffer) {
    compute_eri_erf_coulomb(shell_a, shell_b, shell_c, shell_d,
                             params.omega, buffer);
}

}  // namespace libaccint::kernels
