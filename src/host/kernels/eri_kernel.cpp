// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file eri_kernel.cpp
/// @brief Electron repulsion integral kernel using Rys quadrature with 2D recursion
///
/// Algorithm overview:
/// 1. For each primitive quartet (i,j,k,l):
///    a. Compute bra Gaussian product P = (alpha_i * A + alpha_j * B) / zeta
///    b. Compute ket Gaussian product Q = (alpha_k * C + alpha_l * D) / eta
///    c. Compute reduced exponent rho = zeta*eta/(zeta+eta) and T = rho*|P-Q|^2
///    d. Compute Rys roots and weights for n_roots = (La+Lb+Lc+Ld)/2 + 1
///    e. For each Rys root:
///       - Build 2D recursion tables Ix[i][j][k][l], Iy, Iz
///       - Accumulate Ix*Iy*Iz weighted by Rys weight and prefactor
/// 2. Multiply by contraction coefficients and normalization corrections
///
/// The 2D recursion follows the Rys polynomial approach:
///   - First build angular momentum on the bra side (VRR on centers A, B)
///   - Then build angular momentum on the ket side (VRR on centers C, D)
/// using effective displacements modified by the Rys root parameter.

#include <libaccint/kernels/eri_kernel.hpp>
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

/// @brief Build 2D Rys recursion table for one Cartesian direction (4-index)
///
/// Builds the table I[ia][ib][ic][id] for a single Cartesian direction at one
/// Rys quadrature root. The approach:
///
/// Step 1: Build (ia, 0 | ic, 0) via vertical recurrence on bra (ia) then ket (ic):
///   I(ia+1, 0, ic, 0) = PA_eff * I(ia, 0, ic, 0) + B00 * ia * I(ia-1, 0, ic, 0)
///                       + B01 * ic * I(ia, 0, ic-1, 0)
///   I(ia, 0, ic+1, 0) = QC_eff * I(ia, 0, ic, 0) + B10 * ia * I(ia-1, 0, ic, 0)
///                       + B00p * ic * I(ia, 0, ic-1, 0)
///
/// Step 2: Transfer angular momentum via HRR:
///   I(ia, ib+1, ic, id) = I(ia+1, ib, ic, id) + AB * I(ia, ib, ic, id)
///   I(ia, ib, ic, id+1) = I(ia, ib, ic+1, id) + CD * I(ia, ib, ic, id)
///
/// Parameters (for Rys root u = t^2):
///   B00  = 0.5 * rho / zeta * (1 - u) ... no, let me use the standard formulation:
///
/// Actually, the standard Rys 2D recursion parameters for the 4-center case:
///   rho = zeta * eta / (zeta + eta)
///   B00 = 0.5 / (zeta + eta) * (1-u)   -- coupling between bra and ket
///   B10 = (1-u) / (2*zeta)              -- within bra recursion
///   B01 = (1-u) / (2*eta)               -- within ket recursion (named B00p below)
///
///   PA_eff = (P-A) - u * rho/zeta * (P-Q)  -- effective displacement bra
///   QC_eff = (Q-C) - u * rho/eta * (P-Q)   -- effective displacement ket
///
/// Wait, more precisely for the standard Rys approach with squared roots u:
///   The mapping is that the Rys integral replaces the Boys function approach.
///   For each root u_r with weight w_r, the 1D integral factors as:
///
///   B10 = 0.5/zeta * (1 - rho/zeta * u)    -- half-inverse exponent, bra internal
///   B01 = 0.5/eta * (1 - rho/eta * u)      -- half-inverse exponent, ket internal
///   B00 = 0.5/(zeta+eta) * u               -- coupling term
///
///   PA_eff[d] = (P[d] - A[d]) - rho/zeta * u * (P[d] - Q[d])
///   QC_eff[d] = (Q[d] - C[d]) + rho/eta * u * (P[d] - Q[d])
///
/// Recursion for (a, 0 | c, 0):
///   I(0,0,0,0) = 1
///   I(a+1,0,c,0) = PA_eff * I(a,0,c,0) + a*B10 * I(a-1,0,c,0) + c*B00 * I(a,0,c-1,0)
///   I(a,0,c+1,0) = QC_eff * I(a,0,c,0) + c*B01 * I(a,0,c-1,0) + a*B00 * I(a-1,0,c,0)
///
/// HRR:
///   I(a,b+1,c,d) = I(a+1,b,c,d) + (A-B)[d] * I(a,b,c,d)
///   I(a,b,c,d+1) = I(a,b,c+1,d) + (C-D)[d] * I(a,b,c,d)
///
/// @param La Max angular momentum for center A
/// @param Lb Max angular momentum for center B
/// @param Lc Max angular momentum for center C
/// @param Ld Max angular momentum for center D
/// @param PA_eff Effective displacement P-A (Rys-modified)
/// @param QC_eff Effective displacement Q-C (Rys-modified)
/// @param AB Displacement A-B for HRR
/// @param CD Displacement C-D for HRR
/// @param B10 Bra internal recursion coefficient
/// @param B01 Ket internal recursion coefficient
/// @param B00 Bra-ket coupling coefficient
/// @param[out] I 4D recursion table [La+Lb+1][Lb+1][Lc+Ld+1][Ld+1]
/// @brief Build 2D Rys recursion table using a flat buffer (no heap allocation)
///
/// @param La,Lb,Lc,Ld Angular momentum values
/// @param PA_eff,QC_eff,AB,CD Recursion displacements
/// @param B10,B01,B00 Recursion coefficients
/// @param I Flat buffer of size dim_a * dim_b * dim_c * dim_d (pre-allocated)
/// @param dim_a,dim_b,dim_c,dim_d Table dimensions
void build_2d_rys(int La, int Lb, int Lc, int Ld,
                   Real PA_eff, Real QC_eff, Real AB, Real CD,
                   Real B10, Real B01, Real B00,
                   Real* I, int dim_a, int dim_b, int dim_c, int dim_d) {
    // 4D indexing: I[a][b][c][d] = I[a * (dim_b*dim_c*dim_d) + b * (dim_c*dim_d) + c * dim_d + d]
    const int stride_a = dim_b * dim_c * dim_d;
    const int stride_b = dim_c * dim_d;
    const int stride_c = dim_d;
    auto idx = [=](int a, int b, int c, int d) {
        return a * stride_a + b * stride_b + c * stride_c + d;
    };

    // Zero the buffer
    std::fill_n(I, dim_a * dim_b * dim_c * dim_d, Real{0});

    // Step 1: Build (a, 0 | c, 0) via VRR
    I[idx(0,0,0,0)] = 1.0;

    // First build up 'a' with c=0
    for (int a = 0; a < La + Lb; ++a) {
        I[idx(a+1,0,0,0)] = PA_eff * I[idx(a,0,0,0)];
        if (a > 0) {
            I[idx(a+1,0,0,0)] += static_cast<Real>(a) * B10 * I[idx(a-1,0,0,0)];
        }
    }

    // Now build up 'c' for all 'a'
    for (int c = 0; c < Lc + Ld; ++c) {
        for (int a = 0; a <= La + Lb; ++a) {
            I[idx(a,0,c+1,0)] = QC_eff * I[idx(a,0,c,0)];
            if (c > 0) {
                I[idx(a,0,c+1,0)] += static_cast<Real>(c) * B01 * I[idx(a,0,c-1,0)];
            }
            if (a > 0) {
                I[idx(a,0,c+1,0)] += static_cast<Real>(a) * B00 * I[idx(a-1,0,c,0)];
            }
        }
    }

    // Step 2: HRR to transfer angular momentum to B (bra side)
    for (int b = 0; b < Lb; ++b) {
        for (int a = 0; a <= La + Lb - b - 1; ++a) {
            for (int c = 0; c <= Lc + Ld; ++c) {
                for (int d = 0; d < dim_d; ++d) {
                    I[idx(a,b+1,c,d)] = I[idx(a+1,b,c,d)] + AB * I[idx(a,b,c,d)];
                }
            }
        }
    }

    // Step 3: HRR to transfer angular momentum to D (ket side)
    for (int d = 0; d < Ld; ++d) {
        for (int a = 0; a <= La; ++a) {
            for (int b = 0; b <= Lb; ++b) {
                for (int c = 0; c <= Lc + Ld - d - 1; ++c) {
                    I[idx(a,b,c,d+1)] = I[idx(a,b,c+1,d)] + CD * I[idx(a,b,c,d)];
                }
            }
        }
    }
}

}  // anonymous namespace

void compute_eri(const Shell& shell_a, const Shell& shell_b,
                 const Shell& shell_c, const Shell& shell_d,
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

    const auto& A = shell_a.center();
    const auto& B = shell_b.center();
    const auto& C = shell_c.center();
    const auto& D = shell_d.center();

    // Displacement vectors for HRR
    const Real AB_x = A.x - B.x;
    const Real AB_y = A.y - B.y;
    const Real AB_z = A.z - B.z;
    const Real CD_x = C.x - D.x;
    const Real CD_y = C.y - D.y;
    const Real CD_z = C.z - D.z;

    // Generate Cartesian index tuples
    const auto indices_a = math::generate_cartesian_indices(La);
    const auto indices_b = math::generate_cartesian_indices(Lb);
    const auto indices_c = math::generate_cartesian_indices(Lc);
    const auto indices_d = math::generate_cartesian_indices(Ld);

    // Precompute normalization corrections
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

    // Number of Rys roots
    const int n_rys_roots = (La + Lb + Lc + Ld) / 2 + 1;

    // Temporary storage for Rys roots/weights
    std::vector<double> roots(n_rys_roots);
    std::vector<double> weights(n_rys_roots);

    // Pre-allocate flat buffers for 2D recursion tables (once, not per root)
    const int dim_a = La + Lb + 1;
    const int dim_b = Lb + 1;
    const int dim_c = Lc + Ld + 1;
    const int dim_d = Ld + 1;
    const int table_size = dim_a * dim_b * dim_c * dim_d;
    std::vector<Real> Ix_flat(table_size), Iy_flat(table_size), Iz_flat(table_size);

    // Primitive data
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

            // Bra Gaussian product
            const auto gp_bra = math::compute_gaussian_product(alpha, A, beta, B);
            const Real zeta = gp_bra.zeta;
            const auto& P = gp_bra.P;

            for (Size r = 0; r < shell_c.n_primitives(); ++r) {
                const Real gamma = exp_c[r];
                const Real cc = coeff_c[r];

                for (Size s = 0; s < shell_d.n_primitives(); ++s) {
                    const Real delta = exp_d[s];
                    const Real cd = coeff_d[s];

                    // Ket Gaussian product
                    const auto gp_ket = math::compute_gaussian_product(gamma, C, delta, D);
                    const Real eta = gp_ket.zeta;
                    const auto& Q = gp_ket.P;

                    // Reduced exponent
                    const Real rho = zeta * eta / (zeta + eta);

                    // |P - Q|^2
                    const Real PQ_x = P.x - Q.x;
                    const Real PQ_y = P.y - Q.y;
                    const Real PQ_z = P.z - Q.z;
                    const Real PQ2 = PQ_x * PQ_x + PQ_y * PQ_y + PQ_z * PQ_z;

                    // Boys function argument
                    const Real T = rho * PQ2;

                    // Prefactor: 2 * pi^(5/2) / (zeta * eta * sqrt(zeta + eta)) * K_AB * K_CD
                    const Real pi_52 = constants::PI * constants::PI * std::sqrt(constants::PI);
                    const Real prefactor = 2.0 * pi_52 /
                        (zeta * eta * std::sqrt(zeta + eta)) *
                        gp_bra.K_AB * gp_ket.K_AB;

                    // Combined contraction coefficient
                    const Real coeff = ca * cb * cc * cd * prefactor;

                    // Get Rys roots and weights
                    math::rys_compute(n_rys_roots, T, roots.data(), weights.data());

                    // Loop over Rys quadrature points
                    for (int root = 0; root < n_rys_roots; ++root) {
                        const Real u = roots[root];    // u = t^2 (squared Rys root)
                        const Real w = weights[root];

                        // Recursion coefficients
                        const Real rho_over_zeta = rho / zeta;
                        const Real rho_over_eta = rho / eta;

                        const Real B10 = 0.5 / zeta * (1.0 - rho_over_zeta * u);
                        const Real B01 = 0.5 / eta * (1.0 - rho_over_eta * u);
                        const Real B00 = 0.5 / (zeta + eta) * u;

                        // Effective displacements
                        const Real PA_x_eff = (P.x - A.x) - rho_over_zeta * u * PQ_x;
                        const Real PA_y_eff = (P.y - A.y) - rho_over_zeta * u * PQ_y;
                        const Real PA_z_eff = (P.z - A.z) - rho_over_zeta * u * PQ_z;

                        const Real QC_x_eff = (Q.x - C.x) + rho_over_eta * u * PQ_x;
                        const Real QC_y_eff = (Q.y - C.y) + rho_over_eta * u * PQ_y;
                        const Real QC_z_eff = (Q.z - C.z) + rho_over_eta * u * PQ_z;

                        // Build 2D recursion tables into pre-allocated flat buffers
                        build_2d_rys(La, Lb, Lc, Ld,
                                     PA_x_eff, QC_x_eff, AB_x, CD_x,
                                     B10, B01, B00,
                                     Ix_flat.data(), dim_a, dim_b, dim_c, dim_d);
                        build_2d_rys(La, Lb, Lc, Ld,
                                     PA_y_eff, QC_y_eff, AB_y, CD_y,
                                     B10, B01, B00,
                                     Iy_flat.data(), dim_a, dim_b, dim_c, dim_d);
                        build_2d_rys(La, Lb, Lc, Ld,
                                     PA_z_eff, QC_z_eff, AB_z, CD_z,
                                     B10, B01, B00,
                                     Iz_flat.data(), dim_a, dim_b, dim_c, dim_d);

                        // Flat indexing helper
                        const int stride_a = dim_b * dim_c * dim_d;
                        const int stride_b = dim_c * dim_d;
                        const int stride_c = dim_d;
                        auto idx = [=](int a, int b, int c, int d) {
                            return a * stride_a + b * stride_b + c * stride_c + d;
                        };

                        // Weighted coefficient
                        const Real wcoeff = coeff * w;

                        // Accumulate contributions for all Cartesian component quartets
                        for (int ia = 0; ia < na; ++ia) {
                            const auto& [ax, ay, az] = indices_a[ia];
                            for (int ib = 0; ib < nb; ++ib) {
                                const auto& [bx, by, bz] = indices_b[ib];
                                for (int ic = 0; ic < nc; ++ic) {
                                    const auto& [cx, cy, cz] = indices_c[ic];
                                    for (int id = 0; id < nd; ++id) {
                                        const auto& [dx, dy, dz] = indices_d[id];

                                        const Real val =
                                            Ix_flat[idx(ax,bx,cx,dx)] *
                                            Iy_flat[idx(ay,by,cy,dy)] *
                                            Iz_flat[idx(az,bz,cz,dz)];

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

}  // namespace libaccint::kernels
