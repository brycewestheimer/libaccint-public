// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file three_center_eri_kernel.cpp
/// @brief CPU implementation of three-center ERI (ab|P)
///
/// Computes three-center electron repulsion integrals using Rys quadrature.
/// The integral is:
///   (ab|P) = integral phi_a(r1) * phi_b(r1) * (1/r12) * chi_P(r2) dr1 dr2
///
/// This is treated as a four-center integral (ab|Ps) where s is an
/// s-function at the auxiliary center, then simplified.

#include <libaccint/kernels/three_center_eri_kernel.hpp>
#include <libaccint/kernels/norm_correction.hpp>
#include <libaccint/math/gaussian_product.hpp>
#include <libaccint/math/normalization.hpp>
#include <libaccint/math/cartesian_indices.hpp>
#include <libaccint/math/rys_quadrature.hpp>
#include <libaccint/math/boys_function.hpp>
#include <libaccint/utils/constants.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <cmath>
#include <vector>

namespace libaccint::kernels {

namespace {

/// @brief Build 3D recursion table for three-center integral
///
/// For (ab|P), we have a bra product (ab) at combined center W_bra
/// and a single auxiliary function P. The recursion builds:
///   I[la][lb][lP] for the three angular momenta.
///
/// This is similar to 4-center but with the ket being a single function.
void build_3d_three_center(int La, int Lb, int Lp,
                            Real WA, Real WB, Real WP_ket,
                            Real AB,
                            Real B10, Real B00,
                            std::vector<std::vector<std::vector<Real>>>& I) {
    const int dim_a = La + Lb + 1;
    const int dim_b = Lb + 1;
    const int dim_p = Lp + 1;

    I.assign(dim_a, std::vector<std::vector<Real>>(
        dim_b, std::vector<Real>(dim_p, 0.0)));

    // Base case
    I[0][0][0] = 1.0;

    // VRR on bra (build I[a][0][0])
    for (int a = 0; a < La + Lb; ++a) {
        I[a + 1][0][0] = WA * I[a][0][0];
        if (a > 0) {
            I[a + 1][0][0] += static_cast<Real>(a) * B10 * I[a - 1][0][0];
        }
    }

    // VRR on ket auxiliary (build I[a][0][p])
    for (int p = 0; p < Lp; ++p) {
        for (int a = 0; a <= La + Lb; ++a) {
            I[a][0][p + 1] = WP_ket * I[a][0][p];
            if (p > 0) {
                // Using B00 for ket internal recursion (simplified)
                I[a][0][p + 1] += static_cast<Real>(p) * B00 * I[a][0][p - 1];
            }
            if (a > 0) {
                // Coupling between bra and ket
                I[a][0][p + 1] += static_cast<Real>(a) * B00 * I[a - 1][0][p];
            }
        }
    }

    // HRR to transfer angular momentum from A to B
    // I[a][b+1][p] = I[a+1][b][p] + (A-B) * I[a][b][p]
    // AB is the fixed geometry displacement A[d] - B[d], NOT root-dependent
    for (int b = 0; b < Lb; ++b) {
        for (int a = 0; a <= La + Lb - b - 1; ++a) {
            for (int p = 0; p <= Lp; ++p) {
                I[a][b + 1][p] = I[a + 1][b][p] + AB * I[a][b][p];
            }
        }
    }
}

}  // anonymous namespace

void compute_three_center_eri(const Shell& shell_a,
                               const Shell& shell_b,
                               const Shell& shell_P,
                               TwoElectronBuffer<0>& buffer) {
    const int La = shell_a.angular_momentum();
    const int Lb = shell_b.angular_momentum();
    const int Lp = shell_P.angular_momentum();
    const int na = shell_a.n_functions();
    const int nb = shell_b.n_functions();
    const int np = shell_P.n_functions();

    // Resize buffer: (na * nb) x np as a 4D tensor with dummy dimensions
    buffer.resize(na, nb, np, 1);
    buffer.clear();

    const auto& A = shell_a.center();
    const auto& B = shell_b.center();
    const auto& P = shell_P.center();

    // Bra distance
    const Real AB_x = A[0] - B[0];
    const Real AB_y = A[1] - B[1];
    const Real AB_z = A[2] - B[2];
    const Real AB2 = AB_x * AB_x + AB_y * AB_y + AB_z * AB_z;

    // Total angular momentum
    const int L_total = La + Lb + Lp;

    // Number of Rys roots needed
    const int n_roots = (L_total + 2) / 2;

    // Temporary storage
    std::vector<std::vector<std::vector<Real>>> Ix, Iy, Iz;

    // Loop over primitives (quadruple loop simplified to triple for 3-center)
    for (Size ia = 0; ia < shell_a.n_primitives(); ++ia) {
        const Real alpha_a = shell_a.exponents()[ia];
        const Real coef_a = shell_a.coefficients()[ia];

        for (Size ib = 0; ib < shell_b.n_primitives(); ++ib) {
            const Real alpha_b = shell_b.exponents()[ib];
            const Real coef_b = shell_b.coefficients()[ib];

            // Bra Gaussian product
            const Real zeta = alpha_a + alpha_b;
            const Real K_AB = std::exp(-alpha_a * alpha_b / zeta * AB2);

            // Bra center
            const Real Wx = (alpha_a * A[0] + alpha_b * B[0]) / zeta;
            const Real Wy = (alpha_a * A[1] + alpha_b * B[1]) / zeta;
            const Real Wz = (alpha_a * A[2] + alpha_b * B[2]) / zeta;

            for (Size ip = 0; ip < shell_P.n_primitives(); ++ip) {
                const Real alpha_p = shell_P.exponents()[ip];
                const Real coef_p = shell_P.coefficients()[ip];

                // Combined exponent
                const Real eta = alpha_p;  // Ket is single function
                const Real rho = zeta * eta / (zeta + eta);

                // W to P distance
                const Real WP_x = Wx - P[0];
                const Real WP_y = Wy - P[1];
                const Real WP_z = Wz - P[2];
                const Real WP2 = WP_x * WP_x + WP_y * WP_y + WP_z * WP_z;

                // Boys function argument
                const Real T = rho * WP2;

                // Prefactor
                const Real prefactor = 2.0 * std::pow(constants::PI, 2.5) /
                    (zeta * eta * std::sqrt(zeta + eta)) *
                    K_AB * coef_a * coef_b * coef_p;

                // Rys quadrature points and weights
                std::vector<Real> roots(n_roots), weights(n_roots);
                math::rys_compute(n_roots, T, roots.data(), weights.data());

                // For each Rys root, build recursion tables and accumulate
                for (int r = 0; r < n_roots; ++r) {
                    const Real u = roots[r];
                    const Real w = weights[r];

                    // Recursion parameters
                    const Real B10 = 0.5 / zeta * (1.0 - rho / zeta * u);
                    const Real B01 = 0.5 / eta * (1.0 - rho / eta * u);
                    const Real B00 = 0.5 / (zeta + eta) * u;

                    // Effective displacements
                    const Real WA_eff_x = (Wx - A[0]) - rho / zeta * u * WP_x;
                    const Real WA_eff_y = (Wy - A[1]) - rho / zeta * u * WP_y;
                    const Real WA_eff_z = (Wz - A[2]) - rho / zeta * u * WP_z;

                    const Real WB_eff_x = (Wx - B[0]) - rho / zeta * u * WP_x;
                    const Real WB_eff_y = (Wy - B[1]) - rho / zeta * u * WP_y;
                    const Real WB_eff_z = (Wz - B[2]) - rho / zeta * u * WP_z;

                    const Real PP_eff_x = (P[0] - Wx) + rho / eta * u * WP_x;
                    const Real PP_eff_y = (P[1] - Wy) + rho / eta * u * WP_y;
                    const Real PP_eff_z = (P[2] - Wz) + rho / eta * u * WP_z;

                    // Build 3D recursion tables
                    // AB_x/y/z are fixed geometry displacements for HRR
                    build_3d_three_center(La, Lb, Lp, WA_eff_x, WB_eff_x, PP_eff_x, AB_x, B10, B00, Ix);
                    build_3d_three_center(La, Lb, Lp, WA_eff_y, WB_eff_y, PP_eff_y, AB_y, B10, B00, Iy);
                    build_3d_three_center(La, Lb, Lp, WA_eff_z, WB_eff_z, PP_eff_z, AB_z, B10, B00, Iz);

                    // Accumulate integrals
                    const auto cart_a = math::generate_cartesian_indices(La);
                    const auto cart_b = math::generate_cartesian_indices(Lb);
                    const auto cart_p = math::generate_cartesian_indices(Lp);

                    for (int ja = 0; ja < na; ++ja) {
                        const auto [lax, lay, laz] = cart_a[ja];
                        const Real norm_a = norm_correction(lax, lay, laz);

                        for (int jb = 0; jb < nb; ++jb) {
                            const auto [lbx, lby, lbz] = cart_b[jb];
                            const Real norm_b = norm_correction(lbx, lby, lbz);

                            for (int jp = 0; jp < np; ++jp) {
                                const auto [lpx, lpy, lpz] = cart_p[jp];
                                const Real norm_p = norm_correction(lpx, lpy, lpz);

                                const Real Ix_val = Ix[lax][lbx][lpx];
                                const Real Iy_val = Iy[lay][lby][lpy];
                                const Real Iz_val = Iz[laz][lbz][lpz];

                                const Real contrib = prefactor * w *
                                    Ix_val * Iy_val * Iz_val *
                                    norm_a * norm_b * norm_p;

                                buffer(ja, jb, jp, 0) += contrib;
                            }
                        }
                    }
                }
            }
        }
    }
}

void compute_three_center_eri_block(const Shell& shell_a,
                                     const Shell& shell_b,
                                     const Shell& shell_P,
                                     Real* output,
                                     Size stride_ab,
                                     Size stride_a) {
    TwoElectronBuffer<0> buffer;
    compute_three_center_eri(shell_a, shell_b, shell_P, buffer);

    const int na = shell_a.n_functions();
    const int nb = shell_b.n_functions();
    const int np = shell_P.n_functions();

    const Size actual_stride_ab = (stride_ab == 0) ? np : stride_ab;
    const Size actual_stride_a = (stride_a == 0) ? nb * actual_stride_ab : stride_a;

    for (int ja = 0; ja < na; ++ja) {
        for (int jb = 0; jb < nb; ++jb) {
            for (int jp = 0; jp < np; ++jp) {
                output[ja * actual_stride_a + jb * actual_stride_ab + jp] =
                    buffer(ja, jb, jp, 0);
            }
        }
    }
}

Real compute_three_center_eri_scalar(const Shell& shell_a,
                                      const Shell& shell_b,
                                      const Shell& shell_P) {
    if (shell_a.angular_momentum() != 0 ||
        shell_b.angular_momentum() != 0 ||
        shell_P.angular_momentum() != 0) {
        throw InvalidArgumentException(
            "compute_three_center_eri_scalar requires s-type shells");
    }

    TwoElectronBuffer<0> buffer;
    compute_three_center_eri(shell_a, shell_b, shell_P, buffer);
    return buffer(0, 0, 0, 0);
}

void compute_three_center_tensor(std::span<const Shell> orbital_shells,
                                  std::span<const Shell> auxiliary_shells,
                                  Real* tensor,
                                  Size n_orb,
                                  Size n_aux,
                                  ThreeCenterStorageFormat format) {
    // Compute function offsets for orbital basis
    std::vector<Size> orb_offset(orbital_shells.size());
    Size offset = 0;
    for (Size i = 0; i < orbital_shells.size(); ++i) {
        orb_offset[i] = offset;
        offset += orbital_shells[i].n_functions();
    }

    // Compute function offsets for auxiliary basis
    std::vector<Size> aux_offset(auxiliary_shells.size());
    offset = 0;
    for (Size i = 0; i < auxiliary_shells.size(); ++i) {
        aux_offset[i] = offset;
        offset += auxiliary_shells[i].n_functions();
    }

    // Zero initialize
    const Size tensor_size = (format == ThreeCenterStorageFormat::abP ||
                              format == ThreeCenterStorageFormat::Pab)
                             ? n_orb * n_orb * n_aux
                             : n_orb * (n_orb + 1) / 2 * n_aux;
    for (Size i = 0; i < tensor_size; ++i) {
        tensor[i] = 0.0;
    }

    TwoElectronBuffer<0> buffer;

    // Loop over shell triples
    for (Size sa = 0; sa < orbital_shells.size(); ++sa) {
        for (Size sb = sa; sb < orbital_shells.size(); ++sb) {  // Symmetry: a <= b
            for (Size sp = 0; sp < auxiliary_shells.size(); ++sp) {
                compute_three_center_eri(orbital_shells[sa],
                                          orbital_shells[sb],
                                          auxiliary_shells[sp],
                                          buffer);

                const Size fa = orb_offset[sa];
                const Size fb = orb_offset[sb];
                const Size fp = aux_offset[sp];
                const int na = orbital_shells[sa].n_functions();
                const int nb = orbital_shells[sb].n_functions();
                const int np = auxiliary_shells[sp].n_functions();

                for (int ja = 0; ja < na; ++ja) {
                    for (int jb = 0; jb < nb; ++jb) {
                        for (int jp = 0; jp < np; ++jp) {
                            const Real val = buffer(ja, jb, jp, 0);
                            const Size a = fa + ja;
                            const Size b = fb + jb;
                            const Size p = fp + jp;

                            switch (format) {
                                case ThreeCenterStorageFormat::abP:
                                    tensor[a * n_orb * n_aux + b * n_aux + p] = val;
                                    if (sa != sb || ja != jb) {
                                        tensor[b * n_orb * n_aux + a * n_aux + p] = val;
                                    }
                                    break;
                                case ThreeCenterStorageFormat::Pab:
                                    tensor[p * n_orb * n_orb + a * n_orb + b] = val;
                                    if (sa != sb || ja != jb) {
                                        tensor[p * n_orb * n_orb + b * n_orb + a] = val;
                                    }
                                    break;
                                case ThreeCenterStorageFormat::uabP:
                                case ThreeCenterStorageFormat::Puab:
                                    // Upper triangle with a <= b
                                    if (a <= b) {
                                        const Size pair_idx = a * n_orb - a * (a + 1) / 2 + b;
                                        if (format == ThreeCenterStorageFormat::uabP) {
                                            tensor[pair_idx * n_aux + p] = val;
                                        } else {
                                            tensor[p * (n_orb * (n_orb + 1) / 2) + pair_idx] = val;
                                        }
                                    }
                                    break;
                            }
                        }
                    }
                }
            }
        }
    }
}

void compute_B_tensor(const Real* three_center,
                       const Real* L_inv,
                       Real* B_tensor,
                       Size n_orb,
                       Size n_aux) {
    // B_ab^P = sum_Q (ab|Q) * L^{-1}_{QP}
    // This is a matrix multiply: B = (ab|Q) * L^{-1}
    // where three_center is (n_orb^2) x n_aux and L_inv is n_aux x n_aux

    // For efficiency, use BLAS dgemm if available
    // Here we implement a simple loop version

    const Size n_pairs = n_orb * n_orb;

    for (Size ab = 0; ab < n_pairs; ++ab) {
        for (Size P = 0; P < n_aux; ++P) {
            Real sum = 0.0;
            for (Size Q = 0; Q < n_aux; ++Q) {
                sum += three_center[ab * n_aux + Q] * L_inv[Q * n_aux + P];
            }
            B_tensor[ab * n_aux + P] = sum;
        }
    }
}

}  // namespace libaccint::kernels
