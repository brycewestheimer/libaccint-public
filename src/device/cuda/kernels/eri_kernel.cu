// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file eri_kernel.cu
/// @brief CUDA kernel for electron repulsion integrals using Rys quadrature
///
/// Implements thread-per-quartet ERI computation with 2D Rys recursion.
/// Each CUDA thread computes all integrals for one shell quartet (a,b,c,d).

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/kernels/eri_kernel_cuda.hpp>
#include <libaccint/kernels/eri_kernel_warp_cuda.hpp>
#include <libaccint/memory/device_memory.hpp>

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cmath>

// Include device Boys and Rys function headers
#include "../../cuda/math/boys_device.cuh"
#include "../../cuda/math/rys_device.cuh"

namespace libaccint::kernels::cuda {

// ============================================================================
// Constants
// ============================================================================

// pi^(5/2) = pi^2 * sqrt(pi) ≈ 17.493418327...
constexpr double PI_52 = 17.493418327624862846262821679872;

// Maximum Rys roots for supported AM combinations: (4+4+4+4)/2 + 1 = 9
constexpr int MAX_RYS_ROOTS = 9;

// Maximum 2D table dimension: (La+Lb+1) max = 9 for g+g
constexpr int MAX_2D_DIM = 9;

// ============================================================================
// Device Helper Functions
// ============================================================================

/// @brief Compute double factorial (2n-1)!! = 1*3*5*...*(2n-1)
static __device__ int eri_double_factorial_odd(int n) {
    if (n <= 0) return 1;
    int result = 1;
    for (int i = 1; i <= n; ++i) {
        result *= (2 * i - 1);
    }
    return result;
}

/// @brief Compute normalization correction for Cartesian component (lx, ly, lz)
static __device__ double eri_norm_correction(int lx, int ly, int lz) {
    const double denom = static_cast<double>(
        eri_double_factorial_odd(lx) *
        eri_double_factorial_odd(ly) *
        eri_double_factorial_odd(lz));
    return rsqrt(denom);
}

// ============================================================================
// Device 2D Rys Recursion
// ============================================================================

/// @brief Build 2D Rys recursion table for one Cartesian direction
///
/// Builds the table I[a][b][c][d] for one Cartesian direction at one Rys root.
///
/// Recursion for (a, 0 | c, 0):
///   I(0,0,0,0) = 1
///   I(a+1,0,c,0) = PA_eff * I(a,0,c,0) + a*B10 * I(a-1,0,c,0) + c*B00 * I(a,0,c-1,0)
///   I(a,0,c+1,0) = QC_eff * I(a,0,c,0) + c*B01 * I(a,0,c-1,0) + a*B00 * I(a-1,0,c,0)
///
/// HRR:
///   I(a,b+1,c,d) = I(a+1,b,c,d) + AB * I(a,b,c,d)
///   I(a,b,c,d+1) = I(a,b,c+1,d) + CD * I(a,b,c,d)
///
/// @param La, Lb, Lc, Ld Angular momenta
/// @param PA_eff, QC_eff Effective displacements (Rys-modified)
/// @param AB, CD Displacements for HRR
/// @param B10, B01, B00 Recursion coefficients
/// @param I Output table [MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM]
template <int La, int Lb, int Lc, int Ld>
__device__ void build_2d_rys_device(
    double PA_eff, double QC_eff, double AB, double CD,
    double B10, double B01, double B00,
    double I[MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM]) {

    constexpr int dim_a = La + Lb + 1;
    constexpr int dim_c = Lc + Ld + 1;

    // Zero the table
    #pragma unroll
    for (int a = 0; a < dim_a; ++a) {
        #pragma unroll
        for (int b = 0; b <= Lb; ++b) {
            #pragma unroll
            for (int c = 0; c < dim_c; ++c) {
                #pragma unroll
                for (int d = 0; d <= Ld; ++d) {
                    I[a][b][c][d] = 0.0;
                }
            }
        }
    }

    // Step 1: Build (a, 0 | c, 0) via VRR
    I[0][0][0][0] = 1.0;

    // Build up 'a' with c=0
    #pragma unroll
    for (int a = 0; a < La + Lb; ++a) {
        I[a + 1][0][0][0] = PA_eff * I[a][0][0][0];
        if (a > 0) {
            I[a + 1][0][0][0] += static_cast<double>(a) * B10 * I[a - 1][0][0][0];
        }
    }

    // Build up 'c' for all 'a'
    #pragma unroll
    for (int c = 0; c < Lc + Ld; ++c) {
        #pragma unroll
        for (int a = 0; a <= La + Lb; ++a) {
            I[a][0][c + 1][0] = QC_eff * I[a][0][c][0];
            if (c > 0) {
                I[a][0][c + 1][0] += static_cast<double>(c) * B01 * I[a][0][c - 1][0];
            }
            if (a > 0) {
                I[a][0][c + 1][0] += static_cast<double>(a) * B00 * I[a - 1][0][c][0];
            }
        }
    }

    // Step 2: HRR to transfer angular momentum to B (bra side)
    #pragma unroll
    for (int b = 0; b < Lb; ++b) {
        #pragma unroll
        for (int a = 0; a <= La + Lb - b - 1; ++a) {
            #pragma unroll
            for (int c = 0; c <= Lc + Ld; ++c) {
                #pragma unroll
                for (int d = 0; d <= Ld; ++d) {
                    I[a][b + 1][c][d] = I[a + 1][b][c][d] + AB * I[a][b][c][d];
                }
            }
        }
    }

    // Step 3: HRR to transfer angular momentum to D (ket side)
    #pragma unroll
    for (int d = 0; d < Ld; ++d) {
        #pragma unroll
        for (int a = 0; a <= La; ++a) {
            #pragma unroll
            for (int b = 0; b <= Lb; ++b) {
                #pragma unroll
                for (int c = 0; c <= Lc + Ld - d - 1; ++c) {
                    I[a][b][c][d + 1] = I[a][b][c + 1][d] + CD * I[a][b][c][d];
                }
            }
        }
    }
}

// ============================================================================
// Cartesian Index Tables
// ============================================================================

/// Cartesian indices for angular momentum L
/// idx_table[L][i] = {lx, ly, lz} for the i-th Cartesian component

__constant__ int s_cart_lx[1] = {0};
__constant__ int s_cart_ly[1] = {0};
__constant__ int s_cart_lz[1] = {0};

__constant__ int p_cart_lx[3] = {1, 0, 0};
__constant__ int p_cart_ly[3] = {0, 1, 0};
__constant__ int p_cart_lz[3] = {0, 0, 1};

__constant__ int d_cart_lx[6] = {2, 1, 1, 0, 0, 0};
__constant__ int d_cart_ly[6] = {0, 1, 0, 2, 1, 0};
__constant__ int d_cart_lz[6] = {0, 0, 1, 0, 1, 2};

__constant__ int f_cart_lx[10] = {3, 2, 2, 1, 1, 1, 0, 0, 0, 0};
__constant__ int f_cart_ly[10] = {0, 1, 0, 2, 1, 0, 3, 2, 1, 0};
__constant__ int f_cart_lz[10] = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3};

__constant__ int g_cart_lx[15] = {4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0};
__constant__ int g_cart_ly[15] = {0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0};
__constant__ int g_cart_lz[15] = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4};

__device__ void get_cart_indices(int L, int i, int& lx, int& ly, int& lz) {
    switch (L) {
        case 0:
            lx = s_cart_lx[i]; ly = s_cart_ly[i]; lz = s_cart_lz[i];
            break;
        case 1:
            lx = p_cart_lx[i]; ly = p_cart_ly[i]; lz = p_cart_lz[i];
            break;
        case 2:
            lx = d_cart_lx[i]; ly = d_cart_ly[i]; lz = d_cart_lz[i];
            break;
        case 3:
            lx = f_cart_lx[i]; ly = f_cart_ly[i]; lz = f_cart_lz[i];
            break;
        case 4:
            lx = g_cart_lx[i]; ly = g_cart_ly[i]; lz = g_cart_lz[i];
            break;
        default:
            lx = ly = lz = 0;
    }
}

__device__ int n_cart_functions(int L) {
    return (L + 1) * (L + 2) / 2;
}

// ============================================================================
// ERI Kernel Template
// ============================================================================

template <int La, int Lb, int Lc, int Ld>
__global__ void eri_kernel(
    // Shell A data
    const double* __restrict__ d_exp_a,
    const double* __restrict__ d_coeff_a,
    const double* __restrict__ d_cx_a,
    const double* __restrict__ d_cy_a,
    const double* __restrict__ d_cz_a,
    int n_shells_a, int n_prim_a,
    // Shell B data
    const double* __restrict__ d_exp_b,
    const double* __restrict__ d_coeff_b,
    const double* __restrict__ d_cx_b,
    const double* __restrict__ d_cy_b,
    const double* __restrict__ d_cz_b,
    int n_shells_b, int n_prim_b,
    // Shell C data
    const double* __restrict__ d_exp_c,
    const double* __restrict__ d_coeff_c,
    const double* __restrict__ d_cx_c,
    const double* __restrict__ d_cy_c,
    const double* __restrict__ d_cz_c,
    int n_shells_c, int n_prim_c,
    // Shell D data
    const double* __restrict__ d_exp_d,
    const double* __restrict__ d_coeff_d,
    const double* __restrict__ d_cx_d,
    const double* __restrict__ d_cy_d,
    const double* __restrict__ d_cz_d,
    int n_shells_d, int n_prim_d,
    // Boys function coefficients
    const double* __restrict__ d_boys_coeffs,
    // Output buffer
    double* __restrict__ d_output) {

    // Number of Cartesian functions
    constexpr int na = (La + 1) * (La + 2) / 2;
    constexpr int nb = (Lb + 1) * (Lb + 2) / 2;
    constexpr int nc = (Lc + 1) * (Lc + 2) / 2;
    constexpr int nd = (Ld + 1) * (Ld + 2) / 2;
    constexpr int n_out = na * nb * nc * nd;

    // Number of Rys roots
    constexpr int n_rys = (La + Lb + Lc + Ld) / 2 + 1;

    // Thread ID -> shell quartet (ia, ib, ic, id)
    const int n_quartets = n_shells_a * n_shells_b * n_shells_c * n_shells_d;
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id >= n_quartets) return;

    // Decode shell indices
    int idx = global_id;
    const int id = idx % n_shells_d; idx /= n_shells_d;
    const int ic = idx % n_shells_c; idx /= n_shells_c;
    const int ib = idx % n_shells_b; idx /= n_shells_b;
    const int ia = idx;

    // Load shell centers
    const double Ax = d_cx_a[ia], Ay = d_cy_a[ia], Az = d_cz_a[ia];
    const double Bx = d_cx_b[ib], By = d_cy_b[ib], Bz = d_cz_b[ib];
    const double Cx = d_cx_c[ic], Cy = d_cy_c[ic], Cz = d_cz_c[ic];
    const double Dx = d_cx_d[id], Dy = d_cy_d[id], Dz = d_cz_d[id];

    // Displacement vectors for HRR
    const double AB_x = Ax - Bx, AB_y = Ay - By, AB_z = Az - Bz;
    const double CD_x = Cx - Dx, CD_y = Cy - Dy, CD_z = Cz - Dz;

    // Integral accumulator
    double integrals[n_out];
    #pragma unroll
    for (int i = 0; i < n_out; ++i) {
        integrals[i] = 0.0;
    }

    // Four-fold contraction loop
    for (int p = 0; p < n_prim_a; ++p) {
        const double alpha = d_exp_a[ia * n_prim_a + p];
        const double ca = d_coeff_a[ia * n_prim_a + p];

        for (int q = 0; q < n_prim_b; ++q) {
            const double beta = d_exp_b[ib * n_prim_b + q];
            const double cb = d_coeff_b[ib * n_prim_b + q];

            // Bra Gaussian product
            const double zeta = alpha + beta;
            const double oo_zeta = 1.0 / zeta;
            const double Px = (alpha * Ax + beta * Bx) * oo_zeta;
            const double Py = (alpha * Ay + beta * By) * oo_zeta;
            const double Pz = (alpha * Az + beta * Bz) * oo_zeta;

            // K_AB = exp(-alpha*beta/zeta * |A-B|^2)
            const double AB2 = AB_x * AB_x + AB_y * AB_y + AB_z * AB_z;
            const double K_AB = exp(-alpha * beta * oo_zeta * AB2);

            for (int r = 0; r < n_prim_c; ++r) {
                const double gamma = d_exp_c[ic * n_prim_c + r];
                const double cc = d_coeff_c[ic * n_prim_c + r];

                for (int s = 0; s < n_prim_d; ++s) {
                    const double delta = d_exp_d[id * n_prim_d + s];
                    const double cd = d_coeff_d[id * n_prim_d + s];

                    // Ket Gaussian product
                    const double eta = gamma + delta;
                    const double oo_eta = 1.0 / eta;
                    const double Qx = (gamma * Cx + delta * Dx) * oo_eta;
                    const double Qy = (gamma * Cy + delta * Dy) * oo_eta;
                    const double Qz = (gamma * Cz + delta * Dz) * oo_eta;

                    // K_CD = exp(-gamma*delta/eta * |C-D|^2)
                    const double CD2 = CD_x * CD_x + CD_y * CD_y + CD_z * CD_z;
                    const double K_CD = exp(-gamma * delta * oo_eta * CD2);

                    // Reduced exponent and P-Q
                    const double rho = zeta * eta / (zeta + eta);
                    const double PQ_x = Px - Qx;
                    const double PQ_y = Py - Qy;
                    const double PQ_z = Pz - Qz;
                    const double PQ2 = PQ_x * PQ_x + PQ_y * PQ_y + PQ_z * PQ_z;

                    // Boys function argument
                    const double T = rho * PQ2;

                    // Prefactor: 2 * pi^(5/2) / (zeta * eta * sqrt(zeta+eta)) * K_AB * K_CD
                    const double oo_zeta_eta = oo_zeta * oo_eta;
                    const double prefactor = 2.0 * PI_52 * oo_zeta_eta *
                        rsqrt(zeta + eta) * K_AB * K_CD;

                    // Combined contraction coefficient
                    const double coeff = ca * cb * cc * cd * prefactor;

                    // Get Rys roots and weights
                    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];
                    device::math::rys_quadrature_impl(n_rys, T, d_boys_coeffs, roots, weights);

                    // Loop over Rys quadrature points
                    for (int root = 0; root < n_rys; ++root) {
                        const double u = roots[root];  // u = t^2 (squared Rys root)
                        const double w = weights[root];

                        // Recursion coefficients
                        const double rho_over_zeta = rho * oo_zeta;
                        const double rho_over_eta = rho * oo_eta;

                        const double B10 = 0.5 * oo_zeta * (1.0 - rho_over_zeta * u);
                        const double B01 = 0.5 * oo_eta * (1.0 - rho_over_eta * u);
                        const double B00 = 0.5 / (zeta + eta) * u;

                        // Effective displacements
                        const double PA_x_eff = (Px - Ax) - rho_over_zeta * u * PQ_x;
                        const double PA_y_eff = (Py - Ay) - rho_over_zeta * u * PQ_y;
                        const double PA_z_eff = (Pz - Az) - rho_over_zeta * u * PQ_z;

                        const double QC_x_eff = (Qx - Cx) + rho_over_eta * u * PQ_x;
                        const double QC_y_eff = (Qy - Cy) + rho_over_eta * u * PQ_y;
                        const double QC_z_eff = (Qz - Cz) + rho_over_eta * u * PQ_z;

                        // Build 2D recursion tables
                        double Ix[MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM];
                        double Iy[MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM];
                        double Iz[MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM];

                        build_2d_rys_device<La, Lb, Lc, Ld>(
                            PA_x_eff, QC_x_eff, AB_x, CD_x, B10, B01, B00, Ix);
                        build_2d_rys_device<La, Lb, Lc, Ld>(
                            PA_y_eff, QC_y_eff, AB_y, CD_y, B10, B01, B00, Iy);
                        build_2d_rys_device<La, Lb, Lc, Ld>(
                            PA_z_eff, QC_z_eff, AB_z, CD_z, B10, B01, B00, Iz);

                        // Weighted coefficient
                        const double wcoeff = coeff * w;

                        // Accumulate contributions for all Cartesian component quartets
                        int out_idx = 0;
                        for (int ia_c = 0; ia_c < na; ++ia_c) {
                            int ax, ay, az;
                            get_cart_indices(La, ia_c, ax, ay, az);
                            for (int ib_c = 0; ib_c < nb; ++ib_c) {
                                int bx, by, bz;
                                get_cart_indices(Lb, ib_c, bx, by, bz);
                                for (int ic_c = 0; ic_c < nc; ++ic_c) {
                                    int cx, cy, cz;
                                    get_cart_indices(Lc, ic_c, cx, cy, cz);
                                    for (int id_c = 0; id_c < nd; ++id_c) {
                                        int dx, dy, dz;
                                        get_cart_indices(Ld, id_c, dx, dy, dz);

                                        const double val =
                                            Ix[ax][bx][cx][dx] *
                                            Iy[ay][by][cy][dy] *
                                            Iz[az][bz][cz][dz];

                                        integrals[out_idx++] += wcoeff * val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply normalization corrections and write output
    const size_t base_offset = static_cast<size_t>(global_id) * n_out;
    int out_idx = 0;
    for (int ia_c = 0; ia_c < na; ++ia_c) {
        int ax, ay, az;
        get_cart_indices(La, ia_c, ax, ay, az);
        const double corr_a = eri_norm_correction(ax, ay, az);

        for (int ib_c = 0; ib_c < nb; ++ib_c) {
            int bx, by, bz;
            get_cart_indices(Lb, ib_c, bx, by, bz);
            const double corr_b = eri_norm_correction(bx, by, bz);

            for (int ic_c = 0; ic_c < nc; ++ic_c) {
                int cx, cy, cz;
                get_cart_indices(Lc, ic_c, cx, cy, cz);
                const double corr_c = eri_norm_correction(cx, cy, cz);

                for (int id_c = 0; id_c < nd; ++id_c) {
                    int dx, dy, dz;
                    get_cart_indices(Ld, id_c, dx, dy, dz);
                    const double corr_d = eri_norm_correction(dx, dy, dz);

                    d_output[base_offset + out_idx] =
                        integrals[out_idx] * corr_a * corr_b * corr_c * corr_d;
                    out_idx++;
                }
            }
        }
    }
}

// ============================================================================
// Kernel Launch Functions
// ============================================================================

template <int La, int Lb, int Lc, int Ld>
void launch_eri_kernel_specialized(
    const basis::ShellSetQuartetDeviceData& quartet,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream) {

    const int n_quartets = quartet.a.n_shells * quartet.b.n_shells *
                           quartet.c.n_shells * quartet.d.n_shells;

    if (n_quartets == 0) return;

    // AM-dependent block size for register pressure management
    constexpr int total_am = La + Lb + Lc + Ld;
    constexpr int block_size = (total_am <= 2) ? 256 :
                               (total_am <= 4) ? 128 :
                               (total_am <= 7) ? 64 : 32;
    const int n_blocks = (n_quartets + block_size - 1) / block_size;

    eri_kernel<La, Lb, Lc, Ld><<<n_blocks, block_size, 0, stream>>>(
        quartet.a.d_exponents, quartet.a.d_coefficients,
        quartet.a.d_centers_x, quartet.a.d_centers_y, quartet.a.d_centers_z,
        quartet.a.n_shells, quartet.a.n_primitives,
        quartet.b.d_exponents, quartet.b.d_coefficients,
        quartet.b.d_centers_x, quartet.b.d_centers_y, quartet.b.d_centers_z,
        quartet.b.n_shells, quartet.b.n_primitives,
        quartet.c.d_exponents, quartet.c.d_coefficients,
        quartet.c.d_centers_x, quartet.c.d_centers_y, quartet.c.d_centers_z,
        quartet.c.n_shells, quartet.c.n_primitives,
        quartet.d.d_exponents, quartet.d.d_coefficients,
        quartet.d.d_centers_x, quartet.d.d_centers_y, quartet.d.d_centers_z,
        quartet.d.n_shells, quartet.d.n_primitives,
        d_boys_coeffs,
        d_output);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::invalid_argument(
            std::string("ERI kernel launch failed: ") + cudaGetErrorString(err));
    }
}

// Dispatch helpers bounded by LIBACCINT_MAX_AM to avoid compiling unreachable
// AM combinations in constrained builds.
namespace {

[[noreturn]] void throw_unsupported_eri_am(int la, int lb, int lc, int ld) {
    throw std::invalid_argument(
        "Unsupported angular momentum combination for ERI kernel: (" +
        std::to_string(la) + std::to_string(lb) + "|" +
        std::to_string(lc) + std::to_string(ld) +
        "), LIBACCINT_MAX_AM=" + std::to_string(LIBACCINT_MAX_AM));
}

template <int La, int Lb, int Lc>
void dispatch_eri_ld(
    int ld,
    const basis::ShellSetQuartetDeviceData& quartet,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream) {
    switch (ld) {
        case 0:
            launch_eri_kernel_specialized<La, Lb, Lc, 0>(quartet, d_boys_coeffs, d_output, stream);
            return;
#if LIBACCINT_MAX_AM >= 1
        case 1:
            launch_eri_kernel_specialized<La, Lb, Lc, 1>(quartet, d_boys_coeffs, d_output, stream);
            return;
#endif
#if LIBACCINT_MAX_AM >= 2
        case 2:
            launch_eri_kernel_specialized<La, Lb, Lc, 2>(quartet, d_boys_coeffs, d_output, stream);
            return;
#endif
#if LIBACCINT_MAX_AM >= 3
        case 3:
            launch_eri_kernel_specialized<La, Lb, Lc, 3>(quartet, d_boys_coeffs, d_output, stream);
            return;
#endif
#if LIBACCINT_MAX_AM >= 4
        case 4:
            launch_eri_kernel_specialized<La, Lb, Lc, 4>(quartet, d_boys_coeffs, d_output, stream);
            return;
#endif
        default:
            throw_unsupported_eri_am(La, Lb, Lc, ld);
    }
}

template <int La, int Lb>
void dispatch_eri_lc(
    int lc,
    int ld,
    const basis::ShellSetQuartetDeviceData& quartet,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream) {
    switch (lc) {
        case 0:
            dispatch_eri_ld<La, Lb, 0>(ld, quartet, d_boys_coeffs, d_output, stream);
            return;
#if LIBACCINT_MAX_AM >= 1
        case 1:
            dispatch_eri_ld<La, Lb, 1>(ld, quartet, d_boys_coeffs, d_output, stream);
            return;
#endif
#if LIBACCINT_MAX_AM >= 2
        case 2:
            dispatch_eri_ld<La, Lb, 2>(ld, quartet, d_boys_coeffs, d_output, stream);
            return;
#endif
#if LIBACCINT_MAX_AM >= 3
        case 3:
            dispatch_eri_ld<La, Lb, 3>(ld, quartet, d_boys_coeffs, d_output, stream);
            return;
#endif
#if LIBACCINT_MAX_AM >= 4
        case 4:
            dispatch_eri_ld<La, Lb, 4>(ld, quartet, d_boys_coeffs, d_output, stream);
            return;
#endif
        default:
            throw_unsupported_eri_am(La, Lb, lc, ld);
    }
}

template <int La>
void dispatch_eri_lb(
    int lb,
    int lc,
    int ld,
    const basis::ShellSetQuartetDeviceData& quartet,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream) {
    switch (lb) {
        case 0:
            dispatch_eri_lc<La, 0>(lc, ld, quartet, d_boys_coeffs, d_output, stream);
            return;
#if LIBACCINT_MAX_AM >= 1
        case 1:
            dispatch_eri_lc<La, 1>(lc, ld, quartet, d_boys_coeffs, d_output, stream);
            return;
#endif
#if LIBACCINT_MAX_AM >= 2
        case 2:
            dispatch_eri_lc<La, 2>(lc, ld, quartet, d_boys_coeffs, d_output, stream);
            return;
#endif
#if LIBACCINT_MAX_AM >= 3
        case 3:
            dispatch_eri_lc<La, 3>(lc, ld, quartet, d_boys_coeffs, d_output, stream);
            return;
#endif
#if LIBACCINT_MAX_AM >= 4
        case 4:
            dispatch_eri_lc<La, 4>(lc, ld, quartet, d_boys_coeffs, d_output, stream);
            return;
#endif
        default:
            throw_unsupported_eri_am(La, lb, lc, ld);
    }
}

}  // namespace

void dispatch_eri_kernel(
    const basis::ShellSetQuartetDeviceData& quartet,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream) {
    const int la = quartet.a.angular_momentum;
    const int lb = quartet.b.angular_momentum;
    const int lc = quartet.c.angular_momentum;
    const int ld = quartet.d.angular_momentum;

    if (la < 0 || lb < 0 || lc < 0 || ld < 0) {
        throw std::invalid_argument("Negative angular momentum not supported for ERI kernel");
    }

    if (la > LIBACCINT_MAX_AM || lb > LIBACCINT_MAX_AM ||
        lc > LIBACCINT_MAX_AM || ld > LIBACCINT_MAX_AM) {
        throw_unsupported_eri_am(la, lb, lc, ld);
    }

    switch (la) {
        case 0:
            dispatch_eri_lb<0>(lb, lc, ld, quartet, d_boys_coeffs, d_output, stream);
            return;
#if LIBACCINT_MAX_AM >= 1
        case 1:
            dispatch_eri_lb<1>(lb, lc, ld, quartet, d_boys_coeffs, d_output, stream);
            return;
#endif
#if LIBACCINT_MAX_AM >= 2
        case 2:
            dispatch_eri_lb<2>(lb, lc, ld, quartet, d_boys_coeffs, d_output, stream);
            return;
#endif
#if LIBACCINT_MAX_AM >= 3
        case 3:
            dispatch_eri_lb<3>(lb, lc, ld, quartet, d_boys_coeffs, d_output, stream);
            return;
#endif
#if LIBACCINT_MAX_AM >= 4
        case 4:
            dispatch_eri_lb<4>(lb, lc, ld, quartet, d_boys_coeffs, d_output, stream);
            return;
#endif
        default:
            throw_unsupported_eri_am(la, lb, lc, ld);
    }
}
void launch_eri_kernel(
    const basis::ShellSetQuartetDeviceData& quartet,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream) {

    dispatch_eri_kernel(quartet, d_boys_coeffs, d_output, stream);
}

size_t eri_output_size(const basis::ShellSetQuartetDeviceData& quartet) {
    const int na = (quartet.a.angular_momentum + 1) * (quartet.a.angular_momentum + 2) / 2;
    const int nb = (quartet.b.angular_momentum + 1) * (quartet.b.angular_momentum + 2) / 2;
    const int nc = (quartet.c.angular_momentum + 1) * (quartet.c.angular_momentum + 2) / 2;
    const int nd = (quartet.d.angular_momentum + 1) * (quartet.d.angular_momentum + 2) / 2;
    return quartet.n_quartets() * na * nb * nc * nd;
}

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
