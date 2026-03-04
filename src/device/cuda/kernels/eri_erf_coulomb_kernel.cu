// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file eri_erf_coulomb_kernel.cu
/// @brief CUDA kernel for erf-attenuated Coulomb ERIs using modified Rys quadrature
///
/// Implements thread-per-quartet erf-Coulomb ERI computation with modified
/// Boys function for range-separated operators.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/kernels/eri_erf_coulomb_kernel_cuda.hpp>
#include <libaccint/memory/device_memory.hpp>

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cmath>

// Include device Boys and Rys function headers
#include "../math/boys_device.cuh"
#include "../math/boys_range_separated_device.cuh"
#include "../math/rys_device.cuh"

namespace libaccint::kernels::cuda {

// ============================================================================
// Constants
// ============================================================================

constexpr double PI_52_ERF = 17.493418327624862846262821679872;
constexpr int MAX_RYS_ROOTS_ERF = 5;
constexpr int MAX_2D_DIM_ERF = 5;

// ============================================================================
// Device Helper Functions
// ============================================================================

__device__ int erf_double_factorial_odd(int n) {
    if (n <= 0) return 1;
    int result = 1;
    for (int i = 1; i <= n; ++i) {
        result *= (2 * i - 1);
    }
    return result;
}

__device__ double erf_norm_correction(int lx, int ly, int lz) {
    const double denom = static_cast<double>(
        erf_double_factorial_odd(lx) *
        erf_double_factorial_odd(ly) *
        erf_double_factorial_odd(lz));
    return rsqrt(denom);
}

// ============================================================================
// Device 2D Rys Recursion (modified for erf-attenuated operator)
// ============================================================================

template <int La, int Lb, int Lc, int Ld>
__device__ void build_2d_rys_erf_device(
    double PA_eff, double QC_eff, double AB, double CD,
    double B10, double B01, double B00,
    double I[MAX_2D_DIM_ERF][MAX_2D_DIM_ERF][MAX_2D_DIM_ERF][MAX_2D_DIM_ERF]) {

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

    #pragma unroll
    for (int a = 0; a < La + Lb; ++a) {
        I[a + 1][0][0][0] = PA_eff * I[a][0][0][0];
        if (a > 0) {
            I[a + 1][0][0][0] += static_cast<double>(a) * B10 * I[a - 1][0][0][0];
        }
    }

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

    // Step 2: HRR to transfer angular momentum to B
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

    // Step 3: HRR to transfer angular momentum to D
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
// CUDA Kernel: erf-Coulomb ERI (thread-per-quartet)
// ============================================================================

template <int La, int Lb, int Lc, int Ld>
__global__ void eri_erf_coulomb_kernel(
    // Shell data (SoA layout)
    const double* __restrict__ exp_a,
    const double* __restrict__ exp_b,
    const double* __restrict__ exp_c,
    const double* __restrict__ exp_d,
    const double* __restrict__ coeff_a,
    const double* __restrict__ coeff_b,
    const double* __restrict__ coeff_c,
    const double* __restrict__ coeff_d,
    const double* __restrict__ center_a_x,
    const double* __restrict__ center_a_y,
    const double* __restrict__ center_a_z,
    const double* __restrict__ center_b_x,
    const double* __restrict__ center_b_y,
    const double* __restrict__ center_b_z,
    const double* __restrict__ center_c_x,
    const double* __restrict__ center_c_y,
    const double* __restrict__ center_c_z,
    const double* __restrict__ center_d_x,
    const double* __restrict__ center_d_y,
    const double* __restrict__ center_d_z,
    int n_prim_a, int n_prim_b, int n_prim_c, int n_prim_d,
    int n_shells_a, int n_shells_b, int n_shells_c, int n_shells_d,
    double omega,  // Range-separation parameter
    const double* __restrict__ d_boys_coeffs,
    double* __restrict__ output) {

    // Thread/quartet indexing
    const int quartet_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_quartets = n_shells_a * n_shells_b * n_shells_c * n_shells_d;

    if (quartet_idx >= n_quartets) return;

    // Decode quartet indices
    int temp = quartet_idx;
    const int id = temp % n_shells_d; temp /= n_shells_d;
    const int ic = temp % n_shells_c; temp /= n_shells_c;
    const int ib = temp % n_shells_b; temp /= n_shells_b;
    const int ia = temp;

    // Number of Cartesian functions
    constexpr int na = ((La + 1) * (La + 2)) / 2;
    constexpr int nb = ((Lb + 1) * (Lb + 2)) / 2;
    constexpr int nc = ((Lc + 1) * (Lc + 2)) / 2;
    constexpr int nd = ((Ld + 1) * (Ld + 2)) / 2;
    constexpr int n_integrals = na * nb * nc * nd;

    // Local accumulator for this quartet
    double eri_local[n_integrals];
    #pragma unroll
    for (int i = 0; i < n_integrals; ++i) {
        eri_local[i] = 0.0;
    }

    // Shell centers
    const double Ax = center_a_x[ia];
    const double Ay = center_a_y[ia];
    const double Az = center_a_z[ia];
    const double Bx = center_b_x[ib];
    const double By = center_b_y[ib];
    const double Bz = center_b_z[ib];
    const double Cx = center_c_x[ic];
    const double Cy = center_c_y[ic];
    const double Cz = center_c_z[ic];
    const double Dx = center_d_x[id];
    const double Dy = center_d_y[id];
    const double Dz = center_d_z[id];

    // HRR displacements
    const double ABx = Ax - Bx;
    const double ABy = Ay - By;
    const double ABz = Az - Bz;
    const double CDx = Cx - Dx;
    const double CDy = Cy - Dy;
    const double CDz = Cz - Dz;

    // Number of Rys roots
    constexpr int n_rys_roots = (La + Lb + Lc + Ld) / 2 + 1;

    // Rys roots/weights storage
    double roots[MAX_RYS_ROOTS_ERF];
    double weights[MAX_RYS_ROOTS_ERF];

    // 2D recursion tables
    double Ix[MAX_2D_DIM_ERF][MAX_2D_DIM_ERF][MAX_2D_DIM_ERF][MAX_2D_DIM_ERF];
    double Iy[MAX_2D_DIM_ERF][MAX_2D_DIM_ERF][MAX_2D_DIM_ERF][MAX_2D_DIM_ERF];
    double Iz[MAX_2D_DIM_ERF][MAX_2D_DIM_ERF][MAX_2D_DIM_ERF][MAX_2D_DIM_ERF];

    // Precompute omega-dependent factors
    const double omega2 = omega * omega;
    const double omega2_ratio = omega2 / (omega2 + 1.0);
    const double sqrt_omega2_ratio = sqrt(omega2_ratio);

    // Four-fold contraction loop
    for (int p = 0; p < n_prim_a; ++p) {
        const double alpha = exp_a[ia * n_prim_a + p];
        const double ca = coeff_a[ia * n_prim_a + p];

        for (int q = 0; q < n_prim_b; ++q) {
            const double beta = exp_b[ib * n_prim_b + q];
            const double cb = coeff_b[ib * n_prim_b + q];

            // Bra Gaussian product
            const double zeta = alpha + beta;
            const double oo_zeta = 1.0 / zeta;
            const double Px = (alpha * Ax + beta * Bx) * oo_zeta;
            const double Py = (alpha * Ay + beta * By) * oo_zeta;
            const double Pz = (alpha * Az + beta * Bz) * oo_zeta;

            const double AB2 = ABx*ABx + ABy*ABy + ABz*ABz;
            const double K_AB = exp(-alpha * beta * oo_zeta * AB2);

            for (int r = 0; r < n_prim_c; ++r) {
                const double gamma_val = exp_c[ic * n_prim_c + r];
                const double cc = coeff_c[ic * n_prim_c + r];

                for (int s = 0; s < n_prim_d; ++s) {
                    const double delta = exp_d[id * n_prim_d + s];
                    const double cd_coeff = coeff_d[id * n_prim_d + s];

                    // Ket Gaussian product
                    const double eta = gamma_val + delta;
                    const double oo_eta = 1.0 / eta;
                    const double Qx = (gamma_val * Cx + delta * Dx) * oo_eta;
                    const double Qy = (gamma_val * Cy + delta * Dy) * oo_eta;
                    const double Qz = (gamma_val * Cz + delta * Dz) * oo_eta;

                    const double CD2 = CDx*CDx + CDy*CDy + CDz*CDz;
                    const double K_CD = exp(-gamma_val * delta * oo_eta * CD2);

                    // Reduced exponent
                    const double rho = zeta * eta / (zeta + eta);

                    // P - Q
                    const double PQx = Px - Qx;
                    const double PQy = Py - Qy;
                    const double PQz = Pz - Qz;
                    const double PQ2 = PQx*PQx + PQy*PQy + PQz*PQz;

                    // Standard T and modified T_eff for erf operator
                    const double T = rho * PQ2;
                    const double T_eff = T * omega2_ratio;

                    // Prefactor with erf scaling
                    const double prefactor = 2.0 * PI_52_ERF /
                        (zeta * eta * sqrt(zeta + eta)) * K_AB * K_CD;
                    const double erf_scale = sqrt_omega2_ratio;
                    const double coeff_full = ca * cb * cc * cd_coeff * prefactor * erf_scale;

                    // Get Rys roots and weights for modified T_eff
                    device::math::rys_compute_device(n_rys_roots, T_eff, d_boys_coeffs, roots, weights);

                    // Loop over Rys roots
                    for (int root = 0; root < n_rys_roots; ++root) {
                        const double u = roots[root];
                        const double w = weights[root];

                        // Modified recursion coefficients for erf-attenuated operator
                        const double rho_over_zeta = rho / zeta;
                        const double rho_over_eta = rho / eta;
                        const double u_eff = u * omega2_ratio;

                        const double B10 = 0.5 * oo_zeta * (1.0 - rho_over_zeta * u_eff);
                        const double B01 = 0.5 * oo_eta * (1.0 - rho_over_eta * u_eff);
                        const double B00 = 0.5 / (zeta + eta) * u_eff;

                        // Effective displacements
                        const double PA_x = Px - Ax - rho_over_zeta * u_eff * PQx;
                        const double PA_y = Py - Ay - rho_over_zeta * u_eff * PQy;
                        const double PA_z = Pz - Az - rho_over_zeta * u_eff * PQz;

                        const double QC_x = Qx - Cx + rho_over_eta * u_eff * PQx;
                        const double QC_y = Qy - Cy + rho_over_eta * u_eff * PQy;
                        const double QC_z = Qz - Cz + rho_over_eta * u_eff * PQz;

                        // Build 2D recursion tables
                        build_2d_rys_erf_device<La, Lb, Lc, Ld>(
                            PA_x, QC_x, ABx, CDx, B10, B01, B00, Ix);
                        build_2d_rys_erf_device<La, Lb, Lc, Ld>(
                            PA_y, QC_y, ABy, CDy, B10, B01, B00, Iy);
                        build_2d_rys_erf_device<La, Lb, Lc, Ld>(
                            PA_z, QC_z, ABz, CDz, B10, B01, B00, Iz);

                        const double wcoeff = coeff_full * w;

                        // Accumulate integrals
                        int idx = 0;
                        #pragma unroll
                        for (int ja = 0; ja < na; ++ja) {
                            // Decode Cartesian indices for shell a
                            int ax, ay, az;
                            device::math::cartesian_index_decode<La>(ja, ax, ay, az);

                            #pragma unroll
                            for (int jb = 0; jb < nb; ++jb) {
                                int bx, by, bz;
                                device::math::cartesian_index_decode<Lb>(jb, bx, by, bz);

                                #pragma unroll
                                for (int jc = 0; jc < nc; ++jc) {
                                    int cx, cy, cz;
                                    device::math::cartesian_index_decode<Lc>(jc, cx, cy, cz);

                                    #pragma unroll
                                    for (int jd = 0; jd < nd; ++jd) {
                                        int dx, dy, dz;
                                        device::math::cartesian_index_decode<Ld>(jd, dx, dy, dz);

                                        const double val =
                                            Ix[ax][bx][cx][dx] *
                                            Iy[ay][by][cy][dy] *
                                            Iz[az][bz][cz][dz];

                                        eri_local[idx] += wcoeff * val;
                                        ++idx;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply normalization corrections and write to global memory
    const int output_offset = quartet_idx * n_integrals;
    int idx = 0;
    #pragma unroll
    for (int ja = 0; ja < na; ++ja) {
        int ax, ay, az;
        device::math::cartesian_index_decode<La>(ja, ax, ay, az);
        const double corr_a = erf_norm_correction(ax, ay, az);

        #pragma unroll
        for (int jb = 0; jb < nb; ++jb) {
            int bx, by, bz;
            device::math::cartesian_index_decode<Lb>(jb, bx, by, bz);
            const double corr_b = erf_norm_correction(bx, by, bz);

            #pragma unroll
            for (int jc = 0; jc < nc; ++jc) {
                int cx, cy, cz;
                device::math::cartesian_index_decode<Lc>(jc, cx, cy, cz);
                const double corr_c = erf_norm_correction(cx, cy, cz);

                #pragma unroll
                for (int jd = 0; jd < nd; ++jd) {
                    int dx, dy, dz;
                    device::math::cartesian_index_decode<Ld>(jd, dx, dy, dz);
                    const double corr_d = erf_norm_correction(dx, dy, dz);

                    output[output_offset + idx] =
                        eri_local[idx] * corr_a * corr_b * corr_c * corr_d;
                    ++idx;
                }
            }
        }
    }
}

// ============================================================================
// Kernel Launch Functions
// ============================================================================

void launch_eri_erf_coulomb_kernel(
    const basis::ShellSetQuartetDeviceData& quartet,
    double omega,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream) {

    dispatch_eri_erf_coulomb_kernel(quartet, omega, d_boys_coeffs, d_output, stream);
}

size_t eri_erf_coulomb_output_size(const basis::ShellSetQuartetDeviceData& quartet) {
    const int na = ((quartet.La + 1) * (quartet.La + 2)) / 2;
    const int nb = ((quartet.Lb + 1) * (quartet.Lb + 2)) / 2;
    const int nc = ((quartet.Lc + 1) * (quartet.Lc + 2)) / 2;
    const int nd = ((quartet.Ld + 1) * (quartet.Ld + 2)) / 2;
    const size_t n_quartets = static_cast<size_t>(quartet.n_shells_a) *
                              quartet.n_shells_b * quartet.n_shells_c * quartet.n_shells_d;
    return n_quartets * na * nb * nc * nd;
}

void dispatch_eri_erf_coulomb_kernel(
    const basis::ShellSetQuartetDeviceData& quartet,
    double omega,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream) {

    const int La = quartet.La;
    const int Lb = quartet.Lb;
    const int Lc = quartet.Lc;
    const int Ld = quartet.Ld;

    const int n_quartets = quartet.n_shells_a * quartet.n_shells_b *
                           quartet.n_shells_c * quartet.n_shells_d;

    constexpr int BLOCK_SIZE = 128;
    const int n_blocks = (n_quartets + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Dispatch macro for supported AM combinations
    #define DISPATCH_ERF_KERNEL(la, lb, lc, ld)                                      \
        if (La == la && Lb == lb && Lc == lc && Ld == ld) {                          \
            eri_erf_coulomb_kernel<la, lb, lc, ld><<<n_blocks, BLOCK_SIZE, 0, stream>>>( \
                quartet.exp_a, quartet.exp_b, quartet.exp_c, quartet.exp_d,           \
                quartet.coeff_a, quartet.coeff_b, quartet.coeff_c, quartet.coeff_d,   \
                quartet.center_a_x, quartet.center_a_y, quartet.center_a_z,           \
                quartet.center_b_x, quartet.center_b_y, quartet.center_b_z,           \
                quartet.center_c_x, quartet.center_c_y, quartet.center_c_z,           \
                quartet.center_d_x, quartet.center_d_y, quartet.center_d_z,           \
                quartet.n_prim_a, quartet.n_prim_b, quartet.n_prim_c, quartet.n_prim_d, \
                quartet.n_shells_a, quartet.n_shells_b, quartet.n_shells_c, quartet.n_shells_d, \
                omega, d_boys_coeffs, d_output);                                      \
            return;                                                                   \
        }

    // Generate dispatch for AM combinations up to d (L=2)
    DISPATCH_ERF_KERNEL(0, 0, 0, 0)
    DISPATCH_ERF_KERNEL(0, 0, 0, 1)
    DISPATCH_ERF_KERNEL(0, 0, 1, 0)
    DISPATCH_ERF_KERNEL(0, 0, 1, 1)
    DISPATCH_ERF_KERNEL(0, 1, 0, 0)
    DISPATCH_ERF_KERNEL(0, 1, 0, 1)
    DISPATCH_ERF_KERNEL(0, 1, 1, 0)
    DISPATCH_ERF_KERNEL(0, 1, 1, 1)
    DISPATCH_ERF_KERNEL(1, 0, 0, 0)
    DISPATCH_ERF_KERNEL(1, 0, 0, 1)
    DISPATCH_ERF_KERNEL(1, 0, 1, 0)
    DISPATCH_ERF_KERNEL(1, 0, 1, 1)
    DISPATCH_ERF_KERNEL(1, 1, 0, 0)
    DISPATCH_ERF_KERNEL(1, 1, 0, 1)
    DISPATCH_ERF_KERNEL(1, 1, 1, 0)
    DISPATCH_ERF_KERNEL(1, 1, 1, 1)
    DISPATCH_ERF_KERNEL(0, 0, 0, 2)
    DISPATCH_ERF_KERNEL(0, 0, 2, 0)
    DISPATCH_ERF_KERNEL(0, 2, 0, 0)
    DISPATCH_ERF_KERNEL(2, 0, 0, 0)
    DISPATCH_ERF_KERNEL(0, 0, 2, 2)
    DISPATCH_ERF_KERNEL(0, 2, 0, 2)
    DISPATCH_ERF_KERNEL(0, 2, 2, 0)
    DISPATCH_ERF_KERNEL(2, 0, 0, 2)
    DISPATCH_ERF_KERNEL(2, 0, 2, 0)
    DISPATCH_ERF_KERNEL(2, 2, 0, 0)
    DISPATCH_ERF_KERNEL(2, 2, 2, 2)

    #undef DISPATCH_ERF_KERNEL

    throw std::runtime_error("Unsupported angular momentum combination for erf-Coulomb ERI: (" +
        std::to_string(La) + "," + std::to_string(Lb) + "," +
        std::to_string(Lc) + "," + std::to_string(Ld) + ")");
}

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
