// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file fused_1e_kernel.cu
/// @brief CUDA fused S+T+V one-electron integral kernel implementation
///
/// Computes overlap (S), kinetic (T), and nuclear attraction (V) integrals
/// simultaneously for a ShellSetPair, sharing Gaussian product computation
/// and recursion tables to eliminate redundant work and reduce kernel launches.

#include <libaccint/kernels/fused_1e_kernel_cuda.hpp>
#include <libaccint/memory/device_memory.hpp>
#include <libaccint/utils/error_handling.hpp>

#if LIBACCINT_USE_CUDA

#include <cmath>

// Include device Boys function header
#include "../../cuda/math/boys_device.cuh"

namespace libaccint::kernels::cuda {

using libaccint::device::math::boys_evaluate_array_device;

// ============================================================================
// Device Constants
// ============================================================================

constexpr int FUSED_MAX_AM = 4;
constexpr int FUSED_MAX_AM_PLUS_1 = FUSED_MAX_AM + 1;
constexpr int FUSED_MAX_AM_PLUS_3 = FUSED_MAX_AM + 3;

/// Maximum Rys roots for nuclear integrals: (La + Lb)/2 + 1
/// For (g|g): (4+4)/2 + 1 = 5
constexpr int FUSED_MAX_RYS_ROOTS = 6;

/// Double factorial (2n-1)!! lookup table for normalization
__device__ __constant__ int d_df_odd_fused[FUSED_MAX_AM + 1] = {1, 1, 3, 15, 105};

// ============================================================================
// Device Helper Functions
// ============================================================================

__device__ __forceinline__ double norm_correction_fused(int lx, int ly, int lz) {
    double denom = static_cast<double>(
        d_df_odd_fused[lx] *
        d_df_odd_fused[ly] *
        d_df_odd_fused[lz]);
    return rsqrt(denom);
}

__device__ __forceinline__ void gaussian_product_fused(
    double alpha, double Ax, double Ay, double Az,
    double beta, double Bx, double By, double Bz,
    double& zeta, double& inv_zeta, double& one_over_2zeta,
    double& Px, double& Py, double& Pz,
    double& K_AB) {

    zeta = alpha + beta;
    inv_zeta = 1.0 / zeta;
    one_over_2zeta = 0.5 * inv_zeta;

    Px = (alpha * Ax + beta * Bx) * inv_zeta;
    Py = (alpha * Ay + beta * By) * inv_zeta;
    Pz = (alpha * Az + beta * Bz) * inv_zeta;

    double dx = Ax - Bx;
    double dy = Ay - By;
    double dz = Az - Bz;
    double AB2 = dx * dx + dy * dy + dz * dz;

    double mu = alpha * beta * inv_zeta;
    K_AB = exp(-mu * AB2);
}

/// @brief Build extended 1D overlap recursion table (shared by S and T)
///
/// Extended to Lb+2 for the kinetic energy j+2 shift.
/// Overlap reads subset [0..La][0..Lb], kinetic needs [0..La][0..Lb+2].
template <int La, int Lb>
__device__ __forceinline__ void build_1d_overlap_extended_fused(
    double XPA, double XPB, double one_over_2zeta,
    double I[FUSED_MAX_AM_PLUS_1][FUSED_MAX_AM_PLUS_3]) {

    I[0][0] = 1.0;

    #pragma unroll
    for (int i = 0; i < La; ++i) {
        I[i + 1][0] = XPA * I[i][0];
        if (i > 0) {
            I[i + 1][0] += static_cast<double>(i) * one_over_2zeta * I[i - 1][0];
        }
    }

    constexpr int Lb_ext = Lb + 2;
    #pragma unroll
    for (int j = 0; j < Lb_ext; ++j) {
        #pragma unroll
        for (int i = 0; i <= La; ++i) {
            double val = XPB * I[i][j];
            if (i > 0) {
                val += static_cast<double>(i) * one_over_2zeta * I[i - 1][j];
            }
            if (j > 0) {
                val += static_cast<double>(j) * one_over_2zeta * I[i][j - 1];
            }
            I[i][j + 1] = val;
        }
    }
}

/// @brief 1D kinetic energy contribution
__device__ __forceinline__ double kinetic_1d_fused(
    double S_d[][FUSED_MAX_AM_PLUS_3], int i, int j, double beta) {

    double val = 4.0 * beta * beta * S_d[i][j + 2];
    val -= 2.0 * beta * static_cast<double>(2 * j + 1) * S_d[i][j];
    if (j >= 2) {
        val += static_cast<double>(j * (j - 1)) * S_d[i][j - 2];
    }
    return -0.5 * val;
}

/// @brief Build 1D Rys recursion table for nuclear attraction
template <int La, int Lb>
__device__ __forceinline__ void build_1d_rys_fused(
    double PA_eff, double PB_eff, double B00,
    double I[FUSED_MAX_AM_PLUS_1][FUSED_MAX_AM_PLUS_1]) {

    I[0][0] = 1.0;

    #pragma unroll
    for (int i = 0; i < La; ++i) {
        I[i + 1][0] = PA_eff * I[i][0];
        if (i > 0) {
            I[i + 1][0] += static_cast<double>(i) * B00 * I[i - 1][0];
        }
    }

    #pragma unroll
    for (int j = 0; j < Lb; ++j) {
        #pragma unroll
        for (int i = 0; i <= La; ++i) {
            double val = PB_eff * I[i][j];
            if (i > 0) {
                val += static_cast<double>(i) * B00 * I[i - 1][j];
            }
            if (j > 0) {
                val += static_cast<double>(j) * B00 * I[i][j - 1];
            }
            I[i][j + 1] = val;
        }
    }
}

// ============================================================================
// Rys Quadrature (duplicated from nuclear_kernel.cu to avoid ODR conflicts)
// ============================================================================

__device__ void fused_rys_chebyshev(int n, const double* moments,
                                     double* alpha, double* beta) {
    const int n2 = 2 * n;
    double sigma[3][2 * FUSED_MAX_RYS_ROOTS];

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < n2; ++j)
            sigma[i][j] = 0.0;

    int r_m2 = 0, r_m1 = 1, r_cur = 2;

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

__device__ void fused_tridiag_ql(int n, double* diag, double* offdiag, double* z) {
    if (n == 1) return;

    offdiag[n - 1] = 0.0;

    for (int l = 0; l < n; ++l) {
        int iter = 0;

        while (true) {
            int m = l;
            while (m < n - 1) {
                double tst = fabs(diag[m]) + fabs(diag[m + 1]);
                if (tst + fabs(offdiag[m]) == tst) break;
                ++m;
            }

            if (m == l) break;
            if (++iter > 50) break;

            double g = (diag[l + 1] - diag[l]) / (2.0 * offdiag[l]);
            double r = sqrt(g * g + 1.0);
            g = diag[m] - diag[l] + offdiag[l] / (g + copysign(r, g));

            double s = 1.0, c = 1.0, p = 0.0;

            for (int i = m - 1; i >= l; --i) {
                double f = s * offdiag[i];
                double b = c * offdiag[i];

                if (fabs(f) >= fabs(g)) {
                    c = g / f;
                    r = sqrt(c * c + 1.0);
                    offdiag[i + 1] = f * r;
                    s = 1.0 / r;
                    c *= s;
                } else {
                    s = f / g;
                    r = sqrt(s * s + 1.0);
                    offdiag[i + 1] = g * r;
                    c = 1.0 / r;
                    s *= c;
                }

                g = diag[i + 1] - p;
                r = (diag[i] - g) * s + 2.0 * c * b;
                p = s * r;
                diag[i + 1] = g + p;
                g = c * r - b;

                f = z[i + 1];
                z[i + 1] = s * z[i] + c * f;
                z[i] = c * z[i] - s * f;
            }

            diag[l] -= p;
            offdiag[l] = g;
            offdiag[m] = 0.0;
        }
    }
}

__device__ void fused_rys_quadrature(
    int n_roots, double T, const double* boys_coeffs,
    double* roots, double* weights) {

    double F[2 * FUSED_MAX_RYS_ROOTS];
    boys_evaluate_array_device(2 * n_roots - 1, T, F, boys_coeffs);

    if (n_roots == 1) {
        roots[0] = (F[0] > 1e-30) ? F[1] / F[0] : 0.5;
        weights[0] = F[0];
        return;
    }

    double alpha[FUSED_MAX_RYS_ROOTS];
    double beta[FUSED_MAX_RYS_ROOTS];
    fused_rys_chebyshev(n_roots, F, alpha, beta);

    double diag[FUSED_MAX_RYS_ROOTS];
    double offdiag[FUSED_MAX_RYS_ROOTS];
    double z[FUSED_MAX_RYS_ROOTS];

    for (int i = 0; i < n_roots; ++i) {
        diag[i] = alpha[i];
        offdiag[i] = (i > 0) ? sqrt(max(0.0, beta[i])) : 0.0;
        z[i] = (i == 0) ? 1.0 : 0.0;
    }

    fused_tridiag_ql(n_roots, diag, offdiag, z);

    for (int i = 0; i < n_roots - 1; ++i) {
        for (int j = i + 1; j < n_roots; ++j) {
            if (diag[i] > diag[j]) {
                double tmp = diag[i]; diag[i] = diag[j]; diag[j] = tmp;
                tmp = z[i]; z[i] = z[j]; z[j] = tmp;
            }
        }
    }

    for (int i = 0; i < n_roots; ++i) {
        roots[i] = max(1e-14, min(diag[i], 1.0 - 1e-14));
        weights[i] = max(0.0, beta[0] * z[i] * z[i]);
    }
}

// ============================================================================
// Fused S+T+V Kernel
// ============================================================================

template <int La, int Lb>
__global__ void fused_1e_kernel(
    // Bra shell data
    const double* __restrict__ bra_exponents,
    const double* __restrict__ bra_coefficients,
    const double* __restrict__ bra_centers_x,
    const double* __restrict__ bra_centers_y,
    const double* __restrict__ bra_centers_z,
    int n_bra_shells,
    int n_bra_primitives,
    // Ket shell data
    const double* __restrict__ ket_exponents,
    const double* __restrict__ ket_coefficients,
    const double* __restrict__ ket_centers_x,
    const double* __restrict__ ket_centers_y,
    const double* __restrict__ ket_centers_z,
    int n_ket_shells,
    int n_ket_primitives,
    // Point charges
    const double* __restrict__ charge_x,
    const double* __restrict__ charge_y,
    const double* __restrict__ charge_z,
    const double* __restrict__ charges,
    int n_charges,
    // Boys function coefficients
    const double* __restrict__ boys_coeffs,
    // Outputs (three separate buffers)
    double* __restrict__ output_S,
    double* __restrict__ output_T,
    double* __restrict__ output_V) {

    constexpr int n_a = ((La + 1) * (La + 2)) / 2;
    constexpr int n_b = ((Lb + 1) * (Lb + 2)) / 2;
    constexpr int n_ab = n_a * n_b;

    constexpr int n_rys_roots = (La + Lb) / 2 + 1;

    // Each block handles one shell pair
    const int shell_pair_idx = blockIdx.x;
    const int bra_shell = shell_pair_idx / n_ket_shells;
    const int ket_shell = shell_pair_idx % n_ket_shells;

    if (bra_shell >= n_bra_shells) return;

    // Each thread handles one Cartesian component pair
    const int tid = threadIdx.x;
    if (tid >= n_ab) return;

    const int a_idx = tid / n_b;
    const int b_idx = tid % n_b;

    // Shell centers
    const double Ax = bra_centers_x[bra_shell];
    const double Ay = bra_centers_y[bra_shell];
    const double Az = bra_centers_z[bra_shell];
    const double Bx = ket_centers_x[ket_shell];
    const double By = ket_centers_y[ket_shell];
    const double Bz = ket_centers_z[ket_shell];

    // Decode Cartesian indices for bra
    int lx_a, ly_a, lz_a;
    {
        int idx = 0;
        for (int lx = La; lx >= 0; --lx) {
            for (int ly = La - lx; ly >= 0; --ly) {
                if (idx == a_idx) {
                    lx_a = lx;
                    ly_a = ly;
                    lz_a = La - lx - ly;
                    goto done_a;
                }
                ++idx;
            }
        }
        done_a:;
    }

    // Decode Cartesian indices for ket
    int lx_b, ly_b, lz_b;
    {
        int idx = 0;
        for (int lx = Lb; lx >= 0; --lx) {
            for (int ly = Lb - lx; ly >= 0; --ly) {
                if (idx == b_idx) {
                    lx_b = lx;
                    ly_b = ly;
                    lz_b = Lb - lx - ly;
                    goto done_b;
                }
                ++idx;
            }
        }
        done_b:;
    }

    // Accumulators for all three integral types
    double integral_S = 0.0;
    double integral_T = 0.0;
    double integral_V = 0.0;

    // Extended recursion tables (shared by S and T)
    double Ix[FUSED_MAX_AM_PLUS_1][FUSED_MAX_AM_PLUS_3];
    double Iy[FUSED_MAX_AM_PLUS_1][FUSED_MAX_AM_PLUS_3];
    double Iz[FUSED_MAX_AM_PLUS_1][FUSED_MAX_AM_PLUS_3];

    // Rys recursion tables (reused for each charge center and root)
    double Ix_rys[FUSED_MAX_AM_PLUS_1][FUSED_MAX_AM_PLUS_1];
    double Iy_rys[FUSED_MAX_AM_PLUS_1][FUSED_MAX_AM_PLUS_1];
    double Iz_rys[FUSED_MAX_AM_PLUS_1][FUSED_MAX_AM_PLUS_1];

    double rys_roots[FUSED_MAX_RYS_ROOTS];
    double rys_weights[FUSED_MAX_RYS_ROOTS];

    constexpr double PI = 3.14159265358979323846;

    // Loop over primitive pairs
    for (int p = 0; p < n_bra_primitives; ++p) {
        const double alpha = bra_exponents[bra_shell * n_bra_primitives + p];
        const double ca = bra_coefficients[bra_shell * n_bra_primitives + p];

        for (int q = 0; q < n_ket_primitives; ++q) {
            const double beta = ket_exponents[ket_shell * n_ket_primitives + q];
            const double cb = ket_coefficients[ket_shell * n_ket_primitives + q];

            // ---- Shared: Gaussian product (computed ONCE) ----
            double zeta, inv_zeta, one_over_2zeta;
            double Px, Py, Pz;
            double K_AB;
            gaussian_product_fused(alpha, Ax, Ay, Az, beta, Bx, By, Bz,
                                   zeta, inv_zeta, one_over_2zeta, Px, Py, Pz, K_AB);

            // Shared displacements (computed ONCE)
            const double XPA_x = Px - Ax;
            const double XPA_y = Py - Ay;
            const double XPA_z = Pz - Az;
            const double XPB_x = Px - Bx;
            const double XPB_y = Py - By;
            const double XPB_z = Pz - Bz;

            // ---- Overlap + Kinetic: shared extended recursion tables ----
            build_1d_overlap_extended_fused<La, Lb>(XPA_x, XPB_x, one_over_2zeta, Ix);
            build_1d_overlap_extended_fused<La, Lb>(XPA_y, XPB_y, one_over_2zeta, Iy);
            build_1d_overlap_extended_fused<La, Lb>(XPA_z, XPB_z, one_over_2zeta, Iz);

            // Overlap + Kinetic prefactor: (pi/zeta)^(3/2) * K_AB
            const double prefactor_ST = pow(PI * inv_zeta, 1.5) * K_AB;
            const double prim_coeff = ca * cb * prefactor_ST;

            // Overlap: S = Ix[lx_a][lx_b] * Iy[ly_a][ly_b] * Iz[lz_a][lz_b]
            const double Sx = Ix[lx_a][lx_b];
            const double Sy = Iy[ly_a][ly_b];
            const double Sz = Iz[lz_a][lz_b];
            integral_S += prim_coeff * Sx * Sy * Sz;

            // Kinetic: T = Tx*Sy*Sz + Sx*Ty*Sz + Sx*Sy*Tz
            const double Tx = kinetic_1d_fused(Ix, lx_a, lx_b, beta);
            const double Ty = kinetic_1d_fused(Iy, ly_a, ly_b, beta);
            const double Tz = kinetic_1d_fused(Iz, lz_a, lz_b, beta);
            integral_T += prim_coeff * (Tx * Sy * Sz + Sx * Ty * Sz + Sx * Sy * Tz);

            // ---- Nuclear attraction: Rys quadrature per charge center ----
            const double prefactor_V = (2.0 * PI * inv_zeta) * K_AB;
            const double B00_base = 0.5 * inv_zeta;

            for (int c = 0; c < n_charges; ++c) {
                const double Z_C = charges[c];
                if (Z_C == 0.0) continue;

                const double PC_x = Px - charge_x[c];
                const double PC_y = Py - charge_y[c];
                const double PC_z = Pz - charge_z[c];

                const double T_arg = zeta * (PC_x * PC_x + PC_y * PC_y + PC_z * PC_z);

                fused_rys_quadrature(n_rys_roots, T_arg, boys_coeffs, rys_roots, rys_weights);

                #pragma unroll
                for (int r = 0; r < n_rys_roots; ++r) {
                    const double u = rys_roots[r];
                    const double w = rys_weights[r];

                    const double B00 = B00_base * (1.0 - u);

                    const double PA_x_eff = XPA_x - u * PC_x;
                    const double PA_y_eff = XPA_y - u * PC_y;
                    const double PA_z_eff = XPA_z - u * PC_z;
                    const double PB_x_eff = XPB_x - u * PC_x;
                    const double PB_y_eff = XPB_y - u * PC_y;
                    const double PB_z_eff = XPB_z - u * PC_z;

                    build_1d_rys_fused<La, Lb>(PA_x_eff, PB_x_eff, B00, Ix_rys);
                    build_1d_rys_fused<La, Lb>(PA_y_eff, PB_y_eff, B00, Iy_rys);
                    build_1d_rys_fused<La, Lb>(PA_z_eff, PB_z_eff, B00, Iz_rys);

                    const double coeff = -Z_C * ca * cb * prefactor_V * w;
                    const double val = Ix_rys[lx_a][lx_b] * Iy_rys[ly_a][ly_b] * Iz_rys[lz_a][lz_b];
                    integral_V += coeff * val;
                }
            }
        }
    }

    // Apply normalization correction
    const double corr_a = norm_correction_fused(lx_a, ly_a, lz_a);
    const double corr_b = norm_correction_fused(lx_b, ly_b, lz_b);
    const double norm = corr_a * corr_b;

    // Write all three outputs
    const size_t out_idx = static_cast<size_t>(shell_pair_idx) * n_ab + tid;
    output_S[out_idx] = integral_S * norm;
    output_T[out_idx] = integral_T * norm;
    output_V[out_idx] = integral_V * norm;
}

// ============================================================================
// Kernel Launch Functions
// ============================================================================

template <int La, int Lb>
void launch_fused_1e_kernel_specialized(
    const basis::ShellSetPairDeviceData& pair,
    const operators::DevicePointChargeData& charges,
    const double* d_boys_coeffs,
    const Fused1eOutputPointers& output,
    cudaStream_t stream) {

    constexpr int n_a = ((La + 1) * (La + 2)) / 2;
    constexpr int n_b = ((Lb + 1) * (Lb + 2)) / 2;
    constexpr int n_ab = n_a * n_b;

    const int n_shell_pairs = pair.n_pairs();
    if (n_shell_pairs == 0) return;

    const int threads_per_block = (n_ab + 31) / 32 * 32;  // Round up to warp size
    const int num_blocks = n_shell_pairs;

    fused_1e_kernel<La, Lb><<<num_blocks, threads_per_block, 0, stream>>>(
        pair.bra.d_exponents, pair.bra.d_coefficients,
        pair.bra.d_centers_x, pair.bra.d_centers_y, pair.bra.d_centers_z,
        pair.bra.n_shells, pair.bra.n_primitives,
        pair.ket.d_exponents, pair.ket.d_coefficients,
        pair.ket.d_centers_x, pair.ket.d_centers_y, pair.ket.d_centers_z,
        pair.ket.n_shells, pair.ket.n_primitives,
        charges.d_x, charges.d_y, charges.d_z, charges.d_charges, charges.n_charges,
        d_boys_coeffs,
        output.d_overlap, output.d_kinetic, output.d_nuclear);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw memory::CudaError(cudaGetErrorString(err), __FILE__, __LINE__);
    }
}

#define INSTANTIATE_FUSED_LB(la) \
template void launch_fused_1e_kernel_specialized<la, 0>(const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&, const double*, const Fused1eOutputPointers&, cudaStream_t); \
template void launch_fused_1e_kernel_specialized<la, 1>(const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&, const double*, const Fused1eOutputPointers&, cudaStream_t); \
template void launch_fused_1e_kernel_specialized<la, 2>(const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&, const double*, const Fused1eOutputPointers&, cudaStream_t); \
template void launch_fused_1e_kernel_specialized<la, 3>(const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&, const double*, const Fused1eOutputPointers&, cudaStream_t); \
template void launch_fused_1e_kernel_specialized<la, 4>(const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&, const double*, const Fused1eOutputPointers&, cudaStream_t)

INSTANTIATE_FUSED_LB(0);
INSTANTIATE_FUSED_LB(1);
INSTANTIATE_FUSED_LB(2);
INSTANTIATE_FUSED_LB(3);
INSTANTIATE_FUSED_LB(4);

#undef INSTANTIATE_FUSED_LB

void dispatch_fused_1e_kernel(
    const basis::ShellSetPairDeviceData& pair,
    const operators::DevicePointChargeData& charges,
    const double* d_boys_coeffs,
    const Fused1eOutputPointers& output,
    cudaStream_t stream) {

    const int La = pair.bra.angular_momentum;
    const int Lb = pair.ket.angular_momentum;

    #define DISPATCH_FUSED_LB(la) \
        switch (Lb) { \
            case 0: launch_fused_1e_kernel_specialized<la, 0>(pair, charges, d_boys_coeffs, output, stream); return; \
            case 1: launch_fused_1e_kernel_specialized<la, 1>(pair, charges, d_boys_coeffs, output, stream); return; \
            case 2: launch_fused_1e_kernel_specialized<la, 2>(pair, charges, d_boys_coeffs, output, stream); return; \
            case 3: launch_fused_1e_kernel_specialized<la, 3>(pair, charges, d_boys_coeffs, output, stream); return; \
            case 4: launch_fused_1e_kernel_specialized<la, 4>(pair, charges, d_boys_coeffs, output, stream); return; \
            default: break; \
        }

    switch (La) {
        case 0: DISPATCH_FUSED_LB(0); break;
        case 1: DISPATCH_FUSED_LB(1); break;
        case 2: DISPATCH_FUSED_LB(2); break;
        case 3: DISPATCH_FUSED_LB(3); break;
        case 4: DISPATCH_FUSED_LB(4); break;
        default: break;
    }

    #undef DISPATCH_FUSED_LB

    throw InvalidArgumentException(
        "Unsupported angular momentum combination for fused 1e kernel: La=" +
        std::to_string(La) + ", Lb=" + std::to_string(Lb));
}

void launch_fused_1e_kernel(
    const basis::ShellSetPairDeviceData& pair,
    const operators::DevicePointChargeData& charges,
    const double* d_boys_coeffs,
    const Fused1eOutputPointers& output,
    cudaStream_t stream) {
    dispatch_fused_1e_kernel(pair, charges, d_boys_coeffs, output, stream);
}

size_t fused_1e_output_size(const basis::ShellSetPairDeviceData& pair) {
    const size_t n_funcs_bra = pair.bra.n_functions_per_shell;
    const size_t n_funcs_ket = pair.ket.n_functions_per_shell;
    const size_t n_pairs = static_cast<size_t>(pair.n_pairs());
    return n_funcs_bra * n_funcs_ket * n_pairs;
}

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
