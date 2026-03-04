// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file eri_kernel_warp.cu
/// @brief Warp-per-quartet ERI kernel implementation

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/kernels/eri_kernel_warp_cuda.hpp>
#include <libaccint/utils/error_handling.hpp>
#include "../math/rys_device.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>

namespace cg = cooperative_groups;

namespace libaccint::kernels::cuda {

constexpr int WARP_SIZE = 32;
constexpr double PI_52 = 17.493418327624862846262821679872;
constexpr int MAX_RYS_ROOTS = 9;
constexpr int MAX_2D_DIM = 9;
constexpr int MAX_CART = 35;

__device__ __constant__ int d_cart_lx[MAX_CART] = {
    0,
    1,0,0,
    2,1,1,0,0,0,
    3,2,2,1,1,1,0,0,0,0,
    4,3,3,2,2,2,1,1,1,1,0,0,0,0,0
};
__device__ __constant__ int d_cart_ly[MAX_CART] = {
    0,
    0,1,0,
    0,1,0,2,1,0,
    0,1,0,2,1,0,3,2,1,0,
    0,1,0,2,1,0,3,2,1,0,4,3,2,1,0
};
__device__ __constant__ int d_cart_lz[MAX_CART] = {
    0,
    0,0,1,
    0,0,1,0,1,2,
    0,0,1,0,1,2,0,1,2,3,
    0,0,1,0,1,2,0,1,2,3,0,1,2,3,4
};

__device__ __forceinline__ int cart_offset(int l) {
    return (l == 0) ? 0 :
           (l == 1) ? 1 :
           (l == 2) ? 4 :
           (l == 3) ? 10 : 20;
}

__device__ __forceinline__ int n_cart(int l) {
    return (l + 1) * (l + 2) / 2;
}

static __device__ int eri_double_factorial_odd(int n) {
    if (n <= 0) return 1;
    int result = 1;
    for (int i = 1; i <= n; ++i) {
        result *= (2 * i - 1);
    }
    return result;
}

static __device__ double eri_norm_correction(int lx, int ly, int lz) {
    const double denom = static_cast<double>(
        eri_double_factorial_odd(lx) *
        eri_double_factorial_odd(ly) *
        eri_double_factorial_odd(lz));
    return rsqrt(denom);
}

__device__ void build_2d_rys_runtime(
    int La, int Lb, int Lc, int Ld,
    double PA_eff, double QC_eff, double AB, double CD,
    double B10, double B01, double B00,
    double I[MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM]) {

    const int dim_a = La + Lb + 1;
    const int dim_c = Lc + Ld + 1;

    for (int a = 0; a < dim_a; ++a) {
        for (int b = 0; b <= Lb; ++b) {
            for (int c = 0; c < dim_c; ++c) {
                for (int d = 0; d <= Ld; ++d) {
                    I[a][b][c][d] = 0.0;
                }
            }
        }
    }

    I[0][0][0][0] = 1.0;

    for (int a = 0; a < La + Lb; ++a) {
        I[a + 1][0][0][0] = PA_eff * I[a][0][0][0];
        if (a > 0) {
            I[a + 1][0][0][0] += static_cast<double>(a) * B10 * I[a - 1][0][0][0];
        }
    }

    for (int c = 0; c < Lc + Ld; ++c) {
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

    for (int b = 0; b < Lb; ++b) {
        for (int a = 0; a <= La + Lb - b - 1; ++a) {
            for (int c = 0; c <= Lc + Ld; ++c) {
                for (int d = 0; d <= Ld; ++d) {
                    I[a][b + 1][c][d] = I[a + 1][b][c][d] + AB * I[a][b][c][d];
                }
            }
        }
    }

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

__global__ void eri_warp_kernel_runtime(
    const double* __restrict__ d_exp_a,
    const double* __restrict__ d_coef_a,
    const double* __restrict__ d_centers_x_a,
    const double* __restrict__ d_centers_y_a,
    const double* __restrict__ d_centers_z_a,
    int n_shells_a,
    int K_a,
    const double* __restrict__ d_exp_b,
    const double* __restrict__ d_coef_b,
    const double* __restrict__ d_centers_x_b,
    const double* __restrict__ d_centers_y_b,
    const double* __restrict__ d_centers_z_b,
    int n_shells_b,
    int K_b,
    const double* __restrict__ d_exp_c,
    const double* __restrict__ d_coef_c,
    const double* __restrict__ d_centers_x_c,
    const double* __restrict__ d_centers_y_c,
    const double* __restrict__ d_centers_z_c,
    int n_shells_c,
    int K_c,
    const double* __restrict__ d_exp_d,
    const double* __restrict__ d_coef_d,
    const double* __restrict__ d_centers_x_d,
    const double* __restrict__ d_centers_y_d,
    const double* __restrict__ d_centers_z_d,
    int n_shells_d,
    int K_d,
    int La,
    int Lb,
    int Lc,
    int Ld,
    const double* __restrict__ d_boys_coeffs,
    double* __restrict__ d_output)
{
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    const int lane_id = warp.thread_rank();
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;

    const int n_quartets = n_shells_a * n_shells_b * n_shells_c * n_shells_d;
    if (warp_id >= n_quartets) return;

    int ia, ib, ic, id;
    if (lane_id == 0) {
        int tmp = warp_id;
        id = tmp % n_shells_d; tmp /= n_shells_d;
        ic = tmp % n_shells_c; tmp /= n_shells_c;
        ib = tmp % n_shells_b; tmp /= n_shells_b;
        ia = tmp;
    }
    ia = __shfl_sync(0xffffffff, ia, 0);
    ib = __shfl_sync(0xffffffff, ib, 0);
    ic = __shfl_sync(0xffffffff, ic, 0);
    id = __shfl_sync(0xffffffff, id, 0);

    double Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz;
    if (lane_id == 0) {
        Ax = d_centers_x_a[ia]; Ay = d_centers_y_a[ia]; Az = d_centers_z_a[ia];
        Bx = d_centers_x_b[ib]; By = d_centers_y_b[ib]; Bz = d_centers_z_b[ib];
        Cx = d_centers_x_c[ic]; Cy = d_centers_y_c[ic]; Cz = d_centers_z_c[ic];
        Dx = d_centers_x_d[id]; Dy = d_centers_y_d[id]; Dz = d_centers_z_d[id];
    }
    Ax = __shfl_sync(0xffffffff, Ax, 0); Ay = __shfl_sync(0xffffffff, Ay, 0); Az = __shfl_sync(0xffffffff, Az, 0);
    Bx = __shfl_sync(0xffffffff, Bx, 0); By = __shfl_sync(0xffffffff, By, 0); Bz = __shfl_sync(0xffffffff, Bz, 0);
    Cx = __shfl_sync(0xffffffff, Cx, 0); Cy = __shfl_sync(0xffffffff, Cy, 0); Cz = __shfl_sync(0xffffffff, Cz, 0);
    Dx = __shfl_sync(0xffffffff, Dx, 0); Dy = __shfl_sync(0xffffffff, Dy, 0); Dz = __shfl_sync(0xffffffff, Dz, 0);

    const int na = n_cart(La);
    const int nb = n_cart(Lb);
    const int nc = n_cart(Lc);
    const int nd = n_cart(Ld);
    const int n_out = na * nb * nc * nd;
    const int n_rys = (La + Lb + Lc + Ld) / 2 + 1;

    const int off_a = cart_offset(La);
    const int off_b = cart_offset(Lb);
    const int off_c = cart_offset(Lc);
    const int off_d = cart_offset(Ld);

    const int n_prims = K_a * K_b * K_c * K_d;

    const double AB_x = Ax - Bx;
    const double AB_y = Ay - By;
    const double AB_z = Az - Bz;
    const double CD_x = Cx - Dx;
    const double CD_y = Cy - Dy;
    const double CD_z = Cz - Dz;

    const int out_base = warp_id * n_out;

    for (int out_idx = lane_id; out_idx < n_out; out_idx += WARP_SIZE) {
        int tmp = out_idx;
        const int d_idx = tmp % nd; tmp /= nd;
        const int c_idx = tmp % nc; tmp /= nc;
        const int b_idx = tmp % nb; tmp /= nb;
        const int a_idx = tmp;

        const int lx_a = d_cart_lx[off_a + a_idx];
        const int ly_a = d_cart_ly[off_a + a_idx];
        const int lz_a = d_cart_lz[off_a + a_idx];
        const int lx_b = d_cart_lx[off_b + b_idx];
        const int ly_b = d_cart_ly[off_b + b_idx];
        const int lz_b = d_cart_lz[off_b + b_idx];
        const int lx_c = d_cart_lx[off_c + c_idx];
        const int ly_c = d_cart_ly[off_c + c_idx];
        const int lz_c = d_cart_lz[off_c + c_idx];
        const int lx_d = d_cart_lx[off_d + d_idx];
        const int ly_d = d_cart_ly[off_d + d_idx];
        const int lz_d = d_cart_lz[off_d + d_idx];

        const double corr = eri_norm_correction(lx_a, ly_a, lz_a) *
                            eri_norm_correction(lx_b, ly_b, lz_b) *
                            eri_norm_correction(lx_c, ly_c, lz_c) *
                            eri_norm_correction(lx_d, ly_d, lz_d);

        double integral_val = 0.0;

        for (int prim_idx = 0; prim_idx < n_prims; ++prim_idx) {
            int ptmp = prim_idx;
            const int pd = ptmp % K_d; ptmp /= K_d;
            const int pc = ptmp % K_c; ptmp /= K_c;
            const int pb = ptmp % K_b; ptmp /= K_b;
            const int pa = ptmp;

            const double alpha = d_exp_a[ia * K_a + pa];
            const double beta = d_exp_b[ib * K_b + pb];
            const double gamma = d_exp_c[ic * K_c + pc];
            const double delta = d_exp_d[id * K_d + pd];

            const double ca = d_coef_a[ia * K_a + pa];
            const double cb = d_coef_b[ib * K_b + pb];
            const double cc = d_coef_c[ic * K_c + pc];
            const double cd = d_coef_d[id * K_d + pd];

            const double zeta = alpha + beta;
            const double eta = gamma + delta;
            const double oo_zeta = 1.0 / zeta;
            const double oo_eta = 1.0 / eta;

            const double Px = (alpha * Ax + beta * Bx) * oo_zeta;
            const double Py = (alpha * Ay + beta * By) * oo_zeta;
            const double Pz = (alpha * Az + beta * Bz) * oo_zeta;

            const double Qx = (gamma * Cx + delta * Dx) * oo_eta;
            const double Qy = (gamma * Cy + delta * Dy) * oo_eta;
            const double Qz = (gamma * Cz + delta * Dz) * oo_eta;

            const double AB2 = AB_x * AB_x + AB_y * AB_y + AB_z * AB_z;
            const double CD2 = CD_x * CD_x + CD_y * CD_y + CD_z * CD_z;
            const double K_AB = exp(-alpha * beta * oo_zeta * AB2);
            const double K_CD = exp(-gamma * delta * oo_eta * CD2);

            const double rho = zeta * eta / (zeta + eta);
            const double PQ_x = Px - Qx;
            const double PQ_y = Py - Qy;
            const double PQ_z = Pz - Qz;
            const double PQ2 = PQ_x * PQ_x + PQ_y * PQ_y + PQ_z * PQ_z;
            const double T = rho * PQ2;

            const double prefactor = 2.0 * PI_52 * oo_zeta * oo_eta *
                rsqrt(zeta + eta) * K_AB * K_CD;
            const double coeff = ca * cb * cc * cd * prefactor;

            double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];
            device::math::rys_quadrature_impl(n_rys, T, d_boys_coeffs, roots, weights);

            for (int root = 0; root < n_rys; ++root) {
                const double u = roots[root];
                const double w = weights[root];

                const double rho_over_zeta = rho / zeta;
                const double rho_over_eta = rho / eta;
                const double B10 = 0.5 * oo_zeta * (1.0 - rho_over_zeta * u);
                const double B01 = 0.5 * oo_eta * (1.0 - rho_over_eta * u);
                const double B00 = 0.5 / (zeta + eta) * u;

                const double PA_x_eff = (Px - Ax) - rho_over_zeta * u * PQ_x;
                const double PA_y_eff = (Py - Ay) - rho_over_zeta * u * PQ_y;
                const double PA_z_eff = (Pz - Az) - rho_over_zeta * u * PQ_z;
                const double QC_x_eff = (Qx - Cx) + rho_over_eta * u * PQ_x;
                const double QC_y_eff = (Qy - Cy) + rho_over_eta * u * PQ_y;
                const double QC_z_eff = (Qz - Cz) + rho_over_eta * u * PQ_z;

                double Ix[MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM];
                double Iy[MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM];
                double Iz[MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM][MAX_2D_DIM];

                build_2d_rys_runtime(La, Lb, Lc, Ld,
                    PA_x_eff, QC_x_eff, AB_x, CD_x, B10, B01, B00, Ix);
                build_2d_rys_runtime(La, Lb, Lc, Ld,
                    PA_y_eff, QC_y_eff, AB_y, CD_y, B10, B01, B00, Iy);
                build_2d_rys_runtime(La, Lb, Lc, Ld,
                    PA_z_eff, QC_z_eff, AB_z, CD_z, B10, B01, B00, Iz);

                integral_val += coeff * w *
                    Ix[lx_a][lx_b][lx_c][lx_d] *
                    Iy[ly_a][ly_b][ly_c][ly_d] *
                    Iz[lz_a][lz_b][lz_c][lz_d];
            }
        }

        d_output[out_base + out_idx] = integral_val * corr;
    }
}

void dispatch_eri_warp_kernel(
    const basis::ShellSetQuartetDeviceData& quartet,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream)
{
    const int la = quartet.a.angular_momentum;
    const int lb = quartet.b.angular_momentum;
    const int lc = quartet.c.angular_momentum;
    const int ld = quartet.d.angular_momentum;

    if (la < 0 || lb < 0 || lc < 0 || ld < 0 ||
        la > 4 || lb > 4 || lc > 4 || ld > 4) {
        throw InvalidArgumentException(
            "Unsupported AM for warp ERI kernel: (" +
            std::to_string(la) + "," + std::to_string(lb) + "|" +
            std::to_string(lc) + "," + std::to_string(ld) + ")");
    }

    const int n_quartets = static_cast<int>(quartet.n_quartets());

    const int warps_per_block = 4;
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const int num_blocks = (n_quartets + warps_per_block - 1) / warps_per_block;

    eri_warp_kernel_runtime<<<num_blocks, threads_per_block, 0, stream>>>(
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
        la, lb, lc, ld,
        d_boys_coeffs,
        d_output);
}

size_t eri_warp_output_size(const basis::ShellSetQuartetDeviceData& quartet) {
    const int n_cart_a = (quartet.a.angular_momentum + 1) * (quartet.a.angular_momentum + 2) / 2;
    const int n_cart_b = (quartet.b.angular_momentum + 1) * (quartet.b.angular_momentum + 2) / 2;
    const int n_cart_c = (quartet.c.angular_momentum + 1) * (quartet.c.angular_momentum + 2) / 2;
    const int n_cart_d = (quartet.d.angular_momentum + 1) * (quartet.d.angular_momentum + 2) / 2;
    return quartet.n_quartets() * n_cart_a * n_cart_b * n_cart_c * n_cart_d;
}

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
