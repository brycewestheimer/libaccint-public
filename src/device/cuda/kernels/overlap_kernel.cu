// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file overlap_kernel.cu
/// @brief CUDA overlap integral kernel implementation using Obara-Saika recursion

#include <libaccint/kernels/overlap_kernel_cuda.hpp>
#include <libaccint/memory/device_memory.hpp>
#include <libaccint/utils/error_handling.hpp>

#if LIBACCINT_USE_CUDA

#include <cmath>

namespace libaccint::kernels::cuda {

// ============================================================================
// Device Constants
// ============================================================================

/// Maximum angular momentum supported (up to g-type = 4)
constexpr int MAX_AM = 4;

/// Maximum recursion table dimension (MAX_AM + 1)
constexpr int MAX_AM_PLUS_1 = MAX_AM + 1;

/// Number of Cartesian functions for each AM: 1, 3, 6, 10, ...
__device__ __constant__ int d_n_cart_funcs[MAX_AM + 1] = {1, 3, 6, 10, 15};

/// Double factorial (2n-1)!! lookup table for normalization
__device__ __constant__ int d_double_factorial_odd[MAX_AM + 1] = {1, 1, 3, 15, 105};

// ============================================================================
// Device Helper Functions
// ============================================================================

/**
 * @brief Compute normalization correction factor for Cartesian component
 *
 * The factor is 1/sqrt((2lx-1)!! * (2ly-1)!! * (2lz-1)!!)
 */
__device__ __forceinline__ double norm_correction_device(int lx, int ly, int lz) {
    double denom = static_cast<double>(
        d_double_factorial_odd[lx] *
        d_double_factorial_odd[ly] *
        d_double_factorial_odd[lz]);
    return rsqrt(denom);
}

/**
 * @brief Build 1D overlap recursion table for one Cartesian direction
 *
 * Uses the Obara-Saika recursion:
 *   I(i+1, j) = XPA * I(i, j) + 1/(2*zeta) * [i * I(i-1, j) + j * I(i, j-1)]
 *
 * @tparam La Maximum angular momentum for first center
 * @tparam Lb Maximum angular momentum for second center
 */
template <int La, int Lb>
__device__ __forceinline__ void build_1d_overlap_device(
    double XPA, double XPB, double one_over_2zeta,
    double I[MAX_AM_PLUS_1][MAX_AM_PLUS_1]) {

    // Base case
    I[0][0] = 1.0;

    // Build up first index: I(i+1, 0) for j=0
    #pragma unroll
    for (int i = 0; i < La; ++i) {
        I[i + 1][0] = XPA * I[i][0];
        if (i > 0) {
            I[i + 1][0] += static_cast<double>(i) * one_over_2zeta * I[i - 1][0];
        }
    }

    // Build up second index: I(i, j+1) for each j, all i
    #pragma unroll
    for (int j = 0; j < Lb; ++j) {
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

/**
 * @brief Compute Gaussian product on device
 */
__device__ __forceinline__ void gaussian_product_device(
    double alpha, double Ax, double Ay, double Az,
    double beta, double Bx, double By, double Bz,
    double& zeta, double& one_over_2zeta,
    double& Px, double& Py, double& Pz,
    double& K_AB) {

    zeta = alpha + beta;
    double inv_zeta = 1.0 / zeta;
    one_over_2zeta = 0.5 * inv_zeta;

    // Product center
    Px = (alpha * Ax + beta * Bx) * inv_zeta;
    Py = (alpha * Ay + beta * By) * inv_zeta;
    Pz = (alpha * Az + beta * Bz) * inv_zeta;

    // Distance squared between centers
    double dx = Ax - Bx;
    double dy = Ay - By;
    double dz = Az - Bz;
    double AB2 = dx * dx + dy * dy + dz * dz;

    // Reduced exponent and overlap prefactor
    double mu = alpha * beta * inv_zeta;
    K_AB = exp(-mu * AB2);
}

// ============================================================================
// Main Overlap Kernel
// ============================================================================

/**
 * @brief CUDA kernel for computing overlap integrals
 *
 * Threading model:
 * - One block per shell pair (i, j)
 * - One thread per Cartesian component pair (a, b) within the shell pair
 * - Primitives are handled sequentially within each thread
 *
 * @tparam La Angular momentum of bra shells
 * @tparam Lb Angular momentum of ket shells
 */
template <int La, int Lb>
__global__ void overlap_kernel(
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
    // Output
    double* __restrict__ output) {

    // Number of Cartesian functions per shell
    constexpr int n_a = ((La + 1) * (La + 2)) / 2;
    constexpr int n_b = ((Lb + 1) * (Lb + 2)) / 2;
    constexpr int n_ab = n_a * n_b;

    // Each block handles one shell pair
    const int shell_pair_idx = blockIdx.x;
    const int bra_shell = shell_pair_idx / n_ket_shells;
    const int ket_shell = shell_pair_idx % n_ket_shells;

    if (bra_shell >= n_bra_shells) return;

    // Each thread handles one or more Cartesian component pairs
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

    // Cartesian indices for this component pair
    // Use compile-time generated indices
    int lx_a, ly_a, lz_a;
    int lx_b, ly_b, lz_b;

    // Decode Cartesian indices from linear index
    // For a_idx in canonical order (lx descending, ly descending)
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

    // Accumulator for this Cartesian component pair
    double integral = 0.0;

    // Local recursion tables
    double Ix[MAX_AM_PLUS_1][MAX_AM_PLUS_1];
    double Iy[MAX_AM_PLUS_1][MAX_AM_PLUS_1];
    double Iz[MAX_AM_PLUS_1][MAX_AM_PLUS_1];

    // Loop over primitive pairs
    for (int p = 0; p < n_bra_primitives; ++p) {
        const double alpha = bra_exponents[bra_shell * n_bra_primitives + p];
        const double ca = bra_coefficients[bra_shell * n_bra_primitives + p];

        for (int q = 0; q < n_ket_primitives; ++q) {
            const double beta = ket_exponents[ket_shell * n_ket_primitives + q];
            const double cb = ket_coefficients[ket_shell * n_ket_primitives + q];

            // Compute Gaussian product
            double zeta, one_over_2zeta;
            double Px, Py, Pz;
            double K_AB;
            gaussian_product_device(alpha, Ax, Ay, Az, beta, Bx, By, Bz,
                                    zeta, one_over_2zeta, Px, Py, Pz, K_AB);

            // 3D prefactor: (pi/zeta)^(3/2) * K_AB
            constexpr double PI = 3.14159265358979323846;
            const double prefactor = pow(PI / zeta, 1.5) * K_AB;

            // Distances from product center to shell centers
            const double XPA_x = Px - Ax;
            const double XPA_y = Py - Ay;
            const double XPA_z = Pz - Az;
            const double XPB_x = Px - Bx;
            const double XPB_y = Py - By;
            const double XPB_z = Pz - Bz;

            // Build 1D overlap recursion tables
            build_1d_overlap_device<La, Lb>(XPA_x, XPB_x, one_over_2zeta, Ix);
            build_1d_overlap_device<La, Lb>(XPA_y, XPB_y, one_over_2zeta, Iy);
            build_1d_overlap_device<La, Lb>(XPA_z, XPB_z, one_over_2zeta, Iz);

            // Combined primitive coefficient with prefactor
            const double prim_coeff = ca * cb * prefactor;

            // Compute integral for this Cartesian component pair
            const double val = Ix[lx_a][lx_b] * Iy[ly_a][ly_b] * Iz[lz_a][lz_b];
            integral += prim_coeff * val;
        }
    }

    // Apply normalization correction
    const double corr_a = norm_correction_device(lx_a, ly_a, lz_a);
    const double corr_b = norm_correction_device(lx_b, ly_b, lz_b);
    integral *= corr_a * corr_b;

    // Write to output buffer
    // Layout: [shell_pair_idx][a_idx][b_idx]
    const size_t out_idx = static_cast<size_t>(shell_pair_idx) * n_ab + tid;
    output[out_idx] = integral;
}

// ============================================================================
// Kernel Launch Functions
// ============================================================================

template <int La, int Lb>
void launch_overlap_kernel_specialized(
    const basis::ShellSetPairDeviceData& pair,
    double* d_output,
    cudaStream_t stream) {

    constexpr int n_a = ((La + 1) * (La + 2)) / 2;
    constexpr int n_b = ((Lb + 1) * (Lb + 2)) / 2;
    constexpr int n_ab = n_a * n_b;

    const int n_shell_pairs = pair.n_pairs();
    if (n_shell_pairs == 0) return;

    // Configure kernel launch
    // One block per shell pair, threads handle Cartesian components
    const int threads_per_block = (n_ab + 31) / 32 * 32;  // Round up to warp size
    const int num_blocks = n_shell_pairs;

    overlap_kernel<La, Lb><<<num_blocks, threads_per_block, 0, stream>>>(
        pair.bra.d_exponents, pair.bra.d_coefficients,
        pair.bra.d_centers_x, pair.bra.d_centers_y, pair.bra.d_centers_z,
        pair.bra.n_shells, pair.bra.n_primitives,
        pair.ket.d_exponents, pair.ket.d_coefficients,
        pair.ket.d_centers_x, pair.ket.d_centers_y, pair.ket.d_centers_z,
        pair.ket.n_shells, pair.ket.n_primitives,
        d_output);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw memory::CudaError(cudaGetErrorString(err), __FILE__, __LINE__);
    }
}

// Explicit instantiations for all supported AM pairs
template void launch_overlap_kernel_specialized<0, 0>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<0, 1>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<0, 2>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<1, 0>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<1, 1>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<1, 2>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<2, 0>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<2, 1>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<2, 2>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<0, 3>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<0, 4>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<1, 3>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<1, 4>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<2, 3>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<2, 4>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<3, 0>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<3, 1>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<3, 2>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<3, 3>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<3, 4>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<4, 0>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<4, 1>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<4, 2>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<4, 3>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);
template void launch_overlap_kernel_specialized<4, 4>(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t);

void dispatch_overlap_kernel(
    const basis::ShellSetPairDeviceData& pair,
    double* d_output,
    cudaStream_t stream) {

    const int La = pair.bra.angular_momentum;
    const int Lb = pair.ket.angular_momentum;

    if (La < 0 || La > 4 || Lb < 0 || Lb > 4) {
        throw InvalidArgumentException(
            "Unsupported angular momentum combination: La=" +
            std::to_string(La) + ", Lb=" + std::to_string(Lb));
    }

    // Dispatch using computed index
    const int am_index = La * 5 + Lb;
    switch (am_index) {
        case  0: launch_overlap_kernel_specialized<0, 0>(pair, d_output, stream); break;
        case  1: launch_overlap_kernel_specialized<0, 1>(pair, d_output, stream); break;
        case  2: launch_overlap_kernel_specialized<0, 2>(pair, d_output, stream); break;
        case  3: launch_overlap_kernel_specialized<0, 3>(pair, d_output, stream); break;
        case  4: launch_overlap_kernel_specialized<0, 4>(pair, d_output, stream); break;
        case  5: launch_overlap_kernel_specialized<1, 0>(pair, d_output, stream); break;
        case  6: launch_overlap_kernel_specialized<1, 1>(pair, d_output, stream); break;
        case  7: launch_overlap_kernel_specialized<1, 2>(pair, d_output, stream); break;
        case  8: launch_overlap_kernel_specialized<1, 3>(pair, d_output, stream); break;
        case  9: launch_overlap_kernel_specialized<1, 4>(pair, d_output, stream); break;
        case 10: launch_overlap_kernel_specialized<2, 0>(pair, d_output, stream); break;
        case 11: launch_overlap_kernel_specialized<2, 1>(pair, d_output, stream); break;
        case 12: launch_overlap_kernel_specialized<2, 2>(pair, d_output, stream); break;
        case 13: launch_overlap_kernel_specialized<2, 3>(pair, d_output, stream); break;
        case 14: launch_overlap_kernel_specialized<2, 4>(pair, d_output, stream); break;
        case 15: launch_overlap_kernel_specialized<3, 0>(pair, d_output, stream); break;
        case 16: launch_overlap_kernel_specialized<3, 1>(pair, d_output, stream); break;
        case 17: launch_overlap_kernel_specialized<3, 2>(pair, d_output, stream); break;
        case 18: launch_overlap_kernel_specialized<3, 3>(pair, d_output, stream); break;
        case 19: launch_overlap_kernel_specialized<3, 4>(pair, d_output, stream); break;
        case 20: launch_overlap_kernel_specialized<4, 0>(pair, d_output, stream); break;
        case 21: launch_overlap_kernel_specialized<4, 1>(pair, d_output, stream); break;
        case 22: launch_overlap_kernel_specialized<4, 2>(pair, d_output, stream); break;
        case 23: launch_overlap_kernel_specialized<4, 3>(pair, d_output, stream); break;
        case 24: launch_overlap_kernel_specialized<4, 4>(pair, d_output, stream); break;
        default:
            throw InvalidArgumentException(
                "Unsupported angular momentum combination: La=" +
                std::to_string(La) + ", Lb=" + std::to_string(Lb));
    }
}

void launch_overlap_kernel(
    const basis::ShellSetPairDeviceData& pair,
    double* d_output,
    cudaStream_t stream) {
    dispatch_overlap_kernel(pair, d_output, stream);
}

size_t overlap_output_size(const basis::ShellSetPairDeviceData& pair) {
    const size_t n_funcs_bra = pair.bra.n_functions_per_shell;
    const size_t n_funcs_ket = pair.ket.n_functions_per_shell;
    const size_t n_pairs = static_cast<size_t>(pair.n_pairs());
    return n_funcs_bra * n_funcs_ket * n_pairs;
}

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
