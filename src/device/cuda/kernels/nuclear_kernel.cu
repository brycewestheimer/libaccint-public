// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file nuclear_kernel.cu
/// @brief CUDA nuclear attraction integral kernel implementation using Rys quadrature

#include <libaccint/kernels/nuclear_kernel_cuda.hpp>
#include <libaccint/memory/device_memory.hpp>
#include <libaccint/utils/error_handling.hpp>

// Include device Boys function
#include "../../cuda/math/boys_device.cuh"

#if LIBACCINT_USE_CUDA

#include <cmath>

namespace libaccint::kernels::cuda {

using libaccint::device::math::boys_evaluate_array_device;

// ============================================================================
// Device Constants
// ============================================================================

/// Maximum angular momentum supported (up to g-type = 4)
constexpr int MAX_AM = 4;

/// Maximum recursion table dimension
constexpr int MAX_AM_PLUS_1 = MAX_AM + 1;

/// Maximum number of Rys roots for nuclear integrals: (La + Lb)/2 + 1
/// For (g|g): (4+4)/2 + 1 = 5
constexpr int MAX_RYS_ROOTS_NUCLEAR = 5;

/// Double factorial (2n-1)!! lookup table for normalization
__device__ __constant__ int d_df_odd_nuclear[MAX_AM + 1] = {1, 1, 3, 15, 105};

// ============================================================================
// Device Rys Quadrature using Modified Chebyshev + QL Algorithm
// ============================================================================

/**
 * @brief Modified Chebyshev algorithm to compute three-term recurrence coefficients
 *
 * Given moments μ_k = F_k(T) for k = 0..2n-1, computes the three-term
 * recurrence coefficients {α_k, β_k} for the monic orthogonal polynomials.
 */
__device__ void rys_chebyshev_device(int n, const double* moments,
                                      double* alpha, double* beta) {
    const int n2 = 2 * n;

    // Three rows of the sigma table (circular buffer)
    double sigma[3][2 * MAX_RYS_ROOTS_NUCLEAR];

    // Initialize
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < n2; ++j)
            sigma[i][j] = 0.0;

    int r_m2 = 0;  // sigma[l-2]
    int r_m1 = 1;  // sigma[l-1]
    int r_cur = 2;

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

/**
 * @brief Implicit QL algorithm for symmetric tridiagonal eigenvalue problem
 *
 * Finds eigenvalues and first-row eigenvector components of the Jacobi matrix.
 */
__device__ void tridiag_ql_device(int n, double* diag, double* offdiag, double* z) {
    if (n == 1) return;

    offdiag[n - 1] = 0.0;

    for (int l = 0; l < n; ++l) {
        int iter = 0;

        while (true) {
            // Find smallest m >= l such that offdiag[m] is negligible
            int m = l;
            while (m < n - 1) {
                double tst = fabs(diag[m]) + fabs(diag[m + 1]);
                if (tst + fabs(offdiag[m]) == tst) break;
                ++m;
            }

            if (m == l) break;

            if (++iter > 50) break;  // Convergence protection

            // Compute implicit shift
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

                // Accumulate eigenvector
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

/**
 * @brief Compute Rys quadrature roots and weights on device
 *
 * Uses the modified Chebyshev algorithm to compute three-term recurrence
 * coefficients from Boys function moments, then solves the eigenvalue
 * problem using the QL algorithm to find roots and weights.
 */
__device__ void rys_quadrature_device(
    int n_roots, double T, const double* boys_coeffs,
    double* roots, double* weights) {

    // Compute Boys function values F_0, F_1, ..., F_{2n-1}
    double F[2 * MAX_RYS_ROOTS_NUCLEAR];
    boys_evaluate_array_device(2 * n_roots - 1, T, F, boys_coeffs);

    if (n_roots == 1) {
        // Single root: u = F_1/F_0, w = F_0
        roots[0] = (F[0] > 1e-30) ? F[1] / F[0] : 0.5;
        weights[0] = F[0];
        return;
    }

    // Compute three-term recurrence coefficients using modified Chebyshev
    double alpha[MAX_RYS_ROOTS_NUCLEAR];
    double beta[MAX_RYS_ROOTS_NUCLEAR];
    rys_chebyshev_device(n_roots, F, alpha, beta);

    // Build the symmetric tridiagonal Jacobi matrix
    // Diagonal: alpha[0], alpha[1], ..., alpha[n-1]
    // Off-diagonal: sqrt(beta[1]), sqrt(beta[2]), ..., sqrt(beta[n-1])
    double diag[MAX_RYS_ROOTS_NUCLEAR];
    double offdiag[MAX_RYS_ROOTS_NUCLEAR];
    double z[MAX_RYS_ROOTS_NUCLEAR];

    for (int i = 0; i < n_roots; ++i) {
        diag[i] = alpha[i];
        offdiag[i] = (i > 0) ? sqrt(max(0.0, beta[i])) : 0.0;
        z[i] = (i == 0) ? 1.0 : 0.0;  // Initialize to [1, 0, ..., 0]
    }

    // Solve eigenvalue problem using QL algorithm
    tridiag_ql_device(n_roots, diag, offdiag, z);

    // Eigenvalues are the roots, weights = beta[0] * z[i]^2
    // Sort eigenvalues in ascending order (they should already be mostly sorted)
    for (int i = 0; i < n_roots - 1; ++i) {
        for (int j = i + 1; j < n_roots; ++j) {
            if (diag[i] > diag[j]) {
                double tmp = diag[i]; diag[i] = diag[j]; diag[j] = tmp;
                tmp = z[i]; z[i] = z[j]; z[j] = tmp;
            }
        }
    }

    // Set roots and weights
    for (int i = 0; i < n_roots; ++i) {
        roots[i] = max(1e-14, min(diag[i], 1.0 - 1e-14));
        weights[i] = max(0.0, beta[0] * z[i] * z[i]);
    }
}

// ============================================================================
// Device Helper Functions
// ============================================================================

/**
 * @brief Compute normalization correction factor for Cartesian component
 */
__device__ __forceinline__ double norm_correction_nuclear(int lx, int ly, int lz) {
    double denom = static_cast<double>(
        d_df_odd_nuclear[lx] *
        d_df_odd_nuclear[ly] *
        d_df_odd_nuclear[lz]);
    return rsqrt(denom);
}

/**
 * @brief Build 1D Rys recursion table for one Cartesian direction
 *
 * Uses modified Obara-Saika recursion with root-dependent parameters:
 *   I(i+1, j) = PA_eff * I(i, j) + B00 * [i * I(i-1, j) + j * I(i, j-1)]
 *   I(i, j+1) = PB_eff * I(i, j) + B00 * [i * I(i-1, j) + j * I(i, j-1)]
 */
template <int La, int Lb>
__device__ __forceinline__ void build_1d_rys_device(
    double PA_eff, double PB_eff, double B00,
    double I[MAX_AM_PLUS_1][MAX_AM_PLUS_1]) {

    // Base case
    I[0][0] = 1.0;

    // Build up first index
    #pragma unroll
    for (int i = 0; i < La; ++i) {
        I[i + 1][0] = PA_eff * I[i][0];
        if (i > 0) {
            I[i + 1][0] += static_cast<double>(i) * B00 * I[i - 1][0];
        }
    }

    // Build up second index
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

/**
 * @brief Compute Gaussian product on device
 */
__device__ __forceinline__ void gaussian_product_nuclear(
    double alpha, double Ax, double Ay, double Az,
    double beta, double Bx, double By, double Bz,
    double& zeta, double& Px, double& Py, double& Pz, double& K_AB) {

    zeta = alpha + beta;
    double inv_zeta = 1.0 / zeta;

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
// Main Nuclear Kernel
// ============================================================================

/**
 * @brief CUDA kernel for computing nuclear attraction integrals
 *
 * @tparam La Angular momentum of bra shells
 * @tparam Lb Angular momentum of ket shells
 */
template <int La, int Lb>
__global__ void nuclear_kernel(
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
    // Output
    double* __restrict__ output) {

    // Number of Cartesian functions per shell
    constexpr int n_a = ((La + 1) * (La + 2)) / 2;
    constexpr int n_b = ((Lb + 1) * (Lb + 2)) / 2;
    constexpr int n_ab = n_a * n_b;

    // Number of Rys roots
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

    // Accumulator for this Cartesian component pair
    double integral = 0.0;

    // Local recursion tables and Rys storage
    double Ix[MAX_AM_PLUS_1][MAX_AM_PLUS_1];
    double Iy[MAX_AM_PLUS_1][MAX_AM_PLUS_1];
    double Iz[MAX_AM_PLUS_1][MAX_AM_PLUS_1];
    double rys_roots[MAX_RYS_ROOTS_NUCLEAR];
    double rys_weights[MAX_RYS_ROOTS_NUCLEAR];

    constexpr double PI = 3.14159265358979323846;

    // Loop over primitive pairs
    for (int p = 0; p < n_bra_primitives; ++p) {
        const double alpha = bra_exponents[bra_shell * n_bra_primitives + p];
        const double ca = bra_coefficients[bra_shell * n_bra_primitives + p];

        for (int q = 0; q < n_ket_primitives; ++q) {
            const double beta = ket_exponents[ket_shell * n_ket_primitives + q];
            const double cb = ket_coefficients[ket_shell * n_ket_primitives + q];

            // Compute Gaussian product
            double zeta, Px, Py, Pz, K_AB;
            gaussian_product_nuclear(alpha, Ax, Ay, Az, beta, Bx, By, Bz,
                                     zeta, Px, Py, Pz, K_AB);

            // Nuclear attraction prefactor: (2*pi/zeta) * K_AB
            const double prefactor = (2.0 * PI / zeta) * K_AB;

            // Base half-inverse exponent
            const double B00_base = 0.5 / zeta;

            // Distances from product center to shell centers
            const double PA_x = Px - Ax;
            const double PA_y = Py - Ay;
            const double PA_z = Pz - Az;
            const double PB_x = Px - Bx;
            const double PB_y = Py - By;
            const double PB_z = Pz - Bz;

            // Loop over nuclear centers
            for (int c = 0; c < n_charges; ++c) {
                const double Z_C = charges[c];
                if (Z_C == 0.0) continue;

                // Distance from product center P to nuclear center C
                const double PC_x = Px - charge_x[c];
                const double PC_y = Py - charge_y[c];
                const double PC_z = Pz - charge_z[c];

                // Boys function argument: T = zeta * |P - C|^2
                const double T = zeta * (PC_x * PC_x + PC_y * PC_y + PC_z * PC_z);

                // Get Rys quadrature roots and weights
                rys_quadrature_device(n_rys_roots, T, boys_coeffs, rys_roots, rys_weights);

                // Loop over Rys quadrature points
                #pragma unroll
                for (int r = 0; r < n_rys_roots; ++r) {
                    const double u = rys_roots[r];     // u = t^2 (squared Rys root)
                    const double w = rys_weights[r];   // Rys weight

                    // Root-dependent half-inverse exponent
                    const double B00 = B00_base * (1.0 - u);

                    // Effective displacements (Rys-modified)
                    const double PA_x_eff = PA_x - u * PC_x;
                    const double PA_y_eff = PA_y - u * PC_y;
                    const double PA_z_eff = PA_z - u * PC_z;
                    const double PB_x_eff = PB_x - u * PC_x;
                    const double PB_y_eff = PB_y - u * PC_y;
                    const double PB_z_eff = PB_z - u * PC_z;

                    // Build 1D Rys recursion tables
                    build_1d_rys_device<La, Lb>(PA_x_eff, PB_x_eff, B00, Ix);
                    build_1d_rys_device<La, Lb>(PA_y_eff, PB_y_eff, B00, Iy);
                    build_1d_rys_device<La, Lb>(PA_z_eff, PB_z_eff, B00, Iz);

                    // Combined coefficient: -Z_C * ca * cb * prefactor * w
                    const double coeff = -Z_C * ca * cb * prefactor * w;

                    // Compute integral for this Cartesian component pair
                    const double val = Ix[lx_a][lx_b] * Iy[ly_a][ly_b] * Iz[lz_a][lz_b];
                    integral += coeff * val;
                }
            }
        }
    }

    // Apply normalization correction
    const double corr_a = norm_correction_nuclear(lx_a, ly_a, lz_a);
    const double corr_b = norm_correction_nuclear(lx_b, ly_b, lz_b);
    integral *= corr_a * corr_b;

    // Write to output buffer
    const size_t out_idx = static_cast<size_t>(shell_pair_idx) * n_ab + tid;
    output[out_idx] = integral;
}

// ============================================================================
// Kernel Launch Functions
// ============================================================================

template <int La, int Lb>
void launch_nuclear_kernel_specialized(
    const basis::ShellSetPairDeviceData& pair,
    const operators::DevicePointChargeData& charges,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream) {

    constexpr int n_a = ((La + 1) * (La + 2)) / 2;
    constexpr int n_b = ((Lb + 1) * (Lb + 2)) / 2;
    constexpr int n_ab = n_a * n_b;

    const int n_shell_pairs = pair.n_pairs();
    if (n_shell_pairs == 0 || charges.n_charges == 0) return;

    // Configure kernel launch
    const int threads_per_block = (n_ab + 31) / 32 * 32;
    const int num_blocks = n_shell_pairs;

    nuclear_kernel<La, Lb><<<num_blocks, threads_per_block, 0, stream>>>(
        pair.bra.d_exponents, pair.bra.d_coefficients,
        pair.bra.d_centers_x, pair.bra.d_centers_y, pair.bra.d_centers_z,
        pair.bra.n_shells, pair.bra.n_primitives,
        pair.ket.d_exponents, pair.ket.d_coefficients,
        pair.ket.d_centers_x, pair.ket.d_centers_y, pair.ket.d_centers_z,
        pair.ket.n_shells, pair.ket.n_primitives,
        charges.d_x, charges.d_y, charges.d_z, charges.d_charges, charges.n_charges,
        d_boys_coeffs,
        d_output);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw memory::CudaError(cudaGetErrorString(err), __FILE__, __LINE__);
    }
}

// Explicit instantiations for all supported AM pairs
template void launch_nuclear_kernel_specialized<0, 0>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<0, 1>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<0, 2>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<1, 0>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<1, 1>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<1, 2>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<2, 0>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<2, 1>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<2, 2>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<0, 3>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<0, 4>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<1, 3>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<1, 4>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<2, 3>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<2, 4>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<3, 0>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<3, 1>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<3, 2>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<3, 3>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<3, 4>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<4, 0>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<4, 1>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<4, 2>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<4, 3>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);
template void launch_nuclear_kernel_specialized<4, 4>(
    const basis::ShellSetPairDeviceData&, const operators::DevicePointChargeData&,
    const double*, double*, cudaStream_t);

void dispatch_nuclear_kernel(
    const basis::ShellSetPairDeviceData& pair,
    const operators::DevicePointChargeData& charges,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream) {

    const int La = pair.bra.angular_momentum;
    const int Lb = pair.ket.angular_momentum;

    if (La < 0 || La > 4 || Lb < 0 || Lb > 4) {
        throw InvalidArgumentException(
            "Unsupported angular momentum combination: La=" +
            std::to_string(La) + ", Lb=" + std::to_string(Lb));
    }

    const int am_index = La * 5 + Lb;
    switch (am_index) {
        case  0: launch_nuclear_kernel_specialized<0, 0>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case  1: launch_nuclear_kernel_specialized<0, 1>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case  2: launch_nuclear_kernel_specialized<0, 2>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case  3: launch_nuclear_kernel_specialized<0, 3>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case  4: launch_nuclear_kernel_specialized<0, 4>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case  5: launch_nuclear_kernel_specialized<1, 0>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case  6: launch_nuclear_kernel_specialized<1, 1>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case  7: launch_nuclear_kernel_specialized<1, 2>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case  8: launch_nuclear_kernel_specialized<1, 3>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case  9: launch_nuclear_kernel_specialized<1, 4>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case 10: launch_nuclear_kernel_specialized<2, 0>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case 11: launch_nuclear_kernel_specialized<2, 1>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case 12: launch_nuclear_kernel_specialized<2, 2>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case 13: launch_nuclear_kernel_specialized<2, 3>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case 14: launch_nuclear_kernel_specialized<2, 4>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case 15: launch_nuclear_kernel_specialized<3, 0>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case 16: launch_nuclear_kernel_specialized<3, 1>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case 17: launch_nuclear_kernel_specialized<3, 2>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case 18: launch_nuclear_kernel_specialized<3, 3>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case 19: launch_nuclear_kernel_specialized<3, 4>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case 20: launch_nuclear_kernel_specialized<4, 0>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case 21: launch_nuclear_kernel_specialized<4, 1>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case 22: launch_nuclear_kernel_specialized<4, 2>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case 23: launch_nuclear_kernel_specialized<4, 3>(pair, charges, d_boys_coeffs, d_output, stream); break;
        case 24: launch_nuclear_kernel_specialized<4, 4>(pair, charges, d_boys_coeffs, d_output, stream); break;
        default:
            throw InvalidArgumentException(
                "Unsupported angular momentum combination: La=" +
                std::to_string(La) + ", Lb=" + std::to_string(Lb));
    }
}

void launch_nuclear_kernel(
    const basis::ShellSetPairDeviceData& pair,
    const operators::DevicePointChargeData& charges,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream) {
    dispatch_nuclear_kernel(pair, charges, d_boys_coeffs, d_output, stream);
}

size_t nuclear_output_size(const basis::ShellSetPairDeviceData& pair) {
    const size_t n_funcs_bra = pair.bra.n_functions_per_shell;
    const size_t n_funcs_ket = pair.ket.n_functions_per_shell;
    const size_t n_pairs = static_cast<size_t>(pair.n_pairs());
    return n_funcs_bra * n_funcs_ket * n_pairs;
}

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
