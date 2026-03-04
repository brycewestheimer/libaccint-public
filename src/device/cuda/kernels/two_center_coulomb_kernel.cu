// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file two_center_coulomb_kernel.cu
/// @brief CUDA implementation of two-center Coulomb integrals

#include <libaccint/device/cuda/two_center_coulomb_kernel.cuh>
#include <libaccint/device/cuda/math/boys_function.cuh>
#include <libaccint/device/cuda/math/constants.cuh>

#include <cmath>

namespace libaccint::device::cuda {

namespace {

/// @brief Number of functions for angular momentum L
__device__ __forceinline__ int n_functions(int L) {
    return (L + 1) * (L + 2) / 2;
}

/// @brief Double factorial (2n-1)!! for normalization
__device__ __forceinline__ double double_factorial_odd(int n) {
    if (n <= 0) return 1.0;
    double result = 1.0;
    for (int i = 1; i <= n; i += 2) {
        result *= i;
    }
    return result;
}

/// @brief Normalization correction for Cartesian component
__device__ __forceinline__ double norm_correction(int lx, int ly, int lz) {
    const double denom = double_factorial_odd(2 * lx - 1) *
                         double_factorial_odd(2 * ly - 1) *
                         double_factorial_odd(2 * lz - 1);
    return rsqrt(denom);
}

}  // anonymous namespace

__global__ void two_center_coulomb_kernel(
    const Real* __restrict__ center_x,
    const Real* __restrict__ center_y,
    const Real* __restrict__ center_z,
    const Real* __restrict__ exponents,
    const Real* __restrict__ coefficients,
    const int* __restrict__ angular_momenta,
    const Size* __restrict__ prim_offsets,
    const Size* __restrict__ n_prims_per_shell,
    const Size* __restrict__ func_offsets,
    Size n_shells,
    Size n_aux,
    Real* __restrict__ metric) {

    // Each block handles one shell pair (sP, sQ)
    const Size sP = blockIdx.x;
    const Size sQ = blockIdx.y;

    // Only compute upper triangle (sP <= sQ)
    if (sP > sQ) return;

    // Shell properties
    const int Lp = angular_momenta[sP];
    const int Lq = angular_momenta[sQ];
    const Size np_prim = n_prims_per_shell[sP];
    const Size nq_prim = n_prims_per_shell[sQ];
    const Size prim_off_P = prim_offsets[sP];
    const Size prim_off_Q = prim_offsets[sQ];
    const Size func_off_P = func_offsets[sP];
    const Size func_off_Q = func_offsets[sQ];
    const int nfunc_P = n_functions(Lp);
    const int nfunc_Q = n_functions(Lq);

    // Shell centers
    const Real Px = center_x[sP];
    const Real Py = center_y[sP];
    const Real Pz = center_z[sP];
    const Real Qx = center_x[sQ];
    const Real Qy = center_y[sQ];
    const Real Qz = center_z[sQ];

    // Distance squared
    const Real PQ_x = Px - Qx;
    const Real PQ_y = Py - Qy;
    const Real PQ_z = Pz - Qz;
    const Real PQ2 = PQ_x * PQ_x + PQ_y * PQ_y + PQ_z * PQ_z;

    // Total angular momentum for Boys function
    const int L_total = Lp + Lq;

    // Shared memory for integral accumulation
    extern __shared__ Real shared_integrals[];

    // Thread handles one function pair within the shell pair
    const int tid = threadIdx.x;
    const int n_pairs = nfunc_P * nfunc_Q;

    // Initialize shared memory
    for (int i = tid; i < n_pairs; i += blockDim.x) {
        shared_integrals[i] = 0.0;
    }
    __syncthreads();

    // Loop over primitive pairs (distributed across threads for now)
    for (Size ip = 0; ip < np_prim; ++ip) {
        const Real alpha_p = exponents[prim_off_P + ip];
        const Real coef_p = coefficients[prim_off_P + ip];

        for (Size iq = 0; iq < nq_prim; ++iq) {
            const Real alpha_q = exponents[prim_off_Q + iq];
            const Real coef_q = coefficients[prim_off_Q + iq];

            // Combined exponent
            const Real zeta = alpha_p + alpha_q;
            const Real rho = alpha_p * alpha_q / zeta;

            // Gaussian product center
            const Real Wx = (alpha_p * Px + alpha_q * Qx) / zeta;
            const Real Wy = (alpha_p * Py + alpha_q * Qy) / zeta;
            const Real Wz = (alpha_p * Pz + alpha_q * Qz) / zeta;

            // Boys function argument
            const Real T = rho * PQ2;

            // Prefactor
            const Real K_PQ = exp(-rho * PQ2);
            const Real prefactor = 2.0 * pow(M_PI, 2.5) / 
                                   (zeta * sqrt(zeta)) * K_PQ * coef_p * coef_q;

            // Compute Boys function values (simple approximation for now)
            Real Fm[8];  // Support up to L=7
            compute_boys_array(L_total, T, Fm);

            // Recursion parameters
            const Real one_over_2zeta = 0.5 / zeta;
            const Real WP_x = Wx - Px;
            const Real WP_y = Wy - Py;
            const Real WP_z = Wz - Pz;
            const Real WQ_x = Wx - Qx;
            const Real WQ_y = Wy - Qy;
            const Real WQ_z = Wz - Qz;

            // For each function pair, compute contribution
            // (Simplified for s and p shells; full implementation would
            // build recursion tables in shared memory)
            for (int fp = tid; fp < nfunc_P; fp += blockDim.x) {
                for (int fq = 0; fq < nfunc_Q; ++fq) {
                    // For demonstration, assume s-shell only
                    // Full implementation requires Cartesian index lookup
                    Real integral = prefactor * Fm[0];
                    
                    atomicAdd(&shared_integrals[fp * nfunc_Q + fq], integral);
                }
            }
        }
    }

    __syncthreads();

    // Write results to global memory
    for (int i = tid; i < n_pairs; i += blockDim.x) {
        const int fp = i / nfunc_Q;
        const int fq = i % nfunc_Q;
        const Size global_p = func_off_P + fp;
        const Size global_q = func_off_Q + fq;

        const Real val = shared_integrals[i];

        // Store in both upper and lower triangle (symmetric)
        metric[global_p * n_aux + global_q] = val;
        if (sP != sQ) {
            metric[global_q * n_aux + global_p] = val;
        }
    }
}

void launch_two_center_coulomb_kernel(
    const Real* center_x,
    const Real* center_y,
    const Real* center_z,
    const Real* exponents,
    const Real* coefficients,
    const int* angular_momenta,
    const Size* prim_offsets,
    const Size* n_prims_per_shell,
    const Size* func_offsets,
    Size n_shells,
    Size n_aux,
    Real* metric,
    cudaStream_t stream) {

    // Grid: one block per shell pair
    dim3 grid(n_shells, n_shells);
    dim3 block(256);  // 256 threads per block

    // Shared memory for function pair integrals
    // Maximum 16 x 16 = 256 pairs for up to f-shells
    const size_t shared_mem = 256 * sizeof(Real);

    two_center_coulomb_kernel<<<grid, block, shared_mem, stream>>>(
        center_x, center_y, center_z,
        exponents, coefficients,
        angular_momenta, prim_offsets, n_prims_per_shell, func_offsets,
        n_shells, n_aux, metric);
}

}  // namespace libaccint::device::cuda
