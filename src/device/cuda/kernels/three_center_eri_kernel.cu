// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file three_center_eri_kernel.cu
/// @brief CUDA implementation of three-center ERI integrals

#include <libaccint/device/cuda/three_center_eri_kernel.cuh>
#include <libaccint/device/cuda/math/boys_function.cuh>
#include <libaccint/device/cuda/math/rys_quadrature.cuh>

#include <cublas_v2.h>
#include <cmath>

namespace libaccint::device::cuda {

namespace {

__device__ __forceinline__ int n_functions(int L) {
    return (L + 1) * (L + 2) / 2;
}

}  // anonymous namespace

__global__ void three_center_eri_kernel(
    const Real* __restrict__ orb_center_x,
    const Real* __restrict__ orb_center_y,
    const Real* __restrict__ orb_center_z,
    const Real* __restrict__ orb_exponents,
    const Real* __restrict__ orb_coefficients,
    const int* __restrict__ orb_angular_momenta,
    const Size* __restrict__ orb_prim_offsets,
    const Size* __restrict__ orb_n_prims_per_shell,
    const Size* __restrict__ orb_func_offsets,
    Size n_orb_shells,
    Size n_orb,
    const Real* __restrict__ aux_center_x,
    const Real* __restrict__ aux_center_y,
    const Real* __restrict__ aux_center_z,
    const Real* __restrict__ aux_exponents,
    const Real* __restrict__ aux_coefficients,
    const int* __restrict__ aux_angular_momenta,
    const Size* __restrict__ aux_prim_offsets,
    const Size* __restrict__ aux_n_prims_per_shell,
    const Size* __restrict__ aux_func_offsets,
    Size n_aux_shells,
    Size n_aux,
    Real* __restrict__ tensor) {

    // Each block handles one shell triple (sa, sb, sP)
    // Flatten 3D grid into linear index
    const Size triple_idx = blockIdx.x + blockIdx.y * gridDim.x;
    const Size n_orb_pairs = n_orb_shells * n_orb_shells;

    if (triple_idx >= n_orb_pairs * n_aux_shells) return;

    const Size sP = triple_idx / n_orb_pairs;
    const Size orb_pair = triple_idx % n_orb_pairs;
    const Size sa = orb_pair / n_orb_shells;
    const Size sb = orb_pair % n_orb_shells;

    // Get shell properties
    const int La = orb_angular_momenta[sa];
    const int Lb = orb_angular_momenta[sb];
    const int Lp = aux_angular_momenta[sP];

    const Size na_prim = orb_n_prims_per_shell[sa];
    const Size nb_prim = orb_n_prims_per_shell[sb];
    const Size np_prim = aux_n_prims_per_shell[sP];

    const Size prim_off_a = orb_prim_offsets[sa];
    const Size prim_off_b = orb_prim_offsets[sb];
    const Size prim_off_P = aux_prim_offsets[sP];

    const Size func_off_a = orb_func_offsets[sa];
    const Size func_off_b = orb_func_offsets[sb];
    const Size func_off_P = aux_func_offsets[sP];

    const int nfunc_a = n_functions(La);
    const int nfunc_b = n_functions(Lb);
    const int nfunc_P = n_functions(Lp);

    // Shell centers
    const Real Ax = orb_center_x[sa];
    const Real Ay = orb_center_y[sa];
    const Real Az = orb_center_z[sa];
    const Real Bx = orb_center_x[sb];
    const Real By = orb_center_y[sb];
    const Real Bz = orb_center_z[sb];
    const Real Px = aux_center_x[sP];
    const Real Py = aux_center_y[sP];
    const Real Pz = aux_center_z[sP];

    // Shared memory for integral accumulation
    extern __shared__ Real shared_integrals[];

    const int tid = threadIdx.x;
    const int n_integrals = nfunc_a * nfunc_b * nfunc_P;

    // Initialize shared memory
    for (int i = tid; i < n_integrals; i += blockDim.x) {
        shared_integrals[i] = 0.0;
    }
    __syncthreads();

    // Bra distance
    const Real AB_x = Ax - Bx;
    const Real AB_y = Ay - By;
    const Real AB_z = Az - Bz;
    const Real AB2 = AB_x * AB_x + AB_y * AB_y + AB_z * AB_z;

    // Total angular momentum
    const int L_total = La + Lb + Lp;
    const int n_roots = (L_total + 2) / 2;

    // Loop over primitive triples
    for (Size ia = 0; ia < na_prim; ++ia) {
        const Real alpha_a = orb_exponents[prim_off_a + ia];
        const Real coef_a = orb_coefficients[prim_off_a + ia];

        for (Size ib = 0; ib < nb_prim; ++ib) {
            const Real alpha_b = orb_exponents[prim_off_b + ib];
            const Real coef_b = orb_coefficients[prim_off_b + ib];

            // Bra Gaussian product
            const Real zeta = alpha_a + alpha_b;
            const Real K_AB = exp(-alpha_a * alpha_b / zeta * AB2);

            // Bra center
            const Real Wx = (alpha_a * Ax + alpha_b * Bx) / zeta;
            const Real Wy = (alpha_a * Ay + alpha_b * By) / zeta;
            const Real Wz = (alpha_a * Az + alpha_b * Bz) / zeta;

            for (Size ip = 0; ip < np_prim; ++ip) {
                const Real alpha_p = aux_exponents[prim_off_P + ip];
                const Real coef_p = aux_coefficients[prim_off_P + ip];

                const Real eta = alpha_p;
                const Real rho = zeta * eta / (zeta + eta);

                // W to P distance
                const Real WP_x = Wx - Px;
                const Real WP_y = Wy - Py;
                const Real WP_z = Wz - Pz;
                const Real WP2 = WP_x * WP_x + WP_y * WP_y + WP_z * WP_z;

                const Real T = rho * WP2;

                // Prefactor
                const Real prefactor = 2.0 * pow(M_PI, 2.5) /
                    (zeta * eta * sqrt(zeta + eta)) *
                    K_AB * coef_a * coef_b * coef_p;

                // Compute Rys roots and weights
                Real roots[8], weights[8];
                compute_rys_roots_weights(n_roots, T, roots, weights);

                // For each integral function triple
                for (int fi = tid; fi < n_integrals; fi += blockDim.x) {
                    const int fa = fi / (nfunc_b * nfunc_P);
                    const int fb = (fi / nfunc_P) % nfunc_b;
                    const int fp = fi % nfunc_P;

                    // Accumulate over Rys roots
                    Real integral = 0.0;
                    for (int r = 0; r < n_roots; ++r) {
                        // Simplified: full implementation needs recursion
                        integral += weights[r] * prefactor;
                    }

                    atomicAdd(&shared_integrals[fi], integral);
                }
            }
        }
    }

    __syncthreads();

    // Write results to global memory
    // Tensor format: (a, b, P) with stride (n_orb * n_aux, n_aux, 1)
    for (int i = tid; i < n_integrals; i += blockDim.x) {
        const int fa = i / (nfunc_b * nfunc_P);
        const int fb = (i / nfunc_P) % nfunc_b;
        const int fp = i % nfunc_P;

        const Size global_a = func_off_a + fa;
        const Size global_b = func_off_b + fb;
        const Size global_P = func_off_P + fp;

        const Size tensor_idx = global_a * n_orb * n_aux + global_b * n_aux + global_P;
        tensor[tensor_idx] = shared_integrals[i];
    }
}

void launch_three_center_eri_kernel(
    const Real* orb_center_x,
    const Real* orb_center_y,
    const Real* orb_center_z,
    const Real* orb_exponents,
    const Real* orb_coefficients,
    const int* orb_angular_momenta,
    const Size* orb_prim_offsets,
    const Size* orb_n_prims_per_shell,
    const Size* orb_func_offsets,
    Size n_orb_shells,
    Size n_orb,
    const Real* aux_center_x,
    const Real* aux_center_y,
    const Real* aux_center_z,
    const Real* aux_exponents,
    const Real* aux_coefficients,
    const int* aux_angular_momenta,
    const Size* aux_prim_offsets,
    const Size* aux_n_prims_per_shell,
    const Size* aux_func_offsets,
    Size n_aux_shells,
    Size n_aux,
    Real* tensor,
    cudaStream_t stream) {

    // Total shell triples
    const Size n_triples = n_orb_shells * n_orb_shells * n_aux_shells;

    // Grid configuration
    const Size blocks_per_dim = 256;
    dim3 grid((n_triples + blocks_per_dim - 1) / blocks_per_dim, 1);
    dim3 block(256);

    // Shared memory: max 16 * 16 * 16 = 4096 integrals for up to f-shells
    const size_t shared_mem = 4096 * sizeof(Real);

    three_center_eri_kernel<<<grid, block, shared_mem, stream>>>(
        orb_center_x, orb_center_y, orb_center_z,
        orb_exponents, orb_coefficients,
        orb_angular_momenta, orb_prim_offsets, orb_n_prims_per_shell, orb_func_offsets,
        n_orb_shells, n_orb,
        aux_center_x, aux_center_y, aux_center_z,
        aux_exponents, aux_coefficients,
        aux_angular_momenta, aux_prim_offsets, aux_n_prims_per_shell, aux_func_offsets,
        n_aux_shells, n_aux,
        tensor);
}

void launch_b_tensor_kernel(
    const Real* three_center,
    const Real* L_inv,
    Real* B_tensor,
    Size n_orb,
    Size n_aux,
    cudaStream_t stream) {

    // B = three_center * L_inv
    // three_center: (n_orb^2) x n_aux
    // L_inv: n_aux x n_aux
    // B: (n_orb^2) x n_aux

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    const Real alpha = 1.0;
    const Real beta = 0.0;
    const int m = n_orb * n_orb;
    const int n = n_aux;
    const int k = n_aux;

    // cublasDgemm: C = alpha * A * B + beta * C
    cublasDgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,  // Note: column-major so dimensions swapped
                &alpha,
                L_inv, n,
                three_center, k,
                &beta,
                B_tensor, n);

    cublasDestroy(handle);
}

void launch_df_j_kernel(
    const Real* B_tensor,
    const Real* D,
    Real* J,
    Size n_orb,
    Size n_aux,
    cudaStream_t stream) {

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    // Step 1: gamma_P = sum_ab D_ab * B_ab^P
    // gamma = D^T * B (treating D as n_orb^2 vector)
    // Actually: gamma[P] = sum_ab D[ab] * B[ab, P]
    // This is a GEMV: gamma = B^T * vec(D)

    std::vector<Real> gamma_host(n_aux);
    Real* gamma_dev;
    cudaMalloc(&gamma_dev, n_aux * sizeof(Real));

    const Real alpha = 1.0;
    const Real beta = 0.0;
    const int m = n_aux;
    const int n = n_orb * n_orb;

    // gamma = B^T * D_vec
    cublasDgemv(handle,
                CUBLAS_OP_T,
                n, m,
                &alpha,
                B_tensor, n,
                D, 1,
                &beta,
                gamma_dev, 1);

    // Step 2: J_ab = sum_P B_ab^P * gamma_P
    // J = B * gamma
    cublasDgemv(handle,
                CUBLAS_OP_N,
                n, m,
                &alpha,
                B_tensor, n,
                gamma_dev, 1,
                &beta,
                J, 1);

    cudaFree(gamma_dev);
    cublasDestroy(handle);
}

void launch_df_k_kernel(
    const Real* B_tensor,
    const Real* D,
    Real* K,
    Size n_orb,
    Size n_aux,
    cudaStream_t stream) {

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    // K_ac = sum_P sum_bd B_ab^P * D_bd * B_cd^P
    //
    // For each P:
    //   X_ad = sum_b B(a,b,P) * D(b,d)     -- X = B_P * D
    //   K_ac += sum_d X_ad * B(c,d,P)      -- K += X * B_P^T

    const Real alpha = 1.0;
    Real beta = 0.0;

    // Zero K
    cudaMemsetAsync(K, 0, n_orb * n_orb * sizeof(Real), stream);

    // Workspace for X
    Real* X;
    cudaMalloc(&X, n_orb * n_orb * sizeof(Real));

    for (Size P = 0; P < n_aux; ++P) {
        // B_P is the P-th "slice" of B tensor
        // B_P[a, b] = B_tensor[a * n_orb * n_aux + b * n_aux + P]
        // We need to extract/view this properly

        // For simplicity, we'll do a batched approach
        // In practice, we'd restructure B for efficient GPU access

        // X = B_P * D
        // K += X * B_P^T

        beta = (P == 0) ? 0.0 : 1.0;

        // This is a simplified version; full implementation would
        // use batched GEMM or restructure memory layout
    }

    cudaFree(X);
    cublasDestroy(handle);
}

}  // namespace libaccint::device::cuda
