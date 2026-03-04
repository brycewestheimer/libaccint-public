// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file three_center_eri_kernel.cuh
/// @brief CUDA kernel for three-center ERI integrals
///
/// GPU implementation of (ab|P) integrals for density fitting.

#include <libaccint/core/types.hpp>

#include <cuda_runtime.h>

namespace libaccint::device::cuda {

/// @brief Launch three-center ERI kernel on GPU
///
/// @param orb_data Orbital basis SoA data on device
/// @param aux_data Auxiliary basis SoA data on device
/// @param n_orb Number of orbital functions
/// @param n_aux Number of auxiliary functions
/// @param[out] tensor Output three-center tensor (device memory)
/// @param stream CUDA stream
void launch_three_center_eri_kernel(
    // Orbital basis data
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
    // Auxiliary basis data
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
    // Output
    Real* tensor,
    cudaStream_t stream);

/// @brief GPU kernel for three-center ERI integrals
///
/// Each thread block handles one shell triple (a, b, P).
__global__ void three_center_eri_kernel(
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
    Real* tensor);

/// @brief Launch B tensor computation on GPU
///
/// Computes B_ab^P = sum_Q (ab|Q) * L^{-1}_{QP} using cuBLAS
///
/// @param three_center Three-center integrals (ab|Q) on device
/// @param L_inv Inverse Cholesky factor L^{-1} on device
/// @param[out] B_tensor Output B tensor on device
/// @param n_orb Number of orbital functions
/// @param n_aux Number of auxiliary functions
/// @param stream CUDA stream
void launch_b_tensor_kernel(
    const Real* three_center,
    const Real* L_inv,
    Real* B_tensor,
    Size n_orb,
    Size n_aux,
    cudaStream_t stream);

/// @brief Launch DF-J computation on GPU
///
/// Computes J_ab = sum_P B_ab^P * gamma_P using cuBLAS
///
/// @param B_tensor B tensor on device (n_orb^2 x n_aux)
/// @param D Density matrix on device (n_orb x n_orb)
/// @param[out] J Coulomb matrix on device (n_orb x n_orb)
/// @param n_orb Number of orbital functions
/// @param n_aux Number of auxiliary functions
/// @param stream CUDA stream
void launch_df_j_kernel(
    const Real* B_tensor,
    const Real* D,
    Real* J,
    Size n_orb,
    Size n_aux,
    cudaStream_t stream);

/// @brief Launch DF-K computation on GPU
///
/// Computes K_ac = sum_P sum_bd B_ab^P * D_bd * B_cd^P
///
/// @param B_tensor B tensor on device (n_orb^2 x n_aux)
/// @param D Density matrix on device (n_orb x n_orb)
/// @param[out] K Exchange matrix on device (n_orb x n_orb)
/// @param n_orb Number of orbital functions
/// @param n_aux Number of auxiliary functions
/// @param stream CUDA stream
void launch_df_k_kernel(
    const Real* B_tensor,
    const Real* D,
    Real* K,
    Size n_orb,
    Size n_aux,
    cudaStream_t stream);

}  // namespace libaccint::device::cuda
