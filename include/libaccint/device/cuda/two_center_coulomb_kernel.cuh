// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file two_center_coulomb_kernel.cuh
/// @brief CUDA kernel for two-center Coulomb integrals
///
/// GPU implementation of (P|Q) integrals for density fitting metric.

#include <libaccint/core/types.hpp>

#include <cuda_runtime.h>

namespace libaccint::device::cuda {

/// @brief Launch two-center Coulomb kernel on GPU
///
/// @param aux_data Auxiliary basis SoA data on device
/// @param n_aux Number of auxiliary functions
/// @param n_shells Number of auxiliary shells
/// @param[out] metric Output metric matrix (device memory)
/// @param stream CUDA stream
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
    cudaStream_t stream);

/// @brief GPU kernel for two-center Coulomb integrals
///
/// Each thread block handles one shell pair (P, Q).
__global__ void two_center_coulomb_kernel(
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
    Real* metric);

}  // namespace libaccint::device::cuda
