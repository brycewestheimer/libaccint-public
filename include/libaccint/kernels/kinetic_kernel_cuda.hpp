// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file kinetic_kernel_cuda.hpp
/// @brief CUDA kinetic energy integral kernel interface
///
/// Provides GPU-accelerated kinetic energy integral computation using the
/// relationship between second derivatives of Gaussians and overlap integrals
/// with shifted angular momentum indices.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/basis/device_data.hpp>
#include <libaccint/memory/device_memory.hpp>

#include <cuda_runtime.h>

namespace libaccint::kernels::cuda {

// ============================================================================
// CUDA Kinetic Kernel Interface
// ============================================================================

/**
 * @brief Compute kinetic energy integrals on GPU for a ShellSetPair
 *
 * Launches a CUDA kernel that computes kinetic energy integrals T_uv = -1/2 <u|nabla^2|v>
 * for all shell pairs in the given ShellSetPairDeviceData.
 *
 * @param pair Device data for bra and ket ShellSets
 * @param d_output Device buffer for output integrals [n_bra_funcs * n_ket_funcs * n_pairs]
 * @param stream CUDA stream for kernel launch
 */
void launch_kinetic_kernel(
    const basis::ShellSetPairDeviceData& pair,
    double* d_output,
    cudaStream_t stream = nullptr);

/**
 * @brief Get the output buffer size required for kinetic energy integrals
 *
 * @param pair Device data for bra and ket ShellSets
 * @return Number of doubles needed in the output buffer
 */
[[nodiscard]] size_t kinetic_output_size(const basis::ShellSetPairDeviceData& pair);

/**
 * @brief Template-specialized kinetic kernel launcher
 *
 * @tparam La Angular momentum of bra shells
 * @tparam Lb Angular momentum of ket shells
 * @param pair Device data for bra and ket ShellSets
 * @param d_output Device buffer for output integrals
 * @param stream CUDA stream for kernel launch
 */
template <int La, int Lb>
void launch_kinetic_kernel_specialized(
    const basis::ShellSetPairDeviceData& pair,
    double* d_output,
    cudaStream_t stream = nullptr);

/**
 * @brief Dispatch to the appropriate specialized kernel based on AM
 *
 * @param pair Device data for bra and ket ShellSets
 * @param d_output Device buffer for output integrals
 * @param stream CUDA stream for kernel launch
 * @throws InvalidArgumentException if AM combination is not supported
 */
void dispatch_kinetic_kernel(
    const basis::ShellSetPairDeviceData& pair,
    double* d_output,
    cudaStream_t stream = nullptr);

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
