// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file overlap_kernel_cuda.hpp
/// @brief CUDA overlap integral kernel interface
///
/// Provides GPU-accelerated overlap integral computation using Obara-Saika
/// recursion. The kernel processes ShellSetPairs in parallel for efficient
/// batch computation.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/basis/device_data.hpp>
#include <libaccint/memory/device_memory.hpp>

#include <cuda_runtime.h>

namespace libaccint::kernels::cuda {

// ============================================================================
// CUDA Overlap Kernel Interface
// ============================================================================

/**
 * @brief Compute overlap integrals on GPU for a ShellSetPair
 *
 * Launches a CUDA kernel that computes overlap integrals for all shell pairs
 * in the given ShellSetPairDeviceData. Each thread block handles one shell
 * pair, with threads distributed across Cartesian component pairs.
 *
 * @param pair Device data for bra and ket ShellSets
 * @param d_output Device buffer for output integrals [n_bra_funcs * n_ket_funcs * n_pairs]
 * @param stream CUDA stream for kernel launch
 */
void launch_overlap_kernel(
    const basis::ShellSetPairDeviceData& pair,
    double* d_output,
    cudaStream_t stream = nullptr);

/**
 * @brief Get the output buffer size required for overlap integrals
 *
 * @param pair Device data for bra and ket ShellSets
 * @return Number of doubles needed in the output buffer
 */
[[nodiscard]] size_t overlap_output_size(const basis::ShellSetPairDeviceData& pair);

/**
 * @brief Template-specialized overlap kernel launcher
 *
 * Launches the appropriate kernel based on angular momentum values.
 * Template specialization allows compile-time optimization of loop bounds.
 *
 * @tparam La Angular momentum of bra shells
 * @tparam Lb Angular momentum of ket shells
 * @param pair Device data for bra and ket ShellSets
 * @param d_output Device buffer for output integrals
 * @param stream CUDA stream for kernel launch
 */
template <int La, int Lb>
void launch_overlap_kernel_specialized(
    const basis::ShellSetPairDeviceData& pair,
    double* d_output,
    cudaStream_t stream = nullptr);

// ============================================================================
// Dispatch Function
// ============================================================================

/**
 * @brief Dispatch to the appropriate specialized kernel based on AM
 *
 * Routes to the correct template specialization based on the angular momenta
 * stored in the ShellSetPairDeviceData. Supports AM pairs up to (d|d).
 *
 * @param pair Device data for bra and ket ShellSets
 * @param d_output Device buffer for output integrals
 * @param stream CUDA stream for kernel launch
 * @throws InvalidArgumentException if AM combination is not supported
 */
void dispatch_overlap_kernel(
    const basis::ShellSetPairDeviceData& pair,
    double* d_output,
    cudaStream_t stream = nullptr);

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
