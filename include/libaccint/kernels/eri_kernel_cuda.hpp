// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file eri_kernel_cuda.hpp
/// @brief CUDA electron repulsion integral (ERI) kernel interface
///
/// Provides GPU-accelerated four-center electron repulsion integral computation
/// using Rys quadrature with 2D recursion. The kernel uses a thread-per-quartet
/// parallelization strategy where each CUDA thread computes one shell quartet.
///
/// Algorithm:
///   1. Each thread processes a unique shell quartet (a,b,c,d)
///   2. Loops over all primitive quartets (contraction)
///   3. For each primitive quartet:
///      a. Compute bra product P = (α·A + β·B)/(α+β)
///      b. Compute ket product Q = (γ·C + δ·D)/(γ+δ)
///      c. Compute T = ρ·|P-Q|² and get Rys roots/weights
///      d. Build 2D recursion tables Ix, Iy, Iz for each Rys root
///      e. Sum Ix·Iy·Iz weighted contributions
///   4. Apply normalization corrections

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/basis/device_data.hpp>
#include <libaccint/memory/device_memory.hpp>

#include <cuda_runtime.h>

namespace libaccint::kernels::cuda {

// ============================================================================
// CUDA ERI Kernel Interface
// ============================================================================

/**
 * @brief Compute electron repulsion integrals on GPU for a ShellSetQuartet
 *
 * Launches a CUDA kernel that computes ERIs (ab|cd) for all shell quartets
 * using Rys quadrature with 2D recursion. Uses one thread per shell quartet.
 *
 * @param quartet Device data for the four ShellSets
 * @param d_boys_coeffs Pointer to device Boys function Chebyshev coefficients
 * @param d_output Device buffer for output integrals
 * @param stream CUDA stream for kernel launch
 */
void launch_eri_kernel(
    const basis::ShellSetQuartetDeviceData& quartet,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream = nullptr);

/**
 * @brief Get the output buffer size required for ERI integrals
 *
 * @param quartet Device data for the four ShellSets
 * @return Number of doubles needed in the output buffer
 */
[[nodiscard]] size_t eri_output_size(const basis::ShellSetQuartetDeviceData& quartet);

/**
 * @brief Template-specialized ERI kernel launcher
 *
 * @tparam La Angular momentum of shell A
 * @tparam Lb Angular momentum of shell B
 * @tparam Lc Angular momentum of shell C
 * @tparam Ld Angular momentum of shell D
 * @param quartet Device data for the four ShellSets
 * @param d_boys_coeffs Pointer to device Boys function Chebyshev coefficients
 * @param d_output Device buffer for output integrals
 * @param stream CUDA stream for kernel launch
 */
template <int La, int Lb, int Lc, int Ld>
void launch_eri_kernel_specialized(
    const basis::ShellSetQuartetDeviceData& quartet,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream = nullptr);

/**
 * @brief Dispatch to the appropriate specialized kernel based on AM
 *
 * @param quartet Device data for the four ShellSets
 * @param d_boys_coeffs Pointer to device Boys function Chebyshev coefficients
 * @param d_output Device buffer for output integrals
 * @param stream CUDA stream for kernel launch
 * @throws InvalidArgumentException if AM combination is not supported
 */
void dispatch_eri_kernel(
    const basis::ShellSetQuartetDeviceData& quartet,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream = nullptr);

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
