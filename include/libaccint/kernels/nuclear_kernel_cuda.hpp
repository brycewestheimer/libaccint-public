// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file nuclear_kernel_cuda.hpp
/// @brief CUDA nuclear attraction integral kernel interface
///
/// Provides GPU-accelerated nuclear attraction integral computation using
/// Rys quadrature. The kernel processes ShellSetPairs and point charges
/// in parallel.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/basis/device_data.hpp>
#include <libaccint/operators/device_operator_data.hpp>
#include <libaccint/memory/device_memory.hpp>

#include <cuda_runtime.h>

namespace libaccint::kernels::cuda {

// ============================================================================
// CUDA Nuclear Kernel Interface
// ============================================================================

/**
 * @brief Compute nuclear attraction integrals on GPU for a ShellSetPair
 *
 * Launches a CUDA kernel that computes nuclear attraction integrals
 * V_uv = <u| sum_C -Z_C/|r-R_C| |v> for all shell pairs using Rys quadrature.
 *
 * @note GPU results may exhibit ~0.1 absolute error vs CPU for small molecules
 *       due to device-side Chebyshev Boys function accumulation order. This error
 *       is bounded and does not grow with system size for the current batch kernel.
 *
 * @param pair Device data for bra and ket ShellSets
 * @param charges Device data for point charges (nuclear positions and charges)
 * @param d_boys_coeffs Pointer to device Boys function Chebyshev coefficients
 * @param d_output Device buffer for output integrals
 * @param stream CUDA stream for kernel launch
 */
void launch_nuclear_kernel(
    const basis::ShellSetPairDeviceData& pair,
    const operators::DevicePointChargeData& charges,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream = nullptr);

/**
 * @brief Get the output buffer size required for nuclear attraction integrals
 *
 * @param pair Device data for bra and ket ShellSets
 * @return Number of doubles needed in the output buffer
 */
[[nodiscard]] size_t nuclear_output_size(const basis::ShellSetPairDeviceData& pair);

/**
 * @brief Template-specialized nuclear kernel launcher
 *
 * @tparam La Angular momentum of bra shells
 * @tparam Lb Angular momentum of ket shells
 * @param pair Device data for bra and ket ShellSets
 * @param charges Device data for point charges
 * @param d_boys_coeffs Pointer to device Boys function Chebyshev coefficients
 * @param d_output Device buffer for output integrals
 * @param stream CUDA stream for kernel launch
 */
template <int La, int Lb>
void launch_nuclear_kernel_specialized(
    const basis::ShellSetPairDeviceData& pair,
    const operators::DevicePointChargeData& charges,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream = nullptr);

/**
 * @brief Dispatch to the appropriate specialized kernel based on AM
 *
 * @param pair Device data for bra and ket ShellSets
 * @param charges Device data for point charges
 * @param d_boys_coeffs Pointer to device Boys function Chebyshev coefficients
 * @param d_output Device buffer for output integrals
 * @param stream CUDA stream for kernel launch
 * @throws InvalidArgumentException if AM combination is not supported
 */
void dispatch_nuclear_kernel(
    const basis::ShellSetPairDeviceData& pair,
    const operators::DevicePointChargeData& charges,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream = nullptr);

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
