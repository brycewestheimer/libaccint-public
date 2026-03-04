// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file eri_erf_coulomb_kernel_cuda.hpp
/// @brief CUDA erf-attenuated Coulomb ERI kernel interface
///
/// GPU-accelerated computation of (ab|erf(omega*r12)/r12|cd) integrals
/// using modified Rys quadrature with range-separated Boys function.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/basis/device_data.hpp>
#include <libaccint/memory/device_memory.hpp>

#include <cuda_runtime.h>

namespace libaccint::kernels::cuda {

/**
 * @brief Launch CUDA kernel for erf-attenuated Coulomb ERIs
 *
 * Computes (ab|erf(omega*r12)/r12|cd) for all shell quartets using
 * modified Rys quadrature with range-separated Boys function.
 *
 * @param quartet Device data for the four ShellSets
 * @param omega Range-separation parameter
 * @param d_boys_coeffs Pointer to device Boys function Chebyshev coefficients
 * @param d_output Device buffer for output integrals
 * @param stream CUDA stream for kernel launch
 */
void launch_eri_erf_coulomb_kernel(
    const basis::ShellSetQuartetDeviceData& quartet,
    double omega,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream = nullptr);

/**
 * @brief Get output buffer size for erf-Coulomb ERIs
 *
 * @param quartet Device data for the four ShellSets
 * @return Number of doubles needed in the output buffer
 */
[[nodiscard]] size_t eri_erf_coulomb_output_size(
    const basis::ShellSetQuartetDeviceData& quartet);

/**
 * @brief Dispatch to appropriate specialized kernel based on AM
 *
 * @param quartet Device data for the four ShellSets
 * @param omega Range-separation parameter
 * @param d_boys_coeffs Pointer to device Boys function coefficients
 * @param d_output Device buffer for output integrals
 * @param stream CUDA stream for kernel launch
 */
void dispatch_eri_erf_coulomb_kernel(
    const basis::ShellSetQuartetDeviceData& quartet,
    double omega,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream = nullptr);

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
