// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file fused_1e_kernel_cuda.hpp
/// @brief CUDA fused S+T+V one-electron integral kernel interface
///
/// Provides a single GPU kernel that computes overlap (S), kinetic (T), and
/// nuclear attraction (V) integrals simultaneously for a ShellSetPair. This
/// eliminates 66% of kernel launch overhead compared to running three separate
/// kernels and reuses shared intermediates (Gaussian product, recursion tables).

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/basis/device_data.hpp>
#include <libaccint/operators/device_operator_data.hpp>
#include <libaccint/memory/device_memory.hpp>

#include <cuda_runtime.h>

namespace libaccint::kernels::cuda {

/// @brief Output pointers for the fused 1e kernel
struct Fused1eOutputPointers {
    double* d_overlap;   ///< Device pointer to overlap integrals
    double* d_kinetic;   ///< Device pointer to kinetic integrals
    double* d_nuclear;   ///< Device pointer to nuclear attraction integrals
};

/// @brief Compute fused S+T+V integrals on GPU for a ShellSetPair
///
/// Single kernel launch that computes all three one-electron integral types
/// simultaneously, sharing Gaussian product computation and recursion tables.
///
/// @param pair Device data for bra and ket ShellSets
/// @param charges Device data for point charges (nuclear positions and charges)
/// @param d_boys_coeffs Pointer to device Boys function Chebyshev coefficients
/// @param output Output pointers for the three integral types
/// @param stream CUDA stream for kernel launch
void launch_fused_1e_kernel(
    const basis::ShellSetPairDeviceData& pair,
    const operators::DevicePointChargeData& charges,
    const double* d_boys_coeffs,
    const Fused1eOutputPointers& output,
    cudaStream_t stream = nullptr);

/// @brief Get the output buffer size required per operator for fused 1e integrals
///
/// @param pair Device data for bra and ket ShellSets
/// @return Number of doubles needed per operator in the output buffer
[[nodiscard]] size_t fused_1e_output_size(const basis::ShellSetPairDeviceData& pair);

/// @brief Dispatch to the appropriate specialized fused kernel based on AM
///
/// @param pair Device data for bra and ket ShellSets
/// @param charges Device data for point charges
/// @param d_boys_coeffs Pointer to device Boys function Chebyshev coefficients
/// @param output Output pointers for the three integral types
/// @param stream CUDA stream for kernel launch
/// @throws InvalidArgumentException if AM combination is not supported
void dispatch_fused_1e_kernel(
    const basis::ShellSetPairDeviceData& pair,
    const operators::DevicePointChargeData& charges,
    const double* d_boys_coeffs,
    const Fused1eOutputPointers& output,
    cudaStream_t stream = nullptr);

/// @brief Template-specialized fused 1e kernel launcher
///
/// @tparam La Angular momentum of bra shells
/// @tparam Lb Angular momentum of ket shells
template <int La, int Lb>
void launch_fused_1e_kernel_specialized(
    const basis::ShellSetPairDeviceData& pair,
    const operators::DevicePointChargeData& charges,
    const double* d_boys_coeffs,
    const Fused1eOutputPointers& output,
    cudaStream_t stream = nullptr);

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
