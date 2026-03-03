// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file eri_kernel_warp_cuda.hpp
/// @brief Warp-per-quartet ERI kernel for high angular momentum
///
/// This kernel variant uses one warp (32 threads) per shell quartet,
/// distributing Rys points and Cartesian components across lanes.
/// This reduces register pressure for high-AM integrals at the cost
/// of using more threads per integral.
///
/// Use cases:
/// - (pp|pp) and higher AM quartets where thread-per-quartet shows register spilling
/// - Better occupancy when individual quartets have many Cartesian components

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/basis/device_data.hpp>
#include <cuda_runtime.h>

namespace libaccint::kernels::cuda {

/// @brief Check if warp-per-quartet kernel is preferred for given AM quartet
/// @param la Angular momentum of shell A
/// @param lb Angular momentum of shell B
/// @param lc Angular momentum of shell C
/// @param ld Angular momentum of shell D
/// @return True if warp-per-quartet kernel should be used
inline bool prefer_warp_eri_kernel(int la, int lb, int lc, int ld) {
    // Use warp kernel for high-AM quartets (total AM >= 4)
    const int total_am = la + lb + lc + ld;
    return total_am >= 4;
}

/// @brief Dispatch warp-per-quartet ERI kernel
/// @param quartet Device data for the shell quartet
/// @param d_boys_coeffs Device pointer to Boys function coefficients
/// @param d_output Device output buffer
/// @param stream CUDA stream for execution
void dispatch_eri_warp_kernel(
    const basis::ShellSetQuartetDeviceData& quartet,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream = nullptr);

/// @brief Calculate output buffer size for warp ERI kernel
/// @param quartet Shell quartet specification
/// @return Number of doubles in output buffer
size_t eri_warp_output_size(const basis::ShellSetQuartetDeviceData& quartet);

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
