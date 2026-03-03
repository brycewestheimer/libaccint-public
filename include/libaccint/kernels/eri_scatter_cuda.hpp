// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file eri_scatter_cuda.hpp
/// @brief CUDA kernel for scattering flat ERI output into a 4D tensor on device
///
/// After an ERI kernel computes integrals into a flat buffer ordered by
/// shell quartet (ia, ib, ic, id) with Cartesian components packed per quartet,
/// this kernel scatters those values into the correct positions of a
/// device-resident nbf x nbf x nbf x nbf tensor using function_offset arrays.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/basis/device_data.hpp>
#include <cuda_runtime.h>

namespace libaccint::kernels::cuda {

/// @brief Scatter flat ERI buffer into a 4D tensor on device
///
/// Each thread handles one element from the flat ERI buffer and atomically
/// adds it to the correct position in the output tensor.
///
/// @param d_eri_flat    Flat ERI output from dispatch_eri_kernel [n_quartets * n_cart_abcd]
/// @param quartet       Device data for the four ShellSets (has function offsets)
/// @param d_tensor      Device 4D tensor [nbf * nbf * nbf * nbf], pre-zeroed
/// @param nbf           Number of basis functions
/// @param stream        CUDA stream
void launch_eri_scatter_kernel(
    const double* d_eri_flat,
    const basis::ShellSetQuartetDeviceData& quartet,
    double* d_tensor,
    int nbf,
    cudaStream_t stream = nullptr);

/// @brief Scatter SoA-layout flat ERI buffer into a 4D tensor on device
///
/// SoA layout: d_eri_soa[component * n_quartets + quartet_id]
/// This is the transpose of the AoS layout and provides coalesced GPU reads.
///
/// @param d_eri_soa      SoA ERI output [n_cart_abcd * n_quartets]
/// @param quartet        Device data for the four ShellSets (has function offsets)
/// @param d_tensor       Device 4D tensor [nbf * nbf * nbf * nbf], pre-zeroed
/// @param nbf            Number of basis functions
/// @param stream         CUDA stream
void launch_eri_scatter_kernel_soa(
    const double* d_eri_soa,
    const basis::ShellSetQuartetDeviceData& quartet,
    double* d_tensor,
    int nbf,
    cudaStream_t stream = nullptr);

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
