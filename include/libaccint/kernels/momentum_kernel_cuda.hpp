// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file momentum_kernel_cuda.hpp
/// @brief CUDA momentum integral kernel interface (stub)
///
/// @warning EXPERIMENTAL -- gated by LIBACCINT_ENABLE_EXPERIMENTAL_GPU_PROPERTIES.
/// This API is subject to change or removal without notice in future releases.
///
/// This interface is part of the experimental GPU property path and is guarded
/// at runtime by `LIBACCINT_ENABLE_EXPERIMENTAL_GPU_PROPERTIES`.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/basis/device_data.hpp>
#include <libaccint/memory/device_memory.hpp>

#include <cuda_runtime.h>

namespace libaccint::kernels::cuda {

/// @brief Compute linear momentum integrals on GPU (stub — not yet implemented)
void launch_linear_momentum_kernel(
    const basis::ShellSetPairDeviceData& pair,
    double* d_output,
    cudaStream_t stream = nullptr);

/// @brief Compute angular momentum integrals on GPU (stub — not yet implemented)
void launch_angular_momentum_kernel(
    const basis::ShellSetPairDeviceData& pair,
    const double* d_origin,
    double* d_output,
    cudaStream_t stream = nullptr);

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
