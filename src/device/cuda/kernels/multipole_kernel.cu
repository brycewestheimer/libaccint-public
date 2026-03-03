// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file multipole_kernel.cu
/// @brief CUDA multipole integral kernel stubs (Phase 17)
///
/// These stubs compile but do not perform useful computation.
/// Full GPU implementations will be provided in a future phase.

#include <libaccint/kernels/multipole_kernel_cuda.hpp>
#include <libaccint/utils/error_handling.hpp>

#if LIBACCINT_USE_CUDA

#include <cstdio>

namespace libaccint::kernels::cuda {

void launch_dipole_kernel(
    const basis::ShellSetPairDeviceData& /*pair*/,
    const double* /*d_origin*/,
    double* /*d_output*/,
    cudaStream_t /*stream*/) {
#if !LIBACCINT_ENABLE_EXPERIMENTAL_GPU_PROPERTIES
    throw NotImplementedException(
        "GPU dipole kernels are disabled by default. "
        "Reconfigure with -DLIBACCINT_ENABLE_EXPERIMENTAL_GPU_PROPERTIES=ON "
        "to enable experimental property paths.");
#else
    throw NotImplementedException(
        "GPU dipole kernels (enabled via LIBACCINT_ENABLE_EXPERIMENTAL_GPU_PROPERTIES) "
        "are not yet implemented. Full GPU implementation is pending.");
#endif
}

void launch_quadrupole_kernel(
    const basis::ShellSetPairDeviceData& /*pair*/,
    const double* /*d_origin*/,
    double* /*d_output*/,
    cudaStream_t /*stream*/) {
#if !LIBACCINT_ENABLE_EXPERIMENTAL_GPU_PROPERTIES
    throw NotImplementedException(
        "GPU quadrupole kernels are disabled by default. "
        "Reconfigure with -DLIBACCINT_ENABLE_EXPERIMENTAL_GPU_PROPERTIES=ON "
        "to enable experimental property paths.");
#else
    throw NotImplementedException(
        "GPU quadrupole kernels (enabled via LIBACCINT_ENABLE_EXPERIMENTAL_GPU_PROPERTIES) "
        "are not yet implemented. Full GPU implementation is pending.");
#endif
}

void launch_octupole_kernel(
    const basis::ShellSetPairDeviceData& /*pair*/,
    const double* /*d_origin*/,
    double* /*d_output*/,
    cudaStream_t /*stream*/) {
#if !LIBACCINT_ENABLE_EXPERIMENTAL_GPU_PROPERTIES
    throw NotImplementedException(
        "GPU octupole kernels are disabled by default. "
        "Reconfigure with -DLIBACCINT_ENABLE_EXPERIMENTAL_GPU_PROPERTIES=ON "
        "to enable experimental property paths.");
#else
    throw NotImplementedException(
        "GPU octupole kernels (enabled via LIBACCINT_ENABLE_EXPERIMENTAL_GPU_PROPERTIES) "
        "are not yet implemented. Full GPU implementation is pending.");
#endif
}

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
