// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file distributed_multipole_kernel.cu
/// @brief CUDA distributed multipole kernel stub (Phase 17)

#include <libaccint/kernels/distributed_multipole_kernel_cuda.hpp>
#include <libaccint/utils/error_handling.hpp>

#if LIBACCINT_USE_CUDA

namespace libaccint::kernels::cuda {

void launch_distributed_multipole_kernel(
    const basis::ShellSetPairDeviceData& /*pair*/,
    const double* /*d_site_data*/,
    int /*n_sites*/,
    double* /*d_output*/,
    cudaStream_t /*stream*/) {
#if !LIBACCINT_ENABLE_EXPERIMENTAL_GPU_PROPERTIES
    throw NotImplementedException(
        "GPU distributed multipole kernels are disabled by default. "
        "Reconfigure with -DLIBACCINT_ENABLE_EXPERIMENTAL_GPU_PROPERTIES=ON "
        "to enable experimental property paths.");
#else
    throw NotImplementedException(
        "GPU distributed multipole kernels (enabled via LIBACCINT_ENABLE_EXPERIMENTAL_GPU_PROPERTIES) "
        "are not yet implemented. Full GPU implementation is pending.");
#endif
}

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
