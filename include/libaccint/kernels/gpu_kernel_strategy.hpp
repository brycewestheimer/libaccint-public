// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file gpu_kernel_strategy.hpp
/// @brief AM-specialized GPU kernel pattern selection
///
/// Maps angular momentum combinations to one of three GPU kernel execution
/// patterns, as specified in the design overview:
///
/// 1. **Register Buffering** (low AM, total <= 2):
///    All recursion tables and intermediate values fit in GPU registers.
///    Full loop unrolling with no shared memory. Optimal for (ss), (sp), (pp).
///    Maps to ExecutionStrategy::ThreadPerIntegralGPU.
///
/// 2. **Streaming Writes** (medium AM, total 3-4):
///    Recursion tables partially spill to local memory. Output values are
///    streamed to global memory without intermediate buffering.
///    Maps to ExecutionStrategy::WarpPerQuartetGPU.
///
/// 3. **Shared Memory** (high AM, total >= 5):
///    Recursion tables are stored in shared memory, accessed collaboratively
///    by warps in the same thread block. Required for (dd|dd) and higher.
///    Maps to ExecutionStrategy::BlockPerBatchGPU.

#include <libaccint/core/types.hpp>
#include <libaccint/kernels/execution_strategy.hpp>

namespace libaccint::kernels {

/// @brief GPU kernel memory pattern for a given AM class
enum class GpuKernelPattern : int {
    RegisterBuffering = 0,  ///< All data fits in registers (low AM)
    StreamingWrites = 1,    ///< Partial register, stream to global memory
    SharedMemory = 2,       ///< Shared memory for recursion tables (high AM)
};

/// @brief Select the optimal GPU kernel pattern for one-electron integrals
///
/// @param La Angular momentum of bra shell
/// @param Lb Angular momentum of ket shell
/// @return Optimal GPU kernel pattern
[[nodiscard]] constexpr GpuKernelPattern select_gpu_kernel_pattern(
    int La, int Lb) noexcept {
    const int total_am = La + Lb;
    if (total_am <= 2) {
        return GpuKernelPattern::RegisterBuffering;
    } else if (total_am <= 4) {
        return GpuKernelPattern::StreamingWrites;
    } else {
        return GpuKernelPattern::SharedMemory;
    }
}

/// @brief Select the optimal GPU kernel pattern for two-electron integrals
///
/// @param La Angular momentum of bra shell A
/// @param Lb Angular momentum of bra shell B
/// @param Lc Angular momentum of ket shell C
/// @param Ld Angular momentum of ket shell D
/// @return Optimal GPU kernel pattern
[[nodiscard]] constexpr GpuKernelPattern select_gpu_kernel_pattern(
    int La, int Lb, int Lc, int Ld) noexcept {
    const int total_am = La + Lb + Lc + Ld;
    if (total_am <= 2) {
        return GpuKernelPattern::RegisterBuffering;
    } else if (total_am <= 4) {
        return GpuKernelPattern::StreamingWrites;
    } else {
        return GpuKernelPattern::SharedMemory;
    }
}

/// @brief Map a GPU kernel pattern to the corresponding ExecutionStrategy
///
/// @param pattern The GPU kernel pattern
/// @return Corresponding ExecutionStrategy enum value
[[nodiscard]] constexpr ExecutionStrategy pattern_to_strategy(
    GpuKernelPattern pattern) noexcept {
    switch (pattern) {
        case GpuKernelPattern::RegisterBuffering:
            return ExecutionStrategy::ThreadPerIntegralGPU;
        case GpuKernelPattern::StreamingWrites:
            return ExecutionStrategy::WarpPerQuartetGPU;
        case GpuKernelPattern::SharedMemory:
            return ExecutionStrategy::BlockPerBatchGPU;
    }
    return ExecutionStrategy::ThreadPerIntegralGPU;
}

/// @brief Select the ExecutionStrategy for GPU one-electron integrals
///
/// @param La Angular momentum of bra shell
/// @param Lb Angular momentum of ket shell
/// @return Optimal GPU execution strategy
[[nodiscard]] constexpr ExecutionStrategy select_gpu_strategy(
    int La, int Lb) noexcept {
    return pattern_to_strategy(select_gpu_kernel_pattern(La, Lb));
}

/// @brief Select the ExecutionStrategy for GPU two-electron integrals
///
/// @param La Angular momentum of bra shell A
/// @param Lb Angular momentum of bra shell B
/// @param Lc Angular momentum of ket shell C
/// @param Ld Angular momentum of ket shell D
/// @return Optimal GPU execution strategy
[[nodiscard]] constexpr ExecutionStrategy select_gpu_strategy(
    int La, int Lb, int Lc, int Ld) noexcept {
    return pattern_to_strategy(select_gpu_kernel_pattern(La, Lb, Lc, Ld));
}

/// @brief Get recommended GPU thread block size for a given pattern
///
/// @param pattern The GPU kernel pattern
/// @return Recommended thread block size (threads per block)
[[nodiscard]] constexpr int recommended_block_size(
    GpuKernelPattern pattern) noexcept {
    switch (pattern) {
        case GpuKernelPattern::RegisterBuffering:
            return 256;  // Maximum occupancy, minimal register pressure
        case GpuKernelPattern::StreamingWrites:
            return 128;  // Balance occupancy with register usage
        case GpuKernelPattern::SharedMemory:
            return 64;   // Reduced threads to allow more shared memory
    }
    return 128;
}

/// @brief Get recommended shared memory size (bytes) for a given AM quartet
///
/// For the SharedMemory pattern, estimates how much shared memory is needed
/// for the recursion tables at the given angular momentum combination.
///
/// @param La Angular momentum of bra shell A
/// @param Lb Angular momentum of bra shell B
/// @param Lc Angular momentum of ket shell C
/// @param Ld Angular momentum of ket shell D
/// @return Estimated shared memory requirement in bytes
[[nodiscard]] constexpr Size recommended_shared_memory(
    int La, int Lb, int Lc, int Ld) noexcept {
    // For 2D Rys recursion, each direction needs a table of size:
    //   (La+Lb+1) x (Lb+1) x (Lc+Ld+1) x (Ld+1) for ERI
    // Three directions (x, y, z), each double precision
    const int dim_a = La + Lb + 1;
    const int dim_b = Lb + 1;
    const int dim_c = Lc + Ld + 1;
    const int dim_d = Ld + 1;
    const Size table_size = static_cast<Size>(dim_a * dim_b * dim_c * dim_d);
    const Size bytes_per_table = table_size * sizeof(double);
    return 3 * bytes_per_table;  // Three Cartesian directions
}

/// @brief Convert GpuKernelPattern to string
[[nodiscard]] constexpr const char* to_string(GpuKernelPattern p) noexcept {
    switch (p) {
        case GpuKernelPattern::RegisterBuffering: return "RegisterBuffering";
        case GpuKernelPattern::StreamingWrites:   return "StreamingWrites";
        case GpuKernelPattern::SharedMemory:      return "SharedMemory";
    }
    return "Unknown";
}

}  // namespace libaccint::kernels
