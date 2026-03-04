// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file execution_strategy.hpp
/// @brief Execution strategy enumeration for kernel dispatch decisions

#include <string_view>

namespace libaccint::kernels {

/// @brief Execution strategies for integral kernels
///
/// Defines the different execution backends and parallelization strategies
/// available for computing molecular integrals. The auto-tuning framework
/// uses these to select the optimal strategy based on workload characteristics.
enum class ExecutionStrategy {
    // CPU strategies
    SerialCPU,           ///< Single-threaded, scalar computation
    SimdCPU,             ///< Single-threaded, SIMD vectorized (AVX/AVX-512)
    ThreadedCPU,         ///< OpenMP multi-threaded, scalar
    ThreadedSimdCPU,     ///< OpenMP multi-threaded with SIMD vectorization

    // GPU strategies
    ThreadPerIntegralGPU,  ///< One GPU thread per integral (high parallelism)
    WarpPerQuartetGPU,     ///< One warp (32 threads) per quartet (collaborative)
    BlockPerBatchGPU,      ///< One thread block per ShellSet batch (coalesced)
};

/// @brief Check if a strategy uses GPU execution
/// @param s The execution strategy
/// @return true if the strategy executes on GPU
[[nodiscard]] constexpr bool uses_gpu(ExecutionStrategy s) noexcept {
    switch (s) {
        case ExecutionStrategy::ThreadPerIntegralGPU:
        case ExecutionStrategy::WarpPerQuartetGPU:
        case ExecutionStrategy::BlockPerBatchGPU:
            return true;
        case ExecutionStrategy::SerialCPU:
        case ExecutionStrategy::SimdCPU:
        case ExecutionStrategy::ThreadedCPU:
        case ExecutionStrategy::ThreadedSimdCPU:
            return false;
    }
    return false;
}

/// @brief Check if a strategy uses SIMD vectorization
/// @param s The execution strategy
/// @return true if the strategy uses SIMD instructions
[[nodiscard]] constexpr bool uses_simd(ExecutionStrategy s) noexcept {
    switch (s) {
        case ExecutionStrategy::SimdCPU:
        case ExecutionStrategy::ThreadedSimdCPU:
            return true;
        case ExecutionStrategy::SerialCPU:
        case ExecutionStrategy::ThreadedCPU:
        case ExecutionStrategy::ThreadPerIntegralGPU:
        case ExecutionStrategy::WarpPerQuartetGPU:
        case ExecutionStrategy::BlockPerBatchGPU:
            return false;
    }
    return false;
}

/// @brief Check if a strategy uses multi-threading
/// @param s The execution strategy
/// @return true if the strategy uses multiple CPU threads
[[nodiscard]] constexpr bool uses_threading(ExecutionStrategy s) noexcept {
    switch (s) {
        case ExecutionStrategy::ThreadedCPU:
        case ExecutionStrategy::ThreadedSimdCPU:
            return true;
        case ExecutionStrategy::SerialCPU:
        case ExecutionStrategy::SimdCPU:
        case ExecutionStrategy::ThreadPerIntegralGPU:
        case ExecutionStrategy::WarpPerQuartetGPU:
        case ExecutionStrategy::BlockPerBatchGPU:
            return false;
    }
    return false;
}

/// @brief Convert ExecutionStrategy to string representation
/// @param s The execution strategy
/// @return String name of the strategy
[[nodiscard]] constexpr std::string_view to_string(ExecutionStrategy s) noexcept {
    switch (s) {
        case ExecutionStrategy::SerialCPU:
            return "SerialCPU";
        case ExecutionStrategy::SimdCPU:
            return "SimdCPU";
        case ExecutionStrategy::ThreadedCPU:
            return "ThreadedCPU";
        case ExecutionStrategy::ThreadedSimdCPU:
            return "ThreadedSimdCPU";
        case ExecutionStrategy::ThreadPerIntegralGPU:
            return "ThreadPerIntegralGPU";
        case ExecutionStrategy::WarpPerQuartetGPU:
            return "WarpPerQuartetGPU";
        case ExecutionStrategy::BlockPerBatchGPU:
            return "BlockPerBatchGPU";
    }
    return "Unknown";
}

}  // namespace libaccint::kernels
