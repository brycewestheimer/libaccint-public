// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file cost_model.hpp
/// @brief Analytical cost model for kernel execution strategy selection

#include <libaccint/core/types.hpp>
#include <libaccint/kernels/execution_strategy.hpp>
#include <libaccint/kernels/optimal_dispatch_table.hpp>
#include <libaccint/operators/operator_types.hpp>

namespace libaccint::kernels {

/// @brief GPU cost breakdown for a specific execution strategy
///
/// Contains estimated execution times in nanoseconds for a single
/// GpuExecutionStrategy (ThreadPerQuartet, WarpPerQuartet, BlockPerQuartet).
struct GpuStrategyCost {
    GpuExecutionStrategy strategy{GpuExecutionStrategy::ThreadPerQuartet};
    double launch_ns{0.0};    ///< Kernel launch overhead
    double compute_ns{0.0};   ///< GPU compute time (excluding launch)

    /// @brief Total GPU time for this strategy (launch + compute)
    [[nodiscard]] double total_ns() const noexcept {
        return launch_ns + compute_ns;
    }
};

/// @brief Cost estimation result for different execution strategies
///
/// Contains estimated execution times in nanoseconds for various
/// execution paths. Used by KernelCalculator to select optimal strategy.
struct CostEstimate {
    double cpu_serial_ns{0.0};    ///< Serial CPU execution time
    double cpu_simd_ns{0.0};      ///< SIMD-vectorized CPU execution time
    double cpu_threaded_ns{0.0};  ///< Multi-threaded CPU execution time (best of threaded options)
    double gpu_launch_ns{0.0};    ///< GPU kernel launch overhead
    double gpu_compute_ns{0.0};   ///< GPU compute time (excluding launch)

    /// @brief Get total GPU execution time (launch + compute)
    [[nodiscard]] double gpu_total_ns() const noexcept {
        return gpu_launch_ns + gpu_compute_ns;
    }

    /// @brief Get best CPU execution time
    [[nodiscard]] double best_cpu_ns() const noexcept {
        double best = cpu_serial_ns;
        if (cpu_simd_ns > 0.0 && cpu_simd_ns < best) best = cpu_simd_ns;
        if (cpu_threaded_ns > 0.0 && cpu_threaded_ns < best) best = cpu_threaded_ns;
        return best;
    }

    /// @brief Get the fastest estimated time across all strategies
    [[nodiscard]] double fastest_ns() const noexcept {
        double best = best_cpu_ns();
        double gpu = gpu_total_ns();
        if (gpu > 0.0 && gpu < best) return gpu;
        return best;
    }
};

/// @brief Hardware profile for cost estimation
///
/// Contains hardware characteristics used to calibrate the cost model.
/// Can be auto-detected or manually specified.
struct HardwareProfile {
    int cpu_cores{1};                    ///< Number of CPU cores available
    int simd_width{4};                   ///< SIMD register width in doubles (4 for AVX, 8 for AVX-512)
    double cpu_gflops{10.0};             ///< CPU peak GFLOPS (single core, scalar)
    double gpu_gflops{1000.0};           ///< GPU peak GFLOPS
    double gpu_launch_overhead_ns{5000.0}; ///< GPU kernel launch overhead in nanoseconds
    double memory_bandwidth_gb_s{50.0};  ///< Memory bandwidth in GB/s

    /// @brief Check if this profile has GPU capabilities
    [[nodiscard]] bool has_gpu() const noexcept {
        return gpu_gflops > 0.0;
    }
};

/// @brief Analytical cost model for integral computation
///
/// Estimates execution costs based on workload characteristics and
/// hardware profile. Used by KernelCalculator to select optimal
/// execution strategies without runtime profiling.
class CostModel {
public:
    /// @brief Construct a cost model with auto-detected hardware profile
    CostModel();

    /// @brief Construct a cost model with a specific hardware profile
    /// @param profile Hardware characteristics to use for estimation
    explicit CostModel(HardwareProfile profile);

    /// @brief Estimate execution costs for a given computation
    ///
    /// @param op Operator type (affects FLOP count per integral)
    /// @param am Angular momentum quartet
    /// @param n_primitives Primitive counts per center
    /// @param batch_size Number of integrals in the batch
    /// @return Cost estimates for different execution strategies
    [[nodiscard]] CostEstimate estimate(
        OperatorKind op,
        const AMQuartet& am,
        const std::array<int, 4>& n_primitives,
        Size batch_size) const;

    /// @brief Select the best execution strategy based on cost estimates
    ///
    /// @param cost Cost estimates from estimate()
    /// @param gpu_available Whether GPU backend is available
    /// @return Recommended execution strategy
    [[nodiscard]] ExecutionStrategy select_strategy(
        const CostEstimate& cost,
        bool gpu_available) const;

    /// @brief Detect hardware profile from the current system
    /// @return Detected hardware characteristics
    [[nodiscard]] static HardwareProfile detect_hardware();

    /// @brief Estimate GPU cost for a specific execution strategy
    ///
    /// Returns a strategy-specific cost estimate that accounts for the
    /// parallelisation model:
    /// - ThreadPerQuartet: one thread per quartet, highest parallelism
    /// - WarpPerQuartet: one warp per quartet, higher per-quartet throughput
    /// - BlockPerQuartet: one block per quartet, shared-memory latency factor
    ///
    /// @param op             Operator type (affects FLOP count)
    /// @param am             Angular momentum quartet
    /// @param n_primitives   Primitive counts per centre
    /// @param batch_size     Number of quartets in the batch
    /// @param gpu_strategy   GPU execution strategy to estimate
    /// @return Strategy-specific GPU cost estimate
    [[nodiscard]] GpuStrategyCost estimate_gpu_strategy(
        OperatorKind op,
        const AMQuartet& am,
        const std::array<int, 4>& n_primitives,
        Size batch_size,
        GpuExecutionStrategy gpu_strategy) const;

    /// @brief Select the best execution strategy, considering a specific GPU strategy
    ///
    /// Like select_strategy(), but uses strategy-specific GPU cost estimates
    /// instead of the generic WarpPerQuartet fallback.
    ///
    /// @param cost           CPU cost estimates from estimate()
    /// @param gpu_available  Whether GPU backend is available
    /// @param gpu_strategy   GPU execution strategy to evaluate
    /// @return Recommended execution strategy
    [[nodiscard]] ExecutionStrategy select_strategy(
        const CostEstimate& cost,
        bool gpu_available,
        GpuExecutionStrategy gpu_strategy) const;

    /// @brief Refresh GPU hardware parameters from DeviceResourceTracker
    ///
    /// When a CUDA device is available at runtime, queries the
    /// DeviceResourceTracker singleton for SM count, max threads per SM,
    /// and warp size, and updates the internal hardware profile accordingly.
    void refresh_gpu_params_from_device();

    /// @brief Get the current hardware profile
    [[nodiscard]] const HardwareProfile& profile() const noexcept { return profile_; }

    /// @brief Set the hardware profile
    void set_profile(HardwareProfile profile) noexcept { profile_ = profile; }

private:
    HardwareProfile profile_;

    /// @brief Number of SMs (sourced from DeviceResourceTracker or default)
    int gpu_sm_count_{0};

    /// @brief Maximum threads per SM (sourced from DeviceResourceTracker or default)
    int gpu_max_threads_per_sm_{0};

    /// @brief Warp size (default 32 for NVIDIA)
    int gpu_warp_size_{32};

    /// @brief Estimate FLOP count for a single integral
    [[nodiscard]] double estimate_flops(OperatorKind op, const AMQuartet& am,
                                        const std::array<int, 4>& n_prims) const;

    /// @brief Estimate memory bytes accessed for a batch
    [[nodiscard]] double estimate_memory_bytes(const AMQuartet& am, Size batch_size) const;

    /// @brief Compute GPU compute time for a specific strategy
    ///
    /// @param total_flops       Total FLOPs for the batch
    /// @param n_quartets        Number of shell quartets
    /// @param gpu_strategy      Execution strategy
    /// @return Compute time in nanoseconds
    [[nodiscard]] double compute_gpu_strategy_time(
        double total_flops,
        Size n_quartets,
        GpuExecutionStrategy gpu_strategy) const;
};

}  // namespace libaccint::kernels
