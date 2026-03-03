// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file dispatch_policy.hpp
/// @brief Dispatch policy for routing work between CPU and GPU backends

#include <libaccint/core/backend.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/kernels/execution_strategy.hpp>
#include <libaccint/kernels/kernel_calculator.hpp>
#include <libaccint/kernels/registry_key.hpp>

#include <memory>

namespace libaccint {

// Forward declaration
namespace kernels {
class DispatchRegistry;
}  // namespace kernels

/// @brief Type of work unit for dispatch decisions
///
/// Different work unit types have different optimal backends based on
/// the amount of parallelism available and kernel launch overhead.
enum class WorkUnitType {
    SingleShellPair,    ///< Single (shell_a, shell_b) pair - low parallelism
    SingleShellQuartet, ///< Single (a, b | c, d) quartet - low parallelism
    ShellSetPair,       ///< Batch of shell pairs - medium parallelism
    ShellSetQuartet,    ///< Batch of shell quartets - high parallelism
    FullBasis,          ///< Full basis computation - maximum parallelism
};

/// @brief Hint for backend selection
///
/// Users can provide hints to influence dispatch decisions. The dispatch
/// policy will respect these hints when possible, but may fall back to
/// a different backend if the requested one is unavailable.
enum class BackendHint {
    Auto,       ///< Let the dispatch policy decide (default)
    ForceCPU,   ///< Always use CPU backend
    ForceGPU,   ///< Always use GPU backend (error if unavailable)
    PreferCPU,  ///< Prefer CPU, but use GPU if beneficial
    PreferGPU,  ///< Prefer GPU, fall back to CPU if unavailable
};

/// @brief Configuration parameters for dispatch heuristics
///
/// These thresholds control when work is routed to the GPU versus CPU.
/// Users can tune these values based on their hardware characteristics.
struct DispatchConfig {
    /// @brief Minimum batch size to consider GPU dispatch
    ///
    /// Below this threshold, kernel launch overhead dominates and CPU is faster.
    Size min_gpu_batch_size = 16;

    /// @brief Minimum total primitives to consider GPU dispatch
    ///
    /// GPU benefits from large numbers of primitives to keep SMs busy.
    Size min_gpu_primitives = 1000;

    /// @brief Angular momentum threshold for GPU preference
    ///
    /// High angular momentum quartets have more work per quartet,
    /// making GPU more beneficial even for smaller batches.
    int high_am_threshold = 4;

    /// @brief Minimum shell count for full-basis GPU dispatch
    ///
    /// For full-basis operations, use GPU if we have enough shells.
    Size min_gpu_shells = 10;

    /// @brief Enable auto-tuning via KernelCalculator
    ///
    /// When enabled, uses the KernelCalculator/DispatchRegistry framework
    /// for strategy selection instead of simple heuristics.
    bool enable_auto_tuning = false;

    /// @brief Auto-tuning mode (only used if enable_auto_tuning = true)
    ///
    /// Controls how the auto-tuner makes decisions:
    /// - Analytical: Pure cost model estimation
    /// - ProfileOnce: Profile first call, cache result
    /// - AdaptiveTune: Periodically re-profile
    kernels::KernelCalculator::Mode auto_tune_mode =
        kernels::KernelCalculator::Mode::ProfileOnce;

    /// @brief Minimum batch size before auto-tuning engages
    ///
    /// For very small batches, skip auto-tuning overhead.
    Size auto_tune_min_batch = 100;

    /// @brief Number of GPU execution slots for concurrent access
    ///
    /// Controls how many host threads can execute GPU work simultaneously.
    /// Each slot owns an independent CUDA stream and set of device buffers.
    /// Users with limited GPU memory (6 GB) should use 1-2 slots; those with
    /// 16 GB+ can use 4 or more.
    Size n_gpu_slots = 4;
};

/// @brief Combined dispatch decision with backend and execution strategy
///
/// Extended result type that includes both the backend (CPU/GPU) and
/// the specific execution strategy within that backend.
struct DispatchDecision {
    BackendType backend;                      ///< Selected backend (CPU, CUDA)
    kernels::ExecutionStrategy strategy;      ///< Execution strategy within the backend
};

/// @brief Policy class for selecting compute backend
///
/// DispatchPolicy encapsulates the heuristics for routing integral
/// computation to the optimal backend based on work characteristics.
///
/// The dispatch decision considers:
/// - Work unit type (single shell pair vs batched operations)
/// - Batch size (GPU benefits from large batches)
/// - Total angular momentum (high AM benefits from GPU parallelism)
/// - Primitive count (more primitives = more GPU-friendly)
/// - User hints (force/prefer CPU or GPU)
/// - GPU availability
class DispatchPolicy {
public:
    /// @brief Construct a dispatch policy with default configuration
    DispatchPolicy() = default;

    /// @brief Construct a dispatch policy with custom configuration
    /// @param config Configuration parameters for dispatch heuristics
    explicit DispatchPolicy(DispatchConfig config) : config_(config) {}

    /// @brief Select the optimal backend for a computation
    ///
    /// @param work_type Type of work unit (single shell pair, batch, etc.)
    /// @param batch_size Number of items in the batch (1 for single operations)
    /// @param total_am Sum of angular momentum quantum numbers
    /// @param n_primitives Total number of primitive Gaussians involved
    /// @param hint User-provided backend preference
    /// @param gpu_available Whether a GPU backend is available
    /// @return The selected backend type
    [[nodiscard]] BackendType select_backend(
        WorkUnitType work_type,
        Size batch_size,
        int total_am,
        Size n_primitives,
        BackendHint hint,
        bool gpu_available) const;

    /// @brief Select both backend and execution strategy using auto-tuning
    ///
    /// Extended API that returns both the backend type and the specific
    /// execution strategy. Uses the DispatchRegistry for cached decisions
    /// when auto-tuning is enabled.
    ///
    /// @param key Registry key identifying the computation
    /// @param batch_size Number of items in the batch
    /// @param hint User-provided backend preference
    /// @param gpu_available Whether a GPU backend is available
    /// @return Combined dispatch decision with backend and strategy
    [[nodiscard]] DispatchDecision select_strategy(
        const kernels::RegistryKey& key,
        Size batch_size,
        BackendHint hint,
        bool gpu_available) const;

    /// @brief Get the current configuration
    [[nodiscard]] const DispatchConfig& config() const noexcept { return config_; }

    /// @brief Set the configuration
    void set_config(DispatchConfig config) noexcept { config_ = config; }

    /// @brief Enable or disable auto-tuning
    void set_auto_tuning(bool enable) noexcept { config_.enable_auto_tuning = enable; }

    /// @brief Check if auto-tuning is enabled
    [[nodiscard]] bool auto_tuning_enabled() const noexcept {
        return config_.enable_auto_tuning;
    }

private:
    DispatchConfig config_;

    /// @brief Map execution strategy to backend type
    [[nodiscard]] static BackendType strategy_to_backend(
        kernels::ExecutionStrategy strategy) noexcept;
};

}  // namespace libaccint
