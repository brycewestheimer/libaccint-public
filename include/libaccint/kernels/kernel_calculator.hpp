// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file kernel_calculator.hpp
/// @brief Auto-tuning kernel strategy selector

#include <libaccint/core/types.hpp>
#include <libaccint/kernels/cost_model.hpp>
#include <libaccint/kernels/execution_strategy.hpp>
#include <libaccint/kernels/registry_key.hpp>

#include <chrono>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace libaccint::kernels {

/// @brief Timing record for a specific strategy execution
struct TimingRecord {
    ExecutionStrategy strategy;          ///< Strategy that was executed
    std::chrono::nanoseconds elapsed;    ///< Measured execution time
    Size batch_size;                     ///< Batch size for this measurement
    std::chrono::steady_clock::time_point timestamp;  ///< When measurement was taken
};

/// @brief Auto-tuning kernel strategy selector
///
/// Selects optimal execution strategies based on cost model estimates
/// and optional runtime profiling. Supports multiple tuning modes:
/// - Analytical: Pure cost model (fast, no profiling)
/// - ProfileOnce: Profile first call, cache result
/// - AdaptiveTune: Periodically re-profile for changing conditions
class KernelCalculator {
public:
    /// @brief Tuning mode for strategy selection
    enum class Mode {
        Analytical,    ///< Use cost model only (fast, approximate)
        ProfileOnce,   ///< Profile first call, cache result
        AdaptiveTune   ///< Periodically re-profile (dynamic tuning)
    };

    /// @brief Construct a KernelCalculator with default settings
    /// @param mode Tuning mode to use (default: ProfileOnce)
    explicit KernelCalculator(Mode mode = Mode::ProfileOnce);

    /// @brief Construct with specific cost model
    /// @param cost_model Cost model for analytical estimates
    /// @param mode Tuning mode to use
    KernelCalculator(CostModel cost_model, Mode mode);

    /// @brief Select the optimal execution strategy
    ///
    /// Returns the best strategy based on the current mode:
    /// - Analytical: Uses cost model estimate
    /// - ProfileOnce: Uses cached timing if available, else cost model
    /// - AdaptiveTune: May re-profile periodically
    ///
    /// @param key Registry key identifying the computation
    /// @param batch_size Number of integrals to compute
    /// @return Recommended execution strategy
    [[nodiscard]] ExecutionStrategy select(const RegistryKey& key, Size batch_size) const;

    /// @brief Record a timing measurement for a computation
    ///
    /// Used in ProfileOnce and AdaptiveTune modes to update the
    /// cached strategy selection based on actual performance.
    ///
    /// @param key Registry key for the computation
    /// @param strategy Strategy that was used
    /// @param elapsed Measured execution time
    void record_timing(const RegistryKey& key, ExecutionStrategy strategy,
                       std::chrono::nanoseconds elapsed);

    /// @brief Record timing with batch size information
    ///
    /// @param key Registry key for the computation
    /// @param strategy Strategy that was used
    /// @param elapsed Measured execution time
    /// @param batch_size Batch size that was processed
    void record_timing(const RegistryKey& key, ExecutionStrategy strategy,
                       std::chrono::nanoseconds elapsed, Size batch_size);

    /// @brief Get the underlying cost model
    [[nodiscard]] const CostModel& cost_model() const noexcept { return cost_model_; }

    /// @brief Get a mutable reference to the cost model
    [[nodiscard]] CostModel& cost_model() noexcept { return cost_model_; }

    /// @brief Get the current tuning mode
    [[nodiscard]] Mode mode() const noexcept { return mode_; }

    /// @brief Set the tuning mode
    void set_mode(Mode mode) noexcept { mode_ = mode; }

    /// @brief Clear all cached timing records
    void clear_history();

    /// @brief Get timing history for a specific key
    [[nodiscard]] std::vector<TimingRecord> get_history(const RegistryKey& key) const;

    /// @brief Check if GPU is available for selection
    [[nodiscard]] bool gpu_available() const noexcept { return gpu_available_; }

    /// @brief Set GPU availability
    void set_gpu_available(bool available) noexcept { gpu_available_ = available; }

private:
    CostModel cost_model_;
    Mode mode_;
    bool gpu_available_{false};

    /// @brief Timing history per registry key
    mutable std::unordered_map<RegistryKey, std::vector<TimingRecord>, RegistryKey::Hash> history_;

    /// @brief Cached best strategy per key (for ProfileOnce mode)
    mutable std::unordered_map<RegistryKey, ExecutionStrategy, RegistryKey::Hash> cached_strategies_;

    /// @brief Mutex for thread-safe access
    mutable std::mutex mutex_;

    /// @brief Select strategy using analytical cost model
    [[nodiscard]] ExecutionStrategy select_analytical(const RegistryKey& key, Size batch_size) const;

    /// @brief Select strategy using cached timing or fall back to analytical
    [[nodiscard]] ExecutionStrategy select_with_cache(const RegistryKey& key, Size batch_size) const;

    /// @brief Analyze timing history to find best strategy
    [[nodiscard]] ExecutionStrategy analyze_history(const std::vector<TimingRecord>& records) const;
};

}  // namespace libaccint::kernels
