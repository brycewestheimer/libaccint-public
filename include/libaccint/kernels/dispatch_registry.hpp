// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file dispatch_registry.hpp
/// @brief Cache for optimal kernel dispatch decisions

#include <libaccint/core/backend.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/kernels/execution_strategy.hpp>
#include <libaccint/kernels/kernel_calculator.hpp>
#include <libaccint/kernels/registry_key.hpp>

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <unordered_map>

namespace libaccint::kernels {

/// @brief Cache for optimal kernel dispatch decisions
///
/// Provides O(1) lookup for pre-computed dispatch decisions. Uses a
/// KernelCalculator for strategy selection when cache misses occur.
/// Thread-safe using reader-writer locks for concurrent access.
class DispatchRegistry {
public:
    /// @brief Entry in the dispatch cache
    struct Entry {
        ExecutionStrategy strategy;  ///< Selected execution strategy
        bool was_cached;             ///< True if this was a cache hit
        double estimated_ns;         ///< Estimated execution time in nanoseconds
    };

    /// @brief Statistics for monitoring cache performance
    ///
    /// Fields are atomic to allow safe concurrent reads via shared_lock
    /// while stats are updated (e.g. hit/miss counting in const lookup).
    struct Stats {
        std::atomic<std::size_t> entries{0};  ///< Number of entries in cache
        std::atomic<std::size_t> hits{0};     ///< Number of cache hits
        std::atomic<std::size_t> misses{0};   ///< Number of cache misses

        Stats() = default;
        Stats(const Stats& other)
            : entries(other.entries.load(std::memory_order_relaxed)),
              hits(other.hits.load(std::memory_order_relaxed)),
              misses(other.misses.load(std::memory_order_relaxed)) {}
        Stats& operator=(const Stats& other) {
            entries.store(other.entries.load(std::memory_order_relaxed), std::memory_order_relaxed);
            hits.store(other.hits.load(std::memory_order_relaxed), std::memory_order_relaxed);
            misses.store(other.misses.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return *this;
        }
    };

    /// @brief Construct a DispatchRegistry with default KernelCalculator
    DispatchRegistry();

    /// @brief Construct with a specific KernelCalculator
    /// @param calculator The kernel calculator to use for strategy selection
    explicit DispatchRegistry(std::shared_ptr<KernelCalculator> calculator);

    /// @brief Look up the optimal dispatch decision for a computation
    ///
    /// Returns a cached entry if available, otherwise uses the
    /// KernelCalculator to select a strategy and caches the result.
    ///
    /// @param key Registry key identifying the computation
    /// @param batch_size Number of integrals to compute
    /// @return Dispatch entry with selected strategy
    [[nodiscard]] Entry lookup(const RegistryKey& key, Size batch_size);

    /// @brief Look up without modifying cache (const version)
    ///
    /// @param key Registry key identifying the computation
    /// @param batch_size Number of integrals to compute
    /// @return Dispatch entry (was_cached=false if not in cache)
    [[nodiscard]] Entry lookup(const RegistryKey& key, Size batch_size) const;

    /// @brief Pre-populate the registry with common configurations
    ///
    /// Warms up the cache by computing optimal strategies for all
    /// angular momentum combinations up to max_am.
    ///
    /// @param max_am Maximum angular momentum to warm up
    /// @param backend Backend type to consider
    void warmup(int max_am, BackendType backend);

    /// @brief Clear all cached entries
    void clear() noexcept;

    /// @brief Get cache statistics
    [[nodiscard]] Stats stats() const noexcept;

    /// @brief Get the underlying kernel calculator
    [[nodiscard]] std::shared_ptr<KernelCalculator> calculator() const noexcept {
        return calculator_;
    }

    /// @brief Set GPU availability
    void set_gpu_available(bool available);

    /// @brief Record a timing measurement (passes through to calculator)
    void record_timing(const RegistryKey& key, ExecutionStrategy strategy,
                       std::chrono::nanoseconds elapsed, Size batch_size);

    /// @brief Get the number of cached entries
    [[nodiscard]] std::size_t size() const noexcept;

private:
    std::shared_ptr<KernelCalculator> calculator_;

    /// @brief Cache of dispatch decisions
    mutable std::unordered_map<RegistryKey, Entry, RegistryKey::Hash> cache_;

    /// @brief Statistics
    mutable Stats stats_;

    /// @brief Reader-writer mutex for thread-safe access
    mutable std::shared_mutex mutex_;

    /// @brief Compute a new entry using the calculator
    [[nodiscard]] Entry compute_entry(const RegistryKey& key, Size batch_size) const;
};

/// @brief Get the global dispatch registry singleton
///
/// Provides a shared registry for all Engine instances. Thread-safe.
///
/// @return Reference to the global dispatch registry
[[nodiscard]] DispatchRegistry& get_dispatch_registry();

/// @brief Reset the global dispatch registry (for testing)
void reset_dispatch_registry();

}  // namespace libaccint::kernels
