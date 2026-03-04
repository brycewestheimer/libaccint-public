// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file thread_local_pool.hpp
/// @brief Per-thread memory pool partitioning for parallel integral computation
///
/// Extends the existing MemoryPool with explicit per-thread partitioning that
/// provides deterministic pool assignment independent of thread scheduling.
/// This avoids the thread_local-based approach when explicit control is needed.

#include <libaccint/engine/thread_config.hpp>
#include <libaccint/memory/memory_pool.hpp>

#include <cassert>
#include <memory>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libaccint::memory {

/// @brief Manages a fixed array of MemoryPools, one per thread
///
/// Unlike the thread_local pool from get_thread_local_pool(), this class
/// provides explicit lifecycle control and deterministic pool assignment.
/// It is designed for use in structured parallel regions where the number
/// of threads is known ahead of time.
///
/// Benefits over thread_local:
///   - Pools are destroyed when PartitionedPool is destroyed (predictable cleanup)
///   - Pool count matches the parallel region's thread count exactly
///   - Statistics can be aggregated across all partitions
///   - No dependency on thread_local storage (works with any threading model)
///
/// Usage:
/// @code
///   PartitionedPool pools(n_threads);
///
///   #pragma omp parallel num_threads(n_threads)
///   {
///       auto& pool = pools.thread_pool();
///       auto buf = pool.acquire(1024);
///       // ... use buf ...
///   }
///
///   auto stats = pools.aggregate_stats();
/// @endcode
class PartitionedPool {
public:
    /// @brief Construct with a specific number of partitions
    /// @param n_partitions Number of thread-local pools to create (0 = auto-detect)
    explicit PartitionedPool(int n_partitions = 0) {
        int n = engine::ThreadConfig::resolve(n_partitions);
        pools_.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            pools_.push_back(std::make_unique<MemoryPool>());
        }
    }

    /// @brief Get the pool for the current thread
    ///
    /// Must be called from within a parallel region. The thread ID
    /// is used to index into the pool array.
    ///
    /// @return Reference to the current thread's pool
    [[nodiscard]] MemoryPool& thread_pool() noexcept {
        int tid = 0;
#if defined(_OPENMP)
#if !defined(NDEBUG)
        assert((pools_.size() == 1 || omp_in_parallel()) &&
               "PartitionedPool::thread_pool() called outside parallel region with multi-partition pool");
#endif
        tid = omp_get_thread_num();
#endif
        assert(tid >= 0 && static_cast<std::size_t>(tid) < pools_.size());
        return *pools_[static_cast<std::size_t>(tid)];
    }

    /// @brief Get a specific partition's pool by index
    /// @param partition_id Partition index (0 to n_partitions-1)
    /// @return Reference to the specified pool
    [[nodiscard]] MemoryPool& pool(int partition_id) noexcept {
        assert(partition_id >= 0 &&
               static_cast<std::size_t>(partition_id) < pools_.size());
        return *pools_[static_cast<std::size_t>(partition_id)];
    }

    /// @brief Get the number of partitions
    [[nodiscard]] int n_partitions() const noexcept {
        return static_cast<int>(pools_.size());
    }

    /// @brief Clear all pools, returning all cached memory
    void clear_all() noexcept {
        for (auto& pool : pools_) {
            pool->clear();
        }
    }

    /// @brief Aggregate statistics across all partitions
    [[nodiscard]] MemoryPool::Stats aggregate_stats() const noexcept {
        MemoryPool::Stats total{};
        for (const auto& pool : pools_) {
            auto s = pool->stats();
            total.total_allocations += s.total_allocations;
            total.pool_hits += s.pool_hits;
            total.pool_misses += s.pool_misses;
            total.current_pooled += s.current_pooled;
            total.current_pooled_bytes += s.current_pooled_bytes;
            total.oversized_allocations += s.oversized_allocations;
        }
        return total;
    }

private:
    std::vector<std::unique_ptr<MemoryPool>> pools_;
};

/// @brief RAII guard that creates a PartitionedPool and cleans up on scope exit
///
/// Combines pool creation, usage, and cleanup into a single object.
class ScopedPartitionedPool {
public:
    /// @brief Create a scoped partitioned pool
    /// @param n_partitions Number of partitions (0 = auto-detect)
    explicit ScopedPartitionedPool(int n_partitions = 0)
        : pool_(n_partitions) {}

    /// @brief Get the underlying partitioned pool
    [[nodiscard]] PartitionedPool& pool() noexcept { return pool_; }

    /// @brief Get the pool for the current thread
    [[nodiscard]] MemoryPool& thread_pool() noexcept { return pool_.thread_pool(); }

    ~ScopedPartitionedPool() {
        pool_.clear_all();
    }

    ScopedPartitionedPool(const ScopedPartitionedPool&) = delete;
    ScopedPartitionedPool& operator=(const ScopedPartitionedPool&) = delete;

private:
    PartitionedPool pool_;
};

}  // namespace libaccint::memory

