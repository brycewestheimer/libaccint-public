// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file buffer_pool.hpp
/// @brief BatchBufferPool for efficient IntegralBuffer memory recycling
///
/// Provides a reusable buffer pool for IntegralBuffer allocations in the
/// batch compute API. Pre-warms allocations based on the BasisSet's
/// ShellSetQuartet worklist to eliminate per-call heap allocations.

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell_set_quartet_utils.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/engine/integral_buffer.hpp>

#include <unordered_map>
#include <vector>

namespace libaccint::memory {

/// @brief Pool usage statistics
struct BatchBufferPoolStats {
    Size total_acquires{0};     ///< Total number of acquire() calls
    Size pool_hits{0};          ///< Number of times a pooled buffer was reused
    Size pool_misses{0};        ///< Number of times a new buffer was allocated
    Size total_bytes_allocated{0}; ///< Total bytes allocated (including reused)
    Size total_dropped{0};      ///< Number of buffers dropped due to per-class cap
};

/// @brief Reusable buffer pool for IntegralBuffer storage
///
/// BatchBufferPool manages recycled IntegralBuffer objects keyed by
/// AM class (La, Lb, Lc, Ld). When compute_batch() is called repeatedly
/// with the same quartet types, the pool recycles previously allocated
/// buffers to avoid heap allocations in hot loops.
///
/// Thread-safety: Each thread should use its own BatchBufferPool instance.
/// Use get_thread_local_batch_pool() for convenient thread-local access.
///
/// Usage:
/// @code
///   memory::BatchBufferPool pool;
///   pool.pre_warm(basis);
///
///   for (const auto& q : basis.shell_set_quartets()) {
///       IntegralBuffer buf = pool.acquire();
///       // ... fill buf via Engine ...
///       pool.release(std::move(buf), q);
///   }
/// @endcode
class BatchBufferPool {
public:
    /// @brief Maximum number of buffers to keep per AM class
    static constexpr Size max_buffers_per_class = 8;

    /// @brief Default constructor
    BatchBufferPool() = default;

    /// @brief Pre-warm the pool for a given basis set
    ///
    /// Analyzes the BasisSet's ShellSetQuartet worklist and pre-allocates
    /// one IntegralBuffer per distinct AM class. This avoids cold-start
    /// allocations on the first compute_batch() call.
    ///
    /// @param basis The BasisSet to pre-warm for
    void pre_warm(const BasisSet& basis) {
        const auto& quartets = basis.shell_set_quartets();
        auto groups = group_by_am_class(quartets);

        for (const auto& group : groups) {
            // Create a pre-sized buffer for each AM class
            IntegralBuffer buf;
            free_lists_[group.am_class].push_back(std::move(buf));
        }
    }

    /// @brief Acquire an IntegralBuffer (may be recycled from pool)
    ///
    /// Returns a recycled buffer if one is available for the given AM class,
    /// or a fresh empty buffer otherwise.
    ///
    /// @return An IntegralBuffer ready for use
    [[nodiscard]] IntegralBuffer acquire(const AMClass& am_class = {}) {
        stats_.total_acquires++;

        auto it = free_lists_.find(am_class);
        if (it != free_lists_.end() && !it->second.empty()) {
            stats_.pool_hits++;
            IntegralBuffer buf = std::move(it->second.back());
            it->second.pop_back();
            buf.clear();
            return buf;
        }

        stats_.pool_misses++;
        return IntegralBuffer{};
    }

    /// @brief Release an IntegralBuffer back to the pool for reuse
    ///
    /// @param buffer The buffer to return to the pool
    /// @param quartet The ShellSetQuartet the buffer was used for (for keying)
    void release(IntegralBuffer&& buffer, const ShellSetQuartet& quartet) {
        AMClass key = get_am_class(quartet);
        auto& list = free_lists_[key];
        if (list.size() < max_buffers_per_class) {
            list.push_back(std::move(buffer));
        } else {
            ++stats_.total_dropped;
        }
    }

    /// @brief Release an IntegralBuffer back to the pool with explicit AM class
    void release(IntegralBuffer&& buffer, const AMClass& key) {
        auto& list = free_lists_[key];
        if (list.size() < max_buffers_per_class) {
            list.push_back(std::move(buffer));
        } else {
            ++stats_.total_dropped;
        }
    }

    /// @brief Get pool usage statistics
    [[nodiscard]] const BatchBufferPoolStats& stats() const noexcept {
        return stats_;
    }

    /// @brief Reset statistics counters
    void reset_stats() noexcept {
        stats_ = {};
    }

    /// @brief Clear all pooled buffers (releases memory)
    void clear() {
        free_lists_.clear();
    }

private:
    std::unordered_map<AMClass, std::vector<IntegralBuffer>, AMClassHash> free_lists_;
    BatchBufferPoolStats stats_;
};

/// @brief Get the thread-local BatchBufferPool
///
/// Each thread has its own pool for lock-free operation. Thread-local
/// pools are initialized on first access and persist for the thread lifetime.
///
/// @return Reference to the calling thread's BatchBufferPool
[[nodiscard]] inline BatchBufferPool& get_thread_local_batch_pool() noexcept {
    thread_local BatchBufferPool pool;
    return pool;
}

}  // namespace libaccint::memory
