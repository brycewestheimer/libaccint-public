// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file thread_config.hpp
/// @brief Thread count configuration, auto-detection, and false sharing prevention
///
/// Provides utilities for configuring OpenMP thread parallelism:
/// - ThreadConfig: global thread count management with auto-detection
/// - CacheLineAligned: wrapper to prevent false sharing between thread-local data
/// - Hardware topology querying

#include <libaccint/core/types.hpp>

#include <algorithm>
#include <cstddef>
#include <thread>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libaccint::engine {

// ============================================================================
// Cache Line Constants (False Sharing Prevention)
// ============================================================================

/// @brief Typical CPU cache line size in bytes
///
/// Used for padding thread-local data to prevent false sharing.
/// 64 bytes covers x86_64, ARM Cortex, and most modern architectures.
/// Apple M-series uses 128, but 64 is still safe (just wastes half a line).
inline constexpr std::size_t CACHE_LINE_SIZE = 64;

/// @brief Wrapper that aligns data to cache line boundaries
///
/// Prevents false sharing when multiple threads access adjacent
/// CacheLineAligned objects in an array.
///
/// @code
///   std::vector<CacheLineAligned<double>> per_thread_accumulators(n_threads);
///   // Each accumulator lives on its own cache line
/// @endcode
template<typename T>
struct alignas(CACHE_LINE_SIZE) CacheLineAligned {
    T value{};

    CacheLineAligned() = default;
    explicit CacheLineAligned(const T& v) : value(v) {}
    explicit CacheLineAligned(T&& v) : value(std::move(v)) {}

    operator T&() noexcept { return value; }
    operator const T&() const noexcept { return value; }

    T& get() noexcept { return value; }
    const T& get() const noexcept { return value; }
};

// ============================================================================
// ThreadConfig
// ============================================================================

/// @brief Thread count configuration and auto-detection for CPU parallelism
///
/// ThreadConfig manages the number of OpenMP threads used for parallel
/// integral computation. It supports:
/// - Auto-detection of available hardware threads
/// - Manual override of thread count
/// - Scoped thread count adjustment via ScopedThreadCount
///
/// Usage:
/// @code
///   // Auto-detect (default)
///   int n = ThreadConfig::recommended_threads();
///
///   // Override globally
///   ThreadConfig::set_num_threads(4);
///
///   // Scoped override
///   {
///       ScopedThreadCount guard(8);
///       // Uses 8 threads in this scope
///   }
///   // Reverts to previous setting
/// @endcode
class ThreadConfig {
public:
    /// @brief Get the number of hardware threads available
    /// @return Number of hardware threads (logical cores)
    [[nodiscard]] static int hardware_threads() noexcept {
        int hw = static_cast<int>(std::thread::hardware_concurrency());
        return (hw > 0) ? hw : 1;
    }

    /// @brief Get the recommended number of threads for computation
    ///
    /// Returns the user-configured thread count if set, or auto-detects
    /// from the environment. Detection order:
    ///   1. User-set value via set_num_threads()
    ///   2. OMP_NUM_THREADS environment variable
    ///   3. Hardware thread count
    ///
    /// @return Recommended thread count (always >= 1)
    [[nodiscard]] static int recommended_threads() noexcept {
        if (user_thread_count_ > 0) {
            return user_thread_count_;
        }

#if defined(_OPENMP)
        return omp_get_max_threads();
#else
        return 1;
#endif
    }

    /// @brief Set the number of threads to use
    /// @param n Number of threads (0 = auto-detect, >0 = explicit)
    static void set_num_threads(int n) noexcept {
        user_thread_count_ = (n > 0) ? n : 0;

#if defined(_OPENMP)
        if (n > 0) {
            omp_set_num_threads(n);
        }
#endif
    }

    /// @brief Get the currently configured thread count
    /// @return Current thread count (0 = auto-detect mode)
    [[nodiscard]] static int num_threads() noexcept {
        return user_thread_count_;
    }

    /// @brief Reset to auto-detection mode
    static void reset() noexcept {
        user_thread_count_ = 0;
    }

    /// @brief Check if OpenMP is available at runtime
    [[nodiscard]] static bool openmp_available() noexcept {
#if defined(_OPENMP)
        return true;
#else
        return false;
#endif
    }

    /// @brief Get the effective number of threads that will be used in a parallel region
    ///
    /// Unlike recommended_threads(), this queries the OpenMP runtime directly
    /// to account for nested parallelism and dynamic adjustment.
    ///
    /// @return Effective thread count for the next parallel region
    [[nodiscard]] static int effective_threads() noexcept {
#if defined(_OPENMP)
        return omp_get_max_threads();
#else
        return 1;
#endif
    }

    /// @brief Resolve thread count parameter: 0 means auto-detect
    /// @param n_threads Input thread count (0 = auto)
    /// @return Resolved thread count (always >= 1)
    [[nodiscard]] static int resolve(int n_threads) noexcept {
        if (n_threads > 0) {
            return n_threads;
        }
        return recommended_threads();
    }

private:
    static inline int user_thread_count_{0};
};

// ============================================================================
// ScopedThreadCount
// ============================================================================

/// @brief RAII guard for temporarily changing the thread count
///
/// Saves the current thread count on construction and restores it
/// on destruction. Useful for benchmarking or testing with specific
/// thread counts without permanently modifying global settings.
class ScopedThreadCount {
public:
    /// @brief Set thread count for the scope duration
    /// @param n_threads Thread count to use (>0)
    explicit ScopedThreadCount(int n_threads)
        : saved_count_(ThreadConfig::num_threads()) {
        ThreadConfig::set_num_threads(n_threads);
    }

    ~ScopedThreadCount() {
        ThreadConfig::set_num_threads(saved_count_);
    }

    ScopedThreadCount(const ScopedThreadCount&) = delete;
    ScopedThreadCount& operator=(const ScopedThreadCount&) = delete;

private:
    int saved_count_;
};

}  // namespace libaccint::engine

// Bring into libaccint namespace for convenience
namespace libaccint {
    using engine::ThreadConfig;
    using engine::ScopedThreadCount;
    using engine::CacheLineAligned;
    using engine::CACHE_LINE_SIZE;
}

