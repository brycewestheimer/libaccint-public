// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file diagnostics.hpp
/// @brief Performance diagnostics: timing and counters for LibAccInt
///
/// Provides lightweight instrumentation for measuring kernel execution times,
/// counting integral evaluations, and tracking screening efficiency.
/// All diagnostics are opt-in and zero-overhead when disabled.

#include <libaccint/core/types.hpp>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace libaccint::diagnostics {

/// @brief A single timing measurement
struct TimingRecord {
    std::string name;
    std::chrono::nanoseconds duration{0};
    Size call_count{0};
};

/// @brief RAII timer for automatic scope-based timing
class ScopedTimer {
public:
    /// @brief Start timing with a given name
    explicit ScopedTimer(const std::string& name);

    /// @brief Stop timing and record the result
    ~ScopedTimer();

    // Non-copyable, non-movable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

    /// @brief Get elapsed time so far (without stopping)
    [[nodiscard]] std::chrono::nanoseconds elapsed() const;

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

/// @brief Performance counter types
enum class Counter : int {
    ShellPairsComputed = 0,
    ShellQuartetsComputed,
    ShellQuartetsScreened,
    IntegralsComputed,
    PrimitivePairsComputed,
    KernelInvocations,
    BufferAllocations,
    BufferReuses,
    Count  ///< Number of counter types
};

/// @brief Convert counter enum to string
[[nodiscard]] std::string_view counter_name(Counter c) noexcept;

/// @brief Central diagnostics collector — thread-safe, singleton-accessible
///
/// Usage:
/// @code
///   auto& diag = DiagnosticsCollector::instance();
///   diag.set_enabled(true);
///
///   {
///     ScopedTimer t("overlap_kernel");
///     // ... kernel work ...
///   }
///
///   diag.increment(Counter::IntegralsComputed, 100);
///   auto report = diag.report();
/// @endcode
class DiagnosticsCollector {
public:
    /// @brief Get the global diagnostics instance
    static DiagnosticsCollector& instance() noexcept;

    /// @brief Enable/disable diagnostics collection
    void set_enabled(bool enabled) noexcept;

    /// @brief Check if diagnostics are enabled
    [[nodiscard]] bool is_enabled() const noexcept;

    /// @brief Record a timing measurement
    void record_timing(const std::string& name,
                       std::chrono::nanoseconds duration);

    /// @brief Increment a performance counter
    void increment(Counter c, Size amount = 1);

    /// @brief Get counter value
    [[nodiscard]] Size counter_value(Counter c) const;

    /// @brief Get all timing records
    [[nodiscard]] std::vector<TimingRecord> timing_records() const;

    /// @brief Generate a human-readable report string
    [[nodiscard]] std::string report() const;

    /// @brief Reset all counters and timings
    void reset();

private:
    DiagnosticsCollector();

    mutable std::mutex mutex_;
    std::atomic<bool> enabled_{false};
    std::unordered_map<std::string, TimingRecord> timings_;
    std::array<std::atomic<Size>, static_cast<int>(Counter::Count)> counters_;
};

// ============================================================================
// Convenience Macros
// ============================================================================

// Two-level token concatenation for proper __LINE__ expansion
#define LIBACCINT_CONCAT_IMPL(a, b) a##b
#define LIBACCINT_CONCAT(a, b) LIBACCINT_CONCAT_IMPL(a, b)

/// @brief Time a scope — only active when diagnostics are enabled
#define LIBACCINT_TIMED_SCOPE(name)                                          \
    std::unique_ptr<::libaccint::diagnostics::ScopedTimer>                   \
        LIBACCINT_CONCAT(timed_scope_, __LINE__);                            \
    if (::libaccint::diagnostics::DiagnosticsCollector::instance()            \
            .is_enabled()) {                                                  \
        LIBACCINT_CONCAT(timed_scope_, __LINE__) = std::make_unique<         \
            ::libaccint::diagnostics::ScopedTimer>(name);                    \
    }

/// @brief Increment a performance counter (no-op when disabled)
#define LIBACCINT_COUNT(counter, amount)                                     \
    do {                                                                      \
        if (::libaccint::diagnostics::DiagnosticsCollector::instance()        \
                .is_enabled()) {                                              \
            ::libaccint::diagnostics::DiagnosticsCollector::instance()        \
                .increment(counter, amount);                                 \
        }                                                                     \
    } while (false)

}  // namespace libaccint::diagnostics
