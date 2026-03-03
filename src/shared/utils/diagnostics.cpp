// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file diagnostics.cpp
/// @brief Implementation of performance diagnostics infrastructure

#include <libaccint/utils/diagnostics.hpp>

#include <iomanip>
#include <sstream>

namespace libaccint::diagnostics {

// ============================================================================
// Counter name conversion
// ============================================================================

std::string_view counter_name(Counter c) noexcept {
    switch (c) {
        case Counter::ShellPairsComputed:     return "shell_pairs_computed";
        case Counter::ShellQuartetsComputed:   return "shell_quartets_computed";
        case Counter::ShellQuartetsScreened:   return "shell_quartets_screened";
        case Counter::IntegralsComputed:       return "integrals_computed";
        case Counter::PrimitivePairsComputed:  return "primitive_pairs_computed";
        case Counter::KernelInvocations:       return "kernel_invocations";
        case Counter::BufferAllocations:       return "buffer_allocations";
        case Counter::BufferReuses:            return "buffer_reuses";
        case Counter::Count:                   return "count";
    }
    return "unknown";
}

// ============================================================================
// ScopedTimer
// ============================================================================

ScopedTimer::ScopedTimer(const std::string& name)
    : name_(name), start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    auto duration = std::chrono::steady_clock::now() - start_;
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
    DiagnosticsCollector::instance().record_timing(name_, ns);
}

std::chrono::nanoseconds ScopedTimer::elapsed() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_);
}

// ============================================================================
// DiagnosticsCollector
// ============================================================================

DiagnosticsCollector::DiagnosticsCollector() {
    for (auto& c : counters_) {
        c.store(0, std::memory_order_relaxed);
    }
}

DiagnosticsCollector& DiagnosticsCollector::instance() noexcept {
    static DiagnosticsCollector instance;
    return instance;
}

void DiagnosticsCollector::set_enabled(bool enabled) noexcept {
    enabled_.store(enabled, std::memory_order_release);
}

bool DiagnosticsCollector::is_enabled() const noexcept {
    return enabled_.load(std::memory_order_acquire);
}

void DiagnosticsCollector::record_timing(const std::string& name,
                                         std::chrono::nanoseconds duration) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& record = timings_[name];
    record.name = name;
    record.duration += duration;
    record.call_count++;
}

void DiagnosticsCollector::increment(Counter c, Size amount) {
    if (!enabled_) return;
    auto idx = static_cast<int>(c);
    if (idx >= 0 && idx < static_cast<int>(Counter::Count)) {
        counters_[static_cast<Size>(idx)].fetch_add(amount, std::memory_order_relaxed);
    }
}

Size DiagnosticsCollector::counter_value(Counter c) const {
    auto idx = static_cast<int>(c);
    if (idx >= 0 && idx < static_cast<int>(Counter::Count)) {
        return counters_[static_cast<Size>(idx)].load(std::memory_order_relaxed);
    }
    return 0;
}

std::vector<TimingRecord> DiagnosticsCollector::timing_records() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<TimingRecord> result;
    result.reserve(timings_.size());
    for (const auto& [key, record] : timings_) {
        result.push_back(record);
    }
    return result;
}

std::string DiagnosticsCollector::report() const {
    std::ostringstream oss;
    oss << "=== LibAccInt Performance Diagnostics ===\n\n";

    // Counters
    oss << "Counters:\n";
    for (int i = 0; i < static_cast<int>(Counter::Count); ++i) {
        auto c = static_cast<Counter>(i);
        Size val = counters_[static_cast<Size>(i)].load(std::memory_order_relaxed);
        if (val > 0) {
            oss << "  " << counter_name(c) << ": " << val << "\n";
        }
    }

    // Timings
    std::lock_guard<std::mutex> lock(mutex_);
    if (!timings_.empty()) {
        oss << "\nTimings:\n";
        for (const auto& [name, record] : timings_) {
            double ms = static_cast<double>(record.duration.count()) / 1e6;
            double avg_ms = record.call_count > 0
                                ? ms / static_cast<double>(record.call_count)
                                : 0.0;
            oss << "  " << name << ": "
                << std::fixed << std::setprecision(3) << ms << " ms"
                << " (" << record.call_count << " calls"
                << ", avg=" << std::setprecision(3) << avg_ms << " ms)\n";
        }
    }

    return oss.str();
}

void DiagnosticsCollector::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    timings_.clear();
    for (auto& c : counters_) {
        c.store(0, std::memory_order_relaxed);
    }
    enabled_.store(false, std::memory_order_release);
}

}  // namespace libaccint::diagnostics
