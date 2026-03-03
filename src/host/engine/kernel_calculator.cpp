// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/kernels/kernel_calculator.hpp>

#include <algorithm>
#include <numeric>

namespace libaccint::kernels {

KernelCalculator::KernelCalculator(Mode mode)
    : cost_model_(), mode_(mode), gpu_available_(false) {}

KernelCalculator::KernelCalculator(CostModel cost_model, Mode mode)
    : cost_model_(std::move(cost_model)), mode_(mode), gpu_available_(false) {}

ExecutionStrategy KernelCalculator::select(const RegistryKey& key, Size batch_size) const {
    switch (mode_) {
        case Mode::Analytical:
            return select_analytical(key, batch_size);

        case Mode::ProfileOnce:
        case Mode::AdaptiveTune:
            return select_with_cache(key, batch_size);
    }

    // Fallback
    return select_analytical(key, batch_size);
}

ExecutionStrategy KernelCalculator::select_analytical(const RegistryKey& key, Size batch_size) const {
    // Use cost model to estimate execution times
    CostEstimate cost = cost_model_.estimate(
        key.op_kind,
        key.am,
        key.n_primitives,
        batch_size);

    return cost_model_.select_strategy(cost, gpu_available_);
}

ExecutionStrategy KernelCalculator::select_with_cache(const RegistryKey& key, Size batch_size) const {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if we have a cached strategy
    auto cached_it = cached_strategies_.find(key);
    if (cached_it != cached_strategies_.end()) {
        return cached_it->second;
    }

    // Check if we have timing history
    auto history_it = history_.find(key);
    if (history_it != history_.end() && !history_it->second.empty()) {
        ExecutionStrategy best = analyze_history(history_it->second);
        // Cache the result for ProfileOnce mode
        if (mode_ == Mode::ProfileOnce) {
            cached_strategies_[key] = best;
        }
        return best;
    }

    // Fall back to analytical selection
    return select_analytical(key, batch_size);
}

ExecutionStrategy KernelCalculator::analyze_history(const std::vector<TimingRecord>& records) const {
    if (records.empty()) {
        return ExecutionStrategy::SerialCPU;
    }

    // Group records by strategy and compute average time
    std::unordered_map<ExecutionStrategy, std::vector<double>> times_by_strategy;

    for (const auto& record : records) {
        // Normalize by batch size for fair comparison
        double time_per_integral = static_cast<double>(record.elapsed.count()) /
                                   static_cast<double>(record.batch_size);
        times_by_strategy[record.strategy].push_back(time_per_integral);
    }

    // Find strategy with minimum average time
    ExecutionStrategy best_strategy = ExecutionStrategy::SerialCPU;
    double best_avg_time = std::numeric_limits<double>::max();

    for (const auto& [strategy, times] : times_by_strategy) {
        if (times.empty()) continue;

        // Compute average (could use median for robustness)
        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

        if (avg < best_avg_time) {
            best_avg_time = avg;
            best_strategy = strategy;
        }
    }

    return best_strategy;
}

void KernelCalculator::record_timing(const RegistryKey& key, ExecutionStrategy strategy,
                                      std::chrono::nanoseconds elapsed) {
    record_timing(key, strategy, elapsed, 1);  // Default batch size of 1
}

void KernelCalculator::record_timing(const RegistryKey& key, ExecutionStrategy strategy,
                                      std::chrono::nanoseconds elapsed, Size batch_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    TimingRecord record{
        strategy,
        elapsed,
        batch_size,
        std::chrono::steady_clock::now()
    };

    history_[key].push_back(record);

    // In AdaptiveTune mode, invalidate cached strategy when new data arrives
    if (mode_ == Mode::AdaptiveTune) {
        cached_strategies_.erase(key);
    }

    // Limit history size to avoid unbounded growth
    constexpr size_t MAX_HISTORY_SIZE = 100;
    auto& records = history_[key];
    if (records.size() > MAX_HISTORY_SIZE) {
        // Keep only the most recent records
        records.erase(records.begin(), records.begin() + (records.size() - MAX_HISTORY_SIZE));
    }
}

void KernelCalculator::clear_history() {
    std::lock_guard<std::mutex> lock(mutex_);
    history_.clear();
    cached_strategies_.clear();
}

std::vector<TimingRecord> KernelCalculator::get_history(const RegistryKey& key) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = history_.find(key);
    if (it != history_.end()) {
        return it->second;
    }
    return {};
}

}  // namespace libaccint::kernels
