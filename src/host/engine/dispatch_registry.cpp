// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/kernels/dispatch_registry.hpp>
#include <libaccint/operators/operator_types.hpp>

#include <mutex>

namespace libaccint::kernels {

DispatchRegistry::DispatchRegistry()
    : calculator_(std::make_shared<KernelCalculator>()) {}

DispatchRegistry::DispatchRegistry(std::shared_ptr<KernelCalculator> calculator)
    : calculator_(std::move(calculator)) {
    if (!calculator_) {
        calculator_ = std::make_shared<KernelCalculator>();
    }
}

DispatchRegistry::Entry DispatchRegistry::lookup(const RegistryKey& key, Size batch_size) {
    // Try read lock first
    {
        std::shared_lock<std::shared_mutex> read_lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            stats_.hits.fetch_add(1, std::memory_order_relaxed);
            Entry entry = it->second;
            entry.was_cached = true;
            return entry;
        }
    }

    // Cache miss - need to compute and insert
    std::unique_lock<std::shared_mutex> write_lock(mutex_);

    // Double-check after acquiring write lock (another thread might have inserted)
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        stats_.hits.fetch_add(1, std::memory_order_relaxed);
        Entry entry = it->second;
        entry.was_cached = true;
        return entry;
    }

    // Compute new entry
    stats_.misses.fetch_add(1, std::memory_order_relaxed);
    Entry entry = compute_entry(key, batch_size);
    entry.was_cached = false;

    // Cache it
    cache_[key] = entry;
    stats_.entries.fetch_add(1, std::memory_order_relaxed);

    return entry;
}

DispatchRegistry::Entry DispatchRegistry::lookup(const RegistryKey& key, Size batch_size) const {
    std::shared_lock<std::shared_mutex> read_lock(mutex_);

    auto it = cache_.find(key);
    if (it != cache_.end()) {
        stats_.hits.fetch_add(1, std::memory_order_relaxed);
        Entry entry = it->second;
        entry.was_cached = true;
        return entry;
    }

    // Cache miss - compute but don't cache (const method)
    stats_.misses.fetch_add(1, std::memory_order_relaxed);
    Entry entry = compute_entry(key, batch_size);
    entry.was_cached = false;
    return entry;
}

DispatchRegistry::Entry DispatchRegistry::compute_entry(const RegistryKey& key, Size batch_size) const {
    // Use calculator to select strategy
    ExecutionStrategy strategy = calculator_->select(key, batch_size);

    // Estimate execution time
    CostEstimate cost = calculator_->cost_model().estimate(
        key.op_kind, key.am, key.n_primitives, batch_size);

    double estimated_ns = 0.0;
    switch (strategy) {
        case ExecutionStrategy::SerialCPU:
            estimated_ns = cost.cpu_serial_ns;
            break;
        case ExecutionStrategy::SimdCPU:
            estimated_ns = cost.cpu_simd_ns;
            break;
        case ExecutionStrategy::ThreadedCPU:
        case ExecutionStrategy::ThreadedSimdCPU:
            estimated_ns = cost.cpu_threaded_ns;
            break;
        case ExecutionStrategy::ThreadPerIntegralGPU:
        case ExecutionStrategy::WarpPerQuartetGPU:
        case ExecutionStrategy::BlockPerBatchGPU:
            estimated_ns = cost.gpu_total_ns();
            break;
    }

    return Entry{strategy, false, estimated_ns};
}

void DispatchRegistry::warmup(int max_am, BackendType backend) {
    std::unique_lock<std::shared_mutex> write_lock(mutex_);

    // Common operator kinds to warm up
    std::array<OperatorKind, 4> one_e_ops = {
        OperatorKind::Overlap,
        OperatorKind::Kinetic,
        OperatorKind::Nuclear,
        OperatorKind::PointCharge
    };

    std::array<OperatorKind, 3> two_e_ops = {
        OperatorKind::Coulomb,
        OperatorKind::ErfCoulomb,
        OperatorKind::ErfcCoulomb
    };

    // Common primitive counts
    std::array<int, 3> prim_counts = {3, 6, 10};

    // Warm up one-electron operators
    for (OperatorKind op : one_e_ops) {
        for (int la = 0; la <= max_am; ++la) {
            for (int lb = 0; lb <= max_am; ++lb) {
                for (int np : prim_counts) {
                    RegistryKey key = RegistryKey::for_1e(op, la, lb, np, np, backend);
                    if (cache_.find(key) == cache_.end()) {
                        Entry entry = compute_entry(key, 1);
                        cache_[key] = entry;
                        stats_.entries.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    }

    // Warm up two-electron operators
    for (OperatorKind op : two_e_ops) {
        for (int la = 0; la <= max_am; ++la) {
            for (int lb = 0; lb <= max_am; ++lb) {
                for (int lc = 0; lc <= max_am; ++lc) {
                    for (int ld = 0; ld <= max_am; ++ld) {
                        for (int np : prim_counts) {
                            RegistryKey key = RegistryKey::for_2e(
                                op, la, lb, lc, ld, np, np, np, np, backend);
                            if (cache_.find(key) == cache_.end()) {
                                Entry entry = compute_entry(key, 1);
                                cache_[key] = entry;
                                stats_.entries.fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                    }
                }
            }
        }
    }
}

void DispatchRegistry::clear() noexcept {
    std::unique_lock<std::shared_mutex> write_lock(mutex_);
    cache_.clear();
    stats_.entries.store(0, std::memory_order_relaxed);
    stats_.hits.store(0, std::memory_order_relaxed);
    stats_.misses.store(0, std::memory_order_relaxed);
}

DispatchRegistry::Stats DispatchRegistry::stats() const noexcept {
    std::shared_lock<std::shared_mutex> read_lock(mutex_);
    return stats_;
}

void DispatchRegistry::set_gpu_available(bool available) {
    std::unique_lock<std::shared_mutex> write_lock(mutex_);
    calculator_->set_gpu_available(available);
    // Clear cache when GPU availability changes
    cache_.clear();
    stats_.entries.store(0, std::memory_order_relaxed);
}

void DispatchRegistry::record_timing(const RegistryKey& key, ExecutionStrategy strategy,
                                      std::chrono::nanoseconds elapsed, Size batch_size) {
    calculator_->record_timing(key, strategy, elapsed, batch_size);

    // Invalidate cache entry if using adaptive tuning
    if (calculator_->mode() == KernelCalculator::Mode::AdaptiveTune) {
        std::unique_lock<std::shared_mutex> write_lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            cache_.erase(it);
            stats_.entries.fetch_sub(1, std::memory_order_relaxed);
        }
    }
}

std::size_t DispatchRegistry::size() const noexcept {
    std::shared_lock<std::shared_mutex> read_lock(mutex_);
    return cache_.size();
}

// Global singleton implementation
namespace {
    std::mutex g_registry_mutex;
    std::unique_ptr<DispatchRegistry> g_registry;
}

DispatchRegistry& get_dispatch_registry() {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    if (!g_registry) {
        g_registry = std::make_unique<DispatchRegistry>();
    }
    return *g_registry;
}

void reset_dispatch_registry() {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    g_registry.reset();
}

}  // namespace libaccint::kernels
