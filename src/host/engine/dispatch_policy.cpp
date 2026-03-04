// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file dispatch_policy.cpp
/// @brief Implementation of dispatch policy heuristics

#include <libaccint/engine/dispatch_policy.hpp>
#include <libaccint/kernels/dispatch_registry.hpp>

namespace libaccint {

BackendType DispatchPolicy::select_backend(
    WorkUnitType work_type,
    Size batch_size,
    int total_am,
    Size n_primitives,
    BackendHint hint,
    bool gpu_available) const {

    // Handle force hints first
    if (hint == BackendHint::ForceCPU) {
        return BackendType::CPU;
    }

    if (hint == BackendHint::ForceGPU) {
        // ForceGPU requires GPU to be available
        // If not available, this is a configuration error - caller should check
        // We return CPU as a fallback to avoid crashes
        return gpu_available ? BackendType::CUDA : BackendType::CPU;
    }

    // If GPU is not available, always use CPU
    if (!gpu_available) {
        return BackendType::CPU;
    }

    // Apply heuristics based on work type and characteristics
    bool prefer_gpu = false;

    switch (work_type) {
        case WorkUnitType::SingleShellPair:
        case WorkUnitType::SingleShellQuartet:
            // Single shell operations have too little work to justify GPU kernel launch
            // Exception: very high angular momentum might still benefit
            prefer_gpu = (total_am >= config_.high_am_threshold + 2);
            break;

        case WorkUnitType::ShellSetPair:
            prefer_gpu = (batch_size >= config_.min_gpu_batch_size) ||
                         (n_primitives >= config_.min_gpu_primitives) ||
                         (total_am >= config_.high_am_threshold);
            break;

        case WorkUnitType::ShellSetQuartet:
            prefer_gpu = (batch_size >= config_.min_gpu_batch_size) ||
                         (n_primitives >= config_.min_gpu_primitives) ||
                         (total_am >= config_.high_am_threshold - 1);
            break;

        case WorkUnitType::FullBasis:
            // Full basis operations almost always benefit from GPU
            // Only skip GPU for very small basis sets
            prefer_gpu = (batch_size >= config_.min_gpu_shells);
            break;
    }

    // Apply user preference hints
    if (hint == BackendHint::PreferCPU) {
        // Only use GPU if it's strongly preferred by heuristics
        // Raise the bar for GPU selection
        if (work_type == WorkUnitType::SingleShellPair ||
            work_type == WorkUnitType::SingleShellQuartet) {
            prefer_gpu = false;  // Never use GPU for single operations with PreferCPU
        } else {
            // Require higher thresholds
            prefer_gpu = prefer_gpu &&
                         (batch_size >= config_.min_gpu_batch_size * 2 ||
                          total_am >= config_.high_am_threshold + 2);
        }
    } else if (hint == BackendHint::PreferGPU) {
        // Lower the bar for GPU selection
        if (work_type != WorkUnitType::SingleShellPair &&
            work_type != WorkUnitType::SingleShellQuartet) {
            // For batched operations with PreferGPU, use GPU more readily
            prefer_gpu = prefer_gpu ||
                         (batch_size >= config_.min_gpu_batch_size / 4) ||
                         (total_am >= config_.high_am_threshold - 1);
        }
    }

    return prefer_gpu ? BackendType::CUDA : BackendType::CPU;
}

DispatchDecision DispatchPolicy::select_strategy(
    const kernels::RegistryKey& key,
    Size batch_size,
    BackendHint hint,
    bool gpu_available) const {

    DispatchDecision decision;

    // Handle force hints first - these bypass auto-tuning
    if (hint == BackendHint::ForceCPU) {
        decision.backend = BackendType::CPU;
        decision.strategy = kernels::ExecutionStrategy::ThreadedSimdCPU;
        return decision;
    }

    if (hint == BackendHint::ForceGPU) {
        if (gpu_available) {
            decision.backend = BackendType::CUDA;
            decision.strategy = kernels::ExecutionStrategy::WarpPerQuartetGPU;
        } else {
            // Fallback to CPU if GPU not available
            decision.backend = BackendType::CPU;
            decision.strategy = kernels::ExecutionStrategy::ThreadedSimdCPU;
        }
        return decision;
    }

    // If GPU is not available, use CPU strategies
    if (!gpu_available) {
        decision.backend = BackendType::CPU;
        decision.strategy = kernels::ExecutionStrategy::ThreadedSimdCPU;
        return decision;
    }

    // Check if auto-tuning is enabled and batch is large enough
    if (config_.enable_auto_tuning && batch_size >= config_.auto_tune_min_batch) {
        // Use dispatch registry for optimal strategy selection
        auto& registry = kernels::get_dispatch_registry();
        registry.set_gpu_available(gpu_available);

        auto entry = registry.lookup(key, batch_size);
        decision.strategy = entry.strategy;
        decision.backend = strategy_to_backend(entry.strategy);

        // Apply user preference adjustments
        if (hint == BackendHint::PreferCPU && kernels::uses_gpu(decision.strategy)) {
            // User prefers CPU but auto-tuner selected GPU
            // Check if the difference is significant before overriding
            decision.strategy = kernels::ExecutionStrategy::ThreadedSimdCPU;
            decision.backend = BackendType::CPU;
        } else if (hint == BackendHint::PreferGPU && !kernels::uses_gpu(decision.strategy)) {
            // User prefers GPU but auto-tuner selected CPU
            decision.strategy = kernels::ExecutionStrategy::WarpPerQuartetGPU;
            decision.backend = BackendType::CUDA;
        }

        return decision;
    }

    // Fall back to heuristic-based selection
    int total_am = key.total_am();
    Size n_primitives = key.total_primitives();

    // Determine work unit type from key
    WorkUnitType work_type = key.is_two_electron()
        ? WorkUnitType::ShellSetQuartet
        : WorkUnitType::ShellSetPair;

    BackendType backend = select_backend(work_type, batch_size, total_am,
                                         n_primitives, hint, gpu_available);

    decision.backend = backend;
    if (backend == BackendType::CPU) {
        decision.strategy = kernels::ExecutionStrategy::ThreadedSimdCPU;
    } else {
        decision.strategy = kernels::ExecutionStrategy::WarpPerQuartetGPU;
    }

    return decision;
}

BackendType DispatchPolicy::strategy_to_backend(
    kernels::ExecutionStrategy strategy) noexcept {
    if (kernels::uses_gpu(strategy)) {
        return BackendType::CUDA;
    }
    return BackendType::CPU;
}

}  // namespace libaccint
