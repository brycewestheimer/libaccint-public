// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_dispatch_overhead.cpp
/// @brief Dispatch policy decision overhead benchmarks (Task 15.3.3)
///
/// Measures:
/// - Cost of DispatchPolicy::select_backend() at various work unit sizes
/// - Cost of DispatchPolicy::select_strategy() for different AM combinations
/// - Impact of auto-tuning on dispatch overhead

#include <benchmark/benchmark.h>

#include <libaccint/engine/dispatch_policy.hpp>
#include <libaccint/kernels/execution_strategy.hpp>

#include <vector>

namespace libaccint {

// ============================================================================
// select_backend() Cost per Work Unit Type
// ============================================================================

/// @brief Measure select_backend latency for full-basis dispatch decisions
static void BM_Dispatch_SelectBackend(benchmark::State& state) {
    const auto work_type = static_cast<WorkUnitType>(state.range(0));
    const Size batch_size = static_cast<Size>(state.range(1));

    DispatchConfig config;
    DispatchPolicy policy(config);

    const int total_am = 4;
    const Size n_primitives = 500;
    const BackendHint hint = BackendHint::Auto;
    const bool gpu_avail = false;  // CPU-only test

    for (auto _ : state) {
        auto backend = policy.select_backend(
            work_type, batch_size, total_am, n_primitives, hint, gpu_avail);
        benchmark::DoNotOptimize(backend);
    }

    state.SetLabel("work_type=" + std::to_string(static_cast<int>(work_type))
                   + " batch=" + std::to_string(batch_size));
}
BENCHMARK(BM_Dispatch_SelectBackend)
    // WorkUnitType::FullBasis = 4
    ->Args({4, 5})->Args({4, 20})->Args({4, 100})->Args({4, 500})
    // WorkUnitType::SingleShellQuartet = 1
    ->Args({1, 1})->Args({1, 10})->Args({1, 100});

// ============================================================================
// Hint-Dependent Cost
// ============================================================================

/// @brief Measure select_backend cost with different backend hints
static void BM_Dispatch_HintVariation(benchmark::State& state) {
    const auto hint = static_cast<BackendHint>(state.range(0));

    DispatchConfig config;
    DispatchPolicy policy(config);

    const Size batch_size = 50;
    const int total_am = 4;
    const Size n_primitives = 500;

    for (auto _ : state) {
        auto backend = policy.select_backend(
            WorkUnitType::FullBasis, batch_size, total_am, n_primitives,
            hint, false);
        benchmark::DoNotOptimize(backend);
    }

    // Label hints by name
    const char* hint_names[] = {"Auto", "ForceCPU", "ForceGPU", "PreferCPU", "PreferGPU"};
    state.SetLabel(hint_names[static_cast<int>(hint)]);
}
BENCHMARK(BM_Dispatch_HintVariation)
    ->Arg(0)->Arg(1)->Arg(2)->Arg(3)->Arg(4);

// ============================================================================
// Auto-Tuning Toggle Impact
// ============================================================================

/// @brief Compare dispatch overhead with and without auto-tuning enabled
static void BM_Dispatch_AutoTuning(benchmark::State& state) {
    const bool auto_tune = (state.range(0) == 1);

    DispatchConfig config;
    config.enable_auto_tuning = auto_tune;
    DispatchPolicy policy(config);

    const Size batch_size = 100;
    const int total_am = 6;
    const Size n_primitives = 2000;

    for (auto _ : state) {
        auto backend = policy.select_backend(
            WorkUnitType::FullBasis, batch_size, total_am, n_primitives,
            BackendHint::Auto, false);
        benchmark::DoNotOptimize(backend);
    }

    state.SetLabel(auto_tune ? "auto_tuning=ON" : "auto_tuning=OFF");
}
BENCHMARK(BM_Dispatch_AutoTuning)->Arg(0)->Arg(1);

// ============================================================================
// Config Construction Cost
// ============================================================================

/// @brief Measure DispatchPolicy construction overhead
static void BM_Dispatch_Construction(benchmark::State& state) {
    for (auto _ : state) {
        DispatchConfig config;
        DispatchPolicy policy(config);
        benchmark::DoNotOptimize(policy.auto_tuning_enabled());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Dispatch_Construction);

// ============================================================================
// Batch of Dispatch Decisions (amortized cost)
// ============================================================================

/// @brief Measure amortized cost of many dispatch decisions in a batch
static void BM_Dispatch_BatchDecisions(benchmark::State& state) {
    const Size n_decisions = static_cast<Size>(state.range(0));

    DispatchConfig config;
    DispatchPolicy policy(config);

    for (auto _ : state) {
        Size cpu_count = 0;
        for (Size i = 0; i < n_decisions; ++i) {
            auto backend = policy.select_backend(
                WorkUnitType::SingleShellQuartet,
                i + 1,       // varying batch size
                (i % 8) + 1, // varying AM
                (i + 1) * 10,
                BackendHint::Auto,
                false);
            if (backend == BackendType::CPU) {
                ++cpu_count;
            }
        }
        benchmark::DoNotOptimize(cpu_count);
    }

    state.SetItemsProcessed(
        state.iterations() * static_cast<std::int64_t>(n_decisions));
    state.SetLabel(std::to_string(n_decisions) + " decisions");
}
BENCHMARK(BM_Dispatch_BatchDecisions)->Arg(100)->Arg(1000)->Arg(10000);

}  // namespace libaccint

BENCHMARK_MAIN();
