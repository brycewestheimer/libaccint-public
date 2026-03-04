// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_boys.cpp
/// @brief Boys function benchmark: scalar vs SIMD evaluation

#include <benchmark/benchmark.h>
#include <libaccint/math/boys_function.hpp>
#include <libaccint/utils/simd.hpp>

#include <vector>
#include <random>
#include <cmath>

namespace libaccint {

// ============================================================================
// Test Data Generation
// ============================================================================

std::vector<double> generate_random_T_values(std::size_t n, double T_min, double T_max,
                                              unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(T_min, T_max);
    std::vector<double> values(n);
    for (auto& v : values) {
        v = dist(gen);
    }
    return values;
}

// ============================================================================
// Scalar Boys Function Benchmarks
// ============================================================================

static void BM_Boys_Scalar_Single(benchmark::State& state) {
    const int n_max = static_cast<int>(state.range(0));
    const double T = 5.0;  // Mid-range Chebyshev regime
    std::vector<double> result(static_cast<std::size_t>(n_max) + 1);

    for (auto _ : state) {
        math::boys_evaluate_array(n_max, T, result.data());
        benchmark::DoNotOptimize(result.data());
    }

    state.SetItemsProcessed(state.iterations() * (n_max + 1));
}
BENCHMARK(BM_Boys_Scalar_Single)->Arg(4)->Arg(8)->Arg(12)->Arg(16)->Arg(20);

static void BM_Boys_Scalar_Batch(benchmark::State& state) {
    const std::size_t n_values = static_cast<std::size_t>(state.range(0));
    const int n_max = 8;
    auto T_values = generate_random_T_values(n_values, 0.1, 30.0);
    std::vector<double> result(n_values * (n_max + 1));

    for (auto _ : state) {
        math::boys_evaluate_batch(n_max, T_values.data(),
                                   static_cast<int>(n_values), result.data());
        benchmark::DoNotOptimize(result.data());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n_values) * (n_max + 1));
}
BENCHMARK(BM_Boys_Scalar_Batch)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);

// ============================================================================
// SIMD Boys Function Benchmarks
// ============================================================================

static void BM_Boys_SIMD_Batch(benchmark::State& state) {
    const std::size_t n_values = static_cast<std::size_t>(state.range(0));
    const int n_max = 8;
    auto T_values = generate_random_T_values(n_values, 0.1, 30.0);
    std::vector<double> result(n_values * (n_max + 1));

    for (auto _ : state) {
        math::boys_evaluate_batch_simd(n_max, T_values.data(),
                                        static_cast<int>(n_values), result.data());
        benchmark::DoNotOptimize(result.data());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n_values) * (n_max + 1));
}
BENCHMARK(BM_Boys_SIMD_Batch)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);

// ============================================================================
// Same-Interval SIMD (Best Case)
// ============================================================================

static void BM_Boys_SIMD_SameInterval(benchmark::State& state) {
    const std::size_t n_values = static_cast<std::size_t>(state.range(0));
    const int n_max = 8;

    // Generate T values all in same interval [1.0, 2.0) for best SIMD performance
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(1.0, 1.99);
    std::vector<double> T_values(n_values);
    for (auto& v : T_values) {
        v = dist(gen);
    }

    std::vector<double> result(n_values * (n_max + 1));

    for (auto _ : state) {
        math::boys_evaluate_batch_simd(n_max, T_values.data(),
                                        static_cast<int>(n_values), result.data());
        benchmark::DoNotOptimize(result.data());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n_values) * (n_max + 1));
}
BENCHMARK(BM_Boys_SIMD_SameInterval)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);

// ============================================================================
// Asymptotic Regime Benchmark
// ============================================================================

static void BM_Boys_Asymptotic(benchmark::State& state) {
    const std::size_t n_values = static_cast<std::size_t>(state.range(0));
    const int n_max = 8;

    // Generate T values in asymptotic regime [40, 100)
    auto T_values = generate_random_T_values(n_values, 40.0, 100.0);
    std::vector<double> result(n_values * (n_max + 1));

    for (auto _ : state) {
        math::boys_evaluate_batch(n_max, T_values.data(),
                                   static_cast<int>(n_values), result.data());
        benchmark::DoNotOptimize(result.data());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n_values) * (n_max + 1));
}
BENCHMARK(BM_Boys_Asymptotic)->Arg(64)->Arg(256)->Arg(1024);

// ============================================================================
// Chebyshev Single Evaluation
// ============================================================================

static void BM_Boys_Chebyshev_Single(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const double T = 5.0;

    for (auto _ : state) {
        double result = math::boys_chebyshev(n, T);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Boys_Chebyshev_Single)->Arg(0)->Arg(4)->Arg(8)->Arg(12);

}  // namespace libaccint

BENCHMARK_MAIN();
