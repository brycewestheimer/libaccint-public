// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_memory_pool.cpp
/// @brief Memory pool allocation benchmarks (Task 15.3.2)
///
/// Measures:
/// - Pool vs std::vector allocation throughput at various sizes
/// - Pool reuse (hit rate) after warmup
/// - Multi-threaded pool contention

#include <benchmark/benchmark.h>

#include <libaccint/memory/memory_pool.hpp>

#include <cstring>
#include <thread>
#include <vector>

namespace libaccint {

// ============================================================================
// Pool vs std::vector — single-threaded allocation throughput
// ============================================================================

/// @brief Measure thread-local pool acquire/release throughput
static void BM_Pool_AcquireRelease(benchmark::State& state) {
    const std::size_t alloc_bytes = static_cast<std::size_t>(state.range(0));

    auto& pool = memory::get_thread_local_pool();
    pool.clear();

    for (auto _ : state) {
        auto buf = pool.acquire(alloc_bytes);
        benchmark::DoNotOptimize(buf.data());
    }

    auto stats = pool.stats();
    state.counters["pool_hits"] = static_cast<double>(stats.pool_hits);
    state.counters["pool_misses"] = static_cast<double>(stats.pool_misses);
    state.SetLabel(std::to_string(alloc_bytes) + " bytes");
}
BENCHMARK(BM_Pool_AcquireRelease)
    ->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536)->Arg(262144);

/// @brief Measure std::vector allocation + deallocation for comparison
static void BM_StdVector_AllocDealloc(benchmark::State& state) {
    const std::size_t n_doubles = static_cast<std::size_t>(state.range(0)) / sizeof(double);

    for (auto _ : state) {
        std::vector<double> buf(n_doubles);
        benchmark::DoNotOptimize(buf.data());
    }

    state.SetLabel(std::to_string(state.range(0)) + " bytes (vector)");
}
BENCHMARK(BM_StdVector_AllocDealloc)
    ->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536)->Arg(262144);

// ============================================================================
// Pool Hit Rate — after warmup
// ============================================================================

/// @brief Measure pool hit rate with pre-warmed pool
static void BM_Pool_HitRate(benchmark::State& state) {
    const std::size_t alloc_bytes = static_cast<std::size_t>(state.range(0));

    auto& pool = memory::get_thread_local_pool();
    pool.clear();

    // Warmup: acquire and release once to populate the pool's free list
    {
        auto warmup = pool.acquire(alloc_bytes);
        // PooledBuffer destructor returns memory to pool
    }

    auto stats_before = pool.stats();

    for (auto _ : state) {
        auto buf = pool.acquire(alloc_bytes);
        // Touch the memory to prevent optimization
        std::memset(buf.data(), 0, std::min(alloc_bytes, std::size_t(64)));
        benchmark::DoNotOptimize(buf.data());
        benchmark::ClobberMemory();
    }

    auto stats_after = pool.stats();
    std::size_t new_hits = stats_after.pool_hits - stats_before.pool_hits;
    std::size_t new_misses = stats_after.pool_misses - stats_before.pool_misses;
    std::size_t total_new = new_hits + new_misses;

    if (total_new > 0) {
        state.counters["hit_rate"] =
            static_cast<double>(new_hits) / static_cast<double>(total_new);
    }
}
BENCHMARK(BM_Pool_HitRate)
    ->Arg(256)->Arg(4096)->Arg(65536)->Arg(262144)->Arg(1048576);

// ============================================================================
// Typed Acquire — typed allocation shortcut
// ============================================================================

/// @brief Measure acquire_typed<double> throughput
static void BM_Pool_TypedAcquire(benchmark::State& state) {
    const std::size_t n_doubles = static_cast<std::size_t>(state.range(0));

    auto& pool = memory::get_thread_local_pool();

    for (auto _ : state) {
        auto buf = pool.acquire_typed<double>(n_doubles);
        benchmark::DoNotOptimize(buf.data());
    }

    state.SetLabel(std::to_string(n_doubles) + " doubles");
}
BENCHMARK(BM_Pool_TypedAcquire)->Arg(32)->Arg(128)->Arg(512)->Arg(2048)->Arg(8192);

// ============================================================================
// Pool Clear + Re-Allocate Cost
// ============================================================================

/// @brief Measure cost of clearing pool and re-allocating
static void BM_Pool_ClearAndReallocate(benchmark::State& state) {
    const std::size_t alloc_bytes = static_cast<std::size_t>(state.range(0));

    auto& pool = memory::get_thread_local_pool();

    for (auto _ : state) {
        pool.clear();
        auto buf = pool.acquire(alloc_bytes);
        benchmark::DoNotOptimize(buf.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Pool_ClearAndReallocate)->Arg(1024)->Arg(16384)->Arg(262144);

// ============================================================================
// Global Pool — singleton access pattern
// ============================================================================

/// @brief Measure global pool acquire/release throughput
static void BM_GlobalPool_AcquireRelease(benchmark::State& state) {
    const std::size_t alloc_bytes = static_cast<std::size_t>(state.range(0));

    auto& global = memory::GlobalMemoryPool::instance();
    global.clear();

    for (auto _ : state) {
        auto buf = global.acquire(alloc_bytes);
        benchmark::DoNotOptimize(buf.data());
    }

    auto stats = global.stats();
    state.counters["pool_hits"] = static_cast<double>(stats.pool_hits);
    state.counters["pool_misses"] = static_cast<double>(stats.pool_misses);
}
BENCHMARK(BM_GlobalPool_AcquireRelease)
    ->Arg(256)->Arg(4096)->Arg(65536);

}  // namespace libaccint

BENCHMARK_MAIN();
