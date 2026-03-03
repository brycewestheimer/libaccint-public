// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_cpu_optimization.cpp
/// @brief CPU optimization profiling benchmarks (Tasks 27.2.1–27.2.4)
///
/// Covers:
/// - SIMD vectorization audit (27.2.1)
/// - Cache utilization profiling (27.2.2)
/// - Contraction loop optimization (27.2.3)
/// - Memory allocation hotspot elimination (27.2.4)

#include <benchmark/benchmark.h>

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/config.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/memory/memory_pool.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/utils/diagnostics.hpp>
#include <libaccint/utils/simd.hpp>

#include "bench_helpers.hpp"

#include <random>
#include <vector>

namespace libaccint {

using bench::create_random_density;
using bench::make_h2o_basis;

// ============================================================================
// 27.2.1: SIMD Vectorization Audit — measure SIMD vs scalar throughput
// ============================================================================

static void BM_AutoVec_VectorAdd(benchmark::State& state) {
    const Size n = static_cast<Size>(state.range(0));

    // Aligned buffers for SIMD
    std::vector<double> a(n, 1.0);
    std::vector<double> b(n, 2.0);
    std::vector<double> c(n, 0.0);

    for (auto _ : state) {
        for (Size i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
        benchmark::DoNotOptimize(c.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(n * sizeof(double) * 3));
    state.SetLabel("SIMD ISA: " + std::string(simd::simd_isa_name)
                   + ", width=" + std::to_string(simd::simd_width));
}
BENCHMARK(BM_AutoVec_VectorAdd)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

/// @brief Auto-vectorized dot product benchmark
static void BM_AutoVec_DotProduct(benchmark::State& state) {
    const Size n = static_cast<Size>(state.range(0));
    std::vector<double> a(n, 1.5);
    std::vector<double> b(n, 2.5);
    std::vector<double> c(n, 0.0);
    double acc = 0.0;

    for (auto _ : state) {
        acc = 0.0;
        for (Size i = 0; i < n; ++i) {
            acc += a[i] * b[i];
        }
        benchmark::DoNotOptimize(acc);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_AutoVec_DotProduct)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// ============================================================================
// 27.2.2: Cache Utilization — measure cache sensitivity
// ============================================================================

/// @brief Sequential vs strided access patterns
static void BM_Cache_Sequential(benchmark::State& state) {
    const Size n = static_cast<Size>(state.range(0));
    std::vector<double> data(n, 1.0);
    double sum = 0.0;

    for (auto _ : state) {
        sum = 0.0;
        for (Size i = 0; i < n; ++i) {
            sum += data[i];
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(n * sizeof(double)));
    state.SetLabel("Sequential access");
}
BENCHMARK(BM_Cache_Sequential)
    ->Arg(1024)         // 8 KB — fits in L1
    ->Arg(4096)         // 32 KB — L1 boundary
    ->Arg(32768)        // 256 KB — L2
    ->Arg(524288)       // 4 MB — L3
    ->Arg(4194304);     // 32 MB — exceeds L3

static void BM_Cache_Strided(benchmark::State& state) {
    const Size n = static_cast<Size>(state.range(0));
    const Size stride = 8;  // 64-byte stride (cache line)
    std::vector<double> data(n, 1.0);
    double sum = 0.0;

    for (auto _ : state) {
        sum = 0.0;
        for (Size i = 0; i < n; i += stride) {
            sum += data[i];
        }
        benchmark::DoNotOptimize(sum);
    }

    state.SetBytesProcessed(state.iterations() *
                            static_cast<std::int64_t>((n / stride) * sizeof(double)));
    state.SetLabel("Strided access (stride=8)");
}
BENCHMARK(BM_Cache_Strided)
    ->Arg(1024)->Arg(4096)->Arg(32768)->Arg(524288)->Arg(4194304);

// ============================================================================
// 27.2.3: Contraction Loop — benchmark integral contraction
// ============================================================================

static void BM_Contraction_Overlap(benchmark::State& state) {
    auto basis = make_h2o_basis();
    Engine engine(basis);
    std::vector<Real> S;

    for (auto _ : state) {
        engine.compute_1e(Operator::overlap(), S);
        benchmark::DoNotOptimize(S.data());
    }

    auto nbf = basis.n_basis_functions();
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(nbf * nbf));
    state.SetLabel("Overlap contraction, nbf=" + std::to_string(nbf));
}
BENCHMARK(BM_Contraction_Overlap);

static void BM_Contraction_FockBuild(benchmark::State& state) {
    auto basis = make_h2o_basis();
    const Size nbf = basis.n_basis_functions();
    auto D = create_random_density(nbf);

    Engine engine(basis);
    consumers::FockBuilder fock(nbf);

    for (auto _ : state) {
        fock.reset();
        fock.set_density(D.data(), nbf);
        engine.compute_and_consume(Operator::coulomb(), fock);
        benchmark::DoNotOptimize(fock.get_coulomb_matrix().data());
    }

    Size n_shells = basis.n_shells();
    state.SetItemsProcessed(state.iterations() *
                            static_cast<std::int64_t>(n_shells * n_shells * n_shells * n_shells));
    state.SetLabel("ERI contraction + Fock, nbf=" + std::to_string(nbf));
}
BENCHMARK(BM_Contraction_FockBuild);

// ============================================================================
// 27.2.4: Memory Allocation — pool vs raw allocation
// ============================================================================

static void BM_Alloc_StdVector(benchmark::State& state) {
    const Size alloc_size = static_cast<Size>(state.range(0));

    for (auto _ : state) {
        std::vector<double> buf(alloc_size);
        benchmark::DoNotOptimize(buf.data());
    }

    state.SetLabel("std::vector allocation, size=" + std::to_string(alloc_size));
}
BENCHMARK(BM_Alloc_StdVector)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)->Arg(65536);

static void BM_Alloc_MemoryPool(benchmark::State& state) {
    const Size alloc_size = static_cast<Size>(state.range(0)) * sizeof(double);

    auto& pool = memory::get_thread_local_pool();

    for (auto _ : state) {
        auto buf = pool.acquire(alloc_size);
        benchmark::DoNotOptimize(buf.data());
    }

    state.SetLabel("MemoryPool allocation, bytes=" + std::to_string(alloc_size));
}
BENCHMARK(BM_Alloc_MemoryPool)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)->Arg(65536);

}  // namespace libaccint

BENCHMARK_MAIN();
