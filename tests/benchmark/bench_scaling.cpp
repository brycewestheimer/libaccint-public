// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_scaling.cpp
/// @brief OpenMP thread scaling benchmarks

#include <benchmark/benchmark.h>
#include <libaccint/engine/engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/utils/simd.hpp>

#include "bench_helpers.hpp"

#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libaccint {

using bench::create_random_density;
using bench::create_h2o_sto3g_shells;

// ============================================================================
// Thread Scaling Benchmark
// ============================================================================

static void BM_ThreadScaling_Atomic(benchmark::State& state) {
    const int n_threads = static_cast<int>(state.range(0));

    auto shells = create_h2o_sto3g_shells();
    BasisSet basis(shells);
    const Size nbf = basis.n_basis_functions();
    auto D = create_random_density(nbf);

    Engine engine(basis);
    consumers::FockBuilder fock(nbf);
    fock.set_threading_strategy(consumers::FockThreadingStrategy::Atomic);

    for (auto _ : state) {
        fock.reset();
        fock.set_density(D.data(), nbf);
        engine.compute_and_consume_parallel(Operator::coulomb(), fock, n_threads);
        benchmark::DoNotOptimize(fock.get_coulomb_matrix().data());
    }

    Size n_shells = basis.n_shells();
    Size n_quartets = n_shells * n_shells * n_shells * n_shells;
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n_quartets));
    state.counters["threads"] = static_cast<double>(n_threads);
}
BENCHMARK(BM_ThreadScaling_Atomic)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)
    ->UseRealTime();

static void BM_ThreadScaling_ThreadLocal(benchmark::State& state) {
    const int n_threads = static_cast<int>(state.range(0));

    auto shells = create_h2o_sto3g_shells();
    BasisSet basis(shells);
    const Size nbf = basis.n_basis_functions();
    auto D = create_random_density(nbf);

    Engine engine(basis);
    consumers::FockBuilder fock(nbf);
    fock.set_threading_strategy(consumers::FockThreadingStrategy::ThreadLocal);

    for (auto _ : state) {
        fock.reset();
        fock.set_density(D.data(), nbf);
        fock.prepare_parallel(n_threads);
        engine.compute_and_consume_parallel(Operator::coulomb(), fock, n_threads);
        fock.finalize_parallel();
        benchmark::DoNotOptimize(fock.get_coulomb_matrix().data());
    }

    Size n_shells = basis.n_shells();
    Size n_quartets = n_shells * n_shells * n_shells * n_shells;
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n_quartets));
    state.counters["threads"] = static_cast<double>(n_threads);
}
BENCHMARK(BM_ThreadScaling_ThreadLocal)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)
    ->UseRealTime();

// ============================================================================
// Report Thread Count
// ============================================================================

static void BM_ReportSystemInfo(benchmark::State& state) {
    for (auto _ : state) {
        // No-op, just to report system info
    }

#if defined(_OPENMP)
    state.counters["max_threads"] = static_cast<double>(omp_get_max_threads());
    state.counters["num_procs"] = static_cast<double>(omp_get_num_procs());
#else
    state.counters["max_threads"] = 1.0;
    state.counters["num_procs"] = 1.0;
#endif

    state.counters["simd_width"] = static_cast<double>(simd::simd_width);
}
BENCHMARK(BM_ReportSystemInfo);

}  // namespace libaccint

BENCHMARK_MAIN();
