// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_fock_build.cpp
/// @brief Fock matrix construction benchmarks with threading comparison

#include <benchmark/benchmark.h>
#include <libaccint/engine/engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/consumers/fock_builder.hpp>

#include "bench_helpers.hpp"

#include <vector>

namespace libaccint {

using bench::create_random_density;
using bench::create_h2o_sto3g_shells;

// ============================================================================
// Sequential Fock Build Benchmarks
// ============================================================================

static void BM_FockBuild_Sequential_H2O_STO3G(benchmark::State& state) {
    auto shells = create_h2o_sto3g_shells();
    BasisSet basis(shells);
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
    Size n_quartets = n_shells * n_shells * n_shells * n_shells;
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n_quartets));
}
BENCHMARK(BM_FockBuild_Sequential_H2O_STO3G);

// ============================================================================
// Parallel Fock Build with Atomic Strategy
// ============================================================================

static void BM_FockBuild_Parallel_Atomic(benchmark::State& state) {
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
}
BENCHMARK(BM_FockBuild_Parallel_Atomic)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

// ============================================================================
// Parallel Fock Build with Thread-Local Strategy
// ============================================================================

static void BM_FockBuild_Parallel_ThreadLocal(benchmark::State& state) {
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
}
BENCHMARK(BM_FockBuild_Parallel_ThreadLocal)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

// ============================================================================
// One-Electron Matrix Benchmarks
// ============================================================================

static void BM_OverlapMatrix_H2O(benchmark::State& state) {
    auto shells = create_h2o_sto3g_shells();
    BasisSet basis(shells);
    Engine engine(basis);
    std::vector<Real> S;

    for (auto _ : state) {
        engine.compute_1e(Operator::overlap(), S);
        benchmark::DoNotOptimize(S.data());
    }

    state.SetItemsProcessed(state.iterations() * basis.n_basis_functions() *
                            basis.n_basis_functions());
}
BENCHMARK(BM_OverlapMatrix_H2O);

static void BM_KineticMatrix_H2O(benchmark::State& state) {
    auto shells = create_h2o_sto3g_shells();
    BasisSet basis(shells);
    Engine engine(basis);
    std::vector<Real> T;

    for (auto _ : state) {
        engine.compute_1e(Operator::kinetic(), T);
        benchmark::DoNotOptimize(T.data());
    }

    state.SetItemsProcessed(state.iterations() * basis.n_basis_functions() *
                            basis.n_basis_functions());
}
BENCHMARK(BM_KineticMatrix_H2O);

}  // namespace libaccint

BENCHMARK_MAIN();
