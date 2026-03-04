// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_hf_endtoend.cpp
/// @brief End-to-end Hartree-Fock benchmark (Task 27.1.4)
///
/// Benchmarks complete Fock build for representative molecules and basis sets:
/// - H2O / STO-3G (small, 7 basis functions)
/// - CH4 / STO-3G (small, 9 basis functions)
/// - Water chain variants for system-size scaling

#include <benchmark/benchmark.h>

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/operators/operator.hpp>

#include "bench_helpers.hpp"

#include <vector>

namespace libaccint {

using bench::create_random_density;
using bench::make_nuclear_charges;

// ============================================================================
// H2O / STO-3G Full Fock Build
// ============================================================================

static void BM_HF_H2O_STO3G(benchmark::State& state) {
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {1.430429, 0.0, 1.107157}},
        {1, {-1.430429, 0.0, 1.107157}}
    };
    auto basis = data::create_builtin_basis("sto-3g", atoms);
    const Size nbf = basis.n_basis_functions();
    auto D = create_random_density(nbf);

    Engine engine(basis);
    consumers::FockBuilder fock(nbf);
    auto charges = make_nuclear_charges(atoms);

    // Pre-allocate 1e matrices
    std::vector<Real> S, T, V;

    // Full Fock build: 1e matrices + 2e Coulomb + Exchange
    for (auto _ : state) {
        // 1e matrices
        engine.compute_1e(Operator::overlap(), S);
        engine.compute_1e(Operator::kinetic(), T);
        engine.compute_1e(Operator::nuclear(charges), V);

        // 2e Fock build
        fock.reset();
        fock.set_density(D.data(), nbf);
        engine.compute_and_consume(Operator::coulomb(), fock);

        benchmark::DoNotOptimize(S.data());
        benchmark::DoNotOptimize(T.data());
        benchmark::DoNotOptimize(V.data());
        benchmark::DoNotOptimize(fock.get_coulomb_matrix().data());
    }

    state.SetLabel("H2O/STO-3G, nbf=" + std::to_string(nbf));
}
BENCHMARK(BM_HF_H2O_STO3G);

// ============================================================================
// CH4 / STO-3G Full Fock Build
// ============================================================================

static void BM_HF_CH4_STO3G(benchmark::State& state) {
    const double a = 2.0503;
    const double t = 1.0 / std::sqrt(3.0);
    std::vector<data::Atom> atoms = {
        {6, {0.0, 0.0, 0.0}},
        {1, { a * t,  a * t,  a * t}},
        {1, { a * t, -a * t, -a * t}},
        {1, {-a * t,  a * t, -a * t}},
        {1, {-a * t, -a * t,  a * t}}
    };
    auto basis = data::create_builtin_basis("sto-3g", atoms);
    const Size nbf = basis.n_basis_functions();
    auto D = create_random_density(nbf);

    Engine engine(basis);
    consumers::FockBuilder fock(nbf);
    auto charges = make_nuclear_charges(atoms);

    // Pre-allocate 1e matrices
    std::vector<Real> S, T, V;

    for (auto _ : state) {
        engine.compute_1e(Operator::overlap(), S);
        engine.compute_1e(Operator::kinetic(), T);
        engine.compute_1e(Operator::nuclear(charges), V);

        fock.reset();
        fock.set_density(D.data(), nbf);
        engine.compute_and_consume(Operator::coulomb(), fock);

        benchmark::DoNotOptimize(S.data());
        benchmark::DoNotOptimize(fock.get_coulomb_matrix().data());
    }

    state.SetLabel("CH4/STO-3G, nbf=" + std::to_string(nbf));
}
BENCHMARK(BM_HF_CH4_STO3G);

// ============================================================================
// Water chain scaling (2, 3, 4 water molecules)
// ============================================================================

static void BM_HF_WaterChain(benchmark::State& state) {
    const int n_waters = static_cast<int>(state.range(0));
    const double separation = 5.0;  // Bohr

    std::vector<data::Atom> atoms;
    for (int w = 0; w < n_waters; ++w) {
        double offset = w * separation;
        atoms.push_back({8, {offset, 0.0, 0.0}});
        atoms.push_back({1, {offset + 1.430429, 0.0, 1.107157}});
        atoms.push_back({1, {offset - 1.430429, 0.0, 1.107157}});
    }

    auto basis = data::create_builtin_basis("sto-3g", atoms);
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
    state.SetLabel(std::to_string(n_waters) + " H2O, nbf=" + std::to_string(nbf));
}
BENCHMARK(BM_HF_WaterChain)->Arg(1)->Arg(2)->Arg(3)->Arg(4);

// ============================================================================
// Parallel Fock Build (thread scaling)
// ============================================================================

static void BM_HF_Parallel(benchmark::State& state) {
    const int n_threads = static_cast<int>(state.range(0));

    // 3 water molecules for a reasonable workload
    std::vector<data::Atom> atoms;
    for (int w = 0; w < 3; ++w) {
        double offset = w * 5.0;
        atoms.push_back({8, {offset, 0.0, 0.0}});
        atoms.push_back({1, {offset + 1.430429, 0.0, 1.107157}});
        atoms.push_back({1, {offset - 1.430429, 0.0, 1.107157}});
    }

    auto basis = data::create_builtin_basis("sto-3g", atoms);
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

    state.SetLabel(std::to_string(n_threads) + " threads, nbf=" + std::to_string(nbf));
}
BENCHMARK(BM_HF_Parallel)->Arg(1)->Arg(2)->Arg(4);

}  // namespace libaccint

BENCHMARK_MAIN();
