// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_df_performance.cpp
/// @brief DF performance benchmarks (Tasks 22.4.1 and 22.4.2)
///
/// Benchmarks DF-Fock construction vs system size. Measures timing of
/// metric build, B-tensor formation, and J/K contractions.

#include <benchmark/benchmark.h>

#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/consumers/df_fock_builder.hpp>
#include <libaccint/data/auxiliary_basis_data.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/core/types.hpp>

#include <random>
#include <vector>

namespace libaccint {

// ============================================================================
// Helpers
// ============================================================================

/// @brief Generate a random symmetric positive semi-definite density matrix
std::vector<Real> make_random_density(Size n, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);

    // Build random matrix C
    std::vector<Real> C(n * n);
    for (auto& v : C) {
        v = dist(gen);
    }

    // D = C^T * C (positive semi-definite), normalized
    std::vector<Real> D(n * n, 0.0);
    for (Size i = 0; i < n; ++i) {
        for (Size j = 0; j < n; ++j) {
            Real s = 0.0;
            for (Size k = 0; k < n; ++k) {
                s += C[k * n + i] * C[k * n + j];
            }
            D[i * n + j] = s / static_cast<Real>(n);
        }
    }
    return D;
}

/// @brief Create a chain of H atoms with given spacing
std::vector<data::Atom> make_hydrogen_chain(Size n_atoms,
                                             Real spacing = 2.0) {
    std::vector<data::Atom> atoms;
    atoms.reserve(n_atoms);
    for (Size i = 0; i < n_atoms; ++i) {
        atoms.push_back({1, {static_cast<Real>(i) * spacing, 0.0, 0.0}});
    }
    return atoms;
}

// ============================================================================
// Task 22.4.1: DF Performance Benchmarks
// ============================================================================

/// @brief Benchmark DF-Fock full compute for H2
static void BM_DFFock_H2(benchmark::State& state) {
    auto atoms = make_hydrogen_chain(2);
    auto orbital = data::create_sto3g(atoms);
    auto aux = data::create_builtin_auxiliary_basis("cc-pVDZ-RI", atoms);
    Size n = orbital.n_basis_functions();
    auto D = make_random_density(n);

    for (auto _ : state) {
        consumers::DFFockBuilder builder(orbital, aux);
        builder.set_density(D);
        auto F = builder.compute();
        benchmark::DoNotOptimize(F.data());
    }

    state.SetItemsProcessed(state.iterations());
    state.counters["n_ao"] = static_cast<double>(n);
    state.counters["n_aux"] = static_cast<double>(aux.n_functions());
}
BENCHMARK(BM_DFFock_H2);

/// @brief Benchmark DF-Fock full compute for H2O
static void BM_DFFock_H2O(benchmark::State& state) {
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.2217}},
        {1, {0.0, 1.4309, -0.8867}},
        {1, {0.0, -1.4309, -0.8867}},
    };
    auto orbital = data::create_sto3g(atoms);
    auto aux = data::create_builtin_auxiliary_basis("cc-pVDZ-RI", atoms);
    Size n = orbital.n_basis_functions();
    auto D = make_random_density(n);

    for (auto _ : state) {
        consumers::DFFockBuilder builder(orbital, aux);
        builder.set_density(D);
        auto F = builder.compute();
        benchmark::DoNotOptimize(F.data());
    }

    state.SetItemsProcessed(state.iterations());
    state.counters["n_ao"] = static_cast<double>(n);
    state.counters["n_aux"] = static_cast<double>(aux.n_functions());
}
BENCHMARK(BM_DFFock_H2O);

/// @brief Benchmark DF-Coulomb component only
static void BM_DFCoulomb_H2O(benchmark::State& state) {
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.2217}},
        {1, {0.0, 1.4309, -0.8867}},
        {1, {0.0, -1.4309, -0.8867}},
    };
    auto orbital = data::create_sto3g(atoms);
    auto aux = data::create_builtin_auxiliary_basis("cc-pVDZ-RI", atoms);
    Size n = orbital.n_basis_functions();
    auto D = make_random_density(n);

    consumers::DFFockBuilder builder(orbital, aux);
    builder.set_density(D);
    builder.initialize();

    for (auto _ : state) {
        auto J = builder.compute_coulomb();
        benchmark::DoNotOptimize(J.data());
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_DFCoulomb_H2O);

/// @brief Benchmark DF-Exchange component only
static void BM_DFExchange_H2O(benchmark::State& state) {
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.2217}},
        {1, {0.0, 1.4309, -0.8867}},
        {1, {0.0, -1.4309, -0.8867}},
    };
    auto orbital = data::create_sto3g(atoms);
    auto aux = data::create_builtin_auxiliary_basis("cc-pVDZ-RI", atoms);
    Size n = orbital.n_basis_functions();
    auto D = make_random_density(n);

    consumers::DFFockBuilder builder(orbital, aux);
    builder.set_density(D);
    builder.initialize();

    for (auto _ : state) {
        auto K = builder.compute_exchange();
        benchmark::DoNotOptimize(K.data());
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_DFExchange_H2O);

// ============================================================================
// Task 22.4.2: DF Scaling Analysis
// ============================================================================

/// @brief Benchmark DF-Fock scaling with hydrogen chain length
static void BM_DFFock_Scaling(benchmark::State& state) {
    auto n_atoms = static_cast<Size>(state.range(0));
    auto atoms = make_hydrogen_chain(n_atoms);
    auto orbital = data::create_sto3g(atoms);
    auto aux = data::create_builtin_auxiliary_basis("cc-pVDZ-RI", atoms);
    Size n = orbital.n_basis_functions();
    auto D = make_random_density(n);

    for (auto _ : state) {
        consumers::DFFockBuilder builder(orbital, aux);
        builder.set_density(D);
        auto F = builder.compute();
        benchmark::DoNotOptimize(F.data());
    }

    state.SetItemsProcessed(state.iterations());
    state.counters["n_atoms"] = static_cast<double>(n_atoms);
    state.counters["n_ao"] = static_cast<double>(n);
    state.counters["n_aux"] = static_cast<double>(aux.n_functions());
}
BENCHMARK(BM_DFFock_Scaling)->Arg(2)->Arg(4)->Arg(6)->Arg(8);

/// @brief Benchmark DF-Fock initialization (metric build + Cholesky)
static void BM_DFFock_Init(benchmark::State& state) {
    auto n_atoms = static_cast<Size>(state.range(0));
    auto atoms = make_hydrogen_chain(n_atoms);
    auto orbital = data::create_sto3g(atoms);
    auto aux = data::create_builtin_auxiliary_basis("cc-pVDZ-RI", atoms);

    for (auto _ : state) {
        consumers::DFFockBuilder builder(orbital, aux);
        Size n = orbital.n_basis_functions();
        std::vector<Real> D(n * n, 0.0);
        D[0] = 1.0;
        builder.set_density(D);
        builder.initialize();
        benchmark::DoNotOptimize(&builder);
    }

    state.SetItemsProcessed(state.iterations());
    state.counters["n_atoms"] = static_cast<double>(n_atoms);
    state.counters["n_aux"] = static_cast<double>(aux.n_functions());
}
BENCHMARK(BM_DFFock_Init)->Arg(2)->Arg(4)->Arg(6);

/// @brief Compare different auxiliary basis sets
static void BM_DFFock_AuxBasis(benchmark::State& state) {
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.2217}},
        {1, {0.0, 1.4309, -0.8867}},
        {1, {0.0, -1.4309, -0.8867}},
    };
    auto orbital = data::create_sto3g(atoms);
    Size n = orbital.n_basis_functions();
    auto D = make_random_density(n);

    auto aux_idx = state.range(0);
    std::string aux_name;
    switch (aux_idx) {
        case 0: aux_name = "cc-pVDZ-RI"; break;
        case 1: aux_name = "cc-pVTZ-RI"; break;
        case 2: aux_name = "def2-SVP-JKFIT"; break;
        case 3: aux_name = "def2-TZVP-JKFIT"; break;
        default: aux_name = "cc-pVDZ-RI"; break;
    }
    auto aux = data::create_builtin_auxiliary_basis(aux_name, atoms);

    for (auto _ : state) {
        consumers::DFFockBuilder builder(orbital, aux);
        builder.set_density(D);
        auto F = builder.compute();
        benchmark::DoNotOptimize(F.data());
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel(aux_name);
    state.counters["n_aux"] = static_cast<double>(aux.n_functions());
}
BENCHMARK(BM_DFFock_AuxBasis)->Arg(0)->Arg(1)->Arg(2)->Arg(3);

}  // namespace libaccint

BENCHMARK_MAIN();
