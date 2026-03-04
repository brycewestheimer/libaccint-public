// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_kernel_microbenchmarks.cpp
/// @brief Per-kernel microbenchmarks across all angular momentum combinations
///
/// Task 27.1.3: Sweeps over kernel types (overlap, kinetic, nuclear, ERI)
/// and all supported AM combinations to build a performance matrix.

#include <benchmark/benchmark.h>

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/operators/operator.hpp>

#include <vector>

namespace libaccint {

// ============================================================================
// Shell Factory for AM Sweep
// ============================================================================

/// @brief Create a minimal two-shell basis for a given pair of AMs
static std::pair<BasisSet, std::vector<Shell>>
create_am_pair_basis(int am_a, int am_b) {
    // STO-3G-like exponents and coefficients for any AM
    std::vector<double> exp_a = {3.42525091, 0.62391373, 0.16885540};
    std::vector<double> coef_a = {0.15432897, 0.53532814, 0.44463454};
    std::vector<double> exp_b = {3.42525091, 0.62391373, 0.16885540};
    std::vector<double> coef_b = {0.15432897, 0.53532814, 0.44463454};

    Point3D center_a{0.0, 0.0, 0.0};
    Point3D center_b{2.0, 0.0, 0.0};

    auto am_enum_a = static_cast<AngularMomentum>(am_a);
    auto am_enum_b = static_cast<AngularMomentum>(am_b);

    Shell shell_a(am_enum_a, center_a, exp_a, coef_a);
    shell_a.set_shell_index(0);
    shell_a.set_atom_index(0);
    shell_a.set_function_index(0);

    int nfunc_a = shell_a.n_functions();

    Shell shell_b(am_enum_b, center_b, exp_b, coef_b);
    shell_b.set_shell_index(1);
    shell_b.set_atom_index(1);
    shell_b.set_function_index(nfunc_a);

    std::vector<Shell> shells = {shell_a, shell_b};
    return {BasisSet(shells), shells};
}

// ============================================================================
// 1e Integral Microbenchmarks: Overlap
// ============================================================================

static void BM_Overlap_AM(benchmark::State& state) {
    int am_a = static_cast<int>(state.range(0));
    int am_b = static_cast<int>(state.range(1));

    auto [basis, shells] = create_am_pair_basis(am_a, am_b);
    Engine engine(basis);
    std::vector<Real> S;

    for (auto _ : state) {
        engine.compute_1e(Operator::overlap(), S);
        benchmark::DoNotOptimize(S.data());
    }

    auto nbf = basis.n_basis_functions();
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(nbf * nbf));
    state.SetLabel("(" + std::to_string(am_a) + "," + std::to_string(am_b) + ")");
}

// Register AM sweep for overlap: (0,0) through (3,3)
BENCHMARK(BM_Overlap_AM)
    ->Args({0, 0})->Args({0, 1})->Args({0, 2})->Args({0, 3})
    ->Args({1, 0})->Args({1, 1})->Args({1, 2})->Args({1, 3})
    ->Args({2, 0})->Args({2, 1})->Args({2, 2})->Args({2, 3})
    ->Args({3, 0})->Args({3, 1})->Args({3, 2})->Args({3, 3});

// ============================================================================
// 1e Integral Microbenchmarks: Kinetic
// ============================================================================

static void BM_Kinetic_AM(benchmark::State& state) {
    int am_a = static_cast<int>(state.range(0));
    int am_b = static_cast<int>(state.range(1));

    auto [basis, shells] = create_am_pair_basis(am_a, am_b);
    Engine engine(basis);
    std::vector<Real> T;

    for (auto _ : state) {
        engine.compute_1e(Operator::kinetic(), T);
        benchmark::DoNotOptimize(T.data());
    }

    auto nbf = basis.n_basis_functions();
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(nbf * nbf));
    state.SetLabel("(" + std::to_string(am_a) + "," + std::to_string(am_b) + ")");
}

BENCHMARK(BM_Kinetic_AM)
    ->Args({0, 0})->Args({0, 1})->Args({0, 2})->Args({0, 3})
    ->Args({1, 0})->Args({1, 1})->Args({1, 2})->Args({1, 3})
    ->Args({2, 0})->Args({2, 1})->Args({2, 2})->Args({2, 3})
    ->Args({3, 0})->Args({3, 1})->Args({3, 2})->Args({3, 3});

// ============================================================================
// 1e Integral Microbenchmarks: Nuclear
// ============================================================================

static void BM_Nuclear_AM(benchmark::State& state) {
    int am_a = static_cast<int>(state.range(0));
    int am_b = static_cast<int>(state.range(1));

    auto [basis, shells] = create_am_pair_basis(am_a, am_b);

    // Create nuclear attraction operator with two point charges
    PointChargeParams charges;
    charges.x = {0.0, 2.0};
    charges.y = {0.0, 0.0};
    charges.z = {0.0, 0.0};
    charges.charge = {8.0, 1.0};

    Engine engine(basis);
    std::vector<Real> V;

    for (auto _ : state) {
        engine.compute_1e(Operator::nuclear(charges), V);
        benchmark::DoNotOptimize(V.data());
    }

    auto nbf = basis.n_basis_functions();
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(nbf * nbf));
    state.SetLabel("(" + std::to_string(am_a) + "," + std::to_string(am_b) + ")");
}

BENCHMARK(BM_Nuclear_AM)
    ->Args({0, 0})->Args({0, 1})->Args({0, 2})->Args({0, 3})
    ->Args({1, 0})->Args({1, 1})->Args({1, 2})->Args({1, 3})
    ->Args({2, 0})->Args({2, 1})->Args({2, 2})->Args({2, 3})
    ->Args({3, 0})->Args({3, 1})->Args({3, 2})->Args({3, 3});

// ============================================================================
// 2e ERI Fock Build (indirect measure of ERI kernel performance)
// ============================================================================

static void BM_ERI_FockBuild_AM(benchmark::State& state) {
    int am = static_cast<int>(state.range(0));

    // Create a small basis with shells of the specified AM
    auto [basis, shells] = create_am_pair_basis(am, am);
    const Size nbf = basis.n_basis_functions();

    // Create a random density matrix
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0 / static_cast<Real>(nbf);
    }

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
    state.SetLabel("AM=" + std::to_string(am));
}

BENCHMARK(BM_ERI_FockBuild_AM)->Arg(0)->Arg(1)->Arg(2)->Arg(3);

}  // namespace libaccint

BENCHMARK_MAIN();
