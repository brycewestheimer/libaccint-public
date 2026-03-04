// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_screening.cpp
/// @brief Schwarz screening performance benchmarks (Task 15.3.1)
///
/// Measures:
/// - Schwarz bounds precomputation cost vs system size
/// - Quartet screening (rejection) throughput
/// - Screened vs unscreened Fock build wall time
/// - Pass fraction / rejection rate vs threshold

#include <benchmark/benchmark.h>

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/screening/schwarz_bounds.hpp>
#include <libaccint/screening/screening_options.hpp>

#include "bench_helpers.hpp"

#include <vector>

namespace libaccint {

using bench::create_random_density;
using bench::make_h2o_atoms;

// ============================================================================
// Helper: Build a water chain of specified size
// ============================================================================

static std::vector<data::Atom> make_water_chain(int n_waters,
                                                 double separation = 5.0) {
    std::vector<data::Atom> atoms;
    for (int w = 0; w < n_waters; ++w) {
        double offset = w * separation;
        atoms.push_back({8, {offset, 0.0, 0.0}});
        atoms.push_back({1, {offset + 1.430429, 0.0, 1.107157}});
        atoms.push_back({1, {offset - 1.430429, 0.0, 1.107157}});
    }
    return atoms;
}

// ============================================================================
// Schwarz Bounds Construction Cost
// ============================================================================

/// @brief Measure time to precompute Schwarz bounds for water chains
static void BM_SchwarzBounds_Build(benchmark::State& state) {
    const int n_waters = static_cast<int>(state.range(0));
    auto atoms = make_water_chain(n_waters);
    auto basis = data::create_builtin_basis("sto-3g", atoms);

    for (auto _ : state) {
        screening::SchwarzBounds bounds(basis);
        benchmark::DoNotOptimize(bounds.max_bound());
        benchmark::ClobberMemory();
    }

    state.counters["n_shells"] = static_cast<double>(basis.n_shells());
    state.counters["nbf"] = static_cast<double>(basis.n_basis_functions());
}
BENCHMARK(BM_SchwarzBounds_Build)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

// ============================================================================
// Quartet Screening Throughput
// ============================================================================

/// @brief Measure screening lookup throughput (passes_screening per second)
static void BM_SchwarzBounds_LookupRate(benchmark::State& state) {
    const int n_waters = static_cast<int>(state.range(0));
    auto atoms = make_water_chain(n_waters);
    auto basis = data::create_builtin_basis("sto-3g", atoms);

    screening::SchwarzBounds bounds(basis);
    const Size n_shells = basis.n_shells();
    const Real threshold = 1e-12;

    for (auto _ : state) {
        Size passed = 0;
        for (Size i = 0; i < n_shells; ++i) {
            for (Size j = 0; j <= i; ++j) {
                for (Size k = 0; k <= i; ++k) {
                    Size l_max = (k == i) ? j : k;
                    for (Size l = 0; l <= l_max; ++l) {
                        if (bounds.passes_screening(i, j, k, l, threshold)) {
                            ++passed;
                        }
                    }
                }
            }
        }
        benchmark::DoNotOptimize(passed);
    }

    // Count total unique quartets with permutation symmetry
    Size total = 0;
    for (Size i = 0; i < n_shells; ++i)
        for (Size j = 0; j <= i; ++j)
            for (Size k = 0; k <= i; ++k) {
                Size l_max = (k == i) ? j : k;
                total += l_max + 1;
            }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(total));
    state.SetLabel(std::to_string(total) + " quartets");
}
BENCHMARK(BM_SchwarzBounds_LookupRate)->Arg(1)->Arg(2)->Arg(4);

// ============================================================================
// Pass Fraction vs Threshold
// ============================================================================

/// @brief Measure how many quartets pass at different thresholds
static void BM_SchwarzBounds_PassFraction(benchmark::State& state) {
    const int threshold_exp = static_cast<int>(state.range(0));
    const Real threshold = std::pow(10.0, static_cast<double>(threshold_exp));

    auto atoms = make_water_chain(3);
    auto basis = data::create_builtin_basis("sto-3g", atoms);

    screening::SchwarzBounds bounds(basis);

    for (auto _ : state) {
        Size count = bounds.count_passing_quartets(threshold);
        benchmark::DoNotOptimize(count);
    }

    Real fraction = bounds.estimate_pass_fraction(threshold);
    state.counters["pass_fraction"] = fraction;
    state.counters["threshold"] = threshold;
}
BENCHMARK(BM_SchwarzBounds_PassFraction)
    ->Arg(-8)->Arg(-10)->Arg(-12)->Arg(-14)->Arg(-16);

// ============================================================================
// Screened vs Unscreened Fock Build
// ============================================================================

/// @brief Compare unscreened Fock build cost
static void BM_FockBuild_Unscreened(benchmark::State& state) {
    const int n_waters = static_cast<int>(state.range(0));
    auto atoms = make_water_chain(n_waters);
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

    state.SetLabel("Unscreened, nbf=" + std::to_string(nbf));
}
BENCHMARK(BM_FockBuild_Unscreened)->Arg(1)->Arg(2)->Arg(3)->Arg(4);

/// @brief Compare screened Fock build cost with statistics
static void BM_FockBuild_Screened(benchmark::State& state) {
    const int n_waters = static_cast<int>(state.range(0));
    auto atoms = make_water_chain(n_waters);
    auto basis = data::create_builtin_basis("sto-3g", atoms);
    const Size nbf = basis.n_basis_functions();
    auto D = create_random_density(nbf);

    Engine engine(basis);
    engine.precompute_schwarz_bounds();
    engine.set_density_matrix(D.data(), nbf);

    consumers::FockBuilder fock(nbf);

    auto options = screening::ScreeningOptions::normal();
    options.enable_statistics = true;

    for (auto _ : state) {
        fock.reset();
        fock.set_density(D.data(), nbf);
        engine.compute_and_consume(Operator::coulomb(), fock, options);
        benchmark::DoNotOptimize(fock.get_coulomb_matrix().data());
    }

    state.SetLabel("Screened, nbf=" + std::to_string(nbf));
}
BENCHMARK(BM_FockBuild_Screened)->Arg(1)->Arg(2)->Arg(3)->Arg(4);

}  // namespace libaccint

BENCHMARK_MAIN();
