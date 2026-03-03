// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_comprehensive.cpp
/// @brief Phase 17 — Comprehensive benchmarks across molecules and basis sets
///
/// Measures:
/// 1. One-electron integral timing (overlap, kinetic, nuclear)
/// 2. Two-electron Fock build timing
/// 3. Screened vs unscreened Fock build comparison
/// 4. Thread scaling (1, 2, 4, 8 threads)
/// 5. Individual 1e+2e timing breakdown

#include <benchmark/benchmark.h>

#include <libaccint/engine/engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/data/bse_json_parser.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/screening/schwarz_bounds.hpp>
#include <libaccint/screening/screening_options.hpp>
#include <libaccint/core/types.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace libaccint {

using namespace data;
using namespace consumers;

// =============================================================================
// Helpers
// =============================================================================

namespace {

std::vector<Atom> make_h2o_atoms() {
    return {
        {8, {0.0, 0.0, 0.0}},
        {1, {0.0, 1.43233673, -1.10866041}},
        {1, {0.0, -1.43233673, -1.10866041}},
    };
}

std::vector<Atom> make_ch4_atoms() {
    const double r = 2.049803133;
    const double d = r / std::sqrt(3.0);
    return {
        {6, {0.0, 0.0, 0.0}},
        {1, { d,  d,  d}},
        {1, { d, -d, -d}},
        {1, {-d,  d, -d}},
        {1, {-d, -d,  d}},
    };
}

std::string find_basis_file(const std::string& filename) {
    std::vector<std::string> search_paths = {
        "share/basis_sets/" + filename,
        "../share/basis_sets/" + filename,
        "../../share/basis_sets/" + filename,
        "../../../share/basis_sets/" + filename,
    };
    const char* src_dir = std::getenv("LIBACCINT_SOURCE_DIR");
    if (src_dir) {
        search_paths.insert(search_paths.begin(),
            std::string(src_dir) + "/share/basis_sets/" + filename);
    }

    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path)) return path;
    }
    return {};
}

std::vector<Real> create_random_density(Size nbf, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<Real> D(nbf * nbf);
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i; j < nbf; ++j) {
            double val = dist(gen);
            D[i * nbf + j] = val;
            D[j * nbf + i] = val;
        }
    }
    return D;
}

PointChargeParams make_charges(const std::vector<Atom>& atoms) {
    PointChargeParams charges;
    for (const auto& atom : atoms) {
        charges.x.push_back(atom.position.x);
        charges.y.push_back(atom.position.y);
        charges.z.push_back(atom.position.z);
        charges.charge.push_back(static_cast<Real>(atom.atomic_number));
    }
    return charges;
}

// Molecule/basis fixture holder — avoids repeated loading in tight loops
struct BenchFixture {
    std::string mol_name;
    std::string basis_name;
    std::vector<Atom> atoms;
    BasisSet basis;
    Engine engine;
    std::vector<Real> density;
    PointChargeParams charges;

    BenchFixture(std::string mol, std::string bname,
                 std::vector<Atom> a, BasisSet b)
        : mol_name(std::move(mol)), basis_name(std::move(bname)),
          atoms(std::move(a)), basis(std::move(b)),
          engine(basis),
          density(create_random_density(basis.n_basis_functions())),
          charges(make_charges(atoms)) {}
};

// Lazy-initialised fixtures for each mol/basis combo.
// Returns nullptr if the basis file is not found.
BenchFixture* get_fixture(const std::string& mol, const std::string& basis_file,
                          const std::string& basis_label) {
    // Use a static map keyed on mol+basis
    static std::unordered_map<std::string, std::unique_ptr<BenchFixture>> cache;
    std::string key = mol + "/" + basis_label;

    auto it = cache.find(key);
    if (it != cache.end()) return it->second.get();

    auto atoms = (mol == "H2O") ? make_h2o_atoms() : make_ch4_atoms();
    std::string path = find_basis_file(basis_file);
    if (path.empty()) {
        cache[key] = nullptr;
        return nullptr;
    }
    auto bs = BseJsonParser::parse_file(path, atoms);
    cache[key] = std::make_unique<BenchFixture>(
        mol, basis_label, std::move(atoms), std::move(bs));
    return cache[key].get();
}

// Shorthand molecule/basis descriptors
struct MolBasis {
    const char* mol;
    const char* file;
    const char* label;
};

constexpr MolBasis kCombinations[] = {
    {"H2O", "sto-3g.json",      "STO3G"},
    {"H2O", "6-31g.json",       "631G"},
    {"H2O", "aug-cc-pvdz.json", "augccpVDZ"},
    {"CH4", "sto-3g.json",      "STO3G"},
    {"CH4", "6-31g.json",       "631G"},
    {"CH4", "aug-cc-pvdz.json", "augccpVDZ"},
};

}  // anonymous namespace

// =============================================================================
// 1. One-Electron Integral Benchmarks
// =============================================================================

static void BM_Overlap(benchmark::State& state) {
    auto idx = static_cast<size_t>(state.range(0));
    if (idx >= std::size(kCombinations)) { state.SkipWithError("bad index"); return; }
    const auto& mb = kCombinations[idx];
    auto* f = get_fixture(mb.mol, mb.file, mb.label);
    if (!f) { state.SkipWithError("basis not found"); return; }

    std::vector<Real> S;
    for (auto _ : state) {
        f->engine.compute_1e(Operator::overlap(), S);
        benchmark::DoNotOptimize(S.data());
    }
    Size nbf = f->basis.n_basis_functions();
    state.SetItemsProcessed(state.iterations() *
                            static_cast<int64_t>(nbf * nbf));
    state.SetLabel(std::string(mb.mol) + "/" + mb.label);
}

static void BM_Kinetic(benchmark::State& state) {
    auto idx = static_cast<size_t>(state.range(0));
    if (idx >= std::size(kCombinations)) { state.SkipWithError("bad index"); return; }
    const auto& mb = kCombinations[idx];
    auto* f = get_fixture(mb.mol, mb.file, mb.label);
    if (!f) { state.SkipWithError("basis not found"); return; }

    std::vector<Real> T;
    for (auto _ : state) {
        f->engine.compute_1e(Operator::kinetic(), T);
        benchmark::DoNotOptimize(T.data());
    }
    Size nbf = f->basis.n_basis_functions();
    state.SetItemsProcessed(state.iterations() *
                            static_cast<int64_t>(nbf * nbf));
    state.SetLabel(std::string(mb.mol) + "/" + mb.label);
}

static void BM_Nuclear(benchmark::State& state) {
    auto idx = static_cast<size_t>(state.range(0));
    if (idx >= std::size(kCombinations)) { state.SkipWithError("bad index"); return; }
    const auto& mb = kCombinations[idx];
    auto* f = get_fixture(mb.mol, mb.file, mb.label);
    if (!f) { state.SkipWithError("basis not found"); return; }

    std::vector<Real> V;
    auto nuc_op = Operator::nuclear(f->charges);
    for (auto _ : state) {
        f->engine.compute_1e(nuc_op, V);
        benchmark::DoNotOptimize(V.data());
    }
    Size nbf = f->basis.n_basis_functions();
    state.SetItemsProcessed(state.iterations() *
                            static_cast<int64_t>(nbf * nbf));
    state.SetLabel(std::string(mb.mol) + "/" + mb.label);
}

// Register for each mol/basis index 0..5
BENCHMARK(BM_Overlap)->DenseRange(0, 5)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Kinetic)->DenseRange(0, 5)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Nuclear)->DenseRange(0, 5)->Unit(benchmark::kMicrosecond);

// =============================================================================
// 2. Two-Electron Fock Build Benchmarks
// =============================================================================

static void BM_FockBuild(benchmark::State& state) {
    auto idx = static_cast<size_t>(state.range(0));
    if (idx >= std::size(kCombinations)) { state.SkipWithError("bad index"); return; }
    const auto& mb = kCombinations[idx];
    auto* f = get_fixture(mb.mol, mb.file, mb.label);
    if (!f) { state.SkipWithError("basis not found"); return; }

    Size nbf = f->basis.n_basis_functions();

    for (auto _ : state) {
        FockBuilder fock(nbf);
        fock.set_density(f->density.data(), nbf);
        f->engine.compute_and_consume(Operator::coulomb(), fock);
        benchmark::DoNotOptimize(fock.get_coulomb_matrix().data());
    }
    Size n_shells = f->basis.n_shells();
    state.SetItemsProcessed(state.iterations() *
        static_cast<int64_t>(n_shells * n_shells * n_shells * n_shells));
    state.SetLabel(std::string(mb.mol) + "/" + mb.label);
}

BENCHMARK(BM_FockBuild)->DenseRange(0, 5)->Unit(benchmark::kMicrosecond);

// =============================================================================
// 3. Screened vs Unscreened Fock Build
// =============================================================================

static void BM_FockBuild_Unscreened(benchmark::State& state) {
    auto idx = static_cast<size_t>(state.range(0));
    if (idx >= std::size(kCombinations)) { state.SkipWithError("bad index"); return; }
    const auto& mb = kCombinations[idx];
    auto* f = get_fixture(mb.mol, mb.file, mb.label);
    if (!f) { state.SkipWithError("basis not found"); return; }

    Size nbf = f->basis.n_basis_functions();

    for (auto _ : state) {
        FockBuilder fock(nbf);
        fock.set_density(f->density.data(), nbf);
        f->engine.compute_and_consume(Operator::coulomb(), fock);
        benchmark::DoNotOptimize(fock.get_coulomb_matrix().data());
    }
    state.SetLabel(std::string(mb.mol) + "/" + mb.label + "/unscreened");
}

static void BM_FockBuild_Screened(benchmark::State& state) {
    auto idx = static_cast<size_t>(state.range(0));
    if (idx >= std::size(kCombinations)) { state.SkipWithError("bad index"); return; }
    const auto& mb = kCombinations[idx];
    auto* f = get_fixture(mb.mol, mb.file, mb.label);
    if (!f) { state.SkipWithError("basis not found"); return; }

    Size nbf = f->basis.n_basis_functions();

    // Prepare screening
    f->engine.precompute_schwarz_bounds();
    f->engine.set_density_matrix(f->density.data(), nbf);

    auto opts = screening::ScreeningOptions::normal();

    for (auto _ : state) {
        FockBuilder fock(nbf);
        fock.set_density(f->density.data(), nbf);
        f->engine.compute_and_consume(Operator::coulomb(), fock, opts);
        benchmark::DoNotOptimize(fock.get_coulomb_matrix().data());
    }
    state.SetLabel(std::string(mb.mol) + "/" + mb.label + "/screened");
}

BENCHMARK(BM_FockBuild_Unscreened)->DenseRange(0, 5)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_FockBuild_Screened)->DenseRange(0, 5)->Unit(benchmark::kMicrosecond);

// =============================================================================
// 4. Thread Scaling Benchmarks
// =============================================================================

static void BM_FockBuild_Parallel(benchmark::State& state) {
    auto idx = static_cast<size_t>(state.range(0));
    int n_threads = static_cast<int>(state.range(1));
    if (idx >= std::size(kCombinations)) { state.SkipWithError("bad index"); return; }
    const auto& mb = kCombinations[idx];
    auto* f = get_fixture(mb.mol, mb.file, mb.label);
    if (!f) { state.SkipWithError("basis not found"); return; }

    Size nbf = f->basis.n_basis_functions();

    for (auto _ : state) {
        FockBuilder fock(nbf);
        fock.set_density(f->density.data(), nbf);
        f->engine.compute_and_consume_parallel(Operator::coulomb(), fock,
                                                n_threads);
        benchmark::DoNotOptimize(fock.get_coulomb_matrix().data());
    }
    Size n_shells = f->basis.n_shells();
    state.SetItemsProcessed(state.iterations() *
        static_cast<int64_t>(n_shells * n_shells * n_shells * n_shells));
    state.SetLabel(std::string(mb.mol) + "/" + mb.label +
                   "/threads=" + std::to_string(n_threads));
}

// Register: {mol_basis_index, n_threads}
static void CustomThreadArgs(benchmark::internal::Benchmark* b) {
    for (int idx = 0; idx < 6; ++idx) {
        for (int t : {1, 2, 4, 8}) {
            b->Args({idx, t});
        }
    }
}

BENCHMARK(BM_FockBuild_Parallel)->Apply(CustomThreadArgs)
    ->Unit(benchmark::kMicrosecond);

// =============================================================================
// 5. Combined 1e + 2e Timing Breakdown
// =============================================================================

static void BM_All1e(benchmark::State& state) {
    auto idx = static_cast<size_t>(state.range(0));
    if (idx >= std::size(kCombinations)) { state.SkipWithError("bad index"); return; }
    const auto& mb = kCombinations[idx];
    auto* f = get_fixture(mb.mol, mb.file, mb.label);
    if (!f) { state.SkipWithError("basis not found"); return; }

    auto nuc_op = Operator::nuclear(f->charges);
    std::vector<Real> S, T, V;

    for (auto _ : state) {
        f->engine.compute_1e(Operator::overlap(), S);
        f->engine.compute_1e(Operator::kinetic(), T);
        f->engine.compute_1e(nuc_op, V);
        benchmark::DoNotOptimize(S.data());
        benchmark::DoNotOptimize(T.data());
        benchmark::DoNotOptimize(V.data());
    }
    Size nbf = f->basis.n_basis_functions();
    state.SetItemsProcessed(state.iterations() *
                            static_cast<int64_t>(3 * nbf * nbf));
    state.SetLabel(std::string(mb.mol) + "/" + mb.label + "/all_1e");
}

static void BM_Full1e2e(benchmark::State& state) {
    auto idx = static_cast<size_t>(state.range(0));
    if (idx >= std::size(kCombinations)) { state.SkipWithError("bad index"); return; }
    const auto& mb = kCombinations[idx];
    auto* f = get_fixture(mb.mol, mb.file, mb.label);
    if (!f) { state.SkipWithError("basis not found"); return; }

    auto nuc_op = Operator::nuclear(f->charges);
    Size nbf = f->basis.n_basis_functions();
    std::vector<Real> S, T, V;

    for (auto _ : state) {
        // 1e integrals
        f->engine.compute_1e(Operator::overlap(), S);
        f->engine.compute_1e(Operator::kinetic(), T);
        f->engine.compute_1e(nuc_op, V);

        // 2e Fock build
        FockBuilder fock(nbf);
        fock.set_density(f->density.data(), nbf);
        f->engine.compute_and_consume(Operator::coulomb(), fock);

        benchmark::DoNotOptimize(S.data());
        benchmark::DoNotOptimize(fock.get_coulomb_matrix().data());
    }
    state.SetLabel(std::string(mb.mol) + "/" + mb.label + "/full_1e+2e");
}

BENCHMARK(BM_All1e)->DenseRange(0, 5)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Full1e2e)->DenseRange(0, 5)->Unit(benchmark::kMicrosecond);

}  // namespace libaccint

BENCHMARK_MAIN();
