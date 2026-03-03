// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_gpu_optimization.cpp
/// @brief GPU optimization profiling benchmarks (Tasks 27.3.1–27.3.3)
///
/// Covers:
/// - CUDA kernel occupancy optimization (27.3.1)
/// - GPU memory transfer minimization (27.3.2)
/// - Warp utilization profiling (27.3.3)
///
/// All GPU benchmarks are skipped at runtime when CUDA is not available.

#include <benchmark/benchmark.h>

#include <libaccint/config.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/consumers/fock_builder.hpp>

#if LIBACCINT_USE_CUDA
#include <libaccint/engine/cuda_engine.hpp>
#include <cuda_runtime.h>
#endif

#include "bench_helpers.hpp"

#include <random>
#include <vector>

namespace libaccint {

using bench::create_random_density;

// ============================================================================
// GPU availability check
// ============================================================================

static bool gpu_available() {
#if LIBACCINT_USE_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

// ============================================================================
// 27.3.1: CUDA Kernel Occupancy — GPU Fock builds at varying system sizes
// ============================================================================

static void BM_GPU_Occupancy_Fock(benchmark::State& state) {
    if (!gpu_available()) {
        state.SkipWithMessage("CUDA GPU not available");
        return;
    }

    const int n_waters = static_cast<int>(state.range(0));
    std::vector<data::Atom> atoms;
    for (int w = 0; w < n_waters; ++w) {
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

    for (auto _ : state) {
        fock.reset();
        fock.set_density(D.data(), nbf);
        engine.compute_and_consume(Operator::coulomb(), fock, BackendHint::PreferGPU);
        benchmark::DoNotOptimize(fock.get_coulomb_matrix().data());
    }

    state.SetLabel(std::to_string(n_waters) + " H2O GPU, nbf=" + std::to_string(nbf));
}
BENCHMARK(BM_GPU_Occupancy_Fock)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

// ============================================================================
// 27.3.2: GPU Memory Transfer — 1e integrals (transfer-dominated)
// ============================================================================

static void BM_GPU_Transfer_1e(benchmark::State& state) {
    if (!gpu_available()) {
        state.SkipWithMessage("CUDA GPU not available");
        return;
    }

    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {1.430429, 0.0, 1.107157}},
        {1, {-1.430429, 0.0, 1.107157}}
    };
    auto basis = data::create_builtin_basis("sto-3g", atoms);

    Engine engine(basis);

    std::vector<Real> S;
    for (auto _ : state) {
        engine.compute_1e(Operator::overlap(), S, BackendHint::PreferGPU);
        benchmark::DoNotOptimize(S.data());
    }

    state.SetLabel("GPU 1e overlap transfer");
}
BENCHMARK(BM_GPU_Transfer_1e);

// ============================================================================
// 27.3.3: Warp Utilization — compare CPU vs GPU for same workload
// ============================================================================

static void BM_GPU_vs_CPU_Fock(benchmark::State& state) {
    const bool use_gpu = (state.range(0) == 1);

    if (use_gpu && !gpu_available()) {
        state.SkipWithMessage("CUDA GPU not available");
        return;
    }

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

    const auto hint = use_gpu ? BackendHint::PreferGPU : BackendHint::ForceCPU;

    for (auto _ : state) {
        fock.reset();
        fock.set_density(D.data(), nbf);
        engine.compute_and_consume(Operator::coulomb(), fock, hint);
        benchmark::DoNotOptimize(fock.get_coulomb_matrix().data());
    }

    state.SetLabel(use_gpu ? "GPU" : "CPU");
}
BENCHMARK(BM_GPU_vs_CPU_Fock)->Arg(0)->Arg(1);  // 0=CPU, 1=GPU

}  // namespace libaccint

BENCHMARK_MAIN();
