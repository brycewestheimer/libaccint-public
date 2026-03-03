// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_gpu_integrals.cpp
/// @brief GPU vs CPU performance benchmarks for molecular integrals

#include <benchmark/benchmark.h>
#include <libaccint/config.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/consumers/fock_builder.hpp>

#if LIBACCINT_USE_CUDA
#include <libaccint/engine/cuda_engine.hpp>
#include <libaccint/consumers/fock_builder_gpu.hpp>
#include <cuda_runtime.h>
#endif

#include "bench_helpers.hpp"

#include <vector>

namespace libaccint {

using bench::create_random_density;
using bench::create_h2o_sto3g_shells;
using bench::create_h2o_charges;

// ============================================================================
// CPU Benchmarks
// ============================================================================

static void BM_CPU_Overlap_H2O(benchmark::State& state) {
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
BENCHMARK(BM_CPU_Overlap_H2O);

static void BM_CPU_Kinetic_H2O(benchmark::State& state) {
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
BENCHMARK(BM_CPU_Kinetic_H2O);

static void BM_CPU_Nuclear_H2O(benchmark::State& state) {
    auto shells = create_h2o_sto3g_shells();
    BasisSet basis(shells);
    auto charges = create_h2o_charges();
    Engine engine(basis);
    std::vector<Real> V;

    for (auto _ : state) {
        engine.compute_1e(Operator::nuclear(charges), V);
        benchmark::DoNotOptimize(V.data());
    }

    state.SetItemsProcessed(state.iterations() * basis.n_basis_functions() *
                            basis.n_basis_functions());
}
BENCHMARK(BM_CPU_Nuclear_H2O);

static void BM_CPU_FockBuild_H2O(benchmark::State& state) {
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

    // Count shell quartets processed
    Size n_shells = basis.n_shells();
    Size n_quartets = n_shells * n_shells * n_shells * n_shells;
    state.SetItemsProcessed(state.iterations() * n_quartets);
}
BENCHMARK(BM_CPU_FockBuild_H2O);

// ============================================================================
// GPU Benchmarks
// ============================================================================

#if LIBACCINT_USE_CUDA

static bool cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

static void BM_GPU_Overlap_H2O(benchmark::State& state) {
    if (!cuda_available()) {
        state.SkipWithMessage("No CUDA devices available");
        return;
    }

    auto shells = create_h2o_sto3g_shells();
    BasisSet basis(shells);
    CudaEngine engine(basis);
    std::vector<Real> S;

    for (auto _ : state) {
        engine.compute_overlap_matrix(S);
        engine.synchronize();
        benchmark::DoNotOptimize(S.data());
    }

    state.SetItemsProcessed(state.iterations() * basis.n_basis_functions() *
                            basis.n_basis_functions());
}
BENCHMARK(BM_GPU_Overlap_H2O);

static void BM_GPU_Kinetic_H2O(benchmark::State& state) {
    if (!cuda_available()) {
        state.SkipWithMessage("No CUDA devices available");
        return;
    }

    auto shells = create_h2o_sto3g_shells();
    BasisSet basis(shells);
    CudaEngine engine(basis);
    std::vector<Real> T;

    for (auto _ : state) {
        engine.compute_kinetic_matrix(T);
        engine.synchronize();
        benchmark::DoNotOptimize(T.data());
    }

    state.SetItemsProcessed(state.iterations() * basis.n_basis_functions() *
                            basis.n_basis_functions());
}
BENCHMARK(BM_GPU_Kinetic_H2O);

static void BM_GPU_Nuclear_H2O(benchmark::State& state) {
    if (!cuda_available()) {
        state.SkipWithMessage("No CUDA devices available");
        return;
    }

    auto shells = create_h2o_sto3g_shells();
    BasisSet basis(shells);
    auto charges = create_h2o_charges();
    CudaEngine engine(basis);
    std::vector<Real> V;

    for (auto _ : state) {
        engine.compute_nuclear_matrix(charges, V);
        engine.synchronize();
        benchmark::DoNotOptimize(V.data());
    }

    state.SetItemsProcessed(state.iterations() * basis.n_basis_functions() *
                            basis.n_basis_functions());
}
BENCHMARK(BM_GPU_Nuclear_H2O);

static void BM_GPU_FockBuild_H2O(benchmark::State& state) {
    if (!cuda_available()) {
        state.SkipWithMessage("No CUDA devices available");
        return;
    }

    auto shells = create_h2o_sto3g_shells();
    BasisSet basis(shells);
    const Size nbf = basis.n_basis_functions();
    auto D = create_random_density(nbf);

    CudaEngine engine(basis);
    consumers::GpuFockBuilder fock(nbf, engine.stream());

    for (auto _ : state) {
        fock.reset();
        fock.set_density(D.data(), nbf);
        engine.compute_and_consume_eri(fock);
        fock.synchronize();
        benchmark::DoNotOptimize(fock.d_J());
    }

    Size n_shells = basis.n_shells();
    Size n_quartets = n_shells * n_shells * n_shells * n_shells;
    state.SetItemsProcessed(state.iterations() * n_quartets);
}
BENCHMARK(BM_GPU_FockBuild_H2O);

#endif  // LIBACCINT_USE_CUDA

}  // namespace libaccint

BENCHMARK_MAIN();
