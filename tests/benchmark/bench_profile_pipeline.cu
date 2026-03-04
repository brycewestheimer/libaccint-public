// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_profile_pipeline.cu
/// @brief Profiling-focused benchmark for nsys: sequential vs pipelined ERI
///
/// Runs H2O/aug-cc-pVDZ with NVTX ranges for sequential, device-scatter,
/// and callback paths so nsys can visualize stream overlap.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/engine/cuda_engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/data/basis_parser.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <vector>

using namespace libaccint;

static std::vector<data::Atom> make_h2o() {
    return {
        {8, {0.0, 0.0, 0.0}},
        {1, {1.430429, 0.0, 1.107157}},
        {1, {-1.430429, 0.0, 1.107157}},
    };
}

static std::vector<data::Atom> make_water_tetramer() {
    return {
        {8, {0.000, 0.000, 0.000}},
        {1, {1.430, 0.000, 1.107}},
        {1, {-1.430, 0.000, 1.107}},
        {8, {5.500, 0.000, 0.000}},
        {1, {6.930, 0.000, 1.107}},
        {1, {5.500, 1.430, 1.107}},
        {8, {5.500, 5.500, 0.000}},
        {1, {4.070, 5.500, 1.107}},
        {1, {5.500, 5.500, -1.430}},
        {8, {0.000, 5.500, 0.000}},
        {1, {-1.430, 5.500, -1.107}},
        {1, {0.000, 4.070, -1.107}},
    };
}

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::printf("No CUDA devices\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("GPU: %s (SM %d.%d, %d SMs)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // Use H2O/aug-cc-pVDZ (43 BF, 231 SSQ — enough work to see overlap)
    auto atoms = make_h2o();
    BasisSet basis = data::load_basis_set("aug-cc-pvdz", atoms);
    const Size nbf = basis.n_basis_functions();
    const auto& quartets = basis.shell_set_quartets();
    std::printf("BF: %zu  Shells: %zu  SSQuartets: %zu\n\n",
                nbf, basis.n_shells(), quartets.size());
    std::fflush(stdout);

    CudaEngine engine(basis);

    // Warmup (not profiled)
    {
        std::vector<double> warmup;
        EriPipelineConfig cfg;
        cfg.n_slots = 2;
        engine.compute_eri_pipelined(warmup, cfg);
    }
    cudaDeviceSynchronize();

    // ---- Sequential: one SSQ at a time ----
    nvtxRangePushA("Sequential ERI");
    {
        for (const auto& q : quartets) {
            auto batch = engine.compute_eri_batch_device_handle(q);
            (void)batch;
        }
        engine.synchronize();
    }
    nvtxRangePop();

    cudaDeviceSynchronize();

    // ---- Pipelined host-scatter ----
    nvtxRangePushA("Pipelined Host-Scatter (4 slots)");
    {
        std::vector<double> result;
        EriPipelineConfig cfg;
        cfg.n_slots = 4;
        engine.compute_eri_pipelined(result, cfg);
    }
    nvtxRangePop();

    cudaDeviceSynchronize();

    // ---- Device-scatter ----
    nvtxRangePushA("Device-Scatter (4 streams)");
    {
        std::vector<double> result;
        EriPipelineConfig cfg;
        cfg.n_slots = 4;
        engine.compute_eri_device_scatter(result, cfg);
    }
    nvtxRangePop();

    cudaDeviceSynchronize();

    // ---- Callback mode ----
    nvtxRangePushA("Callback (4 slots)");
    {
        double sum = 0.0;
        EriPipelineConfig cfg;
        cfg.n_slots = 4;
        engine.compute_eri_pipelined(
            [&sum](std::span<const double> chunk, const ShellSetQuartet&) {
                for (double v : chunk) sum += v;
            }, cfg);
    }
    nvtxRangePop();

    cudaDeviceSynchronize();

    // ---- Also profile (H2O)4 callback for the larger system ----
    auto atoms4 = make_water_tetramer();
    BasisSet basis4 = data::load_basis_set("aug-cc-pvdz", atoms4);
    CudaEngine engine4(basis4);
    std::printf("(H2O)4: BF=%zu  SSQ=%zu\n",
                basis4.n_basis_functions(), basis4.shell_set_quartets().size());
    std::fflush(stdout);

    // Warmup
    {
        std::vector<double> w;
        engine4.compute_eri_pipelined(w, {.n_slots = 2});
    }
    cudaDeviceSynchronize();

    nvtxRangePushA("(H2O)4 Sequential");
    {
        for (const auto& q : basis4.shell_set_quartets()) {
            auto batch = engine4.compute_eri_batch_device_handle(q);
            (void)batch;
        }
        engine4.synchronize();
    }
    nvtxRangePop();

    cudaDeviceSynchronize();

    nvtxRangePushA("(H2O)4 Callback (4 slots)");
    {
        double sum = 0.0;
        engine4.compute_eri_pipelined(
            [&sum](std::span<const double> chunk, const ShellSetQuartet&) {
                for (double v : chunk) sum += v;
            }, {.n_slots = 4});
    }
    nvtxRangePop();

    cudaDeviceSynchronize();
    std::printf("Done.\n");
    return 0;
}

#else
#include <cstdio>
int main() { std::printf("CUDA not enabled.\n"); return 0; }
#endif
