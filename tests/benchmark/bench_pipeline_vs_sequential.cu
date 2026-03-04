// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_pipeline_vs_sequential.cu
/// @brief Benchmark comparing multi-stream pipelined vs sequential ERI computation
///
/// Builds the full ERI tensor for a molecule using:
///   1. Sequential: one ShellSetQuartet at a time via compute_eri_batch_device_handle()
///   2. Pipelined: multi-stream ring-buffer via compute_eri_pipelined()
///
/// Also verifies correctness by comparing the two tensors element-wise.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/engine/cuda_engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/data/basis_parser.hpp>

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace libaccint;
using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// Molecule Geometries
// ============================================================================

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

// ============================================================================
// Sequential ERI tensor build (one SSQ at a time, download + scatter)
// ============================================================================

static double build_eri_sequential(CudaEngine& engine,
                                   const BasisSet& basis,
                                   std::vector<double>& result) {
    const Size nbf = basis.n_basis_functions();
    const size_t n = static_cast<size_t>(nbf);
    result.assign(n * n * n * n, 0.0);

    const auto& quartets = basis.shell_set_quartets();

    auto start = Clock::now();

    for (const auto& q : quartets) {
        auto batch = engine.compute_eri_batch_device_handle(q);
        if (!batch) continue;

        // Download to host
        std::vector<double> host_buf(batch.size());
        cudaMemcpyAsync(host_buf.data(), batch.data(),
                        batch.size() * sizeof(double), cudaMemcpyDeviceToHost,
                        batch.stream());
        cudaStreamSynchronize(batch.stream());

        // Scatter into result tensor
        const auto& set_a = q.bra_pair().shell_set_a();
        const auto& set_b = q.bra_pair().shell_set_b();
        const auto& set_c = q.ket_pair().shell_set_a();
        const auto& set_d = q.ket_pair().shell_set_b();

        const int na_funcs = n_cartesian(set_a.angular_momentum());
        const int nb_funcs = n_cartesian(set_b.angular_momentum());
        const int nc_funcs = n_cartesian(set_c.angular_momentum());
        const int nd_funcs = n_cartesian(set_d.angular_momentum());
        const size_t funcs_per_quartet = static_cast<size_t>(na_funcs) * nb_funcs * nc_funcs * nd_funcs;

        size_t flat_idx = 0;
        for (Size i = 0; i < set_a.n_shells(); ++i) {
            const auto& shell_a = set_a.shell(i);
            const Index fi = shell_a.function_index();
            for (Size j = 0; j < set_b.n_shells(); ++j) {
                const auto& shell_b = set_b.shell(j);
                const Index fj = shell_b.function_index();
                for (Size k = 0; k < set_c.n_shells(); ++k) {
                    const auto& shell_c = set_c.shell(k);
                    const Index fk = shell_c.function_index();
                    for (Size l = 0; l < set_d.n_shells(); ++l) {
                        const auto& shell_d = set_d.shell(l);
                        const Index fl = shell_d.function_index();

                        for (int a = 0; a < na_funcs; ++a) {
                            for (int b = 0; b < nb_funcs; ++b) {
                                for (int c = 0; c < nc_funcs; ++c) {
                                    for (int d = 0; d < nd_funcs; ++d) {
                                        size_t src = flat_idx +
                                            a * nb_funcs * nc_funcs * nd_funcs +
                                            b * nc_funcs * nd_funcs + c * nd_funcs + d;
                                        size_t dst =
                                            static_cast<size_t>(fi + a) * n * n * n +
                                            static_cast<size_t>(fj + b) * n * n +
                                            static_cast<size_t>(fk + c) * n +
                                            static_cast<size_t>(fl + d);
                                        result[dst] += host_buf[src];
                                    }
                                }
                            }
                        }
                        flat_idx += funcs_per_quartet;
                    }
                }
            }
        }
    }

    engine.synchronize();
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================================
// Pipelined ERI tensor build
// ============================================================================

static double build_eri_pipelined(CudaEngine& engine,
                                  std::vector<double>& result,
                                  size_t n_slots) {
    EriPipelineConfig config;
    config.n_slots = n_slots;

    auto start = Clock::now();
    engine.compute_eri_pipelined(result, config);
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================================
// Device-scatter ERI tensor build (GPU-side scatter, single D2H)
// ============================================================================

static double build_eri_device_scatter(CudaEngine& engine,
                                       std::vector<double>& result,
                                       size_t n_streams) {
    EriPipelineConfig config;
    config.n_slots = n_streams;

    auto start = Clock::now();
    engine.compute_eri_device_scatter(result, config);
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================================
// Callback mode ERI (measures throughput without building full tensor)
// ============================================================================

static double build_eri_callback(CudaEngine& engine,
                                 double& checksum,
                                 size_t n_slots) {
    EriPipelineConfig config;
    config.n_slots = n_slots;

    double sum = 0.0;
    size_t n_elements = 0;

    auto start = Clock::now();
    engine.compute_eri_pipelined(
        [&sum, &n_elements](std::span<const double> chunk, const ShellSetQuartet&) {
            for (double v : chunk) sum += v;
            n_elements += chunk.size();
        },
        config);
    auto end = Clock::now();

    checksum = sum;
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================================
// Verification
// ============================================================================

static bool verify_tensors(const std::vector<double>& seq,
                           const std::vector<double>& pipe,
                           double tol = 1e-10) {
    if (seq.size() != pipe.size()) {
        std::printf("  ERROR: size mismatch: %zu vs %zu\n", seq.size(), pipe.size());
        return false;
    }

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    size_t n_nonzero = 0;
    size_t n_mismatch = 0;

    for (size_t i = 0; i < seq.size(); ++i) {
        double diff = std::abs(seq[i] - pipe[i]);
        if (diff > max_abs_err) max_abs_err = diff;

        if (std::abs(seq[i]) > 1e-15) {
            n_nonzero++;
            double rel = diff / std::abs(seq[i]);
            if (rel > max_rel_err) max_rel_err = rel;
            if (rel > tol) n_mismatch++;
        }
    }

    std::printf("  Tensor size: %zu elements, %zu nonzero\n", seq.size(), n_nonzero);
    std::printf("  Max absolute error: %.3e\n", max_abs_err);
    std::printf("  Max relative error: %.3e\n", max_rel_err);
    if (n_mismatch > 0)
        std::printf("  WARNING: %zu elements exceed tol %.1e\n", n_mismatch, tol);
    else
        std::printf("  PASS: all elements within tol %.1e\n", tol);

    return n_mismatch == 0;
}

// ============================================================================
// Benchmark Runner
// ============================================================================

static void run_benchmark(const std::string& name,
                          const std::vector<data::Atom>& atoms,
                          const std::string& basis_name) {
    std::printf("\n==== %s / %s ====\n", name.c_str(), basis_name.c_str());
    std::fflush(stdout);

    BasisSet basis = data::load_basis_set(basis_name, atoms);
    const Size nbf = basis.n_basis_functions();
    const Size n_shells = basis.n_shells();
    const auto& quartets = basis.shell_set_quartets();

    std::printf("  Atoms: %zu  Shells: %zu  BF: %zu  SSQuartets: %zu\n",
                atoms.size(), n_shells, nbf, quartets.size());
    std::printf("  ERI tensor: %zu elements (%.1f MB)\n",
                static_cast<size_t>(nbf) * nbf * nbf * nbf,
                static_cast<size_t>(nbf) * nbf * nbf * nbf * sizeof(double) / 1e6);
    std::fflush(stdout);

    CudaEngine engine(basis);

    // Warmup
    {
        std::vector<double> warmup;
        EriPipelineConfig cfg;
        cfg.n_slots = 2;
        engine.compute_eri_pipelined(warmup, cfg);
    }

    // Sequential
    std::printf("\n  [Sequential] Building ERI tensor...\n");
    std::fflush(stdout);
    std::vector<double> eri_seq;
    double t_seq = build_eri_sequential(engine, basis, eri_seq);
    std::printf("  [Sequential] %.3f ms\n", t_seq);
    std::fflush(stdout);

    // Pipelined with host scatter
    for (size_t n_slots : {4}) {
        std::printf("\n  [Pipelined host-scatter n_slots=%zu] Building ERI tensor...\n", n_slots);
        std::fflush(stdout);
        std::vector<double> eri_pipe;
        double t_pipe = build_eri_pipelined(engine, eri_pipe, n_slots);
        std::printf("  [Pipelined host-scatter n_slots=%zu] %.3f ms  (%.3fx vs sequential)\n",
                    n_slots, t_pipe, t_seq / t_pipe);
        verify_tensors(eri_seq, eri_pipe);
        std::fflush(stdout);
    }

    // Device-scatter (GPU-side scatter, single bulk D2H)
    for (size_t n_streams : {4}) {
        std::printf("\n  [Device-scatter n_streams=%zu] Building ERI tensor...\n", n_streams);
        std::fflush(stdout);
        std::vector<double> eri_dev;
        double t_dev = build_eri_device_scatter(engine, eri_dev, n_streams);
        std::printf("  [Device-scatter n_streams=%zu] %.3f ms  (%.3fx vs sequential)\n",
                    n_streams, t_dev, t_seq / t_dev);
        verify_tensors(eri_seq, eri_dev);
        std::fflush(stdout);
    }

    // Callback mode (no tensor build, measures pure throughput)
    {
        std::printf("\n  [Callback n_slots=4] Computing ERIs...\n");
        std::fflush(stdout);
        double checksum = 0.0;
        double t_cb = build_eri_callback(engine, checksum, 4);
        std::printf("  [Callback n_slots=4] %.3f ms  (%.3fx vs sequential)  checksum=%.6e\n",
                    t_cb, t_seq / t_cb, checksum);
        std::fflush(stdout);
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::printf("No CUDA devices available\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("GPU: %s (SM %d.%d, %d SMs)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    std::fflush(stdout);

    // Small system: H2O / STO-3G (7 BF, fast sanity check)
    run_benchmark("H2O", make_h2o(), "sto-3g");

    // Medium system with d functions: H2O / aug-cc-pVDZ
    run_benchmark("H2O", make_h2o(), "aug-cc-pvdz");

    // Larger system with d functions: (H2O)4 / aug-cc-pVDZ
    run_benchmark("(H2O)4", make_water_tetramer(), "aug-cc-pvdz");

    std::printf("\nDone.\n");
    return 0;
}

#else  // LIBACCINT_USE_CUDA

#include <cstdio>
int main() {
    std::printf("CUDA not enabled. Skipping benchmark.\n");
    return 0;
}

#endif
