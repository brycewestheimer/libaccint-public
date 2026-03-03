// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_cpu_vs_gpu.cpp
/// @brief CPU vs GPU performance comparison benchmarks
///
/// Benchmarks one-electron (overlap, kinetic, nuclear) and two-electron (ERI)
/// integral computation on CPU vs CUDA backends for larger molecular systems
/// with 6-31G and aug-cc-pVDZ basis sets.
///
/// One-electron integrals are computed via full-matrix methods that internally
/// iterate over ShellSetPair batches.
///
/// Two-electron integrals are computed by iterating over the ShellSetQuartet
/// worklist and computing each batch, measuring raw integral computation time
/// without consumer overhead.
///
/// Molecular systems:
///   - (H2O)4  water tetramer cluster
///   - C6H14   hexane (larger alkane)
///   - C4H6    1,3-butadiene (conjugated alkene)

#include <libaccint/config.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/engine/cpu_engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/data/basis_parser.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>

#if LIBACCINT_USE_CUDA
#include <libaccint/engine/cuda_engine.hpp>
#include <cuda_runtime.h>
#endif

#include "bench_helpers.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace libaccint;
using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// Molecule Geometries (positions in Bohr)
// ============================================================================

/// @brief Water tetramer (H2O)4 — cyclic hydrogen-bonded cluster
static std::vector<data::Atom> make_water_tetramer() {
    return {
        // Water 1
        {8, {0.000, 0.000, 0.000}},
        {1, {1.430, 0.000, 1.107}},
        {1, {-1.430, 0.000, 1.107}},
        // Water 2
        {8, {5.500, 0.000, 0.000}},
        {1, {6.930, 0.000, 1.107}},
        {1, {5.500, 1.430, 1.107}},
        // Water 3
        {8, {5.500, 5.500, 0.000}},
        {1, {4.070, 5.500, 1.107}},
        {1, {5.500, 5.500, -1.430}},
        // Water 4
        {8, {0.000, 5.500, 0.000}},
        {1, {-1.430, 5.500, -1.107}},
        {1, {0.000, 4.070, -1.107}},
    };
}

/// @brief Hexane C6H14 — larger alkane
static std::vector<data::Atom> make_hexane() {
    const double cc = 2.9;
    const double ch = 2.06;

    std::vector<data::Atom> atoms;
    double cx[6], cy[6];
    for (int i = 0; i < 6; ++i) {
        cx[i] = i * cc * 0.866;
        cy[i] = (i % 2 == 0) ? 0.0 : cc * 0.5;
        atoms.push_back({6, {cx[i], cy[i], 0.0}});
    }
    atoms.push_back({1, {cx[0] - ch * 0.866, cy[0] - ch * 0.5, ch * 0.3}});
    atoms.push_back({1, {cx[0] - ch * 0.866, cy[0] - ch * 0.5, -ch * 0.3}});
    atoms.push_back({1, {cx[0] - ch * 0.5, cy[0] + ch * 0.866, 0.0}});
    for (int i = 1; i < 5; ++i) {
        double sign = (i % 2 == 0) ? -1.0 : 1.0;
        atoms.push_back({1, {cx[i], cy[i] + sign * ch * 0.866, ch * 0.5}});
        atoms.push_back({1, {cx[i], cy[i] + sign * ch * 0.866, -ch * 0.5}});
    }
    atoms.push_back({1, {cx[5] + ch * 0.866, cy[5] + ch * 0.5, ch * 0.3}});
    atoms.push_back({1, {cx[5] + ch * 0.866, cy[5] + ch * 0.5, -ch * 0.3}});
    atoms.push_back({1, {cx[5] + ch * 0.5, cy[5] - ch * 0.866, 0.0}});
    return atoms;
}

/// @brief 1,3-Butadiene C4H6 — conjugated alkene
static std::vector<data::Atom> make_butadiene() {
    const double ccd = 2.53, ccs = 2.79, ch = 2.06;
    return {
        {6, {0.000, 0.000, 0.000}},
        {6, {ccd, 0.000, 0.000}},
        {6, {ccd + ccs * 0.866, ccs * 0.5, 0.000}},
        {6, {ccd + ccs * 0.866 + ccd, ccs * 0.5, 0.000}},
        {1, {-ch * 0.866, -ch * 0.5, 0.0}},
        {1, {-ch * 0.866, ch * 0.5, 0.0}},
        {1, {ccd * 0.5, -ch * 0.866, 0.0}},
        {1, {ccd + ccs * 0.866 * 0.5, ccs * 0.5 + ch * 0.866, 0.0}},
        {1, {ccd + ccs * 0.866 + ccd + ch * 0.866, ccs * 0.5 - ch * 0.5, 0.0}},
        {1, {ccd + ccs * 0.866 + ccd + ch * 0.866, ccs * 0.5 + ch * 0.5, 0.0}},
    };
}

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

struct BenchmarkResult {
    std::string molecule;
    std::string basis_name;
    int n_atoms;
    int n_basis_functions;
    int n_shells;
    std::string integral_type;
    double cpu_time_ms;
    double gpu_time_ms;
    double speedup;
    int n_iterations;
};

template<typename Func>
double time_benchmark(Func&& f, int n_warmup = 1, int n_iter = 5) {
    for (int i = 0; i < n_warmup; ++i) f();
    auto start = Clock::now();
    for (int i = 0; i < n_iter; ++i) f();
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / n_iter;
}

// Prevent compiler from optimizing away results
template<typename T>
void benchmark_do_not_optimize(T* p) {
    asm volatile("" : : "g"(p) : "memory");
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

std::vector<BenchmarkResult> run_benchmarks_for_system(
    const std::string& mol_name,
    const std::vector<data::Atom>& atoms,
    const std::string& basis_name,
    int n_1e_iter,
    int n_2e_iter) {

    std::vector<BenchmarkResult> results;

    BasisSet basis = data::load_basis_set(basis_name, atoms);
    const Size nbf = basis.n_basis_functions();
    const Size n_shells = basis.n_shells();
    const auto& quartets = basis.shell_set_quartets();
    const auto& pairs = basis.shell_set_pairs();

    std::printf("  System: %s / %s\n", mol_name.c_str(), basis_name.c_str());
    std::printf("    Atoms: %zu  Shells: %zu  BF: %zu\n", atoms.size(), n_shells, nbf);
    std::printf("    ShellSetPairs: %zu  ShellSetQuartets: %zu\n", pairs.size(), quartets.size());
    std::fflush(stdout);

    auto charges = bench::make_nuclear_charges(atoms);
    std::vector<Real> S, T, V;

    // ==== CPU One-Electron (skip if n_2e_iter < 0, or always run for comparison) ====
    double cpu_S = 0, cpu_T = 0, cpu_V = 0, cpu_ERI = 0;

    if (n_2e_iter >= 0) {
        Engine engine(basis);

        std::printf("    [CPU] Overlap...  "); std::fflush(stdout);
        cpu_S = time_benchmark([&]() {
            engine.compute_overlap_matrix(S, BackendHint::ForceCPU);
        }, 1, n_1e_iter);
        std::printf("%.3f ms\n", cpu_S); std::fflush(stdout);

        std::printf("    [CPU] Kinetic...  "); std::fflush(stdout);
        cpu_T = time_benchmark([&]() {
            engine.compute_kinetic_matrix(T, BackendHint::ForceCPU);
        }, 1, n_1e_iter);
        std::printf("%.3f ms\n", cpu_T); std::fflush(stdout);

        std::printf("    [CPU] Nuclear...  "); std::fflush(stdout);
        cpu_V = time_benchmark([&]() {
            engine.compute_nuclear_matrix(charges, V, BackendHint::ForceCPU);
        }, 1, n_1e_iter);
        std::printf("%.3f ms\n", cpu_V); std::fflush(stdout);

        // ==== CPU Two-Electron (ERI via ShellSetQuartet batches) ====

        std::printf("    [CPU] ERI (%zu SSQ batches)...  ", quartets.size()); std::fflush(stdout);
        cpu_ERI = time_benchmark([&]() {
            auto buffers = engine.compute_all_2e(Operator::coulomb(), BackendHint::ForceCPU);
            benchmark_do_not_optimize(buffers.data());
        }, 0, n_2e_iter);
        std::printf("%.3f ms\n", cpu_ERI); std::fflush(stdout);
    } else {
        std::printf("    [CPU] Skipped (GPU-only mode)\n");
    }

    // ==== GPU benchmarks ====

    double cpu_STV_combined = cpu_S + cpu_T + cpu_V;  // Sum of individual CPU 1e times
    double gpu_S = 0, gpu_T = 0, gpu_V = 0, gpu_STV_fused = 0, gpu_ERI = 0;

#if LIBACCINT_USE_CUDA
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        CudaEngine cuda_engine(basis);

        std::printf("    [GPU] Overlap...  "); std::fflush(stdout);
        gpu_S = time_benchmark([&]() {
            cuda_engine.compute_overlap_matrix(S);
            cuda_engine.synchronize();
        }, 1, n_1e_iter);
        std::printf("%.3f ms\n", gpu_S); std::fflush(stdout);

        std::printf("    [GPU] Kinetic...  "); std::fflush(stdout);
        gpu_T = time_benchmark([&]() {
            cuda_engine.compute_kinetic_matrix(T);
            cuda_engine.synchronize();
        }, 1, n_1e_iter);
        std::printf("%.3f ms\n", gpu_T); std::fflush(stdout);

        std::printf("    [GPU] Nuclear...  "); std::fflush(stdout);
        gpu_V = time_benchmark([&]() {
            cuda_engine.compute_nuclear_matrix(charges, V);
            cuda_engine.synchronize();
        }, 1, n_1e_iter);
        std::printf("%.3f ms\n", gpu_V); std::fflush(stdout);

        std::printf("    [GPU] Fused S+T+V...  "); std::fflush(stdout);
        gpu_STV_fused = time_benchmark([&]() {
            cuda_engine.compute_all_1e_fused(charges, S, T, V);
            cuda_engine.synchronize();
        }, 1, n_1e_iter);
        std::printf("%.3f ms  (vs %.3f ms individual = %.2fx)\n",
                     gpu_STV_fused, gpu_S + gpu_T + gpu_V,
                     (gpu_STV_fused > 0) ? (gpu_S + gpu_T + gpu_V) / gpu_STV_fused : 0.0);
        std::fflush(stdout);

        if (n_2e_iter > 0) {
            std::printf("    [GPU] ERI (%zu SSQ batches)...  ", quartets.size()); std::fflush(stdout);
            gpu_ERI = time_benchmark([&]() {
                for (const auto& quartet : quartets) {
                    auto batch = cuda_engine.compute_eri_batch_device_handle(quartet);
                    (void)batch;
                }
                cuda_engine.synchronize();
            }, 0, n_2e_iter);
            std::printf("%.3f ms\n", gpu_ERI); std::fflush(stdout);
        }
    } else {
        std::printf("    GPU: No CUDA device available\n");
    }
#else
    std::printf("    GPU: CUDA not enabled\n");
#endif

    auto record = [&](const std::string& type, double cpu_ms, double gpu_ms, int iters) {
        BenchmarkResult r;
        r.molecule = mol_name;
        r.basis_name = basis_name;
        r.n_atoms = static_cast<int>(atoms.size());
        r.n_basis_functions = static_cast<int>(nbf);
        r.n_shells = static_cast<int>(n_shells);
        r.integral_type = type;
        r.cpu_time_ms = cpu_ms;
        r.gpu_time_ms = gpu_ms;
        r.speedup = (gpu_ms > 0) ? cpu_ms / gpu_ms : 0.0;
        r.n_iterations = iters;
        results.push_back(r);
    };

    record("Overlap (S)", cpu_S, gpu_S, n_1e_iter);
    record("Kinetic (T)", cpu_T, gpu_T, n_1e_iter);
    record("Nuclear (V)", cpu_V, gpu_V, n_1e_iter);
    record("S+T+V indiv", cpu_STV_combined, gpu_S + gpu_T + gpu_V, n_1e_iter);
    record("S+T+V fused", cpu_STV_combined, gpu_STV_fused, n_1e_iter);
    record("ERI (2e)", cpu_ERI, gpu_ERI, n_2e_iter);

    return results;
}

// ============================================================================
// Output
// ============================================================================

void print_results_table(const std::vector<BenchmarkResult>& results) {
    std::printf("\n");
    std::printf("%-14s %-12s %5s %5s %-12s %10s %10s %8s\n",
                "Molecule", "Basis", "Sh", "BF", "Integral",
                "CPU (ms)", "GPU (ms)", "Speedup");
    std::printf("%-14s %-12s %5s %5s %-12s %10s %10s %8s\n",
                "--------------", "------------", "-----", "-----",
                "------------", "----------", "----------", "--------");

    for (const auto& r : results) {
        if (r.gpu_time_ms > 0) {
            std::printf("%-14s %-12s %5d %5d %-12s %10.3f %10.3f %7.2fx\n",
                        r.molecule.c_str(), r.basis_name.c_str(),
                        r.n_shells, r.n_basis_functions,
                        r.integral_type.c_str(),
                        r.cpu_time_ms, r.gpu_time_ms, r.speedup);
        } else {
            std::printf("%-14s %-12s %5d %5d %-12s %10.3f %10s %8s\n",
                        r.molecule.c_str(), r.basis_name.c_str(),
                        r.n_shells, r.n_basis_functions,
                        r.integral_type.c_str(),
                        r.cpu_time_ms, "N/A", "N/A");
        }
    }
}

void write_markdown(const std::vector<BenchmarkResult>& results, const char* filename) {
    FILE* f = std::fopen(filename, "w");
    if (!f) {
        std::fprintf(stderr, "ERROR: Cannot open %s for writing\n", filename);
        return;
    }

    std::fprintf(f, "# LibAccInt CPU vs GPU Benchmark Results\n\n");
    std::fprintf(f, "**Date**: %s\n", __DATE__);
    std::fprintf(f, "**Platform**: Linux (WSL2)\n");

#if LIBACCINT_USE_CUDA
    {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::fprintf(f, "**GPU**: %s (Compute %d.%d, %d SMs, %.0f MHz)\n",
                         prop.name, prop.major, prop.minor,
                         prop.multiProcessorCount,
                         prop.clockRate / 1000.0);
            std::fprintf(f, "**GPU Memory**: %.1f GB\n",
                         prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        }
    }
#endif

    std::fprintf(f, "\n## Methodology\n\n");
    std::fprintf(f, "- **1e integrals**: Full-matrix computation via `compute_overlap_matrix()`, "
                    "`compute_kinetic_matrix()`, `compute_nuclear_matrix()`. "
                    "Internally dispatches through ShellSetPair batches.\n");
    std::fprintf(f, "- **2e integrals (ERI)**: Raw integral computation by iterating the "
                    "ShellSetQuartet worklist. CPU uses `Engine::compute_all_2e()` "
                    "(returns IntegralBuffers). GPU uses "
                    "`CudaEngine::compute_eri_batch_device_handle()` per "
                    "ShellSetQuartet with slot-owned device output.\n");
    std::fprintf(f, "- Times are averages over multiple iterations (after warmup).\n");

    std::fprintf(f, "\n## Results\n\n");

    std::string current_group;
    for (const auto& r : results) {
        std::string group = r.molecule + " / " + r.basis_name;
        if (group != current_group) {
            if (!current_group.empty()) std::fprintf(f, "\n");
            current_group = group;
            std::fprintf(f, "### %s (%d atoms, %d shells, %d basis functions)\n\n",
                         group.c_str(), r.n_atoms, r.n_shells, r.n_basis_functions);
            std::fprintf(f, "| Integral | CPU (ms) | GPU (ms) | Speedup |\n");
            std::fprintf(f, "|----------|----------|----------|--------:|\n");
        }

        if (r.gpu_time_ms > 0) {
            std::fprintf(f, "| %s | %.3f | %.3f | %.2fx |\n",
                         r.integral_type.c_str(), r.cpu_time_ms, r.gpu_time_ms, r.speedup);
        } else {
            std::fprintf(f, "| %s | %.3f | N/A | N/A |\n",
                         r.integral_type.c_str(), r.cpu_time_ms);
        }
    }

    std::fprintf(f, "\n## Notes\n\n");
    std::fprintf(f, "- **GPU ERI dispatch**: Each ShellSetQuartet is dispatched as a separate "
                    "kernel launch. Future optimization will fuse multiple quartets into a single "
                    "kernel launch for better GPU utilization.\n");
    std::fprintf(f, "- **1e GPU overhead**: For small basis sets, GPU kernel launch latency "
                    "dominates. GPU advantage appears with larger basis sets where per-batch "
                    "work justifies the launch cost.\n");
    std::fprintf(f, "- **Basis set scaling**: aug-cc-pVDZ has significantly more primitives and "
                    "functions than 6-31G, increasing both the total work and the per-batch work "
                    "available to saturate GPU parallelism.\n");

    std::fclose(f);
    std::printf("\nResults written to %s\n", filename);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::printf("=== LibAccInt CPU vs GPU Benchmark ===\n\n");

#if LIBACCINT_USE_CUDA
    {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::printf("GPU: %s (Compute %d.%d, %d SMs)\n\n",
                        prop.name, prop.major, prop.minor, prop.multiProcessorCount);
        } else {
            std::printf("GPU: No CUDA devices found\n\n");
        }
    }
#else
    std::printf("GPU: CUDA not enabled\n\n");
#endif

    std::vector<BenchmarkResult> all_results;

    struct MolSystem {
        std::string name;
        std::vector<data::Atom> atoms;
    };

    std::vector<MolSystem> systems = {
        {"(H2O)4",  make_water_tetramer()},
        {"C6H14",   make_hexane()},
        {"C4H6",    make_butadiene()},
    };

    std::vector<std::string> basis_sets = {"6-31g", "aug-cc-pvdz"};

    const int n_1e_iter = 5;
    const int n_2e_iter = 1;   // 2e is expensive, single iteration

    for (const auto& sys : systems) {
        for (const auto& basis_name : basis_sets) {
            std::printf("\n--- %s / %s ---\n", sys.name.c_str(), basis_name.c_str());
            std::fflush(stdout);
            try {
                auto results = run_benchmarks_for_system(
                    sys.name, sys.atoms, basis_name, n_1e_iter, n_2e_iter);
                all_results.insert(all_results.end(), results.begin(), results.end());
            } catch (const std::exception& e) {
                std::printf("  ERROR: %s\n", e.what());
            }
            std::fflush(stdout);
        }
    }

    std::printf("\n=== Summary ===\n");
    print_results_table(all_results);

    write_markdown(all_results, "benchmarks.md");

    return 0;
}
