// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_water_cluster_scaling.cpp
/// @brief Water cluster scaling benchmark: CPU vs GPU
///
/// Benchmarks one-electron (S, T, V) and two-electron (ERI, Fock) integral
/// computation for (H2O)_N clusters (N=1,2,4,8) with aug-cc-pVTZ basis set.
/// Standalone benchmark with no Google Benchmark dependency.

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

#if LIBACCINT_USE_CUDA
#include <libaccint/engine/cuda_engine.hpp>
#include <cuda_runtime.h>
#endif

#include "bench_helpers.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace libaccint;
using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// Water Cluster Geometry (must match Python generator and validation test)
// ============================================================================

static std::vector<data::Atom> make_water_cluster(int n) {
    const double r_oh = 1.8088;
    const double theta_hoh = 104.52 * M_PI / 180.0;
    const double half_theta = theta_hoh / 2.0;

    std::vector<data::Atom> atoms;

    if (n == 1) {
        double hy = r_oh * std::sin(half_theta);
        double hz = -r_oh * std::cos(half_theta);
        atoms.push_back({8, {0.0, 0.0, 0.0}});
        atoms.push_back({1, {0.0, hy, hz}});
        atoms.push_back({1, {0.0, -hy, hz}});
        return atoms;
    }

    const double oo_dist = 5.4;
    const double R = oo_dist / (2.0 * std::sin(M_PI / n));

    for (int i = 0; i < n; ++i) {
        double angle_o = 2.0 * M_PI * i / n;
        double ox = R * std::cos(angle_o);
        double oy = R * std::sin(angle_o);
        double oz = 0.0;
        atoms.push_back({8, {ox, oy, oz}});

        int next_i = (i + 1) % n;
        double angle_next = 2.0 * M_PI * next_i / n;
        double nx = R * std::cos(angle_next) - ox;
        double ny = R * std::sin(angle_next) - oy;
        double nlen = std::sqrt(nx * nx + ny * ny);
        double dx = nx / nlen;
        double dy = ny / nlen;

        double px = -dy;
        double py = dx;

        double h1x = ox + r_oh * std::cos(half_theta) * dx;
        double h1y = oy + r_oh * std::cos(half_theta) * dy;
        double h1z = oz - r_oh * std::sin(half_theta);
        atoms.push_back({1, {h1x, h1y, h1z}});

        double h2x = ox + r_oh * std::cos(half_theta) * px;
        double h2y = oy + r_oh * std::cos(half_theta) * py;
        double h2z = oz + r_oh * std::sin(half_theta);
        atoms.push_back({1, {h2x, h2y, h2z}});
    }

    return atoms;
}

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

struct BenchmarkResult {
    std::string system;
    int n_waters;
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

template<typename T>
void benchmark_do_not_optimize(T* p) {
    asm volatile("" : : "g"(p) : "memory");
}

// ============================================================================
// Per-System Benchmark
// ============================================================================

std::vector<BenchmarkResult> run_benchmarks(int n_waters, int n_1e_iter, int n_2e_iter) {
    std::vector<BenchmarkResult> results;

    auto atoms = make_water_cluster(n_waters);
    std::string system_name = "(H2O)_" + std::to_string(n_waters);

    BasisSet basis = data::load_basis_set("aug-cc-pvtz", atoms);
    const Size nbf = basis.n_basis_functions();
    const Size n_shells = basis.n_shells();
    const auto& quartets = basis.shell_set_quartets();

    std::printf("  System: %s / aug-cc-pVTZ\n", system_name.c_str());
    std::printf("    Atoms: %zu  Shells: %zu  BF: %zu\n", atoms.size(), n_shells, nbf);
    std::printf("    ShellSetQuartets: %zu\n", quartets.size());
    std::printf("    1e iters: %d  2e iters: %d\n", n_1e_iter, n_2e_iter);
    std::fflush(stdout);

    auto charges = bench::make_nuclear_charges(atoms);
    auto D = bench::create_random_density(nbf);

    std::vector<Real> S, T, V;
    double cpu_S = 0, cpu_T = 0, cpu_V = 0, cpu_ERI = 0, cpu_Fock = 0;

    Engine engine(basis);

    // ---- CPU 1e ----
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

    // ---- CPU ERI ----
    if (n_2e_iter > 0) {
        std::printf("    [CPU] ERI (%zu SSQ)...  ", quartets.size()); std::fflush(stdout);
        cpu_ERI = time_benchmark([&]() {
            auto buffers = engine.compute_all_2e(Operator::coulomb(), BackendHint::ForceCPU);
            benchmark_do_not_optimize(buffers.data());
        }, 0, n_2e_iter);
        std::printf("%.3f ms\n", cpu_ERI); std::fflush(stdout);

        // ---- CPU Fock ----
        std::printf("    [CPU] Fock (J+K)...  "); std::fflush(stdout);
        cpu_Fock = time_benchmark([&]() {
            consumers::FockBuilder fock(nbf);
            fock.set_density(D.data(), nbf);
            engine.compute_and_consume(Operator::coulomb(), fock);
            benchmark_do_not_optimize(const_cast<Real*>(fock.get_coulomb_matrix().data()));
        }, 0, n_2e_iter);
        std::printf("%.3f ms\n", cpu_Fock); std::fflush(stdout);
    }

    // ---- GPU ----
    double gpu_S = 0, gpu_T = 0, gpu_V = 0, gpu_STV_fused = 0, gpu_ERI = 0, gpu_Fock = 0;

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
        std::printf("%.3f ms\n", gpu_STV_fused); std::fflush(stdout);

        if (n_2e_iter > 0) {
            std::printf("    [GPU] ERI (%zu SSQ)...  ", quartets.size()); std::fflush(stdout);
            gpu_ERI = time_benchmark([&]() {
                for (const auto& quartet : quartets) {
                    auto batch = cuda_engine.compute_eri_batch_device_handle(quartet);
                    (void)batch;
                }
                cuda_engine.synchronize();
            }, 0, n_2e_iter);
            std::printf("%.3f ms\n", gpu_ERI); std::fflush(stdout);

            std::printf("    [GPU] Fock (J+K via Engine)...  "); std::fflush(stdout);
            gpu_Fock = time_benchmark([&]() {
                consumers::FockBuilder fock(nbf);
                fock.set_density(D.data(), nbf);
                engine.compute_and_consume(Operator::coulomb(), fock);
                benchmark_do_not_optimize(const_cast<Real*>(fock.get_coulomb_matrix().data()));
            }, 0, n_2e_iter);
            std::printf("%.3f ms\n", gpu_Fock); std::fflush(stdout);
        }
    } else {
        std::printf("    GPU: No CUDA device available\n");
    }
#else
    std::printf("    GPU: CUDA not enabled\n");
#endif

    double cpu_STV = cpu_S + cpu_T + cpu_V;

    auto record = [&](const std::string& type, double cpu, double gpu, int iters) {
        results.push_back({system_name, n_waters,
            static_cast<int>(atoms.size()), static_cast<int>(nbf),
            static_cast<int>(n_shells), type, cpu, gpu,
            (gpu > 0) ? cpu / gpu : 0.0, iters});
    };

    record("Overlap (S)", cpu_S, gpu_S, n_1e_iter);
    record("Kinetic (T)", cpu_T, gpu_T, n_1e_iter);
    record("Nuclear (V)", cpu_V, gpu_V, n_1e_iter);
    record("S+T+V indiv", cpu_STV, gpu_S + gpu_T + gpu_V, n_1e_iter);
    record("S+T+V fused", cpu_STV, gpu_STV_fused, n_1e_iter);
    if (n_2e_iter > 0) {
        record("ERI (raw)", cpu_ERI, gpu_ERI, n_2e_iter);
        record("Fock (J+K)", cpu_Fock, gpu_Fock, n_2e_iter);
    }

    return results;
}

// ============================================================================
// Output
// ============================================================================

void print_results_table(const std::vector<BenchmarkResult>& results) {
    std::printf("\n");
    std::printf("%-12s %5s %5s %-14s %10s %10s %8s\n",
                "System", "Sh", "BF", "Integral",
                "CPU (ms)", "GPU (ms)", "Speedup");
    std::printf("%-12s %5s %5s %-14s %10s %10s %8s\n",
                "------------", "-----", "-----",
                "--------------", "----------", "----------", "--------");

    for (const auto& r : results) {
        if (r.gpu_time_ms > 0) {
            std::printf("%-12s %5d %5d %-14s %10.3f %10.3f %7.2fx\n",
                        r.system.c_str(), r.n_shells, r.n_basis_functions,
                        r.integral_type.c_str(),
                        r.cpu_time_ms, r.gpu_time_ms, r.speedup);
        } else {
            std::printf("%-12s %5d %5d %-14s %10.3f %10s %8s\n",
                        r.system.c_str(), r.n_shells, r.n_basis_functions,
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

    std::fprintf(f, "# Water Cluster Scaling Benchmark\n\n");
    std::fprintf(f, "**Basis set**: aug-cc-pVTZ (Cartesian)\n");
    std::fprintf(f, "**Date**: %s\n", __DATE__);

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
    std::fprintf(f, "- **Systems**: (H2O)_N for N = 1, 2, 4, 8 with aug-cc-pVTZ\n");
    std::fprintf(f, "- **1e integrals**: Full-matrix via `compute_overlap_matrix()`, "
                    "`compute_kinetic_matrix()`, `compute_nuclear_matrix()`\n");
    std::fprintf(f, "- **2e integrals**: `compute_all_2e()` (CPU), "
                    "per-SSQ `compute_eri_batch_device_handle()` (GPU)\n");
    std::fprintf(f, "- **Fock build**: `compute_and_consume()` with FockBuilder\n");
    std::fprintf(f, "- Times are averages over multiple iterations (after warmup)\n");

    std::fprintf(f, "\n## Results\n\n");

    std::string current_sys;
    for (const auto& r : results) {
        if (r.system != current_sys) {
            if (!current_sys.empty()) std::fprintf(f, "\n");
            current_sys = r.system;
            std::fprintf(f, "### %s (%d atoms, %d shells, %d BF)\n\n",
                         r.system.c_str(), r.n_atoms, r.n_shells, r.n_basis_functions);
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

    // Scaling summary
    std::fprintf(f, "\n## Scaling Summary\n\n");
    std::fprintf(f, "| N | BF | CPU S+T+V (ms) | GPU S+T+V fused (ms) | CPU ERI (ms) | GPU ERI (ms) |\n");
    std::fprintf(f, "|---|----|---------|---------|---------|---------|\n");

    for (const auto& r : results) {
        if (r.integral_type == "S+T+V fused") {
            // Find matching ERI
            double cpu_eri = 0, gpu_eri = 0;
            for (const auto& r2 : results) {
                if (r2.n_waters == r.n_waters && r2.integral_type == "ERI (raw)") {
                    cpu_eri = r2.cpu_time_ms;
                    gpu_eri = r2.gpu_time_ms;
                    break;
                }
            }
            std::fprintf(f, "| %d | %d | %.3f | %.3f | %.3f | %.3f |\n",
                         r.n_waters, r.n_basis_functions,
                         r.cpu_time_ms, r.gpu_time_ms,
                         cpu_eri, gpu_eri);
        }
    }

    std::fclose(f);
    std::printf("\nResults written to %s\n", filename);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::printf("=== Water Cluster Scaling Benchmark ===\n");
    std::printf("    Basis: aug-cc-pVTZ (Cartesian)\n\n");

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

    struct ClusterConfig {
        int n_waters;
        int n_1e_iter;
        int n_2e_iter;
    };

    std::vector<ClusterConfig> configs = {
        {1, 10, 5},
        {2,  5, 3},
        {4,  3, 1},
        {8,  1, 1},
    };

    std::vector<BenchmarkResult> all_results;

    for (const auto& cfg : configs) {
        std::printf("\n--- (H2O)_%d ---\n", cfg.n_waters);
        std::fflush(stdout);
        try {
            auto results = run_benchmarks(cfg.n_waters, cfg.n_1e_iter, cfg.n_2e_iter);
            all_results.insert(all_results.end(), results.begin(), results.end());
        } catch (const std::exception& e) {
            std::printf("  ERROR: %s\n", e.what());
        }
        std::fflush(stdout);
    }

    std::printf("\n=== Summary ===\n");
    print_results_table(all_results);

    write_markdown(all_results, "water_cluster_benchmark.md");

    return 0;
}
