// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_multi_gpu_scaling.cpp
/// @brief Scaling benchmarks for multi-GPU performance

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/consumers/fock_builder_gpu.hpp>
#include <libaccint/consumers/multi_gpu_fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/device/device_manager.hpp>
#include <libaccint/engine/cuda_engine.hpp>
#include <libaccint/engine/multi_gpu_engine.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace libaccint::benchmark {

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

/// @brief Create a BasisSet for a named molecule
BasisSet create_molecule_basis(const std::string& molecule,
                               const std::string& basis_name) {
    std::vector<data::Atom> atoms;
    if (molecule == "water") {
        atoms = {{8, {0.0, 0.0, 0.0}},
                 {1, {1.430429, 0.0, 1.107157}},
                 {1, {-1.430429, 0.0, 1.107157}}};
    } else if (molecule == "benzene") {
        // C6H6 in Bohr (planar, D6h symmetry)
        constexpr double r_cc = 2.640;
        constexpr double r_ch = 4.693;
        for (int i = 0; i < 6; ++i) {
            double angle = i * 3.14159265358979 / 3.0;
            atoms.push_back({6, {r_cc * std::cos(angle), r_cc * std::sin(angle), 0.0}});
            atoms.push_back({1, {r_ch * std::cos(angle), r_ch * std::sin(angle), 0.0}});
        }
    } else {
        throw std::runtime_error("Unknown molecule: " + molecule);
    }
    return data::create_builtin_basis(basis_name, atoms);
}

struct ScalingResult {
    int n_devices;
    double time_ms;
    double speedup;
    double efficiency;
};

class MultiGPUScalingBenchmark {
public:
    explicit MultiGPUScalingBenchmark(int warmup_runs = 2, int benchmark_runs = 5)
        : warmup_runs_(warmup_runs), benchmark_runs_(benchmark_runs) {}
    
    void run_all_benchmarks() {
        auto& dm = device::DeviceManager::instance();
        const int n_devices = dm.device_count();
        
        std::cout << "=== Multi-GPU Scaling Benchmark ===\n";
        std::cout << "Available GPUs: " << n_devices << "\n";
        
        for (int i = 0; i < n_devices; ++i) {
            const auto& props = dm.get_device_properties(i);
            std::cout << "  [" << i << "] " << props.name << "\n";
        }
        std::cout << "\n";
        
        // Run benchmarks for different molecule sizes
        benchmark_molecule("water", "sto-3g");
        benchmark_molecule("water", "6-31g");
        
        if (n_devices >= 2) {
            benchmark_molecule("benzene", "sto-3g");
        }
        
        if (n_devices >= 4) {
            benchmark_molecule("benzene", "6-31g");
        }
    }
    
private:
    void benchmark_molecule(const std::string& molecule, 
                            const std::string& basis_name) {
        std::cout << "Benchmarking: " << molecule << "/" << basis_name << "\n";
        std::cout << std::string(50, '-') << "\n";
        
        auto basis = create_molecule_basis(molecule, basis_name);
        const Size nbf = basis.n_basis_functions();
        
        std::cout << "Basis functions: " << nbf << "\n\n";
        
        // Generate random density
        auto D = random_density(nbf);
        
        auto& dm = device::DeviceManager::instance();
        const int max_devices = dm.device_count();
        
        std::vector<ScalingResult> results;
        
        // Benchmark with increasing device counts
        for (int n = 1; n <= max_devices; ++n) {
            auto result = benchmark_fock_build(basis, D, n);
            results.push_back(result);
        }
        
        // Print results table
        print_results_table(results);
        
        // Print scaling analysis
        print_scaling_analysis(results);
        
        std::cout << "\n";
    }
    
    ScalingResult benchmark_fock_build(const BasisSet& basis,
                                          const std::vector<Real>& D,
                                          int n_devices) {
        const Size nbf = basis.n_basis_functions();
        
        std::vector<int> device_ids;
        for (int i = 0; i < n_devices; ++i) {
            device_ids.push_back(i);
        }
        
        // Warmup runs
        for (int w = 0; w < warmup_runs_; ++w) {
            engine::MultiGPUConfig config;
            config.device_ids = device_ids;
            engine::MultiGPUEngine engine(basis, config);
            
            consumers::MultiGPUFockBuilder fock(nbf, device_ids);
            fock.set_density(D.data(), nbf);
            engine.compute_all_eri(fock);
            fock.get_coulomb_matrix();
        }
        
        // Timed runs
        double total_time = 0.0;
        for (int r = 0; r < benchmark_runs_; ++r) {
            engine::MultiGPUConfig config;
            config.device_ids = device_ids;
            config.collect_stats = true;
            engine::MultiGPUEngine engine(basis, config);
            
            consumers::MultiGPUFockBuilder fock(nbf, device_ids);
            fock.set_density(D.data(), nbf);
            
            auto start = Clock::now();
            engine.compute_all_eri(fock);
            fock.synchronize();
            auto J = fock.get_coulomb_matrix();
            auto end = Clock::now();
            
            total_time += Duration(end - start).count();
        }
        
        double avg_time = total_time / benchmark_runs_;
        
        ScalingResult result;
        result.n_devices = n_devices;
        result.time_ms = avg_time;
        result.speedup = 0.0;  // Set after all results collected
        result.efficiency = 0.0;
        
        return result;
    }
    
    void print_results_table(std::vector<ScalingResult>& results) {
        // Calculate speedups and efficiencies
        double baseline = results[0].time_ms;
        for (auto& r : results) {
            r.speedup = baseline / r.time_ms;
            r.efficiency = r.speedup / r.n_devices * 100.0;
        }
        
        std::cout << std::setw(10) << "GPUs" 
                  << std::setw(15) << "Time (ms)"
                  << std::setw(12) << "Speedup"
                  << std::setw(15) << "Efficiency"
                  << "\n";
        std::cout << std::string(52, '-') << "\n";
        
        for (const auto& r : results) {
            std::cout << std::setw(10) << r.n_devices
                      << std::setw(15) << std::fixed << std::setprecision(2) << r.time_ms
                      << std::setw(12) << std::setprecision(2) << r.speedup << "x"
                      << std::setw(14) << std::setprecision(1) << r.efficiency << "%"
                      << "\n";
        }
    }
    
    void print_scaling_analysis(const std::vector<ScalingResult>& results) {
        std::cout << "\nScaling Analysis:\n";
        
        // Check if we meet the 80% efficiency target
        bool meets_target = true;
        for (const auto& r : results) {
            if (r.n_devices <= 4 && r.efficiency < 80.0) {
                meets_target = false;
                break;
            }
        }
        
        std::cout << "  Quality Gate G15 (>80% efficiency up to 4 GPUs): ";
        if (meets_target) {
            std::cout << "PASS\n";
        } else {
            std::cout << "NEEDS IMPROVEMENT\n";
        }
        
        // Identify bottlenecks
        if (results.size() >= 2) {
            double scaling_drop = results[0].efficiency - results.back().efficiency;
            if (scaling_drop > 20) {
                std::cout << "  Warning: Significant scaling drop detected ("
                          << std::setprecision(1) << scaling_drop << "%)\n";
                std::cout << "  Possible causes: load imbalance, reduction overhead, "
                          << "communication bottleneck\n";
            }
        }
    }
    
    std::vector<Real> random_density(Size nbf, unsigned seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<Real> dist(-1.0, 1.0);
        
        std::vector<Real> D(nbf * nbf, 0.0);
        for (Size i = 0; i < nbf; ++i) {
            for (Size j = i; j < nbf; ++j) {
                Real val = dist(gen);
                D[i * nbf + j] = val;
                D[j * nbf + i] = val;
            }
        }
        return D;
    }
    
    int warmup_runs_;
    int benchmark_runs_;
};

}  // namespace libaccint::benchmark

#endif  // LIBACCINT_USE_CUDA

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    
#if LIBACCINT_USE_CUDA
    try {
        libaccint::benchmark::MultiGPUScalingBenchmark bench;
        bench.run_all_benchmarks();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
#else
    std::cerr << "Multi-GPU benchmarks require CUDA support\n";
    return 1;
#endif
}
