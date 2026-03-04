// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_optimal_dispatch.cpp
/// @brief Comprehensive benchmark of all kernel variants for optimal dispatch table generation
///
/// Benchmarks every available kernel variant (handwritten, generated, cooperative, fused)
/// for each AM combination and outputs a machine-parseable JSON dispatch table that can
/// be loaded by the engine at runtime via LIBACCINT_DISPATCH_TABLE environment variable.
///
/// Usage:
///   bench_optimal_dispatch [--output <path.json>]

#include <libaccint/config.hpp>

#if !LIBACCINT_USE_CUDA
#include <cstdio>
int main() {
    std::printf("CUDA not enabled. Skipping benchmark.\n");
    return 0;
}
#else

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/device_data.hpp>
#include <libaccint/data/basis_parser.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/operators/device_operator_data.hpp>
#include <libaccint/memory/device_memory.hpp>
#include <libaccint/kernels/overlap_kernel_cuda.hpp>
#include <libaccint/kernels/kinetic_kernel_cuda.hpp>
#include <libaccint/kernels/nuclear_kernel_cuda.hpp>
#include <libaccint/kernels/eri_kernel_cuda.hpp>
#include <libaccint/kernels/fused_1e_kernel_cuda.hpp>
#include <libaccint/kernels/generated_kernel_registry_cuda.hpp>
#include <libaccint/kernels/kernel_variant.hpp>
#include <libaccint/core/types.hpp>

#include <nlohmann/json.hpp>
#include <cuda_runtime.h>

// Boys function device initialization
namespace libaccint::device::math {
    double* boys_device_init(cudaStream_t stream);
    double* boys_device_get_coeffs();
    bool boys_device_is_initialized();
    void boys_device_cleanup();
}

#include "bench_helpers.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace libaccint;
using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// Timing helper (same as existing benchmarks)
// ============================================================================

template<typename Func>
double time_kernel_us(Func&& f, cudaStream_t stream, int n_warmup = 3, int n_iter = 20) {
    for (int i = 0; i < n_warmup; ++i) f();
    cudaStreamSynchronize(stream);

    auto start = Clock::now();
    for (int i = 0; i < n_iter; ++i) f();
    cudaStreamSynchronize(stream);
    auto end = Clock::now();

    return std::chrono::duration<double, std::micro>(end - start).count() / n_iter;
}

static const char* am_char(int l) {
    static const char* names[] = {"s", "p", "d", "f", "g", "h"};
    return (l >= 0 && l <= 5) ? names[l] : "?";
}

// ============================================================================
// Result structures
// ============================================================================

struct OneElectronResult {
    int la, lb;
    int n_pairs;
    double hw_overlap_us, gen_overlap_us;
    double hw_kinetic_us, gen_kinetic_us;
    double hw_nuclear_us, gen_nuclear_us;
    double fused_us;
    // Derived
    std::string best_overlap, best_kinetic, best_nuclear;
    bool prefer_fused;
};

struct TwoElectronResult {
    int la, lb, lc, ld;
    int n_quartets;
    double hw_us;
    double gen_us;
    double coop_us;          // -1 if not available
    std::string best_variant;
};

// ============================================================================
// 1e benchmarks
// ============================================================================

std::vector<OneElectronResult> benchmark_1e(
    const BasisSet& basis,
    const std::vector<data::Atom>& atoms,
    cudaStream_t stream,
    const double* d_boys_coeffs) {

    std::vector<OneElectronResult> results;

    auto charges_params = bench::make_nuclear_charges(atoms);
    auto charge_data = operators::upload_point_charges(
        charges_params.x, charges_params.y, charges_params.z, charges_params.charge);

    const auto& pairs = basis.shell_set_pairs();
    const int max_gen_am = kernels::cuda::generated::GENERATED_MAX_AM;

    // Group by AM pair
    std::map<std::pair<int,int>, std::vector<const ShellSetPair*>> am_groups;
    for (const auto& pair : pairs) {
        int la = pair.shell_set_a().angular_momentum();
        int lb = pair.shell_set_b().angular_momentum();
        am_groups[{la, lb}].push_back(&pair);
    }

    for (const auto& [am_pair, pair_list] : am_groups) {
        auto [la, lb] = am_pair;

        // Pick the largest ShellSetPair for GPU occupancy
        const ShellSetPair* best = nullptr;
        int best_count = 0;
        for (const auto* p : pair_list) {
            int count = p->shell_set_a().n_shells() * p->shell_set_b().n_shells();
            if (count > best_count) {
                best = p;
                best_count = count;
            }
        }
        if (!best || best_count == 0) continue;

        // Upload shell data
        auto bra_data = basis::upload_shell_set(best->shell_set_a(), stream);
        auto ket_data = basis::upload_shell_set(best->shell_set_b(), stream);
        basis::ShellSetPairDeviceData pair_device;
        pair_device.bra = bra_data;
        pair_device.ket = ket_data;

        size_t output_size = kernels::cuda::overlap_output_size(pair_device);
        memory::DeviceBuffer<double> d_output(output_size);

        // Allocate fused output (3x for S+T+V)
        size_t fused_size = kernels::cuda::fused_1e_output_size(pair_device);
        memory::DeviceBuffer<double> d_fused(fused_size * 3);

        OneElectronResult r{};
        r.la = la;
        r.lb = lb;
        r.n_pairs = best_count;

        // --- Overlap ---
        r.hw_overlap_us = time_kernel_us([&]() {
            kernels::cuda::dispatch_overlap_kernel(pair_device, d_output.data(), stream);
        }, stream);

        if (la <= max_gen_am && lb <= max_gen_am &&
            kernels::cuda::generated::has_generated_overlap(la, lb)) {
            r.gen_overlap_us = time_kernel_us([&]() {
                kernels::cuda::generated::launch_generated_overlap(
                    pair_device, d_output.data(), stream);
            }, stream);
        } else {
            r.gen_overlap_us = -1.0;
        }

        // --- Kinetic ---
        r.hw_kinetic_us = time_kernel_us([&]() {
            kernels::cuda::dispatch_kinetic_kernel(pair_device, d_output.data(), stream);
        }, stream);

        if (la <= max_gen_am && lb <= max_gen_am &&
            kernels::cuda::generated::has_generated_kinetic(la, lb)) {
            r.gen_kinetic_us = time_kernel_us([&]() {
                kernels::cuda::generated::launch_generated_kinetic(
                    pair_device, d_output.data(), stream);
            }, stream);
        } else {
            r.gen_kinetic_us = -1.0;
        }

        // --- Nuclear ---
        r.hw_nuclear_us = time_kernel_us([&]() {
            kernels::cuda::dispatch_nuclear_kernel(
                pair_device, charge_data, d_boys_coeffs, d_output.data(), stream);
        }, stream);

        if (la <= max_gen_am && lb <= max_gen_am &&
            kernels::cuda::generated::has_generated_nuclear(la, lb)) {
            r.gen_nuclear_us = time_kernel_us([&]() {
                kernels::cuda::generated::launch_generated_nuclear(
                    pair_device, charge_data, d_boys_coeffs, d_output.data(), stream);
            }, stream);
        } else {
            r.gen_nuclear_us = -1.0;
        }

        // --- Fused S+T+V ---
        kernels::cuda::Fused1eOutputPointers fused_ptrs;
        fused_ptrs.d_overlap = d_fused.data();
        fused_ptrs.d_kinetic = d_fused.data() + fused_size;
        fused_ptrs.d_nuclear = d_fused.data() + 2 * fused_size;

        r.fused_us = time_kernel_us([&]() {
            kernels::cuda::dispatch_fused_1e_kernel(
                pair_device, charge_data, d_boys_coeffs, fused_ptrs, stream);
        }, stream);

        // Determine best variants
        r.best_overlap = (r.gen_overlap_us > 0 && r.gen_overlap_us < r.hw_overlap_us)
            ? "generated" : "handwritten";
        r.best_kinetic = (r.gen_kinetic_us > 0 && r.gen_kinetic_us < r.hw_kinetic_us)
            ? "generated" : "handwritten";
        r.best_nuclear = (r.gen_nuclear_us > 0 && r.gen_nuclear_us < r.hw_nuclear_us)
            ? "generated" : "handwritten";

        // Determine if fused beats sum of 3 best individual kernels
        double best_s = (r.gen_overlap_us > 0) ? std::min(r.hw_overlap_us, r.gen_overlap_us) : r.hw_overlap_us;
        double best_t = (r.gen_kinetic_us > 0) ? std::min(r.hw_kinetic_us, r.gen_kinetic_us) : r.hw_kinetic_us;
        double best_v = (r.gen_nuclear_us > 0) ? std::min(r.hw_nuclear_us, r.gen_nuclear_us) : r.hw_nuclear_us;
        r.prefer_fused = (r.fused_us < best_s + best_t + best_v);

        results.push_back(r);

        basis::free_shell_set_device_data(bra_data);
        basis::free_shell_set_device_data(ket_data);
    }

    operators::free_point_charge_device_data(charge_data);
    return results;
}

// ============================================================================
// 2e benchmarks
// ============================================================================

std::vector<TwoElectronResult> benchmark_2e(
    const BasisSet& basis,
    cudaStream_t stream,
    const double* d_boys_coeffs) {

    std::vector<TwoElectronResult> results;

    const auto& quartets = basis.shell_set_quartets();
    const int max_gen_am = kernels::cuda::generated::GENERATED_MAX_AM;

    // Group by AM quartet
    auto key_to_int = [](int a, int b, int c, int d) {
        return ((a * 10 + b) * 10 + c) * 10 + d;
    };
    std::map<int, std::vector<const ShellSetQuartet*>> am_groups;

    for (const auto& q : quartets) {
        int la = q.bra_pair().shell_set_a().angular_momentum();
        int lb = q.bra_pair().shell_set_b().angular_momentum();
        int lc = q.ket_pair().shell_set_a().angular_momentum();
        int ld = q.ket_pair().shell_set_b().angular_momentum();
        am_groups[key_to_int(la, lb, lc, ld)].push_back(&q);
    }

    for (const auto& [key, q_list] : am_groups) {
        int la = key / 1000;
        int lb = (key / 100) % 10;
        int lc = (key / 10) % 10;
        int ld = key % 10;

        // Pick the largest quartet batch
        const ShellSetQuartet* best = nullptr;
        size_t best_count = 0;
        for (const auto* q : q_list) {
            size_t count = static_cast<size_t>(q->bra_pair().shell_set_a().n_shells()) *
                           q->bra_pair().shell_set_b().n_shells() *
                           q->ket_pair().shell_set_a().n_shells() *
                           q->ket_pair().shell_set_b().n_shells();
            if (count > best_count) {
                best = q;
                best_count = count;
            }
        }
        if (!best || best_count == 0) continue;

        // Upload shell data
        auto data_a = basis::upload_shell_set(best->bra_pair().shell_set_a(), stream);
        auto data_b = basis::upload_shell_set(best->bra_pair().shell_set_b(), stream);
        auto data_c = basis::upload_shell_set(best->ket_pair().shell_set_a(), stream);
        auto data_d = basis::upload_shell_set(best->ket_pair().shell_set_b(), stream);

        basis::ShellSetQuartetDeviceData quartet_device;
        quartet_device.a = data_a;
        quartet_device.b = data_b;
        quartet_device.c = data_c;
        quartet_device.d = data_d;

        size_t output_size = kernels::cuda::eri_output_size(quartet_device);
        memory::DeviceBuffer<double> d_output(output_size);

        TwoElectronResult r{};
        r.la = la; r.lb = lb; r.lc = lc; r.ld = ld;
        r.n_quartets = static_cast<int>(best_count);

        // Handwritten
        r.hw_us = time_kernel_us([&]() {
            kernels::cuda::dispatch_eri_kernel(
                quartet_device, d_boys_coeffs, d_output.data(), stream);
        }, stream);

        // Generated
        if (la <= max_gen_am && lb <= max_gen_am &&
            lc <= max_gen_am && ld <= max_gen_am &&
            kernels::cuda::generated::has_generated_eri(la, lb, lc, ld)) {
            r.gen_us = time_kernel_us([&]() {
                kernels::cuda::generated::launch_generated_eri(
                    quartet_device, d_boys_coeffs, d_output.data(), stream);
            }, stream);
        } else {
            r.gen_us = -1.0;
        }

        // Cooperative (block-per-quartet)
        if (la <= max_gen_am && lb <= max_gen_am &&
            lc <= max_gen_am && ld <= max_gen_am &&
            kernels::cuda::generated::has_generated_eri_cooperative(la, lb, lc, ld)) {
            r.coop_us = time_kernel_us([&]() {
                kernels::cuda::generated::launch_generated_eri_cooperative(
                    quartet_device, d_boys_coeffs, d_output.data(), stream);
            }, stream);
        } else {
            r.coop_us = -1.0;
        }

        // Determine best variant
        double best_time = r.hw_us;
        r.best_variant = "handwritten";

        if (r.gen_us > 0 && r.gen_us < best_time) {
            best_time = r.gen_us;
            r.best_variant = "generated";
        }
        if (r.coop_us > 0 && r.coop_us < best_time) {
            r.best_variant = "cooperative";
        }

        results.push_back(r);

        basis::free_shell_set_device_data(data_a);
        basis::free_shell_set_device_data(data_b);
        basis::free_shell_set_device_data(data_c);
        basis::free_shell_set_device_data(data_d);
    }

    return results;
}

// ============================================================================
// JSON output
// ============================================================================

void write_dispatch_json(
    const std::string& path,
    const std::string& gpu_name,
    int compute_major, int compute_minor,
    int max_am,
    const std::vector<OneElectronResult>& results_1e,
    const std::vector<TwoElectronResult>& results_2e) {

    nlohmann::json j;
    j["gpu"] = gpu_name;
    j["compute_capability"] = std::to_string(compute_major) + "." + std::to_string(compute_minor);
    j["max_am"] = max_am;

    auto& one_e = j["one_electron"];
    one_e = nlohmann::json::array();
    for (const auto& r : results_1e) {
        one_e.push_back({
            {"la", r.la}, {"lb", r.lb},
            {"overlap", r.best_overlap},
            {"kinetic", r.best_kinetic},
            {"nuclear", r.best_nuclear},
            {"prefer_fused", r.prefer_fused},
            {"timings", {
                {"hw_overlap_us", r.hw_overlap_us},
                {"gen_overlap_us", r.gen_overlap_us},
                {"hw_kinetic_us", r.hw_kinetic_us},
                {"gen_kinetic_us", r.gen_kinetic_us},
                {"hw_nuclear_us", r.hw_nuclear_us},
                {"gen_nuclear_us", r.gen_nuclear_us},
                {"fused_us", r.fused_us}
            }}
        });
    }

    auto& two_e = j["two_electron"];
    two_e = nlohmann::json::array();
    for (const auto& r : results_2e) {
        nlohmann::json entry = {
            {"la", r.la}, {"lb", r.lb}, {"lc", r.lc}, {"ld", r.ld},
            {"variant", r.best_variant},
            {"timings", {
                {"handwritten_us", r.hw_us},
                {"generated_us", r.gen_us},
                {"cooperative_us", r.coop_us}
            }}
        };
        two_e.push_back(entry);
    }

    std::ofstream file(path);
    if (!file.is_open()) {
        std::fprintf(stderr, "ERROR: Cannot write to %s\n", path.c_str());
        return;
    }
    file << j.dump(2) << "\n";
    std::printf("\nDispatch table written to: %s\n", path.c_str());
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Parse --output flag
    std::string output_path;
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "--output") == 0 || std::strcmp(argv[i], "-o") == 0)
            && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    std::printf("=== Optimal Kernel Dispatch Benchmark ===\n\n");

    // Check CUDA
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::printf("ERROR: No CUDA devices found.\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("GPU: %s (Compute %d.%d, %d SMs)\n\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // Create stream and init Boys tables
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    double* d_boys_coeffs = device::math::boys_device_init(stream);
    cudaStreamSynchronize(stream);

    // Load (H2O)4 / aug-cc-pVDZ
    auto atoms = std::vector<data::Atom>{
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
    BasisSet basis = data::load_basis_set("aug-cc-pvdz", atoms);

    const Size nbf = basis.n_basis_functions();
    const Size n_shells = basis.n_shells();
    const int max_am = basis.max_angular_momentum();

    std::printf("System: (H2O)4 / aug-cc-pVDZ\n");
    std::printf("  Atoms: %zu  Shells: %zu  BF: %zu  Max AM: %d\n\n",
                atoms.size(), n_shells, nbf, max_am);

    // ========================================================================
    // 1e benchmarks
    // ========================================================================
    std::printf("=== One-Electron Integrals ===\n\n");
    std::printf("%-8s %6s  %10s %10s  %10s %10s  %10s %10s  %10s  %s\n",
                "AM", "Pairs",
                "HW-S(us)", "Gen-S(us)",
                "HW-T(us)", "Gen-T(us)",
                "HW-V(us)", "Gen-V(us)",
                "Fused(us)", "Fused?");
    std::printf("%-8s %6s  %10s %10s  %10s %10s  %10s %10s  %10s  %s\n",
                "--------", "------",
                "----------", "----------",
                "----------", "----------",
                "----------", "----------",
                "----------", "------");

    auto results_1e = benchmark_1e(basis, atoms, stream, d_boys_coeffs);

    for (const auto& r : results_1e) {
        auto fmt_time = [](double us) -> std::string {
            if (us < 0) return "N/A";
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%.1f", us);
            return buf;
        };

        std::printf("(%s,%s)    %6d  %10s %10s  %10s %10s  %10s %10s  %10s  %s\n",
                    am_char(r.la), am_char(r.lb), r.n_pairs,
                    fmt_time(r.hw_overlap_us).c_str(), fmt_time(r.gen_overlap_us).c_str(),
                    fmt_time(r.hw_kinetic_us).c_str(), fmt_time(r.gen_kinetic_us).c_str(),
                    fmt_time(r.hw_nuclear_us).c_str(), fmt_time(r.gen_nuclear_us).c_str(),
                    fmt_time(r.fused_us).c_str(),
                    r.prefer_fused ? "YES" : "no");
    }

    // ========================================================================
    // 2e benchmarks
    // ========================================================================
    std::printf("\n=== Two-Electron Integrals (ERI) ===\n\n");
    std::printf("%-14s %10s  %12s %12s %12s  %s\n",
                "AM", "Quartets", "HW (us)", "Gen (us)", "Coop (us)", "Best");
    std::printf("%-14s %10s  %12s %12s %12s  %s\n",
                "--------------", "----------",
                "------------", "------------", "------------", "----------");

    auto results_2e = benchmark_2e(basis, stream, d_boys_coeffs);

    for (const auto& r : results_2e) {
        auto fmt_time = [](double us) -> std::string {
            if (us < 0) return "N/A";
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%.1f", us);
            return buf;
        };

        std::printf("(%s%s|%s%s)      %10d  %12s %12s %12s  %s\n",
                    am_char(r.la), am_char(r.lb), am_char(r.lc), am_char(r.ld),
                    r.n_quartets,
                    fmt_time(r.hw_us).c_str(),
                    fmt_time(r.gen_us).c_str(),
                    fmt_time(r.coop_us).c_str(),
                    r.best_variant.c_str());
    }

    // ========================================================================
    // Summary
    // ========================================================================
    std::printf("\n=== Summary ===\n");

    int fused_count = 0;
    for (const auto& r : results_1e) {
        if (r.prefer_fused) ++fused_count;
    }
    std::printf("  1e: %zu AM pairs benchmarked, %d prefer fused\n",
                results_1e.size(), fused_count);

    int hw_count = 0, gen_count = 0, coop_count = 0;
    for (const auto& r : results_2e) {
        if (r.best_variant == "handwritten") ++hw_count;
        else if (r.best_variant == "generated") ++gen_count;
        else if (r.best_variant == "cooperative") ++coop_count;
    }
    std::printf("  2e: %zu AM quartets benchmarked\n", results_2e.size());
    std::printf("      handwritten wins: %d, generated wins: %d, cooperative wins: %d\n",
                hw_count, gen_count, coop_count);

    // ========================================================================
    // JSON output
    // ========================================================================
    if (!output_path.empty()) {
        write_dispatch_json(output_path,
                           prop.name, prop.major, prop.minor,
                           max_am, results_1e, results_2e);
    }

    // Cleanup
    device::math::boys_device_cleanup();
    cudaStreamDestroy(stream);

    return 0;
}

#endif  // LIBACCINT_USE_CUDA
