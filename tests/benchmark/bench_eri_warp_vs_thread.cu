// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_eri_warp_vs_thread.cu
/// @brief Benchmark comparing warp-per-quartet vs thread-per-quartet ERI kernels
///
/// Measures kernel execution time for both strategies across batch sizes.
/// Also profiles the thread-per-quartet kernel across all AM combinations
/// to establish the performance baseline needed to determine when (if ever)
/// the warp kernel would be beneficial.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/kernels/eri_kernel_cuda.hpp>
#include <libaccint/kernels/eri_kernel_warp_cuda.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/device_data.hpp>
#include <libaccint/memory/device_memory.hpp>
#include <libaccint/memory/stream_management.hpp>

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

namespace libaccint::device::math {
    double* boys_device_init(cudaStream_t stream);
    void boys_device_cleanup();
    double* boys_device_get_coeffs();
    bool boys_device_is_initialized();
}

namespace {

using namespace libaccint;
using namespace libaccint::kernels::cuda;

// Number of timing iterations per benchmark
constexpr int N_WARMUP = 2;
constexpr int N_ITERS = 10;

struct BenchResult {
    double avg_us;   // Average kernel time in microseconds
    double min_us;   // Minimum kernel time
    double max_us;   // Maximum kernel time
};

/// @brief Build a ShellSet with n_shells identical shells at given AM
/// Returns a unique_ptr since ShellSet is non-copyable
std::unique_ptr<ShellSet> make_shell_set(int am, int n_shells, int n_prims,
                                          double center_spread = 2.0) {
    auto set = std::make_unique<ShellSet>(am, n_prims);

    std::vector<double> exponents(n_prims);
    std::vector<double> coefficients(n_prims);
    // STO-3G-like parameters
    if (n_prims >= 1) { exponents[0] = 3.42525091; coefficients[0] = 0.15432897; }
    if (n_prims >= 2) { exponents[1] = 0.62391373; coefficients[1] = 0.53532814; }
    if (n_prims >= 3) { exponents[2] = 0.16885540; coefficients[2] = 0.44463454; }

    int func_idx = 0;
    int n_cart = (am + 1) * (am + 2) / 2;
    for (int i = 0; i < n_shells; ++i) {
        double x = center_spread * static_cast<double>(i) / std::max(n_shells - 1, 1);
        Point3D center{x, 0.0, 0.0};
        Shell shell(static_cast<AngularMomentum>(am), center, exponents, coefficients);
        shell.set_shell_index(i);
        shell.set_atom_index(i);
        shell.set_function_index(func_idx);
        func_idx += n_cart;
        set->add_shell(shell);
    }
    return set;
}

/// @brief Time a kernel dispatch using CUDA events
BenchResult time_kernel(
    const basis::ShellSetQuartetDeviceData& quartet,
    const double* d_boys_coeffs,
    double* d_output,
    cudaStream_t stream,
    bool use_warp_kernel)
{
    memory::EventHandle start, stop;

    // Warmup
    for (int i = 0; i < N_WARMUP; ++i) {
        if (use_warp_kernel) {
            dispatch_eri_warp_kernel(quartet, d_boys_coeffs, d_output, stream);
        } else {
            dispatch_eri_kernel(quartet, d_boys_coeffs, d_output, stream);
        }
    }
    cudaStreamSynchronize(stream);

    // Timed iterations
    std::vector<float> times(N_ITERS);
    for (int i = 0; i < N_ITERS; ++i) {
        start.record(stream);
        if (use_warp_kernel) {
            dispatch_eri_warp_kernel(quartet, d_boys_coeffs, d_output, stream);
        } else {
            dispatch_eri_kernel(quartet, d_boys_coeffs, d_output, stream);
        }
        stop.record(stream);
        cudaStreamSynchronize(stream);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start.get(), stop.get());
        times[i] = ms * 1000.0f;  // Convert to microseconds
    }

    BenchResult result{};
    result.min_us = 1e9;
    result.max_us = 0.0;
    double sum = 0.0;
    for (float t : times) {
        sum += t;
        if (t < result.min_us) result.min_us = t;
        if (t > result.max_us) result.max_us = t;
    }
    result.avg_us = sum / N_ITERS;
    return result;
}

void run_benchmark() {
    // Initialize CUDA
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::printf("No CUDA devices available\n");
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("GPU: %s (SM %d.%d, %d SMs)\n\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    std::fflush(stdout);

    // Initialize Boys function
    double* d_boys_coeffs = nullptr;
    if (!device::math::boys_device_is_initialized()) {
        d_boys_coeffs = device::math::boys_device_init(nullptr);
    } else {
        d_boys_coeffs = device::math::boys_device_get_coeffs();
    }
    cudaDeviceSynchronize();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // =========================================================================
    // Part 1: (ss|ss) Warp vs Thread comparison across batch sizes
    // =========================================================================
    std::printf("=== Part 1: (ss|ss) Warp vs Thread-per-Quartet ===\n");
    std::printf("%-12s  %-12s  %-14s  %-14s  %-10s\n",
                "N_shells", "N_quartets", "Thread (us)", "Warp (us)", "Speedup");
    std::printf("%-12s  %-12s  %-14s  %-14s  %-10s\n",
                "--------", "----------", "-----------", "---------", "-------");

    for (int n_shells : {1, 2, 4, 8, 16, 32}) {
        auto set = make_shell_set(0, n_shells, 3);

        auto data = basis::upload_shell_set(*set, stream);
        cudaStreamSynchronize(stream);

        basis::ShellSetQuartetDeviceData quartet;
        quartet.a = data;
        quartet.b = data;
        quartet.c = data;
        quartet.d = data;

        size_t output_size = eri_output_size(quartet);
        memory::DeviceBuffer<double> d_output(output_size);

        size_t n_quartets = static_cast<size_t>(n_shells) * n_shells * n_shells * n_shells;

        auto thread_result = time_kernel(quartet, d_boys_coeffs, d_output.data(), stream, false);
        auto warp_result = time_kernel(quartet, d_boys_coeffs, d_output.data(), stream, true);

        double speedup = thread_result.avg_us / warp_result.avg_us;

        std::printf("%-12d  %-12zu  %-14.2f  %-14.2f  %-10.3fx\n",
                    n_shells, n_quartets,
                    thread_result.avg_us, warp_result.avg_us, speedup);
        std::fflush(stdout);

        basis::free_shell_set_device_data(data);
    }

    // =========================================================================
    // Part 2: Thread-per-quartet kernel across all AM combinations
    // =========================================================================
    std::printf("\n=== Part 2: Thread-per-Quartet Across AM Combinations ===\n");
    std::printf("%-12s  %-6s  %-12s  %-14s  %-14s  %-12s\n",
                "Quartet", "AM", "N_quartets", "Avg (us)", "Min (us)", "us/quartet");
    std::printf("%-12s  %-6s  %-12s  %-14s  %-14s  %-12s\n",
                "-------", "----", "----------", "--------", "--------", "----------");

    // Use 8 shells for all AM combinations to get meaningful batch sizes
    int n_shells = 8;
    int n_prims = 3;

    for (int la = 0; la <= 2; ++la) {
        for (int lb = la; lb <= 2; ++lb) {
            for (int lc = 0; lc <= la; ++lc) {
                for (int ld = (lc == la ? lb : lc); ld <= 2; ++ld) {
                    int total_am = la + lb + lc + ld;

                    auto set_a = make_shell_set(la, n_shells, n_prims, 2.0);
                    auto set_b = make_shell_set(lb, n_shells, n_prims, 2.5);
                    auto set_c = make_shell_set(lc, n_shells, n_prims, 3.0);
                    auto set_d = make_shell_set(ld, n_shells, n_prims, 3.5);

                    auto data_a = basis::upload_shell_set(*set_a, stream);
                    auto data_b = basis::upload_shell_set(*set_b, stream);
                    auto data_c = basis::upload_shell_set(*set_c, stream);
                    auto data_d = basis::upload_shell_set(*set_d, stream);
                    cudaStreamSynchronize(stream);

                    basis::ShellSetQuartetDeviceData quartet;
                    quartet.a = data_a;
                    quartet.b = data_b;
                    quartet.c = data_c;
                    quartet.d = data_d;

                    size_t output_size = eri_output_size(quartet);
                    memory::DeviceBuffer<double> d_output(output_size);

                    size_t n_quartets_total = static_cast<size_t>(n_shells) * n_shells * n_shells * n_shells;

                    auto result = time_kernel(quartet, d_boys_coeffs, d_output.data(), stream, false);

                    char label[32];
                    std::snprintf(label, sizeof(label), "(%d%d|%d%d)", la, lb, lc, ld);

                    std::printf("%-12s  %-6d  %-12zu  %-14.2f  %-14.2f  %-12.4f\n",
                                label, total_am, n_quartets_total,
                                result.avg_us, result.min_us,
                                result.avg_us / n_quartets_total);
                    std::fflush(stdout);

                    basis::free_shell_set_device_data(data_a);
                    basis::free_shell_set_device_data(data_b);
                    basis::free_shell_set_device_data(data_c);
                    basis::free_shell_set_device_data(data_d);
                }
            }
        }
    }

    // =========================================================================
    // Part 3: Thread-per-quartet register pressure indicator
    // =========================================================================
    std::printf("\n=== Part 3: Occupancy & Register Pressure (Thread-per-Quartet) ===\n");
    std::printf("%-12s  %-6s  %-12s  %-14s  %-14s\n",
                "Quartet", "AM", "Cartesian", "Avg (us)", "Throughput");
    std::printf("%-12s  %-6s  %-12s  %-14s  %-14s\n",
                "-------", "----", "---------", "--------", "----------");

    // Fixed quartet count, vary AM to see register pressure effect
    for (int am : {0, 1, 2}) {
        int la = am, lb = am, lc = am, ld = am;
        int total_am = la + lb + lc + ld;
        int n_cart = (am + 1) * (am + 2) / 2;
        int n_cart_total = n_cart * n_cart * n_cart * n_cart;

        // Use enough shells to saturate GPU
        int ns = 16;
        auto set = make_shell_set(am, ns, n_prims, 4.0);
        auto data = basis::upload_shell_set(*set, stream);
        cudaStreamSynchronize(stream);

        basis::ShellSetQuartetDeviceData quartet;
        quartet.a = data;
        quartet.b = data;
        quartet.c = data;
        quartet.d = data;

        size_t output_size = eri_output_size(quartet);
        memory::DeviceBuffer<double> d_output(output_size);

        size_t n_q = static_cast<size_t>(ns) * ns * ns * ns;

        auto result = time_kernel(quartet, d_boys_coeffs, d_output.data(), stream, false);

        double integrals_per_us = (n_q * n_cart_total) / result.avg_us;

        char label[32];
        std::snprintf(label, sizeof(label), "(%d%d|%d%d)", am, am, am, am);

        std::printf("%-12s  %-6d  %-12d  %-14.2f  %-14.0f ints/us\n",
                    label, total_am, n_cart_total,
                    result.avg_us, integrals_per_us);
        std::fflush(stdout);

        basis::free_shell_set_device_data(data);
    }

    cudaStreamDestroy(stream);
    std::printf("\nDone.\n");
}

}  // anonymous namespace

int main() {
    run_benchmark();
    return 0;
}

#else  // LIBACCINT_USE_CUDA

#include <cstdio>
int main() {
    std::printf("CUDA not enabled. Skipping benchmark.\n");
    return 0;
}

#endif
