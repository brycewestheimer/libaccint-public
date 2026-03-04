// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/kernels/cost_model.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/config.hpp>

#include <algorithm>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#if LIBACCINT_USE_CUDA
#include <libaccint/device/device_resource_tracker.hpp>
#endif

namespace libaccint::kernels {

// Default GPU parameters when no device is available
static constexpr int    kDefaultGpuSmCount         = 50;    // Mid-range GPU
static constexpr int    kDefaultGpuMaxThreadsPerSm = 2048;
static constexpr int    kDefaultGpuWarpSize        = 32;
static constexpr double kDefaultGpuGflops          = 1000.0;

CostModel::CostModel() : profile_(detect_hardware()) {
    refresh_gpu_params_from_device();
}

CostModel::CostModel(HardwareProfile profile) : profile_(std::move(profile)) {
    refresh_gpu_params_from_device();
}

HardwareProfile CostModel::detect_hardware() {
    HardwareProfile profile;

    // Detect CPU cores
#ifdef _OPENMP
    profile.cpu_cores = omp_get_max_threads();
#else
    profile.cpu_cores = 1;
#endif

    // Detect SIMD width (conservative default)
    profile.simd_width = 4;  // AVX default

#if defined(__x86_64__) || defined(_M_X64)
    // Try to detect AVX-512 support
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        if (ebx & (1 << 16)) {  // AVX-512F
            profile.simd_width = 8;
        }
    }
#endif

    // Estimate CPU GFLOPS (conservative estimate for modern CPUs)
    // Assume ~2 GFLOPS per core for double-precision scalar operations
    profile.cpu_gflops = 2.0 * profile.cpu_cores;

    // GPU characteristics (conservative defaults if no GPU available)
    profile.gpu_gflops = 0.0;  // Set to 0 if no GPU
    profile.gpu_launch_overhead_ns = 5000.0;

#if LIBACCINT_USE_CUDA
    // If CUDA is compiled in, assume reasonable GPU characteristics
    // Actual detection would require CUDA runtime
    if (has_cuda_backend()) {
        profile.gpu_gflops = 1000.0;  // Conservative for modern GPUs
        profile.gpu_launch_overhead_ns = 5000.0;
    }
#endif

    // Estimate memory bandwidth (conservative)
    profile.memory_bandwidth_gb_s = 50.0;

    return profile;
}

double CostModel::estimate_flops(OperatorKind op, const AMQuartet& am,
                                  const std::array<int, 4>& n_prims) const {
    // FLOP estimation based on operator type and angular momentum
    // These are empirical formulas calibrated against actual performance

    int total_am = am[0] + am[1] + am[2] + am[3];
    int total_prims = n_prims[0] * n_prims[1] * n_prims[2] * n_prims[3];

    // Base FLOP count depends on operator type
    double base_flops = 0.0;

    switch (op) {
        case OperatorKind::Overlap:
            // Overlap: ~20 FLOPs per primitive pair for low AM
            base_flops = 20.0 * n_prims[0] * n_prims[1];
            break;

        case OperatorKind::Kinetic:
            // Kinetic: ~50 FLOPs per primitive pair (includes Laplacian)
            base_flops = 50.0 * n_prims[0] * n_prims[1];
            break;

        case OperatorKind::Nuclear:
        case OperatorKind::PointCharge:
            // Nuclear attraction: ~100 FLOPs per primitive pair (Boys function)
            base_flops = 100.0 * n_prims[0] * n_prims[1];
            break;

        case OperatorKind::Coulomb:
            // ERI: Complex scaling with AM
            // Base: ~200 FLOPs per primitive quartet for (ss|ss)
            // Scales roughly as (L+1)^4 for Rys quadrature
            {
                int n_rys = (total_am + 2) / 2;  // Number of Rys roots
                double am_factor = std::pow(total_am + 1.0, 2.0);
                base_flops = 200.0 * total_prims * n_rys * am_factor;
            }
            break;

        case OperatorKind::ErfCoulomb:
        case OperatorKind::ErfcCoulomb:
            // Range-separated: Similar to Coulomb with modified Boys function
            {
                int n_rys = (total_am + 2) / 2;
                double am_factor = std::pow(total_am + 1.0, 2.0);
                base_flops = 220.0 * total_prims * n_rys * am_factor;
            }
            break;

        default:
            // Fallback for unhandled operators
            base_flops = 100.0 * total_prims;
            break;
    }

    // Angular momentum scaling for recurrence relations
    // Higher AM requires more recurrence steps
    if (total_am > 0) {
        // Obara-Saika recurrence scales as O(L^3) per Cartesian component
        double am_scaling = 1.0 + 0.5 * total_am + 0.1 * total_am * total_am;
        base_flops *= am_scaling;
    }

    // Output scaling: number of Cartesian components
    int n_cart_a = n_cartesian(am[0]);
    int n_cart_b = n_cartesian(am[1]);
    int n_cart_c = is_two_electron(op) ? n_cartesian(am[2]) : 1;
    int n_cart_d = is_two_electron(op) ? n_cartesian(am[3]) : 1;
    int n_integrals = n_cart_a * n_cart_b * n_cart_c * n_cart_d;

    // Final FLOP count includes some overhead per output integral
    return base_flops + 5.0 * n_integrals;
}

double CostModel::estimate_memory_bytes(const AMQuartet& am, Size batch_size) const {
    // Estimate memory traffic for coefficient/exponent loading and result storage
    int n_cart_a = n_cartesian(am[0]);
    int n_cart_b = n_cartesian(am[1]);
    int n_cart_c = n_cartesian(am[2]);
    int n_cart_d = n_cartesian(am[3]);
    int n_integrals = n_cart_a * n_cart_b * n_cart_c * n_cart_d;

    // Input: coefficients and exponents (estimated)
    double input_bytes = 16.0 * 8;  // 16 doubles per shell (conservative)

    // Output: integral values
    double output_bytes = static_cast<double>(n_integrals) * sizeof(Real);

    return (input_bytes + output_bytes) * static_cast<double>(batch_size);
}

CostEstimate CostModel::estimate(
    OperatorKind op,
    const AMQuartet& am,
    const std::array<int, 4>& n_primitives,
    Size batch_size) const {

    CostEstimate cost;

    // Estimate FLOPs for a single integral
    double flops_per_integral = estimate_flops(op, am, n_primitives);
    double total_flops = flops_per_integral * static_cast<double>(batch_size);

    // Estimate memory traffic
    double memory_bytes = estimate_memory_bytes(am, batch_size);

    // Convert GFLOPS to FLOPS per nanosecond
    double cpu_flops_per_ns = profile_.cpu_gflops * 1e9 / 1e9;  // GFLOPS = 10^9 FLOPS/sec
    double gpu_flops_per_ns = profile_.gpu_gflops * 1e9 / 1e9;

    // CPU serial time
    if (cpu_flops_per_ns > 0) {
        // Account for single-core performance
        double single_core_flops_per_ns = (profile_.cpu_gflops / profile_.cpu_cores) * 1.0;
        cost.cpu_serial_ns = total_flops / single_core_flops_per_ns;
    }

    // CPU SIMD time (assume SIMD_width speedup with 70% efficiency)
    if (profile_.simd_width > 1) {
        double simd_speedup = profile_.simd_width * 0.7;
        cost.cpu_simd_ns = cost.cpu_serial_ns / simd_speedup;
    }

    // CPU threaded time (Amdahl's law with 90% parallel fraction)
    if (profile_.cpu_cores > 1) {
        double parallel_fraction = 0.9;
        double serial_fraction = 1.0 - parallel_fraction;
        double threaded_speedup = 1.0 / (serial_fraction + parallel_fraction / profile_.cpu_cores);

        // Combined threading + SIMD
        double simd_factor = (profile_.simd_width > 1) ? profile_.simd_width * 0.7 : 1.0;
        cost.cpu_threaded_ns = cost.cpu_serial_ns / (threaded_speedup * simd_factor);
    }

    // GPU time
    if (profile_.gpu_gflops > 0) {
        // Phase 4.5: True batched execution - single kernel launch per batch
        cost.gpu_launch_ns = profile_.gpu_launch_overhead_ns;  // Single launch per batch

        // Default GPU compute uses WarpPerQuartet model for backward compatibility
        cost.gpu_compute_ns = compute_gpu_strategy_time(
            total_flops, batch_size, GpuExecutionStrategy::WarpPerQuartet);
    }

    return cost;
}

ExecutionStrategy CostModel::select_strategy(
    const CostEstimate& cost,
    bool gpu_available) const {

    // Collect all viable strategies with their costs
    struct StrategyOption {
        ExecutionStrategy strategy;
        double cost_ns;
    };

    std::vector<StrategyOption> options;

    // Always consider serial CPU
    if (cost.cpu_serial_ns > 0) {
        options.push_back({ExecutionStrategy::SerialCPU, cost.cpu_serial_ns});
    }

    // Consider SIMD if available
    if (cost.cpu_simd_ns > 0 && profile_.simd_width > 1) {
        options.push_back({ExecutionStrategy::SimdCPU, cost.cpu_simd_ns});
    }

    // Consider threaded if available
    if (cost.cpu_threaded_ns > 0 && profile_.cpu_cores > 1) {
        if (profile_.simd_width > 1) {
            options.push_back({ExecutionStrategy::ThreadedSimdCPU, cost.cpu_threaded_ns});
        } else {
            options.push_back({ExecutionStrategy::ThreadedCPU, cost.cpu_threaded_ns});
        }
    }

    // Consider GPU if available
    if (gpu_available && cost.gpu_compute_ns > 0) {
        double gpu_total = cost.gpu_total_ns();
        // Legacy path: map to WarpPerQuartetGPU for backward compatibility
        if (gpu_total > 0) {
            options.push_back({ExecutionStrategy::WarpPerQuartetGPU, gpu_total});
        }
    }

    // Select the strategy with minimum cost
    if (options.empty()) {
        return ExecutionStrategy::SerialCPU;  // Fallback
    }

    auto best = std::min_element(options.begin(), options.end(),
        [](const StrategyOption& a, const StrategyOption& b) {
            return a.cost_ns < b.cost_ns;
        });

    return best->strategy;
}

// ---------------------------------------------------------------------------
// Strategy-specific GPU cost estimation
// ---------------------------------------------------------------------------

double CostModel::compute_gpu_strategy_time(
    double total_flops,
    Size n_quartets,
    GpuExecutionStrategy gpu_strategy) const {

    if (profile_.gpu_gflops <= 0.0 || n_quartets == 0) {
        return 0.0;
    }

    const double gpu_flops_per_ns = profile_.gpu_gflops;  // GFLOPS = FLOPS/ns
    const int sms  = (gpu_sm_count_ > 0) ? gpu_sm_count_ : kDefaultGpuSmCount;
    const int tps  = (gpu_max_threads_per_sm_ > 0) ? gpu_max_threads_per_sm_
                                                    : kDefaultGpuMaxThreadsPerSm;
    const int ws   = (gpu_warp_size_ > 0) ? gpu_warp_size_ : kDefaultGpuWarpSize;
    const int warps_per_sm = tps / ws;

    double compute_ns = 0.0;
    const double flops_per_quartet = (n_quartets > 0)
        ? total_flops / static_cast<double>(n_quartets)
        : total_flops;

    switch (gpu_strategy) {
        case GpuExecutionStrategy::ThreadPerQuartet: {
            // Each thread processes one quartet.
            // Effective throughput = SMs × threads_per_SM × clock.
            // Since gpu_flops_per_ns already encodes peak, scale by
            // occupancy fraction: total_threads / peak_threads.
            const double peak_threads = static_cast<double>(sms) * tps;
            const double active_threads = std::min(
                static_cast<double>(n_quartets), peak_threads);
            const double utilisation = active_threads / peak_threads;

            // Compute-bound time with utilisation scaling
            compute_ns = total_flops / (gpu_flops_per_ns * utilisation);
            break;
        }
        case GpuExecutionStrategy::WarpPerQuartet: {
            // One warp (32 threads) cooperates on each quartet.
            // Throughput per quartet is higher (warp-level parallelism in
            // recurrence relations), but fewer quartets run concurrently.
            const double peak_warps = static_cast<double>(sms) * warps_per_sm;
            const double active_warps = std::min(
                static_cast<double>(n_quartets), peak_warps);
            const double utilisation = active_warps / peak_warps;

            // Warp-collaborative execution: ~0.7× throughput efficiency vs TPQ
            // due to intra-warp synchronisation, but better cache reuse.
            constexpr double kWarpEfficiency = 0.7;
            compute_ns = total_flops / (gpu_flops_per_ns * utilisation * kWarpEfficiency);
            break;
        }
        case GpuExecutionStrategy::BlockPerQuartet: {
            // One thread block per quartet. Leverages shared memory for
            // intermediate recurrence values but is limited to SM count.
            const double active_blocks = std::min(
                static_cast<double>(n_quartets), static_cast<double>(sms));
            const double utilisation = active_blocks / static_cast<double>(sms);

            // Shared-memory latency factor: accessing shared memory is
            // ~5× faster than global, but the block-level synchronisation
            // overhead reduces effective throughput to ~0.5× peak.
            constexpr double kBlockEfficiency = 0.5;
            constexpr double kSharedMemLatencyFactor = 0.85;  // amortised smem benefit
            compute_ns = total_flops / (gpu_flops_per_ns * utilisation
                                         * kBlockEfficiency * kSharedMemLatencyFactor);
            break;
        }
    }

    // Memory-bandwidth bound floor
    const double memory_bytes =
        estimate_memory_bytes({0, 0, 0, 0}, n_quartets);  // rough byte estimate
    const double memory_bound_ns =
        memory_bytes / profile_.memory_bandwidth_gb_s;  // GB/s ≈ bytes/ns
    return std::max(compute_ns, memory_bound_ns);
}

GpuStrategyCost CostModel::estimate_gpu_strategy(
    OperatorKind op,
    const AMQuartet& am,
    const std::array<int, 4>& n_primitives,
    Size batch_size,
    GpuExecutionStrategy gpu_strategy) const {

    GpuStrategyCost result;
    result.strategy = gpu_strategy;

    if (profile_.gpu_gflops <= 0.0) {
        return result;  // No GPU — costs remain at zero
    }

    result.launch_ns  = profile_.gpu_launch_overhead_ns;

    const double flops_per_integral = estimate_flops(op, am, n_primitives);
    const double total_flops = flops_per_integral * static_cast<double>(batch_size);

    result.compute_ns = compute_gpu_strategy_time(total_flops, batch_size, gpu_strategy);
    return result;
}

ExecutionStrategy CostModel::select_strategy(
    const CostEstimate& cost,
    bool gpu_available,
    GpuExecutionStrategy gpu_strategy) const {

    // Reuse the existing CPU-side logic from the two-argument overload
    // by building the same options list, but substitute the GPU cost
    // with the strategy-specific one that is already in `cost`.

    struct StrategyOption {
        ExecutionStrategy strategy;
        double cost_ns;
    };

    std::vector<StrategyOption> options;

    if (cost.cpu_serial_ns > 0) {
        options.push_back({ExecutionStrategy::SerialCPU, cost.cpu_serial_ns});
    }
    if (cost.cpu_simd_ns > 0 && profile_.simd_width > 1) {
        options.push_back({ExecutionStrategy::SimdCPU, cost.cpu_simd_ns});
    }
    if (cost.cpu_threaded_ns > 0 && profile_.cpu_cores > 1) {
        if (profile_.simd_width > 1) {
            options.push_back({ExecutionStrategy::ThreadedSimdCPU, cost.cpu_threaded_ns});
        } else {
            options.push_back({ExecutionStrategy::ThreadedCPU, cost.cpu_threaded_ns});
        }
    }

    // GPU: map GpuExecutionStrategy to the corresponding ExecutionStrategy
    if (gpu_available && cost.gpu_compute_ns > 0) {
        double gpu_total = cost.gpu_total_ns();
        ExecutionStrategy exec = ExecutionStrategy::WarpPerQuartetGPU;  // default
        switch (gpu_strategy) {
            case GpuExecutionStrategy::ThreadPerQuartet:
                exec = ExecutionStrategy::ThreadPerIntegralGPU;
                break;
            case GpuExecutionStrategy::WarpPerQuartet:
                exec = ExecutionStrategy::WarpPerQuartetGPU;
                break;
            case GpuExecutionStrategy::BlockPerQuartet:
                exec = ExecutionStrategy::BlockPerBatchGPU;
                break;
        }
        if (gpu_total > 0) {
            options.push_back({exec, gpu_total});
        }
    }

    if (options.empty()) {
        return ExecutionStrategy::SerialCPU;
    }

    auto best_it = std::min_element(options.begin(), options.end(),
        [](const StrategyOption& a, const StrategyOption& b) {
            return a.cost_ns < b.cost_ns;
        });
    return best_it->strategy;
}

// ---------------------------------------------------------------------------
// DeviceResourceTracker integration
// ---------------------------------------------------------------------------

void CostModel::refresh_gpu_params_from_device() {
#if LIBACCINT_USE_CUDA
    if (has_cuda_backend()) {
        try {
            auto& tracker = device::DeviceResourceTracker::instance();
            gpu_sm_count_            = tracker.total_sms();
            gpu_max_threads_per_sm_  = tracker.max_threads_per_sm();
            gpu_warp_size_           = tracker.warp_size();

            // Update profile GFLOPS from SM count if the profile still has
            // the hardcoded default.  Rough heuristic:
            //   peak_dp_gflops ≈ SMs × (threads/SM) × clock_GHz × 2
            // We don't have clock speed from the tracker, so only override
            // when the profile has the placeholder zero.
            if (profile_.gpu_gflops <= 0.0 && gpu_sm_count_ > 0) {
                profile_.gpu_gflops = kDefaultGpuGflops;
            }
        } catch (...) {
            // DeviceResourceTracker may throw if no CUDA device is
            // initialised — fall through to defaults.
        }
    }
#endif
    // Apply defaults for any parameters that weren't set
    if (gpu_sm_count_ <= 0)            gpu_sm_count_           = kDefaultGpuSmCount;
    if (gpu_max_threads_per_sm_ <= 0)  gpu_max_threads_per_sm_ = kDefaultGpuMaxThreadsPerSm;
    if (gpu_warp_size_ <= 0)           gpu_warp_size_          = kDefaultGpuWarpSize;
}

}  // namespace libaccint::kernels
