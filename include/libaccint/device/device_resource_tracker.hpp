// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file device_resource_tracker.hpp
/// @brief Dynamic GPU resource tracking and occupancy-aware dispatch advice
///
/// DeviceResourceTracker complements DeviceManager (static topology) with
/// runtime resource accounting.  It monitors active kernels, allocated global
/// memory, and per-stream work, and exposes occupancy estimation and batch-
/// sizing helpers so callers can make informed launch decisions.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/core/types.hpp>
#include <libaccint/device/device_properties.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

// Forward-declare the CUDA stream type so the header stays self-contained
// when parsed outside a CUDA TU.
struct CUstream_st;
using cudaStream_t = CUstream_st*;

// Forward-declare dim3 to avoid requiring cuda_runtime.h in every consumer.
#ifndef __CUDA_RUNTIME_H__
struct dim3 {
    unsigned int x{1}, y{1}, z{1};
    constexpr dim3() = default;
    constexpr dim3(unsigned int x_, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};
#endif

namespace libaccint::device {

// ============================================================================
// GPU Kernel Configuration (C++ mirror of the Python GpuKernelConfig)
// ============================================================================

/// @brief GPU parallelisation strategy for integral kernels
enum class GpuExecutionStrategy : uint8_t {
    ThreadPerQuartet,   ///< One thread per shell quartet
    WarpPerQuartet,     ///< One warp (32 threads) per shell quartet
    BlockPerQuartet,    ///< One thread block per shell quartet (cooperative)
};

/// @brief Memory-access pattern for computed integrals on the GPU
enum class GpuMemoryStrategy : uint8_t {
    RegisterBuffer,       ///< All integrals accumulated in registers
    StreamingWrites,      ///< Stream integrals to global memory
    SharedMemoryBuffer,   ///< Shared-memory accumulation before write-out
    SharedMemoryBlocked,  ///< Shared-memory with blocked primitive batching
};

/// @brief Contraction-degree range for GPU K-aware dispatch
///
/// Mirrors libaccint::kernels::cpu::generated::ContractionRange for the GPU
/// path so that the resource tracker can reason about contraction cost.
enum class GpuContractionRange : uint8_t {
    SmallK,   ///< K <= 3
    MediumK,  ///< 4 <= K <= 6
    LargeK,   ///< K > 6
};

/// @brief Integral type tag used by the resource tracker
///
/// A minimal enumeration that covers the integral classes relevant for GPU
/// workload sizing.  Kept separate from the benchmark and codegen enums to
/// avoid circular header dependencies.
enum class GpuIntegralType : uint8_t {
    Overlap,
    Kinetic,
    Nuclear,
    ERI,
};

/// @brief Complete GPU kernel launch configuration
///
/// Combines execution/memory strategy with the numeric parameters that drive
/// the CUDA occupancy formulae.
struct GpuKernelConfig {
    GpuExecutionStrategy execution_strategy = GpuExecutionStrategy::ThreadPerQuartet;
    GpuMemoryStrategy    memory_strategy    = GpuMemoryStrategy::RegisterBuffer;
    int  block_size            = 256;   ///< Threads per block
    int  min_blocks_per_sm     = 1;     ///< __launch_bounds__ min-blocks hint
    int  registers_per_thread  = 32;    ///< Estimated register usage per thread
    int  shared_mem_per_block  = 0;     ///< Dynamic shared memory (bytes)
};

// ============================================================================
// Occupancy & Batch Advice Structs
// ============================================================================

/// @brief Result of a theoretical occupancy calculation
struct OccupancyEstimate {
    double theoretical_occupancy = 0.0;   ///< Fraction in [0.0, 1.0]
    int    active_warps_per_sm   = 0;     ///< Warps that can be resident
    int    max_active_blocks_per_sm = 0;  ///< Blocks that can be resident
    std::string limiting_factor;          ///< Human-readable bottleneck description
};

/// @brief Recommended kernel launch geometry for a batch of work
struct BatchConfig {
    dim3   grid_dim{1};                ///< Grid dimensions
    dim3   block_dim{256};             ///< Block dimensions
    int    shared_mem_bytes     = 0;   ///< Dynamic shared memory per block
    int    num_launches         = 1;   ///< Number of kernel launches needed
    int    work_units_per_launch = 0;  ///< Work units processed per launch
};

// ============================================================================
// RAII Resource Reservation
// ============================================================================

class DeviceResourceTracker;  // forward

/// @brief RAII guard that reserves GPU resources on construction and releases
///        them on destruction.
///
/// Non-copyable, move-only.  Typically obtained via
/// DeviceResourceTracker::reserve().
class ResourceReservation {
public:
    ResourceReservation() = default;

    /// @brief Move-construct (transfers ownership)
    ResourceReservation(ResourceReservation&& other) noexcept;

    /// @brief Move-assign (transfers ownership)
    ResourceReservation& operator=(ResourceReservation&& other) noexcept;

    // Non-copyable
    ResourceReservation(const ResourceReservation&) = delete;
    ResourceReservation& operator=(const ResourceReservation&) = delete;

    /// @brief Release reserved resources
    ~ResourceReservation();

    /// @brief Was a real reservation made?
    [[nodiscard]] bool is_valid() const noexcept { return tracker_ != nullptr; }

    /// @brief Explicitly release resources before destruction
    void release() noexcept;

private:
    friend class DeviceResourceTracker;

    ResourceReservation(DeviceResourceTracker* tracker,
                        size_t reserved_bytes,
                        cudaStream_t stream) noexcept;

    DeviceResourceTracker* tracker_ = nullptr;
    size_t reserved_bytes_ = 0;
    cudaStream_t stream_   = nullptr;
};

// ============================================================================
// Device Resource Tracker
// ============================================================================

/// @brief Dynamic GPU resource tracker for occupancy-aware kernel dispatch
///
/// Complements DeviceManager (static topology queries) with a live view of
/// resource utilisation on the current device.  All mutable state is
/// thread-safe: atomic counters for the fast-path metrics and a mutex for
/// the per-stream tracking map.
///
/// Typical usage:
/// @code
///   auto& tracker = DeviceResourceTracker::instance();
///
///   GpuKernelConfig cfg{.block_size = 128, .registers_per_thread = 40,
///                       .shared_mem_per_block = 4096};
///
///   auto occ = tracker.estimate_occupancy(cfg, n_quartets);
///   if (tracker.can_launch(cfg, n_quartets)) {
///       auto reservation = tracker.reserve(cfg, n_quartets, stream);
///       my_kernel<<<grid, block, smem, stream>>>(...);
///       // reservation released automatically
///   }
/// @endcode
class DeviceResourceTracker {
public:
    // -----------------------------------------------------------------
    // Singleton access
    // -----------------------------------------------------------------

    /// @brief Get the singleton instance (initialises on first call)
    static DeviceResourceTracker& instance();

    // Non-copyable, non-movable
    DeviceResourceTracker(const DeviceResourceTracker&) = delete;
    DeviceResourceTracker& operator=(const DeviceResourceTracker&) = delete;
    DeviceResourceTracker(DeviceResourceTracker&&) = delete;
    DeviceResourceTracker& operator=(DeviceResourceTracker&&) = delete;

    // -----------------------------------------------------------------
    // Static capability queries (cached from device properties)
    // -----------------------------------------------------------------

    /// @brief Number of SMs on the device
    [[nodiscard]] int total_sms() const noexcept { return total_sms_; }

    /// @brief Total global memory (bytes)
    [[nodiscard]] size_t total_global_memory() const noexcept { return total_global_memory_; }

    /// @brief Shared memory per SM (bytes)
    [[nodiscard]] size_t total_shared_memory_per_sm() const noexcept {
        return total_shared_memory_per_sm_;
    }

    /// @brief Maximum resident blocks per SM
    [[nodiscard]] int max_blocks_per_sm() const noexcept { return max_blocks_per_sm_; }

    /// @brief Maximum resident threads per SM
    [[nodiscard]] int max_threads_per_sm() const noexcept { return max_threads_per_sm_; }

    /// @brief Maximum 32-bit registers per SM
    [[nodiscard]] int max_registers_per_sm() const noexcept { return max_registers_per_sm_; }

    /// @brief Warp size (32 for NVIDIA)
    [[nodiscard]] int warp_size() const noexcept { return warp_size_; }

    // -----------------------------------------------------------------
    // Dynamic counters
    // -----------------------------------------------------------------

    /// @brief Number of kernels currently in-flight
    [[nodiscard]] int active_kernels() const noexcept {
        return active_kernels_.load(std::memory_order_relaxed);
    }

    /// @brief Bytes of global memory currently reserved by tracked launches
    [[nodiscard]] size_t allocated_global_bytes() const noexcept {
        return allocated_global_bytes_.load(std::memory_order_relaxed);
    }

    /// @brief Number of streams with active tracked work
    [[nodiscard]] int active_streams() const noexcept {
        return active_streams_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------
    // Occupancy estimation
    // -----------------------------------------------------------------

    /// @brief Estimate theoretical occupancy for a given kernel config
    ///
    /// Uses the well-known CUDA occupancy formulae:
    ///   - blocks limited by threads = max_threads_per_sm / block_size
    ///   - blocks limited by registers = max_registers_per_sm /
    ///                                   (registers_per_thread * block_size)
    ///   - blocks limited by shared mem = shared_memory_per_sm /
    ///                                    shared_mem_per_block
    ///   - active blocks = min(above, max_blocks_per_sm)
    ///   - occupancy = (active_blocks * block_size) / max_threads_per_sm
    ///
    /// @param config   Kernel launch configuration
    /// @param n_work_units  Total work items (for informational context only)
    /// @return Occupancy estimate with limiting-factor description
    [[nodiscard]] OccupancyEstimate estimate_occupancy(
        const GpuKernelConfig& config,
        int n_work_units) const;

    // -----------------------------------------------------------------
    // Dispatch advice
    // -----------------------------------------------------------------

    /// @brief Recommend grid/block dimensions and number of launches
    ///
    /// The recommendation takes the occupancy estimate, device SM count,
    /// and total work into account.
    ///
    /// @param integral_type   Integral class (for heuristic tuning)
    /// @param am_quartet      Angular momentum quartet [la, lb, lc, ld]
    /// @param contraction     Contraction range tier
    /// @param total_work_units Number of shell pairs/quartets to process
    /// @return Recommended BatchConfig
    [[nodiscard]] BatchConfig recommend_batch_config(
        GpuIntegralType integral_type,
        AMQuartet am_quartet,
        GpuContractionRange contraction,
        int total_work_units) const;

    /// @brief Check whether sufficient resources exist for a launch
    ///
    /// Performs a lightweight headroom check against global-memory budget
    /// and SM availability.
    ///
    /// @param config        Kernel configuration
    /// @param n_work_units  Number of work items
    /// @return true if the launch should proceed
    [[nodiscard]] bool can_launch(const GpuKernelConfig& config,
                                  int n_work_units) const;

    /// @brief Wait for GPU resources to become available
    ///
    /// Polls can_launch() in 1ms increments until resources free up or the
    /// timeout expires.  Useful when a short wait can avoid a best-effort
    /// launch or a CPU fallback.
    ///
    /// Thread-safe: uses only atomic reads via can_launch().
    ///
    /// @param config        Kernel configuration
    /// @param n_work_units  Number of work items
    /// @param timeout       Maximum time to wait (default 100ms)
    /// @return true if resources became available, false if timed out
    [[nodiscard]] bool wait_for_resources(
        const GpuKernelConfig& config,
        int n_work_units,
        std::chrono::milliseconds timeout = std::chrono::milliseconds{100});

    // -----------------------------------------------------------------
    // RAII resource reservation
    // -----------------------------------------------------------------

    /// @brief Reserve resources for an upcoming kernel launch
    ///
    /// Atomically increments active_kernels_ and allocated_global_bytes_.
    /// The returned ResourceReservation will decrement them on destruction.
    ///
    /// @param config        Kernel configuration
    /// @param n_work_units  Number of work items
    /// @param stream        CUDA stream (nullptr for the default stream)
    /// @return A move-only reservation guard
    [[nodiscard]] ResourceReservation reserve(
        const GpuKernelConfig& config,
        int n_work_units,
        cudaStream_t stream = nullptr);

    // -----------------------------------------------------------------
    // Stream-aware tracking
    // -----------------------------------------------------------------

    /// @brief Per-stream resource snapshot
    struct StreamResourceState {
        int    kernel_count       = 0;   ///< Active kernels on this stream
        int    estimated_sm_usage = 0;   ///< Estimated SM blocks in use
        size_t allocated_bytes    = 0;   ///< Global bytes allocated
    };

    /// @brief Get a snapshot of the resource state for a given stream
    ///
    /// @param stream  CUDA stream to query
    /// @return State snapshot (zero-initialised if the stream is unknown)
    [[nodiscard]] StreamResourceState get_stream_state(cudaStream_t stream) const;

    // -----------------------------------------------------------------
    // Reinitialisation
    // -----------------------------------------------------------------

    /// @brief Reinitialise static capabilities from the current device
    ///
    /// Call this after changing the active CUDA device via DeviceManager.
    void refresh_device_properties();

    /// @brief Reset all dynamic counters and per-stream state to zero
    void reset_counters() noexcept;

    /// @brief Summary string for logging / diagnostics
    [[nodiscard]] std::string summary() const;

private:
    DeviceResourceTracker();
    ~DeviceResourceTracker() = default;

    // --- internal helpers called by ResourceReservation ----
    friend class ResourceReservation;

    /// @brief Called by ResourceReservation destructor
    void release_reservation(size_t bytes, cudaStream_t stream) noexcept;

    /// @brief Estimate global-memory bytes a launch will consume
    [[nodiscard]] size_t estimate_launch_bytes(
        const GpuKernelConfig& config, int n_work_units) const noexcept;

    /// @brief Estimate number of SM blocks a launch will occupy
    [[nodiscard]] int estimate_sm_blocks(
        const GpuKernelConfig& config, int n_work_units) const noexcept;

    // --- Heuristic helpers for recommend_batch_config ---

    /// @brief Choose block size based on integral type and AM
    [[nodiscard]] int choose_block_size(GpuIntegralType integral_type,
                                        int total_am) const noexcept;

    /// @brief Estimate per-thread register demand from AM class
    [[nodiscard]] int estimate_registers(GpuIntegralType integral_type,
                                          int total_am) const noexcept;

    /// @brief Estimate per-block shared memory from AM class
    [[nodiscard]] int estimate_shared_mem(GpuIntegralType integral_type,
                                           int total_am) const noexcept;

    // =========================================================================
    // Data members
    // =========================================================================

    // Static capabilities (queried once from the CUDA runtime)
    int    total_sms_                  = 0;
    size_t total_global_memory_        = 0;
    size_t total_shared_memory_per_sm_ = 0;
    int    max_blocks_per_sm_          = 0;
    int    max_threads_per_sm_         = 0;
    int    max_registers_per_sm_       = 0;
    int    warp_size_                  = 32;

    // Dynamic counters (updated atomically)
    std::atomic<int>    active_kernels_{0};
    std::atomic<size_t> allocated_global_bytes_{0};
    std::atomic<int>    active_streams_{0};

    // Per-stream tracking (guarded by stream_mutex_)
    mutable std::mutex stream_mutex_;
    std::unordered_map<cudaStream_t, StreamResourceState> stream_states_;

    /// @brief Maximum fraction of global memory we allow tracked launches to
    ///        reserve.  Leaves headroom for driver allocations and other users.
    static constexpr double kGlobalMemoryBudgetFraction = 0.85;

    /// @brief Maximum concurrent kernel launches we consider healthy.
    static constexpr int kMaxConcurrentKernels = 128;
};

}  // namespace libaccint::device

#endif  // LIBACCINT_USE_CUDA
