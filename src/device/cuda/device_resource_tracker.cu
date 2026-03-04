// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file device_resource_tracker.cu
/// @brief CUDA implementation of dynamic GPU resource tracking
///
/// Provides occupancy estimation, batch-sizing advice, resource gating,
/// and RAII reservation for GPU kernel launches.

#include <libaccint/device/device_resource_tracker.hpp>

#if LIBACCINT_USE_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <thread>

namespace libaccint::device {

// ============================================================================
// ResourceReservation implementation
// ============================================================================

ResourceReservation::ResourceReservation(DeviceResourceTracker* tracker,
                                         size_t reserved_bytes,
                                         cudaStream_t stream) noexcept
    : tracker_(tracker),
      reserved_bytes_(reserved_bytes),
      stream_(stream) {}

ResourceReservation::ResourceReservation(ResourceReservation&& other) noexcept
    : tracker_(other.tracker_),
      reserved_bytes_(other.reserved_bytes_),
      stream_(other.stream_) {
    other.tracker_ = nullptr;
    other.reserved_bytes_ = 0;
    other.stream_ = nullptr;
}

ResourceReservation& ResourceReservation::operator=(ResourceReservation&& other) noexcept {
    if (this != &other) {
        release();
        tracker_ = other.tracker_;
        reserved_bytes_ = other.reserved_bytes_;
        stream_ = other.stream_;
        other.tracker_ = nullptr;
        other.reserved_bytes_ = 0;
        other.stream_ = nullptr;
    }
    return *this;
}

ResourceReservation::~ResourceReservation() {
    release();
}

void ResourceReservation::release() noexcept {
    if (tracker_) {
        tracker_->release_reservation(reserved_bytes_, stream_);
        tracker_ = nullptr;
        reserved_bytes_ = 0;
        stream_ = nullptr;
    }
}

// ============================================================================
// DeviceResourceTracker — singleton & construction
// ============================================================================

DeviceResourceTracker& DeviceResourceTracker::instance() {
    static DeviceResourceTracker inst;
    return inst;
}

DeviceResourceTracker::DeviceResourceTracker() {
    refresh_device_properties();
}

void DeviceResourceTracker::refresh_device_properties() {
    int device_id = 0;
    cudaGetDevice(&device_id);

    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        // Graceful fallback — leave capabilities at zero
        return;
    }

    total_sms_                  = props.multiProcessorCount;
    total_global_memory_        = props.totalGlobalMem;
    total_shared_memory_per_sm_ = static_cast<size_t>(props.sharedMemPerMultiprocessor);
    max_blocks_per_sm_          = props.maxBlocksPerMultiProcessor;
    max_threads_per_sm_         = props.maxThreadsPerMultiProcessor;
    max_registers_per_sm_       = props.regsPerMultiprocessor;
    warp_size_                  = props.warpSize;
}

void DeviceResourceTracker::reset_counters() noexcept {
    active_kernels_.store(0, std::memory_order_relaxed);
    allocated_global_bytes_.store(0, std::memory_order_relaxed);
    active_streams_.store(0, std::memory_order_relaxed);

    std::lock_guard lock(stream_mutex_);
    stream_states_.clear();
}

// ============================================================================
// Occupancy estimation
// ============================================================================

OccupancyEstimate DeviceResourceTracker::estimate_occupancy(
    const GpuKernelConfig& config,
    [[maybe_unused]] int n_work_units) const {

    OccupancyEstimate est;

    // Guard against degenerate inputs
    if (config.block_size <= 0 || max_threads_per_sm_ <= 0) {
        est.limiting_factor = "invalid configuration (block_size or max_threads_per_sm is zero)";
        return est;
    }

    // 1. Blocks limited by thread count
    const int blocks_by_threads = max_threads_per_sm_ / config.block_size;

    // 2. Blocks limited by register file
    int blocks_by_registers = max_blocks_per_sm_;  // assume no limit
    if (config.registers_per_thread > 0 && max_registers_per_sm_ > 0) {
        const int regs_per_block = config.registers_per_thread * config.block_size;
        blocks_by_registers = (regs_per_block > 0)
                                  ? max_registers_per_sm_ / regs_per_block
                                  : max_blocks_per_sm_;
    }

    // 3. Blocks limited by shared memory
    int blocks_by_shared = max_blocks_per_sm_;  // assume no limit
    if (config.shared_mem_per_block > 0 && total_shared_memory_per_sm_ > 0) {
        blocks_by_shared = static_cast<int>(
            total_shared_memory_per_sm_ /
            static_cast<size_t>(config.shared_mem_per_block));
    }

    // 4. Active blocks = min(all limits, hardware cap)
    const int active_blocks = std::min({blocks_by_threads,
                                         blocks_by_registers,
                                         blocks_by_shared,
                                         max_blocks_per_sm_});

    est.max_active_blocks_per_sm = std::max(active_blocks, 0);
    est.active_warps_per_sm =
        (warp_size_ > 0)
            ? (est.max_active_blocks_per_sm * config.block_size) / warp_size_
            : 0;
    est.theoretical_occupancy =
        (max_threads_per_sm_ > 0)
            ? static_cast<double>(est.max_active_blocks_per_sm * config.block_size) /
                  static_cast<double>(max_threads_per_sm_)
            : 0.0;
    est.theoretical_occupancy = std::clamp(est.theoretical_occupancy, 0.0, 1.0);

    // Determine the limiting factor
    const int min_limit = std::min({blocks_by_threads, blocks_by_registers,
                                     blocks_by_shared, max_blocks_per_sm_});
    if (min_limit == blocks_by_registers && blocks_by_registers < blocks_by_threads) {
        est.limiting_factor = "registers (" +
                              std::to_string(config.registers_per_thread) +
                              " regs/thread × " +
                              std::to_string(config.block_size) + " threads)";
    } else if (min_limit == blocks_by_shared && blocks_by_shared < blocks_by_threads) {
        est.limiting_factor = "shared memory (" +
                              std::to_string(config.shared_mem_per_block) +
                              " B/block, " +
                              std::to_string(total_shared_memory_per_sm_) +
                              " B/SM)";
    } else if (min_limit == max_blocks_per_sm_ &&
               max_blocks_per_sm_ < blocks_by_threads) {
        est.limiting_factor = "max blocks per SM (" +
                              std::to_string(max_blocks_per_sm_) + ")";
    } else {
        est.limiting_factor = "threads (" +
                              std::to_string(config.block_size) +
                              " threads/block, " +
                              std::to_string(max_threads_per_sm_) +
                              " max threads/SM)";
    }

    return est;
}

// ============================================================================
// Dispatch advice — recommend_batch_config
// ============================================================================

int DeviceResourceTracker::choose_block_size(
    [[maybe_unused]] GpuIntegralType integral_type,
    int total_am) const noexcept {
    // Low AM: maximise occupancy with large blocks
    if (total_am <= 2)  return 256;
    // Medium AM: moderate register pressure
    if (total_am <= 4)  return 128;
    // High AM: heavy register pressure, keep blocks small
    return 64;
}

int DeviceResourceTracker::estimate_registers(
    GpuIntegralType integral_type,
    int total_am) const noexcept {
    // Rough per-thread register estimates calibrated against typical
    // Obara-Saika / Rys-quadrature CUDA kernels at different AM tiers.
    if (integral_type == GpuIntegralType::ERI) {
        if (total_am <= 2)  return 32;
        if (total_am <= 4)  return 48;
        if (total_am <= 8)  return 80;
        return 128;
    }
    // One-electron integrals are lighter
    if (total_am <= 2) return 24;
    if (total_am <= 4) return 36;
    return 56;
}

int DeviceResourceTracker::estimate_shared_mem(
    GpuIntegralType integral_type,
    int total_am) const noexcept {
    // Low-AM kernels fit entirely in registers
    if (total_am <= 2) return 0;
    if (integral_type == GpuIntegralType::ERI) {
        // Shared memory used for Rys weights, intermediate buffers
        if (total_am <= 4) return 4096;
        if (total_am <= 8) return 8192;
        return 16384;
    }
    // 1e shared-memory prefetch of nuclear charges etc.
    if (total_am <= 4) return 2048;
    return 4096;
}

BatchConfig DeviceResourceTracker::recommend_batch_config(
    GpuIntegralType integral_type,
    AMQuartet am_quartet,
    GpuContractionRange contraction,
    int total_work_units) const {

    BatchConfig bc;

    const int total_am = am_quartet[0] + am_quartet[1] +
                         am_quartet[2] + am_quartet[3];

    // --- Choose block size ----------------------------------------------
    const int block_size = choose_block_size(integral_type, total_am);
    bc.block_dim = dim3(static_cast<unsigned>(block_size));

    // --- Estimate occupancy with the chosen block size -------------------
    GpuKernelConfig cfg;
    cfg.block_size           = block_size;
    cfg.registers_per_thread = estimate_registers(integral_type, total_am);
    cfg.shared_mem_per_block = estimate_shared_mem(integral_type, total_am);
    bc.shared_mem_bytes      = cfg.shared_mem_per_block;

    const auto occ = estimate_occupancy(cfg, total_work_units);

    // --- Compute the number of blocks that can run device-wide ----------
    const int blocks_per_sm = std::max(occ.max_active_blocks_per_sm, 1);
    const int device_blocks = blocks_per_sm * total_sms_;

    // --- Grid size should not exceed work items / block_size ------------
    const int blocks_needed =
        (total_work_units + block_size - 1) / block_size;

    // Cap per-launch grid to device capacity to allow overlap with memory
    // transfers on other streams.
    int grid_x = std::min(blocks_needed, device_blocks);

    // For large-K contractions, reduce grid to leave room for memory traffic.
    if (contraction == GpuContractionRange::LargeK) {
        grid_x = std::max(grid_x * 3 / 4, 1);
    }

    bc.grid_dim = dim3(static_cast<unsigned>(grid_x));

    // --- Number of launches ---------------------------------------------
    const int work_per_launch = grid_x * block_size;
    bc.work_units_per_launch  = work_per_launch;
    bc.num_launches = (total_work_units + work_per_launch - 1) / work_per_launch;
    bc.num_launches = std::max(bc.num_launches, 1);

    return bc;
}

// ============================================================================
// Resource gating — can_launch
// ============================================================================

bool DeviceResourceTracker::can_launch(
    const GpuKernelConfig& config,
    int n_work_units) const {

    // 1. Too many concurrent kernels?
    if (active_kernels_.load(std::memory_order_relaxed) >= kMaxConcurrentKernels) {
        return false;
    }

    // 2. Global memory headroom check
    const size_t budget = static_cast<size_t>(
        static_cast<double>(total_global_memory_) * kGlobalMemoryBudgetFraction);
    const size_t currently_used =
        allocated_global_bytes_.load(std::memory_order_relaxed);
    const size_t needed = estimate_launch_bytes(config, n_work_units);
    if (currently_used + needed > budget) {
        return false;
    }

    // 3. SM-block availability (heuristic: we want at least one SM free)
    const auto occ = estimate_occupancy(config, n_work_units);
    const int blocks_needed = estimate_sm_blocks(config, n_work_units);
    const int total_slots = occ.max_active_blocks_per_sm * total_sms_;
    // Very rough: reject if the launch would consume >90 % of SM block slots
    if (total_slots > 0 && blocks_needed > total_slots * 9 / 10) {
        return false;
    }

    return true;
}

// ============================================================================
// Wait for resources
// ============================================================================

bool DeviceResourceTracker::wait_for_resources(
    const GpuKernelConfig& config,
    int n_work_units,
    std::chrono::milliseconds timeout) {

    if (can_launch(config, n_work_units)) {
        return true;
    }

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    static constexpr auto kPollInterval = std::chrono::milliseconds{1};

    while (std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(kPollInterval);
        if (can_launch(config, n_work_units)) {
            return true;
        }
    }

    return false;
}

// ============================================================================
// RAII reservation
// ============================================================================

ResourceReservation DeviceResourceTracker::reserve(
    const GpuKernelConfig& config,
    int n_work_units,
    cudaStream_t stream) {

    const size_t bytes = estimate_launch_bytes(config, n_work_units);
    const int sm_blocks = estimate_sm_blocks(config, n_work_units);

    // Atomically account for the reservation
    active_kernels_.fetch_add(1, std::memory_order_acq_rel);
    allocated_global_bytes_.fetch_add(bytes, std::memory_order_acq_rel);

    // Per-stream tracking
    {
        std::lock_guard lock(stream_mutex_);
        auto& state = stream_states_[stream];
        if (state.kernel_count == 0) {
            // First kernel on this stream — bump active_streams_
            active_streams_.fetch_add(1, std::memory_order_acq_rel);
        }
        state.kernel_count += 1;
        state.estimated_sm_usage += sm_blocks;
        state.allocated_bytes += bytes;
    }

    return ResourceReservation{this, bytes, stream};
}

void DeviceResourceTracker::release_reservation(size_t bytes,
                                                 cudaStream_t stream) noexcept {
    active_kernels_.fetch_sub(1, std::memory_order_acq_rel);
    allocated_global_bytes_.fetch_sub(bytes, std::memory_order_acq_rel);

    std::lock_guard lock(stream_mutex_);
    auto it = stream_states_.find(stream);
    if (it != stream_states_.end()) {
        auto& state = it->second;
        state.kernel_count -= 1;
        state.allocated_bytes =
            (state.allocated_bytes >= bytes) ? state.allocated_bytes - bytes : 0;
        if (state.kernel_count <= 0) {
            // No more active kernels on this stream — remove entry
            stream_states_.erase(it);
            active_streams_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
}

// ============================================================================
// Stream state query
// ============================================================================

DeviceResourceTracker::StreamResourceState
DeviceResourceTracker::get_stream_state(cudaStream_t stream) const {
    std::lock_guard lock(stream_mutex_);
    auto it = stream_states_.find(stream);
    if (it != stream_states_.end()) {
        return it->second;
    }
    return {};  // zero-initialised
}

// ============================================================================
// Estimation helpers
// ============================================================================

size_t DeviceResourceTracker::estimate_launch_bytes(
    const GpuKernelConfig&,
    int n_work_units) const noexcept {
    // Rough model: each work unit needs ~(block_size * 8) bytes for output +
    // input data.  Shared memory is on-chip and not counted against global.
    //
    // For ERI quartets the output size is n_cart(la)*n_cart(lb)*n_cart(lc)*
    // n_cart(ld) doubles.  Without the AM information here we use a coarse
    // multiplier of 256 bytes/work-unit as a conservative middle ground for
    // mixed workloads.
    static constexpr size_t kBytesPerWorkUnit = 256;
    return static_cast<size_t>(n_work_units) * kBytesPerWorkUnit;
}

int DeviceResourceTracker::estimate_sm_blocks(
    const GpuKernelConfig& config,
    int n_work_units) const noexcept {
    if (config.block_size <= 0) return 0;
    return (n_work_units + config.block_size - 1) / config.block_size;
}

// ============================================================================
// Summary
// ============================================================================

std::string DeviceResourceTracker::summary() const {
    std::ostringstream os;
    os << "DeviceResourceTracker {\n"
       << "  SMs:               " << total_sms_ << "\n"
       << "  Global memory:     " << (total_global_memory_ / (1024 * 1024)) << " MiB\n"
       << "  Shared mem/SM:     " << total_shared_memory_per_sm_ << " B\n"
       << "  Max blocks/SM:     " << max_blocks_per_sm_ << "\n"
       << "  Max threads/SM:    " << max_threads_per_sm_ << "\n"
       << "  Max registers/SM:  " << max_registers_per_sm_ << "\n"
       << "  Warp size:         " << warp_size_ << "\n"
       << "  --- dynamic ---\n"
       << "  Active kernels:    " << active_kernels_.load(std::memory_order_relaxed) << "\n"
       << "  Allocated global:  "
       << (allocated_global_bytes_.load(std::memory_order_relaxed) / (1024 * 1024))
       << " MiB\n"
       << "  Active streams:    " << active_streams_.load(std::memory_order_relaxed) << "\n"
       << "}";
    return os.str();
}

}  // namespace libaccint::device

#endif  // LIBACCINT_USE_CUDA
