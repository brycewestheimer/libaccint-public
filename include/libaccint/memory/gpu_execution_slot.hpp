// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file gpu_execution_slot.hpp
/// @brief Per-thread GPU execution slots for concurrent kernel execution
///
/// Provides a pool of execution slots, each containing an independent CUDA
/// stream and set of device/host buffers. Multiple host threads can execute
/// GPU work concurrently by acquiring separate slots, eliminating the need
/// for a global mutex that serializes GPU access.

#include <libaccint/config.hpp>
#include <libaccint/memory/stream_management.hpp>

#if LIBACCINT_USE_CUDA

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

namespace libaccint::memory {

// ============================================================================
// GpuExecutionSlot - Bundle of stream + buffers for one concurrent execution
// ============================================================================

/// @brief A self-contained execution context for concurrent GPU work
///
/// Each slot owns an independent CUDA stream and a set of device output
/// buffers plus host staging buffers.  A thread that acquires a slot can
/// run kernels, transfer data, and synchronize without interfering with
/// other threads that hold different slots.
///
/// The buffer management methods (`ensure_1e_buffer`, etc.) follow the
/// same grow-only / 2x-headroom policy used by the former per-instance
/// buffers in CudaEngine.
struct GpuExecutionSlot {
    /// CUDA stream owned by this slot
    StreamHandle stream;

    // ---- One-electron device output buffer ----
    double* d_1e_output = nullptr;
    size_t  d_1e_capacity = 0;

    // ---- Two-electron device output buffer ----
    double* d_2e_output = nullptr;
    size_t  d_2e_capacity = 0;

    // ---- Fused one-electron device output buffer [S|T|V] ----
    double* d_fused_1e_output = nullptr;
    size_t  d_fused_capacity = 0;

    // ---- Host staging buffers ----
    std::vector<double> h_1e_staging;
    std::vector<double> h_2e_staging;

    /// @brief Construct a slot with a fresh stream
    GpuExecutionSlot();

    /// @brief Destructor - frees device buffers (stream freed by StreamHandle)
    ~GpuExecutionSlot();

    // Move-only
    GpuExecutionSlot(GpuExecutionSlot&& other) noexcept;
    GpuExecutionSlot& operator=(GpuExecutionSlot&& other) noexcept;
    GpuExecutionSlot(const GpuExecutionSlot&) = delete;
    GpuExecutionSlot& operator=(const GpuExecutionSlot&) = delete;

    /// @brief Ensure the 1e output buffer has at least @p required doubles
    void ensure_1e_buffer(size_t required);

    /// @brief Ensure the 2e output buffer has at least @p required doubles
    void ensure_2e_buffer(size_t required);

    /// @brief Ensure the fused 1e output buffer has capacity for 3 * @p per_op_size doubles
    void ensure_fused_1e_buffer(size_t per_op_size);

private:
    /// @brief Free all device buffers
    void free_device_buffers();
};

// ============================================================================
// GpuSlotPool - Thread-safe pool of execution slots
// ============================================================================

/// @brief Thread-safe pool of GpuExecutionSlots for concurrent GPU access
///
/// Maintains a fixed-size pool of execution slots.  Threads acquire a slot
/// before performing GPU work and release it afterwards.  If all slots are
/// in use, `acquire()` blocks until one becomes available.
///
/// Example usage:
/// @code
///     GpuSlotPool pool(4);
///     {
///         ScopedGpuSlot scoped(pool);
///         auto& slot = scoped.slot();
///         slot.ensure_2e_buffer(output_size);
///         my_kernel<<<grid, block, 0, slot.stream.get()>>>(slot.d_2e_output, ...);
///         slot.stream.synchronize();
///     }  // slot automatically released
/// @endcode
class GpuSlotPool {
public:
    /// @brief Construct a pool with @p n_slots execution slots
    /// @param n_slots Number of slots (default: 4)
    explicit GpuSlotPool(size_t n_slots = 4);

    /// @brief Destructor - destroys all slots
    ~GpuSlotPool() = default;

    // Non-copyable, non-movable
    GpuSlotPool(const GpuSlotPool&) = delete;
    GpuSlotPool& operator=(const GpuSlotPool&) = delete;
    GpuSlotPool(GpuSlotPool&&) = delete;
    GpuSlotPool& operator=(GpuSlotPool&&) = delete;

    /// @brief Acquire an execution slot (blocks if none available)
    /// @return Reference to an available slot
    [[nodiscard]] GpuExecutionSlot& acquire();

    /// @brief Release a slot back to the pool
    /// @param slot The slot to release (must have been acquired from this pool)
    void release(GpuExecutionSlot& slot);

    /// @brief Synchronize all streams across all slots
    void synchronize_all();

    /// @brief Get the total number of slots in the pool
    [[nodiscard]] size_t size() const noexcept { return slots_.size(); }

    /// @brief Get the number of currently available (non-acquired) slots
    [[nodiscard]] size_t available() const;

private:
    std::vector<std::unique_ptr<GpuExecutionSlot>> slots_;
    std::queue<GpuExecutionSlot*> available_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

// ============================================================================
// ScopedGpuSlot - RAII helper for automatic slot release
// ============================================================================

/// @brief RAII wrapper that acquires a slot on construction and releases on destruction
///
/// Example usage:
/// @code
///     GpuSlotPool pool(4);
///     {
///         ScopedGpuSlot scoped(pool);
///         // Use scoped.slot().stream, scoped.slot().d_2e_output, etc.
///     }  // slot released here
/// @endcode
class ScopedGpuSlot {
public:
    /// @brief Acquire a slot from a pool
    /// @param pool The pool to acquire from
    explicit ScopedGpuSlot(GpuSlotPool& pool);

    /// @brief Destructor - releases the slot back to the pool
    ~ScopedGpuSlot();

    // Non-copyable, non-movable
    ScopedGpuSlot(const ScopedGpuSlot&) = delete;
    ScopedGpuSlot& operator=(const ScopedGpuSlot&) = delete;
    ScopedGpuSlot(ScopedGpuSlot&&) = delete;
    ScopedGpuSlot& operator=(ScopedGpuSlot&&) = delete;

    /// @brief Get the acquired slot
    [[nodiscard]] GpuExecutionSlot& slot() noexcept { return *slot_; }
    [[nodiscard]] const GpuExecutionSlot& slot() const noexcept { return *slot_; }

    /// @brief Get the slot's CUDA stream for convenience
    [[nodiscard]] gpu_stream_t stream() const noexcept { return slot_->stream.get(); }

private:
    GpuSlotPool& pool_;
    GpuExecutionSlot* slot_;
};

}  // namespace libaccint::memory

#endif  // LIBACCINT_USE_CUDA
