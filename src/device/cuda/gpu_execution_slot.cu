// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file gpu_execution_slot.cu
/// @brief Implementation of GPU execution slot pool for concurrent kernel execution

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/memory/gpu_execution_slot.hpp>
#include <libaccint/utils/error_handling.hpp>
#include <libaccint/core/backend.hpp>

#include <cuda_runtime.h>

namespace libaccint::memory {

// ============================================================================
// GpuExecutionSlot
// ============================================================================

GpuExecutionSlot::GpuExecutionSlot() = default;

GpuExecutionSlot::~GpuExecutionSlot() {
    free_device_buffers();
}

GpuExecutionSlot::GpuExecutionSlot(GpuExecutionSlot&& other) noexcept
    : stream(std::move(other.stream)),
      d_1e_output(other.d_1e_output),
      d_1e_capacity(other.d_1e_capacity),
      d_2e_output(other.d_2e_output),
      d_2e_capacity(other.d_2e_capacity),
      d_fused_1e_output(other.d_fused_1e_output),
      d_fused_capacity(other.d_fused_capacity),
      h_1e_staging(std::move(other.h_1e_staging)),
      h_2e_staging(std::move(other.h_2e_staging)) {
    other.d_1e_output = nullptr;
    other.d_1e_capacity = 0;
    other.d_2e_output = nullptr;
    other.d_2e_capacity = 0;
    other.d_fused_1e_output = nullptr;
    other.d_fused_capacity = 0;
}

GpuExecutionSlot& GpuExecutionSlot::operator=(GpuExecutionSlot&& other) noexcept {
    if (this != &other) {
        free_device_buffers();
        stream = std::move(other.stream);
        d_1e_output = other.d_1e_output;
        d_1e_capacity = other.d_1e_capacity;
        d_2e_output = other.d_2e_output;
        d_2e_capacity = other.d_2e_capacity;
        d_fused_1e_output = other.d_fused_1e_output;
        d_fused_capacity = other.d_fused_capacity;
        h_1e_staging = std::move(other.h_1e_staging);
        h_2e_staging = std::move(other.h_2e_staging);

        other.d_1e_output = nullptr;
        other.d_1e_capacity = 0;
        other.d_2e_output = nullptr;
        other.d_2e_capacity = 0;
        other.d_fused_1e_output = nullptr;
        other.d_fused_capacity = 0;
    }
    return *this;
}

void GpuExecutionSlot::ensure_1e_buffer(size_t required) {
    if (required <= d_1e_capacity) return;

    if (d_1e_output != nullptr) {
        cudaFree(d_1e_output);
    }
    size_t new_capacity = required * 2;
    cudaError_t err = cudaMalloc(&d_1e_output, new_capacity * sizeof(double));
    if (err != cudaSuccess) {
        d_1e_output = nullptr;
        d_1e_capacity = 0;
        throw BackendError(BackendType::CUDA,
            std::string("Failed to allocate 1e slot buffer: ") + cudaGetErrorString(err));
    }
    d_1e_capacity = new_capacity;
}

void GpuExecutionSlot::ensure_2e_buffer(size_t required) {
    if (required <= d_2e_capacity) return;

    if (d_2e_output != nullptr) {
        cudaFree(d_2e_output);
    }
    size_t new_capacity = required * 2;
    cudaError_t err = cudaMalloc(&d_2e_output, new_capacity * sizeof(double));
    if (err != cudaSuccess) {
        d_2e_output = nullptr;
        d_2e_capacity = 0;
        throw BackendError(BackendType::CUDA,
            std::string("Failed to allocate 2e slot buffer: ") + cudaGetErrorString(err));
    }
    d_2e_capacity = new_capacity;
}

void GpuExecutionSlot::ensure_fused_1e_buffer(size_t per_op_size) {
    size_t required = per_op_size * 3;  // [S | T | V] contiguous
    if (required <= d_fused_capacity) return;

    if (d_fused_1e_output != nullptr) {
        cudaFree(d_fused_1e_output);
    }
    size_t new_capacity = required * 2;
    cudaError_t err = cudaMalloc(&d_fused_1e_output, new_capacity * sizeof(double));
    if (err != cudaSuccess) {
        d_fused_1e_output = nullptr;
        d_fused_capacity = 0;
        throw BackendError(BackendType::CUDA,
            std::string("Failed to allocate fused 1e slot buffer: ") + cudaGetErrorString(err));
    }
    d_fused_capacity = new_capacity;
}

void GpuExecutionSlot::free_device_buffers() {
    if (d_1e_output != nullptr) {
        cudaFree(d_1e_output);
        d_1e_output = nullptr;
        d_1e_capacity = 0;
    }
    if (d_2e_output != nullptr) {
        cudaFree(d_2e_output);
        d_2e_output = nullptr;
        d_2e_capacity = 0;
    }
    if (d_fused_1e_output != nullptr) {
        cudaFree(d_fused_1e_output);
        d_fused_1e_output = nullptr;
        d_fused_capacity = 0;
    }
}

// ============================================================================
// GpuSlotPool
// ============================================================================

GpuSlotPool::GpuSlotPool(size_t n_slots) {
    slots_.reserve(n_slots);
    for (size_t i = 0; i < n_slots; ++i) {
        auto slot = std::make_unique<GpuExecutionSlot>();
        available_.push(slot.get());
        slots_.push_back(std::move(slot));
    }
}

GpuExecutionSlot& GpuSlotPool::acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !available_.empty(); });

    GpuExecutionSlot* slot = available_.front();
    available_.pop();
    return *slot;
}

void GpuSlotPool::release(GpuExecutionSlot& slot) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        available_.push(&slot);
    }
    cv_.notify_one();
}

void GpuSlotPool::synchronize_all() {
    // No lock needed - we sync all slots, not just available ones
    for (auto& slot : slots_) {
        if (slot->stream.valid()) {
            slot->stream.synchronize();
        }
    }
}

size_t GpuSlotPool::available() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return available_.size();
}

// ============================================================================
// ScopedGpuSlot
// ============================================================================

ScopedGpuSlot::ScopedGpuSlot(GpuSlotPool& pool)
    : pool_(pool), slot_(&pool.acquire()) {}

ScopedGpuSlot::~ScopedGpuSlot() {
    pool_.release(*slot_);
}

}  // namespace libaccint::memory

#endif  // LIBACCINT_USE_CUDA
