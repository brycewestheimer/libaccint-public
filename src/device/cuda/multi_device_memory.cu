// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file multi_device_memory.cu
/// @brief Implementation of multi-device memory management

#include <libaccint/device/multi_device_memory.hpp>

#if LIBACCINT_USE_CUDA

#include <algorithm>

namespace libaccint::device {

// ============================================================================
// DeviceWorkspace Implementation
// ============================================================================

DeviceWorkspace::DeviceWorkspace(int device_id, size_t /*initial_pool_size*/)
    : device_id_(device_id) {
    ScopedDevice guard(device_id_);
    cudaStreamCreate(&stream_);
}

DeviceWorkspace::~DeviceWorkspace() {
    reset();
    if (stream_ != nullptr) {
        ScopedDevice guard(device_id_);
        cudaStreamDestroy(stream_);
    }
}

DeviceWorkspace::DeviceWorkspace(DeviceWorkspace&& other) noexcept
    : device_id_(other.device_id_),
      stream_(other.stream_),
      allocations_(std::move(other.allocations_)),
      allocated_bytes_(other.allocated_bytes_) {
    other.stream_ = nullptr;
    other.allocated_bytes_ = 0;
}

DeviceWorkspace& DeviceWorkspace::operator=(DeviceWorkspace&& other) noexcept {
    if (this != &other) {
        reset();
        if (stream_ != nullptr) {
            ScopedDevice guard(device_id_);
            cudaStreamDestroy(stream_);
        }
        
        device_id_ = other.device_id_;
        stream_ = other.stream_;
        allocations_ = std::move(other.allocations_);
        allocated_bytes_ = other.allocated_bytes_;
        
        other.stream_ = nullptr;
        other.allocated_bytes_ = 0;
    }
    return *this;
}

void DeviceWorkspace::reset() {
    if (allocations_.empty()) return;
    
    ScopedDevice guard(device_id_);
    for (auto& [ptr, size] : allocations_) {
        gpuFree(ptr);
    }
    allocations_.clear();
    allocated_bytes_ = 0;
}

void DeviceWorkspace::synchronize() {
    ScopedDevice guard(device_id_);
    cudaStreamSynchronize(stream_);
}

// ============================================================================
// MultiDeviceMemoryManager Implementation
// ============================================================================

MultiDeviceMemoryManager::MultiDeviceMemoryManager(
    const std::vector<int>& device_ids,
    size_t pool_size_per_device)
    : device_ids_(device_ids) {
    
    for (int device_id : device_ids_) {
        workspaces_[device_id] = std::make_unique<DeviceWorkspace>(
            device_id, pool_size_per_device);
    }
}

DeviceWorkspace& MultiDeviceMemoryManager::workspace(int device_id) {
    auto it = workspaces_.find(device_id);
    if (it == workspaces_.end()) {
        throw DeviceError(device_id, "Device not managed by this memory manager");
    }
    return *it->second;
}

const DeviceWorkspace& MultiDeviceMemoryManager::workspace(int device_id) const {
    auto it = workspaces_.find(device_id);
    if (it == workspaces_.end()) {
        throw DeviceError(device_id, "Device not managed by this memory manager");
    }
    return *it->second;
}

void MultiDeviceMemoryManager::reset_all() {
    for (auto& [device_id, ws] : workspaces_) {
        ws->reset();
    }
}

void MultiDeviceMemoryManager::reset(int device_id) {
    workspace(device_id).reset();
}

void MultiDeviceMemoryManager::synchronize_all() {
    for (auto& [device_id, ws] : workspaces_) {
        ws->synchronize();
    }
}

void MultiDeviceMemoryManager::synchronize(int device_id) {
    workspace(device_id).synchronize();
}

size_t MultiDeviceMemoryManager::total_allocated_bytes() const {
    size_t total = 0;
    for (const auto& [device_id, ws] : workspaces_) {
        total += ws->allocated_bytes();
    }
    return total;
}

size_t MultiDeviceMemoryManager::allocated_bytes(int device_id) const {
    return workspace(device_id).allocated_bytes();
}

}  // namespace libaccint::device

#endif  // LIBACCINT_USE_CUDA
