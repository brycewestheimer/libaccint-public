// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file multi_device_memory.hpp
/// @brief Per-device memory management for multi-GPU execution
///
/// Provides memory allocation and management that is device-aware, ensuring
/// allocations happen on the correct device and preventing cross-device errors.

#include <libaccint/config.hpp>
#include <libaccint/device/device_manager.hpp>
#include <libaccint/memory/device_memory.hpp>

#include <memory>
#include <unordered_map>
#include <vector>

#if LIBACCINT_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace libaccint::device {

#if LIBACCINT_USE_CUDA
using gpuStream_t = cudaStream_t;
using gpuError_t = cudaError_t;

constexpr gpuError_t gpuSuccess = cudaSuccess;
constexpr auto gpuMemcpyHostToDevice = cudaMemcpyHostToDevice;
constexpr auto gpuMemcpyDeviceToHost = cudaMemcpyDeviceToHost;

inline gpuError_t gpuMalloc(void** ptr, size_t bytes) { return cudaMalloc(ptr, bytes); }
inline gpuError_t gpuFree(void* ptr) { return cudaFree(ptr); }
inline gpuError_t gpuMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    return cudaMemcpy(dst, src, bytes, kind);
}
inline gpuError_t gpuMemcpyPeer(void* dst, int dst_device,
                                const void* src, int src_device,
                                size_t bytes) {
    return cudaMemcpyPeer(dst, dst_device, src, src_device, bytes);
}
inline const char* gpuGetErrorString(gpuError_t err) { return cudaGetErrorString(err); }

#else
using gpuStream_t = void*;
#endif

/// @brief Per-device workspace that owns memory allocated on a specific GPU
///
/// DeviceWorkspace provides RAII-based memory management for a specific device.
/// All allocations made through this workspace are guaranteed to be on the
/// associated device.
///
/// Usage:
/// @code
///   DeviceWorkspace workspace(0, 1024 * 1024);  // 1MB on device 0
///   
///   double* buffer = workspace.allocate<double>(1000);
///   // buffer is guaranteed to be on device 0
///   
///   workspace.reset();  // Free all allocations
/// @endcode
class DeviceWorkspace {
public:
    /// @brief Create a workspace for the specified device
    /// @param device_id The GPU device ID
    /// @param initial_pool_size Initial memory pool size (0 = no pooling)
    explicit DeviceWorkspace(int device_id, size_t initial_pool_size = 0);
    
    /// @brief Destructor - frees all allocations
    ~DeviceWorkspace();
    
    // Move-only
    DeviceWorkspace(DeviceWorkspace&& other) noexcept;
    DeviceWorkspace& operator=(DeviceWorkspace&& other) noexcept;
    DeviceWorkspace(const DeviceWorkspace&) = delete;
    DeviceWorkspace& operator=(const DeviceWorkspace&) = delete;
    
    /// @brief Get the device ID this workspace is associated with
    [[nodiscard]] int device_id() const noexcept { return device_id_; }
    
    /// @brief Allocate device memory on this workspace's device
    /// @tparam T Element type
    /// @param count Number of elements to allocate
    /// @return Device pointer to allocated memory
    template<typename T>
    [[nodiscard]] T* allocate(size_t count);
    
    /// @brief Free a specific allocation
    /// @tparam T Element type
    /// @param ptr Pointer previously returned by allocate()
    template<typename T>
    void deallocate(T* ptr);
    
    /// @brief Free all allocations
    void reset();
    
    /// @brief Get total allocated bytes
    [[nodiscard]] size_t allocated_bytes() const noexcept { return allocated_bytes_; }
    
    /// @brief Get the GPU stream for this workspace
    [[nodiscard]] gpuStream_t stream() const noexcept { return stream_; }
    
    /// @brief Synchronize this workspace's stream
    void synchronize();

private:
    int device_id_;
    gpuStream_t stream_ = nullptr;
    std::unordered_map<void*, size_t> allocations_;  ///< Maps device ptr → allocation size in bytes
    size_t allocated_bytes_ = 0;
};

/// @brief Multi-device memory manager that coordinates allocations across GPUs
///
/// MultiDeviceMemoryManager provides a unified interface for allocating and
/// managing memory across multiple GPU devices. It maintains per-device
/// workspaces and ensures thread-safe access.
///
/// Usage:
/// @code
///   MultiDeviceMemoryManager mgr({0, 1, 2});
///   
///   // Allocate on specific device
///   double* d0_data = mgr.allocate<double>(0, 1000);
///   double* d1_data = mgr.allocate<double>(1, 1000);
///   
///   // Get workspace for direct access
///   auto& ws = mgr.workspace(0);
/// @endcode
class MultiDeviceMemoryManager {
public:
    /// @brief Create a multi-device memory manager for the specified devices
    /// @param device_ids List of device IDs to manage
    /// @param pool_size_per_device Initial pool size per device (0 = no pooling)
    explicit MultiDeviceMemoryManager(
        const std::vector<int>& device_ids,
        size_t pool_size_per_device = 0);
    
    /// @brief Destructor - frees all allocations on all devices
    ~MultiDeviceMemoryManager() = default;
    
    // Move-only
    MultiDeviceMemoryManager(MultiDeviceMemoryManager&&) noexcept = default;
    MultiDeviceMemoryManager& operator=(MultiDeviceMemoryManager&&) noexcept = default;
    MultiDeviceMemoryManager(const MultiDeviceMemoryManager&) = delete;
    MultiDeviceMemoryManager& operator=(const MultiDeviceMemoryManager&) = delete;
    
    /// @brief Get the list of managed device IDs
    [[nodiscard]] const std::vector<int>& device_ids() const noexcept {
        return device_ids_;
    }
    
    /// @brief Get the number of managed devices
    [[nodiscard]] int device_count() const noexcept {
        return static_cast<int>(device_ids_.size());
    }
    
    /// @brief Get the workspace for a specific device
    /// @param device_id The device ID
    /// @return Reference to the device's workspace
    [[nodiscard]] DeviceWorkspace& workspace(int device_id);
    [[nodiscard]] const DeviceWorkspace& workspace(int device_id) const;
    
    /// @brief Allocate memory on a specific device
    /// @tparam T Element type
    /// @param device_id Target device
    /// @param count Number of elements
    /// @return Device pointer
    template<typename T>
    [[nodiscard]] T* allocate(int device_id, size_t count);
    
    /// @brief Deallocate memory on a specific device
    template<typename T>
    void deallocate(int device_id, T* ptr);
    
    /// @brief Free all allocations on all devices
    void reset_all();
    
    /// @brief Free allocations on a specific device
    void reset(int device_id);
    
    /// @brief Synchronize all devices
    void synchronize_all();
    
    /// @brief Synchronize a specific device
    void synchronize(int device_id);
    
    /// @brief Get total allocated bytes across all devices
    [[nodiscard]] size_t total_allocated_bytes() const;
    
    /// @brief Get allocated bytes on a specific device
    [[nodiscard]] size_t allocated_bytes(int device_id) const;
    
    /// @brief Copy data between devices (peer-to-peer if available)
    /// @tparam T Element type
    /// @param dst_device Destination device
    /// @param dst Destination pointer (on dst_device)
    /// @param src_device Source device
    /// @param src Source pointer (on src_device)
    /// @param count Number of elements to copy
    template<typename T>
    void copy_between_devices(int dst_device, T* dst,
                              int src_device, const T* src,
                              size_t count);

private:
    std::vector<int> device_ids_;
    std::unordered_map<int, std::unique_ptr<DeviceWorkspace>> workspaces_;
};

// ============================================================================
// Template Implementations
// ============================================================================

#if LIBACCINT_USE_CUDA

template<typename T>
T* DeviceWorkspace::allocate(size_t count) {
    if (count == 0) return nullptr;
    
    ScopedDevice guard(device_id_);
    
    void* raw_ptr = nullptr;
    const size_t alloc_bytes = count * sizeof(T);
    gpuError_t err = gpuMalloc(&raw_ptr, alloc_bytes);
    T* ptr = static_cast<T*>(raw_ptr);
    if (err != gpuSuccess) {
        throw DeviceError(device_id_, 
            "Memory allocation failed: " + std::string(gpuGetErrorString(err)));
    }
    
    allocations_[static_cast<void*>(ptr)] = alloc_bytes;
    allocated_bytes_ += alloc_bytes;
    return ptr;
}

template<typename T>
void DeviceWorkspace::deallocate(T* ptr) {
    if (ptr == nullptr) return;
    
    ScopedDevice guard(device_id_);
    
    auto it = allocations_.find(static_cast<void*>(ptr));
    if (it != allocations_.end()) {
        allocated_bytes_ -= it->second;
        allocations_.erase(it);
        gpuFree(ptr);
    }
}

template<typename T>
T* MultiDeviceMemoryManager::allocate(int device_id, size_t count) {
    return workspace(device_id).allocate<T>(count);
}

template<typename T>
void MultiDeviceMemoryManager::deallocate(int device_id, T* ptr) {
    workspace(device_id).deallocate(ptr);
}

template<typename T>
void MultiDeviceMemoryManager::copy_between_devices(
    int dst_device, T* dst,
    int src_device, const T* src,
    size_t count) {
    
    if (count == 0) return;
    
    auto& dm = DeviceManager::instance();
    
    // Check if P2P is possible
    if (dm.can_access_peer(dst_device, src_device)) {
        // Use P2P copy
        ScopedDevice guard(dst_device);
        gpuMemcpyPeer(dst, dst_device, src, src_device, count * sizeof(T));
    } else {
        // Fall back to host-staged copy
        std::vector<T> host_buffer(count);
        
        // Copy from source device to host
        {
            ScopedDevice guard(src_device);
            gpuMemcpy(host_buffer.data(), src, count * sizeof(T), 
                       gpuMemcpyDeviceToHost);
        }
        
        // Copy from host to destination device
        {
            ScopedDevice guard(dst_device);
            gpuMemcpy(dst, host_buffer.data(), count * sizeof(T),
                       gpuMemcpyHostToDevice);
        }
    }
}

#endif  // LIBACCINT_USE_CUDA

}  // namespace libaccint::device
