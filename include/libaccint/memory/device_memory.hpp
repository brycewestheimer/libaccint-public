// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file device_memory.hpp
/// @brief Device memory management for GPU backends (CUDA)

#include <libaccint/config.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <cstddef>
#include <string>
#include <utility>

#if LIBACCINT_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace libaccint::memory {

// ============================================================================
// CUDA Error Handling
// ============================================================================

#if LIBACCINT_USE_CUDA

/**
 * @brief Exception for CUDA-specific errors
 *
 * Inherits from BackendException and includes file/line information
 * for better debugging.
 */
class CudaError : public BackendException {
public:
    /**
     * @brief Construct a CudaError with message and source location
     * @param message The error message (typically from cudaGetErrorString)
     * @param file Source file where the error occurred
     * @param line Line number where the error occurred
     */
    CudaError(const std::string& message, const char* file, int line)
        : BackendException("CUDA",
                           message + " (at " + file + ":" + std::to_string(line) + ")"),
          file_(file),
          line_(line) {}

    /// Get the source file where the error occurred
    [[nodiscard]] const char* file() const noexcept { return file_; }

    /// Get the line number where the error occurred
    [[nodiscard]] int line() const noexcept { return line_; }

private:
    const char* file_;
    int line_;
};

/**
 * @brief Check a CUDA API call and throw CudaError on failure
 *
 * Usage:
 * @code
 *     LIBACCINT_CUDA_CHECK(cudaMalloc(&ptr, size));
 * @endcode
 */
#define LIBACCINT_CUDA_CHECK(call)                                              \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            throw libaccint::memory::CudaError(                                 \
                cudaGetErrorString(err),                                        \
                __FILE__, __LINE__);                                            \
        }                                                                       \
    } while (0)

#endif  // LIBACCINT_USE_CUDA

// ============================================================================
// DeviceMemoryManager
// ============================================================================

#if LIBACCINT_USE_CUDA

/**
 * @brief Low-level device memory management utilities
 *
 * Provides static methods for allocating, deallocating, and copying
 * memory between host and device. All CUDA API calls are checked for
 * errors and throw CudaError on failure.
 *
 * For most use cases, prefer DeviceBuffer<T> or PinnedBuffer<T> which
 * provide RAII-based memory management.
 */
class DeviceMemoryManager {
public:
    // ---- Device Memory ----

    /**
     * @brief Allocate device memory
     * @tparam T Element type
     * @param count Number of elements to allocate
     * @return Pointer to allocated device memory
     * @throws CudaError on allocation failure
     */
    template <typename T>
    [[nodiscard]] static T* allocate_device(size_t count) {
        if (count == 0) {
            return nullptr;
        }
        T* ptr = nullptr;
        LIBACCINT_CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
#ifndef NDEBUG
        increment_device_allocations();
#endif
        return ptr;
    }

    /**
     * @brief Free device memory
     * @tparam T Element type
     * @param ptr Pointer to device memory (may be null)
     */
    template <typename T>
    static void deallocate_device(T* ptr) {
        if (ptr) {
            LIBACCINT_CUDA_CHECK(cudaFree(ptr));
#ifndef NDEBUG
            decrement_device_allocations();
#endif
        }
    }

    // ---- Pinned Host Memory ----

    /**
     * @brief Allocate page-locked (pinned) host memory
     *
     * Pinned memory enables faster async transfers between host and device.
     *
     * @tparam T Element type
     * @param count Number of elements to allocate
     * @return Pointer to allocated pinned host memory
     * @throws CudaError on allocation failure
     */
    template <typename T>
    [[nodiscard]] static T* allocate_pinned(size_t count) {
        if (count == 0) {
            return nullptr;
        }
        T* ptr = nullptr;
        LIBACCINT_CUDA_CHECK(cudaMallocHost(&ptr, count * sizeof(T)));
#ifndef NDEBUG
        increment_pinned_allocations();
#endif
        return ptr;
    }

    /**
     * @brief Free pinned host memory
     * @tparam T Element type
     * @param ptr Pointer to pinned host memory (may be null)
     */
    template <typename T>
    static void deallocate_pinned(T* ptr) {
        if (ptr) {
            LIBACCINT_CUDA_CHECK(cudaFreeHost(ptr));
#ifndef NDEBUG
            decrement_pinned_allocations();
#endif
        }
    }

    // ---- Memory Transfers ----

    /**
     * @brief Asynchronously copy data from host to device
     * @tparam T Element type
     * @param dst Device destination pointer
     * @param src Host source pointer
     * @param count Number of elements to copy
     * @param stream CUDA stream for async operation (0 = default stream)
     */
    template <typename T>
    static void copy_to_device(T* dst, const T* src, size_t count,
                               cudaStream_t stream = nullptr) {
        if (count == 0) return;
        LIBACCINT_CUDA_CHECK(cudaMemcpyAsync(
            dst, src, count * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    /**
     * @brief Asynchronously copy data from device to host
     * @tparam T Element type
     * @param dst Host destination pointer
     * @param src Device source pointer
     * @param count Number of elements to copy
     * @param stream CUDA stream for async operation (0 = default stream)
     */
    template <typename T>
    static void copy_to_host(T* dst, const T* src, size_t count,
                             cudaStream_t stream = nullptr) {
        if (count == 0) return;
        LIBACCINT_CUDA_CHECK(cudaMemcpyAsync(
            dst, src, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    /**
     * @brief Asynchronously copy data within device memory
     * @tparam T Element type
     * @param dst Device destination pointer
     * @param src Device source pointer
     * @param count Number of elements to copy
     * @param stream CUDA stream for async operation (0 = default stream)
     */
    template <typename T>
    static void copy_device_to_device(T* dst, const T* src, size_t count,
                                      cudaStream_t stream = nullptr) {
        if (count == 0) return;
        LIBACCINT_CUDA_CHECK(cudaMemcpyAsync(
            dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    }

    /**
     * @brief Synchronize the device (wait for all operations to complete)
     */
    static void synchronize() {
        LIBACCINT_CUDA_CHECK(cudaDeviceSynchronize());
    }

    /**
     * @brief Synchronize a specific stream
     * @param stream The CUDA stream to synchronize
     */
    static void synchronize_stream(cudaStream_t stream) {
        LIBACCINT_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

#ifndef NDEBUG
    // ---- Debug Statistics ----

    /// Get current number of active device allocations
    [[nodiscard]] static size_t active_device_allocations();

    /// Get current number of active pinned allocations
    [[nodiscard]] static size_t active_pinned_allocations();

private:
    static void increment_device_allocations();
    static void decrement_device_allocations();
    static void increment_pinned_allocations();
    static void decrement_pinned_allocations();
#endif
};

// ============================================================================
// DeviceBuffer<T> - RAII wrapper for device memory
// ============================================================================

/**
 * @brief RAII wrapper for device memory
 *
 * Automatically allocates device memory on construction and frees it
 * on destruction. Move-only to prevent double-free.
 *
 * @tparam T Element type stored in the buffer
 *
 * Example usage:
 * @code
 *     // Allocate 1000 doubles on device
 *     DeviceBuffer<double> buffer(1000);
 *
 *     // Upload data from host
 *     std::vector<double> host_data(1000, 1.0);
 *     buffer.upload(host_data.data(), host_data.size());
 *
 *     // Use buffer.data() in kernel calls
 *     my_kernel<<<blocks, threads>>>(buffer.data(), buffer.size());
 *
 *     // Download results
 *     buffer.download(host_data.data(), host_data.size());
 *     // Buffer automatically freed when it goes out of scope
 * @endcode
 */
template <typename T>
class DeviceBuffer {
public:
    /// Construct an empty buffer
    DeviceBuffer() noexcept : ptr_(nullptr), size_(0) {}

    /**
     * @brief Construct a buffer with the given capacity
     * @param count Number of elements to allocate
     * @throws CudaError on allocation failure
     */
    explicit DeviceBuffer(size_t count)
        : ptr_(DeviceMemoryManager::allocate_device<T>(count)), size_(count) {}

    /// Destructor - frees device memory
    ~DeviceBuffer() {
        if (ptr_) {
            DeviceMemoryManager::deallocate_device(ptr_);
        }
    }

    // Move constructor
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    // Move assignment
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                DeviceMemoryManager::deallocate_device(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Delete copy operations
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // ---- Accessors ----

    /// Get pointer to device memory
    [[nodiscard]] T* data() noexcept { return ptr_; }

    /// Get const pointer to device memory
    [[nodiscard]] const T* data() const noexcept { return ptr_; }

    /// Get number of elements in the buffer
    [[nodiscard]] size_t size() const noexcept { return size_; }

    /// Check if the buffer is empty
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /// Get size in bytes
    [[nodiscard]] size_t size_bytes() const noexcept { return size_ * sizeof(T); }

    // ---- Transfer Operations ----

    /**
     * @brief Upload data from host to this device buffer
     * @param host_data Pointer to host data
     * @param count Number of elements to copy (must not exceed buffer size)
     * @param stream CUDA stream for async operation (0 = default stream)
     */
    void upload(const T* host_data, size_t count, cudaStream_t stream = nullptr) {
        DeviceMemoryManager::copy_to_device(ptr_, host_data, count, stream);
    }

    /**
     * @brief Download data from this device buffer to host
     * @param host_data Pointer to host destination
     * @param count Number of elements to copy (must not exceed buffer size)
     * @param stream CUDA stream for async operation (0 = default stream)
     */
    void download(T* host_data, size_t count, cudaStream_t stream = nullptr) const {
        DeviceMemoryManager::copy_to_host(host_data, ptr_, count, stream);
    }

    /**
     * @brief Release ownership of the pointer
     * @return The device pointer (caller is responsible for freeing)
     */
    [[nodiscard]] T* release() noexcept {
        T* tmp = ptr_;
        ptr_ = nullptr;
        size_ = 0;
        return tmp;
    }

    /**
     * @brief Reset the buffer, optionally with a new size
     * @param count New size (0 to just free current memory)
     */
    void reset(size_t count = 0) {
        if (ptr_) {
            DeviceMemoryManager::deallocate_device(ptr_);
        }
        if (count > 0) {
            ptr_ = DeviceMemoryManager::allocate_device<T>(count);
            size_ = count;
        } else {
            ptr_ = nullptr;
            size_ = 0;
        }
    }

private:
    T* ptr_;
    size_t size_;
};

// ============================================================================
// PinnedBuffer<T> - RAII wrapper for pinned host memory
// ============================================================================

/**
 * @brief RAII wrapper for pinned (page-locked) host memory
 *
 * Pinned memory enables faster and truly asynchronous transfers between
 * host and device. Use this for staging buffers when performance is critical.
 *
 * @tparam T Element type stored in the buffer
 *
 * Example usage:
 * @code
 *     // Allocate 1000 doubles of pinned memory
 *     PinnedBuffer<double> pinned(1000);
 *
 *     // Fill with data
 *     for (size_t i = 0; i < 1000; ++i) {
 *         pinned.data()[i] = static_cast<double>(i);
 *     }
 *
 *     // Use for async transfers
 *     DeviceBuffer<double> device(1000);
 *     cudaStream_t stream;
 *     cudaStreamCreate(&stream);
 *     device.upload(pinned.data(), 1000, stream);
 *     // Transfer happens asynchronously while CPU continues
 * @endcode
 */
template <typename T>
class PinnedBuffer {
public:
    /// Construct an empty buffer
    PinnedBuffer() noexcept : ptr_(nullptr), size_(0) {}

    /**
     * @brief Construct a buffer with the given capacity
     * @param count Number of elements to allocate
     * @throws CudaError on allocation failure
     */
    explicit PinnedBuffer(size_t count)
        : ptr_(DeviceMemoryManager::allocate_pinned<T>(count)), size_(count) {}

    /// Destructor - frees pinned memory
    ~PinnedBuffer() {
        if (ptr_) {
            DeviceMemoryManager::deallocate_pinned(ptr_);
        }
    }

    // Move constructor
    PinnedBuffer(PinnedBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    // Move assignment
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                DeviceMemoryManager::deallocate_pinned(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Delete copy operations
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    // ---- Accessors ----

    /// Get pointer to pinned memory
    [[nodiscard]] T* data() noexcept { return ptr_; }

    /// Get const pointer to pinned memory
    [[nodiscard]] const T* data() const noexcept { return ptr_; }

    /// Get number of elements in the buffer
    [[nodiscard]] size_t size() const noexcept { return size_; }

    /// Check if the buffer is empty
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /// Get size in bytes
    [[nodiscard]] size_t size_bytes() const noexcept { return size_ * sizeof(T); }

    // ---- Transfer Operations ----

    /**
     * @brief Upload data from this pinned buffer to a device buffer
     * @param device_ptr Pointer to device memory
     * @param count Number of elements to copy
     * @param stream CUDA stream for async operation
     */
    void upload_to_device(T* device_ptr, size_t count,
                          cudaStream_t stream = nullptr) const {
        DeviceMemoryManager::copy_to_device(device_ptr, ptr_, count, stream);
    }

    /**
     * @brief Download data from a device buffer to this pinned buffer
     * @param device_ptr Pointer to device memory
     * @param count Number of elements to copy
     * @param stream CUDA stream for async operation
     */
    void download_from_device(const T* device_ptr, size_t count,
                              cudaStream_t stream = nullptr) {
        DeviceMemoryManager::copy_to_host(ptr_, device_ptr, count, stream);
    }

    /**
     * @brief Release ownership of the pointer
     * @return The pinned pointer (caller is responsible for freeing)
     */
    [[nodiscard]] T* release() noexcept {
        T* tmp = ptr_;
        ptr_ = nullptr;
        size_ = 0;
        return tmp;
    }

    /**
     * @brief Reset the buffer, optionally with a new size
     * @param count New size (0 to just free current memory)
     */
    void reset(size_t count = 0) {
        if (ptr_) {
            DeviceMemoryManager::deallocate_pinned(ptr_);
        }
        if (count > 0) {
            ptr_ = DeviceMemoryManager::allocate_pinned<T>(count);
            size_ = count;
        } else {
            ptr_ = nullptr;
            size_ = 0;
        }
    }

private:
    T* ptr_;
    size_t size_;
};

#endif  // LIBACCINT_USE_CUDA

}  // namespace libaccint::memory
