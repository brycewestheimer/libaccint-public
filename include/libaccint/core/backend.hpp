// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file backend.hpp
/// @brief Backend abstraction types for LibAccInt

#include <libaccint/config.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <stdexcept>

namespace libaccint {

// ============================================================================
// Backend Type
// ============================================================================

/// Available compute backends
enum class BackendType {
    CPU,   ///< Host CPU (vectorized)
    CUDA,  ///< NVIDIA CUDA
};

/// Convert BackendType to string
[[nodiscard]] inline constexpr std::string_view backend_name(BackendType backend) noexcept {
    switch (backend) {
        case BackendType::CPU:  return "CPU";
        case BackendType::CUDA: return "CUDA";
    }
    return "Unknown";
}

/// Check if a backend is a GPU backend
[[nodiscard]] inline constexpr bool is_gpu_backend(BackendType backend) noexcept {
    return backend == BackendType::CUDA;
}

/// Check if a backend is available at runtime
[[nodiscard]] inline bool is_backend_available(BackendType backend) noexcept {
    switch (backend) {
        case BackendType::CPU:
            return true;  // Always available
        case BackendType::CUDA:
            return has_cuda_backend();
    }
    return false;
}

// ============================================================================
// Stream Handle
// ============================================================================

/// Backend-specific stream implementation
namespace detail {
    /// @brief Concrete stream implementation for all backends
    ///
    /// For CPU: streams are synchronous (no-op).
    /// For GPU backends: wraps the native stream handle (conditionally compiled).
    struct StreamImpl {
        BackendType backend{BackendType::CPU};
#if LIBACCINT_USE_CUDA
        void* native_stream{nullptr};   ///< cudaStream_t (cast from void*)
#endif
        ~StreamImpl();
    };
}

/// Handle to an asynchronous execution stream
///
/// StreamHandle provides a unified interface for managing asynchronous
/// operations across different GPU backends. On CPU, operations are
/// synchronous by default.
class StreamHandle {
public:
    /// Create a synchronous (default) stream
    StreamHandle() = default;

    /// Create an asynchronous stream for the given backend
    static StreamHandle create(BackendType backend);

    /// Synchronize (wait for all operations on this stream to complete)
    void synchronize();

    /// Check if this is a valid (non-null) stream
    [[nodiscard]] bool valid() const noexcept { return impl_ != nullptr; }

    /// Get the backend type for this stream
    [[nodiscard]] BackendType backend() const noexcept { return backend_; }

#if LIBACCINT_USE_CUDA
    /// Get the underlying CUDA stream (CUDA backend only)
    [[nodiscard]] void* cuda_stream() const;
#endif

private:
    std::shared_ptr<detail::StreamImpl> impl_;
    BackendType backend_{BackendType::CPU};
};

// ============================================================================
// Backend Exception
// ============================================================================

/// Exception thrown for backend-related errors
class BackendError : public Exception {
public:
    BackendError(BackendType backend, const std::string& message)
        : Exception(std::string(backend_name(backend)) + ": " + message)
        , backend_(backend)
    {}

    [[nodiscard]] BackendType backend() const noexcept { return backend_; }

private:
    BackendType backend_;
};

// ============================================================================
// Device Info
// ============================================================================

/// Information about a compute device
struct DeviceInfo {
    std::string name;
    Size total_memory{0};       ///< Total device memory in bytes
    Size available_memory{0};   ///< Available device memory in bytes
    int compute_capability{0};  ///< SM version for CUDA
    int multiprocessor_count{0};
    int max_threads_per_block{0};
    int warp_size{0};
};

/// Query device information for a backend
[[nodiscard]] DeviceInfo get_device_info(BackendType backend, int device_id = 0);

/// Get the number of available devices for a backend
[[nodiscard]] int get_device_count(BackendType backend);

/// Set the active device for a backend
void set_device(BackendType backend, int device_id);

}  // namespace libaccint
