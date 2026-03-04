// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file backend.cpp
/// @brief CPU implementations of StreamHandle and device query functions

#include <libaccint/core/backend.hpp>

#include <stdexcept>
#include <thread>

namespace libaccint {

// ============================================================================
// detail::StreamImpl
// ============================================================================

detail::StreamImpl::~StreamImpl() {
    // CPU: nothing to destroy
    // GPU stream destruction is handled by device-specific code
#if LIBACCINT_USE_CUDA
    // cudaStreamDestroy handled in CUDA source files
#endif
}

// ============================================================================
// StreamHandle Implementation
// ============================================================================

StreamHandle StreamHandle::create(BackendType backend) {
    StreamHandle handle;
    handle.backend_ = backend;

    switch (backend) {
        case BackendType::CPU: {
            handle.impl_ = std::make_shared<detail::StreamImpl>();
            handle.impl_->backend = BackendType::CPU;
            break;
        }
        case BackendType::CUDA: {
#if LIBACCINT_USE_CUDA
            // CUDA stream creation delegated to device-specific code
            throw BackendError(BackendType::CUDA,
                "CUDA stream creation should use device-specific factory");
#else
            throw BackendError(BackendType::CUDA,
                "CUDA backend not available in this build");
#endif
        }
    }

    return handle;
}

void StreamHandle::synchronize() {
    if (!impl_) return;  // Default-constructed handle, nothing to sync

    switch (backend_) {
        case BackendType::CPU:
            // CPU is synchronous by default — no-op
            break;
        case BackendType::CUDA:
            // GPU synchronization handled by device-specific code
            break;
    }
}

#if LIBACCINT_USE_CUDA
void* StreamHandle::cuda_stream() const {
    if (!impl_ || backend_ != BackendType::CUDA) {
        throw BackendError(BackendType::CUDA,
            "Cannot get CUDA stream from non-CUDA StreamHandle");
    }
    return impl_->native_stream;
}
#endif

// ============================================================================
// Device Query Functions
// ============================================================================

DeviceInfo get_device_info(BackendType backend, int device_id) {
    switch (backend) {
        case BackendType::CPU: {
            DeviceInfo info;
            info.name = "CPU";
            info.compute_capability = 0;
            info.multiprocessor_count = static_cast<int>(
                std::thread::hardware_concurrency());
            info.max_threads_per_block = 1;
            info.warp_size = 1;
            return info;
        }
        case BackendType::CUDA:
            throw BackendError(BackendType::CUDA,
                "CUDA device query not available — use device-specific API");
    }
    throw std::runtime_error("Unknown backend type");
}

int get_device_count(BackendType backend) {
    switch (backend) {
        case BackendType::CPU:
            return 0;  // CPU has no "devices" in the GPU sense
        case BackendType::CUDA:
#if LIBACCINT_USE_CUDA
            // Delegated to CUDA runtime query in device-specific code
            return 0;
#else
            return 0;
#endif
    }
    return 0;
}

void set_device(BackendType backend, int device_id) {
    switch (backend) {
        case BackendType::CPU:
            // No-op for CPU
            break;
        case BackendType::CUDA:
#if LIBACCINT_USE_CUDA
            // Delegated to device-specific code
#else
            throw BackendError(BackendType::CUDA,
                "CUDA backend not available in this build");
#endif
            break;
    }
}

}  // namespace libaccint
