// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file device_memory.cu
/// @brief CUDA device memory management implementation

#include <libaccint/memory/device_memory.hpp>

#if LIBACCINT_USE_CUDA

#include <atomic>

namespace libaccint::memory {

#ifndef NDEBUG
// ============================================================================
// Debug Build: Memory Tracking
// ============================================================================

namespace {
    /// Counter for active device memory allocations
    std::atomic<size_t> g_device_allocation_count{0};

    /// Counter for active pinned memory allocations
    std::atomic<size_t> g_pinned_allocation_count{0};
}  // anonymous namespace

size_t DeviceMemoryManager::active_device_allocations() {
    return g_device_allocation_count.load(std::memory_order_relaxed);
}

size_t DeviceMemoryManager::active_pinned_allocations() {
    return g_pinned_allocation_count.load(std::memory_order_relaxed);
}

void DeviceMemoryManager::increment_device_allocations() {
    g_device_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void DeviceMemoryManager::decrement_device_allocations() {
    g_device_allocation_count.fetch_sub(1, std::memory_order_relaxed);
}

void DeviceMemoryManager::increment_pinned_allocations() {
    g_pinned_allocation_count.fetch_add(1, std::memory_order_relaxed);
}

void DeviceMemoryManager::decrement_pinned_allocations() {
    g_pinned_allocation_count.fetch_sub(1, std::memory_order_relaxed);
}

#endif  // NDEBUG

}  // namespace libaccint::memory

#endif  // LIBACCINT_USE_CUDA
