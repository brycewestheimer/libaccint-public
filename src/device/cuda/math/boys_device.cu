// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file boys_device.cu
/// @brief CUDA device-side Boys function implementation

#include "boys_device.cuh"
#include <libaccint/memory/device_memory.hpp>
#include <libaccint/math/boys_tables.hpp>

#include <atomic>
#include <mutex>

#if LIBACCINT_USE_CUDA

namespace libaccint::device::math {

// ============================================================================
// Global Memory for Chebyshev Coefficients
// ============================================================================

namespace {

/// Mutex for thread-safe initialization
std::mutex g_init_mutex;

/// Flag indicating if tables are initialized
std::atomic<bool> g_initialized{false};

/// Device memory pointer for Chebyshev coefficients
double* g_d_chebyshev_coeffs = nullptr;

/// Total size of Chebyshev coefficient table
constexpr size_t CHEB_TABLE_SIZE = BOYS_CHEB_N_INTERVALS_DEVICE
                                 * BOYS_CHEB_N_ORDERS_DEVICE
                                 * BOYS_CHEB_N_TERMS_DEVICE;

}  // anonymous namespace

// ============================================================================
// Initialization Implementation
// ============================================================================

double* boys_device_init(cudaStream_t stream) {
    // Fast path: already initialized
    if (g_initialized.load(std::memory_order_acquire)) {
        return g_d_chebyshev_coeffs;
    }

    // Slow path: need to initialize
    std::lock_guard<std::mutex> lock(g_init_mutex);

    // Double-check after acquiring lock
    if (g_initialized.load(std::memory_order_relaxed)) {
        return g_d_chebyshev_coeffs;
    }

    // Allocate device memory for Chebyshev coefficients
    g_d_chebyshev_coeffs = memory::DeviceMemoryManager::allocate_device<double>(CHEB_TABLE_SIZE);

    // Flatten the 3D coefficient table and upload
    // Layout: coefficients[interval][n][term] -> coeffs[interval * 31*12 + n * 12 + term]
    std::vector<double> flat_coeffs(CHEB_TABLE_SIZE);

    for (int interval = 0; interval < BOYS_CHEB_N_INTERVALS_DEVICE; ++interval) {
        for (int n = 0; n < BOYS_CHEB_N_ORDERS_DEVICE; ++n) {
            for (int term = 0; term < BOYS_CHEB_N_TERMS_DEVICE; ++term) {
                const size_t flat_idx = interval * (BOYS_CHEB_N_ORDERS_DEVICE * BOYS_CHEB_N_TERMS_DEVICE)
                                      + n * BOYS_CHEB_N_TERMS_DEVICE
                                      + term;
                flat_coeffs[flat_idx] = libaccint::math::detail::BOYS_CHEBYSHEV_COEFFICIENTS[interval][n][term];
            }
        }
    }

    // Upload to device
    memory::DeviceMemoryManager::copy_to_device(
        g_d_chebyshev_coeffs, flat_coeffs.data(), CHEB_TABLE_SIZE, stream);

    // Ensure upload completes before marking as initialized
    if (stream == nullptr) {
        memory::DeviceMemoryManager::synchronize();
    } else {
        memory::DeviceMemoryManager::synchronize_stream(stream);
    }

    g_initialized.store(true, std::memory_order_release);
    return g_d_chebyshev_coeffs;
}

void boys_device_cleanup() {
    std::lock_guard<std::mutex> lock(g_init_mutex);

    if (g_d_chebyshev_coeffs != nullptr) {
        memory::DeviceMemoryManager::deallocate_device(g_d_chebyshev_coeffs);
        g_d_chebyshev_coeffs = nullptr;
    }

    g_initialized.store(false, std::memory_order_release);
}

double* boys_device_get_coeffs() {
    if (!g_initialized.load(std::memory_order_acquire)) {
        boys_device_init();
    }
    return g_d_chebyshev_coeffs;
}

bool boys_device_is_initialized() {
    return g_initialized.load(std::memory_order_acquire);
}

// ============================================================================
// Test Kernel (for validation)
// ============================================================================

/// Kernel for testing Boys function on device
__global__ void boys_test_kernel(
    int n_max,
    const double* T_values,
    int n_T_values,
    double* results,
    const double* cheb_coeffs) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_T_values) return;

    const double T = T_values[idx];
    double* out = results + idx * (n_max + 1);

    boys_evaluate_array_device(n_max, T, out, cheb_coeffs);
}

/// Host function to launch the test kernel
void boys_device_test(
    int n_max,
    const double* d_T_values,
    int n_T_values,
    double* d_results,
    cudaStream_t stream) {

    const double* cheb_coeffs = boys_device_get_coeffs();

    const int block_size = 256;
    const int n_blocks = (n_T_values + block_size - 1) / block_size;

    boys_test_kernel<<<n_blocks, block_size, 0, stream>>>(
        n_max, d_T_values, n_T_values, d_results, cheb_coeffs);
}

}  // namespace libaccint::device::math

#endif  // LIBACCINT_USE_CUDA
