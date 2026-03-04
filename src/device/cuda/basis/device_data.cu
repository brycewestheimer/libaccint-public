// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file device_data.cu
/// @brief CUDA implementation of device ShellSet data management

#include <libaccint/basis/device_data.hpp>
#include <libaccint/memory/device_memory.hpp>
#include <libaccint/math/normalization.hpp>

#if LIBACCINT_USE_CUDA

namespace libaccint::basis {

using memory::DeviceMemoryManager;

// ============================================================================
// Utility Functions
// ============================================================================

ShellSetDeviceData upload_shell_set(const ShellSet& shell_set, cudaStream_t stream) {
    ShellSetDeviceData result;

    if (shell_set.empty()) {
        return result;
    }

    // Get SoA data from the ShellSet
    const ShellSetDataSoA& soa = shell_set.soa_data();

    const int n_shells = static_cast<int>(shell_set.n_shells());
    const int n_prims = shell_set.n_primitives_per_shell();
    const size_t total_prims = static_cast<size_t>(n_shells) * n_prims;

    // Fill metadata
    result.n_shells = n_shells;
    result.angular_momentum = shell_set.angular_momentum();
    result.n_primitives = n_prims;
    result.n_functions_per_shell = shell_set.n_functions_per_shell();

    // Allocate device memory for primitives
    result.d_exponents = DeviceMemoryManager::allocate_device<double>(total_prims);
    result.d_coefficients = DeviceMemoryManager::allocate_device<double>(total_prims);

    // Allocate device memory for centers
    result.d_centers_x = DeviceMemoryManager::allocate_device<double>(n_shells);
    result.d_centers_y = DeviceMemoryManager::allocate_device<double>(n_shells);
    result.d_centers_z = DeviceMemoryManager::allocate_device<double>(n_shells);

    // Allocate device memory for indices
    result.d_shell_indices = DeviceMemoryManager::allocate_device<int>(n_shells);
    result.d_atom_indices = DeviceMemoryManager::allocate_device<int>(n_shells);
    result.d_function_offsets = DeviceMemoryManager::allocate_device<int>(n_shells);

    // Upload primitive data
    DeviceMemoryManager::copy_to_device(result.d_exponents, soa.exponents.data(), total_prims, stream);
    DeviceMemoryManager::copy_to_device(result.d_coefficients, soa.coefficients.data(), total_prims, stream);

    // Upload center data
    DeviceMemoryManager::copy_to_device(result.d_centers_x, soa.center_x.data(), n_shells, stream);
    DeviceMemoryManager::copy_to_device(result.d_centers_y, soa.center_y.data(), n_shells, stream);
    DeviceMemoryManager::copy_to_device(result.d_centers_z, soa.center_z.data(), n_shells, stream);

    // Convert Index to int for device memory (Index is typically size_t)
    std::vector<int> shell_indices(n_shells);
    std::vector<int> atom_indices(n_shells);
    std::vector<int> function_offsets(n_shells);

    for (int i = 0; i < n_shells; ++i) {
        shell_indices[i] = static_cast<int>(soa.shell_indices[i]);
        atom_indices[i] = static_cast<int>(soa.atom_indices[i]);
        function_offsets[i] = static_cast<int>(soa.function_offsets[i]);
    }

    DeviceMemoryManager::copy_to_device(result.d_shell_indices, shell_indices.data(), n_shells, stream);
    DeviceMemoryManager::copy_to_device(result.d_atom_indices, atom_indices.data(), n_shells, stream);
    DeviceMemoryManager::copy_to_device(result.d_function_offsets, function_offsets.data(), n_shells, stream);

    return result;
}

void free_shell_set_device_data(ShellSetDeviceData& data) {
    if (data.d_exponents) {
        DeviceMemoryManager::deallocate_device(data.d_exponents);
        data.d_exponents = nullptr;
    }
    if (data.d_coefficients) {
        DeviceMemoryManager::deallocate_device(data.d_coefficients);
        data.d_coefficients = nullptr;
    }
    if (data.d_centers_x) {
        DeviceMemoryManager::deallocate_device(data.d_centers_x);
        data.d_centers_x = nullptr;
    }
    if (data.d_centers_y) {
        DeviceMemoryManager::deallocate_device(data.d_centers_y);
        data.d_centers_y = nullptr;
    }
    if (data.d_centers_z) {
        DeviceMemoryManager::deallocate_device(data.d_centers_z);
        data.d_centers_z = nullptr;
    }
    if (data.d_shell_indices) {
        DeviceMemoryManager::deallocate_device(data.d_shell_indices);
        data.d_shell_indices = nullptr;
    }
    if (data.d_atom_indices) {
        DeviceMemoryManager::deallocate_device(data.d_atom_indices);
        data.d_atom_indices = nullptr;
    }
    if (data.d_function_offsets) {
        DeviceMemoryManager::deallocate_device(data.d_function_offsets);
        data.d_function_offsets = nullptr;
    }

    data.n_shells = 0;
    data.angular_momentum = 0;
    data.n_primitives = 0;
    data.n_functions_per_shell = 0;
}

// ============================================================================
// ShellSetDeviceHandle Implementation
// ============================================================================

ShellSetDeviceHandle::ShellSetDeviceHandle(const ShellSet& shell_set, cudaStream_t stream) {
    data_ = upload_shell_set(shell_set, stream);
}

ShellSetDeviceHandle::~ShellSetDeviceHandle() {
    free_device_memory();
}

ShellSetDeviceHandle::ShellSetDeviceHandle(ShellSetDeviceHandle&& other) noexcept
    : data_(other.data_) {
    // Clear the source
    other.data_ = ShellSetDeviceData{};
}

ShellSetDeviceHandle& ShellSetDeviceHandle::operator=(ShellSetDeviceHandle&& other) noexcept {
    if (this != &other) {
        free_device_memory();
        data_ = other.data_;
        other.data_ = ShellSetDeviceData{};
    }
    return *this;
}

void ShellSetDeviceHandle::free_device_memory() {
    free_shell_set_device_data(data_);
}

// ============================================================================
// ShellSetDeviceCache Implementation
// ============================================================================

const ShellSetDeviceData& ShellSetDeviceCache::get_or_upload(
    const ShellSet& shell_set, cudaStream_t stream) {

    const ShellSetKey key = shell_set.key();

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second->data();
    }

    // Not found - upload and cache
    auto handle = std::make_unique<ShellSetDeviceHandle>(shell_set, stream);
    const ShellSetDeviceData& data = handle->data();
    cache_[key] = std::move(handle);

    return data;
}

void ShellSetDeviceCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

size_t ShellSetDeviceCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

bool ShellSetDeviceCache::contains(const ShellSetKey& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.find(key) != cache_.end();
}

}  // namespace libaccint::basis

#endif  // LIBACCINT_USE_CUDA
