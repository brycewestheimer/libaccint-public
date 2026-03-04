// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file device_operator_data.cu
/// @brief CUDA implementation of device operator data management

#include <libaccint/operators/device_operator_data.hpp>
#include <libaccint/memory/device_memory.hpp>
#include <libaccint/utils/error_handling.hpp>

#if LIBACCINT_USE_CUDA

namespace libaccint::operators {

using memory::DeviceMemoryManager;

// ============================================================================
// Utility Functions
// ============================================================================

DevicePointChargeData upload_point_charges(
    std::span<const double> x,
    std::span<const double> y,
    std::span<const double> z,
    std::span<const double> charges,
    cudaStream_t stream) {

    DevicePointChargeData result;

    // Validate inputs
    if (x.size() != y.size() || y.size() != z.size() || z.size() != charges.size()) {
        throw InvalidArgumentException(
            "Point charge arrays must have the same size");
    }

    if (x.empty()) {
        return result;
    }

    const int n = static_cast<int>(x.size());
    result.n_charges = n;

    // Allocate device memory
    result.d_x = DeviceMemoryManager::allocate_device<double>(n);
    result.d_y = DeviceMemoryManager::allocate_device<double>(n);
    result.d_z = DeviceMemoryManager::allocate_device<double>(n);
    result.d_charges = DeviceMemoryManager::allocate_device<double>(n);

    // Upload data
    DeviceMemoryManager::copy_to_device(result.d_x, x.data(), n, stream);
    DeviceMemoryManager::copy_to_device(result.d_y, y.data(), n, stream);
    DeviceMemoryManager::copy_to_device(result.d_z, z.data(), n, stream);
    DeviceMemoryManager::copy_to_device(result.d_charges, charges.data(), n, stream);

    return result;
}

void free_point_charge_device_data(DevicePointChargeData& data) {
    if (data.d_x) {
        DeviceMemoryManager::deallocate_device(data.d_x);
        data.d_x = nullptr;
    }
    if (data.d_y) {
        DeviceMemoryManager::deallocate_device(data.d_y);
        data.d_y = nullptr;
    }
    if (data.d_z) {
        DeviceMemoryManager::deallocate_device(data.d_z);
        data.d_z = nullptr;
    }
    if (data.d_charges) {
        DeviceMemoryManager::deallocate_device(data.d_charges);
        data.d_charges = nullptr;
    }
    data.n_charges = 0;
}

// ============================================================================
// PointChargeDeviceHandle Implementation
// ============================================================================

PointChargeDeviceHandle::PointChargeDeviceHandle(
    std::span<const double> x,
    std::span<const double> y,
    std::span<const double> z,
    std::span<const double> charges,
    cudaStream_t stream) {

    data_ = upload_point_charges(x, y, z, charges, stream);
}

PointChargeDeviceHandle::~PointChargeDeviceHandle() {
    free_device_memory();
}

PointChargeDeviceHandle::PointChargeDeviceHandle(PointChargeDeviceHandle&& other) noexcept
    : data_(other.data_) {
    other.data_ = DevicePointChargeData{};
}

PointChargeDeviceHandle& PointChargeDeviceHandle::operator=(PointChargeDeviceHandle&& other) noexcept {
    if (this != &other) {
        free_device_memory();
        data_ = other.data_;
        other.data_ = DevicePointChargeData{};
    }
    return *this;
}

void PointChargeDeviceHandle::free_device_memory() {
    free_point_charge_device_data(data_);
}

}  // namespace libaccint::operators

#endif  // LIBACCINT_USE_CUDA
