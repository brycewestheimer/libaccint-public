// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file device_operator_data.hpp
/// @brief Device-side data structures for operator parameters
///
/// Provides structures and utilities for uploading operator parameters
/// (e.g., nuclear positions and charges) to GPU memory.

#include <libaccint/config.hpp>

#include <cstddef>
#include <span>
#include <vector>

#if LIBACCINT_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace libaccint::operators {

#if LIBACCINT_USE_CUDA

// ============================================================================
// DevicePointChargeData - Device data for point charges (nuclei)
// ============================================================================

/**
 * @brief Device-side data for a set of point charges
 *
 * Used primarily for nuclear attraction integrals. Stores nuclear positions
 * and charges in SoA format for coalesced memory access.
 */
struct DevicePointChargeData {
    // Positions [n_charges]
    double* d_x{nullptr};
    double* d_y{nullptr};
    double* d_z{nullptr};

    // Charges [n_charges]
    double* d_charges{nullptr};

    // Metadata
    int n_charges{0};

    /// @brief Check if device data is valid
    [[nodiscard]] bool valid() const noexcept {
        return d_x != nullptr && n_charges > 0;
    }
};

// ============================================================================
// PointChargeDeviceHandle - RAII handle for device point charge data
// ============================================================================

/**
 * @brief RAII handle for device point charge data
 *
 * Manages the lifetime of device memory for point charges. Automatically
 * frees all allocated device memory when destroyed.
 *
 * Example usage:
 * @code
 *     std::vector<double> x = {0.0, 1.5};
 *     std::vector<double> y = {0.0, 0.0};
 *     std::vector<double> z = {0.0, 0.0};
 *     std::vector<double> charges = {8.0, 1.0};  // O and H
 *
 *     PointChargeDeviceHandle handle(x, y, z, charges);
 *     // Use handle.data() in kernel launch
 * @endcode
 */
class PointChargeDeviceHandle {
public:
    /// Default constructor (invalid handle)
    PointChargeDeviceHandle() = default;

    /**
     * @brief Constructor that allocates and uploads point charge data
     *
     * @param x X coordinates of point charges
     * @param y Y coordinates of point charges
     * @param z Z coordinates of point charges
     * @param charges Charge values
     * @param stream CUDA stream for async upload
     * @throws InvalidArgumentException if arrays have different sizes
     */
    PointChargeDeviceHandle(
        std::span<const double> x,
        std::span<const double> y,
        std::span<const double> z,
        std::span<const double> charges,
        cudaStream_t stream = nullptr);

    /// Destructor - frees device memory
    ~PointChargeDeviceHandle();

    // Move-only
    PointChargeDeviceHandle(PointChargeDeviceHandle&& other) noexcept;
    PointChargeDeviceHandle& operator=(PointChargeDeviceHandle&& other) noexcept;

    PointChargeDeviceHandle(const PointChargeDeviceHandle&) = delete;
    PointChargeDeviceHandle& operator=(const PointChargeDeviceHandle&) = delete;

    /// Access the device data
    [[nodiscard]] const DevicePointChargeData& data() const noexcept { return data_; }

    /// Check if handle is valid
    [[nodiscard]] bool valid() const noexcept { return data_.valid(); }

    /// Get number of point charges
    [[nodiscard]] int n_charges() const noexcept { return data_.n_charges; }

private:
    DevicePointChargeData data_;

    void free_device_memory();
};

// ============================================================================
// DeviceGaussianGeminalData - Device data for Gaussian geminal operators
// ============================================================================

/**
 * @brief Device-side data for Gaussian geminal operator parameters
 *
 * Used for attenuated Coulomb operators like Erf(ω r₁₂)/r₁₂.
 * Currently stores just the screening parameter(s).
 */
struct DeviceGaussianGeminalData {
    // Geminal exponents for the operator
    double* d_exponents{nullptr};

    // Number of geminal terms
    int n_terms{0};

    /// @brief Attenuation parameter (omega for erf/erfc)
    double omega{0.0};

    /// @brief Check if device data is valid
    [[nodiscard]] bool valid() const noexcept {
        return n_terms >= 0;  // omega-only operators have n_terms = 0
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Upload point charge data to device memory
 *
 * @param x X coordinates of point charges
 * @param y Y coordinates of point charges
 * @param z Z coordinates of point charges
 * @param charges Charge values
 * @param stream CUDA stream for async upload
 * @return Populated DevicePointChargeData (caller owns the pointers)
 */
DevicePointChargeData upload_point_charges(
    std::span<const double> x,
    std::span<const double> y,
    std::span<const double> z,
    std::span<const double> charges,
    cudaStream_t stream = nullptr);

/**
 * @brief Free device memory in a DevicePointChargeData
 *
 * @param data The device data to free (pointers set to nullptr after)
 */
void free_point_charge_device_data(DevicePointChargeData& data);

#endif  // LIBACCINT_USE_CUDA

}  // namespace libaccint::operators
