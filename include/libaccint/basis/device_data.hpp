// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file device_data.hpp
/// @brief Device-side data structures for shell batching
///
/// Provides structures and utilities for uploading ShellSet data to GPU memory
/// in Structure-of-Arrays (SoA) format for efficient kernel access.

#include <libaccint/config.hpp>
#include <libaccint/basis/shell_set.hpp>

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <mutex>

#if LIBACCINT_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace libaccint::basis {

#if LIBACCINT_USE_CUDA

// ============================================================================
// ShellSetDeviceData - Device pointers for a single ShellSet
// ============================================================================

/**
 * @brief Device-side data structure for a ShellSet
 *
 * Contains device pointers to SoA-formatted shell data suitable for
 * efficient GPU kernel access. All data is laid out contiguously:
 *   - exponents[i * K + k] is the k-th exponent of the i-th shell
 *   - coefficients[i * K + k] is the k-th coefficient of the i-th shell
 *
 * This structure is typically managed by ShellSetDeviceCache and should
 * not be created directly by users.
 */
struct ShellSetDeviceData {
    // Primitive data [n_shells * n_primitives]
    double* d_exponents{nullptr};
    double* d_coefficients{nullptr};

    // Shell centers [n_shells]
    double* d_centers_x{nullptr};
    double* d_centers_y{nullptr};
    double* d_centers_z{nullptr};

    // Indexing data [n_shells]
    int* d_shell_indices{nullptr};     ///< Original shell indices in basis
    int* d_atom_indices{nullptr};      ///< Atom each shell belongs to
    int* d_function_offsets{nullptr};  ///< Basis function offset for each shell

    // Metadata (stored on host for kernel configuration)
    int n_shells{0};
    int angular_momentum{0};
    int n_primitives{0};
    int n_functions_per_shell{0};

    /// @brief Check if device data is valid (has been uploaded)
    [[nodiscard]] bool valid() const noexcept {
        return d_exponents != nullptr && n_shells > 0;
    }

    /// @brief Total number of primitive values
    [[nodiscard]] size_t total_primitives() const noexcept {
        return static_cast<size_t>(n_shells) * n_primitives;
    }
};

// ============================================================================
// ShellSetPairDeviceData - Device data for a pair of ShellSets
// ============================================================================

/**
 * @brief Device-side data for a pair of ShellSets (bra and ket)
 *
 * Used for one-electron integral kernels that operate on pairs of shells.
 */
struct ShellSetPairDeviceData {
    ShellSetDeviceData bra;
    ShellSetDeviceData ket;

    /// @brief Total number of shell pairs
    [[nodiscard]] int n_pairs() const noexcept {
        return bra.n_shells * ket.n_shells;
    }

    /// @brief Check if both bra and ket data are valid
    [[nodiscard]] bool valid() const noexcept {
        return bra.valid() && ket.valid();
    }
};

// ============================================================================
// ShellSetQuartetDeviceData - Device data for a quartet of ShellSets
// ============================================================================

/**
 * @brief Device-side data for a quartet of ShellSets
 *
 * Used for two-electron integral kernels (ERIs) that operate on shell quartets.
 */
struct ShellSetQuartetDeviceData {
    ShellSetDeviceData a;  ///< First bra shell
    ShellSetDeviceData b;  ///< Second bra shell
    ShellSetDeviceData c;  ///< First ket shell
    ShellSetDeviceData d;  ///< Second ket shell

    /// @brief Total number of shell quartets
    [[nodiscard]] size_t n_quartets() const noexcept {
        return static_cast<size_t>(a.n_shells) * b.n_shells * c.n_shells * d.n_shells;
    }

    /// @brief Check if all four ShellSets are valid
    [[nodiscard]] bool valid() const noexcept {
        return a.valid() && b.valid() && c.valid() && d.valid();
    }
};

// ============================================================================
// ShellSetDeviceHandle - RAII handle for device ShellSet data
// ============================================================================

/**
 * @brief RAII handle for device ShellSet data
 *
 * Manages the lifetime of device memory for a ShellSet. When destroyed,
 * automatically frees all allocated device memory.
 */
class ShellSetDeviceHandle {
public:
    /// Default constructor (invalid handle)
    ShellSetDeviceHandle() = default;

    /// Constructor that allocates and uploads ShellSet data
    explicit ShellSetDeviceHandle(const ShellSet& shell_set, cudaStream_t stream = nullptr);

    /// Destructor - frees device memory
    ~ShellSetDeviceHandle();

    // Move-only
    ShellSetDeviceHandle(ShellSetDeviceHandle&& other) noexcept;
    ShellSetDeviceHandle& operator=(ShellSetDeviceHandle&& other) noexcept;

    ShellSetDeviceHandle(const ShellSetDeviceHandle&) = delete;
    ShellSetDeviceHandle& operator=(const ShellSetDeviceHandle&) = delete;

    /// Access the device data
    [[nodiscard]] const ShellSetDeviceData& data() const noexcept { return data_; }

    /// Check if handle is valid
    [[nodiscard]] bool valid() const noexcept { return data_.valid(); }

    /// Get the key for this ShellSet
    [[nodiscard]] ShellSetKey key() const noexcept {
        return ShellSetKey{data_.angular_momentum, data_.n_primitives};
    }

private:
    ShellSetDeviceData data_;

    void free_device_memory();
};

// ============================================================================
// ShellSetDeviceCache - Caching layer for device ShellSet data
// ============================================================================

/**
 * @brief Cache for device ShellSet data
 *
 * Maintains a cache of uploaded ShellSet data keyed by {AM, n_primitives}
 * to avoid redundant uploads. Thread-safe for concurrent access.
 *
 * Typical usage:
 * @code
 *     ShellSetDeviceCache cache;
 *     const ShellSetDeviceData& data = cache.get_or_upload(shell_set, stream);
 *     // Use data in kernel launch
 * @endcode
 */
class ShellSetDeviceCache {
public:
    ShellSetDeviceCache() = default;
    ~ShellSetDeviceCache() = default;

    // Non-copyable, non-movable
    ShellSetDeviceCache(const ShellSetDeviceCache&) = delete;
    ShellSetDeviceCache& operator=(const ShellSetDeviceCache&) = delete;

    /**
     * @brief Get or upload ShellSet data
     *
     * Returns cached data if available, otherwise uploads and caches.
     *
     * @param shell_set The ShellSet to get/upload
     * @param stream CUDA stream for async upload
     * @return Reference to device data (valid until cache is cleared)
     */
    [[nodiscard]] const ShellSetDeviceData& get_or_upload(
        const ShellSet& shell_set, cudaStream_t stream = nullptr);

    /**
     * @brief Clear all cached data
     *
     * Frees all device memory. Should be called before CUDA context cleanup.
     */
    void clear();

    /**
     * @brief Get number of cached entries
     */
    [[nodiscard]] size_t size() const;

    /**
     * @brief Check if a ShellSet is cached
     */
    [[nodiscard]] bool contains(const ShellSetKey& key) const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<ShellSetKey, std::unique_ptr<ShellSetDeviceHandle>> cache_;
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Upload a ShellSet to device memory
 *
 * @param shell_set The ShellSet to upload
 * @param stream CUDA stream for async upload
 * @return Populated ShellSetDeviceData (caller owns the pointers)
 */
ShellSetDeviceData upload_shell_set(const ShellSet& shell_set, cudaStream_t stream = nullptr);

/**
 * @brief Free device memory in a ShellSetDeviceData
 *
 * @param data The device data to free (pointers set to nullptr after)
 */
void free_shell_set_device_data(ShellSetDeviceData& data);

#endif  // LIBACCINT_USE_CUDA

}  // namespace libaccint::basis
