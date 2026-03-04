// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file device_manager.hpp
/// @brief Multi-GPU device discovery, selection, and management
///
/// Provides a unified interface for discovering and managing multiple GPUs
/// across CUDA backends. The DeviceManager handles device enumeration,
/// selection, and provides topology information for optimal work distribution.

#include <libaccint/config.hpp>
#include <libaccint/device/device_properties.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace libaccint::device {

/// @brief Error class for device management failures
class DeviceError : public std::runtime_error {
public:
    explicit DeviceError(const std::string& message)
        : std::runtime_error("DeviceError: " + message) {}
    
    DeviceError(int device_id, const std::string& message)
        : std::runtime_error("DeviceError (device " + std::to_string(device_id) + "): " + message) {}
};

/// @brief Central manager for multi-GPU device discovery and selection
///
/// DeviceManager provides a singleton-style interface for managing GPU devices.
/// It handles:
/// - Device enumeration and property queries
/// - Device selection for multi-GPU execution
/// - Peer-to-peer access configuration
/// - Topology-aware device grouping
///
/// @note Thread safety: Device selection (set_active_devices, set_all_devices,
///       set_current_device) must be performed before any computation begins
///       and must not be called concurrently with computation or with each other.
///       Read-only queries (device_count, available_devices, etc.) are safe to
///       call concurrently after initialization.
///
/// Usage:
/// @code
///   auto& mgr = DeviceManager::instance();
///   
///   // Discover and print all devices
///   for (const auto& props : mgr.available_devices()) {
///       std::cout << props.description() << "\n";
///   }
///   
///   // Select devices for multi-GPU execution
///   mgr.set_active_devices({0, 1, 2});
///   
///   // Or select all available
///   mgr.set_all_devices();
/// @endcode
class DeviceManager {
public:
    /// @brief Get the singleton DeviceManager instance
    static DeviceManager& instance();
    
    // Non-copyable, non-movable singleton
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
    DeviceManager(DeviceManager&&) = delete;
    DeviceManager& operator=(DeviceManager&&) = delete;
    
    // =========================================================================
    // Device Discovery
    // =========================================================================
    
    /// @brief Get the total number of visible GPU devices
    /// @note Respects CUDA_VISIBLE_DEVICES
    [[nodiscard]] int device_count() const noexcept { return device_count_; }
    
    /// @brief Check if any GPU devices are available
    [[nodiscard]] bool has_devices() const noexcept { return device_count_ > 0; }
    
    /// @brief Get properties for a specific device
    /// @param device_id Device index (0 to device_count-1)
    /// @throws DeviceError if device_id is invalid
    [[nodiscard]] const DeviceProperties& get_device_properties(int device_id) const;
    
    /// @brief Get properties for all visible devices
    [[nodiscard]] const std::vector<DeviceProperties>& available_devices() const noexcept {
        return device_properties_;
    }
    
    /// @brief Refresh device properties (e.g., free memory)
    /// @param device_id Device to refresh, or -1 for all devices
    void refresh_properties(int device_id = -1);
    
    // =========================================================================
    // Device Selection
    // =========================================================================
    
    /// @brief Set the active devices for multi-GPU execution
    /// @param device_ids Ordered list of device IDs to use
    /// @throws DeviceError if any device_id is invalid
    /// @warning Not thread-safe. Must be called before computation begins and
    ///          must not be called concurrently with other device selection methods.
    void set_active_devices(const std::vector<int>& device_ids);

    /// @brief Select all available devices for multi-GPU execution
    /// @warning Not thread-safe. Must be called before computation begins and
    ///          must not be called concurrently with other device selection methods.
    void set_all_devices();
    
    /// @brief Select a single device (equivalent to cudaSetDevice)
    /// @param device_id The device to make current
    /// @throws DeviceError if device_id is invalid
    void set_current_device(int device_id);
    
    /// @brief Get the currently active device
    [[nodiscard]] int current_device() const;
    
    /// @brief Get the list of active (selected) device IDs
    [[nodiscard]] const std::vector<int>& active_devices() const noexcept {
        return active_devices_;
    }
    
    /// @brief Get the number of active devices
    [[nodiscard]] int active_device_count() const noexcept {
        return static_cast<int>(active_devices_.size());
    }
    
    // =========================================================================
    // Peer-to-Peer Access
    // =========================================================================
    
    /// @brief Check if peer-to-peer access is possible between two devices
    /// @param source_device Device that will access the other's memory
    /// @param target_device Device whose memory will be accessed
    /// @return true if P2P access is possible
    [[nodiscard]] bool can_access_peer(int source_device, int target_device) const;
    
    /// @brief Enable peer-to-peer access between two devices
    /// @param source_device Device that will access the other's memory
    /// @param target_device Device whose memory will be accessed
    /// @throws DeviceError if P2P cannot be enabled
    void enable_peer_access(int source_device, int target_device);
    
    /// @brief Disable peer-to-peer access between two devices
    void disable_peer_access(int source_device, int target_device);
    
    /// @brief Enable P2P access between all active devices where possible
    void enable_all_peer_access();
    
    /// @brief Disable all P2P access between active devices
    void disable_all_peer_access();
    
    /// @brief Get peer access matrix for all devices
    /// @return Matrix of PeerAccessInfo for all device pairs
    [[nodiscard]] std::vector<PeerAccessInfo> get_peer_access_matrix() const;
    
    // =========================================================================
    // Topology Queries
    // =========================================================================
    
    /// @brief Check if two devices are connected via NVLink
    [[nodiscard]] bool is_nvlink_connected(int device1, int device2) const;
    
    /// @brief Get devices grouped by NUMA node for optimal affinity
    /// @return Vector of device ID vectors, one per NUMA node
    [[nodiscard]] std::vector<std::vector<int>> devices_by_numa_node() const;
    
    /// @brief Get an optimal device ordering for the given count
    /// @param count Number of devices to select
    /// @return Device IDs ordered for optimal communication topology
    [[nodiscard]] std::vector<int> select_optimal_devices(int count) const;
    
    // =========================================================================
    // Utility
    // =========================================================================
    
    /// @brief Synchronize all active devices
    void synchronize_all();
    
    /// @brief Synchronize a specific device
    void synchronize_device(int device_id);
    
    /// @brief Reset (reinitialize) a device
    void reset_device(int device_id);
    
    /// @brief Get a summary string for logging
    [[nodiscard]] std::string summary() const;

private:
    DeviceManager();
    ~DeviceManager();
    
    void enumerate_devices();
    void query_device_properties(int device_id);
    
    int device_count_ = 0;
    std::vector<DeviceProperties> device_properties_;
    std::vector<int> active_devices_;
    
    // Peer access tracking (device_count_ x device_count_ matrix, row-major)
    std::vector<bool> peer_access_enabled_;
    std::vector<bool> peer_access_possible_;
};

// ============================================================================
// RAII Helpers
// ============================================================================

/// @brief RAII guard for temporarily switching to a different device
///
/// Saves the current device on construction and restores it on destruction.
/// Useful for multi-GPU operations that need to temporarily operate on
/// a different device.
class ScopedDevice {
public:
    /// @brief Switch to the specified device
    /// @param device_id Device to make current
    explicit ScopedDevice(int device_id);
    
    /// @brief Restore the previous device
    ~ScopedDevice();
    
    // Non-copyable
    ScopedDevice(const ScopedDevice&) = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;
    
    // Moveable
    ScopedDevice(ScopedDevice&& other) noexcept;
    ScopedDevice& operator=(ScopedDevice&& other) noexcept;
    
    /// @brief Get the device this guard switched to
    [[nodiscard]] int device() const noexcept { return device_id_; }
    
private:
    int device_id_;
    int previous_device_;
    bool owns_restore_ = true;
};

}  // namespace libaccint::device
