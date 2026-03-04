// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file device_properties.hpp
/// @brief GPU device properties structure for multi-device management

#include <libaccint/config.hpp>
#include <libaccint/core/types.hpp>

#include <string>
#include <cstddef>

namespace libaccint::device {

/// @brief Structured representation of GPU device properties
///
/// Contains key properties needed for multi-GPU scheduling and load balancing.
/// Works with the CUDA backend.
struct DeviceProperties {
    /// Device ID (index in the visible device list)
    int device_id = -1;
    
    /// Device name (e.g., "NVIDIA A100-SXM4-40GB")
    std::string name;
    
    /// Total global memory in bytes
    size_t total_memory = 0;
    
    /// Free global memory in bytes (at query time)
    size_t free_memory = 0;
    
    /// Number of streaming multiprocessors (SMs)
    int multiprocessor_count = 0;
    
    /// Maximum threads per multiprocessor
    int max_threads_per_mp = 0;
    
    /// Major compute capability
    int major_version = 0;
    
    /// Minor compute capability
    int minor_version = 0;
    
    /// Warp size (32 for NVIDIA GPUs)
    int warp_size = 32;
    
    /// Memory clock rate in kHz
    int memory_clock_khz = 0;
    
    /// Memory bus width in bits
    int memory_bus_width = 0;
    
    /// Peak memory bandwidth in GB/s (derived)
    double peak_memory_bandwidth_gbps = 0.0;
    
    /// Whether this device can use unified addressing
    bool unified_addressing = false;
    
    /// Whether concurrent kernel execution is supported
    bool concurrent_kernels = false;
    
    /// Whether the device supports managed memory
    bool managed_memory = false;
    
    /// PCI bus ID for topology awareness
    int pci_bus_id = 0;
    int pci_device_id = 0;
    int pci_domain_id = 0;
    
    /// @brief Estimate relative compute power for load balancing
    /// @return Normalized compute score (higher = faster)
    [[nodiscard]] double compute_score() const noexcept {
        // Simple heuristic based on SM count and clock
        return static_cast<double>(multiprocessor_count) * max_threads_per_mp;
    }
    
    /// @brief Check if this is a valid, usable device
    [[nodiscard]] bool is_valid() const noexcept {
        return device_id >= 0 && multiprocessor_count > 0;
    }
    
    /// @brief Get a short description string
    [[nodiscard]] std::string description() const {
        return name + " (Device " + std::to_string(device_id) + 
               ", " + std::to_string(total_memory / (1024 * 1024)) + " MB)";
    }
};

/// @brief Peer-to-peer access capability between two devices
struct PeerAccessInfo {
    int source_device = -1;
    int target_device = -1;
    bool can_access = false;
    bool nvlink_connected = false;  ///< True if connected via NVLink (high bandwidth)
};

}  // namespace libaccint::device
