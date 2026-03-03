// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file device_manager.cu
/// @brief CUDA implementation of multi-GPU device management

#include <libaccint/device/device_manager.hpp>

#if LIBACCINT_USE_CUDA

#include <cuda_runtime.h>
#include <algorithm>
#include <sstream>

namespace libaccint::device {

// ============================================================================
// DeviceManager Implementation
// ============================================================================

DeviceManager& DeviceManager::instance() {
    static DeviceManager instance;
    return instance;
}

DeviceManager::DeviceManager() {
    enumerate_devices();
}

DeviceManager::~DeviceManager() {
    // Disable any enabled peer access
    try {
        disable_all_peer_access();
    } catch (...) {
        // Ignore errors during destruction
    }
}

void DeviceManager::enumerate_devices() {
    cudaError_t err = cudaGetDeviceCount(&device_count_);
    if (err != cudaSuccess) {
        device_count_ = 0;
        return;
    }
    
    device_properties_.resize(device_count_);
    
    // Query properties for each device
    for (int i = 0; i < device_count_; ++i) {
        query_device_properties(i);
    }
    
    // Initialize peer access matrices
    const size_t n = device_count_;
    peer_access_enabled_.resize(n * n, false);
    peer_access_possible_.resize(n * n, false);
    
    // Query peer access capability for all pairs
    for (int i = 0; i < device_count_; ++i) {
        for (int j = 0; j < device_count_; ++j) {
            if (i == j) {
                peer_access_possible_[i * n + j] = true;  // Self-access always ok
                continue;
            }
            int can_access = 0;
            cudaDeviceCanAccessPeer(&can_access, i, j);
            peer_access_possible_[i * n + j] = (can_access != 0);
        }
    }
    
    // By default, no devices are active (user must select)
    active_devices_.clear();
}

void DeviceManager::query_device_properties(int device_id) {
    if (device_id < 0 || device_id >= device_count_) {
        throw DeviceError(device_id, "Invalid device ID");
    }
    
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        throw DeviceError(device_id, cudaGetErrorString(err));
    }
    
    DeviceProperties& dp = device_properties_[device_id];
    dp.device_id = device_id;
    dp.name = props.name;
    dp.total_memory = props.totalGlobalMem;
    dp.multiprocessor_count = props.multiProcessorCount;
    dp.max_threads_per_mp = props.maxThreadsPerMultiProcessor;
    dp.major_version = props.major;
    dp.minor_version = props.minor;
    dp.warp_size = props.warpSize;
    dp.memory_clock_khz = props.memoryClockRate;
    dp.memory_bus_width = props.memoryBusWidth;
    dp.unified_addressing = props.unifiedAddressing;
    dp.concurrent_kernels = props.concurrentKernels;
    dp.managed_memory = props.managedMemory;
    dp.pci_bus_id = props.pciBusID;
    dp.pci_device_id = props.pciDeviceID;
    dp.pci_domain_id = props.pciDomainID;
    
    // Calculate peak memory bandwidth in GB/s
    // Formula: (memory_clock * 2 * bus_width / 8) / 1e6
    // Factor of 2 for DDR
    dp.peak_memory_bandwidth_gbps = 
        (static_cast<double>(props.memoryClockRate) * 2.0 * 
         static_cast<double>(props.memoryBusWidth) / 8.0) / 1.0e6;
    
    // Query free memory
    size_t free_mem = 0, total_mem = 0;
    int current_device;
    cudaGetDevice(&current_device);
    cudaSetDevice(device_id);
    cudaMemGetInfo(&free_mem, &total_mem);
    cudaSetDevice(current_device);
    dp.free_memory = free_mem;
}

void DeviceManager::refresh_properties(int device_id) {
    if (device_id == -1) {
        for (int i = 0; i < device_count_; ++i) {
            query_device_properties(i);
        }
    } else {
        query_device_properties(device_id);
    }
}

const DeviceProperties& DeviceManager::get_device_properties(int device_id) const {
    if (device_id < 0 || device_id >= device_count_) {
        throw DeviceError(device_id, "Invalid device ID");
    }
    return device_properties_[device_id];
}

void DeviceManager::set_active_devices(const std::vector<int>& device_ids) {
    // Validate all device IDs
    for (int id : device_ids) {
        if (id < 0 || id >= device_count_) {
            throw DeviceError(id, "Invalid device ID in selection");
        }
    }
    
    // Check for duplicates
    std::vector<int> sorted_ids = device_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    auto it = std::adjacent_find(sorted_ids.begin(), sorted_ids.end());
    if (it != sorted_ids.end()) {
        throw DeviceError(*it, "Duplicate device ID in selection");
    }
    
    active_devices_ = device_ids;
}

void DeviceManager::set_all_devices() {
    active_devices_.clear();
    active_devices_.reserve(device_count_);
    for (int i = 0; i < device_count_; ++i) {
        active_devices_.push_back(i);
    }
}

void DeviceManager::set_current_device(int device_id) {
    if (device_id < 0 || device_id >= device_count_) {
        throw DeviceError(device_id, "Invalid device ID");
    }
    
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        throw DeviceError(device_id, cudaGetErrorString(err));
    }
}

int DeviceManager::current_device() const {
    int device_id = 0;
    cudaGetDevice(&device_id);
    return device_id;
}

bool DeviceManager::can_access_peer(int source_device, int target_device) const {
    if (source_device < 0 || source_device >= device_count_ ||
        target_device < 0 || target_device >= device_count_) {
        return false;
    }
    return peer_access_possible_[source_device * device_count_ + target_device];
}

void DeviceManager::enable_peer_access(int source_device, int target_device) {
    if (source_device == target_device) return;
    
    if (!can_access_peer(source_device, target_device)) {
        throw DeviceError("Peer access not possible between devices " + 
                          std::to_string(source_device) + " and " + 
                          std::to_string(target_device));
    }
    
    const size_t idx = source_device * device_count_ + target_device;
    if (peer_access_enabled_[idx]) return;  // Already enabled
    
    // Switch to source device and enable access to target
    int current_device;
    cudaGetDevice(&current_device);
    cudaSetDevice(source_device);
    
    cudaError_t err = cudaDeviceEnablePeerAccess(target_device, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
        cudaSetDevice(current_device);
        throw DeviceError("Failed to enable peer access: " + 
                          std::string(cudaGetErrorString(err)));
    }
    
    peer_access_enabled_[idx] = true;
    cudaSetDevice(current_device);
}

void DeviceManager::disable_peer_access(int source_device, int target_device) {
    if (source_device == target_device) return;
    
    const size_t idx = source_device * device_count_ + target_device;
    if (!peer_access_enabled_[idx]) return;  // Not enabled
    
    int current_device;
    cudaGetDevice(&current_device);
    cudaSetDevice(source_device);
    
    cudaError_t err = cudaDeviceDisablePeerAccess(target_device);
    if (err != cudaSuccess && err != cudaErrorPeerAccessNotEnabled) {
        cudaSetDevice(current_device);
        throw DeviceError("Failed to disable peer access: " + 
                          std::string(cudaGetErrorString(err)));
    }
    
    peer_access_enabled_[idx] = false;
    cudaSetDevice(current_device);
}

void DeviceManager::enable_all_peer_access() {
    for (int src : active_devices_) {
        for (int dst : active_devices_) {
            if (src != dst && can_access_peer(src, dst)) {
                try {
                    enable_peer_access(src, dst);
                } catch (const DeviceError&) {
                    // Ignore individual failures, some pairs may not support P2P
                }
            }
        }
    }
}

void DeviceManager::disable_all_peer_access() {
    for (int src : active_devices_) {
        for (int dst : active_devices_) {
            if (src != dst) {
                try {
                    disable_peer_access(src, dst);
                } catch (...) {
                    // Ignore errors during cleanup
                }
            }
        }
    }
}

std::vector<PeerAccessInfo> DeviceManager::get_peer_access_matrix() const {
    std::vector<PeerAccessInfo> matrix;
    matrix.reserve(device_count_ * device_count_);
    
    for (int i = 0; i < device_count_; ++i) {
        for (int j = 0; j < device_count_; ++j) {
            PeerAccessInfo info;
            info.source_device = i;
            info.target_device = j;
            info.can_access = peer_access_possible_[i * device_count_ + j];
            info.nvlink_connected = is_nvlink_connected(i, j);
            matrix.push_back(info);
        }
    }
    
    return matrix;
}

bool DeviceManager::is_nvlink_connected(int device1, int device2) const {
    if (device1 == device2) return false;
    if (device1 < 0 || device1 >= device_count_ ||
        device2 < 0 || device2 >= device_count_) {
        return false;
    }
    
    // Check using CUDA P2P link type if available
    // For now, use heuristic: devices on same PCI domain with P2P access
    // may have NVLink
    const auto& props1 = device_properties_[device1];
    const auto& props2 = device_properties_[device2];
    
    // Simple heuristic: if both devices support P2P and are on the same
    // PCI domain, they might have NVLink (actual detection requires nvml)
    return can_access_peer(device1, device2) && 
           props1.pci_domain_id == props2.pci_domain_id;
}

std::vector<std::vector<int>> DeviceManager::devices_by_numa_node() const {
    // Group by PCI domain as a proxy for NUMA node
    std::vector<std::vector<int>> groups;
    
    for (int i = 0; i < device_count_; ++i) {
        const int domain = device_properties_[i].pci_domain_id;
        
        // Find or create group for this domain
        bool found = false;
        for (auto& group : groups) {
            if (!group.empty() && 
                device_properties_[group[0]].pci_domain_id == domain) {
                group.push_back(i);
                found = true;
                break;
            }
        }
        if (!found) {
            groups.push_back({i});
        }
    }
    
    return groups;
}

std::vector<int> DeviceManager::select_optimal_devices(int count) const {
    if (count <= 0 || count > device_count_) {
        throw DeviceError("Invalid device count: " + std::to_string(count));
    }
    
    // Strategy: prioritize devices with P2P connectivity
    // Start with the most capable device, then add devices with P2P access
    std::vector<std::pair<double, int>> scored_devices;
    for (int i = 0; i < device_count_; ++i) {
        scored_devices.emplace_back(device_properties_[i].compute_score(), i);
    }
    
    // Sort by compute score descending
    std::sort(scored_devices.begin(), scored_devices.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::vector<int> selected;
    selected.reserve(count);
    
    // Select top devices, preferring those with P2P to already-selected
    for (const auto& [score, device] : scored_devices) {
        if (static_cast<int>(selected.size()) >= count) break;
        
        // Check if this device has P2P with at least one selected device
        // (or if this is the first device)
        bool has_p2p_connection = selected.empty();
        for (int sel : selected) {
            if (can_access_peer(device, sel) || can_access_peer(sel, device)) {
                has_p2p_connection = true;
                break;
            }
        }
        
        if (has_p2p_connection || selected.size() + 1 == static_cast<size_t>(count)) {
            selected.push_back(device);
        }
    }
    
    // If we couldn't fill with P2P-connected devices, just take top devices
    if (static_cast<int>(selected.size()) < count) {
        for (const auto& [score, device] : scored_devices) {
            if (static_cast<int>(selected.size()) >= count) break;
            if (std::find(selected.begin(), selected.end(), device) == selected.end()) {
                selected.push_back(device);
            }
        }
    }
    
    return selected;
}

void DeviceManager::synchronize_all() {
    int current_device;
    cudaGetDevice(&current_device);
    
    for (int device : active_devices_) {
        cudaSetDevice(device);
        cudaDeviceSynchronize();
    }
    
    cudaSetDevice(current_device);
}

void DeviceManager::synchronize_device(int device_id) {
    if (device_id < 0 || device_id >= device_count_) {
        throw DeviceError(device_id, "Invalid device ID");
    }
    
    int current_device;
    cudaGetDevice(&current_device);
    cudaSetDevice(device_id);
    cudaDeviceSynchronize();
    cudaSetDevice(current_device);
}

void DeviceManager::reset_device(int device_id) {
    if (device_id < 0 || device_id >= device_count_) {
        throw DeviceError(device_id, "Invalid device ID");
    }
    
    int current_device;
    cudaGetDevice(&current_device);
    cudaSetDevice(device_id);
    cudaDeviceReset();
    cudaSetDevice(current_device);
    
    // Re-query properties
    query_device_properties(device_id);
}

std::string DeviceManager::summary() const {
    std::ostringstream oss;
    oss << "DeviceManager: " << device_count_ << " GPU(s) available";
    
    if (device_count_ > 0) {
        oss << "\n";
        for (const auto& props : device_properties_) {
            oss << "  [" << props.device_id << "] " << props.name
                << " (" << props.multiprocessor_count << " SMs, "
                << (props.total_memory / (1024 * 1024)) << " MB)\n";
        }
        
        if (!active_devices_.empty()) {
            oss << "Active devices: {";
            for (size_t i = 0; i < active_devices_.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << active_devices_[i];
            }
            oss << "}";
        }
    }
    
    return oss.str();
}

// ============================================================================
// ScopedDevice Implementation
// ============================================================================

ScopedDevice::ScopedDevice(int device_id)
    : device_id_(device_id) {
    cudaGetDevice(&previous_device_);
    if (device_id_ != previous_device_) {
        cudaSetDevice(device_id_);
    }
}

ScopedDevice::~ScopedDevice() {
    if (owns_restore_ && device_id_ != previous_device_) {
        cudaSetDevice(previous_device_);
    }
}

ScopedDevice::ScopedDevice(ScopedDevice&& other) noexcept
    : device_id_(other.device_id_),
      previous_device_(other.previous_device_),
      owns_restore_(other.owns_restore_) {
    other.owns_restore_ = false;
}

ScopedDevice& ScopedDevice::operator=(ScopedDevice&& other) noexcept {
    if (this != &other) {
        if (owns_restore_ && device_id_ != previous_device_) {
            cudaSetDevice(previous_device_);
        }
        device_id_ = other.device_id_;
        previous_device_ = other.previous_device_;
        owns_restore_ = other.owns_restore_;
        other.owns_restore_ = false;
    }
    return *this;
}

}  // namespace libaccint::device

#endif  // LIBACCINT_USE_CUDA
