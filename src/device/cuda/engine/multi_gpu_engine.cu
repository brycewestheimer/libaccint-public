// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file multi_gpu_engine.cu
/// @brief Multi-GPU engine implementation

#include <libaccint/engine/multi_gpu_engine.hpp>

#if LIBACCINT_USE_CUDA

#include <algorithm>
#include <cmath>
#include <sstream>

namespace libaccint::engine {

// ============================================================================
// Constructor / Destructor
// ============================================================================

MultiGPUEngine::MultiGPUEngine(const BasisSet& basis, MultiGPUConfig config)
    : basis_(&basis),
      config_(std::move(config)),
      distributor_(config_.distribution) {
    initialize();
}

MultiGPUEngine::~MultiGPUEngine() {
    cleanup();
}

MultiGPUEngine::MultiGPUEngine(MultiGPUEngine&& other) noexcept
    : basis_(other.basis_),
      config_(std::move(other.config_)),
      device_ids_(std::move(other.device_ids_)),
      engines_(std::move(other.engines_)),
      compute_streams_(std::move(other.compute_streams_)),
      transfer_streams_(std::move(other.transfer_streams_)),
      distributor_(std::move(other.distributor_)),
      memory_manager_(std::move(other.memory_manager_)),
      stats_(other.stats_),
      work_queues_(std::move(other.work_queues_)),
      total_remaining_(other.total_remaining_.load(std::memory_order_relaxed)),
      initialized_(other.initialized_) {
    other.initialized_ = false;
}

MultiGPUEngine& MultiGPUEngine::operator=(MultiGPUEngine&& other) noexcept {
    if (this != &other) {
        cleanup();
        basis_ = other.basis_;
        config_ = std::move(other.config_);
        device_ids_ = std::move(other.device_ids_);
        engines_ = std::move(other.engines_);
        compute_streams_ = std::move(other.compute_streams_);
        transfer_streams_ = std::move(other.transfer_streams_);
        distributor_ = std::move(other.distributor_);
        memory_manager_ = std::move(other.memory_manager_);
        stats_ = other.stats_;
        work_queues_ = std::move(other.work_queues_);
        total_remaining_.store(other.total_remaining_.load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
        initialized_ = other.initialized_;
        other.initialized_ = false;
    }
    return *this;
}

void MultiGPUEngine::initialize() {
    if (initialized_) return;
    
    auto& dm = device::DeviceManager::instance();
    
    // Determine which devices to use
    if (config_.device_ids.empty()) {
        // Use all available devices
        dm.set_all_devices();
        device_ids_ = dm.active_devices();
    } else {
        dm.set_active_devices(config_.device_ids);
        device_ids_ = config_.device_ids;
    }
    
    if (device_ids_.empty()) {
        throw device::DeviceError("No GPU devices available for multi-GPU execution");
    }
    
    // Enable peer-to-peer access if configured
    if (config_.enable_peer_access) {
        dm.enable_all_peer_access();
    }
    
    // Create per-device engines
    engines_.reserve(device_ids_.size());
    for (int device_id : device_ids_) {
        device::ScopedDevice guard(device_id);
        engines_.push_back(std::make_unique<CudaEngine>(*basis_));
    }
    
    // Create streams for async execution
    if (config_.async_execution) {
        for (int device_id : device_ids_) {
            device::ScopedDevice guard(device_id);
            compute_streams_.emplace_back();
            transfer_streams_.emplace_back();
        }
    }
    
    // Create memory manager
    memory_manager_ = std::make_unique<device::MultiDeviceMemoryManager>(device_ids_);
    
    // Set up device weights for load balancing if not provided
    if (config_.distribution.device_weights.empty()) {
        update_resource_aware_weights();
    }
    
    initialized_ = true;
}

void MultiGPUEngine::cleanup() {
    if (!initialized_) return;
    
    // Synchronize all devices before cleanup
    synchronize_all();
    
    // Clear in reverse order of creation
    transfer_streams_.clear();
    compute_streams_.clear();
    memory_manager_.reset();
    engines_.clear();
    
    // Disable peer access
    if (config_.enable_peer_access) {
        auto& dm = device::DeviceManager::instance();
        dm.disable_all_peer_access();
    }
    
    initialized_ = false;
}

// ============================================================================
// Accessors
// ============================================================================

CudaEngine& MultiGPUEngine::engine(int device_index) {
    if (device_index < 0 || device_index >= static_cast<int>(engines_.size())) {
        throw device::DeviceError("Invalid device index: " + std::to_string(device_index));
    }
    return *engines_[device_index];
}

const CudaEngine& MultiGPUEngine::engine(int device_index) const {
    if (device_index < 0 || device_index >= static_cast<int>(engines_.size())) {
        throw device::DeviceError("Invalid device index: " + std::to_string(device_index));
    }
    return *engines_[device_index];
}

// ============================================================================
// Full Matrix Computation
// ============================================================================

void MultiGPUEngine::compute_overlap_matrix(std::vector<Real>& result) {
    // For 1e integrals, partition shell pairs across devices
    // and reduce results
    
    const Size nbf = basis_->n_basis_functions();
    result.resize(nbf * nbf, 0.0);
    
    // For simplicity, use first device for 1e integrals
    // (1e integrals are typically not the bottleneck)
    device::ScopedDevice guard(device_ids_[0]);
    engines_[0]->compute_overlap_matrix(result);
}

void MultiGPUEngine::compute_kinetic_matrix(std::vector<Real>& result) {
    const Size nbf = basis_->n_basis_functions();
    result.resize(nbf * nbf, 0.0);
    
    device::ScopedDevice guard(device_ids_[0]);
    engines_[0]->compute_kinetic_matrix(result);
}

void MultiGPUEngine::compute_nuclear_matrix(const PointChargeParams& charges,
                                              std::vector<Real>& result) {
    const Size nbf = basis_->n_basis_functions();
    result.resize(nbf * nbf, 0.0);
    
    device::ScopedDevice guard(device_ids_[0]);
    engines_[0]->compute_nuclear_matrix(charges, result);
}

// ============================================================================
// Utility
// ============================================================================

void MultiGPUEngine::synchronize_all() {
    for (size_t d = 0; d < device_ids_.size(); ++d) {
        device::ScopedDevice guard(device_ids_[d]);
        cudaDeviceSynchronize();
    }
}

void MultiGPUEngine::reset_distribution() {
    // Refresh with resource-aware weights
    update_resource_aware_weights();
}

void MultiGPUEngine::update_resource_aware_weights() {
    auto& dm = device::DeviceManager::instance();
    dm.refresh_properties();

    std::vector<double> weights;
    weights.reserve(device_ids_.size());

    for (size_t d = 0; d < device_ids_.size(); ++d) {
        device::ScopedDevice guard(device_ids_[d]);
        const auto& props = dm.get_device_properties(device_ids_[d]);
        double sm_throughput = props.compute_score();

        // Query the per-device resource tracker for live availability
        auto* tracker = engines_[d]->get_tracker();
        if (tracker) {
            tracker->refresh_device_properties();

            // SM availability: fraction of SMs not occupied by active kernels
            int total_sms = tracker->total_sms();
            int active = tracker->active_kernels();
            double sm_available = (total_sms > 0)
                ? std::max(0.0, 1.0 - static_cast<double>(active) / total_sms)
                : 1.0;

            // Memory availability: fraction of usable global memory still free
            size_t allocated = tracker->allocated_global_bytes();
            size_t total_mem = tracker->total_global_memory();
            double mem_available = (total_mem > 0)
                ? std::max(0.0, 1.0 - static_cast<double>(allocated)
                                      / (static_cast<double>(total_mem) * 0.85))
                : 1.0;

            // Combined availability (min of SM and memory headroom, floored)
            double available_fraction = std::max(0.01,
                                                  std::min(sm_available, mem_available));

            weights.push_back(available_fraction * sm_throughput);
        } else {
            // Fallback: static compute score only
            weights.push_back(sm_throughput);
        }
    }

    config_.distribution.device_weights = std::move(weights);
    distributor_.set_config(config_.distribution);
}

std::string MultiGPUEngine::summary() const {
    std::ostringstream oss;
    oss << "MultiGPUEngine: " << device_ids_.size() << " devices\n";
    
    auto& dm = device::DeviceManager::instance();
    for (size_t d = 0; d < device_ids_.size(); ++d) {
        const auto& props = dm.get_device_properties(device_ids_[d]);
        oss << "  [" << d << "] Device " << device_ids_[d] << ": " 
            << props.name << "\n";
    }
    
    if (stats_.total_time_ms > 0) {
        oss << "Last execution stats:\n";
        oss << "  Total time: " << stats_.total_time_ms << " ms\n";
        oss << "  Load balance: " << (stats_.load_balance_efficiency() * 100) << "%\n";
    }
    
    return oss.str();
}

}  // namespace libaccint::engine

#endif  // LIBACCINT_USE_CUDA
