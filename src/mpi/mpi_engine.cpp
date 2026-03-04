// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file mpi_engine.cpp
/// @brief MPI engine implementation

#include <libaccint/mpi/mpi_engine.hpp>

#if LIBACCINT_USE_MPI

#include <libaccint/mpi/mpi_utils.hpp>

#include <algorithm>
#include <chrono>
#include <sstream>

#if LIBACCINT_USE_CUDA
#include <libaccint/device/device_manager.hpp>
#endif

namespace libaccint::mpi {

// ============================================================================
// Constructor / Destructor
// ============================================================================

MPIEngine::MPIEngine(const BasisSet& basis, MPIEngineConfig config)
    : basis_(&basis),
      config_(std::move(config)) {
    initialize();
}

MPIEngine::~MPIEngine() {
    // Cleanup handled by unique_ptr
}

MPIEngine::MPIEngine(MPIEngine&& other) noexcept
    : basis_(other.basis_),
      config_(std::move(other.config_)),
      rank_(other.rank_),
      size_(other.size_),
      local_gpu_ids_(std::move(other.local_gpu_ids_)),
#if LIBACCINT_USE_CUDA
      local_engine_(std::move(other.local_engine_)),
#endif
      stats_(other.stats_),
      initialized_(other.initialized_) {
    other.initialized_ = false;
}

MPIEngine& MPIEngine::operator=(MPIEngine&& other) noexcept {
    if (this != &other) {
        basis_ = other.basis_;
        config_ = std::move(other.config_);
        rank_ = other.rank_;
        size_ = other.size_;
        local_gpu_ids_ = std::move(other.local_gpu_ids_);
#if LIBACCINT_USE_CUDA
        local_engine_ = std::move(other.local_engine_);
#endif
        stats_ = other.stats_;
        initialized_ = other.initialized_;
        other.initialized_ = false;
    }
    return *this;
}

void MPIEngine::initialize() {
    if (initialized_) return;
    
    // Get MPI rank and size
    int err = MPI_Comm_rank(config_.comm, &rank_);
    if (err != MPI_SUCCESS) {
        throw MPIError("MPI_Comm_rank", err);
    }
    err = MPI_Comm_size(config_.comm, &size_);
    if (err != MPI_SUCCESS) {
        throw MPIError("MPI_Comm_size", err);
    }
    
    // Set up GPU mapping
    setup_gpu_mapping();
    
#if LIBACCINT_USE_CUDA
    // Create local multi-GPU engine
    engine::MultiGPUConfig gpu_config;
    gpu_config.device_ids = local_gpu_ids_;
    gpu_config.enable_peer_access = true;
    gpu_config.collect_stats = config_.collect_stats;
    
    local_engine_ = std::make_unique<engine::MultiGPUEngine>(*basis_, gpu_config);
#endif
    
    initialized_ = true;
}

void MPIEngine::setup_gpu_mapping() {
#if LIBACCINT_USE_CUDA
    auto& dm = device::DeviceManager::instance();
    const int n_local_gpus = dm.device_count();
    
    switch (config_.gpu_mapping) {
        case GPUMapping::RoundRobin: {
            // Each rank gets GPU (rank % n_local_gpus)
            if (n_local_gpus > 0) {
                local_gpu_ids_.push_back(rank_ % n_local_gpus);
            }
            break;
        }
        
        case GPUMapping::Packed: {
            // Assign multiple GPUs per rank
            int gpus_to_use = (config_.gpus_per_rank > 0) ? 
                              config_.gpus_per_rank : n_local_gpus;
            gpus_to_use = std::min(gpus_to_use, n_local_gpus);
            
            // Simple packing: rank 0 gets first N, rank 1 gets next N, etc.
            // Wrap around if more ranks than GPUs
            int start_gpu = (rank_ * gpus_to_use) % n_local_gpus;
            for (int i = 0; i < gpus_to_use; ++i) {
                local_gpu_ids_.push_back((start_gpu + i) % n_local_gpus);
            }
            break;
        }
        
        case GPUMapping::Exclusive: {
            // One GPU per rank, assumes n_ranks <= n_gpus * n_nodes
            if (rank_ < n_local_gpus) {
                local_gpu_ids_.push_back(rank_);
            }
            break;
        }
        
        case GPUMapping::UserDefined: {
            local_gpu_ids_ = config_.local_gpu_ids;
            break;
        }
    }
    
    // Validate GPU IDs
    for (int gpu_id : local_gpu_ids_) {
        if (gpu_id < 0 || gpu_id >= n_local_gpus) {
            throw MPIError("Invalid GPU ID " + std::to_string(gpu_id) + 
                           " for rank " + std::to_string(rank_));
        }
    }
#else
    // No GPUs, CPU-only
    local_gpu_ids_.clear();
#endif
}

// ============================================================================
// Communication
// ============================================================================

void MPIEngine::reduce_to_root(const double* local_data, double* global_data,
                                Size count) {
    auto start = std::chrono::steady_clock::now();
    
    chunked_reduce(local_data, global_data, count, 0, config_.comm);
    
    if (config_.collect_stats) {
        auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        stats_.reduction_time_ms += elapsed;
        stats_.communication_time_ms += elapsed;
    }
}

void MPIEngine::allreduce(const double* local_data, double* global_data,
                           Size count) {
    auto start = std::chrono::steady_clock::now();
    
    chunked_allreduce(local_data, global_data, count, config_.comm);
    
    if (config_.collect_stats) {
        auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        stats_.reduction_time_ms += elapsed;
        stats_.communication_time_ms += elapsed;
    }
}

void MPIEngine::barrier() {
    auto start = std::chrono::steady_clock::now();
    
    int err = MPI_Barrier(config_.comm);
    if (err != MPI_SUCCESS) {
        throw MPIError("MPI_Barrier", err);
    }
    
    if (config_.collect_stats) {
        auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        stats_.communication_time_ms += elapsed;
    }
}

// ============================================================================
// Utility
// ============================================================================

std::vector<Size> MPIEngine::get_rank_partition(
    const std::vector<ShellSetQuartet>& quartets) const {
    const Size total = quartets.size();
    
    if (total == 0 || size_ <= 0) {
        return {};
    }
    
    // Estimate cost for each quartet based on total angular momentum.
    // ERI cost scales roughly as O((La+1)*(Lb+1)*(Lc+1)*(Ld+1)) with the
    // number of Cartesian components per shell quartet.  We use
    // (L_bra + 1)^2 * (L_ket + 1)^2 as a simple proxy that captures the
    // quartic scaling with angular momentum.
    std::vector<double> costs(total);
    double total_cost = 0.0;
    for (Size i = 0; i < total; ++i) {
        const auto& q = quartets[i];
        const int L_bra = q.bra_pair().L_total();
        const int L_ket = q.ket_pair().L_total();
        double c = static_cast<double>((L_bra + 1) * (L_bra + 1)) *
                   static_cast<double>((L_ket + 1) * (L_ket + 1));
        costs[i] = c;
        total_cost += c;
    }
    
    // Target cost per rank
    const double target_cost = total_cost / size_;
    
    // Greedy contiguous partitioning: assign consecutive quartets to each
    // rank until the accumulated cost exceeds target_cost * (rank + 1).
    // This preserves locality while balancing computational load.
    std::vector<Size> my_indices;
    double cumulative = 0.0;
    for (Size i = 0; i < total; ++i) {
        cumulative += costs[i];
        
        // Determine which rank this quartet belongs to
        int assigned_rank = static_cast<int>(cumulative / target_cost);
        // Clamp to last rank for rounding edge cases
        if (assigned_rank >= size_) {
            assigned_rank = size_ - 1;
        }
        
        if (assigned_rank == rank_) {
            my_indices.push_back(i);
        }
    }
    
    return my_indices;
}

void MPIEngine::synchronize_local() {
#if LIBACCINT_USE_CUDA
    if (local_engine_) {
        local_engine_->synchronize_all();
    }
#endif
}

std::string MPIEngine::summary() const {
    std::ostringstream oss;
    oss << "MPIEngine: rank " << rank_ << " of " << size_ << "\n";
    oss << "  Local GPUs: {";
    for (size_t i = 0; i < local_gpu_ids_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << local_gpu_ids_[i];
    }
    oss << "}\n";
    
    if (stats_.total_quartets > 0) {
        oss << "  Local quartets: " << stats_.local_quartets 
            << " / " << stats_.total_quartets << "\n";
    }
    
    return oss.str();
}

}  // namespace libaccint::mpi

#endif  // LIBACCINT_USE_MPI
