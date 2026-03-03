// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file mpi_engine.hpp
/// @brief MPI-distributed molecular integral computation
///
/// Extends multi-GPU support to distributed computing across multiple nodes
/// using MPI for communication and coordination.

#include <libaccint/config.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/core/types.hpp>

#include <string>
#include <vector>

#if LIBACCINT_USE_MPI

#include <libaccint/mpi/mpi_guard.hpp>

#include <libaccint/engine/cpu_engine.hpp>
#include <libaccint/operators/operator.hpp>

#if LIBACCINT_USE_CUDA
#include <libaccint/engine/multi_gpu_engine.hpp>
#include <libaccint/consumers/multi_gpu_fock_builder.hpp>
#endif

#include <mpi.h>
#include <chrono>
#include <functional>
#include <memory>

namespace libaccint::mpi {

/// @brief GPU-to-rank mapping strategies
enum class GPUMapping {
    RoundRobin,      ///< Assign GPUs in round-robin order
    Packed,          ///< Assign all local GPUs to this rank
    Exclusive,       ///< One GPU per rank (1:1 mapping)
    UserDefined      ///< Custom mapping provided by user
};

/// @brief Configuration for MPI-distributed execution
struct MPIEngineConfig {
    /// MPI communicator to use (default: MPI_COMM_WORLD)
    MPI_Comm comm = MPI_COMM_WORLD;
    
    /// GPU mapping strategy
    GPUMapping gpu_mapping = GPUMapping::RoundRobin;
    
    /// For UserDefined mapping: which GPUs this rank should use
    std::vector<int> local_gpu_ids;
    
    /// Number of GPUs per rank (for Packed mode, 0 = all available)
    int gpus_per_rank = 0;
    
    /// Collect timing statistics
    bool collect_stats = false;
};

/// @brief Statistics from MPI-distributed execution
struct MPIStats {
    double total_time_ms = 0.0;
    double compute_time_ms = 0.0;
    double communication_time_ms = 0.0;
    double reduction_time_ms = 0.0;
    
    int total_ranks = 0;
    Size local_quartets = 0;
    Size total_quartets = 0;
    
    /// @brief Calculate per-rank efficiency
    [[nodiscard]] double work_distribution_efficiency() const {
        if (total_ranks == 0 || total_quartets == 0) return 0.0;
        Size ideal_per_rank = total_quartets / total_ranks;
        return (ideal_per_rank > 0) ? 
               static_cast<double>(local_quartets) / ideal_per_rank : 1.0;
    }
};

/// @brief MPI-distributed engine for molecular integral computation
///
/// MPIEngine coordinates computation across multiple MPI ranks, with each
/// rank potentially managing multiple GPUs. It handles:
/// - Rank-to-GPU mapping
/// - Distributed work partitioning
/// - MPI-based result reduction
/// - Optional GPU-aware MPI for optimized communication
///
/// Usage:
/// @code
///   MPIGuard mpi(&argc, &argv);
///   
///   BasisSet basis(shells);
///   MPIEngineConfig config;
///   config.gpu_mapping = GPUMapping::Exclusive;
///   
///   MPIEngine engine(basis, config);
///   
///   // Each rank computes its share and reduces
///   MPIFockBuilder fock(engine);
///   engine.compute_all_eri(fock);
///   
///   // Results are reduced to all ranks (or just root)
///   auto J = fock.get_coulomb_matrix();
/// @endcode
class MPIEngine {
public:
    /// @brief Construct an MPI-distributed engine
    /// @param basis Basis set (must be identical on all ranks)
    /// @param config MPI engine configuration
    explicit MPIEngine(const BasisSet& basis, MPIEngineConfig config = {});
    
    /// @brief Destructor
    ~MPIEngine();
    
    // Non-copyable
    MPIEngine(const MPIEngine&) = delete;
    MPIEngine& operator=(const MPIEngine&) = delete;
    
    // Moveable
    MPIEngine(MPIEngine&&) noexcept;
    MPIEngine& operator=(MPIEngine&&) noexcept;
    
    // =========================================================================
    // Accessors
    // =========================================================================
    
    /// @brief Get the basis set
    [[nodiscard]] const BasisSet& basis() const noexcept { return *basis_; }
    
    /// @brief Get MPI rank
    [[nodiscard]] int rank() const noexcept { return rank_; }
    
    /// @brief Get MPI size (total ranks)
    [[nodiscard]] int size() const noexcept { return size_; }
    
    /// @brief Check if this is the root rank
    [[nodiscard]] bool is_root() const noexcept { return rank_ == 0; }
    
    /// @brief Get the communicator
    [[nodiscard]] MPI_Comm comm() const noexcept { return config_.comm; }
    
    /// @brief Get local GPU device IDs assigned to this rank
    [[nodiscard]] const std::vector<int>& local_gpu_ids() const noexcept {
        return local_gpu_ids_;
    }
    
    /// @brief Get the configuration
    [[nodiscard]] const MPIEngineConfig& config() const noexcept {
        return config_;
    }
    
    /// @brief Get statistics from last execution
    [[nodiscard]] const MPIStats& stats() const noexcept { return stats_; }
    
#if LIBACCINT_USE_CUDA
    /// @brief Get the local multi-GPU engine
    [[nodiscard]] engine::MultiGPUEngine& local_engine() { return *local_engine_; }
    [[nodiscard]] const engine::MultiGPUEngine& local_engine() const { 
        return *local_engine_; 
    }
#endif
    
    // =========================================================================
    // Computation
    // =========================================================================
    
    /// @brief Compute all ERIs with distributed work
    ///
    /// Each rank computes its share of shell quartets. Results are NOT
    /// automatically reduced - use the consumer's reduction methods.
    ///
    /// @tparam Consumer Consumer type (e.g., MPIFockBuilder)
    /// @param consumer Consumer for accumulating results
    template<typename Consumer>
    void compute_all_eri(Consumer& consumer);
    
    /// @brief Partition and compute specified quartets
    ///
    /// @tparam Consumer Consumer type
    /// @param quartets Full list of quartets (must be same on all ranks)
    /// @param consumer Consumer for accumulating results
    template<typename Consumer>
    void compute_quartets(const std::vector<ShellSetQuartet>& quartets,
                           Consumer& consumer);
    
    // =========================================================================
    // Communication
    // =========================================================================
    
    /// @brief Reduce a matrix to root rank
    /// @param local_data Local matrix data
    /// @param global_data Output matrix (only valid on root)
    /// @param count Number of elements
    void reduce_to_root(const double* local_data, double* global_data, 
                        Size count);
    
    /// @brief All-reduce a matrix (result on all ranks)
    /// @param local_data Local matrix data
    /// @param global_data Output matrix (valid on all ranks)
    /// @param count Number of elements
    void allreduce(const double* local_data, double* global_data, 
                   Size count);
    
    /// @brief Barrier synchronization
    void barrier();
    
    // =========================================================================
    // Utility
    // =========================================================================
    
    /// @brief Get summary string
    [[nodiscard]] std::string summary() const;
    
    /// @brief Synchronize local computation (GPUs)
    void synchronize_local();

private:
    void initialize();
    void setup_gpu_mapping();
    
    /// @brief Partition quartets for this rank using cost-based load balancing
    [[nodiscard]] std::vector<Size> get_rank_partition(
        const std::vector<ShellSetQuartet>& quartets) const;
    
    const BasisSet* basis_;
    MPIEngineConfig config_;
    
    int rank_ = 0;
    int size_ = 1;
    std::vector<int> local_gpu_ids_;
    
#if LIBACCINT_USE_CUDA
    std::unique_ptr<engine::MultiGPUEngine> local_engine_;
#endif
    
    MPIStats stats_;
    bool initialized_ = false;
};

// ============================================================================
// Template Implementation
// ============================================================================

template<typename Consumer>
void MPIEngine::compute_all_eri(Consumer& consumer) {
    const auto& quartets = basis_->shell_set_quartets();
    compute_quartets(quartets, consumer);
}

template<typename Consumer>
void MPIEngine::compute_quartets(
    const std::vector<ShellSetQuartet>& quartets,
    Consumer& consumer) {
    
    auto total_start = std::chrono::steady_clock::now();
    
    // Get this rank's partition using cost-based load balancing
    auto my_indices = get_rank_partition(quartets);
    
    // Build local quartet list
    std::vector<ShellSetQuartet> local_quartets;
    local_quartets.reserve(my_indices.size());
    for (Size idx : my_indices) {
        local_quartets.push_back(quartets[idx]);
    }
    
    if (config_.collect_stats) {
        stats_.local_quartets = local_quartets.size();
        stats_.total_quartets = quartets.size();
        stats_.total_ranks = size_;
    }
    
    auto compute_start = std::chrono::steady_clock::now();
    
#if LIBACCINT_USE_CUDA
    // Compute on local GPUs
    local_engine_->compute_and_consume(local_quartets, consumer);
#else
    // CPU-only path: use CpuEngine to compute assigned quartets
    engine::CpuEngine cpu_engine(*basis_);
    auto coulomb_op = Operator::coulomb();
    for (const auto& quartet : local_quartets) {
        cpu_engine.compute_shell_set_quartet(coulomb_op, quartet, consumer);
    }
#endif
    
    if (config_.collect_stats) {
        auto compute_end = std::chrono::steady_clock::now();
        stats_.compute_time_ms = std::chrono::duration<double, std::milli>(
            compute_end - compute_start).count();
        stats_.total_time_ms = std::chrono::duration<double, std::milli>(
            compute_end - total_start).count();
    }
}

}  // namespace libaccint::mpi

#else  // !LIBACCINT_USE_MPI

// Stub for non-MPI builds
namespace libaccint::mpi {

enum class GPUMapping { RoundRobin, Packed, Exclusive, UserDefined };

struct MPIEngineConfig {
    void* comm = nullptr;
    GPUMapping gpu_mapping = GPUMapping::RoundRobin;
    std::vector<int> local_gpu_ids;
    int gpus_per_rank = 0;
    bool collect_stats = false;
};

struct MPIStats {
    double total_time_ms = 0.0;
    double compute_time_ms = 0.0;
    double communication_time_ms = 0.0;
    double reduction_time_ms = 0.0;
    int total_ranks = 0;
    Size local_quartets = 0;
    Size total_quartets = 0;
    
    [[nodiscard]] double work_distribution_efficiency() const {
        if (total_ranks == 0 || total_quartets == 0) return 0.0;
        Size ideal_per_rank = total_quartets / total_ranks;
        return (ideal_per_rank > 0) ?
               static_cast<double>(local_quartets) / ideal_per_rank : 1.0;
    }
};

/// @brief Stub MPIEngine for non-MPI builds
class MPIEngine {
public:
    explicit MPIEngine(const BasisSet& /*basis*/, MPIEngineConfig /*config*/ = {}) {}
    [[nodiscard]] int rank() const noexcept { return 0; }
    [[nodiscard]] int size() const noexcept { return 1; }
    [[nodiscard]] bool is_root() const noexcept { return true; }
    [[nodiscard]] const std::vector<int>& local_gpu_ids() const noexcept {
        return gpu_ids_;
    }
    [[nodiscard]] const MPIEngineConfig& config() const noexcept {
        return config_;
    }
    [[nodiscard]] const MPIStats& stats() const noexcept { return stats_; }
    
    template<typename Consumer>
    void compute_all_eri(Consumer& /*consumer*/) {}
    
    template<typename Consumer>
    void compute_quartets(const std::vector<ShellSetQuartet>& /*quartets*/,
                           Consumer& /*consumer*/) {}
    
    void barrier() {}
    void synchronize_local() {}
    [[nodiscard]] std::string summary() const { return "MPIEngine (stub, MPI disabled)"; }
    
private:
    std::vector<int> gpu_ids_;
    MPIEngineConfig config_;
    MPIStats stats_;
};

}  // namespace libaccint::mpi

#endif  // LIBACCINT_USE_MPI
