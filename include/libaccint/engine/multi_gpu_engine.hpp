// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file multi_gpu_engine.hpp
/// @brief Multi-GPU orchestration for molecular integral computation
///
/// Provides a unified interface for coordinating integral computation across
/// multiple GPU devices. The MultiGPUEngine manages work distribution,
/// per-device computation, and result aggregation.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/device/device_manager.hpp>
#include <libaccint/device/multi_device_memory.hpp>
#include <libaccint/device/work_distribution.hpp>
#include <libaccint/engine/cuda_engine.hpp>
#include <libaccint/memory/stream_management.hpp>

#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace libaccint::engine {

/// @brief Adapts FockBuilder to the raw-pointer consumer interface used by MultiGPUEngine
///
/// MultiGPUEngine::compute_and_consume expects consumers with:
///   void accumulate(const double* data, const ShellSetQuartet& quartet)
/// FockBuilder expects:
///   void accumulate_symmetric(const TwoElectronBuffer<0>&, Index, Index, Index, Index,
///                             int, int, int, int, bool, bool, bool)
///
/// This adapter bridges the two interfaces by unpacking the flat ERI buffer
/// into per-shell-quartet slices and calling accumulate_symmetric for each.
class MultiGPUFockAdapter {
public:
    explicit MultiGPUFockAdapter(consumers::FockBuilder& fb) : fock_(fb) {}

    void accumulate(const double* data, const ShellSetQuartet& quartet) {
        const auto& set_a = quartet.bra_pair().shell_set_a();
        const auto& set_b = quartet.bra_pair().shell_set_b();
        const auto& set_c = quartet.ket_pair().shell_set_a();
        const auto& set_d = quartet.ket_pair().shell_set_b();

        const int na_funcs = n_cartesian(set_a.angular_momentum());
        const int nb_funcs = n_cartesian(set_b.angular_momentum());
        const int nc_funcs = n_cartesian(set_c.angular_momentum());
        const int nd_funcs = n_cartesian(set_d.angular_momentum());

        const Size funcs_per_quartet = static_cast<Size>(na_funcs) * nb_funcs * nc_funcs * nd_funcs;

        TwoElectronBuffer<0> buffer;
        buffer.resize(na_funcs, nb_funcs, nc_funcs, nd_funcs);

        const bool ij_same = (&set_a == &set_b);
        const bool kl_same = (&set_c == &set_d);
        const bool braket_same =
            (&quartet.bra_pair().shell_set_a() == &quartet.ket_pair().shell_set_a()) &&
            (&quartet.bra_pair().shell_set_b() == &quartet.ket_pair().shell_set_b());

        size_t flat_idx = 0;
        for (Size i = 0; i < set_a.n_shells(); ++i) {
            const auto& shell_a = set_a.shell(i);
            const Index fi = shell_a.function_index();

            for (Size j = 0; j < set_b.n_shells(); ++j) {
                const auto& shell_b = set_b.shell(j);
                const Index fj = shell_b.function_index();

                for (Size k = 0; k < set_c.n_shells(); ++k) {
                    const auto& shell_c = set_c.shell(k);
                    const Index fk = shell_c.function_index();

                    for (Size l = 0; l < set_d.n_shells(); ++l) {
                        const auto& shell_d = set_d.shell(l);
                        const Index fl = shell_d.function_index();

                        // Copy from flat buffer to 4D buffer
                        for (int a = 0; a < na_funcs; ++a) {
                            for (int b = 0; b < nb_funcs; ++b) {
                                for (int c = 0; c < nc_funcs; ++c) {
                                    for (int d = 0; d < nd_funcs; ++d) {
                                        buffer(a, b, c, d) = data[
                                            flat_idx +
                                            static_cast<size_t>(a) * nb_funcs * nc_funcs * nd_funcs +
                                            static_cast<size_t>(b) * nc_funcs * nd_funcs +
                                            static_cast<size_t>(c) * nd_funcs + d];
                                    }
                                }
                            }
                        }

                        fock_.accumulate_symmetric(buffer, fi, fj, fk, fl,
                                                   na_funcs, nb_funcs, nc_funcs, nd_funcs,
                                                   ij_same, kl_same, braket_same);

                        flat_idx += funcs_per_quartet;
                    }
                }
            }
        }
    }

    // Required by MultiGPUEngine's thread management
    void prepare_parallel(int) {}
    void finalize_parallel() {}

private:
    consumers::FockBuilder& fock_;
};

/// @brief Statistics from multi-GPU execution
struct MultiGPUStats {
    double total_time_ms = 0.0;          ///< Total wall clock time
    double compute_time_ms = 0.0;        ///< Time spent in kernels
    double communication_time_ms = 0.0;  ///< Time in data transfer
    double reduction_time_ms = 0.0;      ///< Time in result reduction
    
    std::vector<double> per_device_time_ms;  ///< Compute time per device
    std::vector<Size> per_device_quartets;   ///< Quartets processed per device
    
    /// @brief Calculate load balance efficiency
    [[nodiscard]] double load_balance_efficiency() const {
        if (per_device_time_ms.empty()) return 0.0;
        double max_time = *std::max_element(per_device_time_ms.begin(), 
                                             per_device_time_ms.end());
        double avg_time = 0.0;
        for (double t : per_device_time_ms) avg_time += t;
        avg_time /= per_device_time_ms.size();
        return (max_time > 0) ? avg_time / max_time : 1.0;
    }
    
    /// @brief Calculate scaling efficiency vs single GPU
    [[nodiscard]] double scaling_efficiency(double single_gpu_time_ms) const {
        if (total_time_ms <= 0 || per_device_time_ms.empty()) return 0.0;
        double ideal_speedup = static_cast<double>(per_device_time_ms.size());
        double actual_speedup = single_gpu_time_ms / total_time_ms;
        return actual_speedup / ideal_speedup;
    }
};

/// @brief Configuration for multi-GPU execution
struct MultiGPUConfig {
    /// Device IDs to use (empty = use all available)
    std::vector<int> device_ids;
    
    /// Work distribution strategy
    device::DistributionConfig distribution;
    
    /// Enable peer-to-peer access between devices
    bool enable_peer_access = true;
    
    /// Number of streams per device
    int streams_per_device = 2;
    
    /// Overlap compute and communication
    bool async_execution = true;
    
    /// Collect timing statistics
    bool collect_stats = false;
};

/// @brief Thread-safe work-stealing queue for multi-GPU load balancing
///
/// Each device owns one queue.  Threads pop from their own queue (LIFO for
/// cache locality) and steal from other devices' queues (FIFO to reduce
/// contention) when their own queue is empty.
class WorkStealingQueue {
public:
    WorkStealingQueue() = default;

    /// @brief Move constructor (needed for vector<WorkStealingQueue>::resize)
    WorkStealingQueue(WorkStealingQueue&& other) noexcept
        : queue_(std::move(other.queue_)) {}

    /// @brief Move assignment
    WorkStealingQueue& operator=(WorkStealingQueue&& other) noexcept {
        if (this != &other) {
            queue_ = std::move(other.queue_);
        }
        return *this;
    }

    // Non-copyable
    WorkStealingQueue(const WorkStealingQueue&) = delete;
    WorkStealingQueue& operator=(const WorkStealingQueue&) = delete;

    /// @brief Initialise the queue with a batch of work-item indices
    void initialize(const std::vector<Size>& indices) {
        std::lock_guard lock(mutex_);
        queue_.assign(indices.begin(), indices.end());
    }

    /// @brief Pop from own queue (LIFO — better cache locality)
    bool try_pop(Size& index) {
        std::lock_guard lock(mutex_);
        if (queue_.empty()) return false;
        index = queue_.back();
        queue_.pop_back();
        return true;
    }

    /// @brief Steal from front of queue (FIFO — minimises contention with owner)
    bool try_steal(Size& index) {
        std::lock_guard lock(mutex_);
        if (queue_.empty()) return false;
        index = queue_.front();
        queue_.pop_front();
        return true;
    }

    /// @brief Check whether the queue is empty
    [[nodiscard]] bool empty() const {
        std::lock_guard lock(mutex_);
        return queue_.empty();
    }

    /// @brief Number of remaining work items
    [[nodiscard]] size_t size() const {
        std::lock_guard lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::deque<Size> queue_;
};

/// @brief Multi-GPU engine for parallel molecular integral computation
///
/// MultiGPUEngine coordinates integral computation across multiple GPU devices.
/// It manages:
/// - Device discovery and selection
/// - Work partitioning using load balancing
/// - Per-device CudaEngine instances
/// - Concurrent execution with streams
/// - Result aggregation
///
/// Usage:
/// @code
///   BasisSet basis(shells);
///   MultiGPUConfig config;
///   config.device_ids = {0, 1, 2, 3};
///   
///   MultiGPUEngine engine(basis, config);
///   
///   // Use with multi-GPU Fock builder
///   MultiGPUFockBuilder fock(basis.n_basis_functions(), engine);
///   engine.compute_and_consume(quartets, fock);
/// @endcode
class MultiGPUEngine {
public:
    /// @brief Construct a multi-GPU engine
    /// @param basis The basis set (shared across devices)
    /// @param config Multi-GPU configuration
    explicit MultiGPUEngine(const BasisSet& basis, 
                             MultiGPUConfig config = {});
    
    /// @brief Destructor
    ~MultiGPUEngine();
    
    // Non-copyable
    MultiGPUEngine(const MultiGPUEngine&) = delete;
    MultiGPUEngine& operator=(const MultiGPUEngine&) = delete;
    
    // Moveable
    MultiGPUEngine(MultiGPUEngine&&) noexcept;
    MultiGPUEngine& operator=(MultiGPUEngine&&) noexcept;
    
    // =========================================================================
    // Accessors
    // =========================================================================
    
    /// @brief Get the basis set
    [[nodiscard]] const BasisSet& basis() const noexcept { return *basis_; }
    
    /// @brief Get active device IDs
    [[nodiscard]] const std::vector<int>& device_ids() const noexcept {
        return device_ids_;
    }
    
    /// @brief Get number of active devices
    [[nodiscard]] int device_count() const noexcept {
        return static_cast<int>(device_ids_.size());
    }
    
    /// @brief Get the CudaEngine for a specific device
    /// @param device_index Index in device_ids_ (not the device ID itself)
    [[nodiscard]] CudaEngine& engine(int device_index);
    [[nodiscard]] const CudaEngine& engine(int device_index) const;
    
    /// @brief Get the configuration
    [[nodiscard]] const MultiGPUConfig& config() const noexcept {
        return config_;
    }
    
    /// @brief Get the most recent execution statistics
    [[nodiscard]] const MultiGPUStats& stats() const noexcept {
        return stats_;
    }
    
    // =========================================================================
    // Full Matrix Computation (Multi-GPU)
    // =========================================================================
    
    /// @brief Compute the full overlap matrix using all devices
    /// @param result Output matrix (nbf x nbf)
    void compute_overlap_matrix(std::vector<Real>& result);
    
    /// @brief Compute the full kinetic matrix using all devices
    /// @param result Output matrix (nbf x nbf)
    void compute_kinetic_matrix(std::vector<Real>& result);
    
    /// @brief Compute the full nuclear attraction matrix using all devices
    /// @param charges Point charge parameters
    /// @param result Output matrix (nbf x nbf)
    void compute_nuclear_matrix(const PointChargeParams& charges,
                                 std::vector<Real>& result);
    
    // =========================================================================
    // Compute-and-Consume Pattern
    // =========================================================================
    
    /// @brief Compute ERIs for quartets and consume results
    ///
    /// Partitions work across devices, computes in parallel, and
    /// calls consumer.accumulate for each batch.
    ///
    /// @tparam Consumer Type with accumulate method
    /// @param quartets Shell set quartets to compute
    /// @param consumer Consumer for results (e.g., FockBuilder)
    template<typename Consumer>
    void compute_and_consume(const std::vector<ShellSetQuartet>& quartets,
                              Consumer& consumer);
    
    /// @brief Compute all quartets from a basis and consume results
    ///
    /// Generates all significant shell quartets from the basis set,
    /// partitions across devices, and computes with consumption.
    ///
    /// @tparam Consumer Type with accumulate method
    /// @param consumer Consumer for results
    template<typename Consumer>
    void compute_all_eri(Consumer& consumer);
    
    // =========================================================================
    // Utility
    // =========================================================================
    
    /// @brief Synchronize all devices
    void synchronize_all();
    
    /// @brief Reset work distribution (useful after topology changes)
    void reset_distribution();
    
    /// @brief Recompute device weights using resource-tracker occupancy data
    ///
    /// Queries each per-device CudaEngine's DeviceResourceTracker for current
    /// SM availability and memory pressure, then sets device weights to
    /// `available_sms_fraction * sm_throughput`.
    void update_resource_aware_weights();

    /// @brief Get a summary string
    [[nodiscard]] std::string summary() const;

private:
    void initialize();
    void cleanup();
    
    /// @brief Partition work and execute across devices
    template<typename ComputeFunc>
    void execute_partitioned(const std::vector<ShellSetQuartet>& quartets,
                              ComputeFunc compute_func);
    
    const BasisSet* basis_;
    MultiGPUConfig config_;
    std::vector<int> device_ids_;
    
    // Per-device engines
    std::vector<std::unique_ptr<CudaEngine>> engines_;
    
    // Per-device streams for async execution
    std::vector<memory::StreamHandle> compute_streams_;
    std::vector<memory::StreamHandle> transfer_streams_;
    
    // Work distribution
    device::WorkDistributor distributor_;
    
    // Memory management
    std::unique_ptr<device::MultiDeviceMemoryManager> memory_manager_;
    
    // Statistics
    MultiGPUStats stats_;
    
    // Per-device work-stealing queues (populated in compute_and_consume)
    std::vector<WorkStealingQueue> work_queues_;

    // Global counter for remaining work items (used by work-stealing threads)
    std::atomic<size_t> total_remaining_{0};

    bool initialized_ = false;
};

// ============================================================================
// Template Implementation
// ============================================================================

template<typename Consumer>
void MultiGPUEngine::compute_and_consume(
    const std::vector<ShellSetQuartet>& quartets,
    Consumer& consumer) {
    
    if (quartets.empty()) return;

    // Refresh device weights based on live resource availability
    update_resource_aware_weights();

    // Partition work across devices using resource-aware weights
    auto partition = distributor_.partition(quartets, device_ids_);
    
    if (config_.collect_stats) {
        stats_ = MultiGPUStats{};
        stats_.per_device_time_ms.resize(device_ids_.size(), 0.0);
        stats_.per_device_quartets.resize(device_ids_.size(), 0);
    }

    // Process each device's work
    // Single-device case: skip thread overhead (no work-stealing needed)
    if (device_ids_.size() == 1) {
        const auto& work = partition[0];
        if (!work.quartet_indices.empty()) {
            device::ScopedDevice guard(device_ids_[0]);
            auto& engine = *engines_[0];

            for (Size idx : work.quartet_indices) {
                const auto& quartet = quartets[idx];
                auto batch = engine.compute_eri_batch_device_handle(quartet);
                if (!batch) {
                    continue;
                }

                std::vector<double> host_eri(batch.size());
                cudaMemcpyAsync(host_eri.data(), batch.data(),
                                batch.size() * sizeof(double), cudaMemcpyDeviceToHost,
                                batch.stream());
                cudaStreamSynchronize(batch.stream());

                if constexpr (requires(Consumer& c, int device_id, const double* data,
                                       const ShellSetQuartet& q) {
                    c.accumulate(device_id, data, q);
                }) {
                    consumer.accumulate(device_ids_[0], host_eri.data(), quartet);
                } else {
                    consumer.accumulate(host_eri.data(), quartet);
                }
            }

            if (config_.collect_stats) {
                stats_.per_device_quartets[0] = work.quartet_indices.size();
            }
        }
    } else {
        // Multi-device: populate per-device work-stealing queues
        const auto n_devices = device_ids_.size();
        work_queues_.resize(n_devices);
        total_remaining_.store(quartets.size(), std::memory_order_relaxed);

        for (size_t d = 0; d < n_devices; ++d) {
            work_queues_[d].initialize(partition[d].quartet_indices);
        }

        // Launch one thread per device with work-stealing support
        std::vector<std::jthread> device_threads;
        device_threads.reserve(n_devices);

        for (size_t d = 0; d < n_devices; ++d) {
            device_threads.emplace_back([&, d]() {
                device::ScopedDevice guard(device_ids_[d]);
                auto& engine = *engines_[d];
                Size processed = 0;

                // Helper: process a single quartet on *this* device
                auto process_quartet = [&](Size idx) {
                    const auto& quartet = quartets[idx];
                    auto batch = engine.compute_eri_batch_device_handle(quartet);
                    if (!batch) {
                        total_remaining_.fetch_sub(1, std::memory_order_relaxed);
                        return;
                    }

                    std::vector<double> host_eri(batch.size());
                    cudaMemcpyAsync(host_eri.data(), batch.data(),
                                    batch.size() * sizeof(double), cudaMemcpyDeviceToHost,
                                    batch.stream());
                    cudaStreamSynchronize(batch.stream());

                    if constexpr (requires(Consumer& c, int device_id, const double* data,
                                           const ShellSetQuartet& q) {
                        c.accumulate(device_id, data, q);
                    }) {
                        consumer.accumulate(device_ids_[d], host_eri.data(), quartet);
                    } else {
                        consumer.accumulate(host_eri.data(), quartet);
                    }
                    ++processed;
                    total_remaining_.fetch_sub(1, std::memory_order_relaxed);
                };

                // Phase 1: drain own queue
                Size idx;
                while (work_queues_[d].try_pop(idx)) {
                    process_quartet(idx);
                }

                // Phase 2: work-stealing — try to steal from other devices
                while (total_remaining_.load(std::memory_order_relaxed) > 0) {
                    bool stolen = false;
                    for (size_t victim = 0; victim < n_devices; ++victim) {
                        if (victim == d) continue;
                        if (work_queues_[victim].try_steal(idx)) {
                            process_quartet(idx);
                            stolen = true;
                            break;  // re-check remaining count before stealing more
                        }
                    }
                    if (!stolen) break;  // all queues empty
                }

                if (config_.collect_stats) {
                    stats_.per_device_quartets[d] = processed;
                }
            });
        }
        // jthread destructor joins automatically
    }
    
    // Synchronize all devices
    synchronize_all();
}

template<typename Consumer>
void MultiGPUEngine::compute_all_eri(Consumer& consumer) {
    // Generate all shell set quartets
    const auto& quartets = basis_->shell_set_quartets();
    compute_and_consume(quartets, consumer);
}

}  // namespace libaccint::engine

#endif  // LIBACCINT_USE_CUDA
