// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file cuda_engine.hpp
/// @brief CUDA GPU backend for molecular integral computation
///
/// Provides a GPU-accelerated engine that computes molecular integrals
/// using CUDA kernels. The engine manages device memory, stream scheduling,
/// and kernel dispatch for efficient batch processing.
///
/// Key differences from CPU Engine:
/// - Batched processing using ShellSets for efficient GPU parallelization
/// - Device memory management with caching
/// - Asynchronous execution using CUDA streams

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/basis/device_data.hpp>
#include <libaccint/engine/engine_backend.hpp>
#include <libaccint/engine/cpu_engine.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/operators/device_operator_data.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/memory/device_memory.hpp>
#include <libaccint/memory/stream_management.hpp>
#include <libaccint/memory/gpu_execution_slot.hpp>
#include <libaccint/kernels/eri_kernel_cuda.hpp>
#include <libaccint/kernels/eri_scatter_cuda.hpp>
#include <libaccint/kernels/optimal_dispatch_table.hpp>
#include <libaccint/consumers/fock_builder_gpu.hpp>
#include <libaccint/consumers/consumer_concepts.hpp>
#include <libaccint/device/device_resource_tracker.hpp>
#include <libaccint/engine/integral_buffer.hpp>

#include <cuda_runtime.h>
#include <type_traits>
#include <functional>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace libaccint {

// Forward declarations for dispatch config
namespace engine {
class MultiGPUEngine;
}
struct DispatchConfig;

/// @brief Configuration for the multi-stream ERI pipeline
struct EriPipelineConfig {
    size_t n_slots = 4;          ///< Ring buffer size (number of concurrent pipeline slots)
    bool use_callback = false;   ///< false = scatter into host tensor, true = invoke user callback
};

/// @brief GPU-accelerated engine for molecular integral computation
///
/// CudaEngine provides GPU-accelerated computation of molecular integrals
/// using CUDA kernels. It manages device resources and provides both
/// shell-pair/quartet level and full-basis computations.
///
/// Usage:
/// @code
///   BasisSet basis(shells);
///   CudaEngine engine(basis);
///
///   // Compute full overlap matrix
///   std::vector<Real> S;
///   engine.compute_overlap_matrix(S);
///
///   // Compute full kinetic matrix
///   std::vector<Real> T;
///   engine.compute_kinetic_matrix(T);
///
///   // Compute full nuclear attraction matrix
///   PointChargeParams charges = {...};
///   std::vector<Real> V;
///   engine.compute_nuclear_matrix(charges, V);
/// @endcode
class CudaEngine {
public:
    /// @brief Owning handle for a device-resident ERI batch
    ///
    /// Keeps the underlying GPU execution slot alive for as long as the batch
    /// is in scope, so the returned device pointer remains valid.
    class DeviceEriBatch {
    public:
        DeviceEriBatch() = default;
        DeviceEriBatch(DeviceEriBatch&&) noexcept = default;
        DeviceEriBatch& operator=(DeviceEriBatch&&) noexcept = default;
        DeviceEriBatch(const DeviceEriBatch&) = delete;
        DeviceEriBatch& operator=(const DeviceEriBatch&) = delete;

        [[nodiscard]] const double* data() const noexcept { return data_; }
        [[nodiscard]] double* data() noexcept { return data_; }
        [[nodiscard]] size_t size() const noexcept { return count_; }
        [[nodiscard]] cudaStream_t stream() const noexcept {
            return slot_ ? slot_->stream() : nullptr;
        }
        [[nodiscard]] bool empty() const noexcept {
            return data_ == nullptr || count_ == 0;
        }
        explicit operator bool() const noexcept { return !empty(); }

    private:
        friend class CudaEngine;

        DeviceEriBatch(std::unique_ptr<memory::ScopedGpuSlot> slot,
                       double* data,
                       size_t count)
            : slot_(std::move(slot)),
              data_(data),
              count_(count) {}

        [[nodiscard]] memory::GpuExecutionSlot& slot() noexcept {
            return slot_->slot();
        }
        [[nodiscard]] const memory::GpuExecutionSlot& slot() const noexcept {
            return slot_->slot();
        }

        std::unique_ptr<memory::ScopedGpuSlot> slot_;
        double* data_{nullptr};
        size_t count_{0};
    };

    /// @brief Construct a CudaEngine with a BasisSet
    /// @param basis The basis set to use (must remain valid for Engine lifetime)
    /// @throws BackendError if CUDA is not available
    explicit CudaEngine(const BasisSet& basis);

    /// @brief Destructor - releases GPU resources
    ~CudaEngine();

    // Non-copyable
    CudaEngine(const CudaEngine&) = delete;
    CudaEngine& operator=(const CudaEngine&) = delete;

    // Moveable
    CudaEngine(CudaEngine&&) noexcept;
    CudaEngine& operator=(CudaEngine&&) noexcept;

    /// @brief Get the basis set
    [[nodiscard]] const BasisSet& basis() const noexcept { return *basis_; }

    /// @brief Get the maximum angular momentum from the basis set
    [[nodiscard]] int max_angular_momentum() const noexcept {
        return basis_->max_angular_momentum();
    }

    /// @brief Check if the engine is properly initialized
    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }

    /// @brief Get the resource tracker for occupancy-aware dispatch
    /// @return Pointer to the DeviceResourceTracker (may be nullptr if not initialised)
    [[nodiscard]] device::DeviceResourceTracker* get_tracker() const noexcept {
        return tracker_;
    }

    /// @brief Set a non-owning pointer to a CpuEngine for small-batch fallback
    /// @param cpu Pointer to a CpuEngine (may be nullptr to disable fallback)
    void set_cpu_fallback(engine::CpuEngine* cpu) noexcept { cpu_fallback_ = cpu; }

    /// @brief Set dispatch configuration for batch-size thresholds and GPU slot count
    /// @param config Dispatch configuration (min_gpu_batch_size, n_gpu_slots)
    void set_dispatch_config(const DispatchConfig& config);

    /// @brief Get the GPU slot pool for concurrent access
    /// @return Pointer to the slot pool (may be nullptr before initialization)
    [[nodiscard]] memory::GpuSlotPool* slot_pool() const noexcept {
        return slot_pool_.get();
    }

    // =========================================================================
    // Full Matrix Computation
    // =========================================================================

    /// @brief Compute the full overlap matrix S
    /// @param result Output matrix (n_basis_functions x n_basis_functions)
    void compute_overlap_matrix(std::vector<Real>& result);

    /// @brief Compute the full kinetic energy matrix T
    /// @param result Output matrix (n_basis_functions x n_basis_functions)
    void compute_kinetic_matrix(std::vector<Real>& result);

    /// @brief Compute the full nuclear attraction matrix V
    /// @param charges Point charge parameters (nuclear positions and charges)
    /// @param result Output matrix (n_basis_functions x n_basis_functions)
    /// @note GPU nuclear attraction may exhibit ~0.1 absolute error vs CPU for
    ///       small molecules due to device-side Chebyshev Boys function accumulation
    ///       order. This error is bounded and does not grow with system size.
    void compute_nuclear_matrix(const PointChargeParams& charges,
                                std::vector<Real>& result);

    /// @brief Compute the full one-electron Hamiltonian (H = T + V)
    /// @param charges Point charge parameters for nuclear attraction
    /// @param result Output matrix (n_basis_functions x n_basis_functions)
    void compute_core_hamiltonian(const PointChargeParams& charges,
                                  std::vector<Real>& result);

    /// @brief Compute all three one-electron matrices (S, T, V) in a single fused pass
    ///
    /// Uses a fused S+T+V kernel that computes overlap, kinetic, and nuclear
    /// attraction integrals simultaneously, sharing Gaussian product computation
    /// and recursion tables. Eliminates 66% of kernel launches compared to
    /// computing S, T, V separately.
    ///
    /// @param charges Point charge parameters for nuclear attraction
    /// @param S_result Output overlap matrix (n_basis_functions x n_basis_functions)
    /// @param T_result Output kinetic matrix (n_basis_functions x n_basis_functions)
    /// @param V_result Output nuclear attraction matrix (n_basis_functions x n_basis_functions)
    void compute_all_1e_fused(const PointChargeParams& charges,
                              std::vector<Real>& S_result,
                              std::vector<Real>& T_result,
                              std::vector<Real>& V_result);

    // =========================================================================
    // Shell Pair Computation (One-Electron)
    // =========================================================================

    /// @brief Compute overlap integrals for a shell pair on GPU
    /// @param shell_a First shell (bra)
    /// @param shell_b Second shell (ket)
    /// @param buffer Output buffer for this shell pair
    [[deprecated("Use compute_shell_set_pair() for batched 1e integrals")]]
    void compute_overlap_shell_pair(const Shell& shell_a,
                                    const Shell& shell_b,
                                    OverlapBuffer& buffer);

    /// @brief Compute kinetic integrals for a shell pair on GPU
    /// @param shell_a First shell (bra)
    /// @param shell_b Second shell (ket)
    /// @param buffer Output buffer for this shell pair
    [[deprecated("Use compute_shell_set_pair() for batched 1e integrals")]]
    void compute_kinetic_shell_pair(const Shell& shell_a,
                                    const Shell& shell_b,
                                    KineticBuffer& buffer);

    /// @brief Compute nuclear attraction integrals for a shell pair on GPU
    /// @param shell_a First shell (bra)
    /// @param shell_b Second shell (ket)
    /// @param charges Point charge parameters
    /// @param buffer Output buffer for this shell pair
    [[deprecated("Use compute_shell_set_pair() for batched 1e integrals")]]
    void compute_nuclear_shell_pair(const Shell& shell_a,
                                    const Shell& shell_b,
                                    const PointChargeParams& charges,
                                    NuclearBuffer& buffer);

    // =========================================================================
    // Shell Quartet Computation (Two-Electron)
    // =========================================================================

    /// @brief Compute ERIs for a shell quartet on GPU
    /// @param shell_a First bra shell
    /// @param shell_b Second bra shell
    /// @param shell_c First ket shell
    /// @param shell_d Second ket shell
    /// @param buffer Output buffer for this shell quartet
    [[deprecated("Use compute_shell_set_quartet() or compute_eri_batch_device() for batched 2e integrals")]]
    void compute_eri_shell_quartet(const Shell& shell_a,
                                   const Shell& shell_b,
                                   const Shell& shell_c,
                                   const Shell& shell_d,
                                   TwoElectronBuffer<0>& buffer);

    // =========================================================================
    // Generic Dispatch Methods (for EngineBackend concept satisfaction)
    // =========================================================================

    /// @brief Compute one-electron integral for a single shell pair
    ///
    /// Generic dispatch method that routes to the appropriate operator-specific
    /// method based on the operator kind. Required for EngineBackend concept.
    ///
    /// @param op The one-electron operator
    /// @param shell_a First shell (bra)
    /// @param shell_b Second shell (ket)
    /// @param buffer Output buffer for this shell pair
    [[deprecated("Use compute_shell_set_pair() for batched 1e integrals")]]
    void compute_1e_shell_pair(const Operator& op,
                               const Shell& shell_a,
                               const Shell& shell_b,
                               OneElectronBuffer<0>& buffer);

    /// @brief Compute two-electron integral for a single shell quartet
    ///
    /// Generic dispatch method that routes to the appropriate operator-specific
    /// method based on the operator kind. Required for EngineBackend concept.
    ///
    /// @param op The two-electron operator (e.g., Coulomb)
    /// @param shell_a First bra shell
    /// @param shell_b Second bra shell
    /// @param shell_c First ket shell
    /// @param shell_d Second ket shell
    /// @param buffer Output buffer for this shell quartet
    [[deprecated("Use compute_shell_set_quartet() or compute_eri_batch_device() for batched 2e integrals")]]
    void compute_2e_shell_quartet(const Operator& op,
                                   const Shell& shell_a,
                                   const Shell& shell_b,
                                   const Shell& shell_c,
                                   const Shell& shell_d,
                                   TwoElectronBuffer<0>& buffer);

    /// @brief Compute one-electron integrals for a ShellSetPair
    ///
    /// Generic dispatch method for batched shell pair computation.
    /// Required for EngineBackend concept.
    ///
    /// @param op The one-electron operator
    /// @param pair ShellSetPair containing shells to process
    /// @param result Output matrix (nbf x nbf, row-major)
    void compute_shell_set_pair(const Operator& op,
                                 const ShellSetPair& pair,
                                 std::vector<Real>& result);

    // =========================================================================
    // Fused Compute-and-Consume
    // =========================================================================

    /// @brief Fused two-electron integral computation and consumption
    ///
    /// Iterates over all shell quartets, computes ERIs on GPU, downloads
    /// results, and calls the consumer's accumulate() method.
    ///
    /// @tparam Consumer Type with an accumulate method
    /// @param consumer The consumer object (e.g., FockBuilder)
    template<typename Consumer>
    [[deprecated("Use compute_shell_set_quartet() or compute_eri_batch_device() for batched 2e integrals")]]
    void compute_and_consume_eri(Consumer& consumer);

    // =========================================================================
    // ShellSetQuartet-Based Computation (Phase 4.5: True Batching)
    // =========================================================================

    /// @brief Compute ERIs for a ShellSetQuartet keeping results on device
    ///
    /// Performs batched ERI computation for all quartets in the ShellSetQuartet,
    /// keeping results on the device for subsequent device-side accumulation.
    /// The returned handle owns the execution slot, so the device pointer stays
    /// valid until the handle is destroyed.
    ///
    /// @param quartet ShellSetQuartet containing shells to process
    /// @return Owning handle for the device-resident ERI batch
    [[nodiscard]] DeviceEriBatch compute_eri_batch_device_handle(
        const ShellSetQuartet& quartet);

    /// @brief Unsupported raw-pointer ERI batch API kept for alpha compatibility
    ///
    /// @deprecated The raw pointer is not lifetime-safe. Use
    ///             compute_eri_batch_device_handle() instead.
    [[deprecated("Unsupported for alpha: use compute_eri_batch_device_handle()")]]
    void compute_eri_batch_device(const ShellSetQuartet& quartet,
                                  double*& d_eri_output,
                                  size_t& eri_count);

    /// @brief Compute two-electron integrals for a ShellSetQuartet (non-consumer)
    ///
    /// Returns an IntegralBuffer with all computed integrals. This bypasses
    /// Engine dispatch and always executes on GPU, with D2H transfer to
    /// populate the host-side IntegralBuffer.
    ///
    /// @param op The two-electron operator
    /// @param quartet ShellSetQuartet to compute
    /// @return IntegralBuffer with computed integrals and metadata
    [[nodiscard]] IntegralBuffer compute_batch(
        const Operator& op,
        const ShellSetQuartet& quartet) {

        if (!op.is_two_electron()) {
            throw InvalidArgumentException(
                "CudaEngine::compute_batch requires a two-electron operator, got: " +
                std::string(to_string(op.kind())));
        }

        IntegralBuffer result;
        result.set_am(quartet.La(), quartet.Lb(), quartet.Lc(), quartet.Ld());

        const auto& bra = quartet.bra_pair();
        const auto& ket = quartet.ket_pair();
        const auto& set_a = bra.shell_set_a();
        const auto& set_b = bra.shell_set_b();
        const auto& set_c = ket.shell_set_a();
        const auto& set_d = ket.shell_set_b();

        const int nf_a = n_cartesian(set_a.angular_momentum());
        const int nf_b = n_cartesian(set_b.angular_momentum());
        const int nf_c = n_cartesian(set_c.angular_momentum());
        const int nf_d = n_cartesian(set_d.angular_momentum());

        const Size n_ints_per_quartet = static_cast<Size>(nf_a * nf_b * nf_c * nf_d);
        const Size total_quartets = set_a.n_shells() * set_b.n_shells() *
                                     set_c.n_shells() * set_d.n_shells();
        result.reserve_2e(n_ints_per_quartet * total_quartets, total_quartets);

        DeviceEriBatch batch = compute_eri_batch_device_handle(quartet);
        if (!batch) {
            return result;
        }

        // Download to host using slot's staging buffer and stream
        auto& slot = batch.slot();
        slot.h_2e_staging.resize(batch.size());
        cudaMemcpyAsync(slot.h_2e_staging.data(), batch.data(),
                        batch.size() * sizeof(double), cudaMemcpyDeviceToHost,
                        batch.stream());
        cudaStreamSynchronize(batch.stream());

        // Populate IntegralBuffer
        size_t flat_idx = 0;
        for (Size ia = 0; ia < set_a.n_shells(); ++ia) {
            const auto& shell_a = set_a.shell(ia);
            for (Size ib = 0; ib < set_b.n_shells(); ++ib) {
                const auto& shell_b = set_b.shell(ib);
                for (Size ic = 0; ic < set_c.n_shells(); ++ic) {
                    const auto& shell_c = set_c.shell(ic);
                    for (Size id = 0; id < set_d.n_shells(); ++id) {
                        const auto& shell_d = set_d.shell(id);

                        const size_t quartet_size = static_cast<size_t>(
                            shell_a.n_functions() * shell_b.n_functions() *
                            shell_c.n_functions() * shell_d.n_functions());
                        result.append_quartet(
                            std::span<const double>(slot.h_2e_staging.data() + flat_idx, quartet_size),
                            shell_a.function_index(),
                            shell_b.function_index(),
                            shell_c.function_index(),
                            shell_d.function_index(),
                            shell_a.n_functions(),
                            shell_b.n_functions(),
                            shell_c.n_functions(),
                            shell_d.n_functions());

                        flat_idx += quartet_size;
                    }
                }
            }
        }

        return result;
    }

    /// @brief Compute and consume two-electron integrals for a ShellSetQuartet
    ///
    /// If Consumer is GPU-capable (e.g., GpuFockBuilder), uses device-side
    /// accumulation. Otherwise, falls back to download + CPU consumer.
    ///
    /// @tparam Consumer Type with an accumulate method
    /// @param op The two-electron operator
    /// @param quartet ShellSetQuartet containing shells to process
    /// @param consumer Consumer object (e.g., FockBuilder or GpuFockBuilder)
    template<typename Consumer>
    void compute_shell_set_quartet(const Operator& op,
                                    const ShellSetQuartet& quartet,
                                    Consumer& consumer,
                                    bool canonical_symmetry = false);

    // =========================================================================
    // Pipelined ERI Computation (Multi-Stream)
    // =========================================================================

    /// @brief Callback type for pipelined ERI computation
    ///
    /// Called for each completed batch with the computed ERI values and
    /// the corresponding ShellSetQuartet metadata.
    using EriCallback = std::function<void(std::span<const double> eri_chunk,
                                           const ShellSetQuartet& quartet)>;

    /// @brief Compute all ERIs using a multi-stream pipeline, scattering into a host tensor
    ///
    /// Uses a ring buffer of pipeline slots with separate compute and transfer
    /// streams to overlap kernel execution with D2H transfers. Adaptively
    /// dispatches to the warp-per-quartet kernel for high-AM quartets.
    ///
    /// @param result Output vector sized for the full ERI tensor (flat, row-major)
    /// @param config Pipeline configuration parameters
    void compute_eri_pipelined(std::vector<double>& result,
                               const EriPipelineConfig& config = {});

    /// @brief Compute all ERIs using a multi-stream pipeline with user callback
    ///
    /// Each completed batch invokes the callback with the ERI chunk and quartet
    /// metadata. The callback can perform Fock accumulation, disk I/O, etc.
    ///
    /// @param callback User function called for each completed ERI batch
    /// @param config Pipeline configuration parameters
    void compute_eri_pipelined(EriCallback callback,
                               const EriPipelineConfig& config = {});

    /// @brief Compute all ERIs using device-side scatter (no host scatter overhead)
    ///
    /// Keeps the entire ERI tensor on device. For each ShellSetQuartet, launches
    /// the ERI kernel followed by a scatter kernel that atomically accumulates
    /// results into a device-resident 4D tensor. A single bulk D2H copy at the
    /// end transfers the full tensor to host. This avoids the per-batch pinned
    /// scatter bottleneck seen with compute_eri_pipelined for large tensors.
    ///
    /// @param result Output vector sized for the full ERI tensor (flat, row-major)
    /// @param config Pipeline configuration parameters (n_slots used for stream count)
    void compute_eri_device_scatter(std::vector<double>& result,
                                    const EriPipelineConfig& config = {});

    // =========================================================================
    // Resource Management
    // =========================================================================

    /// @brief Synchronize all pending GPU operations
    void synchronize();

    /// @brief Get the CUDA stream used by this engine
    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

private:
    friend class engine::MultiGPUEngine;

    /// @brief Initialize GPU resources
    void initialize();

    /// @brief Cleanup GPU resources
    void cleanup();

    /// @brief Generic one-electron matrix computation
    template<typename KernelDispatcher>
    void compute_1e_matrix_impl(KernelDispatcher dispatcher,
                                std::vector<Real>& result);

    const BasisSet* basis_;           ///< Pointer to the basis set
    bool initialized_{false};         ///< Whether GPU resources are initialized
    cudaStream_t stream_{nullptr};    ///< CUDA stream for kernel execution
    double* d_boys_coeffs_{nullptr};  ///< Device Boys function coefficients

    // Device memory cache for shell data (pointer to avoid move issues with mutex)
    std::unique_ptr<basis::ShellSetDeviceCache> shell_cache_;

    // Optimal kernel dispatch table (Phase: Optimal Dispatch)
    std::unique_ptr<kernels::OptimalDispatchTable> dispatch_table_;

    // Runtime GPU resource tracker for occupancy-aware dispatch (Phase 4.2)
    device::DeviceResourceTracker* tracker_{nullptr};

    // CPU fallback for small-batch dispatch (Step 10.2)
    engine::CpuEngine* cpu_fallback_{nullptr};

    // Dispatch thresholds for small-batch guard (Step 10.2)
    Size min_gpu_batch_size_{16};  ///< Cached from DispatchConfig

    /// @brief Dispatch a one-electron kernel using the optimal dispatch table
    /// @param op_kind The operator kind (Overlap, Kinetic, Nuclear)
    /// @param pair_device Device data for the shell set pair
    /// @param d_output Device output buffer
    /// @param stream CUDA stream for kernel launch
    /// @param charge_data Device charge data (only used for Nuclear)
    void dispatch_1e_optimal(OperatorKind op_kind,
                             const basis::ShellSetPairDeviceData& pair_device,
                             double* d_output,
                             cudaStream_t stream,
                             const operators::DevicePointChargeData* charge_data = nullptr);

    /// @brief Dispatch a two-electron kernel using the optimal dispatch table
    /// @param quartet_device Device data for the shell set quartet
    /// @param d_output Device output buffer
    /// @param stream CUDA stream for kernel launch
    void dispatch_2e_optimal(const basis::ShellSetQuartetDeviceData& quartet_device,
                             double* d_output,
                             cudaStream_t stream);

    // Pool of GPU execution slots for concurrent thread access
    std::unique_ptr<memory::GpuSlotPool> slot_pool_;

    /// @brief Number of slots to create (from DispatchConfig)
    Size n_gpu_slots_{4};

    /// @brief Slot-aware overload of compute_eri_batch_device
    void compute_eri_batch_device(const ShellSetQuartet& quartet,
                                   double*& d_eri_output,
                                   size_t& eri_count,
                                   memory::GpuExecutionSlot& slot);

    /// @brief Scatter 1e integrals from flat buffer to result matrix
    void scatter_1e_to_matrix(const ShellSetPair& pair,
                              const std::vector<double>& flat_output,
                              std::vector<Real>& result);

    /// @brief Scatter 2e integrals from SoA-layout flat buffer to result tensor
    ///
    /// SoA layout: flat_output[component * n_quartets + quartet_idx]
    void scatter_2e_soa_to_matrix(const ShellSetQuartet& quartet,
                                   const std::vector<double>& flat_output,
                                   std::vector<double>& result);

    // ---- Pipeline internals ----

    /// @brief A single slot in the ring-buffer ERI pipeline
    struct PipelineSlot {
        double* d_output = nullptr;              ///< Device output segment
        memory::PinnedBuffer<double> h_pinned;   ///< Pinned host staging buffer
        size_t capacity = 0;                     ///< Current buffer capacity (doubles)
        size_t output_size = 0;                  ///< Actual output size for current batch
        memory::EventHandle kernel_done;         ///< Event: kernel finished writing
        memory::EventHandle download_done;       ///< Event: D2H transfer complete
        const ShellSetQuartet* quartet = nullptr; ///< Which quartet this slot holds
        bool in_flight = false;                  ///< Whether this slot has pending work

        PipelineSlot();
        ~PipelineSlot();
        PipelineSlot(PipelineSlot&&) noexcept;
        PipelineSlot& operator=(PipelineSlot&&) noexcept;
        PipelineSlot(const PipelineSlot&) = delete;
        PipelineSlot& operator=(const PipelineSlot&) = delete;

        /// @brief Ensure device and pinned buffers have sufficient capacity
        void ensure_capacity(size_t required);
    };

    /// @brief Core pipeline implementation shared by both overloads
    ///
    /// @param process_slot Callback invoked for each completed slot to consume its results.
    ///                     Receives the slot and the pipeline's basis pointer.
    /// @param config Pipeline configuration
    template<typename SlotProcessor>
    void compute_eri_pipelined_impl(SlotProcessor process_slot,
                                     const EriPipelineConfig& config);

    /// @brief Scatter a completed pipeline slot's ERIs into a flat result vector
    void scatter_eri_slot_to_tensor(const PipelineSlot& slot,
                                    std::vector<double>& result);
};

// =============================================================================
// AM-Type Batching Infrastructure
// =============================================================================
//
// Shell quartets are grouped ("batched") by their angular momentum type
// (la, lb, lc, ld) before being submitted to GPU kernels.  Launching
// many quartets of the *same* AM type in a single kernel improves GPU
// occupancy because:
//   - The kernel register allocation is fixed per AM type, so all
//     threads in a launch use the same register count.
//   - Warp divergence within the kernel is eliminated.
//   - The scheduler can fill all SMs with uniform work.
//
// The batching structure collects quartet metadata during the shell
// iteration and groups it by an AM key.  Once a group reaches a
// configurable batch size, or at the end of the iteration, the group
// is flushed to the GPU as a single kernel launch.

namespace detail {

/// @brief Key for grouping shell quartets by AM type
struct AmQuartetKey {
    int la, lb, lc, ld;

    bool operator==(const AmQuartetKey& o) const noexcept {
        return la == o.la && lb == o.lb && lc == o.lc && ld == o.ld;
    }
};

/// @brief Hash functor for AmQuartetKey
struct AmQuartetKeyHash {
    std::size_t operator()(const AmQuartetKey& k) const noexcept {
        // Pack four small ints (0-6 each) into a single size_t.
        // AM values fit in 3 bits, so 4 * 3 = 12 bits -- no collisions.
        return static_cast<std::size_t>(k.la)
             | (static_cast<std::size_t>(k.lb) << 4)
             | (static_cast<std::size_t>(k.lc) << 8)
             | (static_cast<std::size_t>(k.ld) << 12);
    }
};

/// @brief Metadata for a single shell quartet in a batch
struct QuartetDescriptor {
    Size shell_i, shell_j, shell_k, shell_l;  ///< Shell indices
    Index fi, fj, fk, fl;                      ///< Basis function offsets
    int na, nb, nc, nd;                        ///< Number of functions per shell
};

/// @brief A batch of shell quartets with the same AM type
struct AmBatch {
    AmQuartetKey key;
    std::vector<QuartetDescriptor> quartets;
};

}  // namespace detail

// =============================================================================
// Template Implementation
// =============================================================================

template<typename Consumer>
void CudaEngine::compute_and_consume_eri(Consumer& consumer) {
    static_assert(EriConsumer<Consumer>,
        "Consumer must provide accumulate(const TwoElectronBuffer<0>&, "
        "Index, Index, Index, Index, int, int, int, int)");

    const Size n_shells = basis_->n_shells();

    // ---- Phase 1: Collect and group shell quartets by AM type ----
    //
    // Rather than launching one kernel per quartet, we group quartets
    // by their (la, lb, lc, ld) AM type.  All quartets in a group use
    // the same generated kernel, enabling batched launches.

    using detail::AmQuartetKey;
    using detail::AmQuartetKeyHash;
    using detail::QuartetDescriptor;
    using detail::AmBatch;

    std::unordered_map<AmQuartetKey, std::vector<QuartetDescriptor>,
                       AmQuartetKeyHash> am_groups;

    for (Size i = 0; i < n_shells; ++i) {
        const auto& shell_a = basis_->shell(i);
        const int la = shell_a.angular_momentum();
        const Index fi = shell_a.function_index();
        const int na = shell_a.n_functions();

        for (Size j = 0; j < n_shells; ++j) {
            const auto& shell_b = basis_->shell(j);
            const int lb = shell_b.angular_momentum();
            const Index fj = shell_b.function_index();
            const int nb = shell_b.n_functions();

            for (Size k = 0; k < n_shells; ++k) {
                const auto& shell_c = basis_->shell(k);
                const int lc = shell_c.angular_momentum();
                const Index fk = shell_c.function_index();
                const int nc = shell_c.n_functions();

                for (Size l = 0; l < n_shells; ++l) {
                    const auto& shell_d = basis_->shell(l);
                    const int ld = shell_d.angular_momentum();
                    const Index fl = shell_d.function_index();
                    const int nd = shell_d.n_functions();

                    AmQuartetKey key{la, lb, lc, ld};
                    am_groups[key].push_back(
                        QuartetDescriptor{i, j, k, l, fi, fj, fk, fl,
                                          na, nb, nc, nd});
                }
            }
        }
    }

    // ---- Phase 2: Process each AM group ----
    //
    // Within each group all quartets dispatch to the same generated
    // kernel.  Currently we still launch one kernel per quartet (the
    // generated kernels process one ShellSetQuartet at a time), but
    // grouping ensures temporal locality of the same kernel binary in
    // the instruction cache.
    //
    // TODO(Phase 7.4): Upgrade to true batch kernel launches where
    //   the generated kernel iterates over an array of quartet
    //   descriptors rather than a single quartet.  This would require
    //   codegen changes to emit batch-aware kernels.

    TwoElectronBuffer<0> buffer;

    for (auto& [key, descs] : am_groups) {
        for (const auto& desc : descs) {
            const auto& shell_a = basis_->shell(desc.shell_i);
            const auto& shell_b = basis_->shell(desc.shell_j);
            const auto& shell_c = basis_->shell(desc.shell_k);
            const auto& shell_d = basis_->shell(desc.shell_l);

            // Compute ERIs on GPU
            compute_eri_shell_quartet(shell_a, shell_b,
                                      shell_c, shell_d, buffer);

            // Pass to consumer
            consumer.accumulate(buffer, desc.fi, desc.fj, desc.fk, desc.fl,
                                desc.na, desc.nb, desc.nc, desc.nd);
        }
    }
}

// =============================================================================
// Template Implementation: compute_shell_set_quartet
// =============================================================================

template<typename Consumer>
void CudaEngine::compute_shell_set_quartet(const Operator& op,
                                            const ShellSetQuartet& quartet,
                                            Consumer& consumer,
                                            bool canonical_symmetry) {
    static_assert(EriConsumer<Consumer>,
        "Consumer must provide accumulate(const TwoElectronBuffer<0>&, "
        "Index, Index, Index, Index, int, int, int, int)");

    if (!op.is_two_electron()) {
        throw std::invalid_argument(
            "CudaEngine::compute_shell_set_quartet: requires two-electron operator, got: " +
            std::string(to_string(op.kind())));
    }

    // --- Small-batch CPU fallback guard (Step 10.2) ---
    {
        const auto& set_a = quartet.bra_pair().shell_set_a();
        const auto& set_b = quartet.bra_pair().shell_set_b();
        const auto& set_c = quartet.ket_pair().shell_set_a();
        const auto& set_d = quartet.ket_pair().shell_set_b();
        const bool ij_same = (&set_a == &set_b);
        const bool kl_same = (&set_c == &set_d);
        const bool braket_same =
            (&quartet.bra_pair().shell_set_a() == &quartet.ket_pair().shell_set_a()) &&
            (&quartet.bra_pair().shell_set_b() == &quartet.ket_pair().shell_set_b());
        const Size n_quartets = set_a.n_shells() * set_b.n_shells() *
                                set_c.n_shells() * set_d.n_shells();
        if (n_quartets < min_gpu_batch_size_ && cpu_fallback_ != nullptr) {
#ifdef LIBACCINT_TRACE_DISPATCH
            LIBACCINT_LOG_DEBUG("CudaEngine",
                "Batch too small (n_quartets=" + std::to_string(n_quartets) +
                " < min=" + std::to_string(min_gpu_batch_size_) +
                "), CPU fallback");
#endif
            if constexpr (SymmetryAwareConsumer<Consumer>) {
                if (canonical_symmetry) {
                    TwoElectronBuffer<0> buffer;
                    for (Size i = 0; i < set_a.n_shells(); ++i) {
                        const auto& shell_a = set_a.shell(i);
                        const Index fi = shell_a.function_index();
                        const int na = shell_a.n_functions();

                        for (Size j = 0; j < set_b.n_shells(); ++j) {
                            const auto& shell_b = set_b.shell(j);
                            const Index fj = shell_b.function_index();
                            const int nb = shell_b.n_functions();

                            for (Size k = 0; k < set_c.n_shells(); ++k) {
                                const auto& shell_c = set_c.shell(k);
                                const Index fk = shell_c.function_index();
                                const int nc = shell_c.n_functions();

                                for (Size l = 0; l < set_d.n_shells(); ++l) {
                                    const auto& shell_d = set_d.shell(l);
                                    const Index fl = shell_d.function_index();
                                    const int nd = shell_d.n_functions();

                                    cpu_fallback_->compute_2e_shell_quartet(
                                        op, shell_a, shell_b, shell_c, shell_d, buffer);
                                    consumer.accumulate_symmetric(
                                        buffer, fi, fj, fk, fl, na, nb, nc, nd,
                                        ij_same, kl_same, braket_same);
                                }
                            }
                        }
                    }
                    return;
                }
            }
            cpu_fallback_->compute_shell_set_quartet(op, quartet, consumer);
            return;
        }
    }

    // Check if consumer is GpuFockBuilder (supports device-side accumulation)
    if constexpr (std::is_same_v<Consumer, consumers::GpuFockBuilder>) {
        // Phase 4.5: True batched execution with device-side accumulation
        // ERIs computed on device, accumulated without host transfer
        DeviceEriBatch batch = compute_eri_batch_device_handle(quartet);
        if (batch) {
            // Get the quartet device data for the accumulation kernel
            const auto& set_a = quartet.bra_pair().shell_set_a();
            const auto& set_b = quartet.bra_pair().shell_set_b();
            const auto& set_c = quartet.ket_pair().shell_set_a();
            const auto& set_d = quartet.ket_pair().shell_set_b();

            const auto& data_a = shell_cache_->get_or_upload(set_a, batch.stream());
            const auto& data_b = shell_cache_->get_or_upload(set_b, batch.stream());
            const auto& data_c = shell_cache_->get_or_upload(set_c, batch.stream());
            const auto& data_d = shell_cache_->get_or_upload(set_d, batch.stream());

            basis::ShellSetQuartetDeviceData quartet_device;
            quartet_device.a = data_a;
            quartet_device.b = data_b;
            quartet_device.c = data_c;
            quartet_device.d = data_d;

            // Keep the builder's stream semantics intact while ordering it
            // behind the ERI producer stream.
            memory::EventHandle eri_ready;
            eri_ready.record(batch.stream());
            LIBACCINT_CUDA_CHECK(
                cudaStreamWaitEvent(consumer.stream(), eri_ready.get(), 0));

            // Device-side Fock accumulation
            consumer.accumulate_device_eri_batch(
                batch.data(), quartet_device, basis_->n_basis_functions());
        }
    } else {
        // Fallback for non-GPU consumers: compute on GPU, download, use CPU consumer
        DeviceEriBatch batch = compute_eri_batch_device_handle(quartet);
        if (!batch) {
            return;
        }

        // Download ERIs to host using slot's staging buffer and stream
        auto& slot = batch.slot();
        slot.h_2e_staging.resize(batch.size());
        cudaMemcpyAsync(slot.h_2e_staging.data(), batch.data(),
                        batch.size() * sizeof(double), cudaMemcpyDeviceToHost,
                        batch.stream());
        cudaStreamSynchronize(batch.stream());

        // Scatter to consumer via individual accumulate calls
        const auto& set_a = quartet.bra_pair().shell_set_a();
        const auto& set_b = quartet.bra_pair().shell_set_b();
        const auto& set_c = quartet.ket_pair().shell_set_a();
        const auto& set_d = quartet.ket_pair().shell_set_b();
        const bool ij_same = (&set_a == &set_b);
        const bool kl_same = (&set_c == &set_d);
        const bool braket_same =
            (&quartet.bra_pair().shell_set_a() == &quartet.ket_pair().shell_set_a()) &&
            (&quartet.bra_pair().shell_set_b() == &quartet.ket_pair().shell_set_b());

        const int na_funcs = n_cartesian(set_a.angular_momentum());
        const int nb_funcs = n_cartesian(set_b.angular_momentum());
        const int nc_funcs = n_cartesian(set_c.angular_momentum());
        const int nd_funcs = n_cartesian(set_d.angular_momentum());

        const Size funcs_per_quartet = static_cast<Size>(na_funcs) * nb_funcs * nc_funcs * nd_funcs;

        TwoElectronBuffer<0> buffer;
        buffer.resize(na_funcs, nb_funcs, nc_funcs, nd_funcs);

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
                                        buffer(a, b, c, d) = slot.h_2e_staging[
                                            flat_idx + a * nb_funcs * nc_funcs * nd_funcs +
                                            b * nc_funcs * nd_funcs + c * nd_funcs + d];
                                    }
                                }
                            }
                        }

                        if constexpr (SymmetryAwareConsumer<Consumer>) {
                            if (canonical_symmetry) {
                                consumer.accumulate_symmetric(
                                    buffer, fi, fj, fk, fl,
                                    na_funcs, nb_funcs, nc_funcs, nd_funcs,
                                    ij_same, kl_same, braket_same);
                            } else {
                                consumer.accumulate(
                                    buffer, fi, fj, fk, fl,
                                    na_funcs, nb_funcs, nc_funcs, nd_funcs);
                            }
                        } else {
                            consumer.accumulate(
                                buffer, fi, fj, fk, fl,
                                na_funcs, nb_funcs, nc_funcs, nd_funcs);
                        }

                        flat_idx += funcs_per_quartet;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Template Implementation: compute_eri_pipelined_impl
// =============================================================================

template<typename SlotProcessor>
void CudaEngine::compute_eri_pipelined_impl(SlotProcessor process_slot,
                                             const EriPipelineConfig& config) {
    const auto& quartets = basis_->shell_set_quartets();
    if (quartets.empty()) return;

    const size_t n_slots = config.n_slots;
    if (n_slots == 0) {
        throw InvalidArgumentException(
            "CudaEngine::compute_eri_pipelined requires config.n_slots >= 1");
    }

    // Create compute stream pool and dedicated transfer stream
    memory::StreamPool compute_pool(n_slots);
    memory::StreamHandle transfer_stream;

    // Allocate pipeline slots
    std::vector<PipelineSlot> slots(n_slots);

    // Track which compute stream each slot uses (round-robin index)
    std::vector<memory::StreamHandle*> slot_streams(n_slots, nullptr);

    size_t next_slot = 0;       // Next slot to assign (round-robin)
    size_t oldest_slot = 0;     // Oldest in-flight slot (for drain order)
    size_t in_flight_count = 0; // Number of slots currently in-flight

    for (const auto& q : quartets) {
        // Get ShellSets from quartet
        const auto& set_a = q.bra_pair().shell_set_a();
        const auto& set_b = q.bra_pair().shell_set_b();
        const auto& set_c = q.ket_pair().shell_set_a();
        const auto& set_d = q.ket_pair().shell_set_b();

        if (set_a.n_shells() == 0 || set_b.n_shells() == 0 ||
            set_c.n_shells() == 0 || set_d.n_shells() == 0) {
            continue;
        }

        // If all slots are in-flight, wait for the oldest one and process it
        if (in_flight_count == n_slots) {
            auto& old_slot = slots[oldest_slot];
            old_slot.download_done.synchronize();
            process_slot(old_slot);
            old_slot.in_flight = false;

            // Release compute stream back to pool
            if (slot_streams[oldest_slot]) {
                compute_pool.release(*slot_streams[oldest_slot]);
                slot_streams[oldest_slot] = nullptr;
            }

            in_flight_count--;
            oldest_slot = (oldest_slot + 1) % n_slots;
        }

        // Use the next_slot (round-robin)
        auto& slot = slots[next_slot];

        // Calculate output size for this quartet
        const int na = n_cartesian(set_a.angular_momentum());
        const int nb = n_cartesian(set_b.angular_momentum());
        const int nc = n_cartesian(set_c.angular_momentum());
        const int nd = n_cartesian(set_d.angular_momentum());
        const size_t n_shell_quartets = static_cast<size_t>(set_a.n_shells()) *
                                         set_b.n_shells() * set_c.n_shells() * set_d.n_shells();
        const size_t output_size = n_shell_quartets * na * nb * nc * nd;

        // Ensure slot buffers are large enough
        slot.ensure_capacity(output_size);
        slot.output_size = output_size;
        slot.quartet = &q;

        // Acquire a compute stream
        auto& compute_stream = compute_pool.acquire();
        slot_streams[next_slot] = &compute_stream;

        // Upload shell data (uses cache, so this is cheap if already uploaded)
        const auto& data_a = shell_cache_->get_or_upload(set_a, compute_stream.get());
        const auto& data_b = shell_cache_->get_or_upload(set_b, compute_stream.get());
        const auto& data_c = shell_cache_->get_or_upload(set_c, compute_stream.get());
        const auto& data_d = shell_cache_->get_or_upload(set_d, compute_stream.get());

        // Build quartet device data
        basis::ShellSetQuartetDeviceData quartet_device;
        quartet_device.a = data_a;
        quartet_device.b = data_b;
        quartet_device.c = data_c;
        quartet_device.d = data_d;

        // Launch kernel via optimal dispatch table
        dispatch_2e_optimal(quartet_device, slot.d_output, compute_stream.get());

        // Record kernel completion event on compute stream
        slot.kernel_done.record(compute_stream.get());

        // Queue D2H transfer on the dedicated transfer stream
        transfer_stream.wait_event(slot.kernel_done.get());
        cudaMemcpyAsync(slot.h_pinned.data(), slot.d_output,
                        output_size * sizeof(double), cudaMemcpyDeviceToHost,
                        transfer_stream.get());

        // Record download completion event on transfer stream
        slot.download_done.record(transfer_stream.get());

        slot.in_flight = true;
        in_flight_count++;
        next_slot = (next_slot + 1) % n_slots;
    }

    // Drain all remaining in-flight slots
    for (size_t i = 0; i < in_flight_count; ++i) {
        auto& slot = slots[oldest_slot];
        slot.download_done.synchronize();
        process_slot(slot);
        slot.in_flight = false;

        if (slot_streams[oldest_slot]) {
            compute_pool.release(*slot_streams[oldest_slot]);
            slot_streams[oldest_slot] = nullptr;
        }

        oldest_slot = (oldest_slot + 1) % n_slots;
    }
}

// Static assertion to verify CudaEngine satisfies EngineBackend concept
static_assert(EngineBackend<CudaEngine>, "CudaEngine must satisfy EngineBackend concept");

}  // namespace libaccint

#endif  // LIBACCINT_USE_CUDA
