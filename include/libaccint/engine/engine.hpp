// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file engine.hpp
/// @brief Unified computation orchestrator for molecular integrals
///
/// The Engine class provides a single unified interface that automatically
/// routes work to the optimal backend (CPU or GPU) based on heuristics
/// and user hints. It owns a CpuEngine (always available) and optionally
/// a CudaEngine (when CUDA is enabled and GPU is available).

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/basis/shell_set_quartet_utils.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/config.hpp>
#include <libaccint/consumers/consumer_concepts.hpp>
#include <libaccint/core/backend.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/diagnostics/batch_trace.hpp>
#include <libaccint/engine/cpu_engine.hpp>
#include <libaccint/engine/dispatch_policy.hpp>
#include <libaccint/engine/engine_backend.hpp>
#include <libaccint/engine/integral_buffer.hpp>
#include <libaccint/engine/thread_config.hpp>
#include <libaccint/memory/buffer_pool.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/screening/density_screening.hpp>
#include <libaccint/screening/schwarz_bounds.hpp>
#include <libaccint/screening/screening_options.hpp>
#include <libaccint/utils/error_handling.hpp>

#if LIBACCINT_USE_CUDA
#include <libaccint/engine/cuda_engine.hpp>
#endif

#include <concepts>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace libaccint {

/// @brief Unified computation orchestrator for molecular integrals
///
/// Engine provides a single API that automatically routes work to the optimal
/// backend based on dispatch heuristics. It owns both a CpuEngine (always)
/// and optionally a CudaEngine (when CUDA is available).
///
/// Features:
/// - Intelligent dispatch: Automatically selects CPU or GPU based on work size
/// - User hints: Allow forcing or preferring specific backends
/// - Transparent fallback: Falls back to CPU when GPU is unavailable
/// - Direct backend access: Advanced users can access underlying engines
///
/// Usage:
/// @code
///   BasisSet basis(shells);
///   Engine engine(basis);
///
///   // Automatic dispatch (default)
///   std::vector<Real> S;
///   engine.compute_overlap_matrix(S);
///
///   // Force CPU backend
///   engine.compute_overlap_matrix(S, BackendHint::ForceCPU);
///
///   // Prefer GPU for large computations
///   engine.compute_overlap_matrix(S, BackendHint::PreferGPU);
/// @endcode
class Engine {
public:
    /// @brief Construct an Engine with a BasisSet
    /// @param basis The basis set to use (must remain valid for Engine lifetime)
    /// @param config Dispatch configuration for backend selection heuristics
    explicit Engine(const BasisSet& basis, DispatchConfig config = {})
        : basis_(&basis),
          cpu_engine_(basis),
          dispatch_policy_(config) {
#if LIBACCINT_USE_CUDA
        // Try to initialize CUDA engine if GPU is available
        if (is_backend_available(BackendType::CUDA)) {
            try {
                cuda_engine_.emplace(basis);
                gpu_available_ = true;

                // Wire up CPU fallback and dispatch thresholds (Step 10.2)
                cuda_engine_->set_cpu_fallback(&cpu_engine_);
                cuda_engine_->set_dispatch_config(config);
            } catch (const BackendError&) {
                // GPU not available, fall back to CPU only
                gpu_available_ = false;
            }
        }
#endif
    }

    /// @brief Get the basis set
    [[nodiscard]] const BasisSet& basis() const noexcept { return *basis_; }

    /// @brief Get the maximum angular momentum from the basis set
    [[nodiscard]] int max_angular_momentum() const noexcept {
        return basis_->max_angular_momentum();
    }

    /// @brief Check if GPU backend is available
    [[nodiscard]] bool gpu_available() const noexcept { return gpu_available_; }

    /// @brief Get the dispatch policy
    [[nodiscard]] const DispatchPolicy& dispatch_policy() const noexcept {
        return dispatch_policy_;
    }

    /// @brief Set the dispatch configuration
    void set_dispatch_config(DispatchConfig config) {
        dispatch_policy_.set_config(config);
#if LIBACCINT_USE_CUDA
        if (cuda_engine_) {
            cuda_engine_->set_dispatch_config(config);
        }
#endif
    }

    // =========================================================================
    // Direct Backend Access
    // =========================================================================

    /// @brief Get the CPU engine (always available)
    [[nodiscard]] engine::CpuEngine& cpu_engine() noexcept { return cpu_engine_; }
    [[nodiscard]] const engine::CpuEngine& cpu_engine() const noexcept { return cpu_engine_; }

#if LIBACCINT_USE_CUDA
    /// @brief Get the CUDA engine (nullptr if GPU not available)
    [[nodiscard]] CudaEngine* cuda_engine() noexcept {
        return cuda_engine_ ? &(*cuda_engine_) : nullptr;
    }
    [[nodiscard]] const CudaEngine* cuda_engine() const noexcept {
        return cuda_engine_ ? &(*cuda_engine_) : nullptr;
    }
#endif

    // =========================================================================
    // One-Electron Integrals
    // =========================================================================

    /// @brief Compute one-electron integrals for all shell pairs
    ///
    /// Dispatches to CPU or GPU based on heuristics and user hint.
    ///
    /// @tparam DerivOrder Derivative order (0 = energy)
    /// @param op The one-electron operator (or composed operator)
    /// @param result Output matrix (n_basis_functions x n_basis_functions)
    /// @param hint Backend selection hint (default: Auto)
    template<int DerivOrder = 0>
    void compute_1e(const OneElectronOperator& op,
                    std::vector<Real>& result,
                    BackendHint hint = BackendHint::Auto) {
        static_assert(DerivOrder == 0,
                      "Only DerivOrder == 0 (energy) is supported in this release");

        // For full-basis operations, use dispatch policy
        BackendType backend = dispatch_policy_.select_backend(
            WorkUnitType::FullBasis,
            basis_->n_shells(),
            basis_->max_angular_momentum() * 2,  // rough total AM estimate
            estimate_primitives(),
            hint,
            gpu_available_);

        if (backend == BackendType::CPU) {
            if (use_default_cpu_parallel_1e()) {
                cpu_engine_.compute_1e_parallel<DerivOrder>(
                    op, result, ThreadConfig::effective_threads());
            } else {
                cpu_engine_.compute_1e<DerivOrder>(op, result);
            }
        }
#if LIBACCINT_USE_CUDA
        else if (cuda_engine_) {
            compute_1e<DerivOrder>(op, basis_->shell_set_pairs(), result, hint);
        }
#endif
        else {
            if (use_default_cpu_parallel_1e()) {
                cpu_engine_.compute_1e_parallel<DerivOrder>(
                    op, result, ThreadConfig::effective_threads());
            } else {
                cpu_engine_.compute_1e<DerivOrder>(op, result);
            }
        }
    }

    /// @brief Compute one-electron integrals for a subset of ShellSetPairs
    ///
    /// Iterates the provided pairs and for each pair, iterates the Cartesian
    /// product of shells within the bra and ket ShellSets, computing integrals
    /// and accumulating them into the result matrix. Upper-triangle handling
    /// for self-pairs is applied automatically.
    ///
    /// @tparam DerivOrder Derivative order (0 = energy)
    /// @param op The one-electron operator
    /// @param pairs Span of ShellSetPairs to compute
    /// @param result Output matrix (n_basis_functions x n_basis_functions)
    /// @param hint Backend selection hint (default: Auto)
    template<int DerivOrder = 0>
    void compute_1e(const OneElectronOperator& op,
                    std::span<const ShellSetPair> pairs,
                    std::vector<Real>& result,
                    BackendHint hint = BackendHint::Auto) {
        static_assert(DerivOrder == 0,
                      "Only DerivOrder == 0 (energy) is supported in this release");

        const Size nbf = basis_->n_basis_functions();
        result.assign(nbf * nbf, Real{0.0});
        if (pairs.empty()) {
            return;
        }

        const auto& contributions = op.contributions();
        if (contributions.size() == 1 && contributions.front().scale == Real{1.0}) {
            for (const auto& pair : pairs) {
                compute_shell_set_pair(contributions.front().op, pair, result, hint);
            }
            return;
        }

        std::vector<Real> pair_result(nbf * nbf, Real{0.0});
        for (const auto& contribution : contributions) {
            for (const auto& pair : pairs) {
                std::fill(pair_result.begin(), pair_result.end(), Real{0.0});
                compute_shell_set_pair(contribution.op, pair, pair_result, hint);
                if (contribution.scale == Real{1.0}) {
                    for (Size idx = 0; idx < result.size(); ++idx) {
                        result[idx] += pair_result[idx];
                    }
                } else {
                    for (Size idx = 0; idx < result.size(); ++idx) {
                        result[idx] += contribution.scale * pair_result[idx];
                    }
                }
            }
        }
    }

    /// @brief Compute one-electron integral for a single shell pair
    ///
    /// @param op The one-electron operator
    /// @param shell_a First shell (bra)
    /// @param shell_b Second shell (ket)
    /// @param buffer Output buffer for this shell pair
    /// @param hint Backend selection hint (default: Auto)
    void compute_1e_shell_pair(const Operator& op,
                               const Shell& shell_a,
                               const Shell& shell_b,
                               OneElectronBuffer<0>& buffer,
                               BackendHint hint = BackendHint::Auto) {
        BackendType backend = dispatch_policy_.select_backend(
            WorkUnitType::SingleShellPair,
            1,
            shell_a.angular_momentum() + shell_b.angular_momentum(),
            static_cast<Size>(shell_a.n_primitives() * shell_b.n_primitives()),
            hint,
            gpu_available_);

        if (backend == BackendType::CPU) {
            cpu_engine_.compute_1e_shell_pair(op, shell_a, shell_b, buffer);
        }
#if LIBACCINT_USE_CUDA
        else if (cuda_engine_) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
            cuda_engine_->compute_1e_shell_pair(op, shell_a, shell_b, buffer);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
        }
#endif
        else {
            cpu_engine_.compute_1e_shell_pair(op, shell_a, shell_b, buffer);
        }
    }

    // =========================================================================
    // Two-Electron Integrals
    // =========================================================================

    /// @brief Compute two-electron integral for a single shell quartet
    ///
    /// @param op The two-electron operator (e.g., Coulomb)
    /// @param shell_a First bra shell
    /// @param shell_b Second bra shell
    /// @param shell_c First ket shell
    /// @param shell_d Second ket shell
    /// @param buffer Output buffer for this shell quartet
    /// @param hint Backend selection hint (default: Auto)
    void compute_2e_shell_quartet(const Operator& op,
                                   const Shell& shell_a,
                                   const Shell& shell_b,
                                   const Shell& shell_c,
                                   const Shell& shell_d,
                                   TwoElectronBuffer<0>& buffer,
                                   BackendHint hint = BackendHint::Auto) {
        int total_am = shell_a.angular_momentum() + shell_b.angular_momentum() +
                       shell_c.angular_momentum() + shell_d.angular_momentum();
        Size n_prims = static_cast<Size>(shell_a.n_primitives() * shell_b.n_primitives() *
                                         shell_c.n_primitives() * shell_d.n_primitives());

        BackendType backend = dispatch_policy_.select_backend(
            WorkUnitType::SingleShellQuartet,
            1,
            total_am,
            n_prims,
            hint,
            gpu_available_);

        if (backend == BackendType::CPU) {
            cpu_engine_.compute_2e_shell_quartet(op, shell_a, shell_b, shell_c, shell_d, buffer);
        }
#if LIBACCINT_USE_CUDA
        else if (cuda_engine_) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
            cuda_engine_->compute_2e_shell_quartet(op, shell_a, shell_b, shell_c, shell_d, buffer);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
        }
#endif
        else {
            cpu_engine_.compute_2e_shell_quartet(op, shell_a, shell_b, shell_c, shell_d, buffer);
        }
    }

    // =========================================================================
    // ShellSet-Based Computation
    // =========================================================================

    /// @brief Compute one-electron integrals for a ShellSetPair
    ///
    /// @param op The one-electron operator
    /// @param pair ShellSetPair containing shells to process
    /// @param result Output matrix (nbf x nbf, row-major)
    /// @param hint Backend selection hint (default: Auto)
    void compute_shell_set_pair(const Operator& op,
                                 const ShellSetPair& pair,
                                 std::vector<Real>& result,
                                 BackendHint hint = BackendHint::Auto) {
        Size batch_size = pair.shell_set_a().n_shells() * pair.shell_set_b().n_shells();
        int total_am = pair.shell_set_a().angular_momentum() +
                       pair.shell_set_b().angular_momentum();

        BackendType backend = dispatch_policy_.select_backend(
            WorkUnitType::ShellSetPair,
            batch_size,
            total_am,
            estimate_shell_set_pair_primitives(pair),
            hint,
            gpu_available_);

        if (backend == BackendType::CPU) {
            diagnostics::BatchTracer::instance().trace_1e_dispatch(
                std::string(to_string(op.kind())).c_str(),
                pair.shell_set_a().angular_momentum(),
                pair.shell_set_b().angular_momentum(),
                pair.shell_set_a().n_primitives_per_shell(),
                pair.shell_set_b().n_primitives_per_shell(),
                batch_size, "CPU");
            cpu_engine_.compute_shell_set_pair(op, pair, result);
        }
#if LIBACCINT_USE_CUDA
        else if (cuda_engine_) {
            diagnostics::BatchTracer::instance().trace_1e_dispatch(
                std::string(to_string(op.kind())).c_str(),
                pair.shell_set_a().angular_momentum(),
                pair.shell_set_b().angular_momentum(),
                pair.shell_set_a().n_primitives_per_shell(),
                pair.shell_set_b().n_primitives_per_shell(),
                batch_size, "GPU");
            cuda_engine_->compute_shell_set_pair(op, pair, result);
        }
#endif
        else {
            diagnostics::BatchTracer::instance().trace_1e_dispatch(
                std::string(to_string(op.kind())).c_str(),
                pair.shell_set_a().angular_momentum(),
                pair.shell_set_b().angular_momentum(),
                pair.shell_set_a().n_primitives_per_shell(),
                pair.shell_set_b().n_primitives_per_shell(),
                batch_size, "CPU");
            cpu_engine_.compute_shell_set_pair(op, pair, result);
        }
    }

    /// @brief Compute and consume two-electron integrals for a ShellSetQuartet
    ///
    /// @tparam Consumer Type with an accumulate method
    /// @param op The two-electron operator
    /// @param quartet ShellSetQuartet containing shells to process
    /// @param consumer Consumer object (e.g., FockBuilder)
    /// @param hint Backend selection hint (default: Auto)
    template<typename Consumer>
    void compute_shell_set_quartet(const Operator& op,
                                    const ShellSetQuartet& quartet,
                                    Consumer& consumer,
                                    BackendHint hint = BackendHint::Auto) {
        compute_shell_set_quartet_impl(op, quartet, consumer, hint, false);
    }

    // =========================================================================
    // Fused Compute-and-Consume
    // =========================================================================

    /// @brief Fused two-electron integral computation and consumption
    ///
    /// @tparam Consumer Type with an accumulate method
    /// @param op The two-electron operator (e.g., Coulomb)
    /// @param consumer The consumer object (e.g., FockBuilder)
    /// @param hint Backend selection hint (default: Auto)
    template<typename Consumer>
    void compute_and_consume(const Operator& op, Consumer& consumer,
                             BackendHint hint = BackendHint::Auto) {
        static_assert(TwoElectronConsumer<Consumer>,
            "Consumer must provide accumulate(const TwoElectronBuffer<0>&, "
            "Index, Index, Index, Index, int, int, int, int)");

        if (!op.is_two_electron()) {
            throw InvalidArgumentException(
                "compute_and_consume requires a two-electron operator, got: " +
                std::string(to_string(op.kind())));
        }

        BackendType backend = dispatch_policy_.select_backend(
            WorkUnitType::FullBasis,
            basis_->n_shells(),
            basis_->max_angular_momentum() * 4,
            estimate_primitives(),
            hint,
            gpu_available_);

#if LIBACCINT_USE_CUDA
        if (backend == BackendType::CUDA && cuda_engine_) {
            if constexpr (std::is_same_v<Consumer, consumers::GpuFockBuilder>) {
                const auto& quartets = basis_->shell_set_quartets();
                for (const auto& quartet : quartets) {
                    compute_shell_set_quartet_impl(op, quartet, consumer, hint, false);
                }
            } else if constexpr (SymmetryAwareConsumer<Consumer>) {
                const auto& quartets = basis_->shell_set_quartets();
                for (const auto& quartet : quartets) {
                    compute_shell_set_quartet_impl(op, quartet, consumer, hint, true);
                }
            } else {
                const auto sets = basis_->shell_sets();
                const Size n_sets = sets.size();
                for (Size ia = 0; ia < n_sets; ++ia) {
                    for (Size ib = 0; ib < n_sets; ++ib) {
                        ShellSetPair bra_pair(*sets[ia], *sets[ib]);
                        for (Size ic = 0; ic < n_sets; ++ic) {
                            for (Size id = 0; id < n_sets; ++id) {
                                ShellSetPair ket_pair(*sets[ic], *sets[id]);
                                ShellSetQuartet q(bra_pair, ket_pair);
                                compute_shell_set_quartet_impl(op, q, consumer, hint, false);
                            }
                        }
                    }
                }
            }
            diagnostics::BatchTracer::instance().print_summary();
            return;
        }
#else
        (void)backend;
#endif

        if constexpr (ParallelConsumer<Consumer>) {
            if (use_default_cpu_parallel_2e()) {
                cpu_engine_.compute_and_consume_parallel(
                    op, consumer, ThreadConfig::effective_threads());
            } else {
                cpu_engine_.compute_and_consume(op, consumer);
            }
        } else {
            cpu_engine_.compute_and_consume(op, consumer);
        }
        diagnostics::BatchTracer::instance().print_summary();
    }

    /// @brief Compute and consume two-electron integrals for a provided worklist
    ///
    /// Processes a caller-supplied set of ShellSetQuartets. This enables:
    ///   - Screened execution (caller filters insignificant quartets)
    ///   - Distributed execution (caller partitions quartets across nodes/GPUs)
    ///   - Custom scheduling (caller controls work order)
    ///
    /// Each quartet is dispatched via compute_shell_set_quartet, which
    /// respects the dispatch policy and BackendHint for CPU/GPU routing.
    ///
    /// @tparam Consumer Type with an accumulate method
    /// @param op The two-electron operator (e.g., Coulomb)
    /// @param quartets Span of ShellSetQuartets to compute
    /// @param consumer The consumer object (e.g., FockBuilder)
    /// @param hint Backend selection hint (default: Auto)
    template<typename Consumer>
    void compute_and_consume(const Operator& op,
                             std::span<const ShellSetQuartet> quartets,
                             Consumer& consumer,
                             BackendHint hint = BackendHint::Auto) {
        static_assert(TwoElectronConsumer<Consumer>,
            "Consumer must provide accumulate(const TwoElectronBuffer<0>&, "
            "Index, Index, Index, Index, int, int, int, int)");

        if (!op.is_two_electron()) {
            throw InvalidArgumentException(
                "compute_and_consume requires a two-electron operator, got: " +
                std::string(to_string(op.kind())));
        }

        for (const auto& quartet : quartets) {
            compute_shell_set_quartet(op, quartet, consumer, hint);
        }
        diagnostics::BatchTracer::instance().print_summary();
    }

    /// @brief Parallel two-electron integral computation and consumption
    ///
    /// @tparam Consumer Type with an accumulate method (must be thread-safe)
    /// @param op The two-electron operator (e.g., Coulomb)
    /// @param consumer The consumer object (must be thread-safe)
    /// @param n_threads Number of OpenMP threads to use (0 = auto)
    /// @param hint Backend selection hint (default: Auto)
    template<typename Consumer>
    void compute_and_consume_parallel(const Operator& op, Consumer& consumer,
                                       int n_threads = 0,
                                       BackendHint hint = BackendHint::Auto) {
        static_assert(TwoElectronConsumer<Consumer>,
            "Consumer must provide accumulate(const TwoElectronBuffer<0>&, "
            "Index, Index, Index, Index, int, int, int, int)");
        static_assert(ParallelConsumer<Consumer>,
            "Parallel compute requires a consumer with prepare_parallel(int) "
            "and finalize_parallel()");

        if (!op.is_two_electron()) {
            throw InvalidArgumentException(
                "compute_and_consume_parallel requires a two-electron operator, got: " +
                std::string(to_string(op.kind())));
        }

        // Parallel compute_and_consume always uses CPU with OpenMP
        cpu_engine_.compute_and_consume_parallel(op, consumer, n_threads);
        (void)hint;
    }

    /// @brief Parallel one-electron integral computation
    ///
    /// Parallelizes the shell pair loop across OpenMP threads using
    /// thread-local result matrices with final reduction.
    ///
    /// @tparam DerivOrder Derivative order (0 = energy)
    /// @param op The one-electron operator (or composed operator)
    /// @param result Output matrix (n_basis_functions x n_basis_functions)
    /// @param n_threads Number of OpenMP threads to use (0 = auto)
    template<int DerivOrder = 0>
    void compute_1e_parallel(const OneElectronOperator& op,
                             std::vector<Real>& result,
                             int n_threads = 0) {
        static_assert(DerivOrder == 0,
                      "Only DerivOrder == 0 (energy) is supported in this release");
        cpu_engine_.compute_1e_parallel<DerivOrder>(op, result, n_threads);
    }

    /// @brief Parallel two-electron integral computation with Schwarz screening
    ///
    /// Combines Schwarz screening with OpenMP parallelism. Builds a screened
    /// worklist and distributes it across threads with dynamic scheduling.
    ///
    /// @tparam Consumer Type with an accumulate method (must be thread-safe)
    /// @param op The two-electron operator
    /// @param consumer The consumer object (must be thread-safe)
    /// @param options Screening options (threshold, etc.)
    /// @param n_threads Number of OpenMP threads to use (0 = auto)
    template<typename Consumer>
    void compute_and_consume_screened_parallel(
        const Operator& op, Consumer& consumer,
        const screening::ScreeningOptions& options,
        int n_threads = 0) {
        static_assert(TwoElectronConsumer<Consumer>,
            "Consumer must provide accumulate(const TwoElectronBuffer<0>&, "
            "Index, Index, Index, Index, int, int, int, int)");

        if (!op.is_two_electron()) {
            throw InvalidArgumentException(
                "compute_and_consume_screened_parallel requires a two-electron operator, got: " +
                std::string(to_string(op.kind())));
        }

        // If screening is disabled, use the unscreened parallel path
        if (!options.enabled) {
            compute_and_consume_parallel(op, consumer, n_threads);
            return;
        }

        // Validate density-weighted screening requires density matrix
        if (options.density_weighted && !density_matrix_set()) {
            throw InvalidArgumentException(
                "density_weighted screening requires a density matrix; "
                "call set_density_matrix() before compute_and_consume_screened_parallel()");
        }

        precompute_schwarz_bounds();
        compute_and_consume_screened_parallel_impl(op, consumer, options, n_threads);
    }

    // =========================================================================
    // Unified Compute API
    // =========================================================================

    /// @brief Unified compute method - dispatches based on work unit and operator type
    ///
    /// This is an umbrella API that routes to the appropriate specialized method:
    ///   - Shell pair + 1e buffer → compute_1e_shell_pair
    ///   - Shell quartet + 2e buffer → compute_2e_shell_quartet
    ///   - ShellSetPair + result vector → compute_shell_set_pair
    ///   - ShellSetQuartet + Consumer → compute_shell_set_quartet
    ///   - Consumer only → compute_and_consume (full basis)
    ///   - OneElectronOperator + result → compute_1e (full basis)

    /// @brief Compute one-electron integral for a single shell pair
    ///
    /// @tparam DerivOrder Derivative order (0 = energy)
    /// @param op The one-electron operator
    /// @param shell_a First shell (bra)
    /// @param shell_b Second shell (ket)
    /// @param buffer Output buffer for this shell pair
    /// @param hint Backend selection hint (default: Auto)
    template<int DerivOrder = 0>
    void compute(const Operator& op,
                 const Shell& shell_a,
                 const Shell& shell_b,
                 OneElectronBuffer<DerivOrder>& buffer,
                 BackendHint hint = BackendHint::Auto) {
        static_assert(DerivOrder == 0,
                      "Only DerivOrder == 0 (energy) is supported in this release");
        if (!op.is_one_electron()) {
            throw InvalidArgumentException(
                "compute with shell pair requires one-electron operator, got: " +
                std::string(to_string(op.kind())));
        }
        compute_1e_shell_pair(op, shell_a, shell_b, buffer, hint);
    }

    /// @brief Compute two-electron integral for a single shell quartet
    ///
    /// @tparam DerivOrder Derivative order (0 = energy)
    /// @param op The two-electron operator
    /// @param shell_a First bra shell
    /// @param shell_b Second bra shell
    /// @param shell_c First ket shell
    /// @param shell_d Second ket shell
    /// @param buffer Output buffer for this shell quartet
    /// @param hint Backend selection hint (default: Auto)
    template<int DerivOrder = 0>
    void compute(const Operator& op,
                 const Shell& shell_a, const Shell& shell_b,
                 const Shell& shell_c, const Shell& shell_d,
                 TwoElectronBuffer<DerivOrder>& buffer,
                 BackendHint hint = BackendHint::Auto) {
        static_assert(DerivOrder == 0,
                      "Only DerivOrder == 0 (energy) is supported in this release");
        if (!op.is_two_electron()) {
            throw InvalidArgumentException(
                "compute with shell quartet requires two-electron operator, got: " +
                std::string(to_string(op.kind())));
        }
        compute_2e_shell_quartet(op, shell_a, shell_b, shell_c, shell_d, buffer, hint);
    }

    /// @brief Compute one-electron integrals for a ShellSetPair (batched)
    ///
    /// @param op The one-electron operator
    /// @param pair ShellSetPair containing shells to process
    /// @param result Output matrix (nbf x nbf, row-major)
    /// @param hint Backend selection hint (default: Auto)
    void compute(const Operator& op,
                 const ShellSetPair& pair,
                 std::vector<Real>& result,
                 BackendHint hint = BackendHint::Auto) {
        compute_shell_set_pair(op, pair, result, hint);
    }

    /// @brief Compute and consume two-electron integrals for a ShellSetQuartet
    ///
    /// @tparam Consumer Type with an accumulate method
    /// @param op The two-electron operator
    /// @param quartet ShellSetQuartet containing shells to process
    /// @param consumer Consumer object (e.g., FockBuilder)
    /// @param hint Backend selection hint (default: Auto)
    template<typename Consumer>
    void compute(const Operator& op,
                 const ShellSetQuartet& quartet,
                 Consumer& consumer,
                 BackendHint hint = BackendHint::Auto) {
        compute_shell_set_quartet(op, quartet, consumer, hint);
    }

    /// @brief Compute and consume two-electron integrals for the full basis
    ///
    /// @tparam Consumer Type with an accumulate method
    /// @param op The two-electron operator (e.g., Coulomb)
    /// @param consumer The consumer object (e.g., FockBuilder)
    /// @param hint Backend selection hint (default: Auto)
    template<typename Consumer>
    void compute(const Operator& op,
                 Consumer& consumer,
                 BackendHint hint = BackendHint::Auto) {
        compute_and_consume(op, consumer, hint);
    }

    /// @brief Compute and consume two-electron integrals for a worklist of ShellSetQuartets
    ///
    /// @tparam Consumer Type with an accumulate method
    /// @param op The two-electron operator
    /// @param quartets Span of ShellSetQuartets to compute
    /// @param consumer Consumer object
    /// @param hint Backend selection hint
    template<typename Consumer>
    void compute(const Operator& op,
                 std::span<const ShellSetQuartet> quartets,
                 Consumer& consumer,
                 BackendHint hint = BackendHint::Auto) {
        compute_and_consume(op, quartets, consumer, hint);
    }

    /// @brief Compute one-electron integrals for the full basis
    ///
    /// @tparam DerivOrder Derivative order (0 = energy)
    /// @param op The composed one-electron operator
    /// @param result Output matrix (n_basis_functions x n_basis_functions)
    /// @param hint Backend selection hint (default: Auto)
    template<int DerivOrder = 0>
    void compute(const OneElectronOperator& op,
                 std::vector<Real>& result,
                 BackendHint hint = BackendHint::Auto) {
        static_assert(DerivOrder == 0,
                      "Only DerivOrder == 0 (energy) is supported in this release");
        compute_1e<DerivOrder>(op, result, hint);
    }

    // =========================================================================
    // Non-Consumer compute() Overloads (return IntegralBuffer)
    // =========================================================================

    /// @brief Compute two-electron integrals for a ShellSetQuartet (non-consumer)
    ///
    /// Returns an IntegralBuffer containing all computed integrals for the
    /// shell quartets within the ShellSetQuartet. This is the simplest API
    /// for accessing raw integral values without a consumer callback.
    ///
    /// @param op The two-electron operator
    /// @param quartet ShellSetQuartet containing shells to process
    /// @param hint Backend selection hint (default: Auto)
    /// @return IntegralBuffer with computed integrals and metadata
    [[nodiscard]] IntegralBuffer compute(
        const Operator& op,
        const ShellSetQuartet& quartet,
        BackendHint hint = BackendHint::Auto,
        const screening::ScreeningOptions& screening = screening::ScreeningOptions::none()) {
        return compute_batch(op, quartet, hint, nullptr, screening);
    }

    /// @brief Compute two-electron integrals for a worklist of ShellSetQuartets (non-consumer)
    ///
    /// Returns a vector of IntegralBuffers, one per input quartet. The
    /// result ordering matches the input ordering.
    ///
    /// @param op The two-electron operator
    /// @param quartets Span of ShellSetQuartets to compute
    /// @param hint Backend selection hint (default: Auto)
    /// @return Vector of IntegralBuffers in input order
    [[nodiscard]] std::vector<IntegralBuffer> compute(
        const Operator& op,
        std::span<const ShellSetQuartet> quartets,
        BackendHint hint = BackendHint::Auto,
        const screening::ScreeningOptions& screening = screening::ScreeningOptions::none()) {
        return compute_batch(op, quartets, hint, nullptr, screening);
    }

    /// @brief Compute one-electron integrals for a ShellSetPair (non-consumer)
    ///
    /// Returns an IntegralBuffer containing all computed integrals for the
    /// shell pairs within the ShellSetPair. For one-electron integrals where
    /// the caller wants direct access to per-pair values and metadata.
    ///
    /// @param op The one-electron operator
    /// @param pair ShellSetPair containing shells to process
    /// @param hint Backend selection hint (default: Auto)
    /// @return IntegralBuffer with computed integrals and metadata
    [[nodiscard]] IntegralBuffer compute(
        const Operator& op,
        const ShellSetPair& pair,
        BackendHint hint = BackendHint::Auto) {
        return compute_batch(op, pair, hint);
    }

    // =========================================================================
    // Full Matrix Computation (Convenience Methods)
    // =========================================================================

    /// @brief Compute the full overlap matrix
    /// @param result Output matrix (n_basis_functions x n_basis_functions)
    /// @param hint Backend selection hint (default: Auto)
    void compute_overlap_matrix(std::vector<Real>& result,
                                BackendHint hint = BackendHint::Auto) {
        BackendType backend = dispatch_policy_.select_backend(
            WorkUnitType::FullBasis,
            basis_->n_shells(),
            basis_->max_angular_momentum() * 2,
            estimate_primitives(),
            hint,
            gpu_available_);

        if (backend == BackendType::CPU) {
            cpu_engine_.compute_1e(OneElectronOperator(Operator::overlap()), result);
        }
#if LIBACCINT_USE_CUDA
        else if (cuda_engine_) {
            cuda_engine_->compute_overlap_matrix(result);
        }
#endif
        else {
            cpu_engine_.compute_1e(OneElectronOperator(Operator::overlap()), result);
        }
    }

    /// @brief Compute the full kinetic energy matrix
    /// @param result Output matrix (n_basis_functions x n_basis_functions)
    /// @param hint Backend selection hint (default: Auto)
    void compute_kinetic_matrix(std::vector<Real>& result,
                                BackendHint hint = BackendHint::Auto) {
        BackendType backend = dispatch_policy_.select_backend(
            WorkUnitType::FullBasis,
            basis_->n_shells(),
            basis_->max_angular_momentum() * 2,
            estimate_primitives(),
            hint,
            gpu_available_);

        if (backend == BackendType::CPU) {
            cpu_engine_.compute_1e(OneElectronOperator(Operator::kinetic()), result);
        }
#if LIBACCINT_USE_CUDA
        else if (cuda_engine_) {
            cuda_engine_->compute_kinetic_matrix(result);
        }
#endif
        else {
            cpu_engine_.compute_1e(OneElectronOperator(Operator::kinetic()), result);
        }
    }

    /// @brief Compute the full nuclear attraction matrix
    /// @param charges Point charge parameters (nuclear positions and charges)
    /// @param result Output matrix (n_basis_functions x n_basis_functions)
    /// @param hint Backend selection hint (default: Auto)
    void compute_nuclear_matrix(const PointChargeParams& charges,
                                std::vector<Real>& result,
                                BackendHint hint = BackendHint::Auto) {
        BackendType backend = dispatch_policy_.select_backend(
            WorkUnitType::FullBasis,
            basis_->n_shells(),
            basis_->max_angular_momentum() * 2,
            estimate_primitives(),
            hint,
            gpu_available_);

        if (backend == BackendType::CPU) {
            cpu_engine_.compute_1e(OneElectronOperator(Operator::nuclear(charges)), result);
        }
#if LIBACCINT_USE_CUDA
        else if (cuda_engine_) {
            cuda_engine_->compute_nuclear_matrix(charges, result);
        }
#endif
        else {
            cpu_engine_.compute_1e(OneElectronOperator(Operator::nuclear(charges)), result);
        }
    }

    /// @brief Compute the core Hamiltonian matrix (H = T + V)
    /// @param charges Point charge parameters (nuclear positions and charges)
    /// @param result Output matrix (n_basis_functions x n_basis_functions)
    /// @param hint Backend selection hint (default: Auto)
    void compute_core_hamiltonian(const PointChargeParams& charges,
                                  std::vector<Real>& result,
                                  BackendHint hint = BackendHint::Auto) {
        BackendType backend = dispatch_policy_.select_backend(
            WorkUnitType::FullBasis,
            basis_->n_shells(),
            basis_->max_angular_momentum() * 2,
            estimate_primitives(),
            hint,
            gpu_available_);

        if (backend == BackendType::CPU) {
            OneElectronOperator h_core = Operator::kinetic();
            h_core.add(Operator::nuclear(charges));
            cpu_engine_.compute_1e(h_core, result);
        }
#if LIBACCINT_USE_CUDA
        else if (cuda_engine_) {
            cuda_engine_->compute_core_hamiltonian(charges, result);
        }
#endif
        else {
            OneElectronOperator h_core = Operator::kinetic();
            h_core.add(Operator::nuclear(charges));
            cpu_engine_.compute_1e(h_core, result);
        }
    }

    // =========================================================================
    // Return-Value Convenience Overloads
    // =========================================================================

    /// @brief Compute and return the full overlap matrix
    /// @param hint Backend selection hint (default: Auto)
    /// @return Overlap matrix as a flat vector (n_basis x n_basis)
    [[nodiscard]] std::vector<Real> compute_overlap_matrix(
            BackendHint hint = BackendHint::Auto) {
        std::vector<Real> result;
        compute_overlap_matrix(result, hint);
        return result;
    }

    /// @brief Compute and return the full kinetic energy matrix
    /// @param hint Backend selection hint (default: Auto)
    /// @return Kinetic energy matrix as a flat vector (n_basis x n_basis)
    [[nodiscard]] std::vector<Real> compute_kinetic_matrix(
            BackendHint hint = BackendHint::Auto) {
        std::vector<Real> result;
        compute_kinetic_matrix(result, hint);
        return result;
    }

    /// @brief Compute and return the full nuclear attraction matrix
    /// @param charges Point charge parameters (nuclear positions and charges)
    /// @param hint Backend selection hint (default: Auto)
    /// @return Nuclear attraction matrix as a flat vector (n_basis x n_basis)
    [[nodiscard]] std::vector<Real> compute_nuclear_matrix(
            const PointChargeParams& charges,
            BackendHint hint = BackendHint::Auto) {
        std::vector<Real> result;
        compute_nuclear_matrix(charges, result, hint);
        return result;
    }

    /// @brief Compute and return the core Hamiltonian matrix (H = T + V)
    /// @param charges Point charge parameters (nuclear positions and charges)
    /// @param hint Backend selection hint (default: Auto)
    /// @return Core Hamiltonian matrix as a flat vector (n_basis x n_basis)
    [[nodiscard]] std::vector<Real> compute_core_hamiltonian(
            const PointChargeParams& charges,
            BackendHint hint = BackendHint::Auto) {
        std::vector<Real> result;
        compute_core_hamiltonian(charges, result, hint);
        return result;
    }

    // =========================================================================
    // Legacy API Compatibility
    // =========================================================================

    /// @brief Get the backend type (returns CPU for backward compatibility)
    /// @deprecated Use gpu_available() instead to check for GPU support
    [[nodiscard]] BackendType backend() const noexcept {
        return BackendType::CPU;
    }

    // =========================================================================
    // Batch Compute API (Non-Consuming)
    // =========================================================================

    /// @brief Compute two-electron integrals for a single ShellSetQuartet
    ///
    /// Returns an IntegralBuffer containing all integrals for the shell
    /// quartets within the ShellSetQuartet. The caller gets direct access
    /// to integral values without needing a consumer.
    ///
    /// @param op The two-electron operator
    /// @param quartet ShellSetQuartet to compute
    /// @param hint Backend selection hint (default: Auto)
    /// @param pool Optional buffer pool for memory reuse
    /// @return IntegralBuffer with computed integrals and metadata
    [[nodiscard]] IntegralBuffer compute_batch(
        const Operator& op,
        const ShellSetQuartet& quartet,
        BackendHint hint = BackendHint::Auto,
        memory::BatchBufferPool* pool = nullptr,
        const screening::ScreeningOptions& screening = screening::ScreeningOptions::none()) {

        if (!op.is_two_electron()) {
            throw InvalidArgumentException(
                "compute_batch (2e) requires a two-electron operator, got: " +
                std::string(to_string(op.kind())));
        }

        // Schwarz screening: skip entire ShellSetQuartet if bound is below threshold
        if (screening.enabled) {
            Real bound = quartet.schwarz_bound();
            if (!screening.passes_screening(bound)) {
                return IntegralBuffer{};  // empty buffer = screened out
            }
        }

        IntegralBuffer result;
        if (pool) {
            result = pool->acquire(get_am_class(quartet));
        }

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

        Size batch_size = set_a.n_shells() * set_b.n_shells() *
                          set_c.n_shells() * set_d.n_shells();
        int total_am = set_a.angular_momentum() + set_b.angular_momentum() +
                       set_c.angular_momentum() + set_d.angular_momentum();

        BackendType backend = dispatch_policy_.select_backend(
            WorkUnitType::ShellSetQuartet,
            batch_size,
            total_am,
            estimate_shell_set_quartet_primitives(quartet),
            hint,
            gpu_available_);

#if LIBACCINT_USE_CUDA
        if (backend == BackendType::CUDA && cuda_engine_) {
            diagnostics::BatchTracer::instance().trace_2e_dispatch(
                std::string(to_string(op.kind())).c_str(),
                set_a.angular_momentum(),
                set_b.angular_momentum(),
                set_c.angular_momentum(),
                set_d.angular_momentum(),
                batch_size, "batched-nonconsumer", "GPU");

            // Delegate to CudaEngine::compute_batch() which internally
            // acquires a GPU execution slot for thread-safe concurrent access
            return cuda_engine_->compute_batch(op, quartet);
        }
#else
        (void)backend;
#endif

        // CPU path (default, or when GPU not available/selected)
        diagnostics::BatchTracer::instance().trace_2e_dispatch(
            std::string(to_string(op.kind())).c_str(),
            set_a.angular_momentum(),
            set_b.angular_momentum(),
            set_c.angular_momentum(),
            set_d.angular_momentum(),
            batch_size, "batched-nonconsumer", "CPU");

        TwoElectronBuffer<0> buffer;

        for (Size ia = 0; ia < set_a.n_shells(); ++ia) {
            const auto& shell_a = set_a.shell(ia);
            for (Size ib = 0; ib < set_b.n_shells(); ++ib) {
                const auto& shell_b = set_b.shell(ib);
                for (Size ic = 0; ic < set_c.n_shells(); ++ic) {
                    const auto& shell_c = set_c.shell(ic);
                    for (Size id = 0; id < set_d.n_shells(); ++id) {
                        const auto& shell_d = set_d.shell(id);

                        cpu_engine_.compute_2e_shell_quartet(
                            op, shell_a, shell_b, shell_c, shell_d, buffer);

                        result.append_quartet(
                            buffer.data(),
                            shell_a.function_index(),
                            shell_b.function_index(),
                            shell_c.function_index(),
                            shell_d.function_index(),
                            shell_a.n_functions(),
                            shell_b.n_functions(),
                            shell_c.n_functions(),
                            shell_d.n_functions());
                    }
                }
            }
        }

        return result;
    }

    /// @brief Compute two-electron integrals for multiple ShellSetQuartets
    ///
    /// Returns a vector of IntegralBuffers, one per input quartet.
    /// The result ordering matches the input ordering.
    ///
    /// @param op The two-electron operator
    /// @param quartets Span of ShellSetQuartets to compute
    /// @param hint Backend selection hint (default: Auto)
    /// @param pool Optional buffer pool for memory reuse
    /// @return Vector of IntegralBuffers in input order
    [[nodiscard]] std::vector<IntegralBuffer> compute_batch(
        const Operator& op,
        std::span<const ShellSetQuartet> quartets,
        BackendHint hint = BackendHint::Auto,
        memory::BatchBufferPool* pool = nullptr,
        const screening::ScreeningOptions& screening = screening::ScreeningOptions::none()) {

        if (!op.is_two_electron()) {
            throw InvalidArgumentException(
                "compute_batch (2e multi) requires a two-electron operator, got: " +
                std::string(to_string(op.kind())));
        }

        std::vector<IntegralBuffer> results;
        results.reserve(quartets.size());

        for (const auto& quartet : quartets) {
            results.push_back(compute_batch(op, quartet, hint, pool, screening));
        }
        return results;
    }

    /// @brief Compute all two-electron integrals for the full basis set
    ///
    /// Convenience method that computes integrals for all quartets from
    /// basis.shell_set_quartets().
    ///
    /// @param op The two-electron operator
    /// @param hint Backend selection hint (default: Auto)
    /// @return Vector of IntegralBuffers for all quartets
    [[nodiscard]] std::vector<IntegralBuffer> compute_all_2e(
        const Operator& op,
        BackendHint hint = BackendHint::Auto) {
        return compute_batch(op, basis_->shell_set_quartets(), hint);
    }

    /// @brief Compute two-electron integrals with Schwarz screening
    ///
    /// Computes all quartets whose Schwarz bound exceeds the screening
    /// threshold. Returns results only for non-screened quartets (some
    /// IntegralBuffers in the vector may be empty).
    ///
    /// @param op The two-electron operator
    /// @param screening Screening options (threshold, etc.)
    /// @param hint Backend selection hint (default: Auto)
    /// @return Vector of IntegralBuffers for all quartets (empty = screened)
    [[nodiscard]] std::vector<IntegralBuffer> compute_batch_screened(
        const Operator& op,
        const screening::ScreeningOptions& screening,
        BackendHint hint = BackendHint::Auto) {
        precompute_schwarz_bounds();
        return compute_batch(op, basis_->shell_set_quartets(), hint, nullptr, screening);
    }

    /// @brief Get screening statistics from a batch computation
    ///
    /// Count the number of non-empty and empty IntegralBuffers to determine
    /// screening effectiveness.
    ///
    /// @param results Vector of IntegralBuffers from a screened compute
    /// @return ScreeningStatistics with computed/skipped counts
    [[nodiscard]] static screening::ScreeningStatistics
    compute_screening_statistics(const std::vector<IntegralBuffer>& results) {
        screening::ScreeningStatistics stats;
        stats.total_quartets = results.size();
        for (const auto& buf : results) {
            if (buf.n_shell_quartets() > 0) {
                ++stats.computed_quartets;
            } else {
                ++stats.skipped_quartets;
            }
        }
        return stats;
    }

    // =========================================================================
    // Parallel Batch Compute API
    // =========================================================================

    /// @brief Compute two-electron integrals for multiple ShellSetQuartets in parallel
    ///
    /// Parallelizes over ShellSetQuartets using OpenMP. Each quartet produces
    /// an independent IntegralBuffer, so results are trivially parallel with
    /// no consumer thread-safety requirements.
    ///
    /// @note Thread safety: compute_batch() creates a local TwoElectronBuffer
    /// and IntegralBuffer per call, so there is no shared mutable state between
    /// threads. The CpuEngine::compute_2e_shell_quartet() reads shared basis
    /// data and writes only to the caller-provided buffer.
    /// When the CUDA backend is selected, CudaEngine internally acquires a
    /// GPU execution slot (stream + device buffers) from a pool, enabling
    /// true concurrent GPU execution from multiple host threads.
    ///
    /// @param op The two-electron operator
    /// @param quartets Span of ShellSetQuartets to compute
    /// @param n_threads Number of OpenMP threads (0 = auto)
    /// @param hint Backend selection hint (default: Auto)
    /// @param screening Optional screening options
    /// @return Vector of IntegralBuffers in input order
    [[nodiscard]] std::vector<IntegralBuffer> compute_batch_parallel(
        const Operator& op,
        std::span<const ShellSetQuartet> quartets,
        int n_threads = 0,
        BackendHint hint = BackendHint::Auto,
        const screening::ScreeningOptions& screening = screening::ScreeningOptions::none()) {

        if (!op.is_two_electron()) {
            throw InvalidArgumentException(
                "compute_batch_parallel requires a two-electron operator, got: " +
                std::string(to_string(op.kind())));
        }

        std::vector<IntegralBuffer> results(quartets.size());

        const int actual_threads = engine::ThreadConfig::resolve(n_threads);

        #pragma omp parallel for schedule(dynamic) num_threads(actual_threads) \
            if(quartets.size() > 1)
        for (Size i = 0; i < quartets.size(); ++i) {
            // Each thread creates local TwoElectronBuffer and IntegralBuffer
            // inside compute_batch(), so no shared mutable state exists
            results[i] = compute_batch(op, quartets[i], hint, nullptr, screening);
        }

        return results;
    }

    /// @brief Parallel compute all two-electron integrals for the full basis
    ///
    /// @param op The two-electron operator
    /// @param n_threads Number of OpenMP threads (0 = auto)
    /// @param hint Backend selection hint
    /// @return Vector of IntegralBuffers for all quartets
    [[nodiscard]] std::vector<IntegralBuffer> compute_all_2e_parallel(
        const Operator& op,
        int n_threads = 0,
        BackendHint hint = BackendHint::Auto) {
        return compute_batch_parallel(op, basis_->shell_set_quartets(), n_threads, hint);
    }

    // =========================================================================
    // Convenience ERI Methods
    // =========================================================================

    /// @brief Compute the full 4-index ERI tensor as a flat vector
    ///
    /// Returns (ij|kl) for all basis function indices i,j,k,l scattered
    /// into a flat nbf^4 vector with row-major indexing:
    ///   index = i * nbf^3 + j * nbf^2 + k * nbf + l
    ///
    /// @warning This method allocates O(nbf^4) memory and is guarded by a
    /// runtime byte limit (default 1 GiB, configurable via
    /// LIBACCINT_MAX_ERI_TENSOR_BYTES).
    ///
    /// @param op The two-electron operator (default: Coulomb)
    /// @param hint Backend selection hint (default: Auto)
    /// @return Flat vector of size nbf^4 with ERI values
    /// @throws InvalidArgumentException if requested ERI tensor exceeds
    ///         configured memory safety limit
    [[nodiscard]] std::vector<Real> compute_eri_tensor(
        const Operator& op = Operator::coulomb(),
        BackendHint hint = BackendHint::Auto) {

        if (!op.is_two_electron()) {
            throw InvalidArgumentException(
                "compute_eri_tensor requires a two-electron operator, got: " +
                std::string(to_string(op.kind())));
        }

        const Size nbf = basis_->n_basis_functions();
        auto checked_mul = [](std::size_t a, std::size_t b, std::size_t& out) -> bool {
            if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
                return false;
            }
            out = a * b;
            return true;
        };

        std::size_t nbf2 = 0;
        std::size_t nbf3 = 0;
        std::size_t nbf4 = 0;
        if (!checked_mul(static_cast<std::size_t>(nbf), static_cast<std::size_t>(nbf), nbf2) ||
            !checked_mul(nbf2, static_cast<std::size_t>(nbf), nbf3) ||
            !checked_mul(nbf3, static_cast<std::size_t>(nbf), nbf4)) {
            throw InvalidArgumentException(
                "compute_eri_tensor: nbf^4 overflows size limits for nbf=" +
                std::to_string(nbf));
        }

        std::size_t tensor_bytes = 0;
        if (!checked_mul(nbf4, sizeof(Real), tensor_bytes)) {
            throw InvalidArgumentException(
                "compute_eri_tensor: ERI tensor byte size overflow for nbf=" +
                std::to_string(nbf));
        }

        static const std::size_t max_tensor_bytes = []() {
            constexpr std::size_t kDefaultMaxBytes = 1024ull * 1024ull * 1024ull;  // 1 GiB
            const char* env = std::getenv("LIBACCINT_MAX_ERI_TENSOR_BYTES");
            if (env == nullptr || env[0] == '\0') {
                return kDefaultMaxBytes;
            }
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(env, &end, 10);
            if (end != env && *end == '\0' && parsed > 0) {
                return static_cast<std::size_t>(parsed);
            }
            return kDefaultMaxBytes;
        }();

        if (tensor_bytes > max_tensor_bytes) {
            throw InvalidArgumentException(
                "compute_eri_tensor: requested tensor requires " +
                std::to_string(tensor_bytes) + " bytes (nbf=" + std::to_string(nbf) +
                "), exceeding limit " + std::to_string(max_tensor_bytes) +
                ". Use compute_batch()/consumer APIs or raise LIBACCINT_MAX_ERI_TENSOR_BYTES.");
        }

        std::vector<Real> tensor(nbf4, 0.0);

        auto buffers = compute_all_2e(op, hint);

        for (const auto& buf : buffers) {
            for (Size q = 0; q < buf.n_shell_quartets(); ++q) {
                const auto& meta = buf.quartet_meta(q);
                auto data = buf.quartet_data(q);

                Size idx = 0;
                for (int a = 0; a < meta.na; ++a) {
                    const Size i = meta.fi + a;
                    for (int b = 0; b < meta.nb; ++b) {
                        const Size j = meta.fj + b;
                        for (int c = 0; c < meta.nc; ++c) {
                            const Size k = meta.fk + c;
                            for (int d = 0; d < meta.nd; ++d) {
                                const Size l = meta.fl + d;
                                const Real val = data[idx];
                                // Scatter all 8 permutation symmetries:
                                // (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
                                //         = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)
                                tensor[i * nbf3 + j * nbf2 + k * nbf + l] = val;
                                tensor[j * nbf3 + i * nbf2 + k * nbf + l] = val;
                                tensor[i * nbf3 + j * nbf2 + l * nbf + k] = val;
                                tensor[j * nbf3 + i * nbf2 + l * nbf + k] = val;
                                tensor[k * nbf3 + l * nbf2 + i * nbf + j] = val;
                                tensor[l * nbf3 + k * nbf2 + i * nbf + j] = val;
                                tensor[k * nbf3 + l * nbf2 + j * nbf + i] = val;
                                tensor[l * nbf3 + k * nbf2 + j * nbf + i] = val;
                                ++idx;
                            }
                        }
                    }
                }
            }
        }

        return tensor;
    }

    /// @brief Compute ERIs for a single ShellSetQuartet scattered into a block
    ///
    /// Returns a flat vector with dimensions
    /// n_funcs_a × n_funcs_b × n_funcs_c × n_funcs_d containing all
    /// integrals for the specified ShellSetQuartet. The block uses local
    /// basis-function indexing starting from 0 (not global function indices).
    ///
    /// @param op The two-electron operator
    /// @param quartet ShellSetQuartet to compute
    /// @param hint Backend selection hint (default: Auto)
    /// @return Flat vector of ERI values in row-major order
    [[nodiscard]] std::vector<Real> compute_eri_block(
        const Operator& op,
        const ShellSetQuartet& quartet,
        BackendHint hint = BackendHint::Auto) {

        auto buf = compute_batch(op, quartet, hint);

        // Calculate block dimensions from the ShellSetQuartet
        const auto& set_a = quartet.bra_pair().shell_set_a();
        const auto& set_b = quartet.bra_pair().shell_set_b();
        const auto& set_c = quartet.ket_pair().shell_set_a();
        const auto& set_d = quartet.ket_pair().shell_set_b();

        // Total functions in each shell set
        Size nf_a = 0, nf_b = 0, nf_c = 0, nf_d = 0;
        for (Size s = 0; s < set_a.n_shells(); ++s) nf_a += set_a.shell(s).n_functions();
        for (Size s = 0; s < set_b.n_shells(); ++s) nf_b += set_b.shell(s).n_functions();
        for (Size s = 0; s < set_c.n_shells(); ++s) nf_c += set_c.shell(s).n_functions();
        for (Size s = 0; s < set_d.n_shells(); ++s) nf_d += set_d.shell(s).n_functions();

        std::vector<Real> block(nf_a * nf_b * nf_c * nf_d, 0.0);

        // Base function indices for local indexing
        const Index base_a = set_a.shell(0).function_index();
        const Index base_b = set_b.shell(0).function_index();
        const Index base_c = set_c.shell(0).function_index();
        const Index base_d = set_d.shell(0).function_index();

        for (Size q = 0; q < buf.n_shell_quartets(); ++q) {
            const auto& meta = buf.quartet_meta(q);
            auto data = buf.quartet_data(q);

            Size idx = 0;
            for (int a = 0; a < meta.na; ++a) {
                const Size la = static_cast<Size>(meta.fi - base_a) + a;
                for (int b = 0; b < meta.nb; ++b) {
                    const Size lb = static_cast<Size>(meta.fj - base_b) + b;
                    for (int c = 0; c < meta.nc; ++c) {
                        const Size lc = static_cast<Size>(meta.fk - base_c) + c;
                        for (int d = 0; d < meta.nd; ++d) {
                            const Size ld = static_cast<Size>(meta.fl - base_d) + d;
                            block[la * nf_b * nf_c * nf_d +
                                  lb * nf_c * nf_d +
                                  lc * nf_d + ld] = data[idx];
                            ++idx;
                        }
                    }
                }
            }
        }

        return block;
    }

    /// @brief Compute one-electron integrals for a single ShellSetPair
    ///
    /// Returns an IntegralBuffer containing all integrals for the shell
    /// pairs within the ShellSetPair. Self-pair handling (upper triangle)
    /// is applied automatically.
    ///
    /// @param op The one-electron operator
    /// @param pair ShellSetPair to compute
    /// @param hint Backend selection hint (default: Auto)
    /// @return IntegralBuffer with computed integrals and metadata
    [[nodiscard]] IntegralBuffer compute_batch(
        const Operator& op,
        const ShellSetPair& pair,
        BackendHint hint = BackendHint::Auto) {

        if (!op.is_one_electron()) {
            throw InvalidArgumentException(
                "compute_batch (1e) requires a one-electron operator, got: " +
                std::string(to_string(op.kind())));
        }

        IntegralBuffer result;
        result.set_am(pair.La(), pair.Lb());

        const auto& set_a = pair.shell_set_a();
        const auto& set_b = pair.shell_set_b();
        const bool is_self_pair = (&set_a == &set_b);

        const int nf_a = n_cartesian(set_a.angular_momentum());
        const int nf_b = n_cartesian(set_b.angular_momentum());
        const Size n_ints_per_pair = static_cast<Size>(nf_a * nf_b);

        // Estimate count: full Cartesian product or upper-triangle
        Size estimated_pairs;
        if (is_self_pair) {
            const Size n = set_a.n_shells();
            estimated_pairs = n * (n + 1) / 2;
        } else {
            estimated_pairs = set_a.n_shells() * set_b.n_shells();
        }
        result.reserve_1e(n_ints_per_pair * estimated_pairs, estimated_pairs);

        OneElectronBuffer<0> buffer;

        for (Size i = 0; i < set_a.n_shells(); ++i) {
            const auto& shell_a = set_a.shell(i);
            const Size j_start = is_self_pair ? i : Size{0};

            for (Size j = j_start; j < set_b.n_shells(); ++j) {
                const auto& shell_b = set_b.shell(j);

                // Route through unified shell-pair dispatch so backend hints
                // (ForceCPU/PreferGPU/ForceGPU) are honored for non-consumer 1e.
                compute_1e_shell_pair(op, shell_a, shell_b, buffer, hint);

                result.append_pair(
                    buffer.data(),
                    shell_a.function_index(),
                    shell_b.function_index(),
                    shell_a.n_functions(),
                    shell_b.n_functions());
            }
        }

        return result;
    }

    /// @brief Compute one-electron integrals for multiple ShellSetPairs
    ///
    /// @param op The one-electron operator
    /// @param pairs Span of ShellSetPairs to compute
    /// @param hint Backend selection hint
    /// @return Vector of IntegralBuffers in input order
    [[nodiscard]] std::vector<IntegralBuffer> compute_batch(
        const Operator& op,
        std::span<const ShellSetPair> pairs,
        BackendHint hint = BackendHint::Auto) {

        if (!op.is_one_electron()) {
            throw InvalidArgumentException(
                "compute_batch (1e multi) requires a one-electron operator, got: " +
                std::string(to_string(op.kind())));
        }

        std::vector<IntegralBuffer> results;
        results.reserve(pairs.size());

        for (const auto& pair : pairs) {
            results.push_back(compute_batch(op, pair, hint));
        }
        return results;
    }

    /// @brief Compute all one-electron integrals for the full basis set
    ///
    /// @param op The one-electron operator
    /// @param hint Backend selection hint
    /// @return Vector of IntegralBuffers for all pairs
    [[nodiscard]] std::vector<IntegralBuffer> compute_all_1e(
        const Operator& op,
        BackendHint hint = BackendHint::Auto) {
        return compute_batch(op, basis_->shell_set_pairs(), hint);
    }

    // =========================================================================
    // Schwarz Screening Support
    // =========================================================================

    /// @brief Precompute Schwarz bounds for all shell pairs
    ///
    /// This method computes and caches Schwarz bounds Q[i][j] = sqrt(max |(ij|ij)|)
    /// for all shell pairs in the basis set. This is optional - bounds are computed
    /// lazily when using screened compute methods if not precomputed.
    ///
    /// Precomputing is recommended when:
    /// - Many screened compute_and_consume calls will be made
    /// - Parallel computation is needed (avoids lazy init contention)
    ///
    /// @return Reference to the computed SchwarzBounds object
    const screening::SchwarzBounds& precompute_schwarz_bounds() {
        std::call_once(schwarz_init_flag_, [this]() {
            schwarz_bounds_ = std::make_unique<screening::SchwarzBounds>(*basis_);
        });
        return *schwarz_bounds_;
    }

    /// @brief Check if Schwarz bounds have been precomputed
    [[nodiscard]] bool schwarz_bounds_precomputed() const noexcept {
        return schwarz_bounds_ != nullptr;
    }

    /// @brief Get the Schwarz bounds (computes if not already done)
    [[nodiscard]] const screening::SchwarzBounds& get_schwarz_bounds() {
        return precompute_schwarz_bounds();
    }

    /// @brief Set the density matrix for density-weighted screening
    ///
    /// When density_weighted=true is set in ScreeningOptions, the engine will
    /// use the density matrix to compute tighter screening bounds:
    /// D_max * Q_ij * Q_kl < threshold
    ///
    /// Must be called before compute_and_consume with density_weighted=true.
    /// Should be called each SCF iteration when the density matrix changes.
    ///
    /// @param D Pointer to row-major density matrix (nbf x nbf)
    /// @param nbf Number of basis functions
    void set_density_matrix(const Real* D, Size nbf) {
        if (!density_screening_) {
            density_screening_ = std::make_unique<screening::DensityScreening>(*basis_);
        }
        density_screening_->update_density(D, nbf);
    }

    /// @brief Set the density matrix for density-weighted screening (span version)
    void set_density_matrix(std::span<const Real> D, Size nbf) {
        set_density_matrix(D.data(), nbf);
    }

    /// @brief Check if density matrix has been set for density-weighted screening
    [[nodiscard]] bool density_matrix_set() const noexcept {
        return density_screening_ && density_screening_->is_initialized();
    }

    /// @brief Fused two-electron integral computation with Schwarz screening
    ///
    /// This is the screened version of compute_and_consume. It applies
    /// Schwarz screening to skip insignificant quartets where
    /// Q_ab * Q_cd < threshold.
    ///
    /// @tparam Consumer Type with an accumulate method
    /// @param op The two-electron operator (e.g., Coulomb)
    /// @param consumer The consumer object (e.g., FockBuilder)
    /// @param options Screening options (threshold, etc.)
    /// @param hint Backend selection hint (default: Auto)
    template<typename Consumer>
    void compute_and_consume(const Operator& op, Consumer& consumer,
                             const screening::ScreeningOptions& options,
                             BackendHint hint = BackendHint::Auto) {
        static_assert(TwoElectronConsumer<Consumer>,
            "Consumer must provide accumulate(const TwoElectronBuffer<0>&, "
            "Index, Index, Index, Index, int, int, int, int)");

        if (!op.is_two_electron()) {
            throw InvalidArgumentException(
                "compute_and_consume requires a two-electron operator, got: " +
                std::string(to_string(op.kind())));
        }

        // If screening is disabled, use the unscreened path
        if (!options.enabled) {
            compute_and_consume(op, consumer, hint);
            return;
        }

        // Validate density-weighted screening requires density matrix
        if (options.density_weighted && !density_matrix_set()) {
            throw InvalidArgumentException(
                "density_weighted screening requires a density matrix; "
                "call set_density_matrix() before compute_and_consume()");
        }

        // Ensure Schwarz bounds are computed
        precompute_schwarz_bounds();

        // Use CPU engine for now
        compute_and_consume_screened_impl(op, consumer, options);
        (void)hint;
    }

private:
    [[nodiscard]] bool use_default_cpu_parallel_1e() const noexcept {
        return ThreadConfig::openmp_available() &&
               ThreadConfig::effective_threads() > 1 &&
               basis_->shell_set_pairs().size() > 1;
    }

    [[nodiscard]] bool use_default_cpu_parallel_2e() const noexcept {
        return ThreadConfig::openmp_available() &&
               ThreadConfig::effective_threads() > 1 &&
               basis_->shell_set_quartets().size() > 1;
    }

    template<typename Consumer>
    void compute_shell_set_quartet_impl(const Operator& op,
                                        const ShellSetQuartet& quartet,
                                        Consumer& consumer,
                                        BackendHint hint,
                                        bool canonical_symmetry) {
        static_assert(TwoElectronConsumer<Consumer>,
            "Consumer must provide accumulate(const TwoElectronBuffer<0>&, "
            "Index, Index, Index, Index, int, int, int, int)");

        Size batch_size = quartet.bra_pair().shell_set_a().n_shells() *
                          quartet.bra_pair().shell_set_b().n_shells() *
                          quartet.ket_pair().shell_set_a().n_shells() *
                          quartet.ket_pair().shell_set_b().n_shells();
        int total_am = quartet.bra_pair().shell_set_a().angular_momentum() +
                       quartet.bra_pair().shell_set_b().angular_momentum() +
                       quartet.ket_pair().shell_set_a().angular_momentum() +
                       quartet.ket_pair().shell_set_b().angular_momentum();

        BackendType backend = dispatch_policy_.select_backend(
            WorkUnitType::ShellSetQuartet,
            batch_size,
            total_am,
            estimate_shell_set_quartet_primitives(quartet),
            hint,
            gpu_available_);

#if LIBACCINT_USE_CUDA
        if (backend == BackendType::CUDA && cuda_engine_) {
            diagnostics::BatchTracer::instance().trace_2e_dispatch(
                std::string(to_string(op.kind())).c_str(),
                quartet.bra_pair().shell_set_a().angular_momentum(),
                quartet.bra_pair().shell_set_b().angular_momentum(),
                quartet.ket_pair().shell_set_a().angular_momentum(),
                quartet.ket_pair().shell_set_b().angular_momentum(),
                batch_size, "batched", "GPU");
            cuda_engine_->compute_shell_set_quartet(
                op, quartet, consumer, canonical_symmetry);
            return;
        }
#else
        (void)backend;
        (void)canonical_symmetry;
#endif

        diagnostics::BatchTracer::instance().trace_2e_dispatch(
            std::string(to_string(op.kind())).c_str(),
            quartet.bra_pair().shell_set_a().angular_momentum(),
            quartet.bra_pair().shell_set_b().angular_momentum(),
            quartet.ket_pair().shell_set_a().angular_momentum(),
            quartet.ket_pair().shell_set_b().angular_momentum(),
            batch_size, "batched", "CPU");
        cpu_engine_.compute_shell_set_quartet(op, quartet, consumer);
    }

    /// @brief Screened compute_and_consume implementation
    template<typename Consumer>
    void compute_and_consume_screened_impl(const Operator& op, Consumer& consumer,
                                            const screening::ScreeningOptions& options) {
        const Size n_shells = basis_->n_shells();
        const Real threshold = options.effective_threshold();
        const bool use_density = options.density_weighted && density_screening_;
        TwoElectronBuffer<0> buffer;

        // Use canonical 8-fold iteration when symmetry exploitation is requested
        // and the consumer supports it
        if constexpr (SymmetryAwareConsumer<Consumer>) {
            if (options.use_permutation_symmetry) {
                compute_and_consume_screened_symmetric_impl(op, consumer, options);
                return;
            }
        }

        // Full N^4 iteration with screening
        for (Size i = 0; i < n_shells; ++i) {
            const auto& shell_a = basis_->shell(i);
            const Index fi = shell_a.function_index();
            const int na = shell_a.n_functions();

            for (Size j = 0; j < n_shells; ++j) {
                const auto& shell_b = basis_->shell(j);
                const Index fj = shell_b.function_index();
                const int nb = shell_b.n_functions();

                for (Size k = 0; k < n_shells; ++k) {
                    const auto& shell_c = basis_->shell(k);
                    const Index fk = shell_c.function_index();
                    const int nc = shell_c.n_functions();

                    for (Size l = 0; l < n_shells; ++l) {
                        // Apply Schwarz screening
                        if (!schwarz_bounds_->passes_screening(i, j, k, l, threshold)) {
                            continue;
                        }

                        // Apply density-weighted screening if enabled
                        if (use_density &&
                            !density_screening_->passes_screening(
                                i, j, k, l, *schwarz_bounds_, threshold)) {
                            continue;
                        }

                        const auto& shell_d = basis_->shell(l);
                        const Index fl = shell_d.function_index();
                        const int nd = shell_d.n_functions();

                        // Compute ERIs for this quartet
                        cpu_engine_.compute_2e_shell_quartet(op, shell_a, shell_b,
                                                             shell_c, shell_d, buffer);

                        // Pass to consumer
                        consumer.accumulate(buffer, fi, fj, fk, fl, na, nb, nc, nd);
                    }
                }
            }
        }
    }

    /// @brief Canonical 8-fold symmetry screened implementation
    ///
    /// Iterates only unique shell quartets (i<=j, k<=l, ij<=kl) and
    /// calls consumer.accumulate_symmetric() to scatter contributions
    /// into all permutation-equivalent J/K matrix slots.
    template<SymmetryAwareConsumer Consumer>
    void compute_and_consume_screened_symmetric_impl(
        const Operator& op, Consumer& consumer,
        const screening::ScreeningOptions& options) {
        const Size n = basis_->n_shells();
        const Real threshold = options.effective_threshold();
        const bool use_density = options.density_weighted && density_screening_;
        TwoElectronBuffer<0> buffer;

        // Canonical iteration: i<=j, k<=l, pair(i,j) <= pair(k,l)
        for (Size i = 0; i < n; ++i) {
            for (Size j = i; j < n; ++j) {
                for (Size k = i; k < n; ++k) {
                    const Size l_start = (k == i) ? j : k;
                    for (Size l = l_start; l < n; ++l) {
                        // Apply Schwarz screening
                        if (!schwarz_bounds_->passes_screening(i, j, k, l, threshold)) {
                            continue;
                        }

                        // Apply density-weighted screening if enabled
                        if (use_density &&
                            !density_screening_->passes_screening(
                                i, j, k, l, *schwarz_bounds_, threshold)) {
                            continue;
                        }

                        const auto& shell_a = basis_->shell(i);
                        const auto& shell_b = basis_->shell(j);
                        const auto& shell_c = basis_->shell(k);
                        const auto& shell_d = basis_->shell(l);

                        const Index fi = shell_a.function_index();
                        const Index fj = shell_b.function_index();
                        const Index fk = shell_c.function_index();
                        const Index fl = shell_d.function_index();

                        const int na = shell_a.n_functions();
                        const int nb = shell_b.n_functions();
                        const int nc = shell_c.n_functions();
                        const int nd = shell_d.n_functions();

                        // Compute ERIs once for this canonical quartet
                        cpu_engine_.compute_2e_shell_quartet(op, shell_a, shell_b,
                                                             shell_c, shell_d, buffer);

                        // Scatter into all permutation-equivalent J/K slots
                        consumer.accumulate_symmetric(
                            buffer, fi, fj, fk, fl, na, nb, nc, nd,
                            /*ij_same=*/(i == j),
                            /*kl_same=*/(k == l),
                            /*braket_same=*/(i == k && j == l));
                    }
                }
            }
        }
    }

    /// @brief Screened parallel compute_and_consume implementation
    template<typename Consumer>
    void compute_and_consume_screened_parallel_impl(
        const Operator& op, Consumer& consumer,
        const screening::ScreeningOptions& options,
        [[maybe_unused]] int n_threads) {

        const Size n_shells = basis_->n_shells();
        const Real threshold = options.effective_threshold();
        const bool use_density = options.density_weighted && density_screening_;
        const bool use_symmetry = options.use_permutation_symmetry &&
                                  SymmetryAwareConsumer<Consumer>;

        // Build a screened worklist: only quartets that pass screening
        struct QuartetTask {
            Size i, j, k, l;
        };

        std::vector<QuartetTask> tasks;
        tasks.reserve(n_shells * n_shells);  // Conservative estimate

        if (use_symmetry) {
            // Canonical iteration: i<=j, k<=l, pair(i,j) <= pair(k,l)
            for (Size i = 0; i < n_shells; ++i) {
                for (Size j = i; j < n_shells; ++j) {
                    for (Size k = i; k < n_shells; ++k) {
                        const Size l_start = (k == i) ? j : k;
                        for (Size l = l_start; l < n_shells; ++l) {
                            if (!schwarz_bounds_->passes_screening(i, j, k, l, threshold)) {
                                continue;
                            }
                            if (use_density &&
                                !density_screening_->passes_screening(
                                    i, j, k, l, *schwarz_bounds_, threshold)) {
                                continue;
                            }
                            tasks.push_back({i, j, k, l});
                        }
                    }
                }
            }
        } else {
            // Full N^4 iteration
            for (Size i = 0; i < n_shells; ++i) {
                for (Size j = 0; j < n_shells; ++j) {
                    for (Size k = 0; k < n_shells; ++k) {
                        for (Size l = 0; l < n_shells; ++l) {
                            if (!schwarz_bounds_->passes_screening(i, j, k, l, threshold)) {
                                continue;
                            }
                            if (use_density &&
                                !density_screening_->passes_screening(
                                    i, j, k, l, *schwarz_bounds_, threshold)) {
                                continue;
                            }
                            tasks.push_back({i, j, k, l});
                        }
                    }
                }
            }
        }

        const Size n_tasks = tasks.size();

#if defined(_OPENMP)
        if (n_threads > 0) {
            omp_set_num_threads(n_threads);
        }

        #pragma omp parallel
        {
            TwoElectronBuffer<0> local_buffer;

            #pragma omp for schedule(dynamic, 8)
            for (Size t = 0; t < n_tasks; ++t) {
                const auto& task = tasks[t];

                const auto& shell_a = basis_->shell(task.i);
                const auto& shell_b = basis_->shell(task.j);
                const auto& shell_c = basis_->shell(task.k);
                const auto& shell_d = basis_->shell(task.l);

                const Index fi = shell_a.function_index();
                const Index fj = shell_b.function_index();
                const Index fk = shell_c.function_index();
                const Index fl = shell_d.function_index();

                const int na = shell_a.n_functions();
                const int nb = shell_b.n_functions();
                const int nc = shell_c.n_functions();
                const int nd = shell_d.n_functions();

                cpu_engine_.compute_2e_shell_quartet(op, shell_a, shell_b,
                                                     shell_c, shell_d, local_buffer);

                if constexpr (SymmetryAwareConsumer<Consumer>) {
                    if (use_symmetry) {
                        consumer.accumulate_symmetric(
                            local_buffer, fi, fj, fk, fl, na, nb, nc, nd,
                            /*ij_same=*/(task.i == task.j),
                            /*kl_same=*/(task.k == task.l),
                            /*braket_same=*/(task.i == task.k && task.j == task.l));
                    } else {
                        consumer.accumulate(local_buffer, fi, fj, fk, fl, na, nb, nc, nd);
                    }
                } else {
                    consumer.accumulate(local_buffer, fi, fj, fk, fl, na, nb, nc, nd);
                }
            }
        }
#else
        // Fallback to sequential screened path
        compute_and_consume_screened_impl(op, consumer, options);
#endif
    }
    /// @brief Estimate total number of primitives in the basis set
    [[nodiscard]] Size estimate_primitives() const {
        Size total = 0;
        for (Size i = 0; i < basis_->n_shells(); ++i) {
            total += static_cast<Size>(basis_->shell(i).n_primitives());
        }
        return total * total;  // Rough estimate for pairs
    }

    /// @brief Estimate primitives in a ShellSetPair
    [[nodiscard]] Size estimate_shell_set_pair_primitives(const ShellSetPair& pair) const {
        return static_cast<Size>(pair.shell_set_a().n_primitives_per_shell() *
                                 pair.shell_set_b().n_primitives_per_shell() *
                                 pair.shell_set_a().n_shells() *
                                 pair.shell_set_b().n_shells());
    }

    /// @brief Estimate primitives in a ShellSetQuartet
    [[nodiscard]] Size estimate_shell_set_quartet_primitives(const ShellSetQuartet& quartet) const {
        return static_cast<Size>(
            quartet.bra_pair().shell_set_a().n_primitives_per_shell() *
            quartet.bra_pair().shell_set_b().n_primitives_per_shell() *
            quartet.ket_pair().shell_set_a().n_primitives_per_shell() *
            quartet.ket_pair().shell_set_b().n_primitives_per_shell() *
            quartet.bra_pair().shell_set_a().n_shells() *
            quartet.bra_pair().shell_set_b().n_shells() *
            quartet.ket_pair().shell_set_a().n_shells() *
            quartet.ket_pair().shell_set_b().n_shells());
    }

    const BasisSet* basis_;                ///< Pointer to the basis set
    engine::CpuEngine cpu_engine_;         ///< CPU backend (always available)
    DispatchPolicy dispatch_policy_;       ///< Policy for backend selection
    bool gpu_available_{false};            ///< Whether GPU backend is available

    /// @brief Precomputed Schwarz bounds for screening
    std::unique_ptr<screening::SchwarzBounds> schwarz_bounds_;

    /// @brief Thread-safe once_flag for lazy Schwarz bounds initialization
    mutable std::once_flag schwarz_init_flag_;

    /// @brief Density-weighted screening (optional, set via set_density_matrix)
    std::unique_ptr<screening::DensityScreening> density_screening_;

#if LIBACCINT_USE_CUDA
    std::optional<CudaEngine> cuda_engine_;  ///< Optional CUDA backend
#endif
};

}  // namespace libaccint
