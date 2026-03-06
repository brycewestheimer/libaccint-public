// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file cpu_engine.hpp
/// @brief CPU backend engine for molecular integral computation

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/core/backend.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/engine/engine_backend.hpp>
#include <libaccint/engine/thread_config.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/consumers/consumer_concepts.hpp>
#include <libaccint/engine/integral_buffer.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <functional>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libaccint::engine {

/// @brief CPU backend engine for molecular integral computation
///
/// CpuEngine provides CPU-based computation of molecular integrals using
/// vectorized (SIMD) kernels and OpenMP parallelization. It satisfies the
/// EngineBackend concept and can be used directly or through the unified
/// Engine class.
///
/// Usage:
/// @code
///   BasisSet basis(shells);
///   engine::CpuEngine engine(basis);
///
///   OverlapBuffer buf;
///   engine.compute_1e_shell_pair(Operator::overlap(), shell_a, shell_b, buf);
/// @endcode
class CpuEngine {
public:
    /// @brief Construct a CpuEngine with a BasisSet
    /// @param basis The basis set to use (must remain valid for Engine lifetime)
    explicit CpuEngine(const BasisSet& basis)
        : basis_(&basis) {
        // Pre-allocate internal buffer sized for worst-case shell pair
        int max_nf = n_cartesian(basis.max_angular_momentum());
        buffer_1e_.resize(max_nf, max_nf);
    }

    /// @brief Get the basis set
    [[nodiscard]] const BasisSet& basis() const noexcept { return *basis_; }

    /// @brief Get the maximum angular momentum from the basis set
    [[nodiscard]] int max_angular_momentum() const noexcept {
        return basis_->max_angular_momentum();
    }

    // =========================================================================
    // One-Electron Integrals
    // =========================================================================

    /// @brief Compute one-electron integrals for all shell pairs
    ///
    /// Iterates over all shell pairs in the basis set, computing the
    /// one-electron integrals for the given operator and accumulating
    /// them into the output matrix.
    ///
    /// @tparam DerivOrder Derivative order (0 = energy)
    /// @param op The one-electron operator (or composed operator)
    /// @param result Output matrix (n_basis_functions x n_basis_functions),
    ///               stored in row-major order
    template<int DerivOrder = 0>
    void compute_1e(const OneElectronOperator& op,
                    std::vector<Real>& result) {
        static_assert(DerivOrder == 0,
                      "Only DerivOrder == 0 (energy) is supported in this release");
        compute_1e_impl(op, result);
    }

    /// @brief Compute one-electron integrals with OpenMP parallelism
    ///
    /// Parallelizes the shell pair loop across OpenMP threads using
    /// thread-local result matrices with final reduction.
    ///
    /// @tparam DerivOrder Derivative order (0 = energy)
    /// @param op The one-electron operator (or composed operator)
    /// @param result Output matrix (n_basis_functions x n_basis_functions)
    /// @param n_threads Number of threads (0 = auto-detect)
    template<int DerivOrder = 0>
    void compute_1e_parallel(const OneElectronOperator& op,
                              std::vector<Real>& result,
                              int n_threads = 0) {
        static_assert(DerivOrder == 0,
                      "Only DerivOrder == 0 (energy) is supported in this release");
        compute_1e_parallel_impl(op, result, n_threads);
    }

    /// @brief Compute one-electron integral for a single shell pair
    ///
    /// Computes integrals for the operator between shell_a and shell_b,
    /// storing results in the provided buffer.
    ///
    /// @param op The one-electron operator
    /// @param shell_a First shell (bra)
    /// @param shell_b Second shell (ket)
    /// @param buffer Output buffer for this shell pair
    void compute_1e_shell_pair(const Operator& op,
                               const Shell& shell_a,
                               const Shell& shell_b,
                               OneElectronBuffer<0>& buffer);

    // =========================================================================
    // Two-Electron Integrals
    // =========================================================================

    /// @brief Compute two-electron integral for a single shell quartet
    ///
    /// Dispatches to the appropriate kernel based on operator kind.
    ///
    /// @param op The two-electron operator (e.g., Coulomb)
    /// @param shell_a First bra shell
    /// @param shell_b Second bra shell
    /// @param shell_c First ket shell
    /// @param shell_d Second ket shell
    /// @param buffer Output buffer for this shell quartet
    void compute_2e_shell_quartet(const Operator& op,
                                   const Shell& shell_a,
                                   const Shell& shell_b,
                                   const Shell& shell_c,
                                   const Shell& shell_d,
                                   TwoElectronBuffer<0>& buffer);

    /// @brief Compute two-electron integrals for a ShellSetQuartet (non-consumer)
    ///
    /// Returns an IntegralBuffer with all computed integrals. This bypasses
    /// Engine dispatch and always executes on CPU.
    ///
    /// @param op The two-electron operator
    /// @param quartet ShellSetQuartet to compute
    /// @return IntegralBuffer with computed integrals and metadata
    [[nodiscard]] IntegralBuffer compute_batch(
        const Operator& op,
        const ShellSetQuartet& quartet) {

        if (!op.is_two_electron()) {
            throw InvalidArgumentException(
                "CpuEngine::compute_batch requires a two-electron operator, got: " +
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

        TwoElectronBuffer<0> buffer;

        for (Size ia = 0; ia < set_a.n_shells(); ++ia) {
            const auto& shell_a = set_a.shell(ia);
            for (Size ib = 0; ib < set_b.n_shells(); ++ib) {
                const auto& shell_b = set_b.shell(ib);
                for (Size ic = 0; ic < set_c.n_shells(); ++ic) {
                    const auto& shell_c = set_c.shell(ic);
                    for (Size id = 0; id < set_d.n_shells(); ++id) {
                        const auto& shell_d = set_d.shell(id);

                        compute_2e_shell_quartet(
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

    /// @brief Compute one-electron integrals for a ShellSetPair (non-consumer)
    ///
    /// @param op The one-electron operator
    /// @param pair ShellSetPair to compute
    /// @return IntegralBuffer with computed integrals and metadata
    [[nodiscard]] IntegralBuffer compute_batch(
        const Operator& op,
        const ShellSetPair& pair) {

        if (!op.is_one_electron()) {
            throw InvalidArgumentException(
                "CpuEngine::compute_batch (1e) requires a one-electron operator, got: " +
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

                compute_1e_shell_pair(op, shell_a, shell_b, buffer);

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

    // =========================================================================
    // ShellSet-Based Computation (Batched API)
    // =========================================================================

    /// @brief Compute one-electron integrals for all shell pairs in a ShellSetPair
    ///
    /// Iterates over all shell pairs within the ShellSetPair and computes
    /// integrals for each, accumulating into the result matrix.
    ///
    /// @param op The one-electron operator
    /// @param pair ShellSetPair containing shells to process
    /// @param result Output matrix (nbf x nbf, row-major)
    void compute_shell_set_pair(const Operator& op,
                                 const ShellSetPair& pair,
                                 std::vector<Real>& result);

    /// @brief Compute and consume two-electron integrals for all quartets in a ShellSetQuartet
    ///
    /// Iterates over all shell quartets within the ShellSetQuartet, computes
    /// ERIs, and passes each buffer to the consumer.
    ///
    /// @tparam Consumer Type with an accumulate method
    /// @param op The two-electron operator
    /// @param quartet ShellSetQuartet containing shells to process
    /// @param consumer Consumer object (e.g., FockBuilder)
    template<typename Consumer>
    void compute_shell_set_quartet(const Operator& op,
                                    const ShellSetQuartet& quartet,
                                    Consumer& consumer);

    // =========================================================================
    // Fused Compute-and-Consume
    // =========================================================================

    /// @brief Fused two-electron integral computation and consumption
    ///
    /// Iterates over all shell quartets, computes ERIs into a temporary
    /// buffer, and calls the consumer's accumulate() method for each quartet.
    /// The buffer is reused across iterations.
    ///
    /// The Consumer must provide:
    ///   void accumulate(const TwoElectronBuffer<0>& buffer,
    ///                   Index fa, Index fb, Index fc, Index fd,
    ///                   int na, int nb, int nc, int nd);
    ///
    /// @tparam Consumer Type with an accumulate method
    /// @param op The two-electron operator (e.g., Coulomb)
    /// @param consumer The consumer object (e.g., FockBuilder)
    template<typename Consumer>
    void compute_and_consume(const Operator& op, Consumer& consumer) {
        if (!op.is_two_electron()) {
            throw InvalidArgumentException(
                "compute_and_consume requires a two-electron operator, got: " +
                std::string(to_string(op.kind())));
        }
        compute_and_consume_impl(op, consumer);
    }

    /// @brief Parallel two-electron integral computation and consumption
    ///
    /// OpenMP-parallelized version that distributes shell quartets across threads.
    /// Each thread has its own buffer and accumulates into thread-local storage.
    /// Final reduction is performed via the consumer's reduce() method if available.
    ///
    /// The Consumer must provide:
    ///   - void accumulate(buffer, fa, fb, fc, fd, na, nb, nc, nd)
    ///   - ThreadSafe API: accumulate_atomic() OR thread-local buffers with reduce()
    ///
    /// @tparam Consumer Type with an accumulate method
    /// @param op The two-electron operator (e.g., Coulomb)
    /// @param consumer The consumer object (must be thread-safe)
    /// @param n_threads Number of OpenMP threads to use (0 = auto)
    template<typename Consumer>
    void compute_and_consume_parallel(const Operator& op, Consumer& consumer,
                                       int n_threads = 0);

private:
    /// @brief Implementation of compute_1e for DerivOrder=0
    void compute_1e_impl(const OneElectronOperator& op,
                         std::vector<Real>& result);

    /// @brief Implementation of compute_1e_parallel for DerivOrder=0
    void compute_1e_parallel_impl(const OneElectronOperator& op,
                                   std::vector<Real>& result,
                                   int n_threads);

    /// @brief Implementation of compute_and_consume for two-electron integrals
    template<typename Consumer>
    void compute_and_consume_impl(const Operator& op, Consumer& consumer);

    const BasisSet* basis_;              ///< Pointer to the basis set
    OneElectronBuffer<0> buffer_1e_;     ///< Internal buffer for worst-case shell pair
    TwoElectronBuffer<0> buffer_2e_;     ///< Internal buffer for worst-case shell quartet
};

// =============================================================================
// Template Implementation
// =============================================================================

template<typename Consumer>
void CpuEngine::compute_and_consume_impl(const Operator& op, Consumer& consumer) {
    static_assert(TwoElectronConsumer<Consumer>,
        "Consumer must provide accumulate(const TwoElectronBuffer<0>&, "
        "Index, Index, Index, Index, int, int, int, int)");

    // Drive 2e computation via ShellSetQuartet work units (ADR-0001).
    // Iterate all ShellSet combinations (not upper-triangle) to match
    // the original all-quartets behavior without symmetry exploitation.
    // Shell-level iteration within each quartet remains sequential here;
    // the parallel entry point is compute_shell_set_quartet / compute_and_consume_parallel.
    const auto sets = basis_->shell_sets();
    const Size n_sets = sets.size();

    for (Size ia = 0; ia < n_sets; ++ia) {
        const auto& set_a = *sets[ia];
        for (Size ib = 0; ib < n_sets; ++ib) {
            const auto& set_b = *sets[ib];
            for (Size ic = 0; ic < n_sets; ++ic) {
                const auto& set_c = *sets[ic];
                for (Size id = 0; id < n_sets; ++id) {
                    const auto& set_d = *sets[id];

                    // Iterate all shell combinations within this ShellSetQuartet
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

                                    // Compute ERIs for this shell quartet
                                    compute_2e_shell_quartet(op, shell_a, shell_b,
                                                             shell_c, shell_d, buffer_2e_);

                                    // Pass to consumer
                                    consumer.accumulate(buffer_2e_, fi, fj, fk, fl,
                                                        na, nb, nc, nd);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// ShellSetQuartet-Based Template Implementation
// =============================================================================

template<typename Consumer>
void CpuEngine::compute_shell_set_quartet(const Operator& op,
                                        const ShellSetQuartet& quartet,
                                        Consumer& consumer) {
    static_assert(TwoElectronConsumer<Consumer>,
        "Consumer must provide accumulate(const TwoElectronBuffer<0>&, "
        "Index, Index, Index, Index, int, int, int, int)");
    static_assert(ParallelConsumer<Consumer>,
        "CpuEngine::compute_shell_set_quartet uses OpenMP parallelism; "
        "consumer must provide prepare_parallel(int) and finalize_parallel()");

    // Get ShellSets from the quartet
    const auto& set_a = quartet.bra_pair().shell_set_a();
    const auto& set_b = quartet.bra_pair().shell_set_b();
    const auto& set_c = quartet.ket_pair().shell_set_a();
    const auto& set_d = quartet.ket_pair().shell_set_b();

    const Size na_shells = set_a.n_shells();
    const Size nb_shells = set_b.n_shells();
    const Size nc_shells = set_c.n_shells();
    const Size nd_shells = set_d.n_shells();
    const Size total_quartets = na_shells * nb_shells * nc_shells * nd_shells;

    // Handle empty case
    if (total_quartets == 0) {
        return;
    }

#if defined(_OPENMP)
    // Prepare consumer for parallel execution (allocates thread-local buffers)
    int n_threads = omp_get_max_threads();
    consumer.prepare_parallel(n_threads);

    // OpenMP parallel execution: flatten 4D loop into 1D for better scheduling
    #pragma omp parallel
    {
        // Per-thread buffer to avoid sharing
        TwoElectronBuffer<0> local_buffer;

        #pragma omp for schedule(static)
        for (Size idx = 0; idx < total_quartets; ++idx) {
            // Decode (i, j, k, l) from linear index
            const Size ncd = nc_shells * nd_shells;
            const Size nbcd = nb_shells * ncd;

            const Size i = idx / nbcd;
            Size rem = idx % nbcd;
            const Size j = rem / ncd;
            rem = rem % ncd;
            const Size k = rem / nd_shells;
            const Size l = rem % nd_shells;

            const auto& shell_a = set_a.shell(i);
            const auto& shell_b = set_b.shell(j);
            const auto& shell_c = set_c.shell(k);
            const auto& shell_d = set_d.shell(l);

            const Index fi = shell_a.function_index();
            const Index fj = shell_b.function_index();
            const Index fk = shell_c.function_index();
            const Index fl = shell_d.function_index();

            const int na = shell_a.n_functions();
            const int nb = shell_b.n_functions();
            const int nc = shell_c.n_functions();
            const int nd = shell_d.n_functions();

            // Compute ERIs for this quartet into thread-local buffer
            compute_2e_shell_quartet(
                op, shell_a, shell_b, shell_c, shell_d, local_buffer);

            // Consumer handles thread-safe accumulation (Atomic or ThreadLocal strategy)
            consumer.accumulate(local_buffer, fi, fj, fk, fl, na, nb, nc, nd);
        }
    }

    // Finalize parallel execution (reduces thread-local buffers if using ThreadLocal strategy)
    consumer.finalize_parallel();
#else
    // Sequential fallback when OpenMP not available
    for (Size i = 0; i < na_shells; ++i) {
        const auto& shell_a = set_a.shell(i);
        const Index fi = shell_a.function_index();
        const int na = shell_a.n_functions();

        for (Size j = 0; j < nb_shells; ++j) {
            const auto& shell_b = set_b.shell(j);
            const Index fj = shell_b.function_index();
            const int nb = shell_b.n_functions();

            for (Size k = 0; k < nc_shells; ++k) {
                const auto& shell_c = set_c.shell(k);
                const Index fk = shell_c.function_index();
                const int nc = shell_c.n_functions();

                for (Size l = 0; l < nd_shells; ++l) {
                    const auto& shell_d = set_d.shell(l);
                    const Index fl = shell_d.function_index();
                    const int nd = shell_d.n_functions();

                    // Compute ERIs for this quartet
                    compute_2e_shell_quartet(op, shell_a, shell_b,
                                             shell_c, shell_d, buffer_2e_);

                    // Pass to consumer
                    consumer.accumulate(buffer_2e_, fi, fj, fk, fl,
                                        na, nb, nc, nd);
                }
            }
        }
    }
#endif
}

// =============================================================================
// Parallel Compute-and-Consume Template Implementation
// =============================================================================

template<typename Consumer>
void CpuEngine::compute_and_consume_parallel(const Operator& op, Consumer& consumer,
                                           [[maybe_unused]] int n_threads) {
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

    const Size n_shells = basis_->n_shells();
    const Size n_tasks = n_shells * n_shells * n_shells * n_shells;
    const int actual_threads = ThreadConfig::resolve(n_threads);

    // Prepare consumer lifecycle before entering parallel region.
    // This guarantees safe defaults for consumers that internally switch to
    // thread-local accumulation when parallel execution is requested.
    consumer.prepare_parallel(actual_threads);

#if defined(_OPENMP)
    #pragma omp parallel num_threads(actual_threads)
    {
        // Per-thread buffer
        TwoElectronBuffer<0> local_buffer;

        #pragma omp for schedule(dynamic, 8)
        for (Size t = 0; t < n_tasks; ++t) {
            // Decode flat index into shell quartet indices
            const Size l = t % n_shells;
            const Size k = (t / n_shells) % n_shells;
            const Size j = (t / (n_shells * n_shells)) % n_shells;
            const Size i = t / (n_shells * n_shells * n_shells);

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

            // Compute ERIs
            compute_2e_shell_quartet(
                op, shell_a, shell_b, shell_c, shell_d, local_buffer);

            // Accumulate (consumer must be thread-safe)
            consumer.accumulate(local_buffer, fi, fj, fk, fl, na, nb, nc, nd);
        }
    }
#else
    // Fallback to sequential implementation if OpenMP not available
    compute_and_consume_impl(op, consumer);
#endif

    // Finalize consumer lifecycle (e.g., reduce thread-local buffers).
    consumer.finalize_parallel();
}

// Static assertion to verify CpuEngine satisfies EngineBackend concept
static_assert(EngineBackend<CpuEngine>, "CpuEngine must satisfy EngineBackend concept");

}  // namespace libaccint::engine

// Bring CpuEngine into the libaccint namespace for convenience
namespace libaccint {
    using engine::CpuEngine;
}
