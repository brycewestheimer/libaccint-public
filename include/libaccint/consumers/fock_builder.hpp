// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file fock_builder.hpp
/// @brief FockBuilder consumer for two-electron integral accumulation
///
/// Implements the compute-and-consume pattern by accumulating Coulomb (J)
/// and exchange (K) matrix contributions from each shell quartet's integrals.
///
/// Thread-safety modes:
///   - Sequential (default): Standard non-thread-safe accumulation
///   - Atomic: Uses atomic operations for thread-safe accumulation (Phase 4)
///   - ThreadLocal: Uses thread-local buffers with final reduction (Phase 4)

#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/core/types.hpp>

#include <atomic>
#include <memory>
#include <span>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libaccint::consumers {

/// @brief Threading strategy for FockBuilder
enum class FockThreadingStrategy {
    Sequential,   ///< Non-thread-safe (default)
    Atomic,       ///< Uses atomic operations for thread safety
    ThreadLocal   ///< Uses thread-local buffers with final reduction
};

/// @brief Builds Coulomb (J) and exchange (K) matrices from two-electron integrals
///
/// FockBuilder accumulates J and K contributions using the density matrix D:
///   J_mu_nu     += sum_lambda,sigma (mu nu | lambda sigma) * D_lambda_sigma
///   K_mu_lambda += sum_nu,sigma     (mu nu | lambda sigma) * D_nu_sigma
///
/// The accumulate() method processes integrals from a TwoElectronBuffer
/// for a specific shell quartet, extracting the integral values and
/// distributing them into J and K using the appropriate index mapping.
///
/// Thread-safety: parallel engine entry points invoke prepare/finalize lifecycle
/// methods. If strategy remains Sequential, FockBuilder uses an internal
/// thread-local fallback for safety when multiple threads are requested.
class FockBuilder {
public:
    /// @brief Construct a FockBuilder for a basis with nbf functions
    /// @param nbf Number of basis functions
    /// @throws std::invalid_argument if nbf is zero
    explicit FockBuilder(Size nbf);

    /// @brief Set the density matrix for accumulation
    /// @param D Pointer to row-major nbf x nbf density matrix
    /// @param nbf Number of basis functions (must match constructor)
    /// @throws std::invalid_argument if D is nullptr or nbf does not match the constructor value
    void set_density(const Real* D, Size nbf);

    /// @brief Accumulate J and K contributions from a buffer of integrals
    ///
    /// For a shell quartet (a b | c d) with basis function offsets
    /// (fa, fb, fc, fd), distributes all integrals into J and K matrices.
    ///
    /// @param buffer The computed integrals for this quartet
    /// @param fa Starting basis function index for shell a
    /// @param fb Starting basis function index for shell b
    /// @param fc Starting basis function index for shell c
    /// @param fd Starting basis function index for shell d
    /// @param na Number of functions in shell a
    /// @param nb Number of functions in shell b
    /// @param nc Number of functions in shell c
    /// @param nd Number of functions in shell d
    void accumulate(const TwoElectronBuffer<0>& buffer,
                    Index fa, Index fb, Index fc, Index fd,
                    int na, int nb, int nc, int nd);

    /// @brief Symmetry-aware J/K accumulation from a canonical shell quartet
    ///
    /// For a canonical shell quartet (i,j,k,l) with i<=j, k<=l, (i,j)<=(k,l),
    /// scatter the integral contributions into all 8 (or fewer, for degenerate
    /// quartets) permutation-equivalent J and K matrix slots.
    ///
    /// This allows the engine to iterate only canonical quartets, computing
    /// each ERI at most once instead of up to 8 times.
    ///
    /// @param buffer The computed integrals for this canonical quartet
    /// @param fa Starting basis function index for shell a (bra left)
    /// @param fb Starting basis function index for shell b (bra right)
    /// @param fc Starting basis function index for shell c (ket left)
    /// @param fd Starting basis function index for shell d (ket right)
    /// @param na Number of functions in shell a
    /// @param nb Number of functions in shell b
    /// @param nc Number of functions in shell c
    /// @param nd Number of functions in shell d
    /// @param ij_same True if shell i == shell j (bra degenerate)
    /// @param kl_same True if shell k == shell l (ket degenerate)
    /// @param braket_same True if bra pair == ket pair (i==k && j==l)
    void accumulate_symmetric(const TwoElectronBuffer<0>& buffer,
                              Index fa, Index fb, Index fc, Index fd,
                              int na, int nb, int nc, int nd,
                              bool ij_same, bool kl_same, bool braket_same);

    /// @brief Get the Coulomb matrix J
    /// @return Const span over the nbf x nbf J matrix (row-major)
    [[nodiscard]] std::span<const Real> get_coulomb_matrix() const noexcept {
        return std::span<const Real>(J_);
    }

    /// @brief Get the exchange matrix K
    /// @return Const span over the nbf x nbf K matrix (row-major)
    [[nodiscard]] std::span<const Real> get_exchange_matrix() const noexcept {
        return std::span<const Real>(K_);
    }

    /// @brief Compute the Fock matrix F = H_core + J - exchange_fraction * K
    /// @param H_core Core Hamiltonian matrix (row-major, nbf x nbf)
    /// @param exchange_fraction Fraction of exact exchange (1.0 for RHF)
    /// @return Vector containing the Fock matrix (row-major, nbf x nbf)
    /// @throws std::invalid_argument if H_core size does not equal nbf * nbf
    [[nodiscard]] std::vector<Real> get_fock_matrix(
        std::span<const Real> H_core,
        Real exchange_fraction = 1.0) const;

    /// @brief Reset J and K matrices to zero
    void reset() noexcept;

    /// @brief Get the number of basis functions
    [[nodiscard]] Size nbf() const noexcept { return nbf_; }

    // =========================================================================
    // Thread-Safety Configuration (Phase 4)
    // =========================================================================

    /// @brief Set the threading strategy for accumulation
    /// @param strategy The threading mode to use
    /// @throws NotImplementedException if the requested strategy is not supported on this platform
    ///
    /// Call this before using the FockBuilder in parallel regions.
    /// After changing strategy, call reset() to initialize appropriate buffers.
    void set_threading_strategy(FockThreadingStrategy strategy);

    /// @brief Get the current threading strategy
    [[nodiscard]] FockThreadingStrategy threading_strategy() const noexcept {
        return strategy_;
    }

    /// @brief Prepare thread-local buffers for parallel execution
    /// @param n_threads Number of threads (0 = auto-detect from OMP_NUM_THREADS)
    /// @throws std::invalid_argument if n_threads is negative
    ///
    /// Must be called before parallel region when using ThreadLocal strategy.
    void prepare_parallel(int n_threads = 0);

    /// @brief Reduce thread-local buffers into the main J/K matrices
    ///
    /// Must be called after parallel region when using ThreadLocal strategy.
    /// This performs the final reduction sum.
    void finalize_parallel();

private:
    Size nbf_;                 ///< Number of basis functions
    std::vector<Real> J_;      ///< Coulomb matrix (nbf x nbf, row-major)
    std::vector<Real> K_;      ///< Exchange matrix (nbf x nbf, row-major)
    const Real* D_{nullptr};   ///< Pointer to density matrix

    // Thread-safety members
    FockThreadingStrategy strategy_{FockThreadingStrategy::Sequential};
    bool auto_thread_local_fallback_{false};
    bool auto_atomic_fallback_{false};

    // Thread-local storage for ThreadLocal strategy
    // Each thread has its own J and K buffers
    std::vector<std::vector<Real>> J_thread_local_;
    std::vector<std::vector<Real>> K_thread_local_;
    int n_threads_{1};

    /// @brief Get thread-local buffer index
    [[nodiscard]] int get_thread_id() const noexcept;

    /// @brief Thread-safe atomic accumulation helper
    void accumulate_atomic_impl(const TwoElectronBuffer<0>& buffer,
                                Index fa, Index fb, Index fc, Index fd,
                                int na, int nb, int nc, int nd);

    /// @brief Thread-local accumulation helper
    void accumulate_thread_local_impl(const TwoElectronBuffer<0>& buffer,
                                       Index fa, Index fb, Index fc, Index fd,
                                       int na, int nb, int nc, int nd);
};

}  // namespace libaccint::consumers
