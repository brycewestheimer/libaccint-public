// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file mpi_fock_builder.hpp
/// @brief MPI-distributed Fock matrix builder
///
/// Wraps multiple FockBuilder instances across MPI ranks, handling
/// distributed accumulation and reduction of Coulomb (J) and exchange (K)
/// matrices. Each rank computes its share of shell quartets; results are
/// combined via MPI_Allreduce.

#include <libaccint/config.hpp>

#include <memory>
#include <mutex>
#include <span>
#include <vector>

#if LIBACCINT_USE_MPI

#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/mpi/mpi_guard.hpp>

#include <mpi.h>
#include <stdexcept>

namespace libaccint::mpi {

// Forward declaration
class MPIEngine;

/// @brief MPI-distributed Fock matrix builder
///
/// MPIFockBuilder wraps a local FockBuilder and provides MPI reduction
/// of the accumulated Coulomb (J) and exchange (K) matrices across ranks.
///
/// Usage:
/// @code
///   MPIGuard mpi(&argc, &argv);
///   BasisSet basis(shells);
///   MPIEngineConfig config;
///   MPIEngine engine(basis, config);
///
///   MPIFockBuilder fock(engine.comm(), basis.nbf());
///   fock.set_density(D, basis.nbf());
///   engine.compute_all_eri(fock);
///   fock.allreduce();
///
///   auto J = fock.get_coulomb_matrix();
///   auto K = fock.get_exchange_matrix();
/// @endcode
class MPIFockBuilder {
public:
    /// @brief Construct an MPI-distributed Fock builder
    /// @param comm MPI communicator
    /// @param nbf Number of basis functions
    explicit MPIFockBuilder(MPI_Comm comm, Size nbf);

    /// @brief Destructor
    ~MPIFockBuilder();

    // Non-copyable
    MPIFockBuilder(const MPIFockBuilder&) = delete;
    MPIFockBuilder& operator=(const MPIFockBuilder&) = delete;

    // Moveable
    MPIFockBuilder(MPIFockBuilder&&) noexcept;
    MPIFockBuilder& operator=(MPIFockBuilder&&) noexcept;

    // =========================================================================
    // Density Matrix
    // =========================================================================

    /// @brief Set the density matrix for accumulation
    /// @param D Pointer to row-major nbf x nbf density matrix
    /// @param nbf Number of basis functions (must match constructor)
    void set_density(const Real* D, Size nbf);

    /// @brief Configure local per-rank Fock accumulation threading strategy
    void set_threading_strategy(consumers::FockThreadingStrategy strategy);

    /// @brief Get the configured local threading strategy
    [[nodiscard]] consumers::FockThreadingStrategy threading_strategy() const noexcept;

    /// @brief Prepare the local builder for threaded accumulation
    void prepare_parallel(int n_threads = 0);

    /// @brief Finalize the local builder after threaded accumulation
    void finalize_parallel();

    // =========================================================================
    // Accumulation (local, called per shell quartet)
    // =========================================================================

    /// @brief Accumulate J and K contributions from a buffer of integrals
    ///
    /// Delegates to the local FockBuilder. This is called for each shell
    /// quartet assigned to this rank.
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

    /// @brief Accumulate host-resident flat ERI data for a ShellSetQuartet
    ///
    /// Used by the GPU-backed MPI path after local device-to-host transfer.
    void accumulate(const double* flat_eri, const ShellSetQuartet& quartet);

    // =========================================================================
    // MPI Reduction
    // =========================================================================

    /// @brief All-reduce J and K matrices across all ranks (MPI_SUM)
    ///
    /// After all local shell quartets have been processed, call this to
    /// combine partial J and K matrices from every rank. The result is
    /// available on all ranks.
    void allreduce();

    /// @brief Reduce J and K matrices to the root rank (rank 0)
    ///
    /// After all local shell quartets have been processed, call this to
    /// combine partial J and K matrices. The result is only valid on rank 0.
    void reduce_to_root();

    // =========================================================================
    // Accessors
    // =========================================================================

    /// @brief Get the Coulomb matrix J (valid after reduction)
    [[nodiscard]] std::span<const Real> get_coulomb_matrix() const noexcept;

    /// @brief Get the exchange matrix K (valid after reduction)
    [[nodiscard]] std::span<const Real> get_exchange_matrix() const noexcept;

    /// @brief Compute the Fock matrix F = H_core + J - exchange_fraction * K
    /// @param H_core Core Hamiltonian matrix (row-major, nbf x nbf)
    /// @param exchange_fraction Fraction of exact exchange (1.0 for RHF)
    /// @return Vector containing the Fock matrix (row-major, nbf x nbf)
    [[nodiscard]] std::vector<Real> get_fock_matrix(
        std::span<const Real> H_core,
        Real exchange_fraction = 1.0) const;

    /// @brief Reset J and K matrices to zero on all ranks
    void reset() noexcept;

    /// @brief Get the number of basis functions
    [[nodiscard]] Size nbf() const noexcept;

    /// @brief Get MPI rank
    [[nodiscard]] int rank() const noexcept { return rank_; }

    /// @brief Get MPI size (total ranks)
    [[nodiscard]] int size() const noexcept { return size_; }

    /// @brief Check if this is the root rank
    [[nodiscard]] bool is_root() const noexcept { return rank_ == 0; }

    /// @brief Get the communicator
    [[nodiscard]] MPI_Comm comm() const noexcept { return comm_; }

private:
    MPI_Comm comm_;
    int rank_ = 0;
    int size_ = 1;

    /// Local FockBuilder for this rank's share of shell quartets
    std::unique_ptr<consumers::FockBuilder> local_builder_;
    mutable std::mutex host_accumulate_mutex_;
};

}  // namespace libaccint::mpi

#else  // !LIBACCINT_USE_MPI

// Stub for non-MPI builds — provides the same API shape for test compilation
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/consumers/fock_builder.hpp>

namespace libaccint::mpi {

/// @brief Stub MPIFockBuilder for non-MPI builds
///
/// Wraps a single local FockBuilder. All "MPI" operations are no-ops.
class MPIFockBuilder {
public:
    explicit MPIFockBuilder(void* /*comm*/, Size nbf)
        : local_builder_(std::make_unique<consumers::FockBuilder>(nbf)) {}

    MPIFockBuilder(MPIFockBuilder&& other) noexcept
        : local_builder_(std::move(other.local_builder_)) {}

    MPIFockBuilder& operator=(MPIFockBuilder&& other) noexcept {
        if (this != &other) {
            std::scoped_lock lock(host_accumulate_mutex_, other.host_accumulate_mutex_);
            local_builder_ = std::move(other.local_builder_);
        }
        return *this;
    }
    
    void set_density(const Real* D, Size nbf) {
        local_builder_->set_density(D, nbf);
    }

    void set_threading_strategy(consumers::FockThreadingStrategy strategy) {
        local_builder_->set_threading_strategy(strategy);
    }

    [[nodiscard]] consumers::FockThreadingStrategy threading_strategy() const noexcept {
        return local_builder_->threading_strategy();
    }

    void prepare_parallel(int n_threads = 0) {
        local_builder_->prepare_parallel(n_threads);
    }

    void finalize_parallel() {
        local_builder_->finalize_parallel();
    }
    
    void accumulate(const TwoElectronBuffer<0>& buffer,
                    Index fa, Index fb, Index fc, Index fd,
                    int na, int nb, int nc, int nd) {
        local_builder_->accumulate(buffer, fa, fb, fc, fd, na, nb, nc, nd);
    }

    void accumulate(const double* flat_eri, const ShellSetQuartet& quartet);
    
    void allreduce() {}  // No-op — single rank
    void reduce_to_root() {}  // No-op — single rank
    
    [[nodiscard]] std::span<const Real> get_coulomb_matrix() const noexcept {
        return local_builder_->get_coulomb_matrix();
    }
    [[nodiscard]] std::span<const Real> get_exchange_matrix() const noexcept {
        return local_builder_->get_exchange_matrix();
    }
    [[nodiscard]] std::vector<Real> get_fock_matrix(
        std::span<const Real> H_core, Real exchange_fraction = 1.0) const {
        return local_builder_->get_fock_matrix(H_core, exchange_fraction);
    }
    
    void reset() noexcept { local_builder_->reset(); }
    [[nodiscard]] Size nbf() const noexcept { return local_builder_->nbf(); }
    [[nodiscard]] int rank() const noexcept { return 0; }
    [[nodiscard]] int size() const noexcept { return 1; }
    [[nodiscard]] bool is_root() const noexcept { return true; }

private:
    std::unique_ptr<consumers::FockBuilder> local_builder_;
    mutable std::mutex host_accumulate_mutex_;
};

inline void MPIFockBuilder::accumulate(const double* flat_eri,
                                       const ShellSetQuartet& quartet) {
    std::lock_guard<std::mutex> lock(host_accumulate_mutex_);

    const auto& set_a = quartet.bra_pair().shell_set_a();
    const auto& set_b = quartet.bra_pair().shell_set_b();
    const auto& set_c = quartet.ket_pair().shell_set_a();
    const auto& set_d = quartet.ket_pair().shell_set_b();

    const int na_funcs = n_cartesian(set_a.angular_momentum());
    const int nb_funcs = n_cartesian(set_b.angular_momentum());
    const int nc_funcs = n_cartesian(set_c.angular_momentum());
    const int nd_funcs = n_cartesian(set_d.angular_momentum());
    const Size funcs_per_quartet =
        static_cast<Size>(na_funcs) * nb_funcs * nc_funcs * nd_funcs;

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

                    for (int a = 0; a < na_funcs; ++a) {
                        for (int b = 0; b < nb_funcs; ++b) {
                            for (int c = 0; c < nc_funcs; ++c) {
                                for (int d = 0; d < nd_funcs; ++d) {
                                    buffer(a, b, c, d) = flat_eri[
                                        flat_idx +
                                        static_cast<size_t>(a) * nb_funcs * nc_funcs * nd_funcs +
                                        static_cast<size_t>(b) * nc_funcs * nd_funcs +
                                        static_cast<size_t>(c) * nd_funcs + d];
                                }
                            }
                        }
                    }

                    local_builder_->accumulate_symmetric(
                        buffer, fi, fj, fk, fl,
                        na_funcs, nb_funcs, nc_funcs, nd_funcs,
                        ij_same, kl_same, braket_same);

                    flat_idx += funcs_per_quartet;
                }
            }
        }
    }
}
}  // namespace libaccint::mpi

#endif  // LIBACCINT_USE_MPI
