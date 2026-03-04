// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file mpi_fock_builder.cpp
/// @brief MPI-distributed Fock matrix builder implementation

#include <libaccint/mpi/mpi_fock_builder.hpp>

#if LIBACCINT_USE_MPI

#include <libaccint/mpi/mpi_utils.hpp>

#include <algorithm>

namespace libaccint::mpi {

// ============================================================================
// Constructor / Destructor
// ============================================================================

MPIFockBuilder::MPIFockBuilder(MPI_Comm comm, Size nbf)
    : comm_(comm),
      local_builder_(std::make_unique<consumers::FockBuilder>(nbf)) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
}

MPIFockBuilder::~MPIFockBuilder() = default;

MPIFockBuilder::MPIFockBuilder(MPIFockBuilder&& other) noexcept
    : comm_(other.comm_),
      rank_(other.rank_),
      size_(other.size_),
      local_builder_(std::move(other.local_builder_)) {
}

MPIFockBuilder& MPIFockBuilder::operator=(MPIFockBuilder&& other) noexcept {
    if (this != &other) {
        std::scoped_lock lock(host_accumulate_mutex_, other.host_accumulate_mutex_);
        comm_ = other.comm_;
        rank_ = other.rank_;
        size_ = other.size_;
        local_builder_ = std::move(other.local_builder_);
    }
    return *this;
}

// ============================================================================
// Density Matrix
// ============================================================================

void MPIFockBuilder::set_density(const Real* D, Size nbf) {
    local_builder_->set_density(D, nbf);
}

void MPIFockBuilder::set_threading_strategy(
    consumers::FockThreadingStrategy strategy) {
    local_builder_->set_threading_strategy(strategy);
}

consumers::FockThreadingStrategy MPIFockBuilder::threading_strategy() const noexcept {
    return local_builder_->threading_strategy();
}

void MPIFockBuilder::prepare_parallel(int n_threads) {
    local_builder_->prepare_parallel(n_threads);
}

void MPIFockBuilder::finalize_parallel() {
    local_builder_->finalize_parallel();
}

// ============================================================================
// Accumulation
// ============================================================================

void MPIFockBuilder::accumulate(const TwoElectronBuffer<0>& buffer,
                                 Index fa, Index fb, Index fc, Index fd,
                                 int na, int nb, int nc, int nd) {
    local_builder_->accumulate(buffer, fa, fb, fc, fd, na, nb, nc, nd);
}

void MPIFockBuilder::accumulate(const double* flat_eri,
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

// ============================================================================
// MPI Reduction
// ============================================================================

void MPIFockBuilder::allreduce() {
    const Size n = local_builder_->nbf() * local_builder_->nbf();

    // Allreduce J in-place
    {
        auto j_span = local_builder_->get_coulomb_matrix();
        auto* j_ptr = const_cast<Real*>(j_span.data());
        chunked_allreduce_inplace(j_ptr, n, comm_);
    }

    // Allreduce K in-place
    {
        auto k_span = local_builder_->get_exchange_matrix();
        auto* k_ptr = const_cast<Real*>(k_span.data());
        chunked_allreduce_inplace(k_ptr, n, comm_);
    }
}

void MPIFockBuilder::reduce_to_root() {
    const Size n = local_builder_->nbf() * local_builder_->nbf();

    // Reduce J in-place
    {
        auto j_span = local_builder_->get_coulomb_matrix();
        auto* j_ptr = const_cast<Real*>(j_span.data());
        chunked_reduce_inplace(j_ptr, n, 0, rank_, comm_);
    }

    // Reduce K in-place
    {
        auto k_span = local_builder_->get_exchange_matrix();
        auto* k_ptr = const_cast<Real*>(k_span.data());
        chunked_reduce_inplace(k_ptr, n, 0, rank_, comm_);
    }
}

// ============================================================================
// Accessors
// ============================================================================

std::span<const Real> MPIFockBuilder::get_coulomb_matrix() const noexcept {
    return local_builder_->get_coulomb_matrix();
}

std::span<const Real> MPIFockBuilder::get_exchange_matrix() const noexcept {
    return local_builder_->get_exchange_matrix();
}

std::vector<Real> MPIFockBuilder::get_fock_matrix(
    std::span<const Real> H_core, Real exchange_fraction) const {
    return local_builder_->get_fock_matrix(H_core, exchange_fraction);
}

void MPIFockBuilder::reset() noexcept {
    local_builder_->reset();
}

Size MPIFockBuilder::nbf() const noexcept {
    return local_builder_->nbf();
}

}  // namespace libaccint::mpi

#endif  // LIBACCINT_USE_MPI
