// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file shell_set_pair.cpp
/// @brief ShellSetPair class implementation with Schwarz bound and pair data

#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/kernels/eri_kernel.hpp>
#include <libaccint/math/gaussian_product.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>

#include <cmath>
#include <algorithm>
#include <thread>

namespace libaccint {

// =============================================================================
// Schwarz Bound Implementation
// =============================================================================

Real ShellSetPair::schwarz_bound() const {
    while (true) {
        if (schwarz_cache_->computed.load(std::memory_order_acquire)) {
            return schwarz_cache_->bound;
        }

        bool expected = false;
        if (schwarz_cache_->computing.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            try {
                compute_schwarz_bound_impl();
                schwarz_cache_->computed.store(true, std::memory_order_release);
            } catch (...) {
                schwarz_cache_->computing.store(false, std::memory_order_release);
                throw;
            }
            schwarz_cache_->computing.store(false, std::memory_order_release);
            return schwarz_cache_->bound;
        }

        // Another thread is computing. Wait for state transition; if the
        // computation fails and computing is cleared without computed=true, we
        // loop and one thread will retry computation.
        while (schwarz_cache_->computing.load(std::memory_order_acquire) &&
               !schwarz_cache_->computed.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }
}

void ShellSetPair::precompute_schwarz_bound() const {
    // Simply call schwarz_bound() to trigger lazy computation
    (void)schwarz_bound();
}

void ShellSetPair::compute_schwarz_bound_impl() const {
    // Q_ab = sqrt(max (ab|ab))
    // We need to find the maximum diagonal ERI over all shell pairs (a,b) in this ShellSetPair

    Real max_eri = 0.0;
    TwoElectronBuffer<0> buffer;

    const Size n_shells_a = set_a_->n_shells();
    const Size n_shells_b = set_b_->n_shells();

    // Iterate over all shell pairs in this ShellSetPair
    for (Size i = 0; i < n_shells_a; ++i) {
        const auto& shell_a = set_a_->shell(i);

        for (Size j = 0; j < n_shells_b; ++j) {
            const auto& shell_b = set_b_->shell(j);

            // Compute diagonal ERIs: (ab|ab)
            kernels::compute_eri(shell_a, shell_b, shell_a, shell_b, buffer);

            // Find maximum absolute value over all function indices
            const int na = shell_a.n_functions();
            const int nb = shell_b.n_functions();

            for (int a = 0; a < na; ++a) {
                for (int b = 0; b < nb; ++b) {
                    // Diagonal element: (a,b|a,b)
                    Real eri_val = std::abs(buffer(a, b, a, b));
                    max_eri = std::max(max_eri, eri_val);
                }
            }
        }
    }

    schwarz_cache_->bound = std::sqrt(max_eri);
}

// =============================================================================
// PrimitivePairData Implementation
// =============================================================================

const PrimitivePairData& ShellSetPair::pair_data() const {
    while (true) {
        if (pair_data_cache_->computed.load(std::memory_order_acquire)) {
            return pair_data_cache_->data;
        }

        bool expected = false;
        if (pair_data_cache_->computing.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            try {
                build_pair_data_impl();
                pair_data_cache_->computed.store(true, std::memory_order_release);
            } catch (...) {
                pair_data_cache_->computing.store(false, std::memory_order_release);
                throw;
            }
            pair_data_cache_->computing.store(false, std::memory_order_release);
            return pair_data_cache_->data;
        }

        while (pair_data_cache_->computing.load(std::memory_order_acquire) &&
               !pair_data_cache_->computed.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }
}

void ShellSetPair::precompute_pair_data() const {
    (void)pair_data();
}

void ShellSetPair::build_pair_data_impl() const {
    const Size na_shells = set_a_->n_shells();
    const Size nb_shells = set_b_->n_shells();
    const auto Ka = static_cast<Size>(set_a_->n_primitives_per_shell());
    const auto Kb = static_cast<Size>(set_b_->n_primitives_per_shell());

    auto& data = pair_data_cache_->data;
    data.allocate(na_shells, nb_shells, Ka, Kb);

    for (Size i = 0; i < na_shells; ++i) {
        const auto& shell_a = set_a_->shell(i);
        const auto& A = shell_a.center();
        const auto exp_a = shell_a.exponents();
        const auto coeff_a = shell_a.coefficients();

        for (Size j = 0; j < nb_shells; ++j) {
            const auto& shell_b = set_b_->shell(j);
            const auto& B = shell_b.center();
            const auto exp_b = shell_b.exponents();
            const auto coeff_b = shell_b.coefficients();

            for (Size p = 0; p < Ka; ++p) {
                const Real alpha = exp_a[p];
                const Real ca = coeff_a[p];

                for (Size q = 0; q < Kb; ++q) {
                    const Real beta = exp_b[q];
                    const Real cb = coeff_b[q];

                    const auto gp = math::compute_gaussian_product(alpha, A, beta, B);
                    const Size idx = data.pair_index(i, j, p, q);

                    data.Px[idx] = gp.P.x;
                    data.Py[idx] = gp.P.y;
                    data.Pz[idx] = gp.P.z;
                    data.zeta[idx] = gp.zeta;
                    data.one_over_2zeta[idx] = 0.5 / gp.zeta;
                    data.mu[idx] = gp.mu;
                    data.K_AB[idx] = gp.K_AB;
                    data.coeff_product[idx] = ca * cb;
                    data.PA_x[idx] = gp.P.x - A.x;
                    data.PA_y[idx] = gp.P.y - A.y;
                    data.PA_z[idx] = gp.P.z - A.z;
                    data.PB_x[idx] = gp.P.x - B.x;
                    data.PB_y[idx] = gp.P.y - B.y;
                    data.PB_z[idx] = gp.P.z - B.z;
                }
            }
        }
    }
}

}  // namespace libaccint
