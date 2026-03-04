// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file fock_builder.cpp
/// @brief FockBuilder implementation for Coulomb (J) and exchange (K) matrix construction

#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libaccint::consumers {

namespace {

constexpr std::size_t kDefaultMaxThreadLocalBytes = 512ull * 1024ull * 1024ull;  // 512 MiB

std::size_t max_thread_local_bytes() {
    const char* env = std::getenv("LIBACCINT_MAX_FOCK_THREADLOCAL_BYTES");
    if (env == nullptr || env[0] == '\0') {
        return kDefaultMaxThreadLocalBytes;
    }

    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (end != env && *end == '\0' && parsed > 0) {
        return static_cast<std::size_t>(parsed);
    }
    return kDefaultMaxThreadLocalBytes;
}

bool checked_mul(std::size_t a, std::size_t b, std::size_t& out) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

}  // namespace

FockBuilder::FockBuilder(Size nbf)
    : nbf_(nbf)
    , J_(nbf * nbf, 0.0)
    , K_(nbf * nbf, 0.0)
{}

void FockBuilder::set_density(const Real* D, Size nbf) {
    if (nbf != nbf_) {
        throw InvalidArgumentException(
            "FockBuilder::set_density: nbf mismatch (expected " +
            std::to_string(nbf_) + ", got " + std::to_string(nbf) + ")");
    }
    D_ = D;
}

void FockBuilder::accumulate(const TwoElectronBuffer<0>& buffer,
                              Index fa, Index fb, Index fc, Index fd,
                              int na, int nb, int nc, int nd) {
    if (!D_) {
        throw InvalidStateException("FockBuilder::accumulate called without density matrix");
    }

    // Dispatch based on effective strategy.
    // When auto_thread_local_fallback_ is enabled, Sequential strategy is
    // internally upgraded for thread-safe parallel accumulation.
    const FockThreadingStrategy effective_strategy = auto_atomic_fallback_
        ? FockThreadingStrategy::Atomic
        : (auto_thread_local_fallback_
            ? FockThreadingStrategy::ThreadLocal
            : strategy_);
    switch (effective_strategy) {
        case FockThreadingStrategy::Sequential:
            // Fall through to default implementation below
            break;
        case FockThreadingStrategy::Atomic:
            accumulate_atomic_impl(buffer, fa, fb, fc, fd, na, nb, nc, nd);
            return;
        case FockThreadingStrategy::ThreadLocal:
            accumulate_thread_local_impl(buffer, fa, fb, fc, fd, na, nb, nc, nd);
            return;
    }

    const Size n = nbf_;

    // Accumulate J and K contributions for each integral (mu nu | lambda sigma)
    // The engine iterates over ALL shell quartets (full N^4) so no symmetry
    // factors are needed here. For the upper-triangle-only iteration path,
    // use accumulate_symmetric() which handles 8-fold permutation scattering.
    //
    // J_mu_nu     += (mu nu | lambda sigma) * D_lambda_sigma
    // K_mu_lambda += (mu nu | lambda sigma) * D_nu_sigma

    for (int a = 0; a < na; ++a) {
        const auto mu = static_cast<Size>(fa + a);
        for (int b = 0; b < nb; ++b) {
            const auto nu = static_cast<Size>(fb + b);
            for (int c = 0; c < nc; ++c) {
                const auto lam = static_cast<Size>(fc + c);
                for (int d = 0; d < nd; ++d) {
                    const auto sig = static_cast<Size>(fd + d);

                    const Real eri = buffer(a, b, c, d);

                    // Coulomb: J_mu_nu += (mu nu | lam sig) * D_lam_sig
                    J_[mu * n + nu] += eri * D_[lam * n + sig];

                    // Exchange: K_mu_lam += (mu nu | lam sig) * D_nu_sig
                    K_[mu * n + lam] += eri * D_[nu * n + sig];
                }
            }
        }
    }
}

void FockBuilder::accumulate_symmetric(const TwoElectronBuffer<0>& buffer,
                                        Index fa, Index fb, Index fc, Index fd,
                                        int na, int nb, int nc, int nd,
                                        bool ij_same, bool kl_same, bool braket_same) {
    if (!D_) {
        throw InvalidStateException("FockBuilder::accumulate_symmetric called without density matrix");
    }

    const Size n = nbf_;

    // Accumulate J and K from a canonical shell quartet by scattering
    // contributions from all distinct permutations.
    //
    // For a general quartet (i,j,k,l) with all shells different, there are
    // 8 permutations in the full N^4 loop. The permutation group is:
    //   P1: (i,j,k,l)  P2: (j,i,k,l)  P3: (i,j,l,k)  P4: (j,i,l,k)
    //   P5: (k,l,i,j)  P6: (l,k,i,j)  P7: (k,l,j,i)  P8: (l,k,j,i)
    //
    // When shells are degenerate:
    //   i=j:  P1=P2, P3=P4, P5=P7, P6=P8  → 4 distinct
    //   k=l:  P1=P3, P2=P4, P5=P6, P7=P8  → 4 distinct
    //   ij=kl: P1=P5, P2=P7, P3=P6, P4=P8 → 4 distinct (if i≠j)
    //
    // Each permutation Pm of shell quartet produces an accumulate call:
    //   J[fi_m+a, fj_m+b]  += g * D[fk_m+c, fl_m+d]
    //   K[fi_m+a, fk_m+c]  += g * D[fj_m+b, fl_m+d]
    //
    // Where g = buffer(a,b,c,d) = (mu nu | lam sig) is the same for all
    // permutations due to ERI symmetry.

    for (int a = 0; a < na; ++a) {
        const auto mu = static_cast<Size>(fa + a);
        for (int b = 0; b < nb; ++b) {
            const auto nu = static_cast<Size>(fb + b);
            for (int c = 0; c < nc; ++c) {
                const auto lam = static_cast<Size>(fc + c);
                for (int d = 0; d < nd; ++d) {
                    const auto sig = static_cast<Size>(fd + d);
                    const Real g = buffer(a, b, c, d);

                    // Load density matrix elements (D is symmetric: D[p,q] = D[q,p])
                    const Real D_ls = D_[lam * n + sig];
                    const Real D_sl = D_[sig * n + lam]; // = D_ls
                    const Real D_mn = D_[mu * n + nu];
                    const Real D_nm = D_[nu * n + mu];   // = D_mn
                    const Real D_ns = D_[nu * n + sig];
                    const Real D_ms = D_[mu * n + sig];
                    const Real D_nl = D_[nu * n + lam];
                    const Real D_ml = D_[mu * n + lam];

                    // P1: (i,j,k,l) — always present
                    J_[mu * n + nu]  += g * D_ls;
                    K_[mu * n + lam] += g * D_ns;

                    // P3: (i,j,l,k) — present if k ≠ l
                    if (!kl_same) {
                        J_[mu * n + nu]  += g * D_sl;
                        K_[mu * n + sig] += g * D_nl;
                    }

                    // P2: (j,i,k,l) — present if i ≠ j
                    if (!ij_same) {
                        J_[nu * n + mu]  += g * D_ls;
                        K_[nu * n + lam] += g * D_ms;
                    }

                    // P4: (j,i,l,k) — present if i ≠ j AND k ≠ l
                    if (!ij_same && !kl_same) {
                        J_[nu * n + mu]  += g * D_sl;
                        K_[nu * n + sig] += g * D_ml;
                    }

                    // P5: (k,l,i,j) — present if bra ≠ ket
                    if (!braket_same) {
                        J_[lam * n + sig] += g * D_mn;
                        K_[lam * n + mu]  += g * D_ns; // D[sig,nu] = D[nu,sig]

                        // P7: (k,l,j,i) — present if bra ≠ ket AND i ≠ j
                        if (!ij_same) {
                            J_[lam * n + sig] += g * D_nm;
                            K_[lam * n + nu]  += g * D_ms; // D[sig,mu] = D[mu,sig]
                        }

                        // P6: (l,k,i,j) — present if bra ≠ ket AND k ≠ l
                        if (!kl_same) {
                            J_[sig * n + lam] += g * D_mn;
                            K_[sig * n + mu]  += g * D_nl; // D[lam,nu] = D[nu,lam]
                        }

                        // P8: (l,k,j,i) — present if bra ≠ ket AND i ≠ j AND k ≠ l
                        if (!ij_same && !kl_same) {
                            J_[sig * n + lam] += g * D_nm;
                            K_[sig * n + nu]  += g * D_ml; // D[lam,mu] = D[mu,lam]
                        }
                    }
                }
            }
        }
    }
}

std::vector<Real> FockBuilder::get_fock_matrix(
    std::span<const Real> H_core,
    Real exchange_fraction) const {
    if (H_core.size() != nbf_ * nbf_) {
        throw InvalidArgumentException(
            "FockBuilder::get_fock_matrix: H_core size mismatch");
    }

    std::vector<Real> F(nbf_ * nbf_);
    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        F[i] = H_core[i] + J_[i] - exchange_fraction * K_[i];
    }
    return F;
}

void FockBuilder::reset() noexcept {
    std::fill(J_.begin(), J_.end(), 0.0);
    std::fill(K_.begin(), K_.end(), 0.0);

    // Reset thread-local buffers if using ThreadLocal strategy
    if (strategy_ == FockThreadingStrategy::ThreadLocal || auto_thread_local_fallback_) {
        for (auto& J_local : J_thread_local_) {
            std::fill(J_local.begin(), J_local.end(), 0.0);
        }
        for (auto& K_local : K_thread_local_) {
            std::fill(K_local.begin(), K_local.end(), 0.0);
        }
    }
}

// =============================================================================
// Thread-Safety Implementation (Phase 4)
// =============================================================================

void FockBuilder::set_threading_strategy(FockThreadingStrategy strategy) {
    strategy_ = strategy;
    auto_thread_local_fallback_ = false;
    auto_atomic_fallback_ = false;
    if (strategy_ != FockThreadingStrategy::ThreadLocal) {
        J_thread_local_.clear();
        K_thread_local_.clear();
    }
}

int FockBuilder::get_thread_id() const noexcept {
#if defined(_OPENMP)
    return omp_get_thread_num();
#else
    return 0;
#endif
}

void FockBuilder::prepare_parallel(int n_threads) {
    if (n_threads <= 0) {
#if defined(_OPENMP)
        n_threads = omp_get_max_threads();
#else
        n_threads = 1;
#endif
    }

    n_threads_ = std::max(1, n_threads);
    auto_atomic_fallback_ = false;
    auto_thread_local_fallback_ =
        (strategy_ == FockThreadingStrategy::Sequential && n_threads_ > 1);
    bool use_thread_local =
        (strategy_ == FockThreadingStrategy::ThreadLocal) || auto_thread_local_fallback_;

    if (use_thread_local) {
        std::size_t mat_size = 0;
        if (!checked_mul(static_cast<std::size_t>(nbf_), static_cast<std::size_t>(nbf_), mat_size)) {
            throw MemoryException(
                "FockBuilder::prepare_parallel: nbf^2 overflow while sizing thread-local buffers");
        }

        std::size_t required_scalars = 0;
        if (!checked_mul(static_cast<std::size_t>(n_threads_), mat_size, required_scalars) ||
            !checked_mul(required_scalars, 2u, required_scalars)) {
            throw MemoryException(
                "FockBuilder::prepare_parallel: overflow while sizing thread-local buffers");
        }

        std::size_t required_bytes = 0;
        if (!checked_mul(required_scalars, sizeof(Real), required_bytes)) {
            throw MemoryException(
                "FockBuilder::prepare_parallel: overflow while sizing thread-local buffers in bytes");
        }

        const std::size_t max_bytes = max_thread_local_bytes();
        if (required_bytes > max_bytes) {
            if (strategy_ == FockThreadingStrategy::ThreadLocal) {
                throw MemoryException(
                    "FockBuilder::prepare_parallel: requested ThreadLocal strategy needs " +
                    std::to_string(required_bytes) + " bytes, exceeding limit " +
                    std::to_string(max_bytes) +
                    ". Reduce threads, reduce basis size, or raise LIBACCINT_MAX_FOCK_THREADLOCAL_BYTES.");
            }

            // Safe default: fall back to atomics instead of allocating massive TLS matrices.
            auto_thread_local_fallback_ = false;
            auto_atomic_fallback_ = (n_threads_ > 1);
            use_thread_local = false;
            J_thread_local_.clear();
            K_thread_local_.clear();
        } else {
            J_thread_local_.resize(static_cast<std::size_t>(n_threads_));
            K_thread_local_.resize(static_cast<std::size_t>(n_threads_));

            for (int t = 0; t < n_threads_; ++t) {
                J_thread_local_[t].assign(mat_size, 0.0);
                K_thread_local_[t].assign(mat_size, 0.0);
            }
        }
    }

    if (!use_thread_local) {
        J_thread_local_.clear();
        K_thread_local_.clear();
    }
}

void FockBuilder::finalize_parallel() {
    const bool use_thread_local =
        (strategy_ == FockThreadingStrategy::ThreadLocal) || auto_thread_local_fallback_;
    if (use_thread_local) {
        if (J_thread_local_.empty() || K_thread_local_.empty()) {
            auto_thread_local_fallback_ = false;
            return;
        }

        // Reduce all thread-local buffers into main J and K
        const Size mat_size = nbf_ * nbf_;
        const int n_local = static_cast<int>(J_thread_local_.size());
        const int n_reduce = std::min(n_threads_, n_local);
        for (int t = 0; t < n_reduce; ++t) {
            for (Size i = 0; i < mat_size; ++i) {
                J_[i] += J_thread_local_[t][i];
                K_[i] += K_thread_local_[t][i];
            }
        }

        // Clear thread-local buffers
        J_thread_local_.clear();
        K_thread_local_.clear();
    }
    auto_thread_local_fallback_ = false;
    auto_atomic_fallback_ = false;
}

void FockBuilder::accumulate_atomic_impl(const TwoElectronBuffer<0>& buffer,
                                          Index fa, Index fb, Index fc, Index fd,
                                          int na, int nb, int nc, int nd) {
    const Size n = nbf_;

    for (int a = 0; a < na; ++a) {
        const auto mu = static_cast<Size>(fa + a);
        for (int b = 0; b < nb; ++b) {
            const auto nu = static_cast<Size>(fb + b);
            for (int c = 0; c < nc; ++c) {
                const auto lam = static_cast<Size>(fc + c);
                for (int d = 0; d < nd; ++d) {
                    const auto sig = static_cast<Size>(fd + d);

                    const Real eri = buffer(a, b, c, d);
                    const Real D_lam_sig = D_[lam * n + sig];
                    const Real D_nu_sig = D_[nu * n + sig];

                    // Atomic accumulation for thread safety
#if defined(_OPENMP)
                    #pragma omp atomic
                    J_[mu * n + nu] += eri * D_lam_sig;

                    #pragma omp atomic
                    K_[mu * n + lam] += eri * D_nu_sig;
#else
                    J_[mu * n + nu] += eri * D_lam_sig;
                    K_[mu * n + lam] += eri * D_nu_sig;
#endif
                }
            }
        }
    }
}

void FockBuilder::accumulate_thread_local_impl(const TwoElectronBuffer<0>& buffer,
                                                Index fa, Index fb, Index fc, Index fd,
                                                int na, int nb, int nc, int nd) {
    const int tid = get_thread_id();
    if (tid < 0 || static_cast<std::size_t>(tid) >= J_thread_local_.size() ||
        static_cast<std::size_t>(tid) >= K_thread_local_.size()) {
        throw InvalidStateException(
            "FockBuilder::accumulate_thread_local_impl called without prepared "
            "thread-local buffers for thread id " + std::to_string(tid));
    }
    const Size n = nbf_;

    // Use thread-local buffers for accumulation (no contention)
    auto& J_local = J_thread_local_[tid];
    auto& K_local = K_thread_local_[tid];

    for (int a = 0; a < na; ++a) {
        const auto mu = static_cast<Size>(fa + a);
        for (int b = 0; b < nb; ++b) {
            const auto nu = static_cast<Size>(fb + b);
            for (int c = 0; c < nc; ++c) {
                const auto lam = static_cast<Size>(fc + c);
                for (int d = 0; d < nd; ++d) {
                    const auto sig = static_cast<Size>(fd + d);

                    const Real eri = buffer(a, b, c, d);

                    // Thread-local accumulation (no synchronization needed)
                    J_local[mu * n + nu] += eri * D_[lam * n + sig];
                    K_local[mu * n + lam] += eri * D_[nu * n + sig];
                }
            }
        }
    }
}

}  // namespace libaccint::consumers
