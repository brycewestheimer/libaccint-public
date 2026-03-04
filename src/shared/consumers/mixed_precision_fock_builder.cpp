// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file mixed_precision_fock_builder.cpp
/// @brief Implementation of the mixed-precision Fock matrix builder
///
/// Accumulates float32 integrals into float64 J and K matrices for
/// improved numerical accuracy with reduced computation cost.

#include <libaccint/consumers/mixed_precision_fock_builder.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <algorithm>
#include <cstring>

namespace libaccint::consumers {

MixedPrecisionFockBuilder::MixedPrecisionFockBuilder(Size nbf)
    : nbf_(nbf), J_(nbf * nbf, 0.0), K_(nbf * nbf, 0.0) {}

void MixedPrecisionFockBuilder::set_density(const Real* D, Size nbf) {
    LIBACCINT_ASSERT(nbf == nbf_, "Density matrix size mismatch");
    D_ = D;
}

void MixedPrecisionFockBuilder::accumulate(
    const TwoElectronBuffer<0, float>& buffer,
    Index fa, Index fb, Index fc, Index fd,
    int na, int nb, int nc, int nd) {
    accumulate_impl(buffer, fa, fb, fc, fd, na, nb, nc, nd);
    ++n_float32_accumulations_;
}

void MixedPrecisionFockBuilder::accumulate(
    const TwoElectronBuffer<0, double>& buffer,
    Index fa, Index fb, Index fc, Index fd,
    int na, int nb, int nc, int nd) {
    accumulate_impl(buffer, fa, fb, fc, fd, na, nb, nc, nd);
    ++n_float64_accumulations_;
}

template<typename RealType>
void MixedPrecisionFockBuilder::accumulate_impl(
    const TwoElectronBuffer<0, RealType>& buffer,
    Index fa, Index fb, Index fc, Index fd,
    int na, int nb, int nc, int nd) {

    LIBACCINT_ASSERT(D_ != nullptr, "Density matrix not set");

    const auto n = static_cast<Index>(nbf_);

    for (int a = 0; a < na; ++a) {
        Index mu = fa + a;
        for (int b = 0; b < nb; ++b) {
            Index nu = fb + b;
            for (int c = 0; c < nc; ++c) {
                Index lam = fc + c;
                for (int d = 0; d < nd; ++d) {
                    Index sig = fd + d;

                    // Promote to double for accumulation
                    double integral = static_cast<double>(buffer(a, b, c, d));

                    // Coulomb: J(mu,nu) += (mu nu | lam sig) * D(lam, sig)
                    J_[static_cast<Size>(mu * n + nu)] +=
                        integral * D_[static_cast<Size>(lam * n + sig)];

                    // Exchange: K(mu,lam) += (mu nu | lam sig) * D(nu, sig)
                    K_[static_cast<Size>(mu * n + lam)] +=
                        integral * D_[static_cast<Size>(nu * n + sig)];
                }
            }
        }
    }
}

std::vector<Real> MixedPrecisionFockBuilder::get_fock_matrix(
    std::span<const Real> H_core,
    Real exchange_fraction) const {

    LIBACCINT_ASSERT(H_core.size() == nbf_ * nbf_,
                     "H_core size mismatch");

    std::vector<Real> F(nbf_ * nbf_);
    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        F[i] = H_core[i] + J_[i] - exchange_fraction * K_[i];
    }
    return F;
}

void MixedPrecisionFockBuilder::reset() noexcept {
    std::fill(J_.begin(), J_.end(), 0.0);
    std::fill(K_.begin(), K_.end(), 0.0);
    n_float32_accumulations_ = 0;
    n_float64_accumulations_ = 0;
}

// Explicit template instantiations
template void MixedPrecisionFockBuilder::accumulate_impl<float>(
    const TwoElectronBuffer<0, float>&, Index, Index, Index, Index, int, int, int, int);
template void MixedPrecisionFockBuilder::accumulate_impl<double>(
    const TwoElectronBuffer<0, double>&, Index, Index, Index, Index, int, int, int, int);

}  // namespace libaccint::consumers
