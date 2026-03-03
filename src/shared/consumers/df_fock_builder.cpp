// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file df_fock_builder.cpp
/// @brief Implementation of density-fitted Fock matrix builder

#include <libaccint/consumers/df_fock_builder.hpp>
#include <libaccint/kernels/two_center_coulomb_kernel.hpp>
#include <libaccint/kernels/three_center_eri_kernel.hpp>
#include <libaccint/kernels/overlap_kernel.hpp>
#include <libaccint/math/cholesky.hpp>
#include <libaccint/math/blas_wrappers.hpp>
#include <libaccint/utils/error_handling.hpp>
#include <libaccint/utils/logging.hpp>
#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/data/auxiliary_basis_selector.hpp>

#include <algorithm>
#include <cmath>
#include <map>

namespace libaccint::consumers {

// =============================================================================
// Construction / Destruction
// =============================================================================

DFFockBuilder::DFFockBuilder(const BasisSet& orbital,
                              const AuxiliaryBasisSet& auxiliary,
                              DFFockBuilderConfig config)
    : orbital_(&orbital),
      auxiliary_(&auxiliary),
      config_(config),
      n_orb_(orbital.n_basis_functions()),
      n_aux_(auxiliary.n_functions()) {

    // Allocate storage
    D_.resize(n_orb_ * n_orb_, 0.0);
    J_.resize(n_orb_ * n_orb_, 0.0);
    K_.resize(n_orb_ * n_orb_, 0.0);
}

DFFockBuilder::DFFockBuilder(const BasisSet& orbital,
                              std::unique_ptr<AuxiliaryBasisSet> auxiliary,
                              DFFockBuilderConfig config)
    : orbital_(&orbital),
      auxiliary_(auxiliary.get()),
      owned_auxiliary_(std::move(auxiliary)),
      config_(config),
      n_orb_(orbital.n_basis_functions()),
      n_aux_(owned_auxiliary_ ? owned_auxiliary_->n_functions() : 0) {

    // Allocate storage
    D_.resize(n_orb_ * n_orb_, 0.0);
    J_.resize(n_orb_ * n_orb_, 0.0);
    K_.resize(n_orb_ * n_orb_, 0.0);
}

DFFockBuilder::~DFFockBuilder() = default;

DFFockBuilder::DFFockBuilder(DFFockBuilder&&) noexcept = default;
DFFockBuilder& DFFockBuilder::operator=(DFFockBuilder&&) noexcept = default;

// =============================================================================
// Setup
// =============================================================================

void DFFockBuilder::set_density(std::span<const Real> D) {
    if (D.size() != n_orb_ * n_orb_) {
        throw InvalidArgumentException(
            "Density matrix size mismatch: expected " +
            std::to_string(n_orb_ * n_orb_) + ", got " + std::to_string(D.size()));
    }
    uhf_mode_ = false;
    std::copy(D.begin(), D.end(), D_.begin());
}

void DFFockBuilder::set_density_unrestricted(std::span<const Real> D_alpha,
                                               std::span<const Real> D_beta) {
    if (D_alpha.size() != n_orb_ * n_orb_ || D_beta.size() != n_orb_ * n_orb_) {
        throw InvalidArgumentException("Density matrix size mismatch");
    }

    uhf_mode_ = true;

    // Store separate spin densities for exchange
    D_alpha_.resize(n_orb_ * n_orb_);
    D_beta_.resize(n_orb_ * n_orb_);
    std::copy(D_alpha.begin(), D_alpha.end(), D_alpha_.begin());
    std::copy(D_beta.begin(), D_beta.end(), D_beta_.begin());

    // Total density D = D_alpha + D_beta for Coulomb
    for (Size i = 0; i < n_orb_ * n_orb_; ++i) {
        D_[i] = D_alpha[i] + D_beta[i];
    }

    // Allocate spin-separated K matrices
    K_alpha_.resize(n_orb_ * n_orb_, 0.0);
    K_beta_.resize(n_orb_ * n_orb_, 0.0);
}

void DFFockBuilder::initialize() {
    if (initialized_) {
        return;
    }

    // Allocate metric storage
    metric_.resize(n_aux_ * n_aux_);
    L_inv_.resize(n_aux_ * n_aux_);

    // Compute two-center metric
    compute_metric();

    // Compute B tensor
    compute_b_tensor();

    // Check if B tensor exceeds memory limit and auto-switch to block storage
    const Size b_tensor_bytes = n_orb_ * n_orb_ * n_aux_ * sizeof(Real);
    const Size memory_limit_bytes = config_.memory_limit_mb * 1024ULL * 1024ULL;
    if (b_tensor_bytes > memory_limit_bytes && !block_storage_) {
        // Create owned block storage and partition the B tensor
        df::BlockStorageConfig block_config;
        block_config.memory_limit_mb = config_.memory_limit_mb;
        owned_block_storage_ = std::make_unique<df::ThreeCenterBlockStorage>(
            n_orb_, n_aux_, block_config);
        owned_block_storage_->store_full_tensor(B_tensor_);
        block_storage_ = owned_block_storage_.get();

        // Release the monolithic B tensor — block storage now owns the data
        B_tensor_.clear();
        B_tensor_.shrink_to_fit();
    }

    initialized_ = true;
}

void DFFockBuilder::set_block_storage(df::ThreeCenterBlockStorage* storage) {
    block_storage_ = storage;
}

// =============================================================================
// Metric Computation
// =============================================================================

void DFFockBuilder::compute_metric() {
    // Compute (P|Q) metric integrals based on selected metric type
    switch (config_.metric_type) {
    case DFMetricType::Coulomb:
        kernels::compute_two_center_metric(auxiliary_->shells(),
                                            metric_.data(),
                                            n_aux_);
        break;

    case DFMetricType::Overlap: {
        // Compute overlap metric: S_PQ = <P|Q>
        // Iterate over auxiliary shell pairs and fill the metric
        std::fill(metric_.begin(), metric_.end(), 0.0);
        const auto& aux_shells = auxiliary_->shells();
        Size row_offset = 0;
        for (Size s_i = 0; s_i < aux_shells.size(); ++s_i) {
            Size n_i = static_cast<Size>(aux_shells[s_i].n_functions());
            Size col_offset = 0;
            for (Size s_j = 0; s_j < aux_shells.size(); ++s_j) {
                Size n_j = static_cast<Size>(aux_shells[s_j].n_functions());

                OverlapBuffer buffer;
                kernels::compute_overlap(aux_shells[s_i], aux_shells[s_j], buffer);

                // Fill metric from buffer
                for (Size i = 0; i < n_i; ++i) {
                    for (Size j = 0; j < n_j; ++j) {
                        metric_[(row_offset + i) * n_aux_ + (col_offset + j)] =
                            buffer(i, j);
                    }
                }
                col_offset += n_j;
            }
            row_offset += n_i;
        }
        break;
    }

    case DFMetricType::AttenuatedCoulomb:
        throw InvalidArgumentException(
            "AttenuatedCoulomb metric is not yet supported. "
            "Use Coulomb or Overlap metric instead.");
    }

    // Estimate condition number from diagonal elements (O(n) estimate)
    Real diag_max = 0.0;
    Real diag_min = std::numeric_limits<Real>::max();
    for (Size i = 0; i < n_aux_; ++i) {
        Real d = metric_[i * n_aux_ + i];
        if (d > diag_max) diag_max = d;
        if (d > 0.0 && d < diag_min) diag_min = d;
    }

    Real condition_estimate = (diag_min > 0.0) ? (diag_max / diag_min) : std::numeric_limits<Real>::infinity();

    LIBACCINT_LOG_INFO("DF",
        "Metric matrix condition estimate (diag ratio): " + std::to_string(condition_estimate) +
        " (max=" + std::to_string(diag_max) + ", min=" + std::to_string(diag_min) + ")");

    if (config_.conditioning_hard_limit > 0.0 && condition_estimate > config_.conditioning_hard_limit) {
        throw InvalidArgumentException(
            "DF metric matrix is severely ill-conditioned (condition estimate = " +
            std::to_string(condition_estimate) + ", hard limit = " +
            std::to_string(config_.conditioning_hard_limit) +
            "). The auxiliary basis may be near-linearly-dependent. "
            "Consider using a different auxiliary basis or increasing conditioning_hard_limit.");
    }

    if (condition_estimate > config_.conditioning_threshold) {
        LIBACCINT_LOG_WARNING("DF",
            "DF metric matrix may be ill-conditioned (condition estimate = " +
            std::to_string(condition_estimate) + ", threshold = " +
            std::to_string(config_.conditioning_threshold) +
            "). Results may have reduced numerical accuracy.");
    }

    // Cholesky decomposition: (P|Q) = L * L^T
    std::vector<Real> L(n_aux_ * n_aux_);
    std::copy(metric_.begin(), metric_.end(), L.begin());

    // In-place Cholesky (lower triangular)
    math::cholesky_decompose(L.data(), n_aux_);

    // Compute L^{-1} via forward substitution
    // For now, use a simple inversion approach
    math::triangular_inverse(L.data(), L_inv_.data(), n_aux_, true);  // lower=true
}

// =============================================================================
// B Tensor Computation
// =============================================================================

void DFFockBuilder::compute_b_tensor() {
    // Allocate B tensor storage
    B_tensor_.resize(n_orb_ * n_orb_ * n_aux_);

    // Compute three-center integrals (ab|P)
    std::vector<Real> three_center(n_orb_ * n_orb_ * n_aux_);
    kernels::compute_three_center_tensor(
        orbital_->shells(),
        auxiliary_->shells(),
        three_center.data(),
        n_orb_,
        n_aux_,
        kernels::ThreeCenterStorageFormat::abP);

    // Form B tensor: B_ab^P = sum_Q (ab|Q) * L^{-1}_{QP}
    kernels::compute_B_tensor(three_center.data(),
                               L_inv_.data(),
                               B_tensor_.data(),
                               n_orb_,
                               n_aux_);
}

// =============================================================================
// Fock Matrix Computation
// =============================================================================

std::vector<Real> DFFockBuilder::compute() {
    if (!initialized_) {
        initialize();
    }

    // Reset J and K
    std::fill(J_.begin(), J_.end(), 0.0);
    std::fill(K_.begin(), K_.end(), 0.0);

    if (config_.compute_coulomb) {
        compute_j_internal();
    }

    if (config_.compute_exchange) {
        compute_k_internal();
    }

    // Form F = J - exchange_fraction * K
    std::vector<Real> F(n_orb_ * n_orb_);
    for (Size i = 0; i < n_orb_ * n_orb_; ++i) {
        F[i] = J_[i] - config_.exchange_fraction * K_[i];
    }

    return F;
}

std::vector<Real> DFFockBuilder::compute_coulomb() {
    if (!initialized_) {
        initialize();
    }

    std::fill(J_.begin(), J_.end(), 0.0);
    compute_j_internal();

    return std::vector<Real>(J_.begin(), J_.end());
}

std::vector<Real> DFFockBuilder::compute_exchange() {
    if (!initialized_) {
        initialize();
    }

    std::fill(K_.begin(), K_.end(), 0.0);
    compute_k_internal();

    return std::vector<Real>(K_.begin(), K_.end());
}

void DFFockBuilder::compute_accumulate(std::span<Real> F) {
    if (F.size() != n_orb_ * n_orb_) {
        throw InvalidArgumentException("Fock matrix size mismatch");
    }

    auto result = compute();
    for (Size i = 0; i < n_orb_ * n_orb_; ++i) {
        F[i] += result[i];
    }
}

// =============================================================================
// J and K Internal Implementation
// =============================================================================

void DFFockBuilder::compute_j_internal() {
    // DF-J algorithm:
    // 1. gamma_P = sum_ab D_ab * B_ab^P   →  gamma = B^T * vec(D)  (GEMV)
    // 2. J_ab = sum_P B_ab^P * gamma_P    →  vec(J) = B * gamma    (GEMV)

    const Size n_pair = n_orb_ * n_orb_;
    std::vector<Real> gamma(n_aux_, 0.0);

    if (block_storage_) {
        // Block-by-block processing
        for (Size blk = 0; blk < block_storage_->n_blocks(); ++blk) {
            auto [P_start, P_end] = block_storage_->block_range(blk);
            Size block_size = P_end - P_start;
            auto block_data = block_storage_->get_block(blk);

            // Block data layout: block_data[(a*n_orb+b)*block_size + (P-P_start)]
            // gamma[P_start:P_end] += B_block^T * vec(D)
            math::gemv(/*transpose=*/true, n_pair, block_size,
                        1.0, block_data.data(), D_.data(),
                        0.0, gamma.data() + P_start);
        }

        // Step 2: J_ab = sum_P B_ab^P * gamma_P (block-by-block)
        std::fill(J_.begin(), J_.end(), 0.0);
        for (Size blk = 0; blk < block_storage_->n_blocks(); ++blk) {
            auto [P_start, P_end] = block_storage_->block_range(blk);
            Size block_size = P_end - P_start;
            auto block_data = block_storage_->get_block(blk);

            // J += B_block * gamma[P_start:P_end]
            math::gemv(/*transpose=*/false, n_pair, block_size,
                        1.0, block_data.data(), gamma.data() + P_start,
                        1.0, J_.data());
        }
    } else {
        // Flat-vector path: B_tensor_ is (n_pair x n_aux_), row-major
        // Step 1: gamma = B^T * vec(D)
        math::gemv(/*transpose=*/true, n_pair, n_aux_,
                    1.0, B_tensor_.data(), D_.data(),
                    0.0, gamma.data());

        // Step 2: vec(J) = B * gamma
        math::gemv(/*transpose=*/false, n_pair, n_aux_,
                    1.0, B_tensor_.data(), gamma.data(),
                    0.0, J_.data());
    }
}

void DFFockBuilder::compute_k_internal() {
    // DF-K algorithm:
    // K_ac = sum_P (sum_b B_ab^P * D_bd) * B_cd^P
    //      = sum_P X_ad * B_cd^P
    //
    // For each auxiliary index P:
    //   1. Extract B_P[a][b]   (strided from flat or from block)
    //   2. X = B_P * D         →  GEMM  (n_orb x n_orb)
    //   3. K += X * B_P^T      →  GEMM_BT with beta=1  (n_orb x n_orb)
    //
    // For UHF, compute K_alpha and K_beta separately using respective
    // spin densities, then sum for the total K.

    const Size n2 = n_orb_ * n_orb_;

    auto compute_k_for_density = [&](const std::vector<Real>& D_spin,
                                      std::vector<Real>& K_out) {
        std::fill(K_out.begin(), K_out.end(), 0.0);

        std::vector<Real> B_P(n2);   // contiguous B slice for one aux index
        std::vector<Real> X(n2);     // intermediate: X = B_P * D

        if (block_storage_) {
            // Block-by-block processing
            for (Size blk = 0; blk < block_storage_->n_blocks(); ++blk) {
                auto [P_start, P_end] = block_storage_->block_range(blk);
                Size block_size = P_end - P_start;
                auto block_data = block_storage_->get_block(blk);

                // Process each aux index P within this block
                for (Size p_local = 0; p_local < block_size; ++p_local) {
                    // Extract B_P from block: B_P[ab] = block_data[ab * block_size + p_local]
                    for (Size ab = 0; ab < n2; ++ab) {
                        B_P[ab] = block_data[ab * block_size + p_local];
                    }

                    math::gemm(n_orb_, n_orb_, n_orb_,
                                1.0, B_P.data(), D_spin.data(),
                                0.0, X.data());

                    math::gemm_bt(n_orb_, n_orb_, n_orb_,
                                   1.0, X.data(), B_P.data(),
                                   1.0, K_out.data());
                }
            }
        } else {
            // Flat-vector path
            for (Size P = 0; P < n_aux_; ++P) {
                // Extract B_P: B_P[ab] = B_tensor_[ab * n_aux_ + P]
                for (Size ab = 0; ab < n2; ++ab) {
                    B_P[ab] = B_tensor_[ab * n_aux_ + P];
                }

                math::gemm(n_orb_, n_orb_, n_orb_,
                            1.0, B_P.data(), D_spin.data(),
                            0.0, X.data());

                math::gemm_bt(n_orb_, n_orb_, n_orb_,
                               1.0, X.data(), B_P.data(),
                               1.0, K_out.data());
            }
        }
    };

    if (uhf_mode_) {
        // UHF: compute exchange separately for each spin channel
        compute_k_for_density(D_alpha_, K_alpha_);
        compute_k_for_density(D_beta_, K_beta_);

        // Total K = K_alpha + K_beta
        for (Size i = 0; i < n_orb_ * n_orb_; ++i) {
            K_[i] = K_alpha_[i] + K_beta_[i];
        }
    } else {
        // RHF: compute exchange from total density
        compute_k_for_density(D_, K_);
    }
}

// =============================================================================
// Results Access
// =============================================================================

std::vector<Real> DFFockBuilder::fock_matrix(std::span<const Real> H_core) const {
    std::vector<Real> F(n_orb_ * n_orb_);

    for (Size i = 0; i < n_orb_ * n_orb_; ++i) {
        F[i] = J_[i] - config_.exchange_fraction * K_[i];
        if (!H_core.empty()) {
            F[i] += H_core[i];
        }
    }

    return F;
}

// =============================================================================
// Factory Functions
// =============================================================================

std::unique_ptr<DFFockBuilder> make_df_fock_builder(
    const BasisSet& orbital,
    const std::string& aux_basis_name) {

    // Without atomic numbers, we cannot load an auxiliary basis from built-in data.
    // The user should use the overload that takes atomic_numbers and centers.
    // As a fallback, try to construct an auxiliary basis from the shells' geometry
    // using a default auxiliary basis name.
    if (aux_basis_name.empty()) {
        throw InvalidArgumentException(
            "make_df_fock_builder: cannot auto-select auxiliary basis without atom data. "
            "Use the overload that accepts atomic_numbers and centers, or provide "
            "an explicit auxiliary basis name.");
    }

    // Extract unique atom centers from the orbital basis shells
    std::map<Index, Point3D> atom_centers;
    for (const auto& shell : orbital.shells()) {
        atom_centers[shell.atom_index()] = shell.center();
    }

    // Without atomic numbers, we cannot load from built-in data.
    // Fall back to constructing a minimal s-type aux basis at each center.
    std::vector<Shell> aux_shells;
    int shell_idx = 0;
    for (const auto& [atom_idx, center] : atom_centers) {
        // Create a universal auxiliary s-function with spread of exponents
        Shell aux(0, center, {10.0, 2.0, 0.5, 0.1}, {0.25, 0.25, 0.25, 0.25});
        aux.set_atom_index(static_cast<Index>(atom_idx));
        aux_shells.push_back(std::move(aux));
        ++shell_idx;
    }

    auto aux_basis = std::make_unique<AuxiliaryBasisSet>(
        std::move(aux_shells), FittingType::JKFIT, aux_basis_name);

    return std::make_unique<DFFockBuilder>(orbital, std::move(aux_basis));
}

std::unique_ptr<DFFockBuilder> make_df_fock_builder(
    const BasisSet& orbital,
    std::span<const int> atomic_numbers,
    std::span<const std::array<Real, 3>> centers,
    const std::string& aux_basis_name,
    DFFockBuilderConfig config) {

    // Determine auxiliary basis name
    std::string resolved_name = aux_basis_name;
    if (resolved_name.empty()) {
        // Default to cc-pVDZ-RI as a general-purpose auxiliary basis
        resolved_name = "cc-pVDZ-RI";
    }

    // Load the auxiliary basis from built-in data
    auto auxiliary = std::make_unique<AuxiliaryBasisSet>(
        load_auxiliary_basis(resolved_name, atomic_numbers, centers));

    return std::make_unique<DFFockBuilder>(orbital, std::move(auxiliary), config);
}

}  // namespace libaccint::consumers
