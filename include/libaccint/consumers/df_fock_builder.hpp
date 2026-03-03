// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file df_fock_builder.hpp
/// @brief Density-fitted Fock matrix builder
///
/// Implements DF-J and DF-K construction using three-center integrals:
///   J_ab = sum_P B_ab^P * gamma_P     where gamma_P = sum_cd D_cd * B_cd^P
///   K_ac = sum_P (sum_b B_ab^P * D_bd) * B_cd^P
///
/// The B tensor is: B_ab^P = sum_Q (ab|Q) * (P|Q)^{-1/2}

#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/df/b_tensor_storage.hpp>
#include <libaccint/df/three_center_storage.hpp>

#include <memory>
#include <span>
#include <vector>

namespace libaccint::consumers {

// =============================================================================
// DFMetricType
// =============================================================================

/// @brief Type of metric matrix used for density fitting
///
/// The metric determines the fitting criterion:
///   - Coulomb: (P|Q) = ∫∫ φ_P(r₁) 1/r₁₂ φ_Q(r₂) dr₁ dr₂  (standard DF)
///   - Overlap: (P|Q)_S = ∫ φ_P(r) φ_Q(r) dr  (simpler, cheaper)
///   - AttenuatedCoulomb: erfc-attenuated Coulomb (for range separation)
enum class DFMetricType {
    Coulomb,           ///< Standard Coulomb metric (default)
    Overlap,           ///< Overlap metric
    AttenuatedCoulomb  ///< Attenuated Coulomb metric (not yet supported)
};

// =============================================================================
// DFFockBuilderConfig
// =============================================================================

/// @brief Configuration options for DFFockBuilder
struct DFFockBuilderConfig {
    Real exchange_fraction{1.0};      ///< Fraction of exact exchange (1.0 for HF)
    bool compute_coulomb{true};       ///< Compute Coulomb (J) matrix
    bool compute_exchange{true};      ///< Compute exchange (K) matrix
    bool use_symmetry{true};          ///< Exploit a <-> b symmetry in orbital pairs
    bool incremental{false};          ///< Incremental Fock build mode
    Size memory_limit_mb{4096};       ///< Memory limit for B tensor storage
    int n_threads{0};                 ///< Thread count (0 = auto)

    /// @brief Condition number warning threshold for (P|Q) metric
    ///
    /// If the ratio of max to min diagonal element exceeds this value,
    /// a warning is emitted during initialize(). Default: 1e8.
    Real conditioning_threshold{1e8};

    /// @brief Hard condition number limit for (P|Q) metric
    ///
    /// If the ratio of max to min diagonal element exceeds this value,
    /// initialization throws an error. Default: 1e14. Set to 0 to disable.
    Real conditioning_hard_limit{1e14};

    /// @brief Type of metric matrix for density fitting
    ///
    /// Default is Coulomb. Overlap is simpler and cheaper to compute.
    /// AttenuatedCoulomb is reserved for range-separated DF (not yet supported).
    DFMetricType metric_type{DFMetricType::Coulomb};

    /// @brief Range-separation parameter for attenuated Coulomb metric
    ///
    /// Only used when metric_type is AttenuatedCoulomb. Controls the
    /// range separation: erfc(ω r₁₂) / r₁₂.
    Real range_separation_omega{0.0};
};

// =============================================================================
// DFFockBuilder
// =============================================================================

/// @brief Builds Coulomb and exchange matrices using density fitting
///
/// DFFockBuilder computes the Fock matrix contributions using the DF/RI
/// approximation. The algorithm:
///
/// 1. Compute three-center integrals (ab|P)
/// 2. Compute two-center metric (P|Q) and Cholesky decompose
/// 3. Form B tensor: B_ab^P = sum_Q (ab|Q) * L^{-1}_{QP}
/// 4. For Coulomb: gamma_P = sum_ab D_ab * B_ab^P, then J_ab = sum_P B_ab^P * gamma_P
/// 5. For exchange: K_ac = sum_P sum_b B_ab^P * D_bd * B_cd^P
///
/// Scaling: O(N^2 * N_aux) for J, O(N^2 * N_occ * N_aux) for K
/// Memory: O(N^2 * N_aux) for full B tensor storage
class DFFockBuilder {
public:
    /// @brief Construct a DFFockBuilder (non-owning)
    ///
    /// The caller must ensure the orbital and auxiliary basis sets outlive
    /// the DFFockBuilder.
    ///
    /// @param orbital Orbital basis set
    /// @param auxiliary Auxiliary basis set
    /// @param config Configuration options
    DFFockBuilder(const BasisSet& orbital,
                  const AuxiliaryBasisSet& auxiliary,
                  DFFockBuilderConfig config = {});

    /// @brief Construct a DFFockBuilder that takes ownership of the auxiliary basis
    ///
    /// This overload is used by factory functions to transfer auxiliary basis
    /// ownership into the builder, preventing dangling pointers.
    ///
    /// @param orbital Orbital basis set (must outlive the builder)
    /// @param auxiliary Auxiliary basis set (ownership transferred)
    /// @param config Configuration options
    DFFockBuilder(const BasisSet& orbital,
                  std::unique_ptr<AuxiliaryBasisSet> auxiliary,
                  DFFockBuilderConfig config = {});

    /// @brief Destructor
    ~DFFockBuilder();

    // Disable copy (owns resources)
    DFFockBuilder(const DFFockBuilder&) = delete;
    DFFockBuilder& operator=(const DFFockBuilder&) = delete;

    // Enable move
    DFFockBuilder(DFFockBuilder&&) noexcept;
    DFFockBuilder& operator=(DFFockBuilder&&) noexcept;

    // =========================================================================
    // Setup
    // =========================================================================

    /// @brief Set the density matrix for Fock construction
    ///
    /// @param D Density matrix (n_orb x n_orb, row-major)
    void set_density(std::span<const Real> D);

    /// @brief Set separate alpha and beta density matrices (UHF)
    ///
    /// @param D_alpha Alpha density matrix
    /// @param D_beta Beta density matrix
    void set_density_unrestricted(std::span<const Real> D_alpha,
                                   std::span<const Real> D_beta);

    /// @brief Initialize the DF infrastructure (compute metric, B tensor)
    ///
    /// Call this once before repeated Fock builds. The metric and B tensor
    /// are cached for reuse.
    void initialize();

    /// @brief Check if already initialized
    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }

    /// @brief Set block storage backend for memory-bounded B tensor access
    ///
    /// When set, J/K contractions use block-by-block processing instead of
    /// the monolithic flat vector. This enables DF computation for systems
    /// where the full B tensor exceeds memory_limit_mb.
    ///
    /// The storage must outlive the DFFockBuilder and contain valid B tensor
    /// data before calling compute().
    ///
    /// @param storage Pointer to block storage, or nullptr to revert to flat
    ///                 storage mode. Passing nullptr is safe and disables
    ///                 block-by-block processing.
    void set_block_storage(df::ThreeCenterBlockStorage* storage);

    // =========================================================================
    // Fock Matrix Construction
    // =========================================================================

    /// @brief Compute the DF-Fock matrix
    ///
    /// Computes J and/or K based on configuration and combines them
    /// as F = J - exchange_fraction * K.
    ///
    /// @return Vector containing Fock matrix (n_orb x n_orb, row-major)
    [[nodiscard]] std::vector<Real> compute();

    /// @brief Compute the Coulomb (J) matrix only
    ///
    /// @return Vector containing J matrix (n_orb x n_orb, row-major)
    [[nodiscard]] std::vector<Real> compute_coulomb();

    /// @brief Compute the exchange (K) matrix only
    ///
    /// @return Vector containing K matrix (n_orb x n_orb, row-major)
    [[nodiscard]] std::vector<Real> compute_exchange();

    /// @brief Compute and accumulate Fock matrix into existing buffer
    ///
    /// @param[in,out] F Fock matrix buffer to accumulate into
    void compute_accumulate(std::span<Real> F);

    // =========================================================================
    // Access Results
    // =========================================================================

    /// @brief Get the last computed Coulomb matrix
    [[nodiscard]] std::span<const Real> coulomb_matrix() const noexcept {
        return J_;
    }

    /// @brief Get the last computed exchange matrix
    [[nodiscard]] std::span<const Real> exchange_matrix() const noexcept {
        return K_;
    }

    /// @brief Get the alpha exchange matrix (UHF mode only)
    [[nodiscard]] std::span<const Real> exchange_matrix_alpha() const noexcept {
        return K_alpha_;
    }

    /// @brief Get the beta exchange matrix (UHF mode only)
    [[nodiscard]] std::span<const Real> exchange_matrix_beta() const noexcept {
        return K_beta_;
    }

    /// @brief Check if currently in UHF mode
    [[nodiscard]] bool is_uhf() const noexcept { return uhf_mode_; }

    /// @brief Get the full Fock matrix (J - exchange_fraction * K)
    ///
    /// @param H_core Optional core Hamiltonian to include
    /// @return Fock matrix F = H_core + J - fraction * K
    [[nodiscard]] std::vector<Real> fock_matrix(
        std::span<const Real> H_core = {}) const;

    // =========================================================================
    // Diagnostics
    // =========================================================================

    /// @brief Get the number of orbital basis functions
    [[nodiscard]] Size n_orb() const noexcept { return n_orb_; }

    /// @brief Get the number of auxiliary basis functions
    [[nodiscard]] Size n_aux() const noexcept { return n_aux_; }

    /// @brief Get memory usage for B tensor (bytes)
    [[nodiscard]] Size b_tensor_memory() const noexcept {
        return n_orb_ * n_orb_ * n_aux_ * sizeof(Real);
    }

    /// @brief Get configuration
    [[nodiscard]] const DFFockBuilderConfig& config() const noexcept { return config_; }

private:
    const BasisSet* orbital_;
    const AuxiliaryBasisSet* auxiliary_;
    std::unique_ptr<AuxiliaryBasisSet> owned_auxiliary_;  ///< Owns aux basis when created by factory
    DFFockBuilderConfig config_;
    df::ThreeCenterBlockStorage* block_storage_{nullptr}; ///< Optional block storage backend
    std::unique_ptr<df::ThreeCenterBlockStorage> owned_block_storage_; ///< Auto-created block storage

    Size n_orb_{0};
    Size n_aux_{0};
    bool initialized_{false};

    // Cached data
    std::vector<Real> metric_;          ///< Two-center (P|Q) matrix
    std::vector<Real> L_inv_;           ///< Inverse of Cholesky factor
    std::vector<Real> B_tensor_;        ///< B_ab^P tensor (n_orb^2 x n_aux)

    // Current density and results
    std::vector<Real> D_;               ///< Current density matrix (total for RHF, total for J in UHF)
    std::vector<Real> D_alpha_;         ///< Alpha density matrix (UHF mode)
    std::vector<Real> D_beta_;          ///< Beta density matrix (UHF mode)
    bool uhf_mode_{false};              ///< Whether UHF densities are set
    std::vector<Real> J_;               ///< Coulomb matrix
    std::vector<Real> K_;               ///< Exchange matrix (RHF total K, UHF: K_alpha + K_beta for total)
    std::vector<Real> K_alpha_;         ///< Alpha exchange matrix (UHF mode)
    std::vector<Real> K_beta_;          ///< Beta exchange matrix (UHF mode)

    /// @brief Compute two-center metric and its inverse
    void compute_metric();

    /// @brief Compute B tensor from three-center integrals
    void compute_b_tensor();

    /// @brief Internal J matrix computation
    void compute_j_internal();

    /// @brief Internal K matrix computation
    void compute_k_internal();
};

// =============================================================================
// Factory Functions
// =============================================================================

/// @brief Create a DFFockBuilder with automatic auxiliary basis selection
///
/// Selects an appropriate auxiliary basis for the given orbital basis
/// (e.g., def2-JKFIT for def2 bases, cc-pVXZ-JKFIT for cc-pVXZ).
///
/// @note This overload requires oracle-level information about the orbital
///       basis name that is not stored in the BasisSet itself. If the basis
///       name cannot be inferred, use the overload that accepts atom data.
///
/// @param orbital Orbital basis set
/// @param aux_basis_name Auxiliary basis name (empty = auto-select)
/// @return Configured DFFockBuilder
[[nodiscard]] std::unique_ptr<DFFockBuilder> make_df_fock_builder(
    const BasisSet& orbital,
    const std::string& aux_basis_name = "");

/// @brief Create a DFFockBuilder with explicit atom data for auxiliary basis loading
///
/// This is the recommended factory when atomic numbers and centers are available.
/// It loads the specified (or auto-selected) auxiliary basis from built-in data.
///
/// @param orbital Orbital basis set
/// @param atomic_numbers Atomic numbers for each atom
/// @param centers Atom centers (in Bohr)
/// @param aux_basis_name Auxiliary basis name (empty = auto-select from "cc-pVDZ-RI")
/// @param config Optional configuration
/// @return Configured DFFockBuilder
[[nodiscard]] std::unique_ptr<DFFockBuilder> make_df_fock_builder(
    const BasisSet& orbital,
    std::span<const int> atomic_numbers,
    std::span<const std::array<Real, 3>> centers,
    const std::string& aux_basis_name = "",
    DFFockBuilderConfig config = {});

}  // namespace libaccint::consumers
