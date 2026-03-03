// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file integral_buffer.hpp
/// @brief IntegralBuffer for non-consuming batch compute API
///
/// IntegralBuffer holds computed integral values along with per-shell-pair
/// or per-shell-quartet metadata needed to interpret the buffer contents.
/// Used by Engine::compute_batch() to return integrals directly to the caller.

#include <libaccint/core/types.hpp>
#include <libaccint/core/derivative_utils.hpp>

#include <cassert>
#include <span>
#include <vector>

namespace libaccint {

// =============================================================================
// Metadata Types
// =============================================================================

/// @brief Metadata for a single shell quartet within an IntegralBuffer
struct ShellQuartetMeta {
    Size offset{0};          ///< Offset into the data buffer for this shell quartet
    Index fi{0}, fj{0};      ///< Basis function indices for bra shells
    Index fk{0}, fl{0};      ///< Basis function indices for ket shells
    int na{0}, nb{0};        ///< Number of functions in bra shells
    int nc{0}, nd{0};        ///< Number of functions in ket shells
};

/// @brief Metadata for a single shell pair within a 1e IntegralBuffer
struct ShellPairMeta {
    Size offset{0};          ///< Offset into the data buffer for this shell pair
    Index fi{0}, fj{0};      ///< Basis function indices
    int na{0}, nb{0};        ///< Number of functions per shell
};

// =============================================================================
// IntegralBuffer
// =============================================================================

/// @brief Buffer holding computed integrals for a ShellSetQuartet or ShellSetPair
///
/// IntegralBuffer provides non-owning access to computed integral values.
/// It stores a contiguous buffer of Real values and per-entry metadata
/// (basis function indices and function counts) so callers can interpret
/// the buffer contents without knowledge of the shell structure.
///
/// The buffer supports both 1e (ShellPairMeta) and 2e (ShellQuartetMeta) modes.
/// The mode is determined by which metadata vector is populated.
///
/// Usage:
/// @code
///   IntegralBuffer buf = engine.compute_batch(op, quartet);
///   for (Size i = 0; i < buf.n_shell_quartets(); ++i) {
///       auto meta = buf.quartet_meta(i);
///       auto data = buf.quartet_data(i);
///       // data[a*nb*nc*nd + b*nc*nd + c*nd + d] is integral (a,b|c,d)
///   }
/// @endcode
class IntegralBuffer {
public:
    /// @brief Default constructor (empty buffer)
    IntegralBuffer() = default;

    // =========================================================================
    // 2e (Quartet) Interface
    // =========================================================================

    /// @brief Number of shell quartets in this buffer (2e mode)
    [[nodiscard]] Size n_shell_quartets() const noexcept {
        return quartet_meta_.size();
    }

    /// @brief Access metadata for a specific shell quartet
    [[nodiscard]] const ShellQuartetMeta& quartet_meta(Size idx) const {
        assert(idx < quartet_meta_.size());
        return quartet_meta_[idx];
    }

    /// @brief Access integral values for a specific shell quartet
    [[nodiscard]] std::span<const Real> quartet_data(Size idx) const {
        assert(idx < quartet_meta_.size());
        const auto& meta = quartet_meta_[idx];
        Size size = static_cast<Size>(meta.na) * meta.nb * meta.nc * meta.nd;
        return std::span<const Real>(data_.data() + meta.offset, size);
    }

    // =========================================================================
    // 1e (Pair) Interface
    // =========================================================================

    /// @brief Number of shell pairs in this buffer (1e mode)
    [[nodiscard]] Size n_shell_pairs() const noexcept {
        return pair_meta_.size();
    }

    /// @brief Access metadata for a specific shell pair
    [[nodiscard]] const ShellPairMeta& pair_meta(Size idx) const {
        assert(idx < pair_meta_.size());
        return pair_meta_[idx];
    }

    /// @brief Access integral values for a specific shell pair
    [[nodiscard]] std::span<const Real> pair_data(Size idx) const {
        assert(idx < pair_meta_.size());
        const auto& meta = pair_meta_[idx];
        Size size = static_cast<Size>(meta.na) * meta.nb;
        return std::span<const Real>(data_.data() + meta.offset, size);
    }

    // =========================================================================
    // Derivative Component Accessors
    // =========================================================================

    /// @brief Access a specific derivative component for a shell quartet (2e)
    ///
    /// For DerivOrder=1, component is 0..11 (4 centers x 3 directions):
    ///   component = center * 3 + cart_dir
    ///
    /// The derivative data for each quartet is stored as:
    ///   [component_0 block][component_1 block]...[component_N block]
    /// where each block is na*nb*nc*nd values.
    ///
    /// @param idx Shell quartet index
    /// @param component Derivative component index
    /// @return Span of na*nb*nc*nd integral values for this derivative component
    [[nodiscard]] std::span<const Real> quartet_deriv_data(Size idx, int component) const {
        assert(idx < quartet_meta_.size());
        assert(component >= 0 && component < n_deriv_components_2e());
        const auto& meta = quartet_meta_[idx];
        Size block_size = static_cast<Size>(meta.na) * meta.nb * meta.nc * meta.nd;
        Size offset = meta.offset + static_cast<Size>(component) * block_size;
        return std::span<const Real>(data_.data() + offset, block_size);
    }

    /// @brief Access a specific derivative component for a shell pair (1e)
    [[nodiscard]] std::span<const Real> pair_deriv_data(Size idx, int component) const {
        assert(idx < pair_meta_.size());
        assert(component >= 0 && component < n_deriv_components_1e());
        const auto& meta = pair_meta_[idx];
        Size block_size = static_cast<Size>(meta.na) * meta.nb;
        Size offset = meta.offset + static_cast<Size>(component) * block_size;
        return std::span<const Real>(data_.data() + offset, block_size);
    }

    // =========================================================================
    // Common Interface
    // =========================================================================

    /// @brief Total number of integral values in the buffer
    [[nodiscard]] Size n_integrals() const noexcept { return data_.size(); }

    /// @brief Access all integral values as a contiguous span
    [[nodiscard]] std::span<const Real> data() const noexcept {
        return std::span<const Real>(data_);
    }

    /// @brief Check if the buffer is empty
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    /// @brief Angular momentum of center A (bra, first shell)
    [[nodiscard]] int La() const noexcept { return la_; }

    /// @brief Angular momentum of center B (bra, second shell)
    [[nodiscard]] int Lb() const noexcept { return lb_; }

    /// @brief Angular momentum of center C (ket, first shell) — 2e only
    [[nodiscard]] int Lc() const noexcept { return lc_; }

    /// @brief Angular momentum of center D (ket, second shell) — 2e only
    [[nodiscard]] int Ld() const noexcept { return ld_; }

    // =========================================================================
    // Builder Interface (used internally by Engine)
    // =========================================================================

    /// @brief Set the derivative order for this buffer
    /// @param order 0 for energy, 1 for gradient, 2 for Hessian
    void set_deriv_order(int order) noexcept {
        assert(order >= 0 && order <= 2);
        deriv_order_ = order;
    }

    /// @brief Get the derivative order
    [[nodiscard]] int deriv_order() const noexcept { return deriv_order_; }

    /// @brief Number of derivative components for the current derivative order (1e)
    [[nodiscard]] int n_deriv_components_1e() const noexcept {
        if (deriv_order_ == 0) return 1;
        if (deriv_order_ == 1) return N_DERIV_1E;        // 6
        return N_DERIV_1E * (N_DERIV_1E + 1) / 2;       // 21 (Hessian)
    }

    /// @brief Number of derivative components for the current derivative order (2e)
    [[nodiscard]] int n_deriv_components_2e() const noexcept {
        if (deriv_order_ == 0) return 1;
        if (deriv_order_ == 1) return N_DERIV_2E;        // 12
        return N_DERIV_2E * (N_DERIV_2E + 1) / 2;       // 78 (Hessian)
    }

    /// @brief Reserve space for 2e data
    void reserve_2e(Size n_values, Size n_quartets) {
        data_.reserve(n_values);
        quartet_meta_.reserve(n_quartets);
    }

    /// @brief Reserve space for 1e data
    void reserve_1e(Size n_values, Size n_pairs) {
        data_.reserve(n_values);
        pair_meta_.reserve(n_pairs);
    }

    /// @brief Reserve space for 2e gradient data (DerivOrder=1)
    /// @param n_base_values Base buffer size (without derivative multiplier)
    /// @param n_quartets Number of shell quartets
    void reserve_2e_gradient(Size n_base_values, Size n_quartets) {
        deriv_order_ = 1;
        data_.reserve(n_base_values * N_DERIV_2E);
        quartet_meta_.reserve(n_quartets);
    }

    /// @brief Reserve space for 2e Hessian data (DerivOrder=2)
    void reserve_2e_hessian(Size n_base_values, Size n_quartets) {
        deriv_order_ = 2;
        data_.reserve(n_base_values * n_deriv_components_2e());
        quartet_meta_.reserve(n_quartets);
    }

    /// @brief Reserve space for 1e gradient data (DerivOrder=1)
    void reserve_1e_gradient(Size n_base_values, Size n_pairs) {
        deriv_order_ = 1;
        data_.reserve(n_base_values * N_DERIV_1E);
        pair_meta_.reserve(n_pairs);
    }

    /// @brief Reserve space for 1e Hessian data (DerivOrder=2)
    void reserve_1e_hessian(Size n_base_values, Size n_pairs) {
        deriv_order_ = 2;
        data_.reserve(n_base_values * n_deriv_components_1e());
        pair_meta_.reserve(n_pairs);
    }

    /// @brief Append a shell quartet's worth of integral data (2e)
    void append_quartet(std::span<const Real> values,
                        Index fi, Index fj, Index fk, Index fl,
                        int na, int nb, int nc, int nd) {
        ShellQuartetMeta meta;
        meta.offset = data_.size();
        meta.fi = fi;
        meta.fj = fj;
        meta.fk = fk;
        meta.fl = fl;
        meta.na = na;
        meta.nb = nb;
        meta.nc = nc;
        meta.nd = nd;
        quartet_meta_.push_back(meta);
        data_.insert(data_.end(), values.begin(), values.end());
    }

    /// @brief Append a shell pair's worth of integral data (1e)
    void append_pair(std::span<const Real> values,
                     Index fi, Index fj, int na, int nb) {
        ShellPairMeta meta;
        meta.offset = data_.size();
        meta.fi = fi;
        meta.fj = fj;
        meta.na = na;
        meta.nb = nb;
        pair_meta_.push_back(meta);
        data_.insert(data_.end(), values.begin(), values.end());
    }

    /// @brief Set angular momentum metadata
    void set_am(int la, int lb, int lc = 0, int ld = 0) noexcept {
        la_ = la;
        lb_ = lb;
        lc_ = lc;
        ld_ = ld;
    }

    /// @brief Clear all data and metadata
    void clear() {
        data_.clear();
        quartet_meta_.clear();
        pair_meta_.clear();
        deriv_order_ = 0;
    }

private:
    std::vector<Real> data_;                  ///< Contiguous integral values
    std::vector<ShellQuartetMeta> quartet_meta_;  ///< 2e metadata
    std::vector<ShellPairMeta> pair_meta_;         ///< 1e metadata
    int la_{0}, lb_{0}, lc_{0}, ld_{0};       ///< Angular momentum class
    int deriv_order_{0};                      ///< Derivative order (0=energy, 1=gradient, 2=Hessian)
};

}  // namespace libaccint
