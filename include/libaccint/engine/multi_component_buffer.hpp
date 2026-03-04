// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file multi_component_buffer.hpp
/// @brief Multi-component buffer for vector/tensor property integrals
///
/// MultiComponentBuffer manages N integral matrices for N-component operators
/// such as dipole (3), quadrupole (6), octupole (10), and momentum (3).
/// Storage is component-major: all elements of component 0, then component 1, etc.

#include <libaccint/core/types.hpp>
#include <libaccint/operators/operator_types.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace libaccint {

/// @brief Symmetry type for component matrices
enum class MatrixSymmetry {
    Symmetric,      ///< M(i,j) = M(j,i) — dipole, quadrupole, octupole
    AntiSymmetric,  ///< M(i,j) = -M(j,i) — linear/angular momentum
};

/// @brief Buffer for multi-component integral results (property integrals)
///
/// Manages N integral matrices stored in contiguous memory with component-major
/// layout: [comp0_data | comp1_data | ... | compN_data].
///
/// Each component matrix has dimensions na × nb (rows × cols) for the current
/// shell pair, matching the OneElectronBuffer resize(na, nb) pattern.
/// Supports origin specification for origin-dependent integrals.
///
/// @code
///   MultiComponentBuffer buf(OperatorKind::ElectricDipole);
///   buf.resize(3, 6);  // p-shell × d-shell → 3 × 18 = 54 doubles
///   buf(0, 1, 2) = 0.5; // component 0, bra function 1, ket function 2
/// @endcode
class MultiComponentBuffer {
public:
    /// @brief Construct from an operator kind (auto-determines component count)
    /// @param op The operator kind
    explicit MultiComponentBuffer(OperatorKind op)
        : op_kind_(op)
        , n_components_(static_cast<Size>(component_count(op)))
    {}

    /// @brief Construct with explicit component count
    /// @param n_components Number of integral components
    explicit MultiComponentBuffer(Size n_components)
        : op_kind_(OperatorKind::Overlap) // fallback
        , n_components_(n_components)
    {}

    /// @brief Resize buffer for na × nb component matrices and zero-fill
    /// @param na Number of bra functions (rows)
    /// @param nb Number of ket functions (columns)
    void resize(int na, int nb) {
        na_ = na;
        nb_ = nb;
        data_.assign(total_size(), 0.0);
    }

    /// @brief Zero all elements without deallocation
    void clear() {
        std::fill(data_.begin(), data_.end(), 0.0);
    }

    /// @brief Get number of bra functions (rows)
    [[nodiscard]] int na() const noexcept { return na_; }

    /// @brief Get number of ket functions (columns)
    [[nodiscard]] int nb() const noexcept { return nb_; }

    /// @brief Get the number of components
    [[nodiscard]] Size n_components() const noexcept { return n_components_; }

    /// @brief Get total number of stored doubles
    [[nodiscard]] Size total_size() const noexcept {
        return n_components_ * static_cast<Size>(na_) * static_cast<Size>(nb_);
    }

    /// @brief Get mutable span for component i
    /// @param i Component index (0-based)
    [[nodiscard]] std::span<Real> component(Size i) {
        assert(i < n_components_);
        Size offset = i * static_cast<Size>(na_) * static_cast<Size>(nb_);
        return std::span<Real>(data_.data() + offset,
                               static_cast<Size>(na_) * static_cast<Size>(nb_));
    }

    /// @brief Get const span for component i
    /// @param i Component index (0-based)
    [[nodiscard]] std::span<const Real> component(Size i) const {
        assert(i < n_components_);
        Size offset = i * static_cast<Size>(na_) * static_cast<Size>(nb_);
        return std::span<const Real>(data_.data() + offset,
                                     static_cast<Size>(na_) * static_cast<Size>(nb_));
    }

    /// @brief Access element (component, bra_index, ket_index) — mutable
    [[nodiscard]] Real& operator()(Size comp, int a, int b) {
        assert(comp < n_components_ && a < na_ && b < nb_);
        return data_[comp * static_cast<Size>(na_) * static_cast<Size>(nb_)
                     + static_cast<Size>(a) * static_cast<Size>(nb_)
                     + static_cast<Size>(b)];
    }

    /// @brief Access element (component, bra_index, ket_index) — const
    [[nodiscard]] Real operator()(Size comp, int a, int b) const {
        assert(comp < n_components_ && a < na_ && b < nb_);
        return data_[comp * static_cast<Size>(na_) * static_cast<Size>(nb_)
                     + static_cast<Size>(a) * static_cast<Size>(nb_)
                     + static_cast<Size>(b)];
    }

    /// @brief Get raw pointer to contiguous storage
    [[nodiscard]] Real* data() noexcept { return data_.data(); }

    /// @brief Get const raw pointer to contiguous storage
    [[nodiscard]] const Real* data() const noexcept { return data_.data(); }

    /// @brief Set the gauge/expansion origin
    /// @param origin 3D origin coordinates
    void set_origin(std::array<Real, 3> origin) { origin_ = origin; }

    /// @brief Get the gauge/expansion origin
    [[nodiscard]] std::array<Real, 3> origin() const noexcept { return origin_; }

    /// @brief Get the operator kind
    [[nodiscard]] OperatorKind operator_kind() const noexcept { return op_kind_; }

    /// @brief Get the symmetry type of component matrices
    [[nodiscard]] MatrixSymmetry symmetry_type() const noexcept {
        return is_anti_hermitian(op_kind_) ? MatrixSymmetry::AntiSymmetric
                                           : MatrixSymmetry::Symmetric;
    }

    /// @brief Get a human-readable label for component i
    /// @param i Component index
    /// @return Label string (e.g., "x", "xy", "xxx")
    [[nodiscard]] std::string component_label(Size i) const {
        assert(i < n_components_);
        // Dipole / momentum: x, y, z
        if (n_components_ == 3 && (op_kind_ == OperatorKind::ElectricDipole ||
                                    op_kind_ == OperatorKind::LinearMomentum ||
                                    op_kind_ == OperatorKind::AngularMomentum)) {
            static const char* labels[] = {"x", "y", "z"};
            return labels[i];
        }
        // Quadrupole: xx, xy, xz, yy, yz, zz
        if (n_components_ == 6) {
            static const char* labels[] = {"xx", "xy", "xz", "yy", "yz", "zz"};
            return labels[i];
        }
        // Octupole: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
        if (n_components_ == 10) {
            static const char* labels[] = {
                "xxx", "xxy", "xxz", "xyy", "xyz",
                "xzz", "yyy", "yyz", "yzz", "zzz"
            };
            return labels[i];
        }
        return std::to_string(i);
    }

private:
    OperatorKind op_kind_;
    Size n_components_;
    int na_{0};
    int nb_{0};
    std::vector<Real> data_;
    std::array<Real, 3> origin_{0.0, 0.0, 0.0};
};

}  // namespace libaccint
