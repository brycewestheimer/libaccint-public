// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file three_center_buffer.hpp
/// @brief Buffer for three-center integral storage (density fitting)
///
/// Supports both single-precision (float) and double-precision (double) storage.
/// The precision is selected via the RealType template parameter.

#include <libaccint/core/types.hpp>
#include <libaccint/core/precision.hpp>
#include <vector>
#include <span>
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace libaccint {

/// @brief Buffer for three-center integral storage (density fitting)
/// @tparam DerivOrder Derivative order (0=energy, 1=gradient, 2=hessian)
/// @tparam RealType Precision type (float or double, defaults to Real = double)
///
/// Storage layout for density-fitting integrals (p | a b):
/// - DerivOrder=0: [np][na][nb]
/// - DerivOrder=1: [n_deriv][np][na][nb] where n_deriv = 9 (3 coords x 3 centers)
/// - DerivOrder=2: [n_deriv][np][na][nb] where n_deriv = 45 (Hessian components)
///
/// Memory is contiguous and row-major, suitable for efficient kernel output.
///
/// Usage:
/// @code
///   // Double precision (default)
///   ThreeCenterBuffer<0> eri3_double(10, 6, 6);
///
///   // Single precision
///   ThreeCenterBuffer<0, float> eri3_float(10, 6, 6);
/// @endcode
template<int DerivOrder = 0, typename RealType = Real>
    requires (DerivOrder >= 0 && DerivOrder <= 2) && ValidPrecision<RealType>
class ThreeCenterBuffer {
public:
    /// The scalar type used for storage
    using Scalar = RealType;

    /// Number of derivative components for 3 centers at this derivative order
    static constexpr int N_DERIV = n_derivative_components<DerivOrder, 3>();

    /// Is this single precision?
    static constexpr bool is_single_precision = PrecisionTraits<RealType>::is_single;

    /// Is this double precision?
    static constexpr bool is_double_precision = PrecisionTraits<RealType>::is_double;

    /// Default constructor creates an empty buffer
    ThreeCenterBuffer() = default;

    /// Construct with specified dimensions
    /// @param np Number of auxiliary basis functions
    /// @param na Number of functions in orbital shell a
    /// @param nb Number of functions in orbital shell b
    ThreeCenterBuffer(int np, int na, int nb)
        : np_(np), na_(na), nb_(nb)
    {
        data_.resize(static_cast<Size>(N_DERIV) * np * na * nb);
    }

    /// Resize the buffer to new dimensions
    /// @param np Number of auxiliary basis functions
    /// @param na Number of functions in orbital shell a
    /// @param nb Number of functions in orbital shell b
    ///
    /// Invalidates existing data. Call clear() before use.
    void resize(int np, int na, int nb) {
        np_ = np;
        na_ = na;
        nb_ = nb;
        data_.resize(static_cast<Size>(N_DERIV) * np * na * nb);
    }

    /// Zero all buffer contents
    void clear() noexcept {
        std::fill(data_.begin(), data_.end(), RealType{0});
    }

    // ========================================================================
    // Accessors for DerivOrder=0 (energy integrals)
    // ========================================================================

    /// Access integral value (p | a b) for energy integrals
    /// @param p Index for auxiliary function (0 <= p < np)
    /// @param a Index for orbital shell a (0 <= a < na)
    /// @param b Index for orbital shell b (0 <= b < nb)
    [[nodiscard]] RealType& operator()(int p, int a, int b)
        requires (DerivOrder == 0)
    {
        assert(p >= 0 && p < np_ && "Index p out of bounds");
        assert(a >= 0 && a < na_ && "Index a out of bounds");
        assert(b >= 0 && b < nb_ && "Index b out of bounds");
        return data_[linear_index(p, a, b)];
    }

    /// Access integral value (p | a b) for energy integrals (const)
    /// @param p Index for auxiliary function (0 <= p < np)
    /// @param a Index for orbital shell a (0 <= a < na)
    /// @param b Index for orbital shell b (0 <= b < nb)
    [[nodiscard]] RealType operator()(int p, int a, int b) const
        requires (DerivOrder == 0)
    {
        assert(p >= 0 && p < np_ && "Index p out of bounds");
        assert(a >= 0 && a < na_ && "Index a out of bounds");
        assert(b >= 0 && b < nb_ && "Index b out of bounds");
        return data_[linear_index(p, a, b)];
    }

    // ========================================================================
    // Accessors for DerivOrder=1 (gradient integrals)
    // ========================================================================

    /// Access integral derivative (p | a b) for gradient integrals
    /// @param p Index for auxiliary function (0 <= p < np)
    /// @param a Index for orbital shell a (0 <= a < na)
    /// @param b Index for orbital shell b (0 <= b < nb)
    /// @param deriv Derivative component index (0 <= deriv < N_DERIV)
    [[nodiscard]] RealType& operator()(int p, int a, int b, int deriv)
        requires (DerivOrder == 1)
    {
        assert(p >= 0 && p < np_ && "Index p out of bounds");
        assert(a >= 0 && a < na_ && "Index a out of bounds");
        assert(b >= 0 && b < nb_ && "Index b out of bounds");
        assert(deriv >= 0 && deriv < N_DERIV && "Derivative index out of bounds");
        return data_[linear_index_deriv(p, a, b, deriv)];
    }

    /// Access integral derivative (p | a b) for gradient integrals (const)
    /// @param p Index for auxiliary function (0 <= p < np)
    /// @param a Index for orbital shell a (0 <= a < na)
    /// @param b Index for orbital shell b (0 <= b < nb)
    /// @param deriv Derivative component index (0 <= deriv < N_DERIV)
    [[nodiscard]] RealType operator()(int p, int a, int b, int deriv) const
        requires (DerivOrder == 1)
    {
        assert(p >= 0 && p < np_ && "Index p out of bounds");
        assert(a >= 0 && a < na_ && "Index a out of bounds");
        assert(b >= 0 && b < nb_ && "Index b out of bounds");
        assert(deriv >= 0 && deriv < N_DERIV && "Derivative index out of bounds");
        return data_[linear_index_deriv(p, a, b, deriv)];
    }

    // ========================================================================
    // Accessors for DerivOrder=2 (Hessian integrals)
    // ========================================================================

    /// Access integral second derivative (p | a b) for Hessian integrals
    /// @param p Index for auxiliary function (0 <= p < np)
    /// @param a Index for orbital shell a (0 <= a < na)
    /// @param b Index for orbital shell b (0 <= b < nb)
    /// @param hess_component Hessian component index (0 <= hess_component < N_DERIV)
    [[nodiscard]] RealType& operator()(int p, int a, int b, int hess_component)
        requires (DerivOrder == 2)
    {
        assert(p >= 0 && p < np_ && "Index p out of bounds");
        assert(a >= 0 && a < na_ && "Index a out of bounds");
        assert(b >= 0 && b < nb_ && "Index b out of bounds");
        assert(hess_component >= 0 && hess_component < N_DERIV && "Derivative index out of bounds");
        return data_[linear_index_deriv(p, a, b, hess_component)];
    }

    /// Access integral second derivative (p | a b) for Hessian integrals (const)
    /// @param p Index for auxiliary function (0 <= p < np)
    /// @param a Index for orbital shell a (0 <= a < na)
    /// @param b Index for orbital shell b (0 <= b < nb)
    /// @param hess_component Hessian component index (0 <= hess_component < N_DERIV)
    [[nodiscard]] RealType operator()(int p, int a, int b, int hess_component) const
        requires (DerivOrder == 2)
    {
        assert(p >= 0 && p < np_ && "Index p out of bounds");
        assert(a >= 0 && a < na_ && "Index a out of bounds");
        assert(b >= 0 && b < nb_ && "Index b out of bounds");
        assert(hess_component >= 0 && hess_component < N_DERIV && "Derivative index out of bounds");
        return data_[linear_index_deriv(p, a, b, hess_component)];
    }

    // ========================================================================
    // Data access
    // ========================================================================

    /// Get mutable span over buffer data
    [[nodiscard]] std::span<RealType> data() noexcept {
        return std::span<RealType>(data_);
    }

    /// Get const span over buffer data
    [[nodiscard]] std::span<const RealType> data() const noexcept {
        return std::span<const RealType>(data_);
    }

    /// Get raw pointer to buffer data
    [[nodiscard]] RealType* data_ptr() noexcept {
        return data_.data();
    }

    /// Get const raw pointer to buffer data
    [[nodiscard]] const RealType* data_ptr() const noexcept {
        return data_.data();
    }

    // ========================================================================
    // Dimensions and size queries
    // ========================================================================

    /// Total number of stored values (np * na * nb * N_DERIV)
    [[nodiscard]] Size size() const noexcept {
        return data_.size();
    }

    /// Size in bytes
    [[nodiscard]] Size size_bytes() const noexcept {
        return data_.size() * sizeof(RealType);
    }

    /// Number of integral values (np * na * nb), excludes derivative components
    [[nodiscard]] Size n_integrals() const noexcept {
        return static_cast<Size>(np_) * na_ * nb_;
    }

    /// Number of auxiliary basis functions
    [[nodiscard]] int np() const noexcept { return np_; }

    /// Number of functions in orbital shell a
    [[nodiscard]] int na() const noexcept { return na_; }

    /// Number of functions in orbital shell b
    [[nodiscard]] int nb() const noexcept { return nb_; }

    /// Check if buffer is empty
    [[nodiscard]] bool empty() const noexcept {
        return data_.empty();
    }

    /// @brief Convert buffer to different precision
    /// @tparam TargetReal Target precision type
    /// @return New buffer with converted values
    template<typename TargetReal>
        requires ValidPrecision<TargetReal>
    [[nodiscard]] ThreeCenterBuffer<DerivOrder, TargetReal> to_precision() const {
        ThreeCenterBuffer<DerivOrder, TargetReal> result(np_, na_, nb_);
        for (Size i = 0; i < data_.size(); ++i) {
            result.data()[i] = static_cast<TargetReal>(data_[i]);
        }
        return result;
    }

    /// @brief Copy from buffer of different precision
    /// @tparam SourceReal Source precision type
    /// @param source Source buffer to copy from
    template<typename SourceReal>
        requires ValidPrecision<SourceReal>
    void copy_from(const ThreeCenterBuffer<DerivOrder, SourceReal>& source) {
        resize(source.np(), source.na(), source.nb());
        for (Size i = 0; i < data_.size(); ++i) {
            data_[i] = static_cast<RealType>(source.data()[i]);
        }
    }

private:
    /// Compute linear index for energy integrals (DerivOrder=0)
    /// Layout: p * (na*nb) + a * nb + b
    [[nodiscard]] Size linear_index(int p, int a, int b) const noexcept {
        return static_cast<Size>(p) * na_ * nb_
             + static_cast<Size>(a) * nb_
             + static_cast<Size>(b);
    }

    /// Compute linear index for derivative integrals (DerivOrder >= 1)
    /// Layout: deriv * (np*na*nb) + p * (na*nb) + a * nb + b
    [[nodiscard]] Size linear_index_deriv(int p, int a, int b, int deriv) const noexcept {
        return static_cast<Size>(deriv) * np_ * na_ * nb_
             + static_cast<Size>(p) * na_ * nb_
             + static_cast<Size>(a) * nb_
             + static_cast<Size>(b);
    }

    std::vector<RealType> data_;  ///< Contiguous storage for integral values
    int np_{0};                   ///< Number of auxiliary basis functions
    int na_{0};                   ///< Number of functions in orbital shell a
    int nb_{0};                   ///< Number of functions in orbital shell b
};

// ============================================================================
// Type Aliases (Double Precision - Default)
// ============================================================================

/// Type alias for three-center ERI buffer (energy only) - double precision
using ThreeCenterERIBuffer = ThreeCenterBuffer<0, double>;

/// Type alias for three-center ERI gradient buffer - double precision
using ThreeCenterERIGradientBuffer = ThreeCenterBuffer<1, double>;

// ============================================================================
// Type Aliases (Single Precision)
// ============================================================================

/// Type alias for three-center ERI buffer (energy only) - single precision
using ThreeCenterERIBufferFloat = ThreeCenterBuffer<0, float>;

/// Type alias for three-center ERI gradient buffer - single precision
using ThreeCenterERIGradientBufferFloat = ThreeCenterBuffer<1, float>;

}  // namespace libaccint
