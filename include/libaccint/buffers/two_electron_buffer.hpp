// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file two_electron_buffer.hpp
/// @brief Buffer for two-electron integral storage with derivative and precision support
///
/// Supports both single-precision (float) and double-precision (double) storage.
/// The precision is selected via the RealType template parameter.

#include <libaccint/core/types.hpp>
#include <libaccint/core/precision.hpp>
#include <vector>
#include <span>
#include <stdexcept>
#include <cassert>
#include <cstddef>

namespace libaccint {

/// @brief Buffer for two-electron (4-center) integral storage
/// @tparam DerivOrder Derivative order (0=energy, 1=gradient, 2=hessian)
/// @tparam RealType Precision type (float or double, defaults to Real = double)
///
/// Storage layout:
/// - DerivOrder=0: [na][nb][nc][nd]
/// - DerivOrder=1: [n_deriv][na][nb][nc][nd] where n_deriv = 12 (3 coords x 4 centers)
/// - DerivOrder=2: [n_deriv][na][nb][nc][nd] where n_deriv = 78 (Hessian components)
///
/// Memory is contiguous and row-major, suitable for efficient kernel output.
///
/// Usage:
/// @code
///   // Double precision (default)
///   TwoElectronBuffer<0> eri_double(10, 10, 10, 10);
///
///   // Single precision
///   TwoElectronBuffer<0, float> eri_float(10, 10, 10, 10);
/// @endcode
template<int DerivOrder = 0, typename RealType = Real>
    requires (DerivOrder >= 0 && DerivOrder <= 2) && ValidPrecision<RealType>
class TwoElectronBuffer {
public:
    /// The scalar type used for storage
    using Scalar = RealType;

    /// Number of derivative components for 4 centers at this derivative order
    static constexpr int N_DERIV = n_derivative_components<DerivOrder, 4>();

    /// Is this single precision?
    static constexpr bool is_single_precision = PrecisionTraits<RealType>::is_single;

    /// Is this double precision?
    static constexpr bool is_double_precision = PrecisionTraits<RealType>::is_double;

    /// Default constructor creates an empty buffer
    TwoElectronBuffer() = default;

    /// Construct with specified dimensions
    /// @param na Number of functions in shell a
    /// @param nb Number of functions in shell b
    /// @param nc Number of functions in shell c
    /// @param nd Number of functions in shell d
    TwoElectronBuffer(int na, int nb, int nc, int nd)
        : na_(na), nb_(nb), nc_(nc), nd_(nd)
    {
        data_.resize(static_cast<Size>(N_DERIV) * na * nb * nc * nd);
    }

    /// Resize the buffer to new dimensions
    /// @param na Number of functions in shell a
    /// @param nb Number of functions in shell b
    /// @param nc Number of functions in shell c
    /// @param nd Number of functions in shell d
    ///
    /// Invalidates existing data. Call clear() before use.
    void resize(int na, int nb, int nc, int nd) {
        na_ = na;
        nb_ = nb;
        nc_ = nc;
        nd_ = nd;
        data_.resize(static_cast<Size>(N_DERIV) * na * nb * nc * nd);
    }

    /// Zero all buffer contents
    void clear() noexcept {
        std::fill(data_.begin(), data_.end(), RealType{0});
    }

    // ========================================================================
    // Accessors for DerivOrder=0 (energy integrals)
    // ========================================================================

    /// Access integral value (a b | c d) for energy integrals
    /// @param a Index for shell a (0 <= a < na)
    /// @param b Index for shell b (0 <= b < nb)
    /// @param c Index for shell c (0 <= c < nc)
    /// @param d Index for shell d (0 <= d < nd)
    [[nodiscard]] RealType& operator()(int a, int b, int c, int d)
        requires (DerivOrder == 0)
    {
        assert(a >= 0 && a < na_ && "Index a out of bounds");
        assert(b >= 0 && b < nb_ && "Index b out of bounds");
        assert(c >= 0 && c < nc_ && "Index c out of bounds");
        assert(d >= 0 && d < nd_ && "Index d out of bounds");
        return data_[linear_index(a, b, c, d)];
    }

    /// Access integral value (a b | c d) for energy integrals (const)
    /// @param a Index for shell a (0 <= a < na)
    /// @param b Index for shell b (0 <= b < nb)
    /// @param c Index for shell c (0 <= c < nc)
    /// @param d Index for shell d (0 <= d < nd)
    [[nodiscard]] RealType operator()(int a, int b, int c, int d) const
        requires (DerivOrder == 0)
    {
        assert(a >= 0 && a < na_ && "Index a out of bounds");
        assert(b >= 0 && b < nb_ && "Index b out of bounds");
        assert(c >= 0 && c < nc_ && "Index c out of bounds");
        assert(d >= 0 && d < nd_ && "Index d out of bounds");
        return data_[linear_index(a, b, c, d)];
    }

    // ========================================================================
    // Accessors for DerivOrder=1 (gradient integrals)
    // ========================================================================

    /// Access integral derivative (a b | c d) for gradient integrals
    /// @param a Index for shell a (0 <= a < na)
    /// @param b Index for shell b (0 <= b < nb)
    /// @param c Index for shell c (0 <= c < nc)
    /// @param d Index for shell d (0 <= d < nd)
    /// @param deriv Derivative component index (0 <= deriv < N_DERIV)
    [[nodiscard]] RealType& operator()(int a, int b, int c, int d, int deriv)
        requires (DerivOrder == 1)
    {
        assert(a >= 0 && a < na_ && "Index a out of bounds");
        assert(b >= 0 && b < nb_ && "Index b out of bounds");
        assert(c >= 0 && c < nc_ && "Index c out of bounds");
        assert(d >= 0 && d < nd_ && "Index d out of bounds");
        assert(deriv >= 0 && deriv < N_DERIV && "Derivative index out of bounds");
        return data_[linear_index_deriv(a, b, c, d, deriv)];
    }

    /// Access integral derivative (a b | c d) for gradient integrals (const)
    /// @param a Index for shell a (0 <= a < na)
    /// @param b Index for shell b (0 <= b < nb)
    /// @param c Index for shell c (0 <= c < nc)
    /// @param d Index for shell d (0 <= d < nd)
    /// @param deriv Derivative component index (0 <= deriv < N_DERIV)
    [[nodiscard]] RealType operator()(int a, int b, int c, int d, int deriv) const
        requires (DerivOrder == 1)
    {
        assert(a >= 0 && a < na_ && "Index a out of bounds");
        assert(b >= 0 && b < nb_ && "Index b out of bounds");
        assert(c >= 0 && c < nc_ && "Index c out of bounds");
        assert(d >= 0 && d < nd_ && "Index d out of bounds");
        assert(deriv >= 0 && deriv < N_DERIV && "Derivative index out of bounds");
        return data_[linear_index_deriv(a, b, c, d, deriv)];
    }

    // ========================================================================
    // Accessors for DerivOrder=2 (Hessian integrals)
    // ========================================================================

    /// Access integral second derivative (a b | c d) for Hessian integrals
    /// @param a Index for shell a (0 <= a < na)
    /// @param b Index for shell b (0 <= b < nb)
    /// @param c Index for shell c (0 <= c < nc)
    /// @param d Index for shell d (0 <= d < nd)
    /// @param hess_component Hessian component index (0 <= hess_component < N_DERIV)
    ///        Use hess_component_2e(center_i, dir_i, center_j, dir_j) to compute.
    [[nodiscard]] RealType& operator()(int a, int b, int c, int d, int hess_component)
        requires (DerivOrder == 2)
    {
        assert(a >= 0 && a < na_ && "Index a out of bounds");
        assert(b >= 0 && b < nb_ && "Index b out of bounds");
        assert(c >= 0 && c < nc_ && "Index c out of bounds");
        assert(d >= 0 && d < nd_ && "Index d out of bounds");
        assert(hess_component >= 0 && hess_component < N_DERIV && "Derivative index out of bounds");
        return data_[linear_index_deriv(a, b, c, d, hess_component)];
    }

    /// Access integral second derivative (a b | c d) for Hessian integrals (const)
    /// @param a Index for shell a (0 <= a < na)
    /// @param b Index for shell b (0 <= b < nb)
    /// @param c Index for shell c (0 <= c < nc)
    /// @param d Index for shell d (0 <= d < nd)
    /// @param hess_component Hessian component index (0 <= hess_component < N_DERIV)
    [[nodiscard]] RealType operator()(int a, int b, int c, int d, int hess_component) const
        requires (DerivOrder == 2)
    {
        assert(a >= 0 && a < na_ && "Index a out of bounds");
        assert(b >= 0 && b < nb_ && "Index b out of bounds");
        assert(c >= 0 && c < nc_ && "Index c out of bounds");
        assert(d >= 0 && d < nd_ && "Index d out of bounds");
        assert(hess_component >= 0 && hess_component < N_DERIV && "Derivative index out of bounds");
        return data_[linear_index_deriv(a, b, c, d, hess_component)];
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

    /// Total number of stored values (na * nb * nc * nd * N_DERIV)
    [[nodiscard]] Size size() const noexcept {
        return data_.size();
    }

    /// Size in bytes
    [[nodiscard]] Size size_bytes() const noexcept {
        return data_.size() * sizeof(RealType);
    }

    /// Number of integral values (na * nb * nc * nd), excludes derivative components
    [[nodiscard]] Size n_integrals() const noexcept {
        return static_cast<Size>(na_) * nb_ * nc_ * nd_;
    }

    /// Number of functions in shell a
    [[nodiscard]] int na() const noexcept { return na_; }

    /// Number of functions in shell b
    [[nodiscard]] int nb() const noexcept { return nb_; }

    /// Number of functions in shell c
    [[nodiscard]] int nc() const noexcept { return nc_; }

    /// Number of functions in shell d
    [[nodiscard]] int nd() const noexcept { return nd_; }

    /// Check if buffer is empty
    [[nodiscard]] bool empty() const noexcept {
        return data_.empty();
    }

    /// @brief Convert buffer to different precision
    /// @tparam TargetReal Target precision type
    /// @return New buffer with converted values
    template<typename TargetReal>
        requires ValidPrecision<TargetReal>
    [[nodiscard]] TwoElectronBuffer<DerivOrder, TargetReal> to_precision() const {
        TwoElectronBuffer<DerivOrder, TargetReal> result(na_, nb_, nc_, nd_);
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
    void copy_from(const TwoElectronBuffer<DerivOrder, SourceReal>& source) {
        resize(source.na(), source.nb(), source.nc(), source.nd());
        for (Size i = 0; i < data_.size(); ++i) {
            data_[i] = static_cast<RealType>(source.data()[i]);
        }
    }

private:
    /// Compute linear index for energy integrals (DerivOrder=0)
    /// Layout: a * (nb*nc*nd) + b * (nc*nd) + c * nd + d
    [[nodiscard]] Size linear_index(int a, int b, int c, int d) const noexcept {
        return static_cast<Size>(a) * nb_ * nc_ * nd_
             + static_cast<Size>(b) * nc_ * nd_
             + static_cast<Size>(c) * nd_
             + static_cast<Size>(d);
    }

    /// Compute linear index for derivative integrals (DerivOrder >= 1)
    /// Layout: deriv * (na*nb*nc*nd) + a * (nb*nc*nd) + b * (nc*nd) + c * nd + d
    [[nodiscard]] Size linear_index_deriv(int a, int b, int c, int d, int deriv) const noexcept {
        return static_cast<Size>(deriv) * na_ * nb_ * nc_ * nd_
             + static_cast<Size>(a) * nb_ * nc_ * nd_
             + static_cast<Size>(b) * nc_ * nd_
             + static_cast<Size>(c) * nd_
             + static_cast<Size>(d);
    }

    std::vector<RealType> data_;  ///< Contiguous storage for integral values
    int na_{0};                   ///< Number of functions in shell a
    int nb_{0};                   ///< Number of functions in shell b
    int nc_{0};                   ///< Number of functions in shell c
    int nd_{0};                   ///< Number of functions in shell d
};

// ============================================================================
// Type Aliases (Double Precision - Default)
// ============================================================================

/// Type alias for electron repulsion integral (ERI) buffer (energy only) - double precision
using ERIBuffer = TwoElectronBuffer<0, double>;

/// Type alias for ERI gradient buffer - double precision
using ERIGradientBuffer = TwoElectronBuffer<1, double>;

// ============================================================================
// Type Aliases (Single Precision)
// ============================================================================

/// Type alias for electron repulsion integral (ERI) buffer (energy only) - single precision
using ERIBufferFloat = TwoElectronBuffer<0, float>;

/// Type alias for ERI gradient buffer - single precision
using ERIGradientBufferFloat = TwoElectronBuffer<1, float>;

// ============================================================================
// Type Aliases (Hessian)
// ============================================================================

/// Type alias for ERI Hessian buffer - double precision
using ERIHessianBuffer = TwoElectronBuffer<2, double>;

/// Type alias for ERI Hessian buffer - single precision
using ERIHessianBufferFloat = TwoElectronBuffer<2, float>;

}  // namespace libaccint
