// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file one_electron_buffer.hpp
/// @brief Buffer for one-electron integral matrices
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

/// @brief Buffer for one-electron integrals with optional derivative and precision support
/// @tparam DerivOrder Derivative order (0 = energy, 1 = gradient, 2 = hessian)
/// @tparam RealType Precision type (float or double, defaults to Real = double)
///
/// Memory layout:
/// - DerivOrder=0: Linear index = a * nb + b
/// - DerivOrder=1: Linear index = deriv * (na * nb) + a * nb + b
/// - DerivOrder=2: Linear index = deriv * (na * nb) + a * nb + b
///
/// Storage is row-major with derivative component as the outermost dimension
/// for DerivOrder > 0.
///
/// Usage:
/// @code
///   // Double precision (default)
///   OneElectronBuffer<0> S_double(10, 10);
///
///   // Single precision
///   OneElectronBuffer<0, float> S_float(10, 10);
/// @endcode
template<int DerivOrder = 0, typename RealType = Real>
    requires (DerivOrder >= 0 && DerivOrder <= 2) && ValidPrecision<RealType>
class OneElectronBuffer {
public:
    /// The scalar type used for storage
    using Scalar = RealType;

    /// Number of derivative components (1 for energy, 6 for gradient, 21 for hessian)
    static constexpr int N_DERIV = n_derivative_components<DerivOrder, 2>();

    /// Is this single precision?
    static constexpr bool is_single_precision = PrecisionTraits<RealType>::is_single;

    /// Is this double precision?
    static constexpr bool is_double_precision = PrecisionTraits<RealType>::is_double;

    /// @brief Default constructor creates an empty buffer
    OneElectronBuffer() = default;

    /// @brief Construct buffer with given dimensions
    /// @param na Number of basis functions in shell A
    /// @param nb Number of basis functions in shell B
    OneElectronBuffer(int na, int nb) : na_(na), nb_(nb) {
        data_.resize(N_DERIV * na * nb);
    }

    /// @brief Resize buffer for given shell pair dimensions
    /// @param na Number of basis functions in shell A
    /// @param nb Number of basis functions in shell B
    void resize(int na, int nb) {
        na_ = na;
        nb_ = nb;
        data_.resize(N_DERIV * na * nb);
    }

    /// @brief Zero all elements in the buffer
    void clear() {
        std::fill(data_.begin(), data_.end(), RealType{0});
    }

    /// @brief Access integral value (energy only)
    /// @param a Index in shell A
    /// @param b Index in shell B
    /// @return Reference to integral value
    [[nodiscard]] RealType& operator()(int a, int b) requires (DerivOrder == 0) {
        assert(a >= 0 && a < na_ && "Index a out of bounds");
        assert(b >= 0 && b < nb_ && "Index b out of bounds");
        return data_[a * nb_ + b];
    }

    /// @brief Access integral value (energy only, const)
    /// @param a Index in shell A
    /// @param b Index in shell B
    /// @return Integral value
    [[nodiscard]] RealType operator()(int a, int b) const requires (DerivOrder == 0) {
        assert(a >= 0 && a < na_ && "Index a out of bounds");
        assert(b >= 0 && b < nb_ && "Index b out of bounds");
        return data_[a * nb_ + b];
    }

    /// @brief Access derivative component (gradient/hessian)
    /// @param a Index in shell A
    /// @param b Index in shell B
    /// @param deriv Derivative component index (0-5 for gradient, 0-20 for hessian)
    /// @return Reference to derivative value
    [[nodiscard]] RealType& operator()(int a, int b, int deriv) requires (DerivOrder >= 1) {
        assert(a >= 0 && a < na_ && "Index a out of bounds");
        assert(b >= 0 && b < nb_ && "Index b out of bounds");
        assert(deriv >= 0 && deriv < N_DERIV && "Derivative index out of bounds");
        return data_[deriv * (na_ * nb_) + a * nb_ + b];
    }

    /// @brief Access derivative component (gradient/hessian, const)
    /// @param a Index in shell A
    /// @param b Index in shell B
    /// @param deriv Derivative component index (0-5 for gradient, 0-20 for hessian)
    /// @return Derivative value
    [[nodiscard]] RealType operator()(int a, int b, int deriv) const requires (DerivOrder >= 1) {
        assert(a >= 0 && a < na_ && "Index a out of bounds");
        assert(b >= 0 && b < nb_ && "Index b out of bounds");
        assert(deriv >= 0 && deriv < N_DERIV && "Derivative index out of bounds");
        return data_[deriv * (na_ * nb_) + a * nb_ + b];
    }

    /// @brief Get mutable span of underlying data
    /// @return Span covering all buffer elements
    [[nodiscard]] std::span<RealType> data() noexcept {
        return std::span<RealType>(data_);
    }

    /// @brief Get const span of underlying data
    /// @return Const span covering all buffer elements
    [[nodiscard]] std::span<const RealType> data() const noexcept {
        return std::span<const RealType>(data_);
    }

    /// @brief Get raw pointer to underlying data
    /// @return Pointer to first element
    [[nodiscard]] RealType* data_ptr() noexcept {
        return data_.data();
    }

    /// @brief Get const raw pointer to underlying data
    /// @return Const pointer to first element
    [[nodiscard]] const RealType* data_ptr() const noexcept {
        return data_.data();
    }

    /// @brief Get total number of elements in buffer
    /// @return Total size (N_DERIV * na * nb)
    [[nodiscard]] Size size() const noexcept {
        return data_.size();
    }

    /// @brief Get size in bytes
    /// @return Total size in bytes
    [[nodiscard]] Size size_bytes() const noexcept {
        return data_.size() * sizeof(RealType);
    }

    /// @brief Get number of functions in shell A
    /// @return Number of functions in shell A
    [[nodiscard]] int na() const noexcept {
        return na_;
    }

    /// @brief Get number of functions in shell B
    /// @return Number of functions in shell B
    [[nodiscard]] int nb() const noexcept {
        return nb_;
    }

    /// @brief Check if buffer is empty
    /// @return True if buffer has zero size
    [[nodiscard]] bool empty() const noexcept {
        return data_.empty();
    }

    /// @brief Convert buffer to different precision
    /// @tparam TargetReal Target precision type
    /// @return New buffer with converted values
    template<typename TargetReal>
        requires ValidPrecision<TargetReal>
    [[nodiscard]] OneElectronBuffer<DerivOrder, TargetReal> to_precision() const {
        OneElectronBuffer<DerivOrder, TargetReal> result;
        result.resize(na_, nb_);
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
    void copy_from(const OneElectronBuffer<DerivOrder, SourceReal>& source) {
        resize(source.na(), source.nb());
        for (Size i = 0; i < data_.size(); ++i) {
            data_[i] = static_cast<RealType>(source.data()[i]);
        }
    }

private:
    std::vector<RealType> data_;  ///< Storage for integral values
    int na_{0};                   ///< Number of functions in shell A
    int nb_{0};                   ///< Number of functions in shell B
};

// ============================================================================
// Type Aliases (Double Precision - Default)
// ============================================================================

/// Buffer for overlap integrals (S matrix) - double precision
using OverlapBuffer = OneElectronBuffer<0, double>;

/// Buffer for kinetic energy integrals (T matrix) - double precision
using KineticBuffer = OneElectronBuffer<0, double>;

/// Buffer for nuclear attraction integrals (V matrix) - double precision
using NuclearBuffer = OneElectronBuffer<0, double>;

// ============================================================================
// Type Aliases (Single Precision)
// ============================================================================

/// Buffer for overlap integrals (S matrix) - single precision
using OverlapBufferFloat = OneElectronBuffer<0, float>;

/// Buffer for kinetic energy integrals (T matrix) - single precision
using KineticBufferFloat = OneElectronBuffer<0, float>;

/// Buffer for nuclear attraction integrals (V matrix) - single precision
using NuclearBufferFloat = OneElectronBuffer<0, float>;

// ============================================================================
// Type Aliases (Gradient Buffers)
// ============================================================================

/// Buffer for overlap gradients - double precision
using OverlapGradientBuffer = OneElectronBuffer<1, double>;

/// Buffer for kinetic energy gradients - double precision
using KineticGradientBuffer = OneElectronBuffer<1, double>;

/// Buffer for nuclear attraction gradients - double precision
using NuclearGradientBuffer = OneElectronBuffer<1, double>;

/// Buffer for overlap gradients - single precision
using OverlapGradientBufferFloat = OneElectronBuffer<1, float>;

/// Buffer for kinetic energy gradients - single precision
using KineticGradientBufferFloat = OneElectronBuffer<1, float>;

/// Buffer for nuclear attraction gradients - single precision
using NuclearGradientBufferFloat = OneElectronBuffer<1, float>;

// ============================================================================
// Type Aliases (Hessian Buffers)
// ============================================================================

/// Buffer for overlap Hessians - double precision
using OverlapHessianBuffer = OneElectronBuffer<2, double>;

/// Buffer for kinetic energy Hessians - double precision
using KineticHessianBuffer = OneElectronBuffer<2, double>;

/// Buffer for nuclear attraction Hessians - double precision
using NuclearHessianBuffer = OneElectronBuffer<2, double>;

/// Buffer for overlap Hessians - single precision
using OverlapHessianBufferFloat = OneElectronBuffer<2, float>;

/// Buffer for kinetic energy Hessians - single precision
using KineticHessianBufferFloat = OneElectronBuffer<2, float>;

/// Buffer for nuclear attraction Hessians - single precision
using NuclearHessianBufferFloat = OneElectronBuffer<2, float>;

}  // namespace libaccint
