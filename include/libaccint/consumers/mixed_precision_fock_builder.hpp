// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file mixed_precision_fock_builder.hpp
/// @brief Mixed-precision Fock matrix builder
///
/// Computes integrals in single precision (float32) and accumulates
/// Coulomb (J) and exchange (K) contributions in double precision (float64).
/// This approach provides significant speedup on GPU while maintaining
/// acceptable accuracy for SCF convergence.
///
/// Usage:
/// @code
///   MixedPrecisionFockBuilder builder(nbf);
///   builder.set_density(D_ptr, nbf);
///
///   // Compute in float32, automatically accumulates in float64
///   TwoElectronBuffer<0, float> float_buf(na, nb, nc, nd);
///   // ... compute float integrals ...
///   builder.accumulate(float_buf, fa, fb, fc, fd, na, nb, nc, nd);
///
///   // Retrieve results in double precision
///   auto J = builder.get_coulomb_matrix();
///   auto K = builder.get_exchange_matrix();
/// @endcode

#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/core/precision.hpp>
#include <libaccint/core/types.hpp>

#include <span>
#include <vector>

namespace libaccint::consumers {

/// @brief Mixed-precision Fock matrix builder
///
/// Accumulates Coulomb (J) and exchange (K) contributions from float32
/// integrals into float64 matrices, achieving the accuracy of double
/// precision accumulation with the throughput of single precision computation.
class MixedPrecisionFockBuilder {
public:
    /// @brief Construct a MixedPrecisionFockBuilder for a basis with nbf functions
    /// @param nbf Number of basis functions
    explicit MixedPrecisionFockBuilder(Size nbf);

    /// @brief Set the density matrix for accumulation
    /// @param D Pointer to row-major nbf x nbf density matrix (double precision)
    /// @param nbf Number of basis functions (must match constructor)
    /// @warning The caller must ensure the density matrix pointer remains valid
    ///          for the lifetime of any subsequent accumulate() calls. The builder
    ///          does not copy the matrix.
    void set_density(const Real* D, Size nbf);

    /// @brief Accumulate J and K from float32 integrals into float64 matrices
    ///
    /// The integral values are read as float32 and promoted to float64
    /// before accumulation into J and K matrices.
    ///
    /// @param buffer Float32 integral buffer for this shell quartet
    /// @param fa Starting basis function index for shell a
    /// @param fb Starting basis function index for shell b
    /// @param fc Starting basis function index for shell c
    /// @param fd Starting basis function index for shell d
    /// @param na Number of functions in shell a
    /// @param nb Number of functions in shell b
    /// @param nc Number of functions in shell c
    /// @param nd Number of functions in shell d
    void accumulate(const TwoElectronBuffer<0, float>& buffer,
                    Index fa, Index fb, Index fc, Index fd,
                    int na, int nb, int nc, int nd);

    /// @brief Accumulate J and K from double precision integrals
    ///
    /// Convenience overload for mixed pipelines that sometimes
    /// use double precision (e.g., adaptive mode for high AM).
    ///
    /// @param buffer Double precision integral buffer
    /// @param fa Starting basis function index for shell a
    /// @param fb Starting basis function index for shell b
    /// @param fc Starting basis function index for shell c
    /// @param fd Starting basis function index for shell d
    /// @param na Number of functions in shell a
    /// @param nb Number of functions in shell b
    /// @param nc Number of functions in shell c
    /// @param nd Number of functions in shell d
    void accumulate(const TwoElectronBuffer<0, double>& buffer,
                    Index fa, Index fb, Index fc, Index fd,
                    int na, int nb, int nc, int nd);

    /// @brief Get the Coulomb matrix J (double precision)
    [[nodiscard]] std::span<const Real> get_coulomb_matrix() const noexcept {
        return std::span<const Real>(J_);
    }

    /// @brief Get the exchange matrix K (double precision)
    [[nodiscard]] std::span<const Real> get_exchange_matrix() const noexcept {
        return std::span<const Real>(K_);
    }

    /// @brief Compute the Fock matrix F = H_core + J - exchange_fraction * K
    /// @param H_core Core Hamiltonian matrix (row-major, nbf x nbf)
    /// @param exchange_fraction Fraction of exact exchange (1.0 for RHF)
    /// @return Vector containing the Fock matrix (row-major, nbf x nbf)
    [[nodiscard]] std::vector<Real> get_fock_matrix(
        std::span<const Real> H_core,
        Real exchange_fraction = 1.0) const;

    /// @brief Reset J and K matrices to zero
    void reset() noexcept;

    /// @brief Get the number of basis functions
    [[nodiscard]] Size nbf() const noexcept { return nbf_; }

    /// @brief Get the current mixed precision mode
    [[nodiscard]] MixedPrecisionMode mode() const noexcept {
        return MixedPrecisionMode::Compute32Accumulate64;
    }

    /// @brief Get statistics: number of float32 accumulations
    [[nodiscard]] Size n_float32_accumulations() const noexcept {
        return n_float32_accumulations_;
    }

    /// @brief Get statistics: number of float64 accumulations
    [[nodiscard]] Size n_float64_accumulations() const noexcept {
        return n_float64_accumulations_;
    }

private:
    Size nbf_;                    ///< Number of basis functions
    std::vector<Real> J_;         ///< Coulomb matrix (double precision)
    std::vector<Real> K_;         ///< Exchange matrix (double precision)
    const Real* D_{nullptr};      ///< Pointer to density matrix

    // Statistics
    Size n_float32_accumulations_{0};
    Size n_float64_accumulations_{0};

    /// @brief Template implementation for accumulate
    template<typename RealType>
    void accumulate_impl(const TwoElectronBuffer<0, RealType>& buffer,
                         Index fa, Index fb, Index fc, Index fd,
                         int na, int nb, int nc, int nd);
};

}  // namespace libaccint::consumers
