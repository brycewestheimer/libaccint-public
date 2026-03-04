// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file primitive_pair_data.hpp
/// @brief Pre-computed Gaussian product data for primitive pairs in SoA layout
///
/// PrimitivePairData caches the results of compute_gaussian_product() for all
/// primitive pairs (p, q) across two ShellSets. This avoids recomputing these
/// quantities every time a kernel needs them. The data is stored in SoA layout
/// for efficient vectorized access and GPU compatibility.
///
/// For a ShellSetPair with n_shells_a * n_shells_b shell pairs, each having
/// K_a * K_b primitive pairs, the total number of primitive pairs is:
///   n_pairs = n_shells_a * n_shells_b * K_a * K_b
///
/// Indexing: For shell pair (i, j) with primitive pair (p, q):
///   flat_index = ((i * n_shells_b + j) * K_a + p) * K_b + q

#include <libaccint/core/types.hpp>

#include <vector>

namespace libaccint {

/// @brief Pre-computed Gaussian product data for all primitive pairs in a ShellSetPair
///
/// Stores the results of the Gaussian product theorem for all primitive
/// combinations across two ShellSets, enabling O(1) lookup during integral
/// computation instead of re-deriving these quantities each time.
///
/// All arrays are indexed by a flat primitive-pair index. Use pair_index()
/// to compute the flat index from shell and primitive indices.
struct PrimitivePairData {
    // Product center coordinates (P = (alpha*A + beta*B) / zeta)
    std::vector<Real> Px;               ///< x-coordinates of product centers
    std::vector<Real> Py;               ///< y-coordinates of product centers
    std::vector<Real> Pz;               ///< z-coordinates of product centers

    // Exponent quantities
    std::vector<Real> zeta;             ///< Combined exponent zeta = alpha + beta
    std::vector<Real> one_over_2zeta;   ///< Precomputed 1 / (2 * zeta)
    std::vector<Real> mu;               ///< Reduced exponent mu = alpha*beta / zeta

    // Prefactors
    std::vector<Real> K_AB;             ///< Gaussian overlap prefactor exp(-mu*|A-B|^2)
    std::vector<Real> coeff_product;    ///< Product of contraction coefficients c_a * c_b

    // Displacements from product center to shell centers
    std::vector<Real> PA_x;             ///< P_x - A_x
    std::vector<Real> PA_y;             ///< P_y - A_y
    std::vector<Real> PA_z;             ///< P_z - A_z
    std::vector<Real> PB_x;             ///< P_x - B_x
    std::vector<Real> PB_y;             ///< P_y - B_y
    std::vector<Real> PB_z;             ///< P_z - B_z

    // Dimensions for indexing
    Size n_shells_a{0};                 ///< Number of shells in ShellSet A
    Size n_shells_b{0};                 ///< Number of shells in ShellSet B
    Size K_a{0};                        ///< Number of primitives per shell in A
    Size K_b{0};                        ///< Number of primitives per shell in B
    Size n_total_pairs{0};              ///< Total number of primitive pairs

    /// @brief Compute flat index for a given shell pair and primitive pair
    /// @param shell_i Index of shell in ShellSet A
    /// @param shell_j Index of shell in ShellSet B
    /// @param prim_p Index of primitive in shell A
    /// @param prim_q Index of primitive in shell B
    /// @return Flat index into the SoA arrays
    [[nodiscard]] Size pair_index(Size shell_i, Size shell_j,
                                  Size prim_p, Size prim_q) const noexcept {
        return ((shell_i * n_shells_b + shell_j) * K_a + prim_p) * K_b + prim_q;
    }

    /// @brief Get offset to the start of primitive pairs for a given shell pair
    /// @param shell_i Index of shell in ShellSet A
    /// @param shell_j Index of shell in ShellSet B
    /// @return Flat index of the first primitive pair for this shell pair
    [[nodiscard]] Size shell_pair_offset(Size shell_i, Size shell_j) const noexcept {
        return (shell_i * n_shells_b + shell_j) * K_a * K_b;
    }

    /// @brief Number of primitive pairs per shell pair (K_a * K_b)
    [[nodiscard]] Size primitives_per_shell_pair() const noexcept {
        return K_a * K_b;
    }

    /// @brief Check if this data has been populated
    [[nodiscard]] bool empty() const noexcept {
        return n_total_pairs == 0;
    }

    /// @brief Allocate storage for the given dimensions
    void allocate(Size n_shells_a_, Size n_shells_b_, Size K_a_, Size K_b_) {
        n_shells_a = n_shells_a_;
        n_shells_b = n_shells_b_;
        K_a = K_a_;
        K_b = K_b_;
        n_total_pairs = n_shells_a * n_shells_b * K_a * K_b;

        Px.resize(n_total_pairs);
        Py.resize(n_total_pairs);
        Pz.resize(n_total_pairs);
        zeta.resize(n_total_pairs);
        one_over_2zeta.resize(n_total_pairs);
        mu.resize(n_total_pairs);
        K_AB.resize(n_total_pairs);
        coeff_product.resize(n_total_pairs);
        PA_x.resize(n_total_pairs);
        PA_y.resize(n_total_pairs);
        PA_z.resize(n_total_pairs);
        PB_x.resize(n_total_pairs);
        PB_y.resize(n_total_pairs);
        PB_z.resize(n_total_pairs);
    }
};

}  // namespace libaccint
