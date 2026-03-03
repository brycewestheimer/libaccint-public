// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file builtin_basis.hpp
/// @brief Hard-coded basis set data and factory functions

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/core/types.hpp>
#include <string>
#include <vector>

namespace libaccint::data {

/// @brief Atom specification for basis set construction
struct Atom {
    int atomic_number;  ///< Atomic number (Z)
    Point3D position;   ///< Position in Bohr (atomic units)
};

/// @brief Create an STO-3G basis set for the given atoms
///
/// Hard-coded STO-3G exponents and contraction coefficients for
/// H (Z=1), C (Z=6), N (Z=7), O (Z=8), F (Z=9).
///
/// Shell structure:
///   - H: 1s (3 primitives)
///   - C, N, O, F: 1s, 2sp (inner s + outer s + p), each 3 primitives
///
/// @param atoms Vector of atoms with atomic numbers and positions
/// @return BasisSet containing all shells for the given atoms
/// @throws InvalidArgumentException if an unsupported element is encountered
[[nodiscard]] BasisSet create_sto3g(const std::vector<Atom>& atoms);

/// @brief Create a named built-in basis set
///
/// Currently only "sto-3g" is supported as a built-in.
///
/// @param name Basis set name (case-insensitive)
/// @param atoms Vector of atoms
/// @return BasisSet for the given atoms
/// @throws InvalidArgumentException if name is not a supported built-in
[[nodiscard]] BasisSet create_builtin_basis(const std::string& name,
                                             const std::vector<Atom>& atoms);

}  // namespace libaccint::data
