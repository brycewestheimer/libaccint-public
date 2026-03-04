// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file auxiliary_basis_data.hpp
/// @brief Built-in auxiliary basis set data for density fitting
///
/// Embeds commonly used auxiliary basis sets so users can load standard
/// fitting bases without external files. Supported bases:
///   - cc-pVDZ-RI (Weigend, PCCP 4, 4285, 2002)
///   - cc-pVTZ-RI
///   - def2-SVP-JKFIT (Weigend, JCIM 46, 1804, 2006)
///   - def2-TZVP-JKFIT

#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/core/types.hpp>

#include <string>
#include <vector>

namespace libaccint::data {

/// @brief Create a built-in auxiliary basis set for the given atoms
///
/// @param name Auxiliary basis name (case-insensitive):
///   "cc-pVDZ-RI", "cc-pVTZ-RI", "def2-SVP-JKFIT", "def2-TZVP-JKFIT"
/// @param atoms Vector of atoms with atomic numbers and positions
/// @return AuxiliaryBasisSet constructed from embedded data
/// @throws InvalidArgumentException if name unknown or element unsupported
[[nodiscard]] AuxiliaryBasisSet create_builtin_auxiliary_basis(
    const std::string& name,
    const std::vector<Atom>& atoms);

/// @brief List all available built-in auxiliary basis sets
/// @return Vector of available auxiliary basis set names
[[nodiscard]] std::vector<std::string> list_builtin_auxiliary_bases();

/// @brief Check if a built-in auxiliary basis is available for given elements
/// @param name Auxiliary basis name
/// @param atomic_numbers Atomic numbers to check
/// @return true if available for all specified elements
[[nodiscard]] bool is_builtin_auxiliary_available(
    const std::string& name,
    const std::vector<int>& atomic_numbers);

}  // namespace libaccint::data
