// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file basis_parser.hpp
/// @brief QCSchema JSON basis set parser

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/core/types.hpp>

#include <string>
#include <vector>

namespace libaccint::data {

/// @brief Load a basis set from a QCSchema JSON file
///
/// Parses a basis set file in the QCSchema JSON format used by the
/// Basis Set Exchange (BSE). Constructs Shell objects for each atom
/// and assembles them into a BasisSet.
///
/// Handles:
///   - Standard contracted shells with single angular momentum
///   - General contraction format (multiple coefficient columns)
///   - SP shells (angular_momentum: [0, 1]) split into separate s and p shells
///
/// @param name Basis set name (e.g., "cc-pvdz"). Used to locate the JSON file
///             in the share/basis_sets/ directory.
/// @param atoms Vector of atoms with atomic numbers and positions
/// @return BasisSet constructed from the parsed data
/// @throws InvalidArgumentException if file not found, malformed JSON,
///         or missing element data
[[nodiscard]] BasisSet load_basis_set(const std::string& name,
                                      const std::vector<Atom>& atoms);

/// @brief Load a basis set from a specific JSON file path
///
/// @param file_path Full path to the QCSchema JSON file
/// @param atoms Vector of atoms with atomic numbers and positions
/// @return BasisSet constructed from the parsed data
/// @throws InvalidArgumentException if file not found, malformed JSON,
///         or missing element data
[[nodiscard]] BasisSet load_basis_set_from_file(const std::string& file_path,
                                                 const std::vector<Atom>& atoms);

/// @brief Convert a basis set name to its JSON filename
///
/// Lowercases the name, replaces spaces with hyphens, and maps Pople star
/// notation: `**` → `_ss`, `*` → `_st`.
///
/// @param name Basis set name (e.g., "6-31G*", "cc-pVDZ")
/// @return Filename with .json extension (e.g., "6-31g_st.json", "cc-pvdz.json")
[[nodiscard]] std::string name_to_filename(const std::string& name);

/// @brief List all available bundled basis sets
///
/// Scans the data directory for .json files and returns the stem names
/// (without extension), sorted alphabetically.
///
/// @return Sorted vector of basis set names (e.g., "sto-3g", "cc-pvdz")
/// @throws InvalidArgumentException if data directory cannot be located
[[nodiscard]] std::vector<std::string> list_available_basis_sets();

}  // namespace libaccint::data
