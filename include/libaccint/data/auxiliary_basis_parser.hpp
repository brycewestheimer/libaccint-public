// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file auxiliary_basis_parser.hpp
/// @brief Auxiliary basis set parser supporting Gaussian94 (.gbs) and BSE JSON formats

#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/core/types.hpp>

#include <string>
#include <vector>

namespace libaccint::data {

/// @brief File format for auxiliary basis set files
enum class AuxBasisFileFormat {
    Auto,        ///< Detect from file extension
    Gaussian94,  ///< Gaussian94 .gbs format
    BSE_JSON     ///< Basis Set Exchange JSON (QCSchema) format
};

/// @brief Parse an auxiliary basis set from file
///
/// Supports both Gaussian94 (.gbs) and BSE JSON formats.
/// Format is auto-detected from file extension unless specified.
///
/// @param file_path Path to the auxiliary basis set file
/// @param atoms Vector of atoms with atomic numbers and positions
/// @param format File format (Auto = detect from extension)
/// @param fitting_type Fitting type for the auxiliary basis
/// @return AuxiliaryBasisSet constructed from the parsed data
/// @throws InvalidArgumentException if file not found, malformed, or missing elements
[[nodiscard]] AuxiliaryBasisSet parse_auxiliary_basis_file(
    const std::string& file_path,
    const std::vector<Atom>& atoms,
    AuxBasisFileFormat format = AuxBasisFileFormat::Auto,
    FittingType fitting_type = FittingType::JKFIT);

/// @brief Parse an auxiliary basis set from Gaussian94 format string
///
/// Gaussian94 format:
/// ```
/// ****
/// H     0
/// S   3   1.00
///       9.0000000    0.0000000
///       1.0000000    0.0000000
///       0.2500000    1.0000000
/// ****
/// ```
///
/// @param content String containing Gaussian94 format data
/// @param atoms Vector of atoms with atomic numbers and positions
/// @param fitting_type Fitting type
/// @param name Optional basis set name
/// @return AuxiliaryBasisSet
[[nodiscard]] AuxiliaryBasisSet parse_auxiliary_basis_gaussian94(
    const std::string& content,
    const std::vector<Atom>& atoms,
    FittingType fitting_type = FittingType::JKFIT,
    const std::string& name = "");

/// @brief Parse an auxiliary basis set from BSE JSON string
///
/// @param json_content String containing BSE JSON data
/// @param atoms Vector of atoms with atomic numbers and positions
/// @param fitting_type Fitting type
/// @param name Optional basis set name
/// @return AuxiliaryBasisSet
[[nodiscard]] AuxiliaryBasisSet parse_auxiliary_basis_json(
    const std::string& json_content,
    const std::vector<Atom>& atoms,
    FittingType fitting_type = FittingType::JKFIT,
    const std::string& name = "");

/// @brief Validate auxiliary basis set data
///
/// Checks:
///   - All exponents are positive
///   - Coefficients are non-zero for at least one primitive per shell
///   - Angular momentum is within bounds
///   - Shell centers match atom positions
///
/// @param aux_basis The auxiliary basis set to validate
/// @return true if valid
/// @throws InvalidArgumentException with details if invalid
[[nodiscard]] bool validate_auxiliary_basis(const AuxiliaryBasisSet& aux_basis);

/// @brief Map element symbol to atomic number
/// @param symbol Element symbol (e.g., "H", "He", "Li")
/// @return Atomic number (1-118)
/// @throws InvalidArgumentException if symbol is unknown
[[nodiscard]] int element_symbol_to_z(const std::string& symbol);

/// @brief Map atomic number to element symbol
/// @param z Atomic number (1-118)
/// @return Element symbol
/// @throws InvalidArgumentException if z is out of range
[[nodiscard]] std::string z_to_element_symbol(int z);

}  // namespace libaccint::data
