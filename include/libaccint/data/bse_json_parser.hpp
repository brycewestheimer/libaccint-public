// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file bse_json_parser.hpp
/// @brief Parser for Basis Set Exchange (BSE) JSON format
///
/// Parses basis set files in the MolSSI BSE JSON schema format into
/// libaccint Shell and BasisSet objects. Supports:
///   - Standard contracted shells with single angular momentum
///   - General contraction format (multiple coefficient columns)
///   - SP shells (angular_momentum: [0, 1]) split into separate s and p shells
///   - All elements in the periodic table
///
/// The BSE JSON schema is documented at:
///   https://www.basissetexchange.org

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/core/types.hpp>

#include <string>
#include <vector>

namespace libaccint::data {

/// @brief Parser for Basis Set Exchange JSON format
///
/// This class provides static methods to parse BSE JSON data into
/// libaccint BasisSet objects. It handles the standard MolSSI BSE
/// schema (version 0.1) used by https://www.basissetexchange.org.
///
/// Usage:
/// @code
///   // Parse from a JSON string
///   std::string json_data = read_file("cc-pvdz.json");
///   auto basis = BseJsonParser::parse(json_data, atoms);
///
///   // Parse from a file path
///   auto basis = BseJsonParser::parse_file("/path/to/cc-pvdz.json", atoms);
///
///   // Validate JSON before parsing
///   auto errors = BseJsonParser::validate(json_data);
/// @endcode
class BseJsonParser {
public:
    /// @brief Parse BSE JSON string into a BasisSet
    ///
    /// @param json_string The JSON content as a string
    /// @param atoms Vector of atoms with atomic numbers and positions
    /// @return BasisSet constructed from the parsed data
    /// @throws InvalidArgumentException if JSON is malformed or missing data
    [[nodiscard]] static BasisSet parse(const std::string& json_string,
                                        const std::vector<Atom>& atoms);

    /// @brief Parse BSE JSON file into a BasisSet
    ///
    /// @param file_path Path to the BSE JSON file
    /// @param atoms Vector of atoms with atomic numbers and positions
    /// @return BasisSet constructed from the parsed data
    /// @throws InvalidArgumentException if file not found, malformed JSON,
    ///         or missing element data
    [[nodiscard]] static BasisSet parse_file(const std::string& file_path,
                                              const std::vector<Atom>& atoms);

    /// @brief Validate BSE JSON string
    ///
    /// Checks that the JSON string conforms to the BSE schema without
    /// constructing a BasisSet. Returns a list of validation errors.
    ///
    /// @param json_string The JSON content to validate
    /// @return Vector of error messages (empty if valid)
    [[nodiscard]] static std::vector<std::string> validate(const std::string& json_string);

    /// @brief Get the basis set name from BSE JSON
    ///
    /// @param json_string The JSON content
    /// @return The basis set name from the JSON metadata
    /// @throws InvalidArgumentException if JSON is malformed
    [[nodiscard]] static std::string get_name(const std::string& json_string);

    /// @brief Get the basis set description from BSE JSON
    ///
    /// @param json_string The JSON content
    /// @return The basis set description from the JSON metadata
    /// @throws InvalidArgumentException if JSON is malformed
    [[nodiscard]] static std::string get_description(const std::string& json_string);

    /// @brief Get list of supported element atomic numbers from BSE JSON
    ///
    /// @param json_string The JSON content
    /// @return Vector of atomic numbers for which the basis set has data
    /// @throws InvalidArgumentException if JSON is malformed
    [[nodiscard]] static std::vector<int> get_supported_elements(const std::string& json_string);
};

}  // namespace libaccint::data
