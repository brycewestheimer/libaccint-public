// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bse_json_parser.cpp
/// @brief Implementation of BSE JSON parser

#include <libaccint/data/bse_json_parser.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace libaccint::data {

using json = nlohmann::json;

namespace {

/// @brief Parse shells for a single element from BSE JSON element data
std::vector<Shell> parse_element_shells(const json& element_data,
                                         const Point3D& center) {
    std::vector<Shell> shells;

    if (!element_data.contains("electron_shells")) {
        return shells;
    }

    for (const auto& shell_data : element_data["electron_shells"]) {
        // Parse exponents (stored as strings in BSE JSON)
        std::vector<Real> exponents;
        for (const auto& exp_str : shell_data["exponents"]) {
            exponents.push_back(std::stod(exp_str.get<std::string>()));
        }

        // Parse angular momentum array
        std::vector<int> am_list;
        for (const auto& am : shell_data["angular_momentum"]) {
            am_list.push_back(am.get<int>());
        }

        // Parse coefficient columns
        const auto& coeff_array = shell_data["coefficients"];

        if (am_list.size() == 1) {
            // Standard shell: single angular momentum, possibly general contraction
            int am = am_list[0];
            for (const auto& coeff_col : coeff_array) {
                std::vector<Real> coefficients;
                bool all_zero = true;
                for (const auto& c_str : coeff_col) {
                    double val = std::stod(c_str.get<std::string>());
                    coefficients.push_back(val);
                    if (val != 0.0) all_zero = false;
                }
                // Skip zero-coefficient columns (general contraction padding)
                if (!all_zero) {
                    shells.emplace_back(am, center, exponents, coefficients);
                }
            }
        } else {
            // SP shell or general contraction with multiple angular momenta
            // Each angular momentum gets its own coefficient column
            for (size_t col = 0; col < am_list.size() && col < coeff_array.size(); ++col) {
                int am = am_list[col];
                std::vector<Real> coefficients;
                bool all_zero = true;
                for (const auto& c_str : coeff_array[col]) {
                    double val = std::stod(c_str.get<std::string>());
                    coefficients.push_back(val);
                    if (val != 0.0) all_zero = false;
                }
                if (!all_zero) {
                    shells.emplace_back(am, center, exponents, coefficients);
                }
            }
        }
    }

    return shells;
}

}  // anonymous namespace

BasisSet BseJsonParser::parse(const std::string& json_string,
                               const std::vector<Atom>& atoms) {
    json j;
    try {
        j = json::parse(json_string);
    } catch (const json::parse_error& e) {
        throw InvalidArgumentException(
            "Failed to parse BSE JSON: " + std::string(e.what()));
    }

    // Validate schema type
    if (j.contains("molssi_bse_schema")) {
        const auto& schema = j["molssi_bse_schema"];
        if (schema.contains("schema_type")) {
            std::string schema_type = schema["schema_type"].get<std::string>();
            if (schema_type != "complete" && schema_type != "minimal") {
                throw InvalidArgumentException(
                    "Unsupported BSE schema type: " + schema_type +
                    " (expected 'complete' or 'minimal')");
            }
        }
    }

    if (!j.contains("elements")) {
        throw InvalidArgumentException(
            "BSE JSON missing 'elements' key");
    }

    const auto& elements = j["elements"];
    std::vector<Shell> all_shells;

    for (size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
        const auto& atom = atoms[atom_idx];
        std::string z_str = std::to_string(atom.atomic_number);

        if (!elements.contains(z_str)) {
            throw InvalidArgumentException(
                "BSE JSON has no data for element Z=" + z_str);
        }

        auto shells = parse_element_shells(elements[z_str], atom.position);
        for (auto& shell : shells) {
            shell.set_atom_index(static_cast<Index>(atom_idx));
            all_shells.push_back(std::move(shell));
        }
    }

    return BasisSet(std::move(all_shells));
}

BasisSet BseJsonParser::parse_file(const std::string& file_path,
                                    const std::vector<Atom>& atoms) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw InvalidArgumentException(
            "Cannot open BSE JSON file: " + file_path);
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    return parse(ss.str(), atoms);
}

std::vector<std::string> BseJsonParser::validate(const std::string& json_string) {
    std::vector<std::string> errors;

    json j;
    try {
        j = json::parse(json_string);
    } catch (const json::parse_error& e) {
        errors.push_back("JSON parse error: " + std::string(e.what()));
        return errors;
    }

    // Check for required top-level keys
    if (!j.contains("elements")) {
        errors.push_back("Missing required key: 'elements'");
    }

    // Validate schema if present
    if (j.contains("molssi_bse_schema")) {
        const auto& schema = j["molssi_bse_schema"];
        if (!schema.contains("schema_type")) {
            errors.push_back("Missing 'schema_type' in 'molssi_bse_schema'");
        }
    }

    // Validate elements
    if (j.contains("elements") && j["elements"].is_object()) {
        for (const auto& [z_str, element_data] : j["elements"].items()) {
            // Check element key is a valid integer
            try {
                int z = std::stoi(z_str);
                if (z < 1 || z > 118) {
                    errors.push_back("Invalid atomic number: " + z_str);
                }
            } catch (...) {
                errors.push_back("Non-integer element key: " + z_str);
                continue;
            }

            // Check for electron_shells
            if (!element_data.contains("electron_shells")) {
                errors.push_back("Element " + z_str + " missing 'electron_shells'");
                continue;
            }

            for (size_t i = 0; i < element_data["electron_shells"].size(); ++i) {
                const auto& shell = element_data["electron_shells"][i];
                std::string prefix = "Element " + z_str + " shell " + std::to_string(i) + ": ";

                if (!shell.contains("angular_momentum")) {
                    errors.push_back(prefix + "missing 'angular_momentum'");
                }
                if (!shell.contains("exponents")) {
                    errors.push_back(prefix + "missing 'exponents'");
                }
                if (!shell.contains("coefficients")) {
                    errors.push_back(prefix + "missing 'coefficients'");
                }

                // Check exponent/coefficient size consistency
                if (shell.contains("exponents") && shell.contains("coefficients")) {
                    size_t n_exp = shell["exponents"].size();
                    for (size_t c = 0; c < shell["coefficients"].size(); ++c) {
                        if (shell["coefficients"][c].size() != n_exp) {
                            errors.push_back(prefix + "coefficient column " +
                                             std::to_string(c) +
                                             " size mismatch with exponents");
                        }
                    }
                }
            }
        }
    }

    return errors;
}

std::string BseJsonParser::get_name(const std::string& json_string) {
    json j;
    try {
        j = json::parse(json_string);
    } catch (const json::parse_error& e) {
        throw InvalidArgumentException(
            "Failed to parse BSE JSON: " + std::string(e.what()));
    }

    if (j.contains("name")) {
        return j["name"].get<std::string>();
    }
    return "";
}

std::string BseJsonParser::get_description(const std::string& json_string) {
    json j;
    try {
        j = json::parse(json_string);
    } catch (const json::parse_error& e) {
        throw InvalidArgumentException(
            "Failed to parse BSE JSON: " + std::string(e.what()));
    }

    if (j.contains("description")) {
        return j["description"].get<std::string>();
    }
    return "";
}

std::vector<int> BseJsonParser::get_supported_elements(const std::string& json_string) {
    json j;
    try {
        j = json::parse(json_string);
    } catch (const json::parse_error& e) {
        throw InvalidArgumentException(
            "Failed to parse BSE JSON: " + std::string(e.what()));
    }

    std::vector<int> elements;
    if (j.contains("elements") && j["elements"].is_object()) {
        for (const auto& [z_str, _] : j["elements"].items()) {
            try {
                elements.push_back(std::stoi(z_str));
            } catch (...) {
                // Skip invalid keys
            }
        }
    }

    std::sort(elements.begin(), elements.end());
    return elements;
}

}  // namespace libaccint::data
