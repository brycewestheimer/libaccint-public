// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file auxiliary_basis_selector.cpp
/// @brief Auxiliary basis auto-selection implementation

#include <libaccint/data/auxiliary_basis_selector.hpp>
#include <libaccint/data/auxiliary_basis_data.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <utility>

namespace libaccint::data {

namespace {



/// @brief Normalize basis name to lowercase
std::string normalize(const std::string& name) {
    std::string result = name;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

/// @brief Pairing entry: orbital → (RI auxiliary, JKFIT auxiliary)
struct PairingEntry {
    std::string orbital;
    std::string ri_aux;
    std::string jkfit_aux;
};

/// @brief Get the orbital→auxiliary pairing table
const std::vector<PairingEntry>& get_pairing_table() {
    static const std::vector<PairingEntry> table = {
        // Dunning correlation-consistent
        {"cc-pvdz",       "cc-pVDZ-RI",       "def2-SVP-JKFIT"},
        {"cc-pvtz",       "cc-pVTZ-RI",       "def2-TZVP-JKFIT"},
        {"cc-pvqz",       "cc-pVTZ-RI",       "def2-TZVP-JKFIT"},
        {"aug-cc-pvdz",   "cc-pVDZ-RI",       "def2-SVP-JKFIT"},
        {"aug-cc-pvtz",   "cc-pVTZ-RI",       "def2-TZVP-JKFIT"},

        // Ahlrichs / Karlsruhe
        {"def2-svp",      "def2-SVP-JKFIT",  "def2-SVP-JKFIT"},
        {"def2-tzvp",     "def2-TZVP-JKFIT", "def2-TZVP-JKFIT"},
        {"def2-tzvpp",    "def2-TZVP-JKFIT", "def2-TZVP-JKFIT"},
        {"def2-sv(p)",    "def2-SVP-JKFIT",  "def2-SVP-JKFIT"},

        // Minimal basis fallback
        {"sto-3g",        "cc-pVDZ-RI",   "def2-SVP-JKFIT"},
        {"sto3g",         "cc-pVDZ-RI",   "def2-SVP-JKFIT"},

        // 6-31G family
        {"6-31g",         "cc-pVDZ-RI",   "def2-SVP-JKFIT"},
        {"6-31g*",        "cc-pVDZ-RI",   "def2-SVP-JKFIT"},
        {"6-31g**",       "cc-pVDZ-RI",   "def2-SVP-JKFIT"},
        {"6-311g",        "cc-pVTZ-RI",   "def2-TZVP-JKFIT"},
        {"6-311g*",       "cc-pVTZ-RI",   "def2-TZVP-JKFIT"},
        {"6-311g**",      "cc-pVTZ-RI",   "def2-TZVP-JKFIT"},
    };
    return table;
}

bool auxiliary_basis_is_builtin(const std::string& name) {
    const auto available = list_builtin_auxiliary_bases();
    return std::any_of(available.begin(), available.end(),
                       [&name](const std::string& candidate) {
                           return normalize(candidate) == normalize(name);
                       });
}

}  // anonymous namespace

std::optional<std::string> recommend_auxiliary_basis(
    const std::string& orbital_basis_name,
    FittingType fitting_type) {

    std::string normalized = normalize(orbital_basis_name);

    for (const auto& entry : get_pairing_table()) {
        if (entry.orbital == normalized) {
            const bool want_ri = (fitting_type == FittingType::RI ||
                                  fitting_type == FittingType::RIFIT ||
                                  fitting_type == FittingType::MP2FIT);
            const std::string& selected = want_ri ? entry.ri_aux : entry.jkfit_aux;
            if (auxiliary_basis_is_builtin(selected)) {
                return selected;
            }
            return std::nullopt;
        }
    }

    return std::nullopt;
}

std::vector<std::pair<std::string, std::string>>
    list_orbital_auxiliary_pairings() {

    std::vector<std::pair<std::string, std::string>> result;
    for (const auto& entry : get_pairing_table()) {
        if (auxiliary_basis_is_builtin(entry.jkfit_aux)) {
            result.emplace_back(entry.orbital, entry.jkfit_aux);
        }
    }
    return result;
}

bool has_recommended_auxiliary(const std::string& orbital_basis_name) {
    return recommend_auxiliary_basis(orbital_basis_name).has_value();
}

}  // namespace libaccint::data
