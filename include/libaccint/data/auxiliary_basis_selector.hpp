// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file auxiliary_basis_selector.hpp
/// @brief Auxiliary basis auto-selection utility
///
/// Maps orbital basis set names to recommended auxiliary basis sets
/// following standard pairing conventions in quantum chemistry.

#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/core/types.hpp>

#include <optional>
#include <string>
#include <vector>

namespace libaccint::data {

/// @brief Recommend an auxiliary basis for a given orbital basis
///
/// Standard pairing conventions:
///   - cc-pVDZ  → cc-pVDZ-RI (Coulomb fitting) or def2-SVP-JKFIT (JK fitting)
///   - cc-pVTZ  → cc-pVTZ-RI or def2-TZVP-JKFIT
///   - cc-pVQZ  → cc-pVTZ-RI (fallback) or def2-TZVP-JKFIT
///   - def2-SVP → def2-SVP-JKFIT
///   - def2-TZVP → def2-TZVP-JKFIT
///   - def2-TZVPP → def2-TZVP-JKFIT
///   - STO-3G   → cc-pVDZ-RI (minimal fallback)
///
/// @param orbital_basis_name Name of the orbital basis set
/// @param fitting_type Desired fitting type (RI or JKFIT)
/// @return Recommended auxiliary basis name, or std::nullopt if unknown
[[nodiscard]] std::optional<std::string> recommend_auxiliary_basis(
    const std::string& orbital_basis_name,
    FittingType fitting_type = FittingType::JKFIT);

/// @brief Get all valid orbital→auxiliary pairings
/// @return Vector of pairs (orbital_name, auxiliary_name)
[[nodiscard]] std::vector<std::pair<std::string, std::string>>
    list_orbital_auxiliary_pairings();

/// @brief Check if an orbital basis has a known auxiliary pairing
/// @param orbital_basis_name Name of the orbital basis
/// @return true if a recommended auxiliary basis exists
[[nodiscard]] bool has_recommended_auxiliary(
    const std::string& orbital_basis_name);

}  // namespace libaccint::data
