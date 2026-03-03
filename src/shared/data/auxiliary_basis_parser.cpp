// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file auxiliary_basis_parser.cpp
/// @brief Implementation of auxiliary basis set parser (Gaussian94 + BSE JSON)

#include <libaccint/data/auxiliary_basis_parser.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>
#include <unordered_map>

namespace libaccint::data {

namespace {

// ============================================================================
// Element symbol tables
// ============================================================================

// Element symbols indexed by atomic number (0 = dummy, 1-118 = H through Og)
constexpr std::array<const char*, 119> kElementSymbols = {
    "X",                                                                    // 0: dummy
    "H",  "He",                                                             // 1-2
    "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",                         // 3-10
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",                         // 11-18
    "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", // 19-30
    "Ga", "Ge", "As", "Se", "Br", "Kr",                                     // 31-36
    "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", // 37-48
    "In", "Sn", "Sb", "Te", "I",  "Xe",                                     // 49-54
    "Cs", "Ba",                                                             // 55-56
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",             // 57-66
    "Ho", "Er", "Tm", "Yb", "Lu",                                           // 67-71
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",                   // 72-80
    "Tl", "Pb", "Bi", "Po", "At", "Rn",                                     // 81-86
    "Fr", "Ra",                                                             // 87-88
    "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf",             // 89-98
    "Es", "Fm", "Md", "No", "Lr",                                           // 99-103
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",                   // 104-112
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og"                                      // 113-118
};

/// Look up atomic number from symbol (case-insensitive)
int symbol_to_z_lookup(const std::string& sym) {
    std::string upper = sym;
    if (!upper.empty()) {
        upper[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(upper[0])));
        for (size_t i = 1; i < upper.size(); ++i) {
            upper[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(upper[i])));
        }
    }

    for (int z = 1; z < static_cast<int>(kElementSymbols.size()); ++z) {
        if (upper == kElementSymbols[static_cast<size_t>(z)]) {
            return z;
        }
    }
    return -1;
}

/// @brief Angular momentum letter to integer
int am_letter_to_int(char c) {
    switch (std::toupper(static_cast<unsigned char>(c))) {
        case 'S': return 0;
        case 'P': return 1;
        case 'D': return 2;
        case 'F': return 3;
        case 'G': return 4;
        case 'H': return 5;
        case 'I': return 6;
        default: return -1;
    }
}

/// @brief Trim whitespace from both ends
std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    auto end = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}



/// @brief Parse Gaussian94 format for a single element block
void parse_g94_element_block(
    const std::vector<std::string>& lines,
    size_t& pos,
    int target_z,
    const Point3D& center,
    Index atom_idx,
    std::vector<Shell>& shells) {

    // Lines should start after the element header "SYMBOL 0"
    while (pos < lines.size()) {
        std::string line = trim(lines[pos]);

        // End of element block
        if (line == "****" || line.empty()) {
            break;
        }

        // Parse shell header: "AM  NPRIM  SCALE"
        // e.g., "S   3   1.00" or "SP  3   1.00"
        std::istringstream header_ss(line);
        std::string am_str;
        int n_prim = 0;
        double scale = 1.0;
        header_ss >> am_str >> n_prim >> scale;

        if (am_str.empty() || n_prim <= 0) {
            throw InvalidArgumentException(
                "Malformed Gaussian94 shell header: '" + line + "'");
        }

        ++pos;

        // Read primitive data
        std::vector<Real> exponents;
        std::vector<Real> coefficients;
        std::vector<Real> sp_coefficients;  // second column for SP shells
        bool is_sp = (am_str == "SP" || am_str == "sp" || am_str == "Sp");

        for (int p = 0; p < n_prim && pos < lines.size(); ++p, ++pos) {
            std::string prim_line = trim(lines[pos]);
            // Replace D/d with E for Fortran-style exponents
            std::replace(prim_line.begin(), prim_line.end(), 'D', 'E');
            std::replace(prim_line.begin(), prim_line.end(), 'd', 'e');

            std::istringstream prim_ss(prim_line);
            double exp_val = 0.0;
            double coeff_val = 0.0;
            prim_ss >> exp_val >> coeff_val;
            exponents.push_back(exp_val);
            coefficients.push_back(coeff_val);

            if (is_sp) {
                double sp_coeff = 0.0;
                prim_ss >> sp_coeff;
                sp_coefficients.push_back(sp_coeff);
            }
        }

        if (is_sp) {
            // Create separate s and p shells
            shells.emplace_back(0, center, exponents, coefficients);
            shells.back().set_atom_index(atom_idx);
            shells.emplace_back(1, center, exponents, sp_coefficients);
            shells.back().set_atom_index(atom_idx);
        } else {
            int am = am_letter_to_int(am_str[0]);
            if (am < 0) {
                throw InvalidArgumentException(
                    "Unknown angular momentum '" + am_str + "' in Gaussian94 format");
            }
            shells.emplace_back(am, center, exponents, coefficients);
            shells.back().set_atom_index(atom_idx);
        }
    }
}

/// @brief Parse all element blocks from Gaussian94 content
std::unordered_map<int, std::vector<Shell>> parse_g94_all_elements(
    const std::string& content) {

    std::unordered_map<int, std::vector<Shell>> element_shells;

    // Split into lines
    std::vector<std::string> lines;
    std::istringstream iss(content);
    std::string line;
    while (std::getline(iss, line)) {
        lines.push_back(line);
    }

    // Dummy center for template shells
    Point3D dummy_center{0.0, 0.0, 0.0};

    size_t pos = 0;
    while (pos < lines.size()) {
        std::string trimmed = trim(lines[pos]);

        // Skip comments and empty lines
        if (trimmed.empty() || trimmed[0] == '!' || trimmed[0] == '#') {
            ++pos;
            continue;
        }

        // Skip separators
        if (trimmed == "****") {
            ++pos;
            continue;
        }

        // Try to parse element header: "SYMBOL 0"
        std::istringstream hdr(trimmed);
        std::string symbol;
        int zero = -1;
        hdr >> symbol >> zero;

        if (zero == 0 && !symbol.empty()) {
            int z = symbol_to_z_lookup(symbol);
            if (z > 0) {
                ++pos;
                std::vector<Shell> shells;
                parse_g94_element_block(lines, pos, z, dummy_center, 0, shells);
                element_shells[z] = std::move(shells);
                continue;
            }
        }

        ++pos;
    }

    return element_shells;
}

}  // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

int element_symbol_to_z(const std::string& symbol) {
    int z = symbol_to_z_lookup(symbol);
    if (z < 0) {
        throw InvalidArgumentException("Unknown element symbol: '" + symbol + "'");
    }
    return z;
}

std::string z_to_element_symbol(int z) {
    if (z < 1 || z >= static_cast<int>(kElementSymbols.size())) {
        throw InvalidArgumentException(
            "Atomic number out of range: " + std::to_string(z));
    }
    return kElementSymbols[static_cast<size_t>(z)];
}

AuxiliaryBasisSet parse_auxiliary_basis_file(
    const std::string& file_path,
    const std::vector<Atom>& atoms,
    AuxBasisFileFormat format,
    FittingType fitting_type) {

    // Read entire file
    std::ifstream ifs(file_path);
    if (!ifs.is_open()) {
        throw InvalidArgumentException(
            "Cannot open auxiliary basis file: " + file_path);
    }
    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());

    // Detect format from extension if Auto
    if (format == AuxBasisFileFormat::Auto) {
        std::filesystem::path p(file_path);
        auto ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        if (ext == ".gbs" || ext == ".g94") {
            format = AuxBasisFileFormat::Gaussian94;
        } else if (ext == ".json") {
            format = AuxBasisFileFormat::BSE_JSON;
        } else {
            throw InvalidArgumentException(
                "Cannot detect auxiliary basis file format from extension: " + ext);
        }
    }

    // Extract basis name from filename
    std::filesystem::path p(file_path);
    std::string name = p.stem().string();

    if (format == AuxBasisFileFormat::Gaussian94) {
        return parse_auxiliary_basis_gaussian94(content, atoms, fitting_type, name);
    } else {
        return parse_auxiliary_basis_json(content, atoms, fitting_type, name);
    }
}

AuxiliaryBasisSet parse_auxiliary_basis_gaussian94(
    const std::string& content,
    const std::vector<Atom>& atoms,
    FittingType fitting_type,
    const std::string& name) {

    auto element_shells = parse_g94_all_elements(content);

    std::vector<Shell> all_shells;
    for (Size i = 0; i < atoms.size(); ++i) {
        int z = atoms[i].atomic_number;
        auto it = element_shells.find(z);
        if (it == element_shells.end()) {
            throw InvalidArgumentException(
                "Gaussian94 auxiliary basis does not contain data for element Z=" +
                std::to_string(z));
        }

        // Clone shells with correct center and atom index
        for (const auto& template_shell : it->second) {
            all_shells.emplace_back(
                template_shell.angular_momentum(),
                atoms[i].position,
                std::vector<Real>(template_shell.exponents().begin(),
                                  template_shell.exponents().end()),
                std::vector<Real>(template_shell.coefficients().begin(),
                                  template_shell.coefficients().end()));
            all_shells.back().set_atom_index(static_cast<Index>(i));
        }
    }

    return AuxiliaryBasisSet(std::move(all_shells), fitting_type, name);
}

AuxiliaryBasisSet parse_auxiliary_basis_json(
    const std::string& json_content,
    const std::vector<Atom>& atoms,
    FittingType fitting_type,
    const std::string& name) {

    nlohmann::json basis_json;
    try {
        basis_json = nlohmann::json::parse(json_content);
    } catch (const nlohmann::json::parse_error& e) {
        throw InvalidArgumentException(
            std::string("Malformed auxiliary basis JSON: ") + e.what());
    }

    if (!basis_json.contains("elements")) {
        throw InvalidArgumentException(
            "Auxiliary basis JSON missing 'elements' object");
    }

    const auto& elements = basis_json["elements"];
    std::vector<Shell> all_shells;

    for (Size i = 0; i < atoms.size(); ++i) {
        const std::string z_str = std::to_string(atoms[i].atomic_number);
        if (!elements.contains(z_str)) {
            throw InvalidArgumentException(
                "Auxiliary basis JSON missing element Z=" + z_str);
        }

        const auto& elem = elements[z_str];
        if (!elem.contains("electron_shells")) {
            throw InvalidArgumentException(
                "Auxiliary basis element Z=" + z_str + " missing 'electron_shells'");
        }

        for (const auto& shell_entry : elem["electron_shells"]) {
            const auto& am_array = shell_entry.at("angular_momentum");

            std::vector<Real> exponents;
            for (const auto& exp_str : shell_entry.at("exponents")) {
                exponents.push_back(std::stod(exp_str.get<std::string>()));
            }

            const auto& coeff_rows = shell_entry.at("coefficients");

            if (am_array.size() == 1) {
                int am = am_array[0].get<int>();
                for (Size row = 0; row < coeff_rows.size(); ++row) {
                    std::vector<Real> coefficients;
                    bool all_zero = true;
                    for (const auto& c_str : coeff_rows[row]) {
                        Real c = std::stod(c_str.get<std::string>());
                        coefficients.push_back(c);
                        if (c != 0.0) { all_zero = false; }
                    }
                    if (all_zero) { continue; }
                    all_shells.emplace_back(am, atoms[i].position,
                                            exponents, coefficients);
                    all_shells.back().set_atom_index(static_cast<Index>(i));
                }
            } else {
                // General contraction or SP
                for (Size row = 0; row < am_array.size() && row < coeff_rows.size(); ++row) {
                    int am = am_array[row].get<int>();
                    std::vector<Real> coefficients;
                    bool all_zero = true;
                    for (const auto& c_str : coeff_rows[row]) {
                        Real c = std::stod(c_str.get<std::string>());
                        coefficients.push_back(c);
                        if (c != 0.0) { all_zero = false; }
                    }
                    if (all_zero) { continue; }
                    all_shells.emplace_back(am, atoms[i].position,
                                            exponents, coefficients);
                    all_shells.back().set_atom_index(static_cast<Index>(i));
                }
            }
        }
    }

    return AuxiliaryBasisSet(std::move(all_shells), fitting_type,
                              name.empty() ? "unknown" : name);
}

bool validate_auxiliary_basis(const AuxiliaryBasisSet& aux_basis) {
    if (aux_basis.empty()) {
        throw InvalidArgumentException("Auxiliary basis set is empty");
    }

    for (Size i = 0; i < aux_basis.n_shells(); ++i) {
        const auto& shell = aux_basis.shell(i);

        // Check angular momentum bounds
        if (shell.angular_momentum() < 0 ||
            shell.angular_momentum() > MAX_ANGULAR_MOMENTUM) {
            throw InvalidArgumentException(
                "Shell " + std::to_string(i) + " has invalid angular momentum: " +
                std::to_string(shell.angular_momentum()));
        }

        // Check exponents are positive
        for (Size j = 0; j < shell.n_primitives(); ++j) {
            if (shell.exponents()[j] <= 0.0) {
                throw InvalidArgumentException(
                    "Shell " + std::to_string(i) + " has non-positive exponent: " +
                    std::to_string(shell.exponents()[j]));
            }
        }

        // Check at least one non-zero coefficient
        bool has_nonzero = false;
        for (Size j = 0; j < shell.n_primitives(); ++j) {
            if (shell.coefficients()[j] != 0.0) {
                has_nonzero = true;
                break;
            }
        }
        if (!has_nonzero) {
            throw InvalidArgumentException(
                "Shell " + std::to_string(i) + " has all-zero coefficients");
        }
    }

    return true;
}

}  // namespace libaccint::data
