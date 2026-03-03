// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file basis_parser.cpp
/// @brief QCSchema JSON basis set parser implementation

#include <libaccint/data/basis_parser.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace libaccint::data {

namespace {

std::filesystem::path loaded_module_directory(std::error_code& ec) {
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&loaded_module_directory), &module)) {
        ec = std::error_code(static_cast<int>(GetLastError()),
                             std::system_category());
        return {};
    }

    std::wstring buffer(MAX_PATH, L'\0');
    DWORD length = 0;
    while (true) {
        length = GetModuleFileNameW(module, buffer.data(),
                                    static_cast<DWORD>(buffer.size()));
        if (length == 0) {
            ec = std::error_code(static_cast<int>(GetLastError()),
                                 std::system_category());
            return {};
        }
        if (length < buffer.size()) {
            buffer.resize(length);
            break;
        }
        buffer.resize(buffer.size() * 2);
    }

    return std::filesystem::path(buffer).parent_path();
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&loaded_module_directory), &info) == 0 ||
        info.dli_fname == nullptr) {
        ec = std::make_error_code(std::errc::no_such_file_or_directory);
        return {};
    }

    return std::filesystem::canonical(std::filesystem::path(info.dli_fname), ec)
        .parent_path();
#endif
}

/// @brief Locate the basis set data directory
///
/// Searches for the share/basis_sets/ directory:
/// 1. LIBACCINT_DATA_DIR environment variable (override, highest priority)
/// 2. LIBACCINT_INSTALL_DATADIR compile-time path (installed builds)
/// 3. CWD-relative share/basis_sets/ paths (development fallback)
std::filesystem::path find_data_directory() {
    // 1. Check environment variable first (highest priority override)
    if (const char* env = std::getenv("LIBACCINT_DATA_DIR")) {
        auto p = std::filesystem::path(env);
        if (std::filesystem::exists(p)) {
            return p;
        }
    }

    // 2. Check compile-time install prefix path
#ifdef LIBACCINT_INSTALL_DATADIR
    {
        auto install_path = std::filesystem::path(LIBACCINT_INSTALL_DATADIR);
        if (install_path.is_relative()) {
            std::error_code ec;
            auto module_dir = loaded_module_directory(ec);
            if (!ec) {
                auto resolved = module_dir / install_path;
                if (std::filesystem::exists(resolved)) {
                    return std::filesystem::canonical(resolved);
                }
            }
        }
        if (std::filesystem::exists(install_path)) {
            return std::filesystem::canonical(install_path);
        }
    }
#endif

    // 3. Try relative to current working directory (development builds)
    std::vector<std::filesystem::path> candidates = {
        "share/basis_sets",
        "../share/basis_sets",
        "../../share/basis_sets",
        "../../../share/basis_sets",
    };

    auto cwd = std::filesystem::current_path();
    for (const auto& candidate : candidates) {
        auto full = cwd / candidate;
        if (std::filesystem::exists(full)) {
            return std::filesystem::canonical(full);
        }
    }

    throw InvalidArgumentException(
        "Cannot locate basis set data directory. "
        "Set LIBACCINT_DATA_DIR environment variable or run from project root.");
}

/// @brief Parse shells for a single element from a QCSchema JSON element entry
void parse_element_shells(const nlohmann::json& element_data,
                           const Point3D& center,
                           Index atom_idx,
                           std::vector<Shell>& shells) {
    if (!element_data.contains("electron_shells")) {
        throw InvalidArgumentException(
            "QCSchema element entry missing 'electron_shells' array");
    }

    const auto& shell_list = element_data["electron_shells"];
    for (const auto& shell_entry : shell_list) {
        // Read angular momentum array
        const auto& am_array = shell_entry.at("angular_momentum");

        // Read exponents (array of strings)
        std::vector<Real> exponents;
        for (const auto& exp_str : shell_entry.at("exponents")) {
            exponents.push_back(std::stod(exp_str.get<std::string>()));
        }

        // Read coefficients (array of arrays of strings)
        // Each row is a contraction
        const auto& coeff_rows = shell_entry.at("coefficients");

        if (am_array.size() == 1) {
            // Standard shell: single angular momentum, possibly general contraction
            int am = am_array[0].get<int>();

            for (Size row = 0; row < coeff_rows.size(); ++row) {
                std::vector<Real> coefficients;
                bool all_zero = true;
                for (const auto& c_str : coeff_rows[row]) {
                    Real c = std::stod(c_str.get<std::string>());
                    coefficients.push_back(c);
                    if (c != 0.0) all_zero = false;
                }

                // Skip rows that are all zeros (unused in segmented contraction)
                if (all_zero) continue;

                // For general contractions, extract only the non-zero segment
                // But keep full vectors for now — the Shell constructor handles it
                shells.emplace_back(am, center, exponents, coefficients);
                shells.back().set_atom_index(atom_idx);
            }
        } else if (am_array.size() == 2 && am_array[0].get<int>() == 0 && am_array[1].get<int>() == 1) {
            // SP shell: split into separate s and p shells
            // First coefficient row is s, second is p
            if (coeff_rows.size() < 2) {
                throw InvalidArgumentException(
                    "SP shell requires at least 2 coefficient rows");
            }

            // s shell
            {
                std::vector<Real> s_coefficients;
                for (const auto& c_str : coeff_rows[0]) {
                    s_coefficients.push_back(std::stod(c_str.get<std::string>()));
                }
                shells.emplace_back(0, center, exponents, s_coefficients);
                shells.back().set_atom_index(atom_idx);
            }

            // p shell
            {
                std::vector<Real> p_coefficients;
                for (const auto& c_str : coeff_rows[1]) {
                    p_coefficients.push_back(std::stod(c_str.get<std::string>()));
                }
                shells.emplace_back(1, center, exponents, p_coefficients);
                shells.back().set_atom_index(atom_idx);
            }
        } else if (am_array.size() >= 2) {
            // General contraction with multiple angular momenta
            // Each row corresponds to a different angular momentum
            for (Size row = 0; row < am_array.size() && row < coeff_rows.size(); ++row) {
                int am = am_array[row].get<int>();
                std::vector<Real> coefficients;
                bool all_zero = true;
                for (const auto& c_str : coeff_rows[row]) {
                    Real c = std::stod(c_str.get<std::string>());
                    coefficients.push_back(c);
                    if (c != 0.0) all_zero = false;
                }
                if (all_zero) continue;
                shells.emplace_back(am, center, exponents, coefficients);
                shells.back().set_atom_index(atom_idx);
            }
        }
    }
}

}  // anonymous namespace

std::string name_to_filename(const std::string& name) {
    std::string result = name;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    // Replace spaces with hyphens
    std::replace(result.begin(), result.end(), ' ', '-');
    // Pople star notation: ** before * to avoid partial match
    for (std::string::size_type pos = 0;
         (pos = result.find("**", pos)) != std::string::npos;) {
        result.replace(pos, 2, "_ss");
        pos += 3;
    }
    for (std::string::size_type pos = 0;
         (pos = result.find('*', pos)) != std::string::npos;) {
        result.replace(pos, 1, "_st");
        pos += 3;
    }
    return result + ".json";
}

std::vector<std::string> list_available_basis_sets() {
    auto data_dir = find_data_directory();
    std::vector<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            names.push_back(entry.path().stem().string());
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

BasisSet load_basis_set(const std::string& name,
                         const std::vector<Atom>& atoms) {
    auto data_dir = find_data_directory();
    auto file_path = data_dir / name_to_filename(name);
    return load_basis_set_from_file(file_path.string(), atoms);
}

BasisSet load_basis_set_from_file(const std::string& file_path,
                                   const std::vector<Atom>& atoms) {
    // Read JSON file
    std::ifstream ifs(file_path);
    if (!ifs.is_open()) {
        throw InvalidArgumentException(
            "Cannot open basis set file: " + file_path);
    }

    nlohmann::json basis_json;
    try {
        ifs >> basis_json;
    } catch (const nlohmann::json::parse_error& e) {
        throw InvalidArgumentException(
            "Malformed JSON in basis set file '" + file_path + "': " + e.what());
    }

    // Validate schema
    if (!basis_json.contains("elements")) {
        throw InvalidArgumentException(
            "Basis set file missing 'elements' object: " + file_path);
    }

    const auto& elements = basis_json["elements"];

    // Build shells for each atom
    std::vector<Shell> shells;

    for (Size i = 0; i < atoms.size(); ++i) {
        const auto& atom = atoms[i];
        const auto atom_idx = static_cast<Index>(i);
        const std::string z_str = std::to_string(atom.atomic_number);

        if (!elements.contains(z_str)) {
            throw InvalidArgumentException(
                "Basis set does not contain data for element Z=" + z_str +
                " in file: " + file_path);
        }

        parse_element_shells(elements[z_str], atom.position, atom_idx, shells);
    }

    return BasisSet(std::move(shells));
}

}  // namespace libaccint::data
