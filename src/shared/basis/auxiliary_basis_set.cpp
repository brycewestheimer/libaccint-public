// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file auxiliary_basis_set.cpp
/// @brief Implementation of AuxiliaryBasisSet class

#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/data/auxiliary_basis_data.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <algorithm>
#include <numeric>

namespace libaccint {

// =============================================================================
// Construction
// =============================================================================

AuxiliaryBasisSet::AuxiliaryBasisSet(std::vector<Shell> shells,
                                       FittingType type,
                                       std::string name)
    : shells_(std::move(shells)),
      fitting_type_(type),
      name_(std::move(name)) {
    build_indices();
}

AuxiliaryBasisSet::AuxiliaryBasisSet(AuxiliaryBasisSet&& other) noexcept
    : shells_(std::move(other.shells_)),
      shell_to_func_(std::move(other.shell_to_func_)),
      func_to_shell_(std::move(other.func_to_shell_)),
      n_functions_(other.n_functions_),
      n_primitives_(other.n_primitives_),
      max_am_(other.max_am_),
      fitting_type_(other.fitting_type_),
      name_(std::move(other.name_)),
      orbital_basis_(other.orbital_basis_),
      soa_data_(std::move(other.soa_data_)),
      soa_valid_(other.soa_valid_) {
    // Note: once_flag is default-constructed (fresh); soa_valid_ handles cache state
    other.n_functions_ = 0;
    other.n_primitives_ = 0;
    other.max_am_ = 0;
    other.orbital_basis_ = nullptr;
    other.soa_valid_ = false;
}

AuxiliaryBasisSet& AuxiliaryBasisSet::operator=(AuxiliaryBasisSet&& other) noexcept {
    if (this != &other) {
        shells_ = std::move(other.shells_);
        shell_to_func_ = std::move(other.shell_to_func_);
        func_to_shell_ = std::move(other.func_to_shell_);
        n_functions_ = other.n_functions_;
        n_primitives_ = other.n_primitives_;
        max_am_ = other.max_am_;
        fitting_type_ = other.fitting_type_;
        name_ = std::move(other.name_);
        orbital_basis_ = other.orbital_basis_;
        soa_data_ = std::move(other.soa_data_);
        soa_valid_ = other.soa_valid_;
        // once_flag stays fresh; soa_valid_ manages cache validity
        other.n_functions_ = 0;
        other.n_primitives_ = 0;
        other.max_am_ = 0;
        other.orbital_basis_ = nullptr;
        other.soa_valid_ = false;
    }
    return *this;
}

AuxiliaryBasisSet::AuxiliaryBasisSet(std::span<const Shell> shells,
                                       FittingType type,
                                       std::string name)
    : shells_(shells.begin(), shells.end()),
      fitting_type_(type),
      name_(std::move(name)) {
    build_indices();
}

// =============================================================================
// Shell Access
// =============================================================================

const Shell& AuxiliaryBasisSet::shell(Size i) const {
    if (i >= shells_.size()) {
        throw InvalidArgumentException(
            "Shell index " + std::to_string(i) + " out of bounds (n_shells = " +
            std::to_string(shells_.size()) + ")");
    }
    return shells_[i];
}

// =============================================================================
// Indexing
// =============================================================================

Size AuxiliaryBasisSet::shell_to_function(Size shell_idx) const {
    if (shell_idx >= shells_.size()) {
        throw InvalidArgumentException(
            "Shell index " + std::to_string(shell_idx) + " out of bounds");
    }
    return shell_to_func_[shell_idx];
}

Size AuxiliaryBasisSet::function_to_shell(Size func_idx) const {
    if (func_idx >= n_functions_) {
        throw InvalidArgumentException(
            "Function index " + std::to_string(func_idx) + " out of bounds");
    }
    return func_to_shell_[func_idx];
}

void AuxiliaryBasisSet::build_indices() {
    if (shells_.empty()) {
        return;
    }

    shell_to_func_.resize(shells_.size());
    n_primitives_ = 0;
    n_functions_ = 0;
    max_am_ = 0;

    for (Size i = 0; i < shells_.size(); ++i) {
        shells_[i].set_shell_index(static_cast<Index>(i));
        shells_[i].set_function_index(static_cast<Index>(n_functions_));
        shell_to_func_[i] = n_functions_;
        n_functions_ += shells_[i].n_functions();
        n_primitives_ += shells_[i].n_primitives();
        max_am_ = std::max(max_am_, shells_[i].angular_momentum());
    }

    // Build function-to-shell mapping
    func_to_shell_.resize(n_functions_);
    for (Size i = 0; i < shells_.size(); ++i) {
        const Size start = shell_to_func_[i];
        const Size nfunc = shells_[i].n_functions();
        for (Size j = 0; j < nfunc; ++j) {
            func_to_shell_[start + j] = i;
        }
    }
}

// =============================================================================
// Orbital Basis Pairing
// =============================================================================

void AuxiliaryBasisSet::set_orbital_basis(const BasisSet& orbital) {
    orbital_basis_ = &orbital;
}

const BasisSet& AuxiliaryBasisSet::orbital_basis() const {
    if (orbital_basis_ == nullptr) {
        throw InvalidStateException("No orbital basis set associated");
    }
    return *orbital_basis_;
}

// =============================================================================
// SoA Data
// =============================================================================

const AuxiliaryBasisSetSoA& AuxiliaryBasisSet::soa_data() const {
    std::call_once(soa_init_flag_, [this]() { build_soa_data(); });

    if (!soa_valid_) {
        build_soa_data();
    }

    return soa_data_;
}

void AuxiliaryBasisSet::build_soa_data() const {
    soa_data_.n_shells = shells_.size();
    soa_data_.n_primitives = n_primitives_;
    soa_data_.n_functions = n_functions_;

    // Allocate arrays
    soa_data_.center_x.resize(shells_.size());
    soa_data_.center_y.resize(shells_.size());
    soa_data_.center_z.resize(shells_.size());
    soa_data_.angular_momenta.resize(shells_.size());
    soa_data_.primitive_offsets.resize(shells_.size());
    soa_data_.n_primitives_per_shell.resize(shells_.size());
    soa_data_.function_offsets.resize(shells_.size());
    soa_data_.n_functions_per_shell.resize(shells_.size());

    soa_data_.exponents.resize(n_primitives_);
    soa_data_.coefficients.resize(n_primitives_);

    Size prim_offset = 0;
    for (Size i = 0; i < shells_.size(); ++i) {
        const Shell& s = shells_[i];
        const auto& center = s.center();

        soa_data_.center_x[i] = center[0];
        soa_data_.center_y[i] = center[1];
        soa_data_.center_z[i] = center[2];
        soa_data_.angular_momenta[i] = s.angular_momentum();
        soa_data_.primitive_offsets[i] = prim_offset;
        soa_data_.n_primitives_per_shell[i] = s.n_primitives();
        soa_data_.function_offsets[i] = shell_to_func_[i];
        soa_data_.n_functions_per_shell[i] = s.n_functions();

        // Copy primitive data
        const auto& exps = s.exponents();
        const auto& coeffs = s.coefficients();
        for (Size j = 0; j < s.n_primitives(); ++j) {
            soa_data_.exponents[prim_offset + j] = exps[j];
            soa_data_.coefficients[prim_offset + j] = coeffs[j];
        }

        prim_offset += s.n_primitives();
    }

    soa_valid_ = true;
}

// =============================================================================
// Factory Functions
// =============================================================================

AuxiliaryBasisSet load_auxiliary_basis(
    const std::string& basis_name,
    std::span<const int> atomic_numbers,
    std::span<const std::array<Real, 3>> centers) {

    if (atomic_numbers.size() != centers.size()) {
        throw InvalidArgumentException(
            "Mismatched sizes: atomic_numbers(" + std::to_string(atomic_numbers.size()) +
            ") vs centers(" + std::to_string(centers.size()) + ")");
    }

    // Convert to Atom vector for the data API
    std::vector<data::Atom> atoms;
    atoms.reserve(atomic_numbers.size());
    for (size_t i = 0; i < atomic_numbers.size(); ++i) {
        data::Atom atom;
        atom.atomic_number = atomic_numbers[i];
        atom.position = Point3D{centers[i][0], centers[i][1], centers[i][2]};
        atoms.push_back(atom);
    }
    return data::create_builtin_auxiliary_basis(basis_name, atoms);
}

std::vector<std::string> available_auxiliary_bases() {
    return data::list_builtin_auxiliary_bases();
}

bool is_auxiliary_basis_available(
    const std::string& basis_name,
    std::span<const int> atomic_numbers) {

    std::vector<int> z_vec(atomic_numbers.begin(), atomic_numbers.end());
    return data::is_builtin_auxiliary_available(basis_name, z_vec);
}

}  // namespace libaccint
