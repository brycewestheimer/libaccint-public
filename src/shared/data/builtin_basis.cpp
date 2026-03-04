// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file builtin_basis.cpp
/// @brief Hard-coded STO-3G basis set data
///
/// Data from the Basis Set Exchange (BSE):
///   W.J. Hehre, R.F. Stewart, J.A. Pople, J. Chem. Phys. 51, 2657 (1969).
///
/// STO-3G uses 3 Gaussian primitives to approximate each Slater-type orbital.
/// For first-row atoms (Li-Ne), the basis consists of:
///   - 1s: 3 primitives (inner shell)
///   - 2sp: 3 primitives shared between 2s and 2p (valence shell)
///     The 2s and 2p share exponents but have different contraction coefficients.

#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <algorithm>
#include <cctype>

namespace libaccint::data {

namespace {

/// @brief STO-3G data for hydrogen (Z=1)
/// 1s shell: 3 primitives
void add_hydrogen_sto3g(const Point3D& center, Index atom_idx,
                         std::vector<Shell>& shells) {
    // 1s shell
    shells.emplace_back(
        0, center,
        std::vector<Real>{3.42525091, 0.62391373, 0.16885540},
        std::vector<Real>{0.15432897, 0.53532814, 0.44463454}
    );
    shells.back().set_atom_index(atom_idx);
}

/// @brief STO-3G data for carbon (Z=6)
/// 1s shell: 3 primitives, 2s shell: 3 primitives, 2p shell: 3 primitives
void add_carbon_sto3g(const Point3D& center, Index atom_idx,
                       std::vector<Shell>& shells) {
    // 1s shell (inner)
    shells.emplace_back(
        0, center,
        std::vector<Real>{71.6168370, 13.0450960, 3.5305122},
        std::vector<Real>{0.15432897, 0.53532814, 0.44463454}
    );
    shells.back().set_atom_index(atom_idx);

    // 2s shell (valence) — same exponents as 2p, different coefficients
    shells.emplace_back(
        0, center,
        std::vector<Real>{2.9412494, 0.6834831, 0.2222899},
        std::vector<Real>{-0.09996723, 0.39951283, 0.70011547}
    );
    shells.back().set_atom_index(atom_idx);

    // 2p shell (valence)
    shells.emplace_back(
        1, center,
        std::vector<Real>{2.9412494, 0.6834831, 0.2222899},
        std::vector<Real>{0.15591627, 0.60768372, 0.39195739}
    );
    shells.back().set_atom_index(atom_idx);
}

/// @brief STO-3G data for nitrogen (Z=7)
void add_nitrogen_sto3g(const Point3D& center, Index atom_idx,
                         std::vector<Shell>& shells) {
    // 1s shell (inner)
    shells.emplace_back(
        0, center,
        std::vector<Real>{99.1061690, 18.0523120, 4.8856602},
        std::vector<Real>{0.15432897, 0.53532814, 0.44463454}
    );
    shells.back().set_atom_index(atom_idx);

    // 2s shell (valence)
    shells.emplace_back(
        0, center,
        std::vector<Real>{3.7804559, 0.8784966, 0.2857144},
        std::vector<Real>{-0.09996723, 0.39951283, 0.70011547}
    );
    shells.back().set_atom_index(atom_idx);

    // 2p shell (valence)
    shells.emplace_back(
        1, center,
        std::vector<Real>{3.7804559, 0.8784966, 0.2857144},
        std::vector<Real>{0.15591627, 0.60768372, 0.39195739}
    );
    shells.back().set_atom_index(atom_idx);
}

/// @brief STO-3G data for oxygen (Z=8)
void add_oxygen_sto3g(const Point3D& center, Index atom_idx,
                       std::vector<Shell>& shells) {
    // 1s shell (inner)
    shells.emplace_back(
        0, center,
        std::vector<Real>{130.7093200, 23.8088610, 6.4436083},
        std::vector<Real>{0.15432897, 0.53532814, 0.44463454}
    );
    shells.back().set_atom_index(atom_idx);

    // 2s shell (valence)
    shells.emplace_back(
        0, center,
        std::vector<Real>{5.0331513, 1.1695961, 0.3803890},
        std::vector<Real>{-0.09996723, 0.39951283, 0.70011547}
    );
    shells.back().set_atom_index(atom_idx);

    // 2p shell (valence)
    shells.emplace_back(
        1, center,
        std::vector<Real>{5.0331513, 1.1695961, 0.3803890},
        std::vector<Real>{0.15591627, 0.60768372, 0.39195739}
    );
    shells.back().set_atom_index(atom_idx);
}

/// @brief STO-3G data for fluorine (Z=9)
void add_fluorine_sto3g(const Point3D& center, Index atom_idx,
                         std::vector<Shell>& shells) {
    // 1s shell (inner)
    shells.emplace_back(
        0, center,
        std::vector<Real>{166.6791300, 30.3608120, 8.2168207},
        std::vector<Real>{0.15432897, 0.53532814, 0.44463454}
    );
    shells.back().set_atom_index(atom_idx);

    // 2s shell (valence)
    shells.emplace_back(
        0, center,
        std::vector<Real>{6.4648032, 1.5022812, 0.4885885},
        std::vector<Real>{-0.09996723, 0.39951283, 0.70011547}
    );
    shells.back().set_atom_index(atom_idx);

    // 2p shell (valence)
    shells.emplace_back(
        1, center,
        std::vector<Real>{6.4648032, 1.5022812, 0.4885885},
        std::vector<Real>{0.15591627, 0.60768372, 0.39195739}
    );
    shells.back().set_atom_index(atom_idx);
}

}  // anonymous namespace

BasisSet create_sto3g(const std::vector<Atom>& atoms) {
    std::vector<Shell> shells;

    for (Size i = 0; i < atoms.size(); ++i) {
        const auto& atom = atoms[i];
        const auto atom_idx = static_cast<Index>(i);

        switch (atom.atomic_number) {
            case 1:  // H
                add_hydrogen_sto3g(atom.position, atom_idx, shells);
                break;
            case 6:  // C
                add_carbon_sto3g(atom.position, atom_idx, shells);
                break;
            case 7:  // N
                add_nitrogen_sto3g(atom.position, atom_idx, shells);
                break;
            case 8:  // O
                add_oxygen_sto3g(atom.position, atom_idx, shells);
                break;
            case 9:  // F
                add_fluorine_sto3g(atom.position, atom_idx, shells);
                break;
            default:
                throw InvalidArgumentException(
                    "STO-3G basis not available for element Z=" +
                    std::to_string(atom.atomic_number) +
                    ". Supported: H(1), C(6), N(7), O(8), F(9).");
        }
    }

    return BasisSet(std::move(shells));
}

BasisSet create_builtin_basis(const std::string& name,
                               const std::vector<Atom>& atoms) {
    // Case-insensitive comparison
    std::string lower_name = name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (lower_name == "sto-3g" || lower_name == "sto3g") {
        return create_sto3g(atoms);
    }

    throw InvalidArgumentException(
        "Unknown built-in basis set: '" + name +
        "'. Supported: sto-3g");
}

}  // namespace libaccint::data
