// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_helpers.hpp
/// @brief Shared helper functions for benchmark executables
///
/// Provides common molecule geometries, basis set factories, density matrix
/// generation, and point-charge setup used across multiple benchmark files.

#pragma once

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/operators/operator.hpp>

#include <random>
#include <vector>

namespace libaccint {
namespace bench {

// ============================================================================
// Density Matrix Generation
// ============================================================================

/// @brief Create a random symmetric density matrix
///
/// Generates an nbf x nbf symmetric matrix with values drawn from
/// uniform(-0.5, 0.5). Not positive semi-definite, but sufficient for
/// benchmarking integral contraction and Fock build performance.
inline std::vector<Real> create_random_density(Size nbf, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<Real> D(nbf * nbf);
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i; j < nbf; ++j) {
            double val = dist(gen);
            D[i * nbf + j] = val;
            D[j * nbf + i] = val;
        }
    }
    return D;
}

// ============================================================================
// Water Molecule Geometry
// ============================================================================

/// @brief Standard H2O geometry (Bohr units)
///
/// O at origin, H atoms at +/-1.430429, 0, 1.107157
inline std::vector<data::Atom> make_h2o_atoms() {
    return {
        {8, {0.0, 0.0, 0.0}},
        {1, {1.430429, 0.0, 1.107157}},
        {1, {-1.430429, 0.0, 1.107157}}
    };
}

/// @brief Create H2O BasisSet using the built-in basis data
inline BasisSet make_h2o_basis(const std::string& basis_name = "sto-3g") {
    return data::create_builtin_basis(basis_name, make_h2o_atoms());
}

// ============================================================================
// Manual Shell Construction (STO-3G Water)
// ============================================================================

/// @brief Build STO-3G shells for H2O with explicit exponents/coefficients
///
/// Returns 5 shells: O(1s), O(2s), O(2p), H1(1s), H2(1s) — 7 basis functions.
/// Useful when benchmarks need direct shell-level access.
inline std::vector<Shell> create_h2o_sto3g_shells() {
    std::vector<double> h_exp = {3.42525091, 0.62391373, 0.16885540};
    std::vector<double> h_coef = {0.15432897, 0.53532814, 0.44463454};

    std::vector<double> o_s_exp = {130.70932, 23.808861, 6.4436083};
    std::vector<double> o_s_coef = {0.15432897, 0.53532814, 0.44463454};

    std::vector<double> o_sp_exp = {5.0331513, 1.1695961, 0.38038896};
    std::vector<double> o_2s_coef = {-0.09996723, 0.39951283, 0.70011547};
    std::vector<double> o_2p_coef = {0.15591627, 0.60768372, 0.39195739};

    Point3D O{0.0, 0.0, 0.0};
    Point3D H1{1.430429, 0.0, 1.107157};
    Point3D H2{-1.430429, 0.0, 1.107157};

    std::vector<Shell> shells;

    Shell o_1s(AngularMomentum::S, O, o_s_exp, o_s_coef);
    o_1s.set_shell_index(0); o_1s.set_atom_index(0); o_1s.set_function_index(0);
    shells.push_back(o_1s);

    Shell o_2s(AngularMomentum::S, O, o_sp_exp, o_2s_coef);
    o_2s.set_shell_index(1); o_2s.set_atom_index(0); o_2s.set_function_index(1);
    shells.push_back(o_2s);

    Shell o_2p(AngularMomentum::P, O, o_sp_exp, o_2p_coef);
    o_2p.set_shell_index(2); o_2p.set_atom_index(0); o_2p.set_function_index(2);
    shells.push_back(o_2p);

    Shell h1_1s(AngularMomentum::S, H1, h_exp, h_coef);
    h1_1s.set_shell_index(3); h1_1s.set_atom_index(1); h1_1s.set_function_index(5);
    shells.push_back(h1_1s);

    Shell h2_1s(AngularMomentum::S, H2, h_exp, h_coef);
    h2_1s.set_shell_index(4); h2_1s.set_atom_index(2); h2_1s.set_function_index(6);
    shells.push_back(h2_1s);

    return shells;
}

// ============================================================================
// Point Charges
// ============================================================================

/// @brief Create PointChargeParams for H2O nuclear attraction integrals
inline PointChargeParams create_h2o_charges() {
    PointChargeParams charges;
    charges.x = {0.0, 1.430429, -1.430429};
    charges.y = {0.0, 0.0, 0.0};
    charges.z = {0.0, 1.107157, 1.107157};
    charges.charge = {8.0, 1.0, 1.0};
    return charges;
}

/// @brief Build PointChargeParams from an atom list
inline PointChargeParams make_nuclear_charges(
    const std::vector<data::Atom>& atoms) {
    PointChargeParams charges;
    for (const auto& atom : atoms) {
        charges.x.push_back(atom.position[0]);
        charges.y.push_back(atom.position[1]);
        charges.z.push_back(atom.position[2]);
        charges.charge.push_back(static_cast<Real>(atom.atomic_number));
    }
    return charges;
}

}  // namespace bench
}  // namespace libaccint
