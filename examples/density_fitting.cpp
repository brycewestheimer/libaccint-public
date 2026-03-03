// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.
//
// density_fitting.cpp
//
// Demonstrates density-fitted Hartree–Fock (DF-HF) using LibAccInt's
// DFFockBuilder. Density fitting (also called Resolution of the Identity, RI)
// replaces four-center ERIs with a combination of two- and three-center
// integrals via an auxiliary basis set.
//
// Key concepts:
//   - AuxiliaryBasisSet construction
//   - DFFockBuilder for DF-J and DF-K
//   - Memory-efficient integral evaluation

#include <libaccint/libaccint.hpp>
#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/consumers/df_fock_builder.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace libaccint;

int main() {
    std::cout << "=== LibAccInt Density Fitting Example ===\n";
    std::cout << "LibAccInt version: " << version() << "\n\n";

    // ── Define H₂ molecule ──────────────────────────────────────────────────
    const double bond_length = 1.4;  // bohr

    std::vector<data::Atom> atoms = {
        {1, {0.0, 0.0, 0.0}},
        {1, {0.0, 0.0, bond_length}},
    };

    // ── Create orbital basis set ────────────────────────────────────────────
    BasisSet orbital_basis = data::create_builtin_basis("STO-3G", atoms);
    const Size nbf = orbital_basis.n_basis_functions();

    std::cout << "Molecule: H2\n";
    std::cout << "Orbital basis: STO-3G (" << nbf << " functions, "
              << orbital_basis.n_shells() << " shells)\n";

    // ── Create auxiliary basis set ──────────────────────────────────────────
    // For density fitting, we need an auxiliary basis set with higher angular
    // momentum functions. We construct a simple auxiliary basis manually.
    // In production, you would use a standard auxiliary basis like cc-pVDZ-RI.
    std::vector<Shell> aux_shells;
    for (const auto& atom : atoms) {
        // s-type auxiliary function (diffuse)
        aux_shells.emplace_back(0, atom.position,
                                std::vector<Real>{1.0, 0.3},
                                std::vector<Real>{0.5, 0.5});
        // p-type auxiliary function
        aux_shells.emplace_back(1, atom.position,
                                std::vector<Real>{0.8},
                                std::vector<Real>{1.0});
    }

    AuxiliaryBasisSet aux_basis(std::move(aux_shells));
    std::cout << "Auxiliary basis: custom (" << aux_basis.n_functions()
              << " functions, " << aux_basis.n_shells() << " shells)\n\n";

    // ── Set up density matrix ───────────────────────────────────────────────
    std::vector<Real> D(nbf * nbf, 0.0);
    const Real coeff = 1.0 / std::sqrt(2.0);
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            D[i * nbf + j] = 2.0 * coeff * coeff;
        }
    }

    // ── Standard (non-DF) Fock build for comparison ─────────────────────────
    Engine engine(orbital_basis);

    consumers::FockBuilder standard_fock(nbf);
    standard_fock.set_density(D.data(), nbf);
    engine.compute(Operator::coulomb(), standard_fock);

    auto J_standard = standard_fock.get_coulomb_matrix();

    std::cout << "Standard Coulomb matrix J (non-DF):\n";
    std::cout << std::fixed << std::setprecision(8);
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            std::cout << std::setw(14) << J_standard[i * nbf + j];
        }
        std::cout << '\n';
    }

    // ── Density-fitted Fock build ───────────────────────────────────────────
    // The DFFockBuilder handles three-center integrals internally
    consumers::DFFockBuilderConfig df_config;
    df_config.compute_coulomb = true;
    df_config.compute_exchange = true;
    df_config.exchange_fraction = 1.0;  // Full exchange for HF

    consumers::DFFockBuilder df_fock(orbital_basis, aux_basis, df_config);

    // Set density and compute
    df_fock.set_density(std::span<const Real>(D.data(), nbf * nbf));
    [[maybe_unused]] auto F_df = df_fock.compute();

    auto J_df = df_fock.compute_coulomb();

    std::cout << "\nDF Coulomb matrix J (density-fitted):\n";
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            std::cout << std::setw(14) << J_df[i * nbf + j];
        }
        std::cout << '\n';
    }

    // ── Compare DF vs standard ──────────────────────────────────────────────
    Real max_diff = 0.0;
    Real rms_diff = 0.0;
    for (Size i = 0; i < nbf * nbf; ++i) {
        Real diff = std::abs(J_df[i] - J_standard[i]);
        max_diff = std::max(max_diff, diff);
        rms_diff += diff * diff;
    }
    rms_diff = std::sqrt(rms_diff / static_cast<Real>(nbf * nbf));

    std::cout << "\nDF approximation error:\n";
    std::cout << "  Max |J_DF - J_exact| = " << std::scientific
              << std::setprecision(2) << max_diff << "\n";
    std::cout << "  RMS |J_DF - J_exact| = " << rms_diff << "\n";

    std::cout << "\nNote: DF errors depend on the auxiliary basis set quality.\n";
    std::cout << "Standard auxiliary basis sets (e.g., cc-pVDZ-JKFIT) provide\n";
    std::cout << "errors typically below 1e-4 Hartree in total energy.\n";

    return 0;
}
