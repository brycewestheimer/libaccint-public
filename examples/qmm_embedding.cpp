// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.
//
// qmm_embedding.cpp
//
// Demonstrates QM/MM embedding with distributed multipoles using LibAccInt.
// A water molecule (QM region) is embedded in a field of point charges and
// distributed multipoles representing the MM environment.
//
// Key concepts:
//   - DistributedMultipoleParams for multipole sites
//   - PointChargeParams for classical charges
//   - OneElectronOperator composition for combined QM/MM Hamiltonian
//   - Core Hamiltonian modification with external potentials

#include <libaccint/libaccint.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/operators/one_electron_operator.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace libaccint;

int main() {
    std::cout << "=== LibAccInt QM/MM Embedding Example ===\n";
    std::cout << "LibAccInt version: " << version() << "\n\n";

    // ── QM region: water molecule ───────────────────────────────────────────
    std::vector<data::Atom> qm_atoms = {
        {8, {0.0000,  0.0000, 0.2217}},   // O
        {1, {0.0000,  1.4309, -0.8867}},  // H
        {1, {0.0000, -1.4309, -0.8867}},  // H
    };

    BasisSet basis = data::create_builtin_basis("STO-3G", qm_atoms);
    const Size nbf = basis.n_basis_functions();
    Engine engine(basis);

    std::cout << "QM region: H2O (" << nbf << " basis functions)\n";

    // ── MM environment: point charges representing nearby water molecules ────
    // Two water molecules as TIP3P point charges at ~5 bohr distance
    PointChargeParams mm_charges;

    // MM water 1 (shifted +6 bohr along x)
    mm_charges.x.push_back(6.0);  mm_charges.y.push_back(0.0);
    mm_charges.z.push_back(0.22);  mm_charges.charge.push_back(-0.834);  // O
    mm_charges.x.push_back(6.0);  mm_charges.y.push_back(1.43);
    mm_charges.z.push_back(-0.89);  mm_charges.charge.push_back(0.417);  // H
    mm_charges.x.push_back(6.0);  mm_charges.y.push_back(-1.43);
    mm_charges.z.push_back(-0.89);  mm_charges.charge.push_back(0.417);  // H

    // MM water 2 (shifted -6 bohr along x)
    mm_charges.x.push_back(-6.0);  mm_charges.y.push_back(0.0);
    mm_charges.z.push_back(0.22);  mm_charges.charge.push_back(-0.834);
    mm_charges.x.push_back(-6.0);  mm_charges.y.push_back(1.43);
    mm_charges.z.push_back(-0.89);  mm_charges.charge.push_back(0.417);
    mm_charges.x.push_back(-6.0);  mm_charges.y.push_back(-1.43);
    mm_charges.z.push_back(-0.89);  mm_charges.charge.push_back(0.417);

    std::cout << "MM environment: " << mm_charges.n_centers()
              << " point charges (2 TIP3P waters)\n";

    // ── MM distributed multipoles: add multipole sites in between ───────────
    // This demonstrates DistributedMultipoleParams with charges + dipoles
    DistributedMultipoleParams multipoles;

    // Add a polarizable MM site with charge + dipole at (0, 5, 0)
    multipoles.x.push_back(0.0);
    multipoles.y.push_back(9.45);  // ~5 Å from QM region
    multipoles.z.push_back(0.0);
    multipoles.charges.push_back(-0.5);       // charge
    multipoles.dipole_x.push_back(0.0);       // dipole x
    multipoles.dipole_y.push_back(-0.3);      // dipole y (pointing toward QM)
    multipoles.dipole_z.push_back(0.0);       // dipole z

    std::cout << "Distributed multipole sites: " << multipoles.n_sites()
              << " (max rank " << multipoles.max_rank() << ")\n\n";

    // ── Step 1: Compute pure QM core Hamiltonian ──────────────────────────
    PointChargeParams nuclear_charges;
    for (const auto& atom : qm_atoms) {
        nuclear_charges.x.push_back(atom.position.x);
        nuclear_charges.y.push_back(atom.position.y);
        nuclear_charges.z.push_back(atom.position.z);
        nuclear_charges.charge.push_back(static_cast<Real>(atom.atomic_number));
    }

    std::vector<Real> H_qm(nbf * nbf, 0.0);
    engine.compute_core_hamiltonian(nuclear_charges, H_qm);

    // ── Step 2: Compute MM embedding contribution ───────────────────────────
    // The MM contribution to the QM Hamiltonian includes:
    //   V_mm = sum_A q_A * <mu| 1/|r-R_A| |nu>
    // This is computed using the PointCharge operator
    std::vector<Real> V_mm(nbf * nbf, 0.0);
    auto op_mm = OneElectronOperator(Operator::point_charges(mm_charges));
    engine.compute(op_mm, V_mm);

    // ── Step 3: Compute distributed multipole contribution ──────────────────
    // When a distributed multipole operator is available, add it to the
    // Hamiltonian. For now, compute the point charge part of the multipoles.
    PointChargeParams multipole_charge_part;
    for (Size i = 0; i < multipoles.n_sites(); ++i) {
        multipole_charge_part.x.push_back(multipoles.x[i]);
        multipole_charge_part.y.push_back(multipoles.y[i]);
        multipole_charge_part.z.push_back(multipoles.z[i]);
        multipole_charge_part.charge.push_back(multipoles.charges[i]);
    }

    std::vector<Real> V_multipole(nbf * nbf, 0.0);
    auto op_multipole = OneElectronOperator(Operator::point_charges(multipole_charge_part));
    engine.compute(op_multipole, V_multipole);

    // ── Step 4: Build QM/MM Hamiltonian ─────────────────────────────────────
    // H_qmm = H_qm + V_mm + V_multipole
    std::vector<Real> H_qmm(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf * nbf; ++i) {
        H_qmm[i] = H_qm[i] + V_mm[i] + V_multipole[i];
    }

    // ── Print results ───────────────────────────────────────────────────────
    std::cout << "QM core Hamiltonian (diagonal):\n";
    std::cout << std::fixed << std::setprecision(8);
    for (Size i = 0; i < nbf; ++i) {
        std::cout << "  H_qm[" << i << "," << i << "] = "
                  << std::setw(14) << H_qm[i * nbf + i] << "\n";
    }

    std::cout << "\nMM embedding contribution (diagonal):\n";
    for (Size i = 0; i < nbf; ++i) {
        std::cout << "  V_mm[" << i << "," << i << "] = "
                  << std::setw(14) << V_mm[i * nbf + i] << "\n";
    }

    std::cout << "\nMultipole contribution (diagonal):\n";
    for (Size i = 0; i < nbf; ++i) {
        std::cout << "  V_mp[" << i << "," << i << "] = "
                  << std::setw(14) << V_multipole[i * nbf + i] << "\n";
    }

    std::cout << "\nQM/MM Hamiltonian (diagonal):\n";
    for (Size i = 0; i < nbf; ++i) {
        std::cout << "  H_qmm[" << i << "," << i << "] = "
                  << std::setw(14) << H_qmm[i * nbf + i] << "\n";
    }

    // Compute embedding shift
    double shift = 0.0;
    for (Size i = 0; i < nbf; ++i) {
        shift += std::abs(H_qmm[i * nbf + i] - H_qm[i * nbf + i]);
    }
    std::cout << "\nTotal diagonal embedding shift: "
              << std::setprecision(8) << shift << " Hartree\n";

    return 0;
}
