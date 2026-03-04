// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.
//
// basic_hf_energy.cpp
//
// Demonstrates a basic Hartree–Fock (HF) energy calculation for H₂ using
// LibAccInt. This example:
//   1. Constructs an STO-3G basis set for H₂
//   2. Computes overlap (S), kinetic (T), and nuclear attraction (V) matrices
//   3. Builds the core Hamiltonian H_core = T + V
//   4. Performs a simple SCF loop with fixed density
//   5. Computes the electronic energy
//
// This is a pedagogical example — a real HF code would diagonalize the Fock
// matrix and iterate to self-consistency. Here we use a fixed density to
// showcase the integral computation APIs.

#include <libaccint/libaccint.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using namespace libaccint;

int main() {
    std::cout << "=== LibAccInt Basic HF Energy Example ===\n";
    std::cout << "LibAccInt version: " << version() << "\n\n";

    // ── Step 1: Define the molecular geometry (H₂) ──────────────────────────
    // Positions are in Bohr (atomic units).
    // H₂ bond length: ~1.4 bohr (0.74 Å)
    const double bond_length = 1.4;  // bohr

    std::vector<data::Atom> atoms = {
        {1, {0.0, 0.0, 0.0}},           // H atom at origin
        {1, {0.0, 0.0, bond_length}},   // H atom along z-axis
    };

    // ── Step 2: Create the STO-3G basis set ─────────────────────────────────
    BasisSet basis = data::create_builtin_basis("STO-3G", atoms);
    const Size nbf = basis.n_basis_functions();

    std::cout << "Molecule: H2\n";
    std::cout << "Basis set: STO-3G\n";
    std::cout << "Number of basis functions: " << nbf << "\n";
    std::cout << "Number of shells: " << basis.n_shells() << "\n\n";

    // ── Step 3: Create the computation engine ───────────────────────────────
    Engine engine(basis);

    // ── Step 4: Compute one-electron integrals ──────────────────────────────
    // Overlap matrix S
    std::vector<Real> S(nbf * nbf, 0.0);
    engine.compute_overlap_matrix(S);

    // Kinetic energy matrix T
    std::vector<Real> T(nbf * nbf, 0.0);
    engine.compute_kinetic_matrix(T);

    // Nuclear attraction matrix V
    // Set up nuclear charges using SoA layout
    PointChargeParams nuclear_charges;
    for (const auto& atom : atoms) {
        nuclear_charges.x.push_back(atom.position.x);
        nuclear_charges.y.push_back(atom.position.y);
        nuclear_charges.z.push_back(atom.position.z);
        nuclear_charges.charge.push_back(static_cast<Real>(atom.atomic_number));
    }

    std::vector<Real> V(nbf * nbf, 0.0);
    engine.compute_nuclear_matrix(nuclear_charges, V);

    // Core Hamiltonian: H_core = T + V
    std::vector<Real> H_core(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf * nbf; ++i) {
        H_core[i] = T[i] + V[i];
    }

    // Print matrices
    std::cout << "Overlap matrix S:\n";
    std::cout << std::fixed << std::setprecision(8);
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            std::cout << std::setw(14) << S[i * nbf + j];
        }
        std::cout << '\n';
    }
    std::cout << '\n';

    std::cout << "Core Hamiltonian H_core = T + V:\n";
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            std::cout << std::setw(14) << H_core[i * nbf + j];
        }
        std::cout << '\n';
    }
    std::cout << '\n';

    // ── Step 5: Build two-electron integrals and Fock matrix ────────────────
    // For this demo, we use a simple density matrix D = 0.5 * I
    // (In a real SCF, D is built from eigenvectors of the Fock matrix)
    std::vector<Real> D(nbf * nbf, 0.0);
    // Simple approximation: uniform density
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 0.5;
    }

    // Build Fock matrix using fused compute-and-consume pattern
    consumers::FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);
    engine.compute(Operator::coulomb(), fock);

    // Get J and K matrices
    auto J = fock.get_coulomb_matrix();
    auto K = fock.get_exchange_matrix();

    // Compute the Fock matrix: F = H_core + J - 0.5*K (RHF)
    // Using exchange_fraction = 0.5 for RHF with the closed-shell formulation
    auto F = fock.get_fock_matrix(
        std::span<const Real>(H_core),
        0.5  // exchange fraction for RHF closed-shell
    );

    std::cout << "Fock matrix F = H_core + J - 0.5*K:\n";
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            std::cout << std::setw(14) << F[i * nbf + j];
        }
        std::cout << '\n';
    }
    std::cout << '\n';

    // ── Step 6: Compute electronic energy ───────────────────────────────────
    // E_elec = 0.5 * Tr[D * (H_core + F)]
    Real E_elec = 0.0;
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            E_elec += D[i * nbf + j] * (H_core[i * nbf + j] + F[i * nbf + j]);
        }
    }
    E_elec *= 0.5;

    // Nuclear repulsion energy: V_nn = Z_A * Z_B / R_AB
    Real V_nn = static_cast<Real>(atoms[0].atomic_number * atoms[1].atomic_number)
                / bond_length;

    Real E_total = E_elec + V_nn;

    std::cout << "Electronic energy: " << std::setprecision(10) << E_elec << " Hartree\n";
    std::cout << "Nuclear repulsion: " << V_nn << " Hartree\n";
    std::cout << "Total energy:      " << E_total << " Hartree\n";
    std::cout << "\nNote: This uses a non-SCF density, so the energy is approximate.\n";
    std::cout << "A full SCF implementation would iterate to self-consistency.\n";

    return 0;
}
