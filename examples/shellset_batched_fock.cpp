// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.
//
// shellset_batched_fock.cpp
//
// Demonstrates full-basis Fock matrix construction using the fused
// compute-and-consume path.  A FockBuilder consumer accumulates the Coulomb
// (J) and exchange (K) matrices from two-electron integrals without ever
// materializing the full ERI tensor.

#include <libaccint/libaccint.hpp>
#include <iomanip>
#include <iostream>

using namespace libaccint;

int main() {
    // ── Build a minimal H₂ basis (STO-3G) ──────────────────────────────────
    Shell s1(0,                                          // angular momentum (s)
             Point3D{0.0, 0.0, 0.0},                  // center
             {3.42525091, 0.62391373, 0.16885540},      // exponents
             {0.15432897, 0.53532814, 0.44463454});     // contraction coefficients

    Shell s2(0,
             Point3D{0.0, 0.0, 1.4},                  // 1.4 bohr ≈ 0.74 Å
             {3.42525091, 0.62391373, 0.16885540},
             {0.15432897, 0.53532814, 0.44463454});

    BasisSet basis({s1, s2});
    Engine engine(basis);

    Size nbf = basis.n_basis_functions();
    std::cout << "H2 / STO-3G — " << nbf << " basis functions\n\n";

    // ── Set up a dummy density matrix (identity) ────────────────────────────
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0;
    }

    // ── Full-basis fused Fock build ─────────────────────────────────────────
    consumers::FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);
    engine.compute(Operator::coulomb(), fock);

    auto J = fock.get_coulomb_matrix();
    auto K = fock.get_exchange_matrix();

    // ── Print the Coulomb matrix ────────────────────────────────────────────
    std::cout << "Coulomb matrix J:\n";
    std::cout << std::fixed << std::setprecision(8);
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            std::cout << std::setw(14) << J[i * nbf + j];
        }
        std::cout << '\n';
    }

    std::cout << "\nExchange matrix K:\n";
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            std::cout << std::setw(14) << K[i * nbf + j];
        }
        std::cout << '\n';
    }

    return 0;
}
