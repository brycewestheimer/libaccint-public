// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.
//
// shellset_batched_1e.cpp
//
// Demonstrates batched one-electron integral evaluation using ShellSetPairs.
// Computes the overlap matrix S for a minimal H2 system (STO-3G) by iterating
// over basis.shell_set_pairs() — the preferred high-throughput path.

#include <libaccint/libaccint.hpp>
#include <iomanip>
#include <iostream>

using namespace libaccint;

int main() {
    // ── Build a minimal H₂ basis (STO-3G) ──────────────────────────────────
    // Each hydrogen has a single s-type shell with 3 primitives.
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

    // ── Batched 1e overlap via ShellSetPair iteration ───────────────────────
    const auto& pairs = basis.shell_set_pairs();
    std::vector<Real> S(nbf * nbf, 0.0);

    for (const auto& pair : pairs) {
        engine.compute(Operator::overlap(), pair, S);
    }

    // ── Print the overlap matrix ────────────────────────────────────────────
    std::cout << "Overlap matrix S:\n";
    std::cout << std::fixed << std::setprecision(8);
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            std::cout << std::setw(14) << S[i * nbf + j];
        }
        std::cout << '\n';
    }

    return 0;
}
