// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file shell_set_triple.cpp
/// @brief ShellSetTriple class implementation for three-center integral computation

#include <libaccint/basis/shell_set_triple.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/auxiliary_basis_set.hpp>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace libaccint {

// =============================================================================
// ShellSetTriple Implementation
// =============================================================================

ShellSetTriple::ShellSetTriple(const ShellSet& shell_set_a,
                               const ShellSet& shell_set_b,
                               const ShellSet& shell_set_P,
                               bool symmetric)
    : symmetric_(symmetric) {

    // Set the key from the first shells' properties
    key_ = ShellSetTripleKey(
        shell_set_a.angular_momentum(),
        shell_set_b.angular_momentum(),
        shell_set_P.angular_momentum(),
        shell_set_a.n_primitives_per_shell(),
        shell_set_b.n_primitives_per_shell(),
        shell_set_P.n_primitives_per_shell()
    );

    // Build all (a, b, P) triples
    const Size n_a = shell_set_a.n_shells();
    const Size n_b = shell_set_b.n_shells();
    const Size n_P = shell_set_P.n_shells();

    triples_.reserve(n_a * n_b * n_P);

    for (Size i = 0; i < n_a; ++i) {
        const auto& shell_a = shell_set_a.shell(i);
        Size shell_a_idx = static_cast<Size>(shell_a.shell_index());

        Size j_start = symmetric ? i : 0;
        for (Size j = j_start; j < n_b; ++j) {
            const auto& shell_b = shell_set_b.shell(j);
            Size shell_b_idx = static_cast<Size>(shell_b.shell_index());

            for (Size k = 0; k < n_P; ++k) {
                const auto& shell_P = shell_set_P.shell(k);
                Size shell_P_idx = static_cast<Size>(shell_P.shell_index());

                ShellTriple triple;
                triple.shell_a_idx = shell_a_idx;
                triple.shell_b_idx = shell_b_idx;
                triple.shell_P_idx = shell_P_idx;
                triple.shell_a = &shell_a;
                triple.shell_b = &shell_b;
                triple.shell_P = &shell_P;

                triples_.push_back(triple);
            }
        }
    }
}

int ShellSetTriple::n_functions_a() const noexcept {
    return n_cartesian(key_.am_a);
}

int ShellSetTriple::n_functions_b() const noexcept {
    return n_cartesian(key_.am_b);
}

int ShellSetTriple::n_functions_P() const noexcept {
    return n_cartesian(key_.am_P);
}

const ShellTriple& ShellSetTriple::at(Size idx) const {
    if (idx >= triples_.size()) {
        throw std::out_of_range(
            "ShellSetTriple::at: index " + std::to_string(idx) +
            " out of range [0, " + std::to_string(triples_.size()) + ")");
    }
    return triples_[idx];
}

// =============================================================================
// Factory Functions
// =============================================================================

std::vector<ShellSetTriple> generate_shell_set_triples(
    const BasisSet& orbital,
    const AuxiliaryBasisSet& auxiliary,
    bool symmetric) {

    std::vector<ShellSetTriple> result;

    auto orbital_sets = orbital.shell_sets();
    if (orbital_sets.empty() || auxiliary.empty()) {
        return result;
    }

    // Group auxiliary shells into ShellSets by (am, n_primitives).
    // ShellSet contains std::once_flag (non-movable), so use unique_ptr.
    std::vector<std::unique_ptr<ShellSet>> aux_shell_sets;
    {
        std::map<std::pair<int,int>, size_t> group_index;
        for (Size i = 0; i < auxiliary.n_shells(); ++i) {
            const auto& shell = auxiliary.shell(i);
            auto key = std::make_pair(shell.angular_momentum(),
                                       static_cast<int>(shell.n_primitives()));
            auto it = group_index.find(key);
            if (it == group_index.end()) {
                group_index[key] = aux_shell_sets.size();
                aux_shell_sets.push_back(
                    std::make_unique<ShellSet>(key.first, key.second));
                aux_shell_sets.back()->add_shell(shell);
            } else {
                aux_shell_sets[it->second]->add_shell(shell);
            }
        }
    }

    // Generate all (orbital_a, orbital_b, aux_P) combinations.
    //
    // Orbital shell pointers are stable because orbital ShellSets are owned by
    // BasisSet. Auxiliary grouping ShellSets above are temporary; immediately
    // rebind auxiliary pointers to AuxiliaryBasisSet-owned shells.
    for (Size i = 0; i < orbital_sets.size(); ++i) {
        Size j_start = symmetric ? i : 0;
        for (Size j = j_start; j < orbital_sets.size(); ++j) {
            for (const auto& aux_set : aux_shell_sets) {
                result.emplace_back(*orbital_sets[i], *orbital_sets[j],
                                    *aux_set, symmetric && (i == j));
                auto& batch = result.back();
                for (auto& triple : batch.triples_) {
                    triple.shell_P = &auxiliary.shell(triple.shell_P_idx);
                }
            }
        }
    }

    return result;
}

Size estimate_triple_cost(const ShellSetTriple& triple) {
    // Rough cost estimate based on angular momentum and primitives
    const int la = triple.am_a();
    const int lb = triple.am_b();
    const int lP = triple.am_P();
    const int na = triple.n_primitives_a();
    const int nb = triple.n_primitives_b();
    const int nP = triple.n_primitives_P();

    // Number of Cartesian functions
    const int fa = n_cartesian(la);
    const int fb = n_cartesian(lb);
    const int fP = n_cartesian(lP);

    // Cost ~ n_triples * n_prims * n_functions
    return static_cast<Size>(triple.size()) *
           static_cast<Size>(na * nb * nP) *
           static_cast<Size>(fa * fb * fP);
}

}  // namespace libaccint
