// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file shell_set_quartet_utils.hpp
/// @brief Utility functions for ShellSetQuartet generation, filtering,
///        sorting, grouping, and cost estimation.
///
/// These utilities support advanced quartet iteration patterns beyond
/// the basic upper-triangle enumeration provided by BasisSet::shell_set_quartets().

#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/core/types.hpp>

#include <algorithm>
#include <functional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace libaccint {

// =============================================================================
// AMClass — angular momentum class key
// =============================================================================

/// @brief Key identifying an angular momentum class (La, Lb, Lc, Ld)
struct AMClass {
    int La{0};
    int Lb{0};
    int Lc{0};
    int Ld{0};

    [[nodiscard]] constexpr bool operator==(const AMClass&) const noexcept = default;

    /// @brief Lexicographic comparison for sorting by AM class
    [[nodiscard]] constexpr bool operator<(const AMClass& rhs) const noexcept {
        return std::tie(La, Lb, Lc, Ld) < std::tie(rhs.La, rhs.Lb, rhs.Lc, rhs.Ld);
    }

    /// @brief Total angular momentum
    [[nodiscard]] constexpr int total() const noexcept { return La + Lb + Lc + Ld; }
};

/// @brief Hash for AMClass
struct AMClassHash {
    [[nodiscard]] std::size_t operator()(const AMClass& k) const noexcept {
        // Pack into single integer for fast hashing (AM values typically 0-6)
        return static_cast<std::size_t>(k.La) * 343 +
               static_cast<std::size_t>(k.Lb) * 49 +
               static_cast<std::size_t>(k.Lc) * 7 +
               static_cast<std::size_t>(k.Ld);
    }
};

/// @brief Extract the AMClass from a ShellSetQuartet
[[nodiscard]] inline AMClass get_am_class(const ShellSetQuartet& q) noexcept {
    return {q.La(), q.Lb(), q.Lc(), q.Ld()};
}

// =============================================================================
// Sorting Utilities
// =============================================================================

/// @brief Sort quartets by ascending total angular momentum
///
/// Returns a new vector with quartets sorted by La+Lb+Lc+Ld. Within the
/// same total AM, the original order is preserved (stable sort).
///
/// @param quartets Source quartets
/// @return Sorted copy of the quartets (as pointers into the original span)
[[nodiscard]] inline std::vector<const ShellSetQuartet*>
sort_by_total_am(std::span<const ShellSetQuartet> quartets) {
    std::vector<const ShellSetQuartet*> sorted;
    sorted.reserve(quartets.size());
    for (const auto& q : quartets) {
        sorted.push_back(&q);
    }
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const ShellSetQuartet* a, const ShellSetQuartet* b) {
                         return a->L_total() < b->L_total();
                     });
    return sorted;
}

// =============================================================================
// Grouping Utilities
// =============================================================================

/// @brief A group of quartets sharing the same AM class
struct AMClassGroup {
    AMClass am_class;
    std::vector<const ShellSetQuartet*> quartets;
};

/// @brief Group quartets by their (La, Lb, Lc, Ld) angular momentum class
///
/// Returns a vector of groups, each containing all quartets with the same
/// AM class. Groups are sorted by AMClass lexicographic order.
///
/// @param quartets Source quartets
/// @return Vector of AMClassGroup, sorted by AM class
[[nodiscard]] inline std::vector<AMClassGroup>
group_by_am_class(std::span<const ShellSetQuartet> quartets) {
    std::unordered_map<AMClass, std::vector<const ShellSetQuartet*>, AMClassHash> groups;

    for (const auto& q : quartets) {
        groups[get_am_class(q)].push_back(&q);
    }

    std::vector<AMClassGroup> result;
    result.reserve(groups.size());
    for (auto& [am, qs] : groups) {
        result.push_back(AMClassGroup{am, std::move(qs)});
    }

    // Sort groups by AM class for deterministic ordering
    std::sort(result.begin(), result.end(),
              [](const AMClassGroup& a, const AMClassGroup& b) {
                  return a.am_class < b.am_class;
              });

    return result;
}

// =============================================================================
// Cost Estimation
// =============================================================================

/// @brief Estimate the computational cost of a ShellSetQuartet
///
/// The cost metric is proportional to the expected computation time,
/// accounting for angular momentum scaling and primitive count.
/// Useful for load balancing in distributed and multi-GPU settings.
///
/// @param quartet The ShellSetQuartet to estimate cost for
/// @return Relative cost metric (higher = more expensive)
[[nodiscard]] inline double estimate_quartet_cost(const ShellSetQuartet& quartet) noexcept {
    const auto& bra = quartet.bra_pair();
    const auto& ket = quartet.ket_pair();

    // Number of shell quartets
    const double n_shell_quartets = static_cast<double>(quartet.n_quartets());

    // Functions per shell quartet
    const double nf = static_cast<double>(
        n_cartesian(bra.La()) * n_cartesian(bra.Lb()) *
        n_cartesian(ket.La()) * n_cartesian(ket.Lb()));

    // Number of primitive combinations
    const double n_prims = static_cast<double>(
        bra.shell_set_a().n_primitives_per_shell() *
        bra.shell_set_b().n_primitives_per_shell() *
        ket.shell_set_a().n_primitives_per_shell() *
        ket.shell_set_b().n_primitives_per_shell());

    // AM scaling: recursion work scales roughly as (L+1)^2 per center
    const double am_factor = static_cast<double>(
        (bra.La() + 1) * (bra.Lb() + 1) *
        (ket.La() + 1) * (ket.Lb() + 1));

    return n_shell_quartets * nf * n_prims * am_factor;
}

// =============================================================================
// Quartet Generation from Arbitrary Pair Sets
// =============================================================================

/// @brief Generate all quartets from two sets of ShellSetPairs
///
/// Generates all combinations (bra, ket) from the two sets of pairs.
/// If the two spans point to the same data, uses upper-triangle enumeration.
///
/// @param bra_pairs Set of bra ShellSetPairs
/// @param ket_pairs Set of ket ShellSetPairs
/// @return Vector of ShellSetQuartets referencing pairs in the input spans
[[nodiscard]] inline std::vector<ShellSetQuartet>
generate_quartets(std::span<const ShellSetPair> bra_pairs,
                  std::span<const ShellSetPair> ket_pairs) {
    std::vector<ShellSetQuartet> result;

    // Check if this is a self-pairing (same span)
    const bool is_self = (bra_pairs.data() == ket_pairs.data() &&
                          bra_pairs.size() == ket_pairs.size());

    if (is_self) {
        const Size n = bra_pairs.size();
        result.reserve(n * (n + 1) / 2);
        for (Size i = 0; i < n; ++i) {
            for (Size j = i; j < n; ++j) {
                result.emplace_back(bra_pairs[i], ket_pairs[j]);
            }
        }
    } else {
        result.reserve(bra_pairs.size() * ket_pairs.size());
        for (const auto& bra : bra_pairs) {
            for (const auto& ket : ket_pairs) {
                result.emplace_back(bra, ket);
            }
        }
    }

    return result;
}

// =============================================================================
// Symmetry-Unique Filtering
// =============================================================================

/// @brief Filter quartets to include only symmetry-unique representatives
///
/// For the upper-triangle ShellSetQuartet worklist produced by
/// BasisSet::shell_set_quartets(), the main additional symmetry is the
/// bra/ket exchange when the bra and ket pairs have the same AM signature.
/// This function returns only the canonical representatives.
///
/// @param quartets Source quartets (should already be upper-triangle over pairs)
/// @return Vector of pointers to symmetry-unique quartets
[[nodiscard]] inline std::vector<const ShellSetQuartet*>
filter_symmetry_unique(std::span<const ShellSetQuartet> quartets) {
    std::vector<const ShellSetQuartet*> result;
    result.reserve(quartets.size());

    for (const auto& q : quartets) {
        // A quartet is canonical if bra_pair AM tuple <= ket_pair AM tuple
        // (lexicographically) or if bra == ket (diagonal)
        const auto bra_am = std::make_pair(q.La(), q.Lb());
        const auto ket_am = std::make_pair(q.Lc(), q.Ld());

        if (bra_am <= ket_am) {
            result.push_back(&q);
        }
        // Note: when bra_am > ket_am in an upper-triangle worklist,
        // the transposed pair is also present (since the worklist is
        // over pair indices, not AM). We only keep the canonical form.
    }

    return result;
}

}  // namespace libaccint
