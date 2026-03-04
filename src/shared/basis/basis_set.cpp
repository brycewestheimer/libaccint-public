// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file basis_set.cpp
/// @brief BasisSet class implementation

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <cassert>
#include <memory>
#include <mutex>
#include <string>

namespace libaccint {

// =============================================================================
// Construction
// =============================================================================

BasisSet::BasisSet(std::vector<Shell> shells)
    : shells_(std::move(shells)) {
    if (shells_.empty()) {
        return;
    }

    assign_indices();
    organize_into_shell_sets();
}

// =============================================================================
// Shell Access
// =============================================================================

const Shell& BasisSet::shell(Size i) const {
    if (i >= shells_.size()) {
        throw InvalidArgumentException(
            "BasisSet::shell: index " + std::to_string(i) +
            " out of range [0, " + std::to_string(shells_.size()) + ")");
    }
    return shells_[i];
}

// =============================================================================
// ShellSet Access
// =============================================================================

std::vector<const ShellSet*> BasisSet::shell_sets() const {
    std::vector<const ShellSet*> result;
    result.reserve(shell_sets_.size());
    for (const auto& ptr : shell_sets_) {
        result.push_back(ptr.get());
    }
    return result;
}

const ShellSet* BasisSet::shell_set(int am, int n_primitives) const {
    ShellSetKey key{am, n_primitives};
    auto it = shell_set_index_.find(key);
    if (it == shell_set_index_.end()) {
        return nullptr;
    }
    return shell_sets_[it->second].get();
}

std::vector<const ShellSet*> BasisSet::shell_sets_with_am(int am) const {
    std::vector<const ShellSet*> result;
    for (const auto& ptr : shell_sets_) {
        if (ptr->angular_momentum() == am) {
            result.push_back(ptr.get());
        }
    }
    return result;
}

// =============================================================================
// Pair/Quartet Generation
// =============================================================================

const std::vector<ShellSetPair>& BasisSet::generate_shell_set_pairs_impl() const {
    if (pairs_generated_.load(std::memory_order_acquire)) {
        return pairs_;
    }

    std::lock_guard<std::mutex> lock(*cache_mutex_);
    if (pairs_generated_.load(std::memory_order_acquire)) {
        return pairs_;
    }

    pairs_.clear();
    const Size n = shell_sets_.size();

    // Reserve upper-triangle count: n*(n+1)/2
    pairs_.reserve(n * (n + 1) / 2);

    for (Size i = 0; i < n; ++i) {
        for (Size j = i; j < n; ++j) {
            pairs_.emplace_back(*shell_sets_[i], *shell_sets_[j]);
        }
    }

    pairs_generated_.store(true, std::memory_order_release);
    return pairs_;
}

const std::vector<ShellSetQuartet>& BasisSet::generate_shell_set_quartets_impl() const {
    if (quartets_generated_.load(std::memory_order_acquire)) {
        return quartets_;
    }

    // Ensure pairs are generated and cached (acquires mutex internally if needed)
    generate_shell_set_pairs_impl();

    std::lock_guard<std::mutex> lock(*cache_mutex_);
    if (quartets_generated_.load(std::memory_order_acquire)) {
        return quartets_;
    }

    const auto& pairs = pairs_;
    const Size n = pairs.size();

    // Debug assertion: pair vector address stability
    [[maybe_unused]] const auto* pairs_data_ptr = pairs_.data();

    quartets_.clear();
    // Reserve upper-triangle count: n*(n+1)/2
    quartets_.reserve(n * (n + 1) / 2);

    for (Size i = 0; i < n; ++i) {
        for (Size j = i; j < n; ++j) {
            quartets_.emplace_back(pairs[i], pairs[j]);
        }
    }

    // Verify pair vector was not reallocated during quartet generation
    assert(pairs_.data() == pairs_data_ptr &&
           "ShellSetPair vector reallocated during quartet generation — pointer stability violated");

    quartets_generated_.store(true, std::memory_order_release);
    return quartets_;
}

// =============================================================================
// Atom Queries
// =============================================================================

std::vector<const Shell*> BasisSet::shells_on_atom(Index atom_idx) const {
    std::vector<const Shell*> result;
    for (const auto& s : shells_) {
        if (s.atom_index() == atom_idx) {
            result.push_back(&s);
        }
    }
    return result;
}

// =============================================================================
// Private Helpers
// =============================================================================

void BasisSet::assign_indices() {
    Index bf_offset = 0;
    Index bf_offset_sph = 0;
    max_am_ = 0;
    max_primitives_ = 0;

    for (Size i = 0; i < shells_.size(); ++i) {
        shells_[i].set_shell_index(static_cast<Index>(i));
        shells_[i].set_function_index(bf_offset);
        bf_offset += shells_[i].n_functions();
        bf_offset_sph += n_spherical(shells_[i].angular_momentum());

        if (shells_[i].angular_momentum() > max_am_) {
            max_am_ = shells_[i].angular_momentum();
        }

        const int n_prim = static_cast<int>(shells_[i].n_primitives());
        if (n_prim > max_primitives_) {
            max_primitives_ = n_prim;
        }
    }

    n_basis_functions_ = static_cast<Size>(bf_offset);
    n_basis_functions_sph_ = static_cast<Size>(bf_offset_sph);
}

Size BasisSet::n_basis_functions_spherical() const noexcept {
    return n_basis_functions_sph_;
}

void BasisSet::organize_into_shell_sets() {
    shell_sets_.clear();
    shell_set_index_.clear();

    for (const auto& s : shells_) {
        ShellSetKey key{s.angular_momentum(),
                        static_cast<int>(s.n_primitives())};

        auto it = shell_set_index_.find(key);
        if (it == shell_set_index_.end()) {
            // Create a new ShellSet for this key
            Size idx = shell_sets_.size();
            shell_sets_.push_back(
                std::make_unique<ShellSet>(key.angular_momentum, key.n_primitives));
            shell_set_index_[key] = idx;
            shell_sets_[idx]->add_shell(s);
        } else {
            shell_sets_[it->second]->add_shell(s);
        }
    }
}

}  // namespace libaccint
