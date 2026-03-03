// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file shell_set.cpp
/// @brief ShellSet class implementation

#include <libaccint/basis/shell_set.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <string>

namespace libaccint {

// =============================================================================
// ShellSet Construction
// =============================================================================

ShellSet::ShellSet(std::span<const std::reference_wrapper<const Shell>> shells) {
    if (shells.empty()) {
        throw InvalidArgumentException(
            "ShellSet: cannot construct from empty shell list");
    }

    // Extract AM and K from the first shell
    const Shell& first = shells[0].get();
    am_ = first.angular_momentum();
    n_primitives_ = static_cast<int>(first.n_primitives());

    // Validate all shells have matching AM and K
    for (Size i = 1; i < shells.size(); ++i) {
        const Shell& s = shells[i].get();
        if (s.angular_momentum() != am_) {
            throw InvalidArgumentException(
                "ShellSet: angular momentum mismatch at shell " +
                std::to_string(i) + " (expected L=" +
                std::to_string(am_) + ", got L=" +
                std::to_string(s.angular_momentum()) + ")");
        }
        if (static_cast<int>(s.n_primitives()) != n_primitives_) {
            throw InvalidArgumentException(
                "ShellSet: primitive count mismatch at shell " +
                std::to_string(i) + " (expected K=" +
                std::to_string(n_primitives_) + ", got K=" +
                std::to_string(s.n_primitives()) + ")");
        }
    }

    // Store copies
    shells_.reserve(shells.size());
    for (const auto& ref : shells) {
        shells_.push_back(ref.get());
    }
}

ShellSet::ShellSet(int am, int n_primitives)
    : am_(am)
    , n_primitives_(n_primitives) {
    if (am < 0 || am > MAX_ANGULAR_MOMENTUM) {
        throw InvalidArgumentException(
            "ShellSet: angular momentum must be in [0, " +
            std::to_string(MAX_ANGULAR_MOMENTUM) + "], got " +
            std::to_string(am));
    }
    if (n_primitives < 1) {
        throw InvalidArgumentException(
            "ShellSet: must have at least 1 primitive, got " +
            std::to_string(n_primitives));
    }
}

void ShellSet::add_shell(const Shell& shell) {
    if (shell.angular_momentum() != am_) {
        throw InvalidArgumentException(
            "ShellSet::add_shell: angular momentum mismatch (expected L=" +
            std::to_string(am_) + ", got L=" +
            std::to_string(shell.angular_momentum()) + ")");
    }
    if (static_cast<int>(shell.n_primitives()) != n_primitives_) {
        throw InvalidArgumentException(
            "ShellSet::add_shell: primitive count mismatch (expected K=" +
            std::to_string(n_primitives_) + ", got K=" +
            std::to_string(shell.n_primitives()) + ")");
    }

    shells_.push_back(shell);

    // Invalidate cached SoA data by resetting the once_flag and state.
    // Note: std::once_flag is not resettable, so we reconstruct it via
    // placement new. This is safe because no concurrent call_once can be
    // in progress during add_shell (adding shells and reading SoA data
    // concurrently is a data race on shells_ regardless).
    soa_initialized_ = false;
    new (&soa_init_flag_) std::once_flag{};
}

// =============================================================================
// Accessors
// =============================================================================

const Shell& ShellSet::shell(Size i) const {
    if (i >= shells_.size()) {
        throw InvalidArgumentException(
            "ShellSet::shell: index " + std::to_string(i) +
            " out of range [0, " + std::to_string(shells_.size()) + ")");
    }
    return shells_[i];
}

// =============================================================================
// SoA Data
// =============================================================================

const ShellSetDataSoA& ShellSet::soa_data() const {
    std::call_once(soa_init_flag_, [this]() {
        build_soa_data();
    });
    return soa_data_;
}

void ShellSet::build_soa_data() const {
    LIBACCINT_ASSERT(!shells_.empty(),
        "ShellSet::build_soa_data: cannot build SoA data for empty ShellSet");

    const Size n = shells_.size();
    const Size k = static_cast<Size>(n_primitives_);
    const Size total_prims = n * k;

    ShellSetDataSoA data;

    // Reserve all arrays
    data.center_x.reserve(n);
    data.center_y.reserve(n);
    data.center_z.reserve(n);
    data.exponents.reserve(total_prims);
    data.coefficients.reserve(total_prims);
    data.shell_indices.reserve(n);
    data.atom_indices.reserve(n);
    data.function_offsets.reserve(n);

    // Populate arrays
    for (const auto& shell : shells_) {
        // Centers
        data.center_x.push_back(shell.center().x);
        data.center_y.push_back(shell.center().y);
        data.center_z.push_back(shell.center().z);

        // Tracking indices
        data.shell_indices.push_back(shell.shell_index());
        data.atom_indices.push_back(shell.atom_index());
        data.function_offsets.push_back(shell.function_index());

        // Primitive data: contiguous per shell
        auto exps = shell.exponents();
        auto coeffs = shell.coefficients();
        data.exponents.insert(data.exponents.end(), exps.begin(), exps.end());
        data.coefficients.insert(data.coefficients.end(), coeffs.begin(), coeffs.end());
    }

    soa_data_ = std::move(data);
    soa_initialized_ = true;
}

}  // namespace libaccint
