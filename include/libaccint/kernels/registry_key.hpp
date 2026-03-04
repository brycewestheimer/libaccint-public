// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file registry_key.hpp
/// @brief Registry key for kernel dispatch decisions

#include <libaccint/core/backend.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/operators/operator_types.hpp>

#include <array>
#include <cstddef>
#include <functional>

namespace libaccint::kernels {

/// @brief Key for kernel dispatch registry
///
/// Uniquely identifies a computation scenario for dispatch decisions.
/// Used by DispatchRegistry for O(1) lookup of optimal execution strategies.
struct RegistryKey {
    OperatorKind op_kind;                  ///< Type of operator (Overlap, Kinetic, Coulomb, etc.)
    AMQuartet am;                          ///< Angular momentum (La, Lb, Lc, Ld) - use {La, Lb, 0, 0} for 1e
    std::array<int, 4> n_primitives;       ///< Primitive counts per center
    BackendType available_backend;          ///< Available backend (CPU, CUDA)

    /// @brief Default constructor
    RegistryKey() = default;

    /// @brief Full constructor
    RegistryKey(OperatorKind kind, AMQuartet angular_momentum,
                std::array<int, 4> prims, BackendType backend)
        : op_kind(kind), am(angular_momentum), n_primitives(prims), available_backend(backend) {}

    /// @brief Factory method for one-electron integrals
    /// @param kind Operator kind (Overlap, Kinetic, Nuclear)
    /// @param la Angular momentum of shell A
    /// @param lb Angular momentum of shell B
    /// @param na_prim Number of primitives in shell A
    /// @param nb_prim Number of primitives in shell B
    /// @param backend Available backend
    /// @return RegistryKey for this one-electron integral type
    [[nodiscard]] static RegistryKey for_1e(OperatorKind kind, int la, int lb,
                                            int na_prim, int nb_prim,
                                            BackendType backend);

    /// @brief Factory method for two-electron integrals
    /// @param kind Operator kind (Coulomb, ErfCoulomb, etc.)
    /// @param la Angular momentum of shell A
    /// @param lb Angular momentum of shell B
    /// @param lc Angular momentum of shell C
    /// @param ld Angular momentum of shell D
    /// @param na Number of primitives in shell A
    /// @param nb Number of primitives in shell B
    /// @param nc Number of primitives in shell C
    /// @param nd Number of primitives in shell D
    /// @param backend Available backend
    /// @return RegistryKey for this two-electron integral type
    [[nodiscard]] static RegistryKey for_2e(OperatorKind kind,
                                            int la, int lb, int lc, int ld,
                                            int na, int nb, int nc, int nd,
                                            BackendType backend);

    /// @brief Check if this key represents a one-electron integral
    [[nodiscard]] bool is_one_electron() const noexcept {
        return libaccint::is_one_electron(op_kind);
    }

    /// @brief Check if this key represents a two-electron integral
    [[nodiscard]] bool is_two_electron() const noexcept {
        return libaccint::is_two_electron(op_kind);
    }

    /// @brief Get total angular momentum
    [[nodiscard]] int total_am() const noexcept {
        return am[0] + am[1] + am[2] + am[3];
    }

    /// @brief Get total number of primitives (product)
    [[nodiscard]] Size total_primitives() const noexcept {
        return static_cast<Size>(n_primitives[0]) * static_cast<Size>(n_primitives[1]) *
               static_cast<Size>(n_primitives[2]) * static_cast<Size>(n_primitives[3]);
    }

    /// @brief Equality comparison
    bool operator==(const RegistryKey& other) const = default;

    /// @brief Hash function object for use with unordered containers
    struct Hash {
        [[nodiscard]] std::size_t operator()(const RegistryKey& k) const noexcept;
    };
};

}  // namespace libaccint::kernels
