// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

#include <libaccint/operators/operator_types.hpp>
#include <memory>
#include <utility>

namespace libaccint {

/// @brief Operator class representing quantum chemical operators
///
/// Uses the factory pattern to construct operators with their associated parameters.
/// Parameters are stored in a shared_ptr for efficient copying of operators with
/// large parameter data (e.g., many point charges).
class Operator {
public:
    // Factory functions for constructing operators

    /// @brief Create overlap operator (S)
    [[nodiscard]] static Operator overlap();

    /// @brief Create kinetic energy operator (T)
    [[nodiscard]] static Operator kinetic();

    /// @brief Create Coulomb operator (1/r12) for electron repulsion
    [[nodiscard]] static Operator coulomb();

    /// @brief Create nuclear attraction operator with point charges
    /// @param params Point charge positions and charges
    [[nodiscard]] static Operator nuclear(PointChargeParams params);

    /// @brief Create general point charge operator
    /// @param params Point charge positions and charges
    [[nodiscard]] static Operator point_charges(PointChargeParams params);

    /// @brief Create error function Coulomb operator erf(omega*r12)/r12
    /// @param omega Range separation parameter
    [[nodiscard]] static Operator erf_coulomb(Real omega);

    /// @brief Create complementary error function Coulomb operator erfc(omega*r12)/r12
    /// @param omega Range separation parameter
    [[nodiscard]] static Operator erfc_coulomb(Real omega);

    /// @brief Create linear momentum operator -i∇ (3-component, anti-Hermitian)
    /// @param params Origin parameters (default: origin at {0,0,0})
    [[nodiscard]] static Operator linear_momentum(OriginParams params = {});

    /// @brief Create angular momentum operator r×(-i∇) (3-component, anti-Hermitian)
    /// @param params Origin parameters (default: origin at {0,0,0})
    [[nodiscard]] static Operator angular_momentum(OriginParams params = {});

    /// @brief Create electric dipole moment operator <μ|r|ν> (3-component)
    /// @param params Origin parameters (default: origin at {0,0,0})
    [[nodiscard]] static Operator electric_dipole(OriginParams params = {});

    /// @brief Create electric quadrupole moment operator <μ|rr|ν> (6-component)
    /// @param params Origin parameters (default: origin at {0,0,0})
    [[nodiscard]] static Operator electric_quadrupole(OriginParams params = {});

    /// @brief Create electric octupole moment operator <μ|rrr|ν> (10-component)
    /// @param params Origin parameters (default: origin at {0,0,0})
    [[nodiscard]] static Operator electric_octupole(OriginParams params = {});

    /// @brief Create distributed multipole operator
    /// @param params Distributed multipole site data
    [[nodiscard]] static Operator distributed_multipole(DistributedMultipoleParams params);

    /// @brief Create projection operator
    /// @param params Projection operator coefficients and weights
    [[nodiscard]] static Operator projection(ProjectionOperatorParams params);

    // Accessors

    /// @brief Get the operator kind
    [[nodiscard]] OperatorKind kind() const noexcept { return kind_; }

    /// @brief Get the operator parameters
    [[nodiscard]] const OperatorParams& params() const noexcept { return *params_; }

    /// @brief Check if this is a one-electron operator
    [[nodiscard]] bool is_one_electron() const noexcept {
        return libaccint::is_one_electron(kind_);
    }

    /// @brief Check if this is a two-electron operator
    [[nodiscard]] bool is_two_electron() const noexcept {
        return libaccint::is_two_electron(kind_);
    }

    /// @brief Get parameters as a specific type
    /// @tparam T The parameter type to retrieve
    /// @throws std::bad_variant_access if the parameter type doesn't match
    template<typename T>
    [[nodiscard]] const T& params_as() const {
        return std::get<T>(*params_);
    }

    /// @brief Visit parameters with a visitor
    /// @param visitor Callable accepting any OperatorParams alternative
    /// @return The result of the visitor invocation
    template<typename Visitor>
    decltype(auto) visit_params(Visitor&& visitor) const {
        return std::visit(std::forward<Visitor>(visitor), *params_);
    }

private:
    /// @brief Private constructor for factory pattern
    /// @param kind The operator kind
    /// @param params The operator parameters (moved into shared_ptr)
    Operator(OperatorKind kind, OperatorParams params);

    OperatorKind kind_;
    std::shared_ptr<const OperatorParams> params_;
};

}  // namespace libaccint
