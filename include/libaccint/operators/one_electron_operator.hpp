// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file one_electron_operator.hpp
/// @brief Composed one-electron operator for multi-contribution evaluation

#include <libaccint/operators/operator.hpp>
#include <span>
#include <vector>

namespace libaccint {

/// @brief Composition of multiple one-electron operators with scale factors
///
/// Represents a sum of one-electron operators, each with a scale factor.
/// Example: H = T + V_nuc would be represented as:
/// OneElectronOperator h = Operator::kinetic() + Operator::nuclear(charges);
///
/// Operators can be combined with operator+ and scaled with operator*.
/// This class is used by the Engine to compute multiple one-electron integrals
/// in a single pass, improving performance.
class OneElectronOperator {
public:
    /// @brief A single contribution to the composed operator
    struct Contribution {
        Operator op;      ///< The operator
        Real scale{1.0};  ///< Scale factor applied to this operator
    };

    /// @brief Construct from a single operator (implicit conversion allowed)
    /// @param op The operator to wrap
    /// @throws InvalidArgumentException if op is not a one-electron operator
    OneElectronOperator(Operator op);

    /// @brief Add a contribution to the composed operator
    /// @param op The operator to add
    /// @param scale The scale factor (default 1.0)
    /// @throws InvalidArgumentException if op is not a one-electron operator
    void add(Operator op, Real scale = 1.0);

    /// @brief Combine two OneElectronOperators (concatenates contributions)
    /// @param lhs Left-hand side operator
    /// @param rhs Right-hand side operator
    /// @return Combined operator with all contributions from both operands
    friend OneElectronOperator operator+(OneElectronOperator lhs, const OneElectronOperator& rhs);

    /// @brief Scale all contributions by a factor (left multiplication)
    /// @param scale The scale factor
    /// @param op The operator to scale
    /// @return New operator with all contributions scaled
    friend OneElectronOperator operator*(Real scale, OneElectronOperator op);

    /// @brief Scale all contributions by a factor (right multiplication)
    /// @param op The operator to scale
    /// @param scale The scale factor
    /// @return New operator with all contributions scaled
    friend OneElectronOperator operator*(OneElectronOperator op, Real scale);

    /// @brief Get all contributions
    /// @return Span of contributions
    [[nodiscard]] std::span<const Contribution> contributions() const noexcept;

    /// @brief Get the number of contributions
    /// @return Number of contributions
    [[nodiscard]] Size n_contributions() const noexcept;

    /// @brief Check if any contribution is a projection operator
    /// @return True if any contribution has OperatorKind::ProjectionOperator
    [[nodiscard]] bool has_projection_terms() const noexcept;

    /// @brief Check if any contribution is a potential term
    /// @return True if any contribution is Nuclear or PointCharge
    [[nodiscard]] bool has_potential_terms() const noexcept;

private:
    std::vector<Contribution> contributions_;
};

}  // namespace libaccint
