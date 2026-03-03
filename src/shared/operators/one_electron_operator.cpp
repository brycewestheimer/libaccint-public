// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/utils/error_handling.hpp>
#include <algorithm>

namespace libaccint {

OneElectronOperator::OneElectronOperator(Operator op) {
    if (!op.is_one_electron()) {
        throw InvalidArgumentException(
            "Operator must be a one-electron operator for OneElectronOperator construction"
        );
    }
    contributions_.push_back(Contribution{std::move(op), 1.0});
}

void OneElectronOperator::add(Operator op, Real scale) {
    if (!op.is_one_electron()) {
        throw InvalidArgumentException(
            "Only one-electron operators can be added to OneElectronOperator"
        );
    }
    contributions_.push_back(Contribution{std::move(op), scale});
}

OneElectronOperator operator+(OneElectronOperator lhs, const OneElectronOperator& rhs) {
    // Append all contributions from rhs to lhs
    lhs.contributions_.reserve(lhs.contributions_.size() + rhs.contributions_.size());
    lhs.contributions_.insert(
        lhs.contributions_.end(),
        rhs.contributions_.begin(),
        rhs.contributions_.end()
    );
    return lhs;
}

OneElectronOperator operator*(Real scale, OneElectronOperator op) {
    // Scale all contributions
    for (auto& contrib : op.contributions_) {
        contrib.scale *= scale;
    }
    return op;
}

OneElectronOperator operator*(OneElectronOperator op, Real scale) {
    // Delegate to left multiplication
    return scale * std::move(op);
}

std::span<const OneElectronOperator::Contribution> OneElectronOperator::contributions() const noexcept {
    return contributions_;
}

Size OneElectronOperator::n_contributions() const noexcept {
    return contributions_.size();
}

bool OneElectronOperator::has_projection_terms() const noexcept {
    return std::any_of(
        contributions_.begin(),
        contributions_.end(),
        [](const Contribution& c) {
            return c.op.kind() == OperatorKind::ProjectionOperator;
        }
    );
}

bool OneElectronOperator::has_potential_terms() const noexcept {
    return std::any_of(
        contributions_.begin(),
        contributions_.end(),
        [](const Contribution& c) {
            return c.op.kind() == OperatorKind::Nuclear ||
                   c.op.kind() == OperatorKind::PointCharge;
        }
    );
}

}  // namespace libaccint
