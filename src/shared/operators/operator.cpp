// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/operators/operator.hpp>
#include <utility>

namespace libaccint {

// Private constructor
Operator::Operator(OperatorKind kind, OperatorParams params)
    : kind_(kind)
    , params_(std::make_shared<const OperatorParams>(std::move(params)))
{}

// Factory functions

Operator Operator::overlap() {
    return Operator(OperatorKind::Overlap, std::monostate{});
}

Operator Operator::kinetic() {
    return Operator(OperatorKind::Kinetic, std::monostate{});
}

Operator Operator::coulomb() {
    return Operator(OperatorKind::Coulomb, std::monostate{});
}

Operator Operator::nuclear(PointChargeParams params) {
    return Operator(OperatorKind::Nuclear, std::move(params));
}

Operator Operator::point_charges(PointChargeParams params) {
    return Operator(OperatorKind::PointCharge, std::move(params));
}

Operator Operator::erf_coulomb(Real omega) {
    return Operator(OperatorKind::ErfCoulomb, RangeSeparatedParams{omega});
}

Operator Operator::erfc_coulomb(Real omega) {
    return Operator(OperatorKind::ErfcCoulomb, RangeSeparatedParams{omega});
}

Operator Operator::linear_momentum(OriginParams params) {
    return Operator(OperatorKind::LinearMomentum, std::move(params));
}

Operator Operator::angular_momentum(OriginParams params) {
    return Operator(OperatorKind::AngularMomentum, std::move(params));
}

Operator Operator::electric_dipole(OriginParams params) {
    return Operator(OperatorKind::ElectricDipole, std::move(params));
}

Operator Operator::electric_quadrupole(OriginParams params) {
    return Operator(OperatorKind::ElectricQuadrupole, std::move(params));
}

Operator Operator::electric_octupole(OriginParams params) {
    return Operator(OperatorKind::ElectricOctupole, std::move(params));
}

Operator Operator::distributed_multipole(DistributedMultipoleParams params) {
    return Operator(OperatorKind::DistributedMultipole, std::move(params));
}

Operator Operator::projection(ProjectionOperatorParams params) {
    return Operator(OperatorKind::ProjectionOperator, std::move(params));
}

}  // namespace libaccint
