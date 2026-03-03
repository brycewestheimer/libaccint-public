// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/operators/operator.hpp>
#include <gtest/gtest.h>
#include <variant>

using namespace libaccint;

// Test fixture for Operator tests
class OperatorTest : public ::testing::Test {
protected:
    static constexpr Real omega_test = 0.3;
};

// Factory construction tests

TEST_F(OperatorTest, OverlapFactory) {
    auto op = Operator::overlap();
    EXPECT_EQ(op.kind(), OperatorKind::Overlap);
    EXPECT_TRUE(op.is_one_electron());
    EXPECT_FALSE(op.is_two_electron());
    EXPECT_TRUE(std::holds_alternative<std::monostate>(op.params()));
}

TEST_F(OperatorTest, KineticFactory) {
    auto op = Operator::kinetic();
    EXPECT_EQ(op.kind(), OperatorKind::Kinetic);
    EXPECT_TRUE(op.is_one_electron());
    EXPECT_FALSE(op.is_two_electron());
    EXPECT_TRUE(std::holds_alternative<std::monostate>(op.params()));
}

TEST_F(OperatorTest, CoulombFactory) {
    auto op = Operator::coulomb();
    EXPECT_EQ(op.kind(), OperatorKind::Coulomb);
    EXPECT_FALSE(op.is_one_electron());
    EXPECT_TRUE(op.is_two_electron());
    EXPECT_TRUE(std::holds_alternative<std::monostate>(op.params()));
}

TEST_F(OperatorTest, NuclearFactory) {
    PointChargeParams params;
    params.x = {0.0, 1.0};
    params.y = {0.0, 0.0};
    params.z = {0.0, 0.0};
    params.charge = {1.0, -1.0};

    auto op = Operator::nuclear(params);
    EXPECT_EQ(op.kind(), OperatorKind::Nuclear);
    EXPECT_TRUE(op.is_one_electron());
    EXPECT_FALSE(op.is_two_electron());
    EXPECT_TRUE(std::holds_alternative<PointChargeParams>(op.params()));

    const auto& retrieved = std::get<PointChargeParams>(op.params());
    EXPECT_EQ(retrieved.n_centers(), 2u);
    EXPECT_EQ(retrieved.x, params.x);
    EXPECT_EQ(retrieved.y, params.y);
    EXPECT_EQ(retrieved.z, params.z);
    EXPECT_EQ(retrieved.charge, params.charge);
}

TEST_F(OperatorTest, PointChargesFactory) {
    PointChargeParams params;
    params.x = {0.0, 1.0, 2.0};
    params.y = {0.0, 0.0, 0.0};
    params.z = {0.0, 0.0, 0.0};
    params.charge = {1.0, -1.0, 0.5};

    auto op = Operator::point_charges(params);
    EXPECT_EQ(op.kind(), OperatorKind::PointCharge);
    EXPECT_TRUE(op.is_one_electron());
    EXPECT_FALSE(op.is_two_electron());
    EXPECT_TRUE(std::holds_alternative<PointChargeParams>(op.params()));

    const auto& retrieved = std::get<PointChargeParams>(op.params());
    EXPECT_EQ(retrieved.n_centers(), 3u);
}

TEST_F(OperatorTest, ErfCoulombFactory) {
    auto op = Operator::erf_coulomb(omega_test);
    EXPECT_EQ(op.kind(), OperatorKind::ErfCoulomb);
    EXPECT_FALSE(op.is_one_electron());
    EXPECT_TRUE(op.is_two_electron());
    EXPECT_TRUE(std::holds_alternative<RangeSeparatedParams>(op.params()));

    const auto& retrieved = std::get<RangeSeparatedParams>(op.params());
    EXPECT_DOUBLE_EQ(retrieved.omega, omega_test);
}

TEST_F(OperatorTest, ErfcCoulombFactory) {
    auto op = Operator::erfc_coulomb(omega_test);
    EXPECT_EQ(op.kind(), OperatorKind::ErfcCoulomb);
    EXPECT_FALSE(op.is_one_electron());
    EXPECT_TRUE(op.is_two_electron());
    EXPECT_TRUE(std::holds_alternative<RangeSeparatedParams>(op.params()));

    const auto& retrieved = std::get<RangeSeparatedParams>(op.params());
    EXPECT_DOUBLE_EQ(retrieved.omega, omega_test);
}

// Kind/params round-trip tests

TEST_F(OperatorTest, KindParamsRoundTrip) {
    PointChargeParams params;
    params.x = {0.0};
    params.y = {0.0};
    params.z = {0.0};
    params.charge = {1.0};

    auto op = Operator::nuclear(params);

    EXPECT_EQ(op.kind(), OperatorKind::Nuclear);
    EXPECT_TRUE(std::holds_alternative<PointChargeParams>(op.params()));

    const auto& retrieved = std::get<PointChargeParams>(op.params());
    EXPECT_EQ(retrieved.charge.size(), 1u);
    EXPECT_DOUBLE_EQ(retrieved.charge[0], 1.0);
}

// params_as<T>() tests

TEST_F(OperatorTest, ParamsAsCorrectType) {
    PointChargeParams params;
    params.x = {0.0, 1.0};
    params.y = {0.0, 0.0};
    params.z = {0.0, 0.0};
    params.charge = {1.0, -1.0};

    auto op = Operator::nuclear(params);

    const auto& retrieved = op.params_as<PointChargeParams>();
    EXPECT_EQ(retrieved.n_centers(), 2u);
    EXPECT_EQ(retrieved.x[0], 0.0);
    EXPECT_EQ(retrieved.x[1], 1.0);
}

TEST_F(OperatorTest, ParamsAsWrongTypeThrows) {
    auto op = Operator::overlap();

    // Trying to get PointChargeParams from an operator with std::monostate should throw
    EXPECT_THROW(op.params_as<PointChargeParams>(), std::bad_variant_access);
}

TEST_F(OperatorTest, ParamsAsRangeSeparated) {
    auto op = Operator::erf_coulomb(omega_test);

    const auto& params = op.params_as<RangeSeparatedParams>();
    EXPECT_DOUBLE_EQ(params.omega, omega_test);
}

// Visitor dispatch tests

TEST_F(OperatorTest, VisitorDispatchMonostate) {
    auto op = Operator::overlap();

    bool visited_monostate = false;
    op.visit_params([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            visited_monostate = true;
        }
    });

    EXPECT_TRUE(visited_monostate);
}

TEST_F(OperatorTest, VisitorDispatchPointCharge) {
    PointChargeParams params;
    params.x = {0.0};
    params.y = {0.0};
    params.z = {0.0};
    params.charge = {1.0};

    auto op = Operator::nuclear(params);

    bool visited_point_charge = false;
    Size n_centers = 0;

    op.visit_params([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, PointChargeParams>) {
            visited_point_charge = true;
            n_centers = arg.n_centers();
        }
    });

    EXPECT_TRUE(visited_point_charge);
    EXPECT_EQ(n_centers, 1u);
}

TEST_F(OperatorTest, VisitorDispatchRangeSeparated) {
    auto op = Operator::erfc_coulomb(omega_test);

    bool visited_range_separated = false;
    Real omega_retrieved = 0.0;

    op.visit_params([&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, RangeSeparatedParams>) {
            visited_range_separated = true;
            omega_retrieved = arg.omega;
        }
    });

    EXPECT_TRUE(visited_range_separated);
    EXPECT_DOUBLE_EQ(omega_retrieved, omega_test);
}

TEST_F(OperatorTest, VisitorWithReturnValue) {
    auto op = Operator::erf_coulomb(omega_test);

    auto omega = op.visit_params([](auto&& arg) -> Real {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, RangeSeparatedParams>) {
            return arg.omega;
        }
        return 0.0;
    });

    EXPECT_DOUBLE_EQ(omega, omega_test);
}

// Copy semantics tests

TEST_F(OperatorTest, CopySharesUnderlyingData) {
    // Create operator with large parameter data
    PointChargeParams params;
    for (int i = 0; i < 1000; ++i) {
        params.x.push_back(static_cast<Real>(i));
        params.y.push_back(static_cast<Real>(i));
        params.z.push_back(static_cast<Real>(i));
        params.charge.push_back(static_cast<Real>(i));
    }

    auto op1 = Operator::nuclear(params);
    auto op2 = op1;  // Copy

    // Both should have the same kind
    EXPECT_EQ(op1.kind(), op2.kind());

    // Both should access the same underlying data (shared_ptr)
    const auto& params1 = op1.params_as<PointChargeParams>();
    const auto& params2 = op2.params_as<PointChargeParams>();

    // Check that data is identical
    EXPECT_EQ(params1.n_centers(), params2.n_centers());
    EXPECT_EQ(params1.n_centers(), 1000u);

    // Verify pointer equality (shared ownership)
    EXPECT_EQ(&op1.params(), &op2.params());
}

TEST_F(OperatorTest, CopyIndependentKind) {
    auto op1 = Operator::overlap();
    auto op2 = op1;

    // After copy, both should have same kind
    EXPECT_EQ(op1.kind(), OperatorKind::Overlap);
    EXPECT_EQ(op2.kind(), OperatorKind::Overlap);
}

// One-electron vs two-electron classification tests

TEST_F(OperatorTest, OneElectronClassification) {
    EXPECT_TRUE(Operator::overlap().is_one_electron());
    EXPECT_TRUE(Operator::kinetic().is_one_electron());

    PointChargeParams params;
    params.x = {0.0};
    params.y = {0.0};
    params.z = {0.0};
    params.charge = {1.0};
    EXPECT_TRUE(Operator::nuclear(params).is_one_electron());
    EXPECT_TRUE(Operator::point_charges(params).is_one_electron());
}

TEST_F(OperatorTest, TwoElectronClassification) {
    EXPECT_TRUE(Operator::coulomb().is_two_electron());
    EXPECT_TRUE(Operator::erf_coulomb(0.3).is_two_electron());
    EXPECT_TRUE(Operator::erfc_coulomb(0.3).is_two_electron());
}

TEST_F(OperatorTest, MutuallyExclusiveClassification) {
    // One-electron operators should not be two-electron
    EXPECT_FALSE(Operator::overlap().is_two_electron());
    EXPECT_FALSE(Operator::kinetic().is_two_electron());

    // Two-electron operators should not be one-electron
    EXPECT_FALSE(Operator::coulomb().is_one_electron());
    EXPECT_FALSE(Operator::erf_coulomb(0.3).is_one_electron());
}

// ============================================================================
// Factory tests for property integral operators (Phase 5)
// ============================================================================

TEST_F(OperatorTest, LinearMomentumFactory) {
    auto op = Operator::linear_momentum();
    EXPECT_EQ(op.kind(), OperatorKind::LinearMomentum);
    EXPECT_TRUE(op.is_one_electron());
    EXPECT_FALSE(op.is_two_electron());
    EXPECT_TRUE(std::holds_alternative<OriginParams>(op.params()));

    const auto& params = op.params_as<OriginParams>();
    EXPECT_DOUBLE_EQ(params.origin[0], 0.0);
    EXPECT_DOUBLE_EQ(params.origin[1], 0.0);
    EXPECT_DOUBLE_EQ(params.origin[2], 0.0);
}

TEST_F(OperatorTest, AngularMomentumFactory) {
    auto op = Operator::angular_momentum();
    EXPECT_EQ(op.kind(), OperatorKind::AngularMomentum);
    EXPECT_TRUE(op.is_one_electron());
    EXPECT_FALSE(op.is_two_electron());
    EXPECT_TRUE(std::holds_alternative<OriginParams>(op.params()));

    const auto& params = op.params_as<OriginParams>();
    EXPECT_DOUBLE_EQ(params.origin[0], 0.0);
    EXPECT_DOUBLE_EQ(params.origin[1], 0.0);
    EXPECT_DOUBLE_EQ(params.origin[2], 0.0);
}

TEST_F(OperatorTest, ElectricDipoleFactory) {
    auto op = Operator::electric_dipole();
    EXPECT_EQ(op.kind(), OperatorKind::ElectricDipole);
    EXPECT_TRUE(op.is_one_electron());
    EXPECT_FALSE(op.is_two_electron());
    EXPECT_TRUE(std::holds_alternative<OriginParams>(op.params()));
}

TEST_F(OperatorTest, ElectricDipoleWithCustomOrigin) {
    OriginParams origin_params;
    origin_params.origin = {1.0, 2.0, 3.0};

    auto op = Operator::electric_dipole(origin_params);
    EXPECT_EQ(op.kind(), OperatorKind::ElectricDipole);

    const auto& params = op.params_as<OriginParams>();
    EXPECT_DOUBLE_EQ(params.origin[0], 1.0);
    EXPECT_DOUBLE_EQ(params.origin[1], 2.0);
    EXPECT_DOUBLE_EQ(params.origin[2], 3.0);
}

TEST_F(OperatorTest, ElectricQuadrupoleFactory) {
    auto op = Operator::electric_quadrupole();
    EXPECT_EQ(op.kind(), OperatorKind::ElectricQuadrupole);
    EXPECT_TRUE(op.is_one_electron());
    EXPECT_FALSE(op.is_two_electron());
    EXPECT_TRUE(std::holds_alternative<OriginParams>(op.params()));
}

TEST_F(OperatorTest, ElectricOctupoleFactory) {
    auto op = Operator::electric_octupole();
    EXPECT_EQ(op.kind(), OperatorKind::ElectricOctupole);
    EXPECT_TRUE(op.is_one_electron());
    EXPECT_FALSE(op.is_two_electron());
    EXPECT_TRUE(std::holds_alternative<OriginParams>(op.params()));
}

TEST_F(OperatorTest, DistributedMultipoleFactory) {
    DistributedMultipoleParams dm_params;
    dm_params.x = {0.0, 1.0};
    dm_params.y = {0.0, 0.0};
    dm_params.z = {0.0, 0.0};
    dm_params.charges = {1.0, -1.0};

    auto op = Operator::distributed_multipole(dm_params);
    EXPECT_EQ(op.kind(), OperatorKind::DistributedMultipole);
    EXPECT_TRUE(op.is_one_electron());
    EXPECT_FALSE(op.is_two_electron());
    EXPECT_TRUE(std::holds_alternative<DistributedMultipoleParams>(op.params()));

    const auto& retrieved = op.params_as<DistributedMultipoleParams>();
    EXPECT_EQ(retrieved.n_sites(), 2u);
    EXPECT_EQ(retrieved.charges[0], 1.0);
    EXPECT_EQ(retrieved.charges[1], -1.0);
    EXPECT_EQ(retrieved.max_rank(), 0);
}

TEST_F(OperatorTest, ProjectionFactory) {
    ProjectionOperatorParams proj_params;
    proj_params.n_basis = 2;
    proj_params.n_projectors = 1;
    proj_params.coefficients = {0.7, 0.7};
    proj_params.weights = {1.0};

    auto op = Operator::projection(proj_params);
    EXPECT_EQ(op.kind(), OperatorKind::ProjectionOperator);
    EXPECT_TRUE(op.is_one_electron());
    EXPECT_FALSE(op.is_two_electron());
    EXPECT_TRUE(std::holds_alternative<ProjectionOperatorParams>(op.params()));

    const auto& retrieved = op.params_as<ProjectionOperatorParams>();
    EXPECT_EQ(retrieved.n_basis, 2u);
    EXPECT_EQ(retrieved.n_projectors, 1u);
    EXPECT_TRUE(retrieved.is_valid());
    EXPECT_DOUBLE_EQ(retrieved.weights[0], 1.0);
}
