// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_operator_types.cpp
/// @brief Unit tests for operator types, classifications, and parameters

#include <gtest/gtest.h>
#include <libaccint/operators/operator_types.hpp>

namespace libaccint {

// ============================================================================
// Test OperatorKind Enum Values
// ============================================================================

TEST(OperatorTypesTest, OperatorKindEnumValues) {
    // Verify all operator kinds exist and can be constructed
    [[maybe_unused]] auto overlap = OperatorKind::Overlap;
    [[maybe_unused]] auto kinetic = OperatorKind::Kinetic;
    [[maybe_unused]] auto nuclear = OperatorKind::Nuclear;
    [[maybe_unused]] auto point_charge = OperatorKind::PointCharge;
    [[maybe_unused]] auto distributed_multipole = OperatorKind::DistributedMultipole;
    [[maybe_unused]] auto projection = OperatorKind::ProjectionOperator;
    [[maybe_unused]] auto coulomb = OperatorKind::Coulomb;
    [[maybe_unused]] auto erf_coulomb = OperatorKind::ErfCoulomb;
    [[maybe_unused]] auto erfc_coulomb = OperatorKind::ErfcCoulomb;
    [[maybe_unused]] auto electric_dipole = OperatorKind::ElectricDipole;
    [[maybe_unused]] auto electric_quadrupole = OperatorKind::ElectricQuadrupole;
    [[maybe_unused]] auto electric_octupole = OperatorKind::ElectricOctupole;
    [[maybe_unused]] auto linear_momentum = OperatorKind::LinearMomentum;
    [[maybe_unused]] auto angular_momentum = OperatorKind::AngularMomentum;

    // Test passes if all enum values are accessible
    SUCCEED();
}

// ============================================================================
// Test is_one_electron Classification
// ============================================================================

TEST(OperatorTypesTest, OneElectronClassification) {
    // One-electron operators should return true
    EXPECT_TRUE(is_one_electron(OperatorKind::Overlap));
    EXPECT_TRUE(is_one_electron(OperatorKind::Kinetic));
    EXPECT_TRUE(is_one_electron(OperatorKind::Nuclear));
    EXPECT_TRUE(is_one_electron(OperatorKind::PointCharge));
    EXPECT_TRUE(is_one_electron(OperatorKind::DistributedMultipole));
    EXPECT_TRUE(is_one_electron(OperatorKind::ProjectionOperator));
    EXPECT_TRUE(is_one_electron(OperatorKind::ElectricDipole));
    EXPECT_TRUE(is_one_electron(OperatorKind::ElectricQuadrupole));
    EXPECT_TRUE(is_one_electron(OperatorKind::ElectricOctupole));
    EXPECT_TRUE(is_one_electron(OperatorKind::LinearMomentum));
    EXPECT_TRUE(is_one_electron(OperatorKind::AngularMomentum));

    // Two-electron operators should return false
    EXPECT_FALSE(is_one_electron(OperatorKind::Coulomb));
    EXPECT_FALSE(is_one_electron(OperatorKind::ErfCoulomb));
    EXPECT_FALSE(is_one_electron(OperatorKind::ErfcCoulomb));
}

// ============================================================================
// Test is_two_electron Classification
// ============================================================================

TEST(OperatorTypesTest, TwoElectronClassification) {
    // Two-electron operators should return true
    EXPECT_TRUE(is_two_electron(OperatorKind::Coulomb));
    EXPECT_TRUE(is_two_electron(OperatorKind::ErfCoulomb));
    EXPECT_TRUE(is_two_electron(OperatorKind::ErfcCoulomb));

    // One-electron operators should return false
    EXPECT_FALSE(is_two_electron(OperatorKind::Overlap));
    EXPECT_FALSE(is_two_electron(OperatorKind::Kinetic));
    EXPECT_FALSE(is_two_electron(OperatorKind::Nuclear));
    EXPECT_FALSE(is_two_electron(OperatorKind::PointCharge));
    EXPECT_FALSE(is_two_electron(OperatorKind::DistributedMultipole));
    EXPECT_FALSE(is_two_electron(OperatorKind::ProjectionOperator));
    EXPECT_FALSE(is_two_electron(OperatorKind::ElectricDipole));
    EXPECT_FALSE(is_two_electron(OperatorKind::ElectricQuadrupole));
    EXPECT_FALSE(is_two_electron(OperatorKind::ElectricOctupole));
    EXPECT_FALSE(is_two_electron(OperatorKind::LinearMomentum));
    EXPECT_FALSE(is_two_electron(OperatorKind::AngularMomentum));
}

// ============================================================================
// Test Classification Consistency
// ============================================================================

TEST(OperatorTypesTest, ClassificationConsistency) {
    // Every operator should be exactly one of one-electron or two-electron
    const std::array all_operators = {
        OperatorKind::Overlap,
        OperatorKind::Kinetic,
        OperatorKind::Nuclear,
        OperatorKind::PointCharge,
        OperatorKind::DistributedMultipole,
        OperatorKind::ProjectionOperator,
        OperatorKind::Coulomb,
        OperatorKind::ErfCoulomb,
        OperatorKind::ErfcCoulomb,
        OperatorKind::ElectricDipole,
        OperatorKind::ElectricQuadrupole,
        OperatorKind::ElectricOctupole,
        OperatorKind::LinearMomentum,
        OperatorKind::AngularMomentum,
    };

    for (auto op : all_operators) {
        // Each operator should be exactly one type (XOR)
        EXPECT_TRUE(is_one_electron(op) != is_two_electron(op))
            << "Operator " << to_string(op) << " classification inconsistent";
    }
}

// ============================================================================
// Test to_string Conversion
// ============================================================================

TEST(OperatorTypesTest, ToStringConversion) {
    // Test all operator kinds return non-empty strings
    EXPECT_EQ(to_string(OperatorKind::Overlap), "Overlap");
    EXPECT_EQ(to_string(OperatorKind::Kinetic), "Kinetic");
    EXPECT_EQ(to_string(OperatorKind::Nuclear), "Nuclear");
    EXPECT_EQ(to_string(OperatorKind::PointCharge), "PointCharge");
    EXPECT_EQ(to_string(OperatorKind::DistributedMultipole), "DistributedMultipole");
    EXPECT_EQ(to_string(OperatorKind::ProjectionOperator), "ProjectionOperator");
    EXPECT_EQ(to_string(OperatorKind::Coulomb), "Coulomb");
    EXPECT_EQ(to_string(OperatorKind::ErfCoulomb), "ErfCoulomb");
    EXPECT_EQ(to_string(OperatorKind::ErfcCoulomb), "ErfcCoulomb");
    EXPECT_EQ(to_string(OperatorKind::ElectricDipole), "ElectricDipole");
    EXPECT_EQ(to_string(OperatorKind::ElectricQuadrupole), "ElectricQuadrupole");
    EXPECT_EQ(to_string(OperatorKind::ElectricOctupole), "ElectricOctupole");
    EXPECT_EQ(to_string(OperatorKind::LinearMomentum), "LinearMomentum");
    EXPECT_EQ(to_string(OperatorKind::AngularMomentum), "AngularMomentum");

    // Verify all strings are non-empty
    const std::array all_operators = {
        OperatorKind::Overlap,
        OperatorKind::Kinetic,
        OperatorKind::Nuclear,
        OperatorKind::PointCharge,
        OperatorKind::DistributedMultipole,
        OperatorKind::ProjectionOperator,
        OperatorKind::Coulomb,
        OperatorKind::ErfCoulomb,
        OperatorKind::ErfcCoulomb,
        OperatorKind::ElectricDipole,
        OperatorKind::ElectricQuadrupole,
        OperatorKind::ElectricOctupole,
        OperatorKind::LinearMomentum,
        OperatorKind::AngularMomentum,
    };

    for (auto op : all_operators) {
        auto str = to_string(op);
        EXPECT_FALSE(str.empty()) << "to_string returned empty for operator";
        EXPECT_GT(str.size(), 0) << "to_string returned zero-length for operator";
    }
}

// ============================================================================
// Test PointChargeParams
// ============================================================================

TEST(OperatorTypesTest, PointChargeParamsEmpty) {
    PointChargeParams params;

    // Empty params should have zero centers
    EXPECT_EQ(params.n_centers(), 0);
    EXPECT_TRUE(params.x.empty());
    EXPECT_TRUE(params.y.empty());
    EXPECT_TRUE(params.z.empty());
    EXPECT_TRUE(params.charge.empty());
}

TEST(OperatorTypesTest, PointChargeParamsSingleCenter) {
    PointChargeParams params;
    params.x.push_back(1.0);
    params.y.push_back(2.0);
    params.z.push_back(3.0);
    params.charge.push_back(6.0);  // Carbon atom

    EXPECT_EQ(params.n_centers(), 1);
    EXPECT_EQ(params.x[0], 1.0);
    EXPECT_EQ(params.y[0], 2.0);
    EXPECT_EQ(params.z[0], 3.0);
    EXPECT_EQ(params.charge[0], 6.0);
}

TEST(OperatorTypesTest, PointChargeParamsMultipleCenters) {
    PointChargeParams params;

    // Add three centers (e.g., water molecule: O, H, H)
    params.x = {0.0, 1.0, -1.0};
    params.y = {0.0, 0.5, 0.5};
    params.z = {0.0, 0.0, 0.0};
    params.charge = {8.0, 1.0, 1.0};  // O=8, H=1, H=1

    EXPECT_EQ(params.n_centers(), 3);
    EXPECT_EQ(params.x.size(), 3);
    EXPECT_EQ(params.y.size(), 3);
    EXPECT_EQ(params.z.size(), 3);
    EXPECT_EQ(params.charge.size(), 3);

    // Verify values
    EXPECT_DOUBLE_EQ(params.charge[0], 8.0);
    EXPECT_DOUBLE_EQ(params.charge[1], 1.0);
    EXPECT_DOUBLE_EQ(params.charge[2], 1.0);
}

// ============================================================================
// Test RangeSeparatedParams
// ============================================================================

TEST(OperatorTypesTest, RangeSeparatedParamsDefault) {
    RangeSeparatedParams params;

    // Default omega should be zero
    EXPECT_DOUBLE_EQ(params.omega, 0.0);
}

TEST(OperatorTypesTest, RangeSeparatedParamsCustomOmega) {
    RangeSeparatedParams params;
    params.omega = 0.3;  // Typical value for range-separated functionals

    EXPECT_DOUBLE_EQ(params.omega, 0.3);
}

// ============================================================================
// Test OperatorParams Variant
// ============================================================================

TEST(OperatorTypesTest, OperatorParamsMonostate) {
    OperatorParams params;

    // Default-constructed variant should hold std::monostate
    EXPECT_TRUE(std::holds_alternative<std::monostate>(params));
}

TEST(OperatorTypesTest, OperatorParamsPointCharge) {
    PointChargeParams pc_params;
    pc_params.x = {0.0};
    pc_params.y = {0.0};
    pc_params.z = {0.0};
    pc_params.charge = {1.0};

    OperatorParams params = pc_params;

    // Verify variant holds PointChargeParams
    EXPECT_TRUE(std::holds_alternative<PointChargeParams>(params));
    EXPECT_FALSE(std::holds_alternative<std::monostate>(params));

    // Verify data is preserved
    const auto& retrieved = std::get<PointChargeParams>(params);
    EXPECT_EQ(retrieved.n_centers(), 1);
    EXPECT_DOUBLE_EQ(retrieved.charge[0], 1.0);
}

TEST(OperatorTypesTest, OperatorParamsRangeSeparated) {
    RangeSeparatedParams rs_params;
    rs_params.omega = 0.4;

    OperatorParams params = rs_params;

    // Verify variant holds RangeSeparatedParams
    EXPECT_TRUE(std::holds_alternative<RangeSeparatedParams>(params));
    EXPECT_FALSE(std::holds_alternative<std::monostate>(params));

    // Verify data is preserved
    const auto& retrieved = std::get<RangeSeparatedParams>(params);
    EXPECT_DOUBLE_EQ(retrieved.omega, 0.4);
}

TEST(OperatorTypesTest, OperatorParamsDistributedMultipole) {
    DistributedMultipoleParams dm_params;
    OperatorParams params = dm_params;

    // Verify variant can hold DistributedMultipoleParams
    EXPECT_TRUE(std::holds_alternative<DistributedMultipoleParams>(params));
}

TEST(OperatorTypesTest, OperatorParamsProjectionOperator) {
    ProjectionOperatorParams po_params;
    OperatorParams params = po_params;

    // Verify variant can hold ProjectionOperatorParams
    EXPECT_TRUE(std::holds_alternative<ProjectionOperatorParams>(params));
}

// ============================================================================
// Test Variant Type Switching
// ============================================================================

TEST(OperatorTypesTest, OperatorParamsSwitching) {
    OperatorParams params;

    // Start with monostate
    EXPECT_TRUE(std::holds_alternative<std::monostate>(params));

    // Switch to PointChargeParams
    PointChargeParams pc;
    pc.charge = {1.0};
    params = pc;
    EXPECT_TRUE(std::holds_alternative<PointChargeParams>(params));

    // Switch to RangeSeparatedParams
    RangeSeparatedParams rs;
    rs.omega = 0.5;
    params = rs;
    EXPECT_TRUE(std::holds_alternative<RangeSeparatedParams>(params));
    EXPECT_FALSE(std::holds_alternative<PointChargeParams>(params));
}

}  // namespace libaccint
