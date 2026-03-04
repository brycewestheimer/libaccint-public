// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_property_integrals.cpp
/// @brief Unit tests for property integral classification functions

#include <libaccint/operators/operator_types.hpp>
#include <gtest/gtest.h>

namespace libaccint::testing {

// ============================================================================
// All 14 OperatorKind values for comprehensive testing
// ============================================================================

static constexpr std::array all_operator_kinds = {
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

// ============================================================================
// is_multi_component tests
// ============================================================================

TEST(PropertyIntegralClassificationTest, MultiComponentTrue) {
    EXPECT_TRUE(is_multi_component(OperatorKind::ElectricDipole));
    EXPECT_TRUE(is_multi_component(OperatorKind::ElectricQuadrupole));
    EXPECT_TRUE(is_multi_component(OperatorKind::ElectricOctupole));
    EXPECT_TRUE(is_multi_component(OperatorKind::LinearMomentum));
    EXPECT_TRUE(is_multi_component(OperatorKind::AngularMomentum));
}

TEST(PropertyIntegralClassificationTest, MultiComponentFalse) {
    EXPECT_FALSE(is_multi_component(OperatorKind::Overlap));
    EXPECT_FALSE(is_multi_component(OperatorKind::Kinetic));
    EXPECT_FALSE(is_multi_component(OperatorKind::Nuclear));
    EXPECT_FALSE(is_multi_component(OperatorKind::PointCharge));
    EXPECT_FALSE(is_multi_component(OperatorKind::DistributedMultipole));
    EXPECT_FALSE(is_multi_component(OperatorKind::ProjectionOperator));
    EXPECT_FALSE(is_multi_component(OperatorKind::Coulomb));
    EXPECT_FALSE(is_multi_component(OperatorKind::ErfCoulomb));
    EXPECT_FALSE(is_multi_component(OperatorKind::ErfcCoulomb));
}

// ============================================================================
// is_anti_hermitian tests
// ============================================================================

TEST(PropertyIntegralClassificationTest, AntiHermitianTrue) {
    EXPECT_TRUE(is_anti_hermitian(OperatorKind::LinearMomentum));
    EXPECT_TRUE(is_anti_hermitian(OperatorKind::AngularMomentum));
}

TEST(PropertyIntegralClassificationTest, AntiHermitianFalse) {
    EXPECT_FALSE(is_anti_hermitian(OperatorKind::Overlap));
    EXPECT_FALSE(is_anti_hermitian(OperatorKind::Kinetic));
    EXPECT_FALSE(is_anti_hermitian(OperatorKind::Nuclear));
    EXPECT_FALSE(is_anti_hermitian(OperatorKind::PointCharge));
    EXPECT_FALSE(is_anti_hermitian(OperatorKind::DistributedMultipole));
    EXPECT_FALSE(is_anti_hermitian(OperatorKind::ProjectionOperator));
    EXPECT_FALSE(is_anti_hermitian(OperatorKind::Coulomb));
    EXPECT_FALSE(is_anti_hermitian(OperatorKind::ErfCoulomb));
    EXPECT_FALSE(is_anti_hermitian(OperatorKind::ErfcCoulomb));
    EXPECT_FALSE(is_anti_hermitian(OperatorKind::ElectricDipole));
    EXPECT_FALSE(is_anti_hermitian(OperatorKind::ElectricQuadrupole));
    EXPECT_FALSE(is_anti_hermitian(OperatorKind::ElectricOctupole));
}

// ============================================================================
// is_property_integral tests
// ============================================================================

TEST(PropertyIntegralClassificationTest, PropertyIntegralTrue) {
    EXPECT_TRUE(is_property_integral(OperatorKind::ElectricDipole));
    EXPECT_TRUE(is_property_integral(OperatorKind::ElectricQuadrupole));
    EXPECT_TRUE(is_property_integral(OperatorKind::ElectricOctupole));
    EXPECT_TRUE(is_property_integral(OperatorKind::LinearMomentum));
    EXPECT_TRUE(is_property_integral(OperatorKind::AngularMomentum));
}

TEST(PropertyIntegralClassificationTest, PropertyIntegralFalse) {
    EXPECT_FALSE(is_property_integral(OperatorKind::Overlap));
    EXPECT_FALSE(is_property_integral(OperatorKind::Kinetic));
    EXPECT_FALSE(is_property_integral(OperatorKind::Nuclear));
    EXPECT_FALSE(is_property_integral(OperatorKind::PointCharge));
    EXPECT_FALSE(is_property_integral(OperatorKind::DistributedMultipole));
    EXPECT_FALSE(is_property_integral(OperatorKind::ProjectionOperator));
    EXPECT_FALSE(is_property_integral(OperatorKind::Coulomb));
    EXPECT_FALSE(is_property_integral(OperatorKind::ErfCoulomb));
    EXPECT_FALSE(is_property_integral(OperatorKind::ErfcCoulomb));
}

// ============================================================================
// component_count tests
// ============================================================================

TEST(PropertyIntegralClassificationTest, ComponentCountMultiComponent) {
    EXPECT_EQ(component_count(OperatorKind::ElectricDipole), 3);
    EXPECT_EQ(component_count(OperatorKind::ElectricQuadrupole), 6);
    EXPECT_EQ(component_count(OperatorKind::ElectricOctupole), 10);
    EXPECT_EQ(component_count(OperatorKind::LinearMomentum), 3);
    EXPECT_EQ(component_count(OperatorKind::AngularMomentum), 3);
}

TEST(PropertyIntegralClassificationTest, ComponentCountSingleComponent) {
    EXPECT_EQ(component_count(OperatorKind::Overlap), 1);
    EXPECT_EQ(component_count(OperatorKind::Kinetic), 1);
    EXPECT_EQ(component_count(OperatorKind::Nuclear), 1);
    EXPECT_EQ(component_count(OperatorKind::PointCharge), 1);
    EXPECT_EQ(component_count(OperatorKind::DistributedMultipole), 1);
    EXPECT_EQ(component_count(OperatorKind::ProjectionOperator), 1);
    EXPECT_EQ(component_count(OperatorKind::Coulomb), 1);
    EXPECT_EQ(component_count(OperatorKind::ErfCoulomb), 1);
    EXPECT_EQ(component_count(OperatorKind::ErfcCoulomb), 1);
}

// ============================================================================
// Cross-classification consistency tests
// ============================================================================

TEST(PropertyIntegralClassificationTest, AllPropertyIntegralsAreOneElectron) {
    for (auto kind : all_operator_kinds) {
        if (is_property_integral(kind)) {
            EXPECT_TRUE(is_one_electron(kind))
                << operator_name(kind) << " is a property integral but not one-electron";
        }
    }
}

TEST(PropertyIntegralClassificationTest, AllMultiComponentArePropertyIntegrals) {
    for (auto kind : all_operator_kinds) {
        if (is_multi_component(kind)) {
            EXPECT_TRUE(is_property_integral(kind))
                << operator_name(kind) << " is multi-component but not a property integral";
        }
    }
}

TEST(PropertyIntegralClassificationTest, AllAntiHermitianAreMultiComponent) {
    for (auto kind : all_operator_kinds) {
        if (is_anti_hermitian(kind)) {
            EXPECT_TRUE(is_multi_component(kind))
                << operator_name(kind) << " is anti-Hermitian but not multi-component";
        }
    }
}

TEST(PropertyIntegralClassificationTest, ComponentCountConsistentWithMultiComponent) {
    for (auto kind : all_operator_kinds) {
        if (is_multi_component(kind)) {
            EXPECT_GT(component_count(kind), 1)
                << operator_name(kind) << " is multi-component but has component_count <= 1";
        } else {
            EXPECT_EQ(component_count(kind), 1)
                << operator_name(kind) << " is not multi-component but has component_count != 1";
        }
    }
}

}  // namespace libaccint::testing
