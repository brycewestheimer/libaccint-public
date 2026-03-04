// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_constants.cpp
/// @brief Unit tests for physical and mathematical constants

#include <gtest/gtest.h>
#include <libaccint/utils/constants.hpp>
#include <cmath>

namespace libaccint {

// ============================================================================
// Mathematical Constants Tests
// ============================================================================

TEST(ConstantsTest, PiValue) {
    // PI should be approximately 3.14159265358979323846
    EXPECT_DOUBLE_EQ(constants::PI, 3.14159265358979323846);
    EXPECT_GT(constants::PI, 3.14);
    EXPECT_LT(constants::PI, 3.15);
}

TEST(ConstantsTest, SqrtPiValue) {
    // SQRT_PI should be approximately 1.77245385090551602729
    EXPECT_DOUBLE_EQ(constants::SQRT_PI, 1.77245385090551602729);
    EXPECT_GT(constants::SQRT_PI, 1.77);
    EXPECT_LT(constants::SQRT_PI, 1.78);
}

TEST(ConstantsTest, SqrtPiSquared) {
    // SQRT_PI squared should equal PI
    double result = constants::SQRT_PI * constants::SQRT_PI;
    EXPECT_NEAR(result, constants::PI, 1e-15);
}

TEST(ConstantsTest, InvSqrtPiValue) {
    // INV_SQRT_PI should be approximately 0.56418958354775628695
    EXPECT_DOUBLE_EQ(constants::INV_SQRT_PI, 0.56418958354775628695);
    EXPECT_GT(constants::INV_SQRT_PI, 0.56);
    EXPECT_LT(constants::INV_SQRT_PI, 0.57);
}

TEST(ConstantsTest, InvSqrtPiProduct) {
    // INV_SQRT_PI * SQRT_PI should equal 1
    double result = constants::INV_SQRT_PI * constants::SQRT_PI;
    EXPECT_NEAR(result, 1.0, 1e-15);
}

TEST(ConstantsTest, TwoPiValue) {
    // TWO_PI should be 2 * PI
    EXPECT_DOUBLE_EQ(constants::TWO_PI, 2.0 * constants::PI);
    EXPECT_GT(constants::TWO_PI, 6.28);
    EXPECT_LT(constants::TWO_PI, 6.29);
}

// ============================================================================
// Physical Constants (CODATA 2018) Tests
// ============================================================================

TEST(ConstantsTest, BohrRadiusValue) {
    // BOHR_RADIUS should be approximately 0.529177210903 × 10⁻¹⁰ m
    EXPECT_DOUBLE_EQ(constants::BOHR_RADIUS, 0.529177210903e-10);
    EXPECT_GT(constants::BOHR_RADIUS, 0.5e-10);
    EXPECT_LT(constants::BOHR_RADIUS, 0.6e-10);
}

TEST(ConstantsTest, HartreeEnergyValue) {
    // HARTREE_ENERGY should be approximately 4.3597447222071 × 10⁻¹⁸ J
    EXPECT_DOUBLE_EQ(constants::HARTREE_ENERGY, 4.3597447222071e-18);
    EXPECT_GT(constants::HARTREE_ENERGY, 4.3e-18);
    EXPECT_LT(constants::HARTREE_ENERGY, 4.4e-18);
}

TEST(ConstantsTest, SpeedOfLightValue) {
    // SPEED_OF_LIGHT should be exactly 299792458 m/s
    EXPECT_DOUBLE_EQ(constants::SPEED_OF_LIGHT, 299792458.0);
    EXPECT_GT(constants::SPEED_OF_LIGHT, 2.99e8);
    EXPECT_LT(constants::SPEED_OF_LIGHT, 3.00e8);
}

TEST(ConstantsTest, ElementaryChargeValue) {
    // ELEMENTARY_CHARGE should be approximately 1.602176634 × 10⁻¹⁹ C
    EXPECT_DOUBLE_EQ(constants::ELEMENTARY_CHARGE, 1.602176634e-19);
    EXPECT_GT(constants::ELEMENTARY_CHARGE, 1.6e-19);
    EXPECT_LT(constants::ELEMENTARY_CHARGE, 1.7e-19);
}

TEST(ConstantsTest, PlanckConstantValue) {
    // PLANCK_CONSTANT should be approximately 6.62607015 × 10⁻³⁴ J⋅s
    EXPECT_DOUBLE_EQ(constants::PLANCK_CONSTANT, 6.62607015e-34);
    EXPECT_GT(constants::PLANCK_CONSTANT, 6.6e-34);
    EXPECT_LT(constants::PLANCK_CONSTANT, 6.7e-34);
}

TEST(ConstantsTest, AvogadroConstantValue) {
    // AVOGADRO_CONSTANT should be approximately 6.02214076 × 10²³ mol⁻¹
    EXPECT_DOUBLE_EQ(constants::AVOGADRO_CONSTANT, 6.02214076e23);
    EXPECT_GT(constants::AVOGADRO_CONSTANT, 6.0e23);
    EXPECT_LT(constants::AVOGADRO_CONSTANT, 6.1e23);
}

// ============================================================================
// Conversion Factor Tests
// ============================================================================

TEST(ConstantsTest, BohrToAngstromValue) {
    // BOHR_TO_ANGSTROM should be approximately 0.529177210903
    EXPECT_DOUBLE_EQ(constants::BOHR_TO_ANGSTROM, 0.529177210903);
    EXPECT_GT(constants::BOHR_TO_ANGSTROM, 0.529);
    EXPECT_LT(constants::BOHR_TO_ANGSTROM, 0.530);
}

TEST(ConstantsTest, AngstromToBohrValue) {
    // ANGSTROM_TO_BOHR should be approximately 1.8897261246257702
    EXPECT_DOUBLE_EQ(constants::ANGSTROM_TO_BOHR, 1.8897261246257702);
    EXPECT_GT(constants::ANGSTROM_TO_BOHR, 1.88);
    EXPECT_LT(constants::ANGSTROM_TO_BOHR, 1.90);
}

TEST(ConstantsTest, BohrAngstromRoundTrip) {
    // Converting Bohr to Angstrom and back should return to 1
    double value_in_bohr = 1.5;
    double value_in_angstrom = value_in_bohr * constants::BOHR_TO_ANGSTROM;
    double back_to_bohr = value_in_angstrom * constants::ANGSTROM_TO_BOHR;
    EXPECT_NEAR(back_to_bohr, value_in_bohr, 1e-14);
}

TEST(ConstantsTest, BohrAngstromRoundTripConversionFactors) {
    // BOHR_TO_ANGSTROM * ANGSTROM_TO_BOHR should equal 1
    double result = constants::BOHR_TO_ANGSTROM * constants::ANGSTROM_TO_BOHR;
    EXPECT_NEAR(result, 1.0, 1e-15);
}

TEST(ConstantsTest, HartreeToEvValue) {
    // HARTREE_TO_EV should be approximately 27.211386245988
    EXPECT_DOUBLE_EQ(constants::HARTREE_TO_EV, 27.211386245988);
    EXPECT_GT(constants::HARTREE_TO_EV, 27.0);
    EXPECT_LT(constants::HARTREE_TO_EV, 27.5);
}

TEST(ConstantsTest, EvToHartreeValue) {
    // EV_TO_HARTREE should be 1/HARTREE_TO_EV
    EXPECT_DOUBLE_EQ(constants::EV_TO_HARTREE, 1.0 / constants::HARTREE_TO_EV);
    EXPECT_GT(constants::EV_TO_HARTREE, 0.03);
    EXPECT_LT(constants::EV_TO_HARTREE, 0.04);
}

TEST(ConstantsTest, HartreeEvRoundTrip) {
    // Converting Hartree to eV and back should return to 1
    double value_in_hartree = 2.5;
    double value_in_ev = value_in_hartree * constants::HARTREE_TO_EV;
    double back_to_hartree = value_in_ev * constants::EV_TO_HARTREE;
    EXPECT_NEAR(back_to_hartree, value_in_hartree, 1e-13);
}

TEST(ConstantsTest, HartreeEvRoundTripConversionFactors) {
    // HARTREE_TO_EV * EV_TO_HARTREE should equal 1
    double result = constants::HARTREE_TO_EV * constants::EV_TO_HARTREE;
    EXPECT_NEAR(result, 1.0, 1e-15);
}

TEST(ConstantsTest, HartreeToKcalMolValue) {
    // HARTREE_TO_KCAL_MOL should be approximately 627.509474063
    EXPECT_DOUBLE_EQ(constants::HARTREE_TO_KCAL_MOL, 627.509474063);
    EXPECT_GT(constants::HARTREE_TO_KCAL_MOL, 627.0);
    EXPECT_LT(constants::HARTREE_TO_KCAL_MOL, 628.0);
}

TEST(ConstantsTest, KcalMolToHartreeValue) {
    // KCAL_MOL_TO_HARTREE should be 1/HARTREE_TO_KCAL_MOL
    EXPECT_DOUBLE_EQ(constants::KCAL_MOL_TO_HARTREE, 1.0 / constants::HARTREE_TO_KCAL_MOL);
    EXPECT_GT(constants::KCAL_MOL_TO_HARTREE, 0.0015);
    EXPECT_LT(constants::KCAL_MOL_TO_HARTREE, 0.0016);
}

TEST(ConstantsTest, HartreeKcalMolRoundTrip) {
    // Converting Hartree to kcal/mol and back should return to 1
    double value_in_hartree = 3.7;
    double value_in_kcal_mol = value_in_hartree * constants::HARTREE_TO_KCAL_MOL;
    double back_to_hartree = value_in_kcal_mol * constants::KCAL_MOL_TO_HARTREE;
    EXPECT_NEAR(back_to_hartree, value_in_hartree, 1e-13);
}

TEST(ConstantsTest, HartreeKcalMolRoundTripConversionFactors) {
    // HARTREE_TO_KCAL_MOL * KCAL_MOL_TO_HARTREE should equal 1
    double result = constants::HARTREE_TO_KCAL_MOL * constants::KCAL_MOL_TO_HARTREE;
    EXPECT_NEAR(result, 1.0, 1e-15);
}

TEST(ConstantsTest, HartreeToKjMolValue) {
    // HARTREE_TO_KJ_MOL should be approximately 2625.499638
    EXPECT_DOUBLE_EQ(constants::HARTREE_TO_KJ_MOL, 2625.499638);
    EXPECT_GT(constants::HARTREE_TO_KJ_MOL, 2625.0);
    EXPECT_LT(constants::HARTREE_TO_KJ_MOL, 2626.0);
}

// ============================================================================
// Threshold Constants Tests
// ============================================================================

TEST(ConstantsTest, IntegralThresholdValue) {
    // INTEGRAL_THRESHOLD should be 1e-12
    EXPECT_DOUBLE_EQ(constants::INTEGRAL_THRESHOLD, 1e-12);
    EXPECT_GT(constants::INTEGRAL_THRESHOLD, 0.0);
    EXPECT_LT(constants::INTEGRAL_THRESHOLD, 1e-11);
}

TEST(ConstantsTest, SchwarzThresholdValue) {
    // SCHWARZ_THRESHOLD should be 1e-10
    EXPECT_DOUBLE_EQ(constants::SCHWARZ_THRESHOLD, 1e-10);
    EXPECT_GT(constants::SCHWARZ_THRESHOLD, 0.0);
    EXPECT_LT(constants::SCHWARZ_THRESHOLD, 1e-9);
}

TEST(ConstantsTest, BoysPrecisionValue) {
    // BOYS_PRECISION should be 1e-14
    EXPECT_DOUBLE_EQ(constants::BOYS_PRECISION, 1e-14);
    EXPECT_GT(constants::BOYS_PRECISION, 0.0);
    EXPECT_LT(constants::BOYS_PRECISION, 1e-13);
}

TEST(ConstantsTest, ThresholdOrdering) {
    // Verify threshold ordering: BOYS_PRECISION < INTEGRAL_THRESHOLD < SCHWARZ_THRESHOLD
    EXPECT_LT(constants::BOYS_PRECISION, constants::INTEGRAL_THRESHOLD);
    EXPECT_LT(constants::INTEGRAL_THRESHOLD, constants::SCHWARZ_THRESHOLD);
}

// ============================================================================
// Atomic Mass Tests
// ============================================================================

TEST(ConstantsTest, ProtonMassValue) {
    // PROTON_MASS should be approximately 1.007276466621
    EXPECT_DOUBLE_EQ(constants::PROTON_MASS, 1.007276466621);
    EXPECT_GT(constants::PROTON_MASS, 1.0);
    EXPECT_LT(constants::PROTON_MASS, 1.01);
}

TEST(ConstantsTest, NeutronMassValue) {
    // NEUTRON_MASS should be approximately 1.00866491595
    EXPECT_DOUBLE_EQ(constants::NEUTRON_MASS, 1.00866491595);
    EXPECT_GT(constants::NEUTRON_MASS, 1.0);
    EXPECT_LT(constants::NEUTRON_MASS, 1.01);
}

TEST(ConstantsTest, ElectronMassValue) {
    // ELECTRON_MASS should be approximately 5.48579909065e-4
    EXPECT_DOUBLE_EQ(constants::ELECTRON_MASS, 5.48579909065e-4);
    EXPECT_GT(constants::ELECTRON_MASS, 5.0e-4);
    EXPECT_LT(constants::ELECTRON_MASS, 6.0e-4);
}

TEST(ConstantsTest, AtomicMassOrdering) {
    // Verify mass ordering: ELECTRON_MASS << PROTON_MASS ≈ NEUTRON_MASS
    EXPECT_LT(constants::ELECTRON_MASS, constants::PROTON_MASS);
    EXPECT_LT(constants::ELECTRON_MASS, constants::NEUTRON_MASS);
    EXPECT_NEAR(constants::PROTON_MASS, constants::NEUTRON_MASS, 0.01);
}

}  // namespace libaccint
