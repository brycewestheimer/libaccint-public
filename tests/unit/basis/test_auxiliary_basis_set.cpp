// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_auxiliary_basis_set.cpp
/// @brief Unit tests for AuxiliaryBasisSet class

#include <gtest/gtest.h>

#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/utils/error_handling.hpp>

namespace libaccint::testing {

// =============================================================================
// Test Fixtures
// =============================================================================

class AuxiliaryBasisSetTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create some test auxiliary shells
        // s-shell at origin
        Shell s_shell(0, Point3D{0.0, 0.0, 0.0}, {2.0, 0.5}, {0.5, 0.5});
        s_shell.set_atom_index(0);
        s_shell.set_shell_index(0);
        test_shells_.push_back(std::move(s_shell));

        // p-shell at (1, 0, 0)
        Shell p_shell(1, Point3D{1.0, 0.0, 0.0}, {1.5}, {1.0});
        p_shell.set_atom_index(1);
        p_shell.set_shell_index(1);
        test_shells_.push_back(std::move(p_shell));

        // d-shell at (0, 1, 0)
        Shell d_shell(2, Point3D{0.0, 1.0, 0.0}, {1.0, 0.3}, {0.6, 0.4});
        d_shell.set_atom_index(1);
        d_shell.set_shell_index(2);
        test_shells_.push_back(std::move(d_shell));
    }

    std::vector<Shell> test_shells_;
};

// =============================================================================
// Construction Tests
// =============================================================================

TEST_F(AuxiliaryBasisSetTest, DefaultConstruction) {
    AuxiliaryBasisSet aux;

    EXPECT_EQ(aux.n_shells(), 0u);
    EXPECT_EQ(aux.n_functions(), 0u);
    EXPECT_TRUE(aux.empty());
}

TEST_F(AuxiliaryBasisSetTest, ConstructionFromShells) {
    AuxiliaryBasisSet aux(test_shells_, FittingType::JKFIT, "test-JKFIT");

    EXPECT_EQ(aux.n_shells(), 3u);
    EXPECT_EQ(aux.fitting_type(), FittingType::JKFIT);
    EXPECT_EQ(aux.name(), "test-JKFIT");
    EXPECT_FALSE(aux.empty());
}

TEST_F(AuxiliaryBasisSetTest, FunctionCount) {
    AuxiliaryBasisSet aux(test_shells_);

    // s: 1 function, p: 3 functions, d: 6 functions
    EXPECT_EQ(aux.n_functions(), 1u + 3u + 6u);
}

TEST_F(AuxiliaryBasisSetTest, PrimitiveCount) {
    AuxiliaryBasisSet aux(test_shells_);

    // s: 2 primitives, p: 1 primitive, d: 2 primitives
    EXPECT_EQ(aux.n_primitives(), 2u + 1u + 2u);
}

TEST_F(AuxiliaryBasisSetTest, MaxAngularMomentum) {
    AuxiliaryBasisSet aux(test_shells_);

    EXPECT_EQ(aux.max_angular_momentum(), 2);  // d-shell
}

// =============================================================================
// Indexing Tests
// =============================================================================

TEST_F(AuxiliaryBasisSetTest, ShellAccess) {
    AuxiliaryBasisSet aux(test_shells_);

    EXPECT_EQ(aux.shell(0).angular_momentum(), 0);
    EXPECT_EQ(aux.shell(1).angular_momentum(), 1);
    EXPECT_EQ(aux.shell(2).angular_momentum(), 2);
}

TEST_F(AuxiliaryBasisSetTest, ShellAccessOutOfBounds) {
    AuxiliaryBasisSet aux(test_shells_);

    EXPECT_THROW(aux.shell(5), InvalidArgumentException);
}

TEST_F(AuxiliaryBasisSetTest, ShellToFunction) {
    AuxiliaryBasisSet aux(test_shells_);

    EXPECT_EQ(aux.shell_to_function(0), 0u);   // s-shell starts at 0
    EXPECT_EQ(aux.shell_to_function(1), 1u);   // p-shell starts at 1
    EXPECT_EQ(aux.shell_to_function(2), 4u);   // d-shell starts at 1+3=4
}

TEST_F(AuxiliaryBasisSetTest, FunctionToShell) {
    AuxiliaryBasisSet aux(test_shells_);

    // Function 0 is in s-shell (0)
    EXPECT_EQ(aux.function_to_shell(0), 0u);

    // Functions 1-3 are in p-shell (1)
    EXPECT_EQ(aux.function_to_shell(1), 1u);
    EXPECT_EQ(aux.function_to_shell(2), 1u);
    EXPECT_EQ(aux.function_to_shell(3), 1u);

    // Functions 4-9 are in d-shell (2)
    EXPECT_EQ(aux.function_to_shell(4), 2u);
    EXPECT_EQ(aux.function_to_shell(9), 2u);
}

TEST_F(AuxiliaryBasisSetTest, ShellAndFunctionIndicesAreAssigned) {
    AuxiliaryBasisSet aux(test_shells_);

    EXPECT_EQ(aux.shell(0).shell_index(), 0);
    EXPECT_EQ(aux.shell(1).shell_index(), 1);
    EXPECT_EQ(aux.shell(2).shell_index(), 2);

    EXPECT_EQ(aux.shell(0).function_index(), 0);
    EXPECT_EQ(aux.shell(1).function_index(), 1);
    EXPECT_EQ(aux.shell(2).function_index(), 4);
}

// =============================================================================
// Fitting Type Tests
// =============================================================================

TEST_F(AuxiliaryBasisSetTest, FittingTypeDefault) {
    AuxiliaryBasisSet aux(test_shells_);

    EXPECT_EQ(aux.fitting_type(), FittingType::JKFIT);
}

TEST_F(AuxiliaryBasisSetTest, FittingTypeRI) {
    AuxiliaryBasisSet aux(test_shells_, FittingType::RI);

    EXPECT_EQ(aux.fitting_type(), FittingType::RI);
}

TEST_F(AuxiliaryBasisSetTest, FittingTypeToString) {
    EXPECT_STREQ(fitting_type_to_string(FittingType::RI), "RI");
    EXPECT_STREQ(fitting_type_to_string(FittingType::JKFIT), "JKFIT");
    EXPECT_STREQ(fitting_type_to_string(FittingType::JFIT), "JFIT");
    EXPECT_STREQ(fitting_type_to_string(FittingType::MP2FIT), "MP2FIT");
}

// =============================================================================
// SoA Data Tests
// =============================================================================

TEST_F(AuxiliaryBasisSetTest, SoADataLayout) {
    AuxiliaryBasisSet aux(test_shells_);

    const auto& soa = aux.soa_data();

    EXPECT_EQ(soa.n_shells, 3u);
    EXPECT_EQ(soa.n_functions, 10u);
    EXPECT_EQ(soa.n_primitives, 5u);

    // Check center coordinates
    EXPECT_DOUBLE_EQ(soa.center_x[0], 0.0);
    EXPECT_DOUBLE_EQ(soa.center_x[1], 1.0);
    EXPECT_DOUBLE_EQ(soa.center_y[2], 1.0);

    // Check angular momenta
    EXPECT_EQ(soa.angular_momenta[0], 0);
    EXPECT_EQ(soa.angular_momenta[1], 1);
    EXPECT_EQ(soa.angular_momenta[2], 2);
}

// =============================================================================
// Orbital Basis Pairing Tests
// =============================================================================

TEST_F(AuxiliaryBasisSetTest, NoOrbitalBasisByDefault) {
    AuxiliaryBasisSet aux(test_shells_);

    EXPECT_FALSE(aux.has_orbital_basis());
    EXPECT_THROW(aux.orbital_basis(), InvalidStateException);
}

TEST_F(AuxiliaryBasisSetTest, SetOrbitalBasis) {
    AuxiliaryBasisSet aux(test_shells_);

    // Create a minimal orbital basis
    std::vector<Shell> orb_shells;
    orb_shells.push_back(Shell(0, Point3D{0.0, 0.0, 0.0}, {1.0}, {1.0}));
    BasisSet orbital(std::move(orb_shells));

    aux.set_orbital_basis(orbital);

    EXPECT_TRUE(aux.has_orbital_basis());
    EXPECT_EQ(&aux.orbital_basis(), &orbital);
}

TEST_F(AuxiliaryBasisSetTest, ClearOrbitalBasis) {
    AuxiliaryBasisSet aux(test_shells_);

    std::vector<Shell> orb_shells;
    orb_shells.push_back(Shell(0, Point3D{0.0, 0.0, 0.0}, {1.0}, {1.0}));
    BasisSet orbital(std::move(orb_shells));

    aux.clear_orbital_basis();

    EXPECT_FALSE(aux.has_orbital_basis());
}

// =============================================================================
// Factory Function Tests
// =============================================================================

TEST(AuxiliaryBasisFactoryTest, AvailableBases) {
    auto bases = available_auxiliary_bases();

    EXPECT_FALSE(bases.empty());
    EXPECT_TRUE(std::find(bases.begin(), bases.end(), "def2-SVP-JKFIT") != bases.end());
    EXPECT_TRUE(std::find(bases.begin(), bases.end(), "cc-pVDZ-RI") != bases.end());
}

TEST(AuxiliaryBasisFactoryTest, IsAvailable) {
    std::vector<int> light_atoms = {1, 6, 8};  // H, C, O

    EXPECT_TRUE(is_auxiliary_basis_available("def2-SVP-JKFIT", light_atoms));
    EXPECT_FALSE(is_auxiliary_basis_available("nonexistent-basis", light_atoms));
}

}  // namespace libaccint::testing
