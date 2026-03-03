// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_auxiliary_basis_data.cpp
/// @brief Tests for built-in auxiliary basis set data

#include <libaccint/data/auxiliary_basis_data.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <gtest/gtest.h>

namespace libaccint::data {
namespace {

// ============================================================================
// Built-in basis availability tests
// ============================================================================

TEST(BuiltinAuxBasis, ListAvailable) {
    auto bases = list_builtin_auxiliary_bases();
    EXPECT_GE(bases.size(), 4u);

    // Check all four required bases are present
    auto has = [&](const std::string& name) {
        return std::find(bases.begin(), bases.end(), name) != bases.end();
    };
    EXPECT_TRUE(has("cc-pVDZ-RI"));
    EXPECT_TRUE(has("cc-pVTZ-RI"));
    EXPECT_TRUE(has("def2-SVP-JKFIT"));
    EXPECT_TRUE(has("def2-TZVP-JKFIT"));
}

TEST(BuiltinAuxBasis, IsAvailableForCommonElements) {
    std::vector<int> common = {1, 6, 7, 8, 9};
    EXPECT_TRUE(is_builtin_auxiliary_available("cc-pVDZ-RI", common));
    EXPECT_TRUE(is_builtin_auxiliary_available("cc-pVTZ-RI", common));
    EXPECT_TRUE(is_builtin_auxiliary_available("def2-SVP-JKFIT", common));
    EXPECT_TRUE(is_builtin_auxiliary_available("def2-TZVP-JKFIT", common));
}

TEST(BuiltinAuxBasis, NotAvailableForUnsupportedElement) {
    std::vector<int> heavy = {50};  // Tin
    EXPECT_FALSE(is_builtin_auxiliary_available("cc-pVDZ-RI", heavy));
}

TEST(BuiltinAuxBasis, UnknownBasisNotAvailable) {
    std::vector<int> h = {1};
    EXPECT_FALSE(is_builtin_auxiliary_available("nonexistent-basis", h));
}

// ============================================================================
// Basis creation tests
// ============================================================================

class BuiltinAuxBasisCreation : public ::testing::TestWithParam<std::string> {
protected:
    std::vector<Atom> h2o_atoms() {
        return {
            {8, {0.0, 0.0, 0.2217}},
            {1, {0.0, 1.4309, -0.8867}},
            {1, {0.0, -1.4309, -0.8867}},
        };
    }
};

TEST_P(BuiltinAuxBasisCreation, CreateForH2O) {
    const auto& basis_name = GetParam();
    auto atoms = h2o_atoms();
    auto aux = create_builtin_auxiliary_basis(basis_name, atoms);

    EXPECT_FALSE(aux.empty());
    EXPECT_GT(aux.n_shells(), 0u);
    EXPECT_GT(aux.n_functions(), 0u);
    EXPECT_GT(aux.n_primitives(), 0u);
    EXPECT_EQ(aux.name(), basis_name);
}

TEST_P(BuiltinAuxBasisCreation, ShellProperties) {
    const auto& basis_name = GetParam();
    auto atoms = h2o_atoms();
    auto aux = create_builtin_auxiliary_basis(basis_name, atoms);

    // All shells should have positive exponents and valid AM
    for (Size i = 0; i < aux.n_shells(); ++i) {
        const auto& shell = aux.shell(i);
        EXPECT_GE(shell.angular_momentum(), 0);
        EXPECT_LE(shell.angular_momentum(), MAX_ANGULAR_MOMENTUM);
        EXPECT_GT(shell.n_primitives(), 0u);

        for (Size j = 0; j < shell.n_primitives(); ++j) {
            EXPECT_GT(shell.exponents()[j], 0.0);
        }
    }
}

TEST_P(BuiltinAuxBasisCreation, FunctionOffsetsConsistent) {
    const auto& basis_name = GetParam();
    auto atoms = h2o_atoms();
    auto aux = create_builtin_auxiliary_basis(basis_name, atoms);

    // Function offsets should be monotonically increasing
    Size total = 0;
    for (Size i = 0; i < aux.n_shells(); ++i) {
        EXPECT_EQ(aux.shell_to_function(i), total);
        total += static_cast<Size>(aux.shell(i).n_functions());
    }
    EXPECT_EQ(total, aux.n_functions());
}

INSTANTIATE_TEST_SUITE_P(
    AllBuiltinBases,
    BuiltinAuxBasisCreation,
    ::testing::Values(
        "cc-pVDZ-RI",
        "cc-pVTZ-RI",
        "def2-SVP-JKFIT",
        "def2-TZVP-JKFIT"
    ));

// ============================================================================
// Error handling tests
// ============================================================================

TEST(BuiltinAuxBasis, UnknownBasisThrows) {
    std::vector<Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    EXPECT_THROW(
        create_builtin_auxiliary_basis("nonexistent", atoms),
        InvalidArgumentException);
}

TEST(BuiltinAuxBasis, UnsupportedElementThrows) {
    std::vector<Atom> atoms = {{50, {0.0, 0.0, 0.0}}};  // Sn
    EXPECT_THROW(
        create_builtin_auxiliary_basis("cc-pVDZ-RI", atoms),
        InvalidArgumentException);
}

// ============================================================================
// Relative size tests (TZ > DZ expected)
// ============================================================================

TEST(BuiltinAuxBasis, TZLargerThanDZ) {
    std::vector<Atom> atoms = {{8, {0.0, 0.0, 0.0}}};

    auto dz = create_builtin_auxiliary_basis("cc-pVDZ-RI", atoms);
    auto tz = create_builtin_auxiliary_basis("cc-pVTZ-RI", atoms);

    EXPECT_GT(tz.n_functions(), dz.n_functions());
    EXPECT_GE(tz.max_angular_momentum(), dz.max_angular_momentum());
}

TEST(BuiltinAuxBasis, TZVPLargerThanSVP) {
    std::vector<Atom> atoms = {{6, {0.0, 0.0, 0.0}}};

    auto svp = create_builtin_auxiliary_basis("def2-SVP-JKFIT", atoms);
    auto tzvp = create_builtin_auxiliary_basis("def2-TZVP-JKFIT", atoms);

    EXPECT_GT(tzvp.n_functions(), svp.n_functions());
}

// ============================================================================
// SoA data tests
// ============================================================================

TEST(BuiltinAuxBasis, SoADataConsistent) {
    std::vector<Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {0.0, 0.0, 1.5}},
    };

    auto aux = create_builtin_auxiliary_basis("cc-pVDZ-RI", atoms);
    const auto& soa = aux.soa_data();

    EXPECT_EQ(soa.n_shells, aux.n_shells());
    EXPECT_EQ(soa.n_functions, aux.n_functions());
    EXPECT_EQ(soa.n_primitives, aux.n_primitives());

    EXPECT_EQ(soa.center_x.size(), aux.n_shells());
    EXPECT_EQ(soa.exponents.size(), aux.n_primitives());
    EXPECT_EQ(soa.angular_momenta.size(), aux.n_shells());
}

}  // anonymous namespace
}  // namespace libaccint::data
