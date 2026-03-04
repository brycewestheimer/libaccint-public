// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_data_integration.cpp
/// @brief Integration tests for basis & data module

#include <gtest/gtest.h>

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/data/basis_parser.hpp>
#include <libaccint/data/auxiliary_basis_data.hpp>
#include <libaccint/data/auxiliary_basis_selector.hpp>
#include <libaccint/data/auxiliary_basis_parser.hpp>
#include <libaccint/utils/error_handling.hpp>

namespace libaccint::testing {

namespace {

/// H2O atoms
std::vector<data::Atom> h2o_atoms() {
    return {
        {8, {0.0, 0.0, 0.0}},
        {1, {0.0, 1.43233673, -1.10866041}},
        {1, {0.0, -1.43233673, -1.10866041}}
    };
}

}  // anonymous namespace

// =============================================================================
// STO-3G Basis Creation Tests
// =============================================================================

TEST(DataIntegrationTest, STO3GH2OShellCount) {
    auto basis = data::create_sto3g(h2o_atoms());

    // STO-3G H2O: O has 1s + 2s + 2p = 3 shells, each H has 1s = 1 shell
    // Total: 3 + 1 + 1 = 5 shells
    EXPECT_EQ(basis.n_shells(), 5u);
}

TEST(DataIntegrationTest, STO3GH2OBasisFunctions) {
    auto basis = data::create_sto3g(h2o_atoms());

    // 7 basis functions: O 1s(1) + O 2s(1) + O 2p(3) + H1 1s(1) + H2 1s(1)
    EXPECT_EQ(basis.n_basis_functions(), 7u);
}

TEST(DataIntegrationTest, STO3GH2OMaxAM) {
    auto basis = data::create_sto3g(h2o_atoms());

    // Max AM should be 1 (p-shell)
    EXPECT_EQ(basis.max_angular_momentum(), 1);
}

TEST(DataIntegrationTest, STO3GViaBuiltinFactory) {
    auto basis = data::create_builtin_basis("sto-3g", h2o_atoms());
    EXPECT_EQ(basis.n_shells(), 5u);
}

TEST(DataIntegrationTest, STO3GCaseInsensitive) {
    auto basis = data::create_builtin_basis("STO-3G", h2o_atoms());
    EXPECT_EQ(basis.n_shells(), 5u);
}

// =============================================================================
// Auxiliary Basis Selector Tests
// =============================================================================

TEST(DataIntegrationTest, SelectorRecommendsDunning) {
    auto rec = data::recommend_auxiliary_basis("cc-pVDZ", FittingType::RI);
    ASSERT_TRUE(rec.has_value());
    EXPECT_EQ(*rec, "cc-pVDZ-RI");
}

TEST(DataIntegrationTest, SelectorRecommendsDunningJKFIT) {
    auto rec = data::recommend_auxiliary_basis("cc-pVDZ", FittingType::JKFIT);
    ASSERT_TRUE(rec.has_value());
    EXPECT_EQ(*rec, "def2-SVP-JKFIT");
}

TEST(DataIntegrationTest, SelectorRecommendsDef2) {
    auto rec = data::recommend_auxiliary_basis("def2-SVP", FittingType::JKFIT);
    ASSERT_TRUE(rec.has_value());
    EXPECT_EQ(*rec, "def2-SVP-JKFIT");
}

TEST(DataIntegrationTest, SelectorRecommendsSTO3G) {
    auto rec_ri = data::recommend_auxiliary_basis("STO-3G", FittingType::RI);
    ASSERT_TRUE(rec_ri.has_value());
    EXPECT_EQ(*rec_ri, "cc-pVDZ-RI");

    auto rec_jk = data::recommend_auxiliary_basis("STO-3G", FittingType::JKFIT);
    ASSERT_TRUE(rec_jk.has_value());
    EXPECT_EQ(*rec_jk, "def2-SVP-JKFIT");
}

TEST(DataIntegrationTest, SelectorUnknownReturnsNullopt) {
    auto rec = data::recommend_auxiliary_basis("completely-unknown-basis");
    EXPECT_FALSE(rec.has_value());
}

TEST(DataIntegrationTest, HasRecommendedAuxiliary) {
    EXPECT_TRUE(data::has_recommended_auxiliary("cc-pVDZ"));
    EXPECT_TRUE(data::has_recommended_auxiliary("def2-SVP"));
    EXPECT_FALSE(data::has_recommended_auxiliary("unknown-basis"));
}

TEST(DataIntegrationTest, ListPairings) {
    auto pairings = data::list_orbital_auxiliary_pairings();
    EXPECT_FALSE(pairings.empty());

    // Should contain at least the main basis sets
    bool has_cc_pvdz = false;
    for (const auto& [orb, aux] : pairings) {
        if (orb == "cc-pvdz") has_cc_pvdz = true;
    }
    EXPECT_TRUE(has_cc_pvdz);
}

// =============================================================================
// Recommend → Create → Verify Round-Trip
// =============================================================================

TEST(DataIntegrationTest, RoundTripRI) {
    // 1. Recommend an auxiliary basis for cc-pVDZ
    auto rec = data::recommend_auxiliary_basis("cc-pVDZ", FittingType::RI);
    ASSERT_TRUE(rec.has_value());

    // 2. Create the recommended auxiliary basis for H2O
    auto atoms = h2o_atoms();
    auto aux = data::create_builtin_auxiliary_basis(*rec, atoms);

    // 3. Verify properties
    EXPECT_FALSE(aux.empty());
    EXPECT_GT(aux.n_shells(), 0u);
    EXPECT_GT(aux.n_functions(), 0u);
    EXPECT_EQ(aux.fitting_type(), FittingType::RI);

    // Verify all shells have valid properties
    for (Size i = 0; i < aux.n_shells(); ++i) {
        const auto& shell = aux.shell(i);
        EXPECT_GE(shell.angular_momentum(), 0);
        EXPECT_GT(shell.n_primitives(), 0u);
    }
}

TEST(DataIntegrationTest, RoundTripJKFIT) {
    auto rec = data::recommend_auxiliary_basis("def2-SVP", FittingType::JKFIT);
    ASSERT_TRUE(rec.has_value());

    auto aux = data::create_builtin_auxiliary_basis(*rec, h2o_atoms());

    EXPECT_FALSE(aux.empty());
    EXPECT_EQ(aux.fitting_type(), FittingType::JKFIT);
}

// =============================================================================
// Auxiliary Basis Data Tests
// =============================================================================

TEST(DataIntegrationTest, ListBuiltinBases) {
    auto bases = data::list_builtin_auxiliary_bases();
    EXPECT_GE(bases.size(), 4u);  // At least 4 built-in bases
}

TEST(DataIntegrationTest, BuiltinAvailability) {
    std::vector<int> h2o_z = {8, 1, 1};
    EXPECT_TRUE(data::is_builtin_auxiliary_available("cc-pVDZ-RI", h2o_z));
    EXPECT_TRUE(data::is_builtin_auxiliary_available("def2-SVP-JKFIT", h2o_z));
    EXPECT_FALSE(data::is_builtin_auxiliary_available("nonexistent", h2o_z));
}

TEST(DataIntegrationTest, BuiltinUnsupportedElement) {
    // Potassium (Z=19) is not in the cc-pVDZ-RI embedded data
    std::vector<int> potassium = {19};
    EXPECT_FALSE(data::is_builtin_auxiliary_available("cc-pVDZ-RI", potassium));
}

// =============================================================================
// Element Symbol Tests
// =============================================================================

TEST(DataIntegrationTest, ElementSymbolLookup) {
    EXPECT_EQ(data::element_symbol_to_z("H"), 1);
    EXPECT_EQ(data::element_symbol_to_z("C"), 6);
    EXPECT_EQ(data::element_symbol_to_z("O"), 8);
    EXPECT_EQ(data::z_to_element_symbol(1), "H");
    EXPECT_EQ(data::z_to_element_symbol(8), "O");
}

// =============================================================================
// Validate Auxiliary Basis Tests
// =============================================================================

TEST(DataIntegrationTest, ValidateBuiltinBasis) {
    auto aux = data::create_builtin_auxiliary_basis("cc-pVDZ-RI", h2o_atoms());
    EXPECT_TRUE(data::validate_auxiliary_basis(aux));
}

TEST(DataIntegrationTest, ValidateEmptyBasisThrows) {
    AuxiliaryBasisSet empty;
    EXPECT_THROW(data::validate_auxiliary_basis(empty), InvalidArgumentException);
}

// =============================================================================
// Pople Star Notation Loading Tests
// =============================================================================

TEST(DataIntegrationTest, LoadBasisSet631GStar) {
    auto basis = data::load_basis_set("6-31G*", h2o_atoms());
    // 6-31G* adds d-polarization on heavy atoms
    EXPECT_GT(basis.n_shells(), 5u);
    EXPECT_GT(basis.n_basis_functions(), 7u);
}

TEST(DataIntegrationTest, LoadBasisSet631GDoubleStar) {
    auto basis = data::load_basis_set("6-31G**", h2o_atoms());
    // 6-31G** adds d on heavy + p on H
    EXPECT_GT(basis.n_basis_functions(), 7u);
}

TEST(DataIntegrationTest, LoadBasisSetDef2SVP) {
    auto basis = data::load_basis_set("def2-SVP", h2o_atoms());
    EXPECT_GT(basis.n_shells(), 0u);
    EXPECT_GT(basis.n_basis_functions(), 0u);
}

TEST(DataIntegrationTest, LoadBasisSetSTO6G) {
    auto basis = data::load_basis_set("STO-6G", h2o_atoms());
    // STO-6G should have the same shell structure as STO-3G (5 shells, 7 functions)
    EXPECT_EQ(basis.n_shells(), 5u);
    EXPECT_EQ(basis.n_basis_functions(), 7u);
}

// =============================================================================
// list_available_basis_sets() Tests
// =============================================================================

TEST(DataIntegrationTest, ListAvailableBasisSets) {
    auto names = data::list_available_basis_sets();
    // We have 43 bundled basis sets
    EXPECT_GE(names.size(), 40u);
}

TEST(DataIntegrationTest, ListIsSorted) {
    auto names = data::list_available_basis_sets();
    ASSERT_FALSE(names.empty());
    for (size_t i = 1; i < names.size(); ++i) {
        EXPECT_LE(names[i - 1], names[i])
            << "List not sorted at index " << i;
    }
}

TEST(DataIntegrationTest, ListContainsKnownBases) {
    auto names = data::list_available_basis_sets();
    auto has = [&](const std::string& name) {
        return std::find(names.begin(), names.end(), name) != names.end();
    };
    EXPECT_TRUE(has("sto-3g"));
    EXPECT_TRUE(has("cc-pvdz"));
    EXPECT_TRUE(has("def2-svp"));
    EXPECT_TRUE(has("6-31g_st"));
}

}  // namespace libaccint::testing
