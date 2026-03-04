// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_load_auxiliary_basis.cpp
/// @brief Unit tests for load_auxiliary_basis factory function

#include <gtest/gtest.h>

#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/utils/error_handling.hpp>

namespace libaccint::testing {

namespace {

/// H2O atom data
const std::vector<int> h2o_Z = {8, 1, 1};
const std::vector<std::array<Real, 3>> h2o_centers = {
    {{0.0, 0.0, 0.0}},
    {{0.0, 1.43233673, -1.10866041}},
    {{0.0, -1.43233673, -1.10866041}}
};

}  // anonymous namespace

// =============================================================================
// load_auxiliary_basis Tests
// =============================================================================

TEST(LoadAuxiliaryBasisTest, LoadCcPvdzRI) {
    auto aux = load_auxiliary_basis("cc-pVDZ-RI", h2o_Z, h2o_centers);

    EXPECT_FALSE(aux.empty());
    EXPECT_GT(aux.n_shells(), 0u);
    EXPECT_GT(aux.n_functions(), 0u);
    EXPECT_EQ(aux.name(), "cc-pVDZ-RI");
    EXPECT_EQ(aux.fitting_type(), FittingType::RI);
}

TEST(LoadAuxiliaryBasisTest, LoadCcPvtzRI) {
    auto aux = load_auxiliary_basis("cc-pVTZ-RI", h2o_Z, h2o_centers);

    EXPECT_FALSE(aux.empty());
    EXPECT_GT(aux.n_shells(), 0u);
    // cc-pVTZ-RI should have more functions than cc-pVDZ-RI
    auto aux_dz = load_auxiliary_basis("cc-pVDZ-RI", h2o_Z, h2o_centers);
    EXPECT_GT(aux.n_functions(), aux_dz.n_functions());
}

TEST(LoadAuxiliaryBasisTest, LoadDef2SvpJkfit) {
    auto aux = load_auxiliary_basis("def2-SVP-JKFIT", h2o_Z, h2o_centers);

    EXPECT_FALSE(aux.empty());
    EXPECT_GT(aux.n_shells(), 0u);
    EXPECT_EQ(aux.fitting_type(), FittingType::JKFIT);
}

TEST(LoadAuxiliaryBasisTest, LoadDef2TzvpJkfit) {
    auto aux = load_auxiliary_basis("def2-TZVP-JKFIT", h2o_Z, h2o_centers);

    EXPECT_FALSE(aux.empty());
    EXPECT_GT(aux.n_shells(), 0u);
    EXPECT_EQ(aux.fitting_type(), FittingType::JKFIT);
}

TEST(LoadAuxiliaryBasisTest, MismatchedSizesThrows) {
    std::vector<int> z = {8, 1};  // 2 atoms
    // 3 centers — mismatch
    EXPECT_THROW(
        load_auxiliary_basis("cc-pVDZ-RI", z, h2o_centers),
        InvalidArgumentException);
}

TEST(LoadAuxiliaryBasisTest, UnknownBasisThrows) {
    EXPECT_THROW(
        load_auxiliary_basis("nonexistent-basis", h2o_Z, h2o_centers),
        InvalidArgumentException);
}

TEST(LoadAuxiliaryBasisTest, CaseInsensitive) {
    // Should accept case-insensitive name matching
    auto aux = load_auxiliary_basis("CC-PVDZ-RI", h2o_Z, h2o_centers);
    EXPECT_FALSE(aux.empty());
}

TEST(LoadAuxiliaryBasisTest, ShellProperties) {
    auto aux = load_auxiliary_basis("cc-pVDZ-RI", h2o_Z, h2o_centers);

    // All shells should have positive exponents
    for (Size i = 0; i < aux.n_shells(); ++i) {
        const auto& shell = aux.shell(i);
        EXPECT_GT(shell.n_primitives(), 0u);
        for (Size j = 0; j < shell.n_primitives(); ++j) {
            EXPECT_GT(shell.exponents()[j], 0.0);
        }
    }
}

// =============================================================================
// available_auxiliary_bases Tests
// =============================================================================

TEST(AvailableAuxiliaryBasesTest, ReturnsKnownBases) {
    auto bases = available_auxiliary_bases();

    EXPECT_FALSE(bases.empty());
    // At least the 4 built-in bases should be listed
    bool has_cc_pvdz_ri = false;
    bool has_def2_svp_jkfit = false;
    for (const auto& name : bases) {
        if (name == "cc-pVDZ-RI") has_cc_pvdz_ri = true;
        if (name == "def2-SVP-JKFIT") has_def2_svp_jkfit = true;
    }
    EXPECT_TRUE(has_cc_pvdz_ri);
    EXPECT_TRUE(has_def2_svp_jkfit);
}

// =============================================================================
// is_auxiliary_basis_available Tests
// =============================================================================

TEST(IsAuxiliaryBasisAvailableTest, KnownBasis) {
    std::vector<int> z = {1, 8};
    EXPECT_TRUE(is_auxiliary_basis_available("cc-pVDZ-RI", z));
}

TEST(IsAuxiliaryBasisAvailableTest, UnknownBasis) {
    std::vector<int> z = {1, 8};
    EXPECT_FALSE(is_auxiliary_basis_available("nonexistent-basis", z));
}

TEST(IsAuxiliaryBasisAvailableTest, UnsupportedElement) {
    // Potassium (Z=19) is not in the cc-pVDZ-RI embedded data
    std::vector<int> z = {19};
    EXPECT_FALSE(is_auxiliary_basis_available("cc-pVDZ-RI", z));
}

}  // namespace libaccint::testing
