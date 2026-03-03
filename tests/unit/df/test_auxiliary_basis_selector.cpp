// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_auxiliary_basis_selector.cpp
/// @brief Tests for auxiliary basis auto-selection utility

#include <libaccint/data/auxiliary_basis_selector.hpp>

#include <gtest/gtest.h>

namespace libaccint::data {
namespace {

// ============================================================================
// Auto-selection tests
// ============================================================================

TEST(AuxBasisSelector, DunningDZ) {
    auto rec = recommend_auxiliary_basis("cc-pVDZ", FittingType::RI);
    ASSERT_TRUE(rec.has_value());
    EXPECT_EQ(rec.value(), "cc-pVDZ-RI");
}

TEST(AuxBasisSelector, DunningTZ) {
    auto rec = recommend_auxiliary_basis("cc-pVTZ", FittingType::RI);
    ASSERT_TRUE(rec.has_value());
    EXPECT_EQ(rec.value(), "cc-pVTZ-RI");
}

TEST(AuxBasisSelector, Def2SVP) {
    auto rec = recommend_auxiliary_basis("def2-SVP", FittingType::JKFIT);
    ASSERT_TRUE(rec.has_value());
    EXPECT_EQ(rec.value(), "def2-SVP-JKFIT");
}

TEST(AuxBasisSelector, Def2TZVP) {
    auto rec = recommend_auxiliary_basis("def2-TZVP");
    ASSERT_TRUE(rec.has_value());
    EXPECT_EQ(rec.value(), "def2-TZVP-JKFIT");
}

TEST(AuxBasisSelector, Def2TZVPP) {
    auto rec = recommend_auxiliary_basis("def2-TZVPP");
    ASSERT_TRUE(rec.has_value());
    EXPECT_EQ(rec.value(), "def2-TZVP-JKFIT");
}

TEST(AuxBasisSelector, STO3GFallback) {
    auto rec = recommend_auxiliary_basis("STO-3G");
    ASSERT_TRUE(rec.has_value());
    // Default fitting type is JKFIT; STO-3G maps to def2-SVP-JKFIT
    EXPECT_EQ(rec.value(), "def2-SVP-JKFIT");
}

TEST(AuxBasisSelector, CaseInsensitive) {
    auto rec1 = recommend_auxiliary_basis("CC-PVDZ");
    auto rec2 = recommend_auxiliary_basis("cc-pvdz");
    ASSERT_TRUE(rec1.has_value());
    ASSERT_TRUE(rec2.has_value());
    EXPECT_EQ(rec1.value(), rec2.value());
}

TEST(AuxBasisSelector, UnknownBasisReturnsNullopt) {
    auto rec = recommend_auxiliary_basis("completely-unknown-basis");
    EXPECT_FALSE(rec.has_value());
}

TEST(AuxBasisSelector, HasRecommended) {
    EXPECT_TRUE(has_recommended_auxiliary("cc-pVDZ"));
    EXPECT_TRUE(has_recommended_auxiliary("def2-SVP"));
    EXPECT_FALSE(has_recommended_auxiliary("unknown-basis"));
}

TEST(AuxBasisSelector, PairingListNotEmpty) {
    auto pairings = list_orbital_auxiliary_pairings();
    EXPECT_GT(pairings.size(), 0u);

    // Each pairing should have non-empty strings
    for (const auto& [orbital, aux] : pairings) {
        EXPECT_FALSE(orbital.empty());
        EXPECT_FALSE(aux.empty());
    }
}

TEST(AuxBasisSelector, PopleBasisMapping) {
    auto rec = recommend_auxiliary_basis("6-31G*");
    ASSERT_TRUE(rec.has_value());
    // Default fitting type is JKFIT; Pople bases map to def2-SVP-JKFIT
    EXPECT_EQ(rec.value(), "def2-SVP-JKFIT");
}

TEST(AuxBasisSelector, RIvJKFIT) {
    // RI fitting type should give RI auxiliary
    auto ri = recommend_auxiliary_basis("cc-pVDZ", FittingType::RI);
    ASSERT_TRUE(ri.has_value());
    EXPECT_NE(ri.value().find("RI"), std::string::npos);

    // JKFIT fitting type gives JKFIT auxiliary (for def2 family)
    auto jk = recommend_auxiliary_basis("def2-SVP", FittingType::JKFIT);
    ASSERT_TRUE(jk.has_value());
    EXPECT_NE(jk.value().find("JKFIT"), std::string::npos);
}

}  // anonymous namespace
}  // namespace libaccint::data
