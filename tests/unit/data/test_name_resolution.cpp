// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_name_resolution.cpp
/// @brief Unit tests for basis set name-to-filename resolution

#include <gtest/gtest.h>

#include <libaccint/data/basis_parser.hpp>

namespace libaccint::testing {

// =============================================================================
// name_to_filename() Unit Tests
// =============================================================================

TEST(NameResolutionTest, SimpleLowercase) {
    EXPECT_EQ(data::name_to_filename("sto-3g"), "sto-3g.json");
}

TEST(NameResolutionTest, CaseInsensitive) {
    EXPECT_EQ(data::name_to_filename("cc-pVDZ"), "cc-pvdz.json");
    EXPECT_EQ(data::name_to_filename("CC-PVDZ"), "cc-pvdz.json");
    EXPECT_EQ(data::name_to_filename("STO-3G"), "sto-3g.json");
}

TEST(NameResolutionTest, SpacesToHyphens) {
    EXPECT_EQ(data::name_to_filename("cc pVDZ"), "cc-pvdz.json");
}

TEST(NameResolutionTest, PopleSingleStar) {
    EXPECT_EQ(data::name_to_filename("6-31G*"), "6-31g_st.json");
    EXPECT_EQ(data::name_to_filename("6-311G*"), "6-311g_st.json");
    EXPECT_EQ(data::name_to_filename("6-31+G*"), "6-31+g_st.json");
    EXPECT_EQ(data::name_to_filename("6-311+G*"), "6-311+g_st.json");
}

TEST(NameResolutionTest, PopleDoubleStar) {
    EXPECT_EQ(data::name_to_filename("6-31G**"), "6-31g_ss.json");
    EXPECT_EQ(data::name_to_filename("6-311G**"), "6-311g_ss.json");
    EXPECT_EQ(data::name_to_filename("6-31++G**"), "6-31++g_ss.json");
    EXPECT_EQ(data::name_to_filename("6-311++G**"), "6-311++g_ss.json");
}

TEST(NameResolutionTest, DoubleStarBeforeSingleStar) {
    // Ensure ** is replaced first, not two separate * replacements
    auto result = data::name_to_filename("6-31G**");
    EXPECT_EQ(result, "6-31g_ss.json");
    // Should NOT produce "6-31g_st_st.json"
    EXPECT_NE(result, "6-31g_st_st.json");
}

TEST(NameResolutionTest, PassthroughAlreadyEncoded) {
    // Names that already use _st / _ss encoding pass through
    EXPECT_EQ(data::name_to_filename("6-31g_st"), "6-31g_st.json");
    EXPECT_EQ(data::name_to_filename("6-31g_ss"), "6-31g_ss.json");
}

TEST(NameResolutionTest, DunningPassthrough) {
    EXPECT_EQ(data::name_to_filename("cc-pVTZ"), "cc-pvtz.json");
    EXPECT_EQ(data::name_to_filename("aug-cc-pVDZ"), "aug-cc-pvdz.json");
}

TEST(NameResolutionTest, Def2Passthrough) {
    EXPECT_EQ(data::name_to_filename("def2-SVP"), "def2-svp.json");
    EXPECT_EQ(data::name_to_filename("def2-TZVPP"), "def2-tzvpp.json");
}

}  // namespace libaccint::testing
