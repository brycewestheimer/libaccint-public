// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_element_symbols.cpp
/// @brief Unit tests for element_symbol_to_z and z_to_element_symbol

#include <gtest/gtest.h>

#include <libaccint/data/auxiliary_basis_parser.hpp>
#include <libaccint/utils/error_handling.hpp>

namespace libaccint::testing {

// =============================================================================
// element_symbol_to_z Tests
// =============================================================================

TEST(ElementSymbolTest, Hydrogen) {
    EXPECT_EQ(data::element_symbol_to_z("H"), 1);
}

TEST(ElementSymbolTest, Helium) {
    EXPECT_EQ(data::element_symbol_to_z("He"), 2);
}

TEST(ElementSymbolTest, Carbon) {
    EXPECT_EQ(data::element_symbol_to_z("C"), 6);
}

TEST(ElementSymbolTest, Oxygen) {
    EXPECT_EQ(data::element_symbol_to_z("O"), 8);
}

TEST(ElementSymbolTest, Iron) {
    EXPECT_EQ(data::element_symbol_to_z("Fe"), 26);
}

TEST(ElementSymbolTest, Krypton) {
    EXPECT_EQ(data::element_symbol_to_z("Kr"), 36);
}

TEST(ElementSymbolTest, Iodine) {
    EXPECT_EQ(data::element_symbol_to_z("I"), 53);
}

TEST(ElementSymbolTest, Gold) {
    EXPECT_EQ(data::element_symbol_to_z("Au"), 79);
}

TEST(ElementSymbolTest, Uranium) {
    EXPECT_EQ(data::element_symbol_to_z("U"), 92);
}

TEST(ElementSymbolTest, Oganesson) {
    EXPECT_EQ(data::element_symbol_to_z("Og"), 118);
}

TEST(ElementSymbolTest, CaseInsensitiveUpperCase) {
    EXPECT_EQ(data::element_symbol_to_z("HE"), 2);
    EXPECT_EQ(data::element_symbol_to_z("FE"), 26);
}

TEST(ElementSymbolTest, CaseInsensitiveLowerCase) {
    EXPECT_EQ(data::element_symbol_to_z("he"), 2);
    EXPECT_EQ(data::element_symbol_to_z("fe"), 26);
}

TEST(ElementSymbolTest, CaseInsensitiveMixed) {
    EXPECT_EQ(data::element_symbol_to_z("hE"), 2);
    EXPECT_EQ(data::element_symbol_to_z("oG"), 118);
}

TEST(ElementSymbolTest, UnknownSymbolThrows) {
    EXPECT_THROW(data::element_symbol_to_z("Xx"), InvalidArgumentException);
    EXPECT_THROW(data::element_symbol_to_z("Zz"), InvalidArgumentException);
    EXPECT_THROW(data::element_symbol_to_z(""), InvalidArgumentException);
}

// =============================================================================
// z_to_element_symbol Tests
// =============================================================================

TEST(ZToSymbolTest, Hydrogen) {
    EXPECT_EQ(data::z_to_element_symbol(1), "H");
}

TEST(ZToSymbolTest, Carbon) {
    EXPECT_EQ(data::z_to_element_symbol(6), "C");
}

TEST(ZToSymbolTest, Iron) {
    EXPECT_EQ(data::z_to_element_symbol(26), "Fe");
}

TEST(ZToSymbolTest, Oganesson) {
    EXPECT_EQ(data::z_to_element_symbol(118), "Og");
}

TEST(ZToSymbolTest, OutOfRangeThrows) {
    EXPECT_THROW(data::z_to_element_symbol(0), InvalidArgumentException);
    EXPECT_THROW(data::z_to_element_symbol(-1), InvalidArgumentException);
    EXPECT_THROW(data::z_to_element_symbol(119), InvalidArgumentException);
}

// =============================================================================
// Round-trip Tests
// =============================================================================

TEST(ElementRoundtripTest, SelectedElements) {
    // Test round-trip for a selection of elements across the periodic table
    const std::vector<std::pair<int, std::string>> elements = {
        {1, "H"}, {2, "He"}, {6, "C"}, {8, "O"}, {9, "F"},
        {11, "Na"}, {17, "Cl"}, {26, "Fe"}, {29, "Cu"}, {35, "Br"},
        {36, "Kr"}, {47, "Ag"}, {53, "I"}, {54, "Xe"}, {74, "W"},
        {78, "Pt"}, {79, "Au"}, {82, "Pb"}, {92, "U"}, {118, "Og"}
    };

    for (const auto& [z, sym] : elements) {
        EXPECT_EQ(data::z_to_element_symbol(z), sym) << "Z=" << z;
        EXPECT_EQ(data::element_symbol_to_z(sym), z) << "Symbol=" << sym;
    }
}

TEST(ElementRoundtripTest, AllFirstRow) {
    // First row transition metals (Sc=21 through Zn=30)
    const std::vector<std::string> syms = {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"
    };
    for (int i = 0; i < static_cast<int>(syms.size()); ++i) {
        int z = 21 + i;
        EXPECT_EQ(data::element_symbol_to_z(syms[static_cast<size_t>(i)]), z);
        EXPECT_EQ(data::z_to_element_symbol(z), syms[static_cast<size_t>(i)]);
    }
}

TEST(ElementRoundtripTest, Lanthanides) {
    EXPECT_EQ(data::element_symbol_to_z("La"), 57);
    EXPECT_EQ(data::element_symbol_to_z("Lu"), 71);
    EXPECT_EQ(data::z_to_element_symbol(57), "La");
    EXPECT_EQ(data::z_to_element_symbol(71), "Lu");
}

TEST(ElementRoundtripTest, Actinides) {
    EXPECT_EQ(data::element_symbol_to_z("Ac"), 89);
    EXPECT_EQ(data::element_symbol_to_z("Lr"), 103);
    EXPECT_EQ(data::z_to_element_symbol(89), "Ac");
    EXPECT_EQ(data::z_to_element_symbol(103), "Lr");
}

TEST(ElementRoundtripTest, SuperheavyElements) {
    EXPECT_EQ(data::element_symbol_to_z("Rf"), 104);
    EXPECT_EQ(data::element_symbol_to_z("Cn"), 112);
    EXPECT_EQ(data::element_symbol_to_z("Nh"), 113);
    EXPECT_EQ(data::element_symbol_to_z("Fl"), 114);
    EXPECT_EQ(data::element_symbol_to_z("Ts"), 117);
}

}  // namespace libaccint::testing
