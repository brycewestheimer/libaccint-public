// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_auxiliary_basis_parser.cpp
/// @brief Tests for auxiliary basis set parser (Gaussian94 + BSE JSON)

#include <libaccint/data/auxiliary_basis_parser.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <gtest/gtest.h>

namespace libaccint::data {
namespace {

// ============================================================================
// Element symbol tests
// ============================================================================

TEST(ElementSymbol, HydrogenToZ) {
    EXPECT_EQ(element_symbol_to_z("H"), 1);
    EXPECT_EQ(element_symbol_to_z("h"), 1);
}

TEST(ElementSymbol, CarbonToZ) {
    EXPECT_EQ(element_symbol_to_z("C"), 6);
    EXPECT_EQ(element_symbol_to_z("c"), 6);
}

TEST(ElementSymbol, OxygenToZ) {
    EXPECT_EQ(element_symbol_to_z("O"), 8);
}

TEST(ElementSymbol, FluorineToZ) {
    EXPECT_EQ(element_symbol_to_z("F"), 9);
}

TEST(ElementSymbol, UnknownThrows) {
    EXPECT_THROW(element_symbol_to_z("Xx"), InvalidArgumentException);
}

TEST(ElementSymbol, ZToSymbol) {
    EXPECT_EQ(z_to_element_symbol(1), "H");
    EXPECT_EQ(z_to_element_symbol(6), "C");
    EXPECT_EQ(z_to_element_symbol(8), "O");
}

TEST(ElementSymbol, InvalidZThrows) {
    EXPECT_THROW(z_to_element_symbol(0), InvalidArgumentException);
    EXPECT_THROW(z_to_element_symbol(119), InvalidArgumentException);
}

// ============================================================================
// Gaussian94 parser tests
// ============================================================================

class Gaussian94ParserTest : public ::testing::Test {
protected:
    // Simple Gaussian94 format auxiliary basis for H and O
    static constexpr const char* simple_g94 = R"(
****
H     0
S   1   1.00
      9.000000    1.000000
S   1   1.00
      1.500000    1.000000
P   1   1.00
      2.500000    1.000000
****
O     0
S   1   1.00
    340.000000    1.000000
S   1   1.00
     63.000000    1.000000
P   1   1.00
     40.000000    1.000000
D   1   1.00
     13.000000    1.000000
****
)";

    std::vector<Atom> h2o_atoms() {
        return {
            {8, {0.0, 0.0, 0.2217}},
            {1, {0.0, 1.4309, -0.8867}},
            {1, {0.0, -1.4309, -0.8867}},
        };
    }
};

TEST_F(Gaussian94ParserTest, ParseSimple) {
    auto atoms = h2o_atoms();
    auto aux = parse_auxiliary_basis_gaussian94(
        simple_g94, atoms, FittingType::RI, "test-aux");

    EXPECT_FALSE(aux.empty());
    EXPECT_GT(aux.n_shells(), 0u);
    EXPECT_GT(aux.n_functions(), 0u);
    EXPECT_EQ(aux.fitting_type(), FittingType::RI);
    EXPECT_EQ(aux.name(), "test-aux");
}

TEST_F(Gaussian94ParserTest, CorrectShellCount) {
    auto atoms = h2o_atoms();
    auto aux = parse_auxiliary_basis_gaussian94(simple_g94, atoms);

    // O: 2s + 1p + 1d = 4 shells
    // H1: 2s + 1p = 3 shells
    // H2: 2s + 1p = 3 shells
    // Total: 10 shells
    EXPECT_EQ(aux.n_shells(), 10u);
}

TEST_F(Gaussian94ParserTest, MissingElementThrows) {
    std::vector<Atom> atoms = {{2, {0.0, 0.0, 0.0}}};  // He not in data
    EXPECT_THROW(
        parse_auxiliary_basis_gaussian94(simple_g94, atoms),
        InvalidArgumentException);
}

TEST_F(Gaussian94ParserTest, EmptyContentThrows) {
    auto atoms = h2o_atoms();
    // Empty content produces no element data → missing element
    EXPECT_THROW(
        parse_auxiliary_basis_gaussian94("", atoms),
        InvalidArgumentException);
}

// ============================================================================
// BSE JSON parser tests
// ============================================================================

class BSEJsonParserTest : public ::testing::Test {
protected:
    // Simple BSE JSON format for H
    static constexpr const char* simple_json = R"({
        "elements": {
            "1": {
                "electron_shells": [
                    {
                        "angular_momentum": [0],
                        "exponents": ["9.0000000", "1.5000000"],
                        "coefficients": [
                            ["1.0000000", "0.0000000"],
                            ["0.0000000", "1.0000000"]
                        ]
                    },
                    {
                        "angular_momentum": [1],
                        "exponents": ["2.5000000"],
                        "coefficients": [
                            ["1.0000000"]
                        ]
                    }
                ]
            }
        }
    })";

    std::vector<Atom> h_atom() {
        return {{1, {0.0, 0.0, 0.0}}};
    }
};

TEST_F(BSEJsonParserTest, ParseSimple) {
    auto atoms = h_atom();
    auto aux = parse_auxiliary_basis_json(simple_json, atoms, FittingType::JKFIT, "test-jk");

    EXPECT_FALSE(aux.empty());
    EXPECT_GT(aux.n_shells(), 0u);
    EXPECT_EQ(aux.fitting_type(), FittingType::JKFIT);
    EXPECT_EQ(aux.name(), "test-jk");
}

TEST_F(BSEJsonParserTest, CorrectShellCount) {
    auto atoms = h_atom();
    auto aux = parse_auxiliary_basis_json(simple_json, atoms);

    // 2 s-shells (from general contraction) + 1 p-shell = 3 shells
    EXPECT_EQ(aux.n_shells(), 3u);
}

TEST_F(BSEJsonParserTest, MalformedJsonThrows) {
    auto atoms = h_atom();
    EXPECT_THROW(
        parse_auxiliary_basis_json("{bad json", atoms),
        InvalidArgumentException);
}

TEST_F(BSEJsonParserTest, MissingElementsThrows) {
    auto atoms = h_atom();
    EXPECT_THROW(
        parse_auxiliary_basis_json(R"({"data": 42})", atoms),
        InvalidArgumentException);
}

TEST_F(BSEJsonParserTest, MissingAtomicNumberThrows) {
    std::vector<Atom> atoms = {{2, {0.0, 0.0, 0.0}}};  // He not in JSON
    EXPECT_THROW(
        parse_auxiliary_basis_json(simple_json, atoms),
        InvalidArgumentException);
}

// ============================================================================
// Validation tests
// ============================================================================

TEST(AuxBasisValidation, ValidBasis) {
    std::vector<Shell> shells = {
        Shell(0, Point3D{0.0, 0.0, 0.0}, {1.0}, {1.0}),
        Shell(1, Point3D{0.0, 0.0, 0.0}, {0.5}, {1.0}),
    };
    AuxiliaryBasisSet aux(std::move(shells), FittingType::RI, "test");
    EXPECT_TRUE(validate_auxiliary_basis(aux));
}

TEST(AuxBasisValidation, EmptyBasisThrows) {
    AuxiliaryBasisSet aux;
    EXPECT_THROW(validate_auxiliary_basis(aux), InvalidArgumentException);
}

}  // anonymous namespace
}  // namespace libaccint::data
