// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_parser_errors.cpp
/// @brief Unit tests for auxiliary basis parser error handling

#include <gtest/gtest.h>

#include <libaccint/data/auxiliary_basis_parser.hpp>
#include <libaccint/data/auxiliary_basis_data.hpp>
#include <libaccint/utils/error_handling.hpp>

namespace libaccint::testing {

namespace {

/// Minimal atom list for parsing tests
std::vector<data::Atom> test_atoms() {
    return {
        {1, {0.0, 0.0, 0.0}},
        {8, {0.0, 0.0, 1.5}}
    };
}

}  // anonymous namespace

// =============================================================================
// JSON Parser Error Tests
// =============================================================================

TEST(ParserErrorTest, EmptyJSON) {
    EXPECT_THROW(
        data::parse_auxiliary_basis_json("", test_atoms()),
        InvalidArgumentException);
}

TEST(ParserErrorTest, InvalidJSON) {
    EXPECT_THROW(
        data::parse_auxiliary_basis_json("{not valid json", test_atoms()),
        InvalidArgumentException);
}

TEST(ParserErrorTest, MissingElementsField) {
    // Valid JSON but missing required 'elements' key
    std::string json = R"({"name": "test"})";
    EXPECT_THROW(
        data::parse_auxiliary_basis_json(json, test_atoms()),
        InvalidArgumentException);
}

TEST(ParserErrorTest, MissingElementEntry) {
    // JSON with 'elements' but missing required element (Z=8 for oxygen)
    std::string json = R"({
        "elements": {
            "1": {
                "electron_shells": [
                    {
                        "angular_momentum": [0],
                        "exponents": ["1.0"],
                        "coefficients": [["1.0"]]
                    }
                ]
            }
        }
    })";
    // Atoms include oxygen (Z=8) which is not in the JSON
    EXPECT_THROW(
        data::parse_auxiliary_basis_json(json, test_atoms()),
        InvalidArgumentException);
}

TEST(ParserErrorTest, ValidMinimalJSON) {
    // Both H and O present
    std::string json = R"({
        "elements": {
            "1": {
                "electron_shells": [
                    {
                        "angular_momentum": [0],
                        "exponents": ["1.0"],
                        "coefficients": [["1.0"]]
                    }
                ]
            },
            "8": {
                "electron_shells": [
                    {
                        "angular_momentum": [0],
                        "exponents": ["10.0"],
                        "coefficients": [["1.0"]]
                    }
                ]
            }
        }
    })";
    EXPECT_NO_THROW(
        data::parse_auxiliary_basis_json(json, test_atoms()));
}

// =============================================================================
// Gaussian94 Parser Error Tests
// =============================================================================

TEST(ParserErrorTest, EmptyGaussian94) {
    // Empty input should produce empty basis; parse doesn't fail on empty
    // But creating an AuxiliaryBasisSet that would be valid is the question
    auto atoms = test_atoms();
    // With empty content, the element shells for H/O won't be found
    EXPECT_THROW(
        data::parse_auxiliary_basis_gaussian94("", atoms),
        InvalidArgumentException);
}

TEST(ParserErrorTest, Gaussian94MissingElement) {
    // Only hydrogen data, but atoms include oxygen
    std::string g94 = R"(****
H     0
S   1   1.00
      1.0000000    1.0000000
****)";
    EXPECT_THROW(
        data::parse_auxiliary_basis_gaussian94(g94, test_atoms()),
        InvalidArgumentException);
}

TEST(ParserErrorTest, Gaussian94ValidMinimal) {
    std::string g94 = R"(****
H     0
S   1   1.00
      1.0000000    1.0000000
****
O     0
S   1   1.00
     10.0000000    1.0000000
****)";
    EXPECT_NO_THROW(
        data::parse_auxiliary_basis_gaussian94(g94, test_atoms()));
}

// =============================================================================
// Element Symbol Error Tests
// =============================================================================

TEST(ParserErrorTest, InvalidElementSymbol) {
    EXPECT_THROW(data::element_symbol_to_z("Xx"), InvalidArgumentException);
    EXPECT_THROW(data::element_symbol_to_z("Zz"), InvalidArgumentException);
    EXPECT_THROW(data::element_symbol_to_z(""), InvalidArgumentException);
    EXPECT_THROW(data::element_symbol_to_z("Abc"), InvalidArgumentException);
}

TEST(ParserErrorTest, InvalidAtomicNumber) {
    EXPECT_THROW(data::z_to_element_symbol(0), InvalidArgumentException);
    EXPECT_THROW(data::z_to_element_symbol(-5), InvalidArgumentException);
    EXPECT_THROW(data::z_to_element_symbol(119), InvalidArgumentException);
    EXPECT_THROW(data::z_to_element_symbol(1000), InvalidArgumentException);
}

// =============================================================================
// Validation Error Tests
// =============================================================================

TEST(ParserErrorTest, ValidateEmptyBasis) {
    AuxiliaryBasisSet empty;
    EXPECT_THROW(data::validate_auxiliary_basis(empty), InvalidArgumentException);
}

TEST(ParserErrorTest, ValidateValidBasis) {
    auto aux = data::create_builtin_auxiliary_basis("cc-pVDZ-RI",
        std::vector<data::Atom>{{8, {0.0, 0.0, 0.0}}, {1, {0.0, 0.0, 1.5}}});
    EXPECT_TRUE(data::validate_auxiliary_basis(aux));
}

// =============================================================================
// File Parser Error Tests
// =============================================================================

TEST(ParserErrorTest, NonexistentFile) {
    EXPECT_THROW(
        data::parse_auxiliary_basis_file("/nonexistent/path/basis.gbs",
                                         test_atoms()),
        InvalidArgumentException);
}

TEST(ParserErrorTest, UnknownFileExtension) {
    // Auto-detection should fail on unknown extension
    EXPECT_THROW(
        data::parse_auxiliary_basis_file("/tmp/basis.xyz", test_atoms()),
        InvalidArgumentException);
}

}  // namespace libaccint::testing
