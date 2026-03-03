// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_bse_parser.cpp
/// @brief Tests for BSE JSON parser (Task 28.2.1)

#include <libaccint/data/bse_json_parser.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace libaccint;
using namespace libaccint::data;

namespace {

// Minimal valid BSE JSON for hydrogen (STO-3G style)
const std::string MINIMAL_BSE_JSON = R"({
  "molssi_bse_schema": {
    "schema_type": "complete",
    "schema_version": "0.1"
  },
  "name": "Test-Basis",
  "description": "Test basis set",
  "elements": {
    "1": {
      "electron_shells": [
        {
          "function_type": "gto",
          "region": "",
          "angular_momentum": [0],
          "exponents": [
            "3.42525091",
            "0.62391373",
            "0.16885540"
          ],
          "coefficients": [
            [
              "0.15432897",
              "0.53532814",
              "0.44463454"
            ]
          ]
        }
      ]
    }
  }
})";

// BSE JSON with SP shell (s+p combined)
const std::string SP_SHELL_JSON = R"({
  "molssi_bse_schema": {
    "schema_type": "complete",
    "schema_version": "0.1"
  },
  "name": "SP-Test",
  "description": "Test SP shells",
  "elements": {
    "6": {
      "electron_shells": [
        {
          "function_type": "gto",
          "region": "",
          "angular_momentum": [0],
          "exponents": ["71.6168370", "13.0450960", "3.5305122"],
          "coefficients": [["0.15432897", "0.53532814", "0.44463454"]]
        },
        {
          "function_type": "gto",
          "region": "",
          "angular_momentum": [0, 1],
          "exponents": ["2.9412494", "0.6834831", "0.2222899"],
          "coefficients": [
            ["-0.09996723", "0.39951283", "0.70011547"],
            ["0.15591627", "0.60768372", "0.39195739"]
          ]
        }
      ]
    }
  }
})";

// Invalid JSON
const std::string INVALID_JSON = R"({ not valid json })";

// Missing elements key
const std::string MISSING_ELEMENTS_JSON = R"({
  "molssi_bse_schema": {"schema_type": "complete", "schema_version": "0.1"},
  "name": "Bad"
})";

// Empty elements
const std::string EMPTY_ELEMENTS_JSON = R"({
  "elements": {}
})";

}  // anonymous namespace

// ============================================================================
// Parse Tests
// ============================================================================

TEST(BseJsonParserTest, ParseMinimalBasis) {
    std::vector<Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    BasisSet basis = BseJsonParser::parse(MINIMAL_BSE_JSON, atoms);

    EXPECT_EQ(basis.n_shells(), 1u);
    EXPECT_EQ(basis.n_basis_functions(), 1u);  // 1 s-function
    EXPECT_EQ(basis.shell(0).angular_momentum(), 0);
    EXPECT_EQ(basis.shell(0).n_primitives(), 3u);
}

TEST(BseJsonParserTest, ParseMultipleAtoms) {
    std::vector<Atom> atoms = {
        {1, {0.0, 0.0, 0.0}},
        {1, {0.0, 0.0, 1.4}},
    };
    BasisSet basis = BseJsonParser::parse(MINIMAL_BSE_JSON, atoms);

    EXPECT_EQ(basis.n_shells(), 2u);
    EXPECT_EQ(basis.n_basis_functions(), 2u);
}

TEST(BseJsonParserTest, ParseSPShell) {
    std::vector<Atom> atoms = {{6, {0.0, 0.0, 0.0}}};
    BasisSet basis = BseJsonParser::parse(SP_SHELL_JSON, atoms);

    // Carbon STO-3G: 1s (1 shell) + 2s + 2p (SP splits into 2 shells)
    EXPECT_EQ(basis.n_shells(), 3u);

    // Check angular momenta
    int total_functions = 0;
    bool found_s = false;
    bool found_p = false;
    for (Size i = 0; i < basis.n_shells(); ++i) {
        const auto& shell = basis.shell(i);
        total_functions += shell.n_functions();
        if (shell.angular_momentum() == 0) found_s = true;
        if (shell.angular_momentum() == 1) found_p = true;
    }

    EXPECT_TRUE(found_s);
    EXPECT_TRUE(found_p);
    EXPECT_EQ(total_functions, 5);  // 1s + 2s + 2p = 1+1+3 = 5
}

TEST(BseJsonParserTest, ParseInvalidJsonThrows) {
    std::vector<Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    EXPECT_THROW(BseJsonParser::parse(INVALID_JSON, atoms),
                 InvalidArgumentException);
}

TEST(BseJsonParserTest, ParseMissingElementsThrows) {
    std::vector<Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    EXPECT_THROW(BseJsonParser::parse(MISSING_ELEMENTS_JSON, atoms),
                 InvalidArgumentException);
}

TEST(BseJsonParserTest, ParseMissingElementDataThrows) {
    std::vector<Atom> atoms = {{2, {0.0, 0.0, 0.0}}};  // He not in minimal JSON
    EXPECT_THROW(BseJsonParser::parse(MINIMAL_BSE_JSON, atoms),
                 InvalidArgumentException);
}

TEST(BseJsonParserTest, ParseEmptyAtomList) {
    std::vector<Atom> atoms;
    BasisSet basis = BseJsonParser::parse(MINIMAL_BSE_JSON, atoms);
    EXPECT_EQ(basis.n_shells(), 0u);
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST(BseJsonParserTest, ValidateValidJson) {
    auto errors = BseJsonParser::validate(MINIMAL_BSE_JSON);
    EXPECT_TRUE(errors.empty()) << "Unexpected error: " << (errors.empty() ? "" : errors[0]);
}

TEST(BseJsonParserTest, ValidateInvalidJson) {
    auto errors = BseJsonParser::validate(INVALID_JSON);
    EXPECT_FALSE(errors.empty());
}

TEST(BseJsonParserTest, ValidateMissingElements) {
    auto errors = BseJsonParser::validate(MISSING_ELEMENTS_JSON);
    EXPECT_FALSE(errors.empty());
}

// ============================================================================
// Metadata Tests
// ============================================================================

TEST(BseJsonParserTest, GetName) {
    std::string name = BseJsonParser::get_name(MINIMAL_BSE_JSON);
    EXPECT_EQ(name, "Test-Basis");
}

TEST(BseJsonParserTest, GetDescription) {
    std::string desc = BseJsonParser::get_description(MINIMAL_BSE_JSON);
    EXPECT_EQ(desc, "Test basis set");
}

TEST(BseJsonParserTest, GetSupportedElements) {
    auto elements = BseJsonParser::get_supported_elements(MINIMAL_BSE_JSON);
    EXPECT_EQ(elements.size(), 1u);
    EXPECT_EQ(elements[0], 1);  // Hydrogen
}

TEST(BseJsonParserTest, GetSupportedElementsSP) {
    auto elements = BseJsonParser::get_supported_elements(SP_SHELL_JSON);
    EXPECT_EQ(elements.size(), 1u);
    EXPECT_EQ(elements[0], 6);  // Carbon
}

// ============================================================================
// Integration with existing basis set data
// ============================================================================

TEST(BseJsonParserTest, ParseFileNonexistentThrows) {
    std::vector<Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    EXPECT_THROW(BseJsonParser::parse_file("/nonexistent/path.json", atoms),
                 InvalidArgumentException);
}

TEST(BseJsonParserTest, AtomIndicesAssigned) {
    std::vector<Atom> atoms = {
        {1, {0.0, 0.0, 0.0}},
        {1, {0.0, 0.0, 1.4}},
    };
    BasisSet basis = BseJsonParser::parse(MINIMAL_BSE_JSON, atoms);

    EXPECT_EQ(basis.shell(0).atom_index(), 0);
    EXPECT_EQ(basis.shell(1).atom_index(), 1);
}
