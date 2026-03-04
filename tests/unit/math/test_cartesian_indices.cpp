// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_cartesian_indices.cpp
/// @brief Unit tests for Cartesian index utilities

#include <gtest/gtest.h>
#include <libaccint/math/cartesian_indices.hpp>
#include <libaccint/core/types.hpp>

namespace libaccint::math {

// ============================================================================
// Test generate_cartesian_indices function
// ============================================================================

TEST(CartesianIndicesTest, GenerateIndicesCountMatchesFormula) {
    // Test that the number of generated indices matches n_cartesian(L) for each L
    for (int L = 0; L <= 6; ++L) {
        auto indices = generate_cartesian_indices(L);
        EXPECT_EQ(static_cast<int>(indices.size()), n_cartesian(L))
            << "Mismatch for L=" << L;
    }
}

TEST(CartesianIndicesTest, GenerateIndicesL0) {
    auto indices = generate_cartesian_indices(0);
    EXPECT_EQ(indices.size(), 1);
    EXPECT_EQ(indices[0], (std::array<int, 3>{0, 0, 0}));
}

TEST(CartesianIndicesTest, GenerateIndicesL1) {
    auto indices = generate_cartesian_indices(1);
    EXPECT_EQ(indices.size(), 3);
    EXPECT_EQ(indices[0], (std::array<int, 3>{1, 0, 0}));
    EXPECT_EQ(indices[1], (std::array<int, 3>{0, 1, 0}));
    EXPECT_EQ(indices[2], (std::array<int, 3>{0, 0, 1}));
}

TEST(CartesianIndicesTest, GenerateIndicesL2) {
    auto indices = generate_cartesian_indices(2);
    EXPECT_EQ(indices.size(), 6);
    EXPECT_EQ(indices[0], (std::array<int, 3>{2, 0, 0}));
    EXPECT_EQ(indices[1], (std::array<int, 3>{1, 1, 0}));
    EXPECT_EQ(indices[2], (std::array<int, 3>{1, 0, 1}));
    EXPECT_EQ(indices[3], (std::array<int, 3>{0, 2, 0}));
    EXPECT_EQ(indices[4], (std::array<int, 3>{0, 1, 1}));
    EXPECT_EQ(indices[5], (std::array<int, 3>{0, 0, 2}));
}

TEST(CartesianIndicesTest, GenerateIndicesL3) {
    auto indices = generate_cartesian_indices(3);
    EXPECT_EQ(indices.size(), 10);
    EXPECT_EQ(indices[0], (std::array<int, 3>{3, 0, 0}));
    EXPECT_EQ(indices[1], (std::array<int, 3>{2, 1, 0}));
    EXPECT_EQ(indices[2], (std::array<int, 3>{2, 0, 1}));
    EXPECT_EQ(indices[3], (std::array<int, 3>{1, 2, 0}));
    EXPECT_EQ(indices[4], (std::array<int, 3>{1, 1, 1}));
    EXPECT_EQ(indices[5], (std::array<int, 3>{1, 0, 2}));
    EXPECT_EQ(indices[6], (std::array<int, 3>{0, 3, 0}));
    EXPECT_EQ(indices[7], (std::array<int, 3>{0, 2, 1}));
    EXPECT_EQ(indices[8], (std::array<int, 3>{0, 1, 2}));
    EXPECT_EQ(indices[9], (std::array<int, 3>{0, 0, 3}));
}

TEST(CartesianIndicesTest, GenerateIndicesComponentsSum) {
    // Test that all components sum to L for each generated index
    for (int L = 0; L <= 6; ++L) {
        auto indices = generate_cartesian_indices(L);
        for (const auto& idx : indices) {
            int sum = idx[0] + idx[1] + idx[2];
            EXPECT_EQ(sum, L)
                << "Components don't sum to L for L=" << L
                << ", index=(" << idx[0] << "," << idx[1] << "," << idx[2] << ")";
        }
    }
}

// ============================================================================
// Test cartesian_index function
// ============================================================================

TEST(CartesianIndicesTest, CartesianIndexL0) {
    // L=0: only (0,0,0)
    EXPECT_EQ(cartesian_index(0, 0, 0), 0);
}

TEST(CartesianIndicesTest, CartesianIndexL1) {
    // L=1: (1,0,0), (0,1,0), (0,0,1)
    EXPECT_EQ(cartesian_index(1, 0, 0), 0);
    EXPECT_EQ(cartesian_index(0, 1, 0), 1);
    EXPECT_EQ(cartesian_index(0, 0, 1), 2);
}

TEST(CartesianIndicesTest, CartesianIndexL2) {
    // L=2: (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)
    EXPECT_EQ(cartesian_index(2, 0, 0), 0);
    EXPECT_EQ(cartesian_index(1, 1, 0), 1);
    EXPECT_EQ(cartesian_index(1, 0, 1), 2);
    EXPECT_EQ(cartesian_index(0, 2, 0), 3);
    EXPECT_EQ(cartesian_index(0, 1, 1), 4);
    EXPECT_EQ(cartesian_index(0, 0, 2), 5);
}

TEST(CartesianIndicesTest, CartesianIndexL3) {
    // L=3: (3,0,0), (2,1,0), (2,0,1), (1,2,0), (1,1,1), (1,0,2), (0,3,0), (0,2,1), (0,1,2), (0,0,3)
    EXPECT_EQ(cartesian_index(3, 0, 0), 0);
    EXPECT_EQ(cartesian_index(2, 1, 0), 1);
    EXPECT_EQ(cartesian_index(2, 0, 1), 2);
    EXPECT_EQ(cartesian_index(1, 2, 0), 3);
    EXPECT_EQ(cartesian_index(1, 1, 1), 4);
    EXPECT_EQ(cartesian_index(1, 0, 2), 5);
    EXPECT_EQ(cartesian_index(0, 3, 0), 6);
    EXPECT_EQ(cartesian_index(0, 2, 1), 7);
    EXPECT_EQ(cartesian_index(0, 1, 2), 8);
    EXPECT_EQ(cartesian_index(0, 0, 3), 9);
}

TEST(CartesianIndicesTest, CartesianIndexRoundTrip) {
    // Test round-trip: generate indices and verify that cartesian_index
    // returns the correct sequential indices (0, 1, 2, ...)
    for (int L = 0; L <= 6; ++L) {
        auto indices = generate_cartesian_indices(L);
        for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
            int idx = cartesian_index(indices[i][0], indices[i][1], indices[i][2]);
            EXPECT_EQ(idx, i)
                << "Round-trip failed for L=" << L << " at position " << i
                << " with tuple (" << indices[i][0] << "," << indices[i][1] << "," << indices[i][2] << ")";
        }
    }
}

// ============================================================================
// Test constexpr compatibility
// ============================================================================

TEST(CartesianIndicesTest, CartesianIndexConstexpr) {
    // Verify that cartesian_index is constexpr by using it in a constexpr context
    constexpr int idx1 = cartesian_index(1, 0, 0);
    constexpr int idx2 = cartesian_index(0, 1, 0);
    constexpr int idx3 = cartesian_index(0, 0, 1);

    EXPECT_EQ(idx1, 0);
    EXPECT_EQ(idx2, 1);
    EXPECT_EQ(idx3, 2);
}

// ============================================================================
// Test edge cases
// ============================================================================

TEST(CartesianIndicesTest, GenerateIndicesL6) {
    auto indices = generate_cartesian_indices(6);
    EXPECT_EQ(indices.size(), 28);  // n_cartesian(6) = 28

    // Check first and last
    EXPECT_EQ(indices[0], (std::array<int, 3>{6, 0, 0}));
    EXPECT_EQ(indices[27], (std::array<int, 3>{0, 0, 6}));
}

TEST(CartesianIndicesTest, CartesianIndexL4) {
    auto indices = generate_cartesian_indices(4);
    EXPECT_EQ(indices.size(), 15);  // n_cartesian(4) = 15

    // Verify all indices map correctly
    for (int i = 0; i < 15; ++i) {
        int computed_idx = cartesian_index(indices[i][0], indices[i][1], indices[i][2]);
        EXPECT_EQ(computed_idx, i);
    }
}

TEST(CartesianIndicesTest, CartesianIndexL5) {
    auto indices = generate_cartesian_indices(5);
    EXPECT_EQ(indices.size(), 21);  // n_cartesian(5) = 21

    // Verify all indices map correctly
    for (int i = 0; i < 21; ++i) {
        int computed_idx = cartesian_index(indices[i][0], indices[i][1], indices[i][2]);
        EXPECT_EQ(computed_idx, i);
    }
}

}  // namespace libaccint::math
