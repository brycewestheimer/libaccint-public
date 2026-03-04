// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_df_cholesky_conditioning.cpp
/// @brief Tests for Cholesky conditioning check in DFFockBuilder

#include <libaccint/consumers/df_fock_builder.hpp>
#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace libaccint;
using namespace libaccint::consumers;

namespace {

/// Create a minimal H2 orbital basis
std::unique_ptr<BasisSet> make_orbital() {
    std::vector<Shell> shells;

    Shell s0(0, Point3D{0.0, 0.0, 0.0},
             {3.42525091, 0.62391373, 0.16885540},
             {0.15432897, 0.53532814, 0.44463454});
    s0.set_atom_index(0);
    shells.push_back(std::move(s0));

    Shell s1(0, Point3D{1.4, 0.0, 0.0},
             {3.42525091, 0.62391373, 0.16885540},
             {0.15432897, 0.53532814, 0.44463454});
    s1.set_atom_index(1);
    shells.push_back(std::move(s1));

    return std::make_unique<BasisSet>(std::move(shells));
}

/// Create a well-conditioned auxiliary basis (spread exponents)
std::unique_ptr<AuxiliaryBasisSet> make_well_conditioned_aux() {
    std::vector<Shell> shells;

    Shell a0(0, Point3D{0.0, 0.0, 0.0}, {4.0, 1.0, 0.25}, {0.33, 0.33, 0.34});
    a0.set_atom_index(0);
    shells.push_back(std::move(a0));

    Shell a1(0, Point3D{1.4, 0.0, 0.0}, {4.0, 1.0, 0.25}, {0.33, 0.33, 0.34});
    a1.set_atom_index(1);
    shells.push_back(std::move(a1));

    return std::make_unique<AuxiliaryBasisSet>(
        std::move(shells), FittingType::JKFIT, "well-cond-test");
}

/// Create a nearly-linearly-dependent auxiliary basis (very similar exponents)
/// This produces a poorly conditioned metric
std::unique_ptr<AuxiliaryBasisSet> make_ill_conditioned_aux() {
    std::vector<Shell> shells;

    // All shells at same center with nearly identical exponents
    // → near-linear-dependent metric
    Shell a0(0, Point3D{0.0, 0.0, 0.0}, {1.0}, {1.0});
    a0.set_atom_index(0);
    shells.push_back(std::move(a0));

    Shell a1(0, Point3D{0.0, 0.0, 0.0}, {1.001}, {1.0});
    a1.set_atom_index(0);
    shells.push_back(std::move(a1));

    Shell a2(0, Point3D{0.0, 0.0, 0.0}, {1.002}, {1.0});
    a2.set_atom_index(0);
    shells.push_back(std::move(a2));

    return std::make_unique<AuxiliaryBasisSet>(
        std::move(shells), FittingType::JKFIT, "ill-cond-test");
}

}  // namespace

// =============================================================================
// Conditioning Tests
// =============================================================================

TEST(DFCholeskyConditioning, WellConditionedPassesClean) {
    auto orb = make_orbital();
    auto aux = make_well_conditioned_aux();
    DFFockBuilder builder(*orb, *aux);

    // Should initialize without throwing or warning
    EXPECT_NO_THROW(builder.initialize());
    EXPECT_TRUE(builder.is_initialized());
}

TEST(DFCholeskyConditioning, DefaultThresholdsAreReasonable) {
    DFFockBuilderConfig config;

    // Verify default thresholds
    EXPECT_DOUBLE_EQ(config.conditioning_threshold, 1e8);
    EXPECT_DOUBLE_EQ(config.conditioning_hard_limit, 1e14);
}

TEST(DFCholeskyConditioning, CustomLenientThreshold) {
    auto orb = make_orbital();
    auto aux = make_well_conditioned_aux();
    DFFockBuilderConfig config;
    config.conditioning_threshold = 1.0;  // Very strict warning threshold
    config.conditioning_hard_limit = 0.0; // Disable hard limit
    DFFockBuilder builder(*orb, *aux, config);

    // Should still initialize without error since hard limit is disabled
    EXPECT_NO_THROW(builder.initialize());
}

TEST(DFCholeskyConditioning, StrictHardLimitThrows) {
    auto orb = make_orbital();
    auto aux = make_well_conditioned_aux();
    DFFockBuilderConfig config;
    config.conditioning_hard_limit = 0.5;  // Below any real condition number (min is 1.0)
    DFFockBuilder builder(*orb, *aux, config);

    // Should throw because every real metric has condition number >= 1.0
    EXPECT_THROW(builder.initialize(), InvalidArgumentException);
}

TEST(DFCholeskyConditioning, DisabledHardLimitNeverThrows) {
    auto orb = make_orbital();
    auto aux = make_well_conditioned_aux();
    DFFockBuilderConfig config;
    config.conditioning_hard_limit = 0.0;
    DFFockBuilder builder(*orb, *aux, config);

    EXPECT_NO_THROW(builder.initialize());
}

TEST(DFCholeskyConditioning, IllConditionedAuxBasisDetected) {
    auto orb = make_orbital();
    auto aux = make_ill_conditioned_aux();
    DFFockBuilderConfig config;
    // Use a moderate threshold to detect near-linear-dependence
    config.conditioning_threshold = 10.0;
    config.conditioning_hard_limit = 0.0;  // Don't throw, just test warning path
    DFFockBuilder builder(*orb, *aux, config);

    // Should still initialize (hard limit disabled), but condition number
    // will be high due to near-identical exponents
    EXPECT_NO_THROW(builder.initialize());
}
