// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_screening_options.cpp
/// @brief Unit tests for ScreeningOptions and ScreeningStatistics

#include <libaccint/screening/screening_options.hpp>
#include <gtest/gtest.h>

#include <sstream>

using namespace libaccint;
using namespace libaccint::screening;

// =============================================================================
// ScreeningOptions Tests
// =============================================================================

TEST(ScreeningOptionsTest, DefaultValues) {
    ScreeningOptions opts;

    EXPECT_DOUBLE_EQ(opts.threshold, 1e-12);
    EXPECT_TRUE(opts.enabled);
    EXPECT_FALSE(opts.density_weighted);
    EXPECT_FALSE(opts.enable_statistics);
    EXPECT_EQ(opts.verbosity, 0);
}

TEST(ScreeningOptionsTest, NonePreset) {
    ScreeningOptions opts = ScreeningOptions::none();

    EXPECT_DOUBLE_EQ(opts.threshold, 0.0);
    EXPECT_FALSE(opts.enabled);
}

TEST(ScreeningOptionsTest, LoosePreset) {
    ScreeningOptions opts = ScreeningOptions::loose();

    EXPECT_DOUBLE_EQ(opts.threshold, 1e-10);
    EXPECT_TRUE(opts.enabled);
}

TEST(ScreeningOptionsTest, NormalPreset) {
    ScreeningOptions opts = ScreeningOptions::normal();

    EXPECT_DOUBLE_EQ(opts.threshold, 1e-12);
    EXPECT_TRUE(opts.enabled);
}

TEST(ScreeningOptionsTest, TightPreset) {
    ScreeningOptions opts = ScreeningOptions::tight();

    EXPECT_DOUBLE_EQ(opts.threshold, 1e-14);
    EXPECT_TRUE(opts.enabled);
}

TEST(ScreeningOptionsTest, FromPreset) {
    EXPECT_DOUBLE_EQ(ScreeningOptions::from_preset(ScreeningPreset::None).threshold, 0.0);
    EXPECT_DOUBLE_EQ(ScreeningOptions::from_preset(ScreeningPreset::Loose).threshold, 1e-10);
    EXPECT_DOUBLE_EQ(ScreeningOptions::from_preset(ScreeningPreset::Normal).threshold, 1e-12);
    EXPECT_DOUBLE_EQ(ScreeningOptions::from_preset(ScreeningPreset::Tight).threshold, 1e-14);
}

TEST(ScreeningOptionsTest, EffectiveThreshold) {
    ScreeningOptions disabled{.enabled = false};
    EXPECT_DOUBLE_EQ(disabled.effective_threshold(), 0.0);

    ScreeningOptions negative{.threshold = -1e-10, .enabled = true};
    EXPECT_DOUBLE_EQ(negative.effective_threshold(), 0.0);

    ScreeningOptions normal = ScreeningOptions::normal();
    EXPECT_DOUBLE_EQ(normal.effective_threshold(), 1e-12);
}

TEST(ScreeningOptionsTest, PassesScreening) {
    ScreeningOptions opts{.threshold = 1e-10, .enabled = true};

    // Schwarz product above threshold passes
    EXPECT_TRUE(opts.passes_screening(1e-8));
    EXPECT_TRUE(opts.passes_screening(1e-10));  // Equal passes

    // Schwarz product below threshold fails
    EXPECT_FALSE(opts.passes_screening(1e-12));
    EXPECT_FALSE(opts.passes_screening(1e-15));
}

TEST(ScreeningOptionsTest, DisabledAlwaysPasses) {
    ScreeningOptions opts{.threshold = 1e-10, .enabled = false};

    // All products pass when screening is disabled
    EXPECT_TRUE(opts.passes_screening(1e-15));
    EXPECT_TRUE(opts.passes_screening(0.0));
}

TEST(ScreeningOptionsTest, PresetName) {
    EXPECT_EQ(ScreeningOptions::none().preset_name(), "None");
    EXPECT_EQ(ScreeningOptions::loose().preset_name(), "Loose");
    EXPECT_EQ(ScreeningOptions::normal().preset_name(), "Normal");
    EXPECT_EQ(ScreeningOptions::tight().preset_name(), "Tight");

    ScreeningOptions custom{.threshold = 1e-11};
    EXPECT_EQ(custom.preset_name(), "Custom");
}

// =============================================================================
// ScreeningStatistics Tests
// =============================================================================

TEST(ScreeningStatisticsTest, DefaultValues) {
    ScreeningStatistics stats;

    EXPECT_EQ(stats.total_quartets, 0u);
    EXPECT_EQ(stats.computed_quartets, 0u);
    EXPECT_EQ(stats.skipped_quartets, 0u);
}

TEST(ScreeningStatisticsTest, Efficiency) {
    ScreeningStatistics stats;
    stats.total_quartets = 100;
    stats.computed_quartets = 40;
    stats.skipped_quartets = 60;

    EXPECT_DOUBLE_EQ(stats.efficiency(), 0.6);
    EXPECT_DOUBLE_EQ(stats.skip_percentage(), 60.0);
}

TEST(ScreeningStatisticsTest, EfficiencyZeroTotal) {
    ScreeningStatistics stats;
    // Avoid division by zero
    EXPECT_DOUBLE_EQ(stats.efficiency(), 0.0);
}

TEST(ScreeningStatisticsTest, Reset) {
    ScreeningStatistics stats;
    stats.total_quartets = 100;
    stats.computed_quartets = 40;
    stats.skipped_quartets = 60;

    stats.reset();

    EXPECT_EQ(stats.total_quartets, 0u);
    EXPECT_EQ(stats.computed_quartets, 0u);
    EXPECT_EQ(stats.skipped_quartets, 0u);
}

TEST(ScreeningStatisticsTest, Merge) {
    ScreeningStatistics stats1;
    stats1.total_quartets = 100;
    stats1.computed_quartets = 40;
    stats1.skipped_quartets = 60;

    ScreeningStatistics stats2;
    stats2.total_quartets = 50;
    stats2.computed_quartets = 20;
    stats2.skipped_quartets = 30;

    stats1.merge(stats2);

    EXPECT_EQ(stats1.total_quartets, 150u);
    EXPECT_EQ(stats1.computed_quartets, 60u);
    EXPECT_EQ(stats1.skipped_quartets, 90u);
}

