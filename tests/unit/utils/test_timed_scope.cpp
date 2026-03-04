// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_timed_scope.cpp
/// @brief Unit tests for LIBACCINT_TIMED_SCOPE macro (Task 1.3.3)

#include <libaccint/utils/diagnostics.hpp>

#include <gtest/gtest.h>
#include <chrono>
#include <thread>

using namespace libaccint::diagnostics;

class TimedScopeTest : public ::testing::Test {
protected:
    void SetUp() override {
        DiagnosticsCollector::instance().reset();
    }

    void TearDown() override {
        DiagnosticsCollector::instance().reset();
    }
};

// Verify two LIBACCINT_TIMED_SCOPE invocations on different lines in the same
// scope compile without error (tests the __LINE__ concatenation fix from 1.1.1)
TEST_F(TimedScopeTest, MultipleUsagesInSameScope) {
    DiagnosticsCollector::instance().set_enabled(true);

    LIBACCINT_TIMED_SCOPE("scope_a");
    LIBACCINT_TIMED_SCOPE("scope_b");

    // If this compiles, the __LINE__-based unique naming works correctly
    SUCCEED();
}

TEST_F(TimedScopeTest, RecordsTiming) {
    DiagnosticsCollector::instance().set_enabled(true);

    {
        LIBACCINT_TIMED_SCOPE("timed_test");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    auto records = DiagnosticsCollector::instance().timing_records();
    // Should have at least 1 timing record for "timed_test"
    bool found = false;
    for (const auto& rec : records) {
        if (rec.name == "timed_test") {
            found = true;
            EXPECT_GE(rec.call_count, 1u);
            EXPECT_GT(rec.duration.count(), 0);
        }
    }
    EXPECT_TRUE(found) << "Expected timing record 'timed_test' not found";
}

TEST_F(TimedScopeTest, NoOpWhenDisabled) {
    // Diagnostics disabled by default after reset
    EXPECT_FALSE(DiagnosticsCollector::instance().is_enabled());

    {
        LIBACCINT_TIMED_SCOPE("disabled_test");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    auto records = DiagnosticsCollector::instance().timing_records();
    // Should have no timing records since diagnostics are disabled
    for (const auto& rec : records) {
        EXPECT_NE(rec.name, "disabled_test")
            << "Timing recorded despite diagnostics being disabled";
    }
}

TEST_F(TimedScopeTest, EmptyBody) {
    DiagnosticsCollector::instance().set_enabled(true);

    {
        LIBACCINT_TIMED_SCOPE("empty_body");
        // Empty scope -- should not crash
    }

    auto records = DiagnosticsCollector::instance().timing_records();
    bool found = false;
    for (const auto& rec : records) {
        if (rec.name == "empty_body") {
            found = true;
            EXPECT_GE(rec.call_count, 1u);
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(TimedScopeTest, NestedScopes) {
    DiagnosticsCollector::instance().set_enabled(true);

    {
        LIBACCINT_TIMED_SCOPE("outer");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        {
            LIBACCINT_TIMED_SCOPE("inner");
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    auto records = DiagnosticsCollector::instance().timing_records();
    bool found_outer = false, found_inner = false;
    for (const auto& rec : records) {
        if (rec.name == "outer") {
            found_outer = true;
            EXPECT_GE(rec.call_count, 1u);
        }
        if (rec.name == "inner") {
            found_inner = true;
            EXPECT_GE(rec.call_count, 1u);
        }
    }
    EXPECT_TRUE(found_outer);
    EXPECT_TRUE(found_inner);
}

TEST_F(TimedScopeTest, RepeatedScope) {
    DiagnosticsCollector::instance().set_enabled(true);

    for (int i = 0; i < 5; ++i) {
        LIBACCINT_TIMED_SCOPE("repeated");
    }

    auto records = DiagnosticsCollector::instance().timing_records();
    for (const auto& rec : records) {
        if (rec.name == "repeated") {
            EXPECT_EQ(rec.call_count, 5u);
        }
    }
}
