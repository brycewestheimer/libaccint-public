// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_diagnostics.cpp
/// @brief Tests for performance diagnostics infrastructure (Task 25.3.3)

#include <libaccint/utils/diagnostics.hpp>

#include <gtest/gtest.h>
#include <chrono>
#include <thread>

using namespace libaccint::diagnostics;

class DiagnosticsTest : public ::testing::Test {
protected:
    void SetUp() override {
        DiagnosticsCollector::instance().reset();
    }

    void TearDown() override {
        DiagnosticsCollector::instance().reset();
    }
};

TEST_F(DiagnosticsTest, DefaultDisabled) {
    EXPECT_FALSE(DiagnosticsCollector::instance().is_enabled());
}

TEST_F(DiagnosticsTest, EnableDisable) {
    DiagnosticsCollector::instance().set_enabled(true);
    EXPECT_TRUE(DiagnosticsCollector::instance().is_enabled());

    DiagnosticsCollector::instance().set_enabled(false);
    EXPECT_FALSE(DiagnosticsCollector::instance().is_enabled());
}

TEST_F(DiagnosticsTest, IncrementCounters) {
    DiagnosticsCollector::instance().set_enabled(true);

    DiagnosticsCollector::instance().increment(Counter::IntegralsComputed, 100);
    DiagnosticsCollector::instance().increment(Counter::IntegralsComputed, 50);
    DiagnosticsCollector::instance().increment(Counter::KernelInvocations);

    EXPECT_EQ(DiagnosticsCollector::instance().counter_value(
        Counter::IntegralsComputed), 150u);
    EXPECT_EQ(DiagnosticsCollector::instance().counter_value(
        Counter::KernelInvocations), 1u);
    EXPECT_EQ(DiagnosticsCollector::instance().counter_value(
        Counter::ShellPairsComputed), 0u);
}

TEST_F(DiagnosticsTest, CounterIgnoredWhenDisabled) {
    // Disabled by default
    DiagnosticsCollector::instance().increment(Counter::IntegralsComputed, 100);
    EXPECT_EQ(DiagnosticsCollector::instance().counter_value(
        Counter::IntegralsComputed), 0u);
}

TEST_F(DiagnosticsTest, ScopedTimer) {
    DiagnosticsCollector::instance().set_enabled(true);

    {
        ScopedTimer timer("test_scope");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    auto records = DiagnosticsCollector::instance().timing_records();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].name, "test_scope");
    EXPECT_EQ(records[0].call_count, 1u);
    EXPECT_GT(records[0].duration.count(), 0);
}

TEST_F(DiagnosticsTest, ScopedTimerElapsed) {
    DiagnosticsCollector::instance().set_enabled(true);

    ScopedTimer timer("elapsed_test");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    auto elapsed = timer.elapsed();
    EXPECT_GT(elapsed.count(), 0);
}

TEST_F(DiagnosticsTest, MultipleTimingsSameName) {
    DiagnosticsCollector::instance().set_enabled(true);

    for (int i = 0; i < 3; ++i) {
        ScopedTimer timer("repeated");
        // Small work to ensure timing > 0
        volatile int x = 0;
        for (int j = 0; j < 1000; ++j) x += j;
        (void)x;
    }

    auto records = DiagnosticsCollector::instance().timing_records();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].call_count, 3u);
}

TEST_F(DiagnosticsTest, Report) {
    DiagnosticsCollector::instance().set_enabled(true);

    DiagnosticsCollector::instance().increment(Counter::IntegralsComputed, 1000);
    {
        ScopedTimer timer("test_kernel");
        volatile int x = 0;
        for (int j = 0; j < 100; ++j) x += j;
        (void)x;
    }

    auto report = DiagnosticsCollector::instance().report();
    EXPECT_NE(report.find("Performance Diagnostics"), std::string::npos);
    EXPECT_NE(report.find("integrals_computed"), std::string::npos);
    EXPECT_NE(report.find("1000"), std::string::npos);
    EXPECT_NE(report.find("test_kernel"), std::string::npos);
}

TEST_F(DiagnosticsTest, Reset) {
    DiagnosticsCollector::instance().set_enabled(true);
    DiagnosticsCollector::instance().increment(Counter::IntegralsComputed, 100);
    {
        ScopedTimer timer("reset_test");
    }

    DiagnosticsCollector::instance().reset();
    EXPECT_FALSE(DiagnosticsCollector::instance().is_enabled());
    EXPECT_EQ(DiagnosticsCollector::instance().counter_value(
        Counter::IntegralsComputed), 0u);
    EXPECT_TRUE(DiagnosticsCollector::instance().timing_records().empty());
}

TEST_F(DiagnosticsTest, CounterNames) {
    EXPECT_EQ(counter_name(Counter::ShellPairsComputed), "shell_pairs_computed");
    EXPECT_EQ(counter_name(Counter::ShellQuartetsComputed), "shell_quartets_computed");
    EXPECT_EQ(counter_name(Counter::ShellQuartetsScreened), "shell_quartets_screened");
    EXPECT_EQ(counter_name(Counter::IntegralsComputed), "integrals_computed");
    EXPECT_EQ(counter_name(Counter::PrimitivePairsComputed), "primitive_pairs_computed");
    EXPECT_EQ(counter_name(Counter::KernelInvocations), "kernel_invocations");
    EXPECT_EQ(counter_name(Counter::BufferAllocations), "buffer_allocations");
    EXPECT_EQ(counter_name(Counter::BufferReuses), "buffer_reuses");
}

TEST_F(DiagnosticsTest, ThreadSafeCounters) {
    DiagnosticsCollector::instance().set_enabled(true);

    constexpr int n_threads = 4;
    constexpr int n_increments = 1000;

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([]() {
            for (int i = 0; i < n_increments; ++i) {
                DiagnosticsCollector::instance().increment(
                    Counter::IntegralsComputed);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(DiagnosticsCollector::instance().counter_value(
        Counter::IntegralsComputed),
        static_cast<libaccint::Size>(n_threads * n_increments));
}

TEST_F(DiagnosticsTest, MacroTIMED_SCOPE) {
    DiagnosticsCollector::instance().set_enabled(true);

    {
        LIBACCINT_TIMED_SCOPE("macro_scope");
        volatile int x = 0;
        for (int j = 0; j < 100; ++j) x += j;
        (void)x;
    }

    auto records = DiagnosticsCollector::instance().timing_records();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].name, "macro_scope");
}

TEST_F(DiagnosticsTest, MacroCOUNT) {
    DiagnosticsCollector::instance().set_enabled(true);

    LIBACCINT_COUNT(Counter::KernelInvocations, 5);

    EXPECT_EQ(DiagnosticsCollector::instance().counter_value(
        Counter::KernelInvocations), 5u);
}
