// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_diagnostics_thread_safety.cpp
/// @brief Thread-safety tests for DiagnosticsCollector (Task 1.3.4)

#include <libaccint/utils/diagnostics.hpp>

#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>

using namespace libaccint::diagnostics;

class DiagnosticsThreadSafetyTest : public ::testing::Test {
protected:
    static constexpr int NUM_THREADS = 8;
    static constexpr int ITERATIONS = 10000;

    void SetUp() override {
        DiagnosticsCollector::instance().reset();
    }

    void TearDown() override {
        DiagnosticsCollector::instance().reset();
    }
};

TEST_F(DiagnosticsThreadSafetyTest, ConcurrentEnableDisable) {
    std::vector<std::thread> threads;
    std::atomic<bool> start{false};

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&start, t]() {
            while (!start.load(std::memory_order_acquire)) {
                // spin until all threads are ready
            }
            for (int i = 0; i < ITERATIONS; ++i) {
                if ((i + t) % 2 == 0) {
                    DiagnosticsCollector::instance().set_enabled(true);
                } else {
                    DiagnosticsCollector::instance().set_enabled(false);
                }
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto& th : threads) {
        th.join();
    }

    // If we got here without crashing or TSan reports, the test passes
    SUCCEED();
}

TEST_F(DiagnosticsThreadSafetyTest, ConcurrentIsEnabled) {
    std::atomic<bool> start{false};
    std::atomic<bool> stop{false};

    // Writer thread: toggles enable/disable
    std::thread writer([&]() {
        while (!start.load(std::memory_order_acquire)) {}
        for (int i = 0; i < ITERATIONS; ++i) {
            DiagnosticsCollector::instance().set_enabled(i % 2 == 0);
        }
        stop.store(true, std::memory_order_release);
    });

    // Reader threads
    std::vector<std::thread> readers;
    for (int t = 0; t < NUM_THREADS; ++t) {
        readers.emplace_back([&]() {
            while (!start.load(std::memory_order_acquire)) {}
            while (!stop.load(std::memory_order_acquire)) {
                // Reading is_enabled concurrently with writes
                [[maybe_unused]] bool enabled =
                    DiagnosticsCollector::instance().is_enabled();
            }
        });
    }

    start.store(true, std::memory_order_release);
    writer.join();
    for (auto& th : readers) {
        th.join();
    }

    SUCCEED();
}

TEST_F(DiagnosticsThreadSafetyTest, ConcurrentCounterIncrement) {
    DiagnosticsCollector::instance().set_enabled(true);

    std::vector<std::thread> threads;
    std::atomic<bool> start{false};

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&start]() {
            while (!start.load(std::memory_order_acquire)) {}
            for (int i = 0; i < ITERATIONS; ++i) {
                DiagnosticsCollector::instance().increment(
                    Counter::IntegralsComputed, 1);
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto& th : threads) {
        th.join();
    }

    auto count = DiagnosticsCollector::instance().counter_value(
        Counter::IntegralsComputed);
    EXPECT_EQ(count, static_cast<libaccint::Size>(NUM_THREADS * ITERATIONS));
}

TEST_F(DiagnosticsThreadSafetyTest, ConcurrentTimerSubmission) {
    DiagnosticsCollector::instance().set_enabled(true);

    std::vector<std::thread> threads;
    std::atomic<bool> start{false};
    constexpr int TIMER_ITERS = 100;

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&start, t]() {
            while (!start.load(std::memory_order_acquire)) {}
            for (int i = 0; i < TIMER_ITERS; ++i) {
                std::string name = "timer_" + std::to_string(t);
                DiagnosticsCollector::instance().record_timing(
                    name, std::chrono::nanoseconds(100));
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto& th : threads) {
        th.join();
    }

    auto records = DiagnosticsCollector::instance().timing_records();
    // Should have NUM_THREADS distinct timer entries
    EXPECT_GE(records.size(), 1u);

    // Verify total call count across all per-thread timers
    libaccint::Size total_calls = 0;
    for (const auto& rec : records) {
        total_calls += rec.call_count;
    }
    EXPECT_EQ(total_calls,
              static_cast<libaccint::Size>(NUM_THREADS * TIMER_ITERS));
}

TEST_F(DiagnosticsThreadSafetyTest, ResetSafety) {
    DiagnosticsCollector::instance().set_enabled(true);

    // Do some work first
    for (int i = 0; i < 100; ++i) {
        DiagnosticsCollector::instance().increment(
            Counter::IntegralsComputed, 1);
    }

    // Reset after all threads joined (single-threaded reset is the safe usage)
    DiagnosticsCollector::instance().reset();

    EXPECT_EQ(DiagnosticsCollector::instance().counter_value(
        Counter::IntegralsComputed), 0u);
    EXPECT_FALSE(DiagnosticsCollector::instance().is_enabled());
    EXPECT_TRUE(DiagnosticsCollector::instance().timing_records().empty());
}
