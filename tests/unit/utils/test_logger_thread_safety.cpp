// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_logger_thread_safety.cpp
/// @brief Thread-safety tests for Logger (Task 1.3.5)

#include <libaccint/utils/logging.hpp>

#include <gtest/gtest.h>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace libaccint::logging;
using libaccint::Size;

class LoggerThreadSafetyTest : public ::testing::Test {
protected:
    static constexpr int NUM_THREADS = 8;
    static constexpr int MESSAGES_PER_THREAD = 500;

    void SetUp() override {
        Logger::instance().reset();
    }

    void TearDown() override {
        Logger::instance().reset();
    }
};

TEST_F(LoggerThreadSafetyTest, ConcurrentLogging) {
    auto sink = std::make_shared<StringBufferSink>();
    Logger::instance().add_sink(sink);
    Logger::instance().set_level(LogLevel::Trace);

    std::vector<std::thread> threads;
    std::atomic<bool> start{false};

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&start, t]() {
            while (!start.load(std::memory_order_acquire)) {}
            for (int i = 0; i < MESSAGES_PER_THREAD; ++i) {
                Logger::instance().log(
                    LogLevel::Info, "test",
                    "Thread " + std::to_string(t) + " msg " + std::to_string(i));
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto& th : threads) {
        th.join();
    }

    // All messages should have been captured
    EXPECT_EQ(sink->count(),
              static_cast<Size>(NUM_THREADS * MESSAGES_PER_THREAD));
}

TEST_F(LoggerThreadSafetyTest, ConcurrentLevelChange) {
    auto sink = std::make_shared<StringBufferSink>();
    Logger::instance().add_sink(sink);
    Logger::instance().set_level(LogLevel::Trace);

    std::atomic<bool> start{false};
    std::atomic<bool> stop{false};

    // Writer thread: continuously changes log level
    std::thread level_changer([&]() {
        while (!start.load(std::memory_order_acquire)) {}
        LogLevel levels[] = {
            LogLevel::Trace, LogLevel::Debug, LogLevel::Info,
            LogLevel::Warning, LogLevel::Error
        };
        for (int i = 0; !stop.load(std::memory_order_acquire); ++i) {
            Logger::instance().set_level(levels[i % 5]);
        }
    });

    // Logger threads
    std::vector<std::thread> loggers;
    for (int t = 0; t < NUM_THREADS; ++t) {
        loggers.emplace_back([&start, &stop, t]() {
            while (!start.load(std::memory_order_acquire)) {}
            for (int i = 0; i < MESSAGES_PER_THREAD; ++i) {
                Logger::instance().log(
                    LogLevel::Info, "test",
                    "Thread " + std::to_string(t) + " msg " + std::to_string(i));
            }
            stop.store(true, std::memory_order_release);
        });
    }

    start.store(true, std::memory_order_release);
    for (auto& th : loggers) {
        th.join();
    }
    stop.store(true, std::memory_order_release);
    level_changer.join();

    // Some messages may have been filtered by level changes; total is <=
    // NUM_THREADS * MESSAGES_PER_THREAD. No crash/TSan violation is the test.
    SUCCEED();
}

TEST_F(LoggerThreadSafetyTest, ConcurrentIsEnabled) {
    std::atomic<bool> start{false};
    std::atomic<bool> stop{false};
    constexpr int ITERATIONS = 10000;

    // Writer thread: toggles log level
    std::thread writer([&]() {
        while (!start.load(std::memory_order_acquire)) {}
        for (int i = 0; i < ITERATIONS; ++i) {
            Logger::instance().set_level(
                (i % 2 == 0) ? LogLevel::Info : LogLevel::Off);
        }
        stop.store(true, std::memory_order_release);
    });

    // Reader threads
    std::vector<std::thread> readers;
    for (int t = 0; t < NUM_THREADS; ++t) {
        readers.emplace_back([&]() {
            while (!start.load(std::memory_order_acquire)) {}
            while (!stop.load(std::memory_order_acquire)) {
                [[maybe_unused]] bool en =
                    Logger::instance().is_enabled(LogLevel::Info);
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

TEST_F(LoggerThreadSafetyTest, MessageIntegrity) {
    auto sink = std::make_shared<StringBufferSink>();
    Logger::instance().add_sink(sink);
    Logger::instance().set_level(LogLevel::Trace);

    std::vector<std::thread> threads;
    std::atomic<bool> start{false};

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&start, t]() {
            while (!start.load(std::memory_order_acquire)) {}
            for (int i = 0; i < 50; ++i) {
                // Each thread logs a unique identifiable message
                std::string msg = "MARKER_T" + std::to_string(t)
                                + "_I" + std::to_string(i);
                Logger::instance().log(LogLevel::Info, "integrity", msg);
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto& th : threads) {
        th.join();
    }

    // Verify each marker message appears intact
    auto contents = sink->contents();
    for (int t = 0; t < NUM_THREADS; ++t) {
        for (int i = 0; i < 50; ++i) {
            std::string marker = "MARKER_T" + std::to_string(t)
                               + "_I" + std::to_string(i);
            EXPECT_NE(contents.find(marker), std::string::npos)
                << "Missing message: " << marker;
        }
    }
}

TEST_F(LoggerThreadSafetyTest, SinkManagement) {
    // Test add_sink and clear_sinks are safe when called sequentially
    // (no concurrent logging)
    auto sink1 = std::make_shared<StringBufferSink>();
    auto sink2 = std::make_shared<StringBufferSink>();

    Logger::instance().add_sink(sink1);
    Logger::instance().add_sink(sink2);
    Logger::instance().set_level(LogLevel::Info);

    Logger::instance().log(LogLevel::Info, "test", "Before clear");
    EXPECT_EQ(sink1->count(), 1u);
    EXPECT_EQ(sink2->count(), 1u);

    Logger::instance().clear_sinks();

    Logger::instance().log(LogLevel::Info, "test", "After clear");
    // Sinks should no longer receive messages
    EXPECT_EQ(sink1->count(), 1u);
    EXPECT_EQ(sink2->count(), 1u);
}
