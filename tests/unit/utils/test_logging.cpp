// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_logging.cpp
/// @brief Tests for structured logging infrastructure (Task 25.3.2)

#include <libaccint/utils/logging.hpp>

#include <gtest/gtest.h>
#include <memory>
#include <thread>

using namespace libaccint::logging;
using libaccint::Size;

class LoggingTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::instance().reset();
    }

    void TearDown() override {
        Logger::instance().reset();
    }
};

TEST_F(LoggingTest, DefaultLevelIsOff) {
    EXPECT_EQ(Logger::instance().level(), LogLevel::Off);
    EXPECT_FALSE(Logger::instance().is_enabled(LogLevel::Info));
    EXPECT_FALSE(Logger::instance().is_enabled(LogLevel::Error));
}

TEST_F(LoggingTest, SetLevel) {
    Logger::instance().set_level(LogLevel::Info);
    EXPECT_EQ(Logger::instance().level(), LogLevel::Info);
    EXPECT_TRUE(Logger::instance().is_enabled(LogLevel::Info));
    EXPECT_TRUE(Logger::instance().is_enabled(LogLevel::Warning));
    EXPECT_TRUE(Logger::instance().is_enabled(LogLevel::Error));
    EXPECT_FALSE(Logger::instance().is_enabled(LogLevel::Debug));
    EXPECT_FALSE(Logger::instance().is_enabled(LogLevel::Trace));
}

TEST_F(LoggingTest, StringBufferSink) {
    auto sink = std::make_shared<StringBufferSink>();
    Logger::instance().add_sink(sink);
    Logger::instance().set_level(LogLevel::Debug);

    Logger::instance().log(LogLevel::Info, "engine", "Test message");
    Logger::instance().log(LogLevel::Debug, "basis", "Debug message");

    EXPECT_EQ(sink->count(), 2u);
    EXPECT_EQ(sink->count_at_level(LogLevel::Info), 1u);
    EXPECT_EQ(sink->count_at_level(LogLevel::Debug), 1u);

    auto contents = sink->contents();
    EXPECT_NE(contents.find("Test message"), std::string::npos);
    EXPECT_NE(contents.find("Debug message"), std::string::npos);
}

TEST_F(LoggingTest, FiltersBelowLevel) {
    auto sink = std::make_shared<StringBufferSink>();
    Logger::instance().add_sink(sink);
    Logger::instance().set_level(LogLevel::Warning);

    Logger::instance().log(LogLevel::Info, "engine", "Should be filtered");
    Logger::instance().log(LogLevel::Debug, "engine", "Should be filtered too");
    Logger::instance().log(LogLevel::Warning, "engine", "Should appear");

    EXPECT_EQ(sink->count(), 1u);
    EXPECT_EQ(sink->count_at_level(LogLevel::Warning), 1u);
}

TEST_F(LoggingTest, MultipleSinks) {
    auto sink1 = std::make_shared<StringBufferSink>();
    auto sink2 = std::make_shared<StringBufferSink>();
    Logger::instance().add_sink(sink1);
    Logger::instance().add_sink(sink2);
    Logger::instance().set_level(LogLevel::Info);

    Logger::instance().log(LogLevel::Info, "test", "Both sinks");

    EXPECT_EQ(sink1->count(), 1u);
    EXPECT_EQ(sink2->count(), 1u);
}

TEST_F(LoggingTest, ClearSinks) {
    auto sink = std::make_shared<StringBufferSink>();
    Logger::instance().add_sink(sink);
    Logger::instance().set_level(LogLevel::Info);

    Logger::instance().log(LogLevel::Info, "test", "Before clear");
    EXPECT_EQ(sink->count(), 1u);

    Logger::instance().clear_sinks();
    Logger::instance().log(LogLevel::Info, "test", "After clear");
    // sink is still there but disconnected, count should stay 1
    EXPECT_EQ(sink->count(), 1u);
}

TEST_F(LoggingTest, LogLevelStrings) {
    EXPECT_EQ(log_level_string(LogLevel::Trace), "TRACE");
    EXPECT_EQ(log_level_string(LogLevel::Debug), "DEBUG");
    EXPECT_EQ(log_level_string(LogLevel::Info), "INFO");
    EXPECT_EQ(log_level_string(LogLevel::Warning), "WARN");
    EXPECT_EQ(log_level_string(LogLevel::Error), "ERROR");
    EXPECT_EQ(log_level_string(LogLevel::Fatal), "FATAL");
    EXPECT_EQ(log_level_string(LogLevel::Off), "OFF");
}

TEST_F(LoggingTest, LogEntryHasTimestamp) {
    auto sink = std::make_shared<StringBufferSink>();
    Logger::instance().add_sink(sink);
    Logger::instance().set_level(LogLevel::Info);

    auto before = std::chrono::steady_clock::now();
    Logger::instance().log(LogLevel::Info, "test", "Timestamp test");
    auto after = std::chrono::steady_clock::now();

    ASSERT_EQ(sink->count(), 1u);
    const auto& entry = sink->entries()[0];
    EXPECT_GE(entry.timestamp, before);
    EXPECT_LE(entry.timestamp, after);
}

TEST_F(LoggingTest, LogEntryCategory) {
    auto sink = std::make_shared<StringBufferSink>();
    Logger::instance().add_sink(sink);
    Logger::instance().set_level(LogLevel::Info);

    Logger::instance().log(LogLevel::Info, "my_category", "Cat test");

    ASSERT_EQ(sink->count(), 1u);
    EXPECT_EQ(sink->entries()[0].category, "my_category");
}

TEST_F(LoggingTest, MacroLogging) {
    auto sink = std::make_shared<StringBufferSink>();
    Logger::instance().add_sink(sink);
    Logger::instance().set_level(LogLevel::Debug);

    LIBACCINT_LOG_INFO("engine", "Macro info");
    LIBACCINT_LOG_DEBUG("basis", "Macro debug");
    LIBACCINT_LOG_WARNING("kernel", "Macro warning");
    LIBACCINT_LOG_TRACE("test", "Should not appear");  // Below Debug

    EXPECT_EQ(sink->count(), 3u);
}

TEST_F(LoggingTest, ThreadSafety) {
    auto sink = std::make_shared<StringBufferSink>();
    Logger::instance().add_sink(sink);
    Logger::instance().set_level(LogLevel::Info);

    constexpr int n_threads = 4;
    constexpr int n_messages = 100;

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([t]() {
            for (int i = 0; i < n_messages; ++i) {
                Logger::instance().log(LogLevel::Info,
                    "thread_" + std::to_string(t),
                    "Message " + std::to_string(i));
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(sink->count(), static_cast<Size>(n_threads * n_messages));
}

TEST_F(LoggingTest, BufferClear) {
    auto sink = std::make_shared<StringBufferSink>();
    Logger::instance().add_sink(sink);
    Logger::instance().set_level(LogLevel::Info);

    Logger::instance().log(LogLevel::Info, "test", "Message 1");
    EXPECT_EQ(sink->count(), 1u);

    sink->clear();
    EXPECT_EQ(sink->count(), 0u);
}

TEST_F(LoggingTest, Reset) {
    auto sink = std::make_shared<StringBufferSink>();
    Logger::instance().add_sink(sink);
    Logger::instance().set_level(LogLevel::Info);
    Logger::instance().log(LogLevel::Info, "test", "Before reset");
    EXPECT_EQ(sink->count(), 1u);

    Logger::instance().reset();
    EXPECT_EQ(Logger::instance().level(), LogLevel::Off);
    // Sink was disconnected by reset
    Logger::instance().log(LogLevel::Info, "test", "After reset");
    EXPECT_EQ(sink->count(), 1u);  // Should not increase
}
