// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file logging.hpp
/// @brief Optional structured logging infrastructure for LibAccInt
///
/// Provides a lightweight, zero-overhead (when disabled) logging system with
/// configurable log levels and structured output. Thread-safe by design.

#include <libaccint/core/types.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace libaccint::logging {

/// @brief Log severity levels
enum class LogLevel : int {
    Trace = 0,   ///< Fine-grained diagnostic information
    Debug = 1,   ///< Debugging information
    Info = 2,    ///< General informational messages
    Warning = 3, ///< Potential issues
    Error = 4,   ///< Errors that allow continued operation
    Fatal = 5,   ///< Unrecoverable errors
    Off = 6,     ///< Logging disabled
};

/// @brief Convert LogLevel to string
[[nodiscard]] std::string_view log_level_string(LogLevel level) noexcept;

/// @brief Structured log entry
struct LogEntry {
    LogLevel level{LogLevel::Info};
    std::string message;
    std::string category;      ///< e.g., "engine", "basis", "kernel"
    std::string source_file;
    int source_line{0};
    std::chrono::steady_clock::time_point timestamp;
};

/// @brief Log sink interface — destinations for log output
class LogSink {
public:
    virtual ~LogSink() = default;
    virtual void write(const LogEntry& entry) = 0;
};

/// @brief Sink that writes to a string buffer (for testing)
class StringBufferSink : public LogSink {
public:
    void write(const LogEntry& entry) override;

    /// @brief Get all captured entries
    [[nodiscard]] const std::vector<LogEntry>& entries() const noexcept {
        return entries_;
    }

    /// @brief Get all captured messages as one string
    [[nodiscard]] std::string contents() const;

    /// @brief Clear the buffer
    void clear() noexcept { entries_.clear(); }

    /// @brief Get number of captured entries
    [[nodiscard]] Size count() const noexcept { return entries_.size(); }

    /// @brief Count entries at a specific level
    [[nodiscard]] Size count_at_level(LogLevel level) const noexcept;

private:
    std::vector<LogEntry> entries_;
};

/// @brief Sink that writes to stderr
class StderrSink : public LogSink {
public:
    void write(const LogEntry& entry) override;
};

/// @brief Central logger — thread-safe, singleton-accessible
///
/// Usage:
/// @code
///   auto& logger = Logger::instance();
///   logger.set_level(LogLevel::Debug);
///   logger.add_sink(std::make_shared<StderrSink>());
///   logger.log(LogLevel::Info, "engine", "Computing overlap matrix");
/// @endcode
class Logger {
public:
    /// @brief Get the global logger instance
    static Logger& instance() noexcept;

    /// @brief Set minimum log level (messages below this are discarded)
    void set_level(LogLevel level) noexcept;

    /// @brief Get current minimum log level
    [[nodiscard]] LogLevel level() const noexcept;

    /// @brief Check if a log level is enabled
    [[nodiscard]] bool is_enabled(LogLevel level) const noexcept;

    /// @brief Add a log sink
    void add_sink(std::shared_ptr<LogSink> sink);

    /// @brief Remove all sinks
    void clear_sinks();

    /// @brief Log a message
    void log(LogLevel level, std::string_view category,
             std::string message,
             const char* file = "", int line = 0);

    /// @brief Reset logger to default state (for testing)
    void reset();

private:
    Logger() = default;

    mutable std::mutex mutex_;
    std::atomic<LogLevel> min_level_{LogLevel::Off};  // Disabled by default
    std::vector<std::shared_ptr<LogSink>> sinks_;
};

// ============================================================================
// Convenience Macros
// ============================================================================

/// @brief Log a message at the given level with category
#define LIBACCINT_LOG(level, category, msg)                                  \
    do {                                                                      \
        auto& logger_ = ::libaccint::logging::Logger::instance();            \
        if (logger_.is_enabled(level)) {                                     \
            logger_.log(level, category, msg, __FILE__, __LINE__);           \
        }                                                                     \
    } while (false)

#define LIBACCINT_LOG_TRACE(category, msg)   LIBACCINT_LOG(::libaccint::logging::LogLevel::Trace, category, msg)
#define LIBACCINT_LOG_DEBUG(category, msg)   LIBACCINT_LOG(::libaccint::logging::LogLevel::Debug, category, msg)
#define LIBACCINT_LOG_INFO(category, msg)    LIBACCINT_LOG(::libaccint::logging::LogLevel::Info, category, msg)
#define LIBACCINT_LOG_WARNING(category, msg) LIBACCINT_LOG(::libaccint::logging::LogLevel::Warning, category, msg)
#define LIBACCINT_LOG_ERROR(category, msg)   LIBACCINT_LOG(::libaccint::logging::LogLevel::Error, category, msg)

}  // namespace libaccint::logging
