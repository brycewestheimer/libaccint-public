// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file logging.cpp
/// @brief Implementation of structured logging infrastructure

#include <libaccint/utils/logging.hpp>

#include <iostream>

namespace libaccint::logging {

// ============================================================================
// LogLevel string conversion
// ============================================================================

std::string_view log_level_string(LogLevel level) noexcept {
    switch (level) {
        case LogLevel::Trace:   return "TRACE";
        case LogLevel::Debug:   return "DEBUG";
        case LogLevel::Info:    return "INFO";
        case LogLevel::Warning: return "WARN";
        case LogLevel::Error:   return "ERROR";
        case LogLevel::Fatal:   return "FATAL";
        case LogLevel::Off:     return "OFF";
    }
    return "UNKNOWN";
}

// ============================================================================
// StringBufferSink
// ============================================================================

void StringBufferSink::write(const LogEntry& entry) {
    entries_.push_back(entry);
}

std::string StringBufferSink::contents() const {
    std::ostringstream oss;
    for (const auto& entry : entries_) {
        oss << "[" << log_level_string(entry.level) << "] "
            << "[" << entry.category << "] "
            << entry.message << "\n";
    }
    return oss.str();
}

Size StringBufferSink::count_at_level(LogLevel level) const noexcept {
    Size count = 0;
    for (const auto& entry : entries_) {
        if (entry.level == level) ++count;
    }
    return count;
}

// ============================================================================
// StderrSink
// ============================================================================

void StderrSink::write(const LogEntry& entry) {
    std::cerr << "[" << log_level_string(entry.level) << "] "
              << "[" << entry.category << "] "
              << entry.message << "\n";
}

// ============================================================================
// Logger
// ============================================================================

Logger& Logger::instance() noexcept {
    static Logger instance;
    return instance;
}

void Logger::set_level(LogLevel level) noexcept {
    min_level_.store(level, std::memory_order_release);
}

LogLevel Logger::level() const noexcept {
    return min_level_.load(std::memory_order_acquire);
}

bool Logger::is_enabled(LogLevel level) const noexcept {
    return static_cast<int>(level) >= static_cast<int>(min_level_.load(std::memory_order_acquire));
}

void Logger::add_sink(std::shared_ptr<LogSink> sink) {
    std::lock_guard<std::mutex> lock(mutex_);
    sinks_.push_back(std::move(sink));
}

void Logger::clear_sinks() {
    std::lock_guard<std::mutex> lock(mutex_);
    sinks_.clear();
}

void Logger::log(LogLevel level, std::string_view category,
                 std::string message,
                 const char* file, int line) {
    if (!is_enabled(level)) return;

    LogEntry entry;
    entry.level = level;
    entry.message = std::move(message);
    entry.category = std::string(category);
    entry.source_file = file ? file : "";
    entry.source_line = line;
    entry.timestamp = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& sink : sinks_) {
        sink->write(entry);
    }
}

void Logger::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    min_level_.store(LogLevel::Off, std::memory_order_release);
    sinks_.clear();
}

}  // namespace libaccint::logging
