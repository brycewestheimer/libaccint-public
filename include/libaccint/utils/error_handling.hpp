// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

#include <stdexcept>
#include <string>

namespace libaccint {

/**
 * @brief Base exception class for LibAccInt errors
 */
class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * @brief Invalid input parameter exception
 */
class InvalidArgumentException : public Exception {
public:
    explicit InvalidArgumentException(const std::string& message)
        : Exception("Invalid argument: " + message) {}
};

/**
 * @brief Invalid configuration or state exception
 */
class InvalidStateException : public Exception {
public:
    explicit InvalidStateException(const std::string& message)
        : Exception("Invalid state: " + message) {}
};

/**
 * @brief Feature not yet implemented
 */
class NotImplementedException : public Exception {
public:
    explicit NotImplementedException(const std::string& message)
        : Exception("Not implemented: " + message) {}

    NotImplementedException(const std::string& feature, const std::string& phase)
        : Exception("Not implemented: " + feature + " (planned for " + phase + ")") {}
};

/**
 * @brief Resource allocation failure
 */
class MemoryException : public Exception {
public:
    explicit MemoryException(const std::string& message)
        : Exception("Memory error: " + message) {}
};

/**
 * @brief Numerical error (overflow, underflow, convergence failure)
 */
class NumericalException : public Exception {
public:
    explicit NumericalException(const std::string& message)
        : Exception("Numerical error: " + message) {}
};

// ============================================================================
// Macros for common error patterns
// ============================================================================

/**
 * @brief Assert a condition and throw InvalidStateException if false
 *
 * Includes file and line information in the error message.
 *
 * @param cond The condition to assert
 * @param msg The error message if condition is false
 */
#define LIBACCINT_ASSERT(cond, msg)                                          \
    do {                                                                      \
        if (!(cond)) {                                                        \
            throw libaccint::InvalidStateException(                           \
                std::string(msg) + " (at " + __FILE__ + ":" +                \
                std::to_string(__LINE__) + ")"                               \
            );                                                                \
        }                                                                     \
    } while (false)

/**
 * @brief Throw NotImplementedException for a feature
 *
 * @param feature The name of the feature that is not implemented
 */
#define LIBACCINT_NOT_IMPLEMENTED(feature)                                   \
    throw libaccint::NotImplementedException(feature)

} // namespace libaccint
