// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_error_handling.cpp
/// @brief Unit tests for error handling and exception classes

#include <gtest/gtest.h>
#include <libaccint/utils/error_handling.hpp>

namespace libaccint {

// ============================================================================
// Test Exception base class
// ============================================================================

TEST(ExceptionTest, BaseException) {
    const std::string msg = "Test error message";
    Exception ex(msg);
    EXPECT_EQ(ex.what(), msg);
}

// ============================================================================
// Test InvalidArgumentException
// ============================================================================

TEST(ExceptionTest, InvalidArgumentExceptionThrow) {
    EXPECT_THROW(
        throw InvalidArgumentException("invalid param"),
        InvalidArgumentException
    );
}

TEST(ExceptionTest, InvalidArgumentExceptionCatchByType) {
    try {
        throw InvalidArgumentException("invalid param");
    } catch (const InvalidArgumentException& e) {
        EXPECT_STREQ(e.what(), "Invalid argument: invalid param");
    }
}

TEST(ExceptionTest, InvalidArgumentExceptionCatchByBase) {
    try {
        throw InvalidArgumentException("invalid param");
    } catch (const Exception& e) {
        EXPECT_STREQ(e.what(), "Invalid argument: invalid param");
    }
}

TEST(ExceptionTest, InvalidArgumentExceptionMessage) {
    InvalidArgumentException ex("param name is null");
    EXPECT_STREQ(ex.what(), "Invalid argument: param name is null");
}

// ============================================================================
// Test InvalidStateException
// ============================================================================

TEST(ExceptionTest, InvalidStateExceptionThrow) {
    EXPECT_THROW(
        throw InvalidStateException("engine not initialized"),
        InvalidStateException
    );
}

TEST(ExceptionTest, InvalidStateExceptionCatchByType) {
    try {
        throw InvalidStateException("engine not initialized");
    } catch (const InvalidStateException& e) {
        EXPECT_STREQ(e.what(), "Invalid state: engine not initialized");
    }
}

TEST(ExceptionTest, InvalidStateExceptionCatchByBase) {
    try {
        throw InvalidStateException("engine not initialized");
    } catch (const Exception& e) {
        EXPECT_STREQ(e.what(), "Invalid state: engine not initialized");
    }
}

TEST(ExceptionTest, InvalidStateExceptionMessage) {
    InvalidStateException ex("invalid configuration");
    EXPECT_STREQ(ex.what(), "Invalid state: invalid configuration");
}

// ============================================================================
// Test NotImplementedException
// ============================================================================

TEST(ExceptionTest, NotImplementedExceptionWithMessage) {
    try {
        throw NotImplementedException("feature X");
    } catch (const NotImplementedException& e) {
        EXPECT_STREQ(e.what(), "Not implemented: feature X");
    }
}

TEST(ExceptionTest, NotImplementedExceptionWithFeatureAndPhase) {
    try {
        throw NotImplementedException("GPU acceleration", "Phase 4");
    } catch (const NotImplementedException& e) {
        EXPECT_STREQ(e.what(), "Not implemented: GPU acceleration (planned for Phase 4)");
    }
}

TEST(ExceptionTest, NotImplementedExceptionCatchByType) {
    EXPECT_THROW(
        throw NotImplementedException("feature Y"),
        NotImplementedException
    );
}

TEST(ExceptionTest, NotImplementedExceptionCatchByBase) {
    try {
        throw NotImplementedException("feature Z");
    } catch (const Exception& e) {
        EXPECT_STREQ(e.what(), "Not implemented: feature Z");
    }
}

// ============================================================================
// Test MemoryException
// ============================================================================

TEST(ExceptionTest, MemoryExceptionThrow) {
    EXPECT_THROW(
        throw MemoryException("allocation failed"),
        MemoryException
    );
}

TEST(ExceptionTest, MemoryExceptionCatchByType) {
    try {
        throw MemoryException("out of memory");
    } catch (const MemoryException& e) {
        EXPECT_STREQ(e.what(), "Memory error: out of memory");
    }
}

TEST(ExceptionTest, MemoryExceptionCatchByBase) {
    try {
        throw MemoryException("allocation failed");
    } catch (const Exception& e) {
        EXPECT_STREQ(e.what(), "Memory error: allocation failed");
    }
}

TEST(ExceptionTest, MemoryExceptionMessage) {
    MemoryException ex("device memory exhausted");
    EXPECT_STREQ(ex.what(), "Memory error: device memory exhausted");
}

// ============================================================================
// Test BackendException
// ============================================================================

TEST(ExceptionTest, BackendExceptionThrow) {
    EXPECT_THROW(
        throw BackendException("CUDA", "kernel launch failed"),
        BackendException
    );
}

TEST(ExceptionTest, BackendExceptionCatchByType) {
    try {
        throw BackendException("CUDA", "memory copy failed");
    } catch (const BackendException& e) {
        EXPECT_STREQ(e.what(), "CUDA error: memory copy failed");
    }
}

TEST(ExceptionTest, BackendExceptionCatchByBase) {
    try {
        throw BackendException("CUDA", "synchronization timeout");
    } catch (const Exception& e) {
        EXPECT_STREQ(e.what(), "CUDA error: synchronization timeout");
    }
}

TEST(ExceptionTest, BackendExceptionMessage) {
    BackendException ex("CUDA", "invalid device");
    EXPECT_STREQ(ex.what(), "CUDA error: invalid device");
}

// ============================================================================
// Test NumericalException
// ============================================================================

TEST(ExceptionTest, NumericalExceptionThrow) {
    EXPECT_THROW(
        throw NumericalException("convergence failed"),
        NumericalException
    );
}

TEST(ExceptionTest, NumericalExceptionCatchByType) {
    try {
        throw NumericalException("overflow detected");
    } catch (const NumericalException& e) {
        EXPECT_STREQ(e.what(), "Numerical error: overflow detected");
    }
}

TEST(ExceptionTest, NumericalExceptionCatchByBase) {
    try {
        throw NumericalException("underflow in integral");
    } catch (const Exception& e) {
        EXPECT_STREQ(e.what(), "Numerical error: underflow in integral");
    }
}

TEST(ExceptionTest, NumericalExceptionMessage) {
    NumericalException ex("singular matrix");
    EXPECT_STREQ(ex.what(), "Numerical error: singular matrix");
}

// ============================================================================
// Test LIBACCINT_ASSERT macro
// ============================================================================

TEST(MacroTest, AssertDoesNotThrowWhenTrue) {
    // Should not throw when condition is true
    EXPECT_NO_THROW(LIBACCINT_ASSERT(true, "test message"));
}

TEST(MacroTest, AssertThrowsWhenFalse) {
    // Should throw InvalidStateException when condition is false
    EXPECT_THROW(
        LIBACCINT_ASSERT(false, "test message"),
        InvalidStateException
    );
}

TEST(MacroTest, AssertIncludesFileAndLine) {
    try {
        LIBACCINT_ASSERT(false, "custom error");
    } catch (const InvalidStateException& e) {
        std::string what(e.what());
        // Check that file name and line number are included
        EXPECT_TRUE(what.find("test_error_handling.cpp") != std::string::npos);
        EXPECT_TRUE(what.find("custom error") != std::string::npos);
    }
}

TEST(MacroTest, AssertWithComplexCondition) {
    int x = 5;
    int y = 10;
    EXPECT_NO_THROW(LIBACCINT_ASSERT(x < y, "x should be less than y"));
    EXPECT_THROW(
        LIBACCINT_ASSERT(x > y, "x should be greater than y"),
        InvalidStateException
    );
}

// ============================================================================
// Test LIBACCINT_NOT_IMPLEMENTED macro
// ============================================================================

TEST(MacroTest, NotImplementedThrows) {
    EXPECT_THROW(
        LIBACCINT_NOT_IMPLEMENTED("test feature"),
        NotImplementedException
    );
}

TEST(MacroTest, NotImplementedMessage) {
    try {
        LIBACCINT_NOT_IMPLEMENTED("distributed memory support");
    } catch (const NotImplementedException& e) {
        EXPECT_STREQ(e.what(), "Not implemented: distributed memory support");
    }
}

TEST(MacroTest, NotImplementedCatchByBase) {
    try {
        LIBACCINT_NOT_IMPLEMENTED("advanced feature");
    } catch (const Exception& e) {
        EXPECT_STREQ(e.what(), "Not implemented: advanced feature");
    }
}

// ============================================================================
// Test exception polymorphism
// ============================================================================

TEST(PolymorphismTest, CatchAllExceptionTypes) {
    // Test that all exception types can be caught by Exception
    std::vector<std::function<void()>> throwers = {
        []() { throw InvalidArgumentException("arg"); },
        []() { throw InvalidStateException("state"); },
        []() { throw NotImplementedException("feature"); },
        []() { throw MemoryException("memory"); },
        []() { throw BackendException("CUDA", "error"); },
        []() { throw NumericalException("numerical"); },
    };

    for (auto& thrower : throwers) {
        EXPECT_THROW(
            try {
                thrower();
            } catch (const Exception& e) {
                // Caught successfully by base class
                throw;
            },
            Exception
        );
    }
}

// ============================================================================
// Test catching by std::runtime_error (Task 1.4.5)
// ============================================================================

TEST(ExceptionHierarchy, CatchByRuntimeError) {
    // All libaccint::Exception subclasses must be catchable via std::runtime_error
    std::vector<std::function<void()>> throwers = {
        []() { throw InvalidArgumentException("arg"); },
        []() { throw InvalidStateException("state"); },
        []() { throw NotImplementedException("feature"); },
        []() { throw MemoryException("memory"); },
        []() { throw BackendException("CUDA", "error"); },
        []() { throw NumericalException("numerical"); },
    };

    for (auto& thrower : throwers) {
        EXPECT_THROW(
            try {
                thrower();
            } catch (const std::runtime_error& e) {
                // Caught successfully by std::runtime_error
                EXPECT_NE(e.what(), nullptr);
                throw;
            },
            std::runtime_error
        );
    }
}

TEST(ExceptionHierarchy, InvalidArgumentException_IsRuntimeError) {
    try {
        throw InvalidArgumentException("test param");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("Invalid argument"), std::string::npos);
        EXPECT_NE(msg.find("test param"), std::string::npos);
    }
}

TEST(ExceptionHierarchy, BackendException_IsRuntimeError) {
    try {
        throw BackendException("CUDA", "device error");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("CUDA"), std::string::npos);
        EXPECT_NE(msg.find("device error"), std::string::npos);
    }
}

// ============================================================================
// Test what() message contains contextual prefix (Task 1.4.5)
// ============================================================================

TEST(ExceptionHierarchy, WhatMessage) {
    // Each exception type should include its prefix in what()
    {
        InvalidArgumentException e("msg");
        EXPECT_NE(std::string(e.what()).find("Invalid argument:"), std::string::npos);
    }
    {
        InvalidStateException e("msg");
        EXPECT_NE(std::string(e.what()).find("Invalid state:"), std::string::npos);
    }
    {
        NotImplementedException e("msg");
        EXPECT_NE(std::string(e.what()).find("Not implemented:"), std::string::npos);
    }
    {
        MemoryException e("msg");
        EXPECT_NE(std::string(e.what()).find("Memory error:"), std::string::npos);
    }
    {
        BackendException e("GPU", "msg");
        EXPECT_NE(std::string(e.what()).find("GPU error:"), std::string::npos);
    }
    {
        NumericalException e("msg");
        EXPECT_NE(std::string(e.what()).find("Numerical error:"), std::string::npos);
    }
}

// ============================================================================
// Test preserving user message (Task 1.4.5)
// ============================================================================

TEST(ExceptionHierarchy, PreservesUserMessage) {
    std::string user_msg = "specific error detail 12345";

    InvalidArgumentException e1(user_msg);
    EXPECT_NE(std::string(e1.what()).find(user_msg), std::string::npos);

    InvalidStateException e2(user_msg);
    EXPECT_NE(std::string(e2.what()).find(user_msg), std::string::npos);

    NotImplementedException e3(user_msg);
    EXPECT_NE(std::string(e3.what()).find(user_msg), std::string::npos);

    MemoryException e4(user_msg);
    EXPECT_NE(std::string(e4.what()).find(user_msg), std::string::npos);

    NumericalException e5(user_msg);
    EXPECT_NE(std::string(e5.what()).find(user_msg), std::string::npos);
}

// ============================================================================
// Test empty message construction (Task 1.4.5)
// ============================================================================

TEST(ExceptionHierarchy, EmptyMessage) {
    // Constructing with an empty string should not crash
    EXPECT_NO_THROW({
        InvalidArgumentException e1("");
        EXPECT_NE(e1.what(), nullptr);
    });
    EXPECT_NO_THROW({
        InvalidStateException e2("");
        EXPECT_NE(e2.what(), nullptr);
    });
    EXPECT_NO_THROW({
        NotImplementedException e3("");
        EXPECT_NE(e3.what(), nullptr);
    });
    EXPECT_NO_THROW({
        MemoryException e4("");
        EXPECT_NE(e4.what(), nullptr);
    });
    EXPECT_NO_THROW({
        BackendException e5("", "");
        EXPECT_NE(e5.what(), nullptr);
    });
    EXPECT_NO_THROW({
        NumericalException e6("");
        EXPECT_NE(e6.what(), nullptr);
    });
}

// ============================================================================
// Test LIBACCINT_ASSERT with message content (Task 1.4.5)
// ============================================================================

TEST(Assert, ThrowsOnFalse_WithMessage) {
    try {
        LIBACCINT_ASSERT(false, "assertion test message");
        FAIL() << "Expected InvalidStateException";
    } catch (const InvalidStateException& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("assertion test message"), std::string::npos);
    }
}

TEST(Assert, NoThrowOnTrue) {
    EXPECT_NO_THROW(LIBACCINT_ASSERT(true, "should not throw"));
}

TEST(Assert, CatchableAsRuntimeError) {
    // LIBACCINT_ASSERT throws InvalidStateException which is Exception which is runtime_error
    EXPECT_THROW(
        LIBACCINT_ASSERT(false, "test"),
        std::runtime_error
    );
}

}  // namespace libaccint
