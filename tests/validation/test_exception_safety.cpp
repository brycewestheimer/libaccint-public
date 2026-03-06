// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_exception_safety.cpp
/// @brief Exception safety verification tests (Task 25.4.2)
///
/// Verifies strong/basic exception safety guarantees:
/// - Objects remain valid after failed operations
/// - No resource leaks on exception paths
/// - Error messages are descriptive

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/core/backend.hpp>
#include <libaccint/utils/error_handling.hpp>
#include <libaccint/utils/input_validation.hpp>

#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

using namespace libaccint;

// ============================================================================
// Exception Hierarchy Tests
// ============================================================================

TEST(ExceptionSafety, ExceptionHierarchy) {
    // All custom exceptions derive from Exception -> std::runtime_error
    EXPECT_THROW(
        throw InvalidArgumentException("test"),
        Exception);
    EXPECT_THROW(
        throw InvalidArgumentException("test"),
        std::runtime_error);
    EXPECT_THROW(
        throw InvalidStateException("test"),
        Exception);
    EXPECT_THROW(
        throw NotImplementedException("test"),
        Exception);
    EXPECT_THROW(
        throw MemoryException("test"),
        Exception);
    EXPECT_THROW(
        throw BackendError(BackendType::CUDA, "test"),
        Exception);
    EXPECT_THROW(
        throw NumericalException("test"),
        Exception);
}

TEST(ExceptionSafety, ExceptionMessages) {
    // Verify error messages include useful context
    try {
        throw InvalidArgumentException("exponent must be positive");
    } catch (const Exception& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("Invalid argument"), std::string::npos);
        EXPECT_NE(msg.find("exponent must be positive"), std::string::npos);
    }

    try {
        throw InvalidStateException("engine not initialized");
    } catch (const Exception& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("Invalid state"), std::string::npos);
        EXPECT_NE(msg.find("engine not initialized"), std::string::npos);
    }

    try {
        throw BackendError(BackendType::CUDA, "device not found");
    } catch (const Exception& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("CUDA"), std::string::npos);
        EXPECT_NE(msg.find("device not found"), std::string::npos);
    }
}

TEST(ExceptionSafety, NotImplementedWithPhase) {
    try {
        throw NotImplementedException("feature_x", "Phase 30");
    } catch (const Exception& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("feature_x"), std::string::npos);
        EXPECT_NE(msg.find("Phase 30"), std::string::npos);
    }
}

// ============================================================================
// Shell Exception Safety
// ============================================================================

TEST(ExceptionSafety, ShellInvalidAMDoesNotLeak) {
    // Constructing a shell with invalid AM should throw but not leak
    EXPECT_THROW(
        Shell(-1, {0.0, 0.0, 0.0}, {1.0}, {1.0}),
        InvalidArgumentException);
}

TEST(ExceptionSafety, ShellEmptyExponentsDoesNotLeak) {
    EXPECT_THROW(
        Shell(0, {0.0, 0.0, 0.0}, {}, {}),
        InvalidArgumentException);
}

TEST(ExceptionSafety, ShellMismatchedArraysDoesNotLeak) {
    EXPECT_THROW(
        Shell(0, {0.0, 0.0, 0.0}, {1.0, 2.0}, {1.0}),
        InvalidArgumentException);
}

// ============================================================================
// BasisSet Exception Safety
// ============================================================================

TEST(ExceptionSafety, BasisSetShellIndexOutOfBounds) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);

    // Accessing valid index should work
    EXPECT_NO_THROW([[maybe_unused]] auto& s = basis.shell(0));

    // Accessing out-of-bounds index should throw
    EXPECT_THROW([[maybe_unused]] auto& s = basis.shell(999), InvalidArgumentException);
}

TEST(ExceptionSafety, BasisSetRemainsValidAfterBadAccess) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);

    // Bad access
    try {
        [[maybe_unused]] auto& s = basis.shell(999);
    } catch (const InvalidArgumentException&) {
        // Expected
    }

    // BasisSet should still be usable
    EXPECT_GT(basis.n_shells(), 0u);
    EXPECT_NO_THROW([[maybe_unused]] auto& s = basis.shell(0));
}

// ============================================================================
// Engine Exception Safety
// ============================================================================

TEST(ExceptionSafety, EngineWithValidBasis) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);

    EXPECT_NO_THROW({
        Engine engine(basis);
        std::vector<Real> S;
        engine.compute_overlap_matrix(S);
    });
}

// ============================================================================
// Builtin Basis Exception Safety
// ============================================================================

TEST(ExceptionSafety, UnsupportedElement) {
    // Try to create STO-3G for an unsupported element
    std::vector<data::Atom> atoms = {{99, {0.0, 0.0, 0.0}}};
    EXPECT_THROW(data::create_sto3g(atoms), InvalidArgumentException);
}

TEST(ExceptionSafety, UnsupportedBasisName) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    EXPECT_THROW(data::create_builtin_basis("nonexistent-basis", atoms),
                 InvalidArgumentException);
}

// ============================================================================
// Validation Exception Safety
// ============================================================================

TEST(ExceptionSafety, ValidationThrowsDescriptive) {
    // validate_shell_params should throw with useful info
    try {
        std::vector<Real> exp = {-1.0};
        std::vector<Real> coeff = {1.0};
        validation::validate_shell_params(0, exp, coeff);
        FAIL() << "Should have thrown";
    } catch (const InvalidArgumentException& e) {
        std::string msg = e.what();
        // Should mention exponent and the problematic value
        EXPECT_NE(msg.find("Exponent"), std::string::npos);
    }
}

TEST(ExceptionSafety, ValidationDoesNotCorruptState) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);

    // Multiple failed validations should not corrupt the basis
    for (int i = 0; i < 10; ++i) {
        try {
            validation::validate_shell_index(basis, 999);
        } catch (const InvalidArgumentException&) {
            // Expected
        }
    }

    // Basis should still work normally
    auto result = validation::validate_basis_set(basis);
    EXPECT_TRUE(result);
}

// ============================================================================
// LIBACCINT_ASSERT Macro Tests
// ============================================================================

TEST(ExceptionSafety, AssertMacroTrue) {
    EXPECT_NO_THROW(LIBACCINT_ASSERT(true, "should not throw"));
}

TEST(ExceptionSafety, AssertMacroFalse) {
    EXPECT_THROW(
        LIBACCINT_ASSERT(false, "assertion failed"),
        InvalidStateException);
}

TEST(ExceptionSafety, AssertMacroMessage) {
    try {
        LIBACCINT_ASSERT(false, "custom error message");
        FAIL() << "Should have thrown";
    } catch (const InvalidStateException& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("custom error message"), std::string::npos);
    }
}

// ============================================================================
// Strong Guarantee Tests
// ============================================================================

TEST(ExceptionSafety, ShellStrongGuarantee) {
    // If Shell construction fails, no resources are leaked
    // (verified by ASAN in sanitizer builds)
    for (int i = 0; i < 100; ++i) {
        try {
            Shell s(-1, {0.0, 0.0, 0.0}, {1.0}, {1.0});
        } catch (const InvalidArgumentException&) {
            // Expected — no leak
        }
    }
}

TEST(ExceptionSafety, BasisSetStrongGuarantee) {
    // After a failed shell access, BasisSet is fully functional
    std::vector<data::Atom> atoms = {
        {1, {0.0, 0.0, 0.0}},
        {6, {2.0, 0.0, 0.0}}
    };
    auto basis = data::create_sto3g(atoms);

    try {
        [[maybe_unused]] auto& s = basis.shell(999);
    } catch (const InvalidArgumentException&) {}

    // All operations should still work
    Engine engine(basis);
    std::vector<Real> S;
    engine.compute_overlap_matrix(S);
    EXPECT_GT(S.size(), 0u);
}
