// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_input_validation.cpp
/// @brief Tests for public API input validation (Task 25.4.1)

#include <libaccint/utils/input_validation.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/data/builtin_basis.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

using namespace libaccint;
using namespace libaccint::validation;

// ============================================================================
// Shell Validation Tests
// ============================================================================

TEST(InputValidation, ValidShellParams) {
    std::vector<Real> exp = {3.42525091, 0.62391373, 0.16885540};
    std::vector<Real> coeff = {0.15432897, 0.53532814, 0.44463454};
    EXPECT_NO_THROW(validate_shell_params(0, exp, coeff));
}

TEST(InputValidation, NegativeAngularMomentum) {
    std::vector<Real> exp = {1.0};
    std::vector<Real> coeff = {1.0};
    EXPECT_THROW(validate_shell_params(-1, exp, coeff), InvalidArgumentException);
}

TEST(InputValidation, AngularMomentumTooHigh) {
    std::vector<Real> exp = {1.0};
    std::vector<Real> coeff = {1.0};
    EXPECT_THROW(validate_shell_params(MAX_ANGULAR_MOMENTUM + 1, exp, coeff),
                 InvalidArgumentException);
}

TEST(InputValidation, EmptyExponents) {
    std::vector<Real> exp;
    std::vector<Real> coeff = {1.0};
    EXPECT_THROW(validate_shell_params(0, exp, coeff), InvalidArgumentException);
}

TEST(InputValidation, EmptyCoefficients) {
    std::vector<Real> exp = {1.0};
    std::vector<Real> coeff;
    EXPECT_THROW(validate_shell_params(0, exp, coeff), InvalidArgumentException);
}

TEST(InputValidation, MismatchedSizes) {
    std::vector<Real> exp = {1.0, 2.0};
    std::vector<Real> coeff = {1.0};
    EXPECT_THROW(validate_shell_params(0, exp, coeff), InvalidArgumentException);
}

TEST(InputValidation, NonPositiveExponent) {
    std::vector<Real> exp = {1.0, -0.5};
    std::vector<Real> coeff = {1.0, 1.0};
    EXPECT_THROW(validate_shell_params(0, exp, coeff), InvalidArgumentException);
}

TEST(InputValidation, ZeroExponent) {
    std::vector<Real> exp = {0.0};
    std::vector<Real> coeff = {1.0};
    EXPECT_THROW(validate_shell_params(0, exp, coeff), InvalidArgumentException);
}

TEST(InputValidation, NaNExponent) {
    std::vector<Real> exp = {std::numeric_limits<Real>::quiet_NaN()};
    std::vector<Real> coeff = {1.0};
    EXPECT_THROW(validate_shell_params(0, exp, coeff), InvalidArgumentException);
}

TEST(InputValidation, InfCoefficient) {
    std::vector<Real> exp = {1.0};
    std::vector<Real> coeff = {std::numeric_limits<Real>::infinity()};
    EXPECT_THROW(validate_shell_params(0, exp, coeff), InvalidArgumentException);
}

TEST(InputValidation, ValidShell) {
    Shell s(0, {0.0, 0.0, 0.0}, {1.0}, {1.0});
    auto result = validate_shell(s);
    EXPECT_TRUE(result);
}

TEST(InputValidation, InvalidShellDefault) {
    Shell s;  // Default-constructed
    auto result = validate_shell(s);
    EXPECT_FALSE(result);
}

// ============================================================================
// BasisSet Validation Tests
// ============================================================================

TEST(InputValidation, ValidBasisSet) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);
    auto result = validate_basis_set(basis);
    EXPECT_TRUE(result);
}

TEST(InputValidation, EmptyBasisSet) {
    BasisSet empty;
    auto result = validate_basis_set(empty);
    EXPECT_FALSE(result);
}

TEST(InputValidation, MatrixSizeValid) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);
    Size nbf = basis.n_basis_functions();
    EXPECT_NO_THROW(validate_matrix_size(basis, nbf * nbf));
}

TEST(InputValidation, MatrixSizeInvalid) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);
    EXPECT_THROW(validate_matrix_size(basis, 0), InvalidArgumentException);
    EXPECT_THROW(validate_matrix_size(basis, 999), InvalidArgumentException);
}

TEST(InputValidation, DensityMatrixValid) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);
    Size nbf = basis.n_basis_functions();
    std::vector<Real> density(nbf * nbf, 0.0);
    EXPECT_NO_THROW(validate_density_matrix(basis, density));
}

TEST(InputValidation, DensityMatrixWrongSize) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);
    std::vector<Real> density(5, 0.0);
    EXPECT_THROW(validate_density_matrix(basis, density), InvalidArgumentException);
}

TEST(InputValidation, DensityMatrixNaN) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);
    Size nbf = basis.n_basis_functions();
    std::vector<Real> density(nbf * nbf, 0.0);
    density[0] = std::numeric_limits<Real>::quiet_NaN();
    EXPECT_THROW(validate_density_matrix(basis, density), NumericalException);
}

// ============================================================================
// Index Validation Tests
// ============================================================================

TEST(InputValidation, ShellIndexValid) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);
    EXPECT_NO_THROW(validate_shell_index(basis, 0));
}

TEST(InputValidation, ShellIndexOutOfBounds) {
    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    auto basis = data::create_sto3g(atoms);
    EXPECT_THROW(validate_shell_index(basis, 999), InvalidArgumentException);
}

TEST(InputValidation, AtomIndexValid) {
    EXPECT_NO_THROW(validate_atom_index(0, 5));
    EXPECT_NO_THROW(validate_atom_index(4, 5));
}

TEST(InputValidation, AtomIndexOutOfBounds) {
    EXPECT_THROW(validate_atom_index(5, 5), InvalidArgumentException);
    EXPECT_THROW(validate_atom_index(100, 5), InvalidArgumentException);
}

// ============================================================================
// Numerical Validation Tests
// ============================================================================

TEST(InputValidation, FiniteValid) {
    std::vector<Real> data = {1.0, 2.0, -3.0, 0.0};
    EXPECT_NO_THROW(validate_finite(data));
}

TEST(InputValidation, FiniteNaN) {
    std::vector<Real> data = {1.0, std::numeric_limits<Real>::quiet_NaN()};
    EXPECT_THROW(validate_finite(data), NumericalException);
}

TEST(InputValidation, FiniteInf) {
    std::vector<Real> data = {1.0, std::numeric_limits<Real>::infinity()};
    EXPECT_THROW(validate_finite(data), NumericalException);
}

TEST(InputValidation, PositiveExponents) {
    std::vector<Real> exp = {1.0, 2.0, 0.5};
    EXPECT_NO_THROW(validate_positive_exponents(exp));
}

TEST(InputValidation, NonPositiveExponents) {
    std::vector<Real> exp = {1.0, -0.1};
    EXPECT_THROW(validate_positive_exponents(exp), InvalidArgumentException);
}

TEST(InputValidation, ScreeningThresholdValid) {
    EXPECT_NO_THROW(validate_screening_threshold(1e-10));
    EXPECT_NO_THROW(validate_screening_threshold(0.0));
    EXPECT_NO_THROW(validate_screening_threshold(1.0));
}

TEST(InputValidation, ScreeningThresholdNegative) {
    EXPECT_THROW(validate_screening_threshold(-1e-10), InvalidArgumentException);
}

TEST(InputValidation, ScreeningThresholdTooLarge) {
    EXPECT_THROW(validate_screening_threshold(10.0), InvalidArgumentException);
}

TEST(InputValidation, ScreeningThresholdNaN) {
    EXPECT_THROW(validate_screening_threshold(
        std::numeric_limits<Real>::quiet_NaN()), InvalidArgumentException);
}

// ============================================================================
// Operator Validation Tests
// ============================================================================

TEST(InputValidation, NuclearDataValid) {
    std::vector<Real> charges = {1.0, 6.0};
    std::vector<Point3D> positions = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    EXPECT_NO_THROW(validate_nuclear_data(charges, positions));
}

TEST(InputValidation, NuclearDataSizeMismatch) {
    std::vector<Real> charges = {1.0, 6.0};
    std::vector<Point3D> positions = {{0.0, 0.0, 0.0}};
    EXPECT_THROW(validate_nuclear_data(charges, positions), InvalidArgumentException);
}

TEST(InputValidation, NuclearDataEmpty) {
    std::vector<Real> charges;
    std::vector<Point3D> positions;
    EXPECT_THROW(validate_nuclear_data(charges, positions), InvalidArgumentException);
}

TEST(InputValidation, OmegaValid) {
    EXPECT_NO_THROW(validate_omega(0.33));
    EXPECT_NO_THROW(validate_omega(1.0));
}

TEST(InputValidation, OmegaNonPositive) {
    EXPECT_THROW(validate_omega(0.0), InvalidArgumentException);
    EXPECT_THROW(validate_omega(-1.0), InvalidArgumentException);
}

TEST(InputValidation, OmegaNaN) {
    EXPECT_THROW(validate_omega(std::numeric_limits<Real>::quiet_NaN()),
                 InvalidArgumentException);
}
