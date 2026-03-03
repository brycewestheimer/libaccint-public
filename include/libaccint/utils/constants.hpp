// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file constants.hpp
/// @brief Physical and mathematical constants for LibAccInt
///
/// All physical constants are based on CODATA 2018 recommendations.
/// Mathematical constants are computed to full double precision.

namespace libaccint::constants {

// ============================================================================
// Mathematical Constants
// ============================================================================

/// @brief Pi (π) to full double precision
inline constexpr double PI = 3.14159265358979323846;

// Validate PI is within expected range
static_assert(PI > 3.14 && PI < 3.15, "PI constant out of expected range");

/// @brief Square root of pi (√π)
inline constexpr double SQRT_PI = 1.77245385090551602729;

// Validate SQRT_PI: √π * √π should equal π
static_assert(SQRT_PI * SQRT_PI > 3.14 && SQRT_PI * SQRT_PI < 3.15,
              "SQRT_PI squared should be approximately PI");

/// @brief Inverse square root of pi (1/√π)
inline constexpr double INV_SQRT_PI = 0.56418958354775628695;

// Validate INV_SQRT_PI: 1/√π * √π should be close to 1
static_assert(INV_SQRT_PI * SQRT_PI > 0.99 && INV_SQRT_PI * SQRT_PI < 1.01,
              "INV_SQRT_PI * SQRT_PI should be approximately 1");

/// @brief 2π (two pi)
inline constexpr double TWO_PI = 2.0 * PI;

static_assert(TWO_PI > 6.28 && TWO_PI < 6.29, "TWO_PI constant out of expected range");

// ============================================================================
// Physical Constants (CODATA 2018, SI units)
// ============================================================================

/// @brief Bohr radius (a₀) in meters
///
/// The Bohr radius is the characteristic length scale in atomic units.
/// CODATA 2018 value: 0.529177210903 × 10⁻¹⁰ m
inline constexpr double BOHR_RADIUS = 0.529177210903e-10;

static_assert(BOHR_RADIUS > 0.5e-10 && BOHR_RADIUS < 0.6e-10,
              "BOHR_RADIUS out of expected range");

/// @brief Hartree energy (Eₕ) in joules
///
/// The Hartree is the unit of energy in atomic units.
/// CODATA 2018 value: 4.3597447222071 × 10⁻¹⁸ J
inline constexpr double HARTREE_ENERGY = 4.3597447222071e-18;

static_assert(HARTREE_ENERGY > 4.3e-18 && HARTREE_ENERGY < 4.4e-18,
              "HARTREE_ENERGY out of expected range");

/// @brief Speed of light (c) in m/s (exact value by definition)
inline constexpr double SPEED_OF_LIGHT = 299792458.0;

static_assert(SPEED_OF_LIGHT > 2.99e8 && SPEED_OF_LIGHT < 3.00e8,
              "SPEED_OF_LIGHT out of expected range");

/// @brief Elementary charge (e) in coulombs (exact value by definition as of 2019)
inline constexpr double ELEMENTARY_CHARGE = 1.602176634e-19;

static_assert(ELEMENTARY_CHARGE > 1.6e-19 && ELEMENTARY_CHARGE < 1.7e-19,
              "ELEMENTARY_CHARGE out of expected range");

/// @brief Planck constant (h) in J⋅s (exact value by definition as of 2019)
inline constexpr double PLANCK_CONSTANT = 6.62607015e-34;

static_assert(PLANCK_CONSTANT > 6.6e-34 && PLANCK_CONSTANT < 6.7e-34,
              "PLANCK_CONSTANT out of expected range");

/// @brief Avogadro constant (Nₐ) in mol⁻¹ (exact value by definition as of 2019)
inline constexpr double AVOGADRO_CONSTANT = 6.02214076e23;

static_assert(AVOGADRO_CONSTANT > 6.0e23 && AVOGADRO_CONSTANT < 6.1e23,
              "AVOGADRO_CONSTANT out of expected range");

// ============================================================================
// Conversion Factors
// ============================================================================

/// @brief Bohr to Angstrom conversion (1 a₀ = ? Å)
inline constexpr double BOHR_TO_ANGSTROM = 0.529177210903;

static_assert(BOHR_TO_ANGSTROM > 0.529 && BOHR_TO_ANGSTROM < 0.530,
              "BOHR_TO_ANGSTROM out of expected range");

/// @brief Angstrom to Bohr conversion (1 Å = ? a₀)
inline constexpr double ANGSTROM_TO_BOHR = 1.8897261246257702;

static_assert(ANGSTROM_TO_BOHR > 1.88 && ANGSTROM_TO_BOHR < 1.90,
              "ANGSTROM_TO_BOHR out of expected range");

// Validate round-trip conversion
static_assert(BOHR_TO_ANGSTROM * ANGSTROM_TO_BOHR > 0.99 &&
              BOHR_TO_ANGSTROM * ANGSTROM_TO_BOHR < 1.01,
              "BOHR_TO_ANGSTROM * ANGSTROM_TO_BOHR should be approximately 1");

/// @brief Hartree to electron volt conversion (1 Eₕ = ? eV)
inline constexpr double HARTREE_TO_EV = 27.211386245988;

static_assert(HARTREE_TO_EV > 27.0 && HARTREE_TO_EV < 27.5,
              "HARTREE_TO_EV out of expected range");

/// @brief Electron volt to Hartree conversion (1 eV = ? Eₕ)
inline constexpr double EV_TO_HARTREE = 1.0 / HARTREE_TO_EV;

static_assert(EV_TO_HARTREE > 0.03 && EV_TO_HARTREE < 0.04,
              "EV_TO_HARTREE out of expected range");

// Validate round-trip conversion
static_assert(HARTREE_TO_EV * EV_TO_HARTREE > 0.99 &&
              HARTREE_TO_EV * EV_TO_HARTREE < 1.01,
              "HARTREE_TO_EV * EV_TO_HARTREE should be approximately 1");

/// @brief Hartree to kcal/mol conversion (1 Eₕ = ? kcal/mol)
inline constexpr double HARTREE_TO_KCAL_MOL = 627.509474063;

static_assert(HARTREE_TO_KCAL_MOL > 627.0 && HARTREE_TO_KCAL_MOL < 628.0,
              "HARTREE_TO_KCAL_MOL out of expected range");

/// @brief kcal/mol to Hartree conversion (1 kcal/mol = ? Eₕ)
inline constexpr double KCAL_MOL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL_MOL;

static_assert(KCAL_MOL_TO_HARTREE > 0.0015 && KCAL_MOL_TO_HARTREE < 0.0016,
              "KCAL_MOL_TO_HARTREE out of expected range");

// Validate round-trip conversion
static_assert(HARTREE_TO_KCAL_MOL * KCAL_MOL_TO_HARTREE > 0.99 &&
              HARTREE_TO_KCAL_MOL * KCAL_MOL_TO_HARTREE < 1.01,
              "HARTREE_TO_KCAL_MOL * KCAL_MOL_TO_HARTREE should be approximately 1");

/// @brief Hartree to kJ/mol conversion (1 Eₕ = ? kJ/mol)
inline constexpr double HARTREE_TO_KJ_MOL = 2625.499638;

static_assert(HARTREE_TO_KJ_MOL > 2625.0 && HARTREE_TO_KJ_MOL < 2626.0,
              "HARTREE_TO_KJ_MOL out of expected range");

// ============================================================================
// Numerical Thresholds
// ============================================================================

/// @brief Default integral threshold for two-electron integrals
///
/// Integrals smaller than this threshold are neglected.
inline constexpr double INTEGRAL_THRESHOLD = 1e-12;

static_assert(INTEGRAL_THRESHOLD > 0.0 && INTEGRAL_THRESHOLD < 1e-11,
              "INTEGRAL_THRESHOLD must be positive and small");

/// @brief Schwarz screening threshold for integral prescreening
///
/// Used in Schwarz inequality integral screening.
inline constexpr double SCHWARZ_THRESHOLD = 1e-10;

static_assert(SCHWARZ_THRESHOLD > 0.0 && SCHWARZ_THRESHOLD < 1e-9,
              "SCHWARZ_THRESHOLD must be positive and small");

/// @brief Precision threshold for Boys function evaluation
///
/// Controls the accuracy of Boys function polynomial expansions.
inline constexpr double BOYS_PRECISION = 1e-14;

static_assert(BOYS_PRECISION > 0.0 && BOYS_PRECISION < 1e-13,
              "BOYS_PRECISION must be positive and small");

// ============================================================================
// Atomic Masses (in atomic mass units)
// ============================================================================

/// @brief Proton mass in atomic mass units
inline constexpr double PROTON_MASS = 1.007276466621;

static_assert(PROTON_MASS > 1.0 && PROTON_MASS < 1.01,
              "PROTON_MASS out of expected range");

/// @brief Neutron mass in atomic mass units
inline constexpr double NEUTRON_MASS = 1.00866491595;

static_assert(NEUTRON_MASS > 1.0 && NEUTRON_MASS < 1.01,
              "NEUTRON_MASS out of expected range");

/// @brief Electron mass in atomic mass units
inline constexpr double ELECTRON_MASS = 5.48579909065e-4;

static_assert(ELECTRON_MASS > 5.0e-4 && ELECTRON_MASS < 6.0e-4,
              "ELECTRON_MASS out of expected range");

}  // namespace libaccint::constants
