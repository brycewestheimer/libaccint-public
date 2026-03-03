// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

#include <libaccint/config/max_am.hpp>
#include <libaccint/kernels/contraction_range.hpp>

#include <span>

/// @file generated_kernel_registry.hpp
/// @brief Registry for AM-specialized generated CPU kernels
///
/// Provides function pointer dispatch tables for generated kernels.
/// Maps (integral_type, la, lb) to the corresponding code-generated
/// kernel function. When a generated kernel is available for a given
/// AM pair, the dispatch system can use it instead of the generic
/// runtime-recursive implementation.
///
/// To regenerate kernels: libaccint-codegen --max-am <N> --backends cpu --output src/generated

namespace libaccint::kernels::cpu::generated {

// Import ContractionRange from the canonical definition in libaccint::kernels
using kernels::ContractionRange;

/// @brief Maximum angular momentum supported by generated 1e kernels
/// Derived from LIBACCINT_MAX_AM (set by CMake or defaulting to 4).
/// Currently generated kernels cover AM 0–4 (S through G).
constexpr int GENERATED_TABLE_MAX_AM = 4;
constexpr int GENERATED_MAX_AM =
    (LIBACCINT_MAX_AM < GENERATED_TABLE_MAX_AM) ? LIBACCINT_MAX_AM : GENERATED_TABLE_MAX_AM;

/// @brief Maximum angular momentum supported by generated ERI kernels
/// Must match the max AM of generated ERI kernel files.
/// Currently generated kernels cover AM 0–4 (S through G).
constexpr int GENERATED_ERI_MAX_AM =
    (LIBACCINT_MAX_AM < GENERATED_TABLE_MAX_AM) ? LIBACCINT_MAX_AM : GENERATED_TABLE_MAX_AM;

// =====================================================================
// Legacy raw-pointer function pointer types (backward compatibility)
// =====================================================================

/// @brief Function pointer type for generated 1e integral kernels (legacy)
///
/// Parameters match the raw-pointer interface used by generated code:
///   exponents_a, coefficients_a, center_a, n_prim_a,
///   exponents_b, coefficients_b, center_b, n_prim_b,
///   output
using OneElectronKernelFn = void(*)(
    const double*, const double*, const double*, int,
    const double*, const double*, const double*, int,
    double*);

/// @brief Function pointer type for generated nuclear attraction integral kernels (legacy)
///
/// Parameters match the raw-pointer interface used by generated code:
///   exponents_a, coefficients_a, center_a, n_prim_a,
///   exponents_b, coefficients_b, center_b, n_prim_b,
///   charges, charge_positions, n_charges,
///   output
using NuclearKernelFn = void(*)(
    const double*, const double*, const double*, int,
    const double*, const double*, const double*, int,
    const double*, const double*, int,
    double*);

/// @brief Function pointer type for generated 2e ERI kernels (legacy)
///
/// Parameters match the raw-pointer interface used by generated code:
///   exponents_a, coefficients_a, center_a, n_prim_a,
///   exponents_b, coefficients_b, center_b, n_prim_b,
///   exponents_c, coefficients_c, center_c, n_prim_c,
///   exponents_d, coefficients_d, center_d, n_prim_d,
///   output
using TwoElectronKernelFn = void(*)(
    const double*, const double*, const double*, int,
    const double*, const double*, const double*, int,
    const double*, const double*, const double*, int,
    const double*, const double*, const double*, int,
    double*);

// =====================================================================
// Modern std::span function pointer types (C++20)
// =====================================================================

/// @brief Function pointer type for 1e integral kernels using std::span
///
/// Parameters: exponents_a, coefficients_a,
///   center_a_x, center_a_y, center_a_z, n_prim_a,
///   exponents_b, coefficients_b,
///   center_b_x, center_b_y, center_b_z, n_prim_b,
///   output
using OneElectronKernelSpanFn = void(*)(
    std::span<const double>, std::span<const double>,
    double, double, double, int,
    std::span<const double>, std::span<const double>,
    double, double, double, int,
    std::span<double>);

/// @brief Function pointer type for nuclear attraction kernels using std::span
///
/// Parameters: exponents_a, coefficients_a,
///   center_a_x, center_a_y, center_a_z, n_prim_a,
///   exponents_b, coefficients_b,
///   center_b_x, center_b_y, center_b_z, n_prim_b,
///   charges, charge_positions, n_charges,
///   output
using NuclearKernelSpanFn = void(*)(
    std::span<const double>, std::span<const double>,
    double, double, double, int,
    std::span<const double>, std::span<const double>,
    double, double, double, int,
    std::span<const double>, std::span<const double>, int,
    std::span<double>);

/// @brief Function pointer type for 2e ERI kernels using std::span
///
/// Parameters: exponents_a, coefficients_a,
///   center_a_x, center_a_y, center_a_z, n_prim_a,
///   ... same for B, C, D ...
///   output
using TwoElectronKernelSpanFn = void(*)(
    std::span<const double>, std::span<const double>,
    double, double, double, int,
    std::span<const double>, std::span<const double>,
    double, double, double, int,
    std::span<const double>, std::span<const double>,
    double, double, double, int,
    std::span<const double>, std::span<const double>,
    double, double, double, int,
    std::span<double>);

// =====================================================================
// Availability queries
// =====================================================================

/// @brief Check if a generated overlap kernel is available for a given AM pair
[[nodiscard]] constexpr bool has_generated_overlap(int la, int lb) noexcept {
    return la >= 0 && la <= GENERATED_MAX_AM &&
           lb >= 0 && lb <= GENERATED_MAX_AM;
}

/// @brief Check if a generated kinetic kernel is available for a given AM pair
[[nodiscard]] constexpr bool has_generated_kinetic(int la, int lb) noexcept {
    return la >= 0 && la <= GENERATED_MAX_AM &&
           lb >= 0 && lb <= GENERATED_MAX_AM;
}

/// @brief Check if a generated nuclear kernel is available for a given AM pair
[[nodiscard]] constexpr bool has_generated_nuclear(int la, int lb) noexcept {
    return la >= 0 && la <= GENERATED_MAX_AM &&
           lb >= 0 && lb <= GENERATED_MAX_AM;
}

/// @brief Check if a generated ERI kernel is available for a given AM quartet
[[nodiscard]] constexpr bool has_generated_eri(int la, int lb, int lc, int ld) noexcept {
    return la >= 0 && la <= GENERATED_ERI_MAX_AM &&
           lb >= 0 && lb <= GENERATED_ERI_MAX_AM &&
           lc >= 0 && lc <= GENERATED_ERI_MAX_AM &&
           ld >= 0 && ld <= GENERATED_ERI_MAX_AM;
}

// =====================================================================
// Legacy dispatch (raw-pointer interface, backward compatibility)
// =====================================================================

/// @brief Get the generated overlap kernel function for a given AM pair
///
/// @param la Angular momentum of bra shell (0-4)
/// @param lb Angular momentum of ket shell (0-4)
/// @return Function pointer, or nullptr if no generated kernel exists
[[nodiscard]] OneElectronKernelFn get_generated_overlap(int la, int lb) noexcept;

/// @brief Get the generated kinetic kernel function for a given AM pair
///
/// @param la Angular momentum of bra shell (0-4)
/// @param lb Angular momentum of ket shell (0-4)
/// @return Function pointer, or nullptr if no generated kernel exists
[[nodiscard]] OneElectronKernelFn get_generated_kinetic(int la, int lb) noexcept;

/// @brief Get the generated nuclear kernel function for a given AM pair
///
/// @param la Angular momentum of bra shell (0-4)
/// @param lb Angular momentum of ket shell (0-4)
/// @return Function pointer, or nullptr if no generated kernel exists
[[nodiscard]] NuclearKernelFn get_generated_nuclear(int la, int lb) noexcept;

/// @brief Get the generated ERI kernel function for a given AM quartet
///
/// @param la Angular momentum of shell A (0-4)
/// @param lb Angular momentum of shell B (0-4)
/// @param lc Angular momentum of shell C (0-4)
/// @param ld Angular momentum of shell D (0-4)
/// @return Function pointer, or nullptr if no generated kernel exists
[[nodiscard]] TwoElectronKernelFn get_generated_eri(int la, int lb, int lc, int ld) noexcept;

// =====================================================================
// K-range-aware dispatch (modern std::span interface)
// =====================================================================

/// @brief Get the generated overlap kernel (std::span) for a given AM pair and K-range
///
/// @param la Angular momentum of bra shell (0-4)
/// @param lb Angular momentum of ket shell (0-4)
/// @param cr Contraction range (SmallK, MediumK, LargeK)
/// @return Function pointer, or nullptr if no generated kernel exists
[[nodiscard]] OneElectronKernelSpanFn get_generated_overlap(
    int la, int lb, ContractionRange cr) noexcept;

/// @brief Get the generated kinetic kernel (std::span) for a given AM pair and K-range
///
/// @param la Angular momentum of bra shell (0-4)
/// @param lb Angular momentum of ket shell (0-4)
/// @param cr Contraction range (SmallK, MediumK, LargeK)
/// @return Function pointer, or nullptr if no generated kernel exists
[[nodiscard]] OneElectronKernelSpanFn get_generated_kinetic(
    int la, int lb, ContractionRange cr) noexcept;

/// @brief Get the generated nuclear kernel (std::span) for a given AM pair and K-range
///
/// @param la Angular momentum of bra shell (0-4)
/// @param lb Angular momentum of ket shell (0-4)
/// @param cr Contraction range (SmallK, MediumK, LargeK)
/// @return Function pointer, or nullptr if no generated kernel exists
[[nodiscard]] NuclearKernelSpanFn get_generated_nuclear(
    int la, int lb, ContractionRange cr) noexcept;

/// @brief Get the generated ERI kernel (std::span) for a given AM quartet and K-range
///
/// @param la Angular momentum of shell A (0-4)
/// @param lb Angular momentum of shell B (0-4)
/// @param lc Angular momentum of shell C (0-4)
/// @param ld Angular momentum of shell D (0-4)
/// @param cr Contraction range (SmallK, MediumK, LargeK)
/// @return Function pointer, or nullptr if no generated kernel exists
[[nodiscard]] TwoElectronKernelSpanFn get_generated_eri(
    int la, int lb, int lc, int ld, ContractionRange cr) noexcept;

}  // namespace libaccint::kernels::cpu::generated
