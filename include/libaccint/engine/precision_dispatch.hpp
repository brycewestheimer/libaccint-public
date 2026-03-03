// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file precision_dispatch.hpp
/// @brief Precision-aware dispatch utilities for template kernel dispatch
///
/// Provides infrastructure for dispatching integral computations to the
/// appropriate precision type (float or double). Supports runtime precision
/// selection via the Precision enum and compile-time dispatch via templates.
///
/// Key components:
/// - PrecisionConfig: Runtime configuration for precision selection
/// - dispatch_on_precision(): Runtime-to-compile-time dispatch bridge
/// - compute_overlap_typed(): Precision-templated kernel wrappers
///
/// Usage:
/// @code
///   PrecisionConfig config;
///   config.compute_precision = Precision::Float32;
///   config.accumulate_precision = Precision::Float64;
///   config.mode = MixedPrecisionMode::Compute32Accumulate64;
///
///   // Dispatch based on runtime precision
///   dispatch_on_precision(config.compute_precision, [&](auto tag) {
///       using T = typename decltype(tag)::type;
///       // T is float or double
///   });
/// @endcode

#include <libaccint/core/precision.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/kernels/overlap_kernel.hpp>
#include <libaccint/kernels/kinetic_kernel.hpp>
#include <libaccint/kernels/nuclear_kernel.hpp>
#include <libaccint/kernels/eri_kernel.hpp>
#include <libaccint/operators/operator_types.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <functional>
#include <type_traits>

namespace libaccint::engine {

// ============================================================================
// Precision Configuration
// ============================================================================

/// @brief Runtime precision configuration for integral computation
struct PrecisionConfig {
    /// Precision for computing integrals
    Precision compute_precision = Precision::Float64;

    /// Precision for accumulating results (Fock matrix, etc.)
    Precision accumulate_precision = Precision::Float64;

    /// Mixed precision mode
    MixedPrecisionMode mode = MixedPrecisionMode::Pure64;

    /// Whether to enable adaptive precision selection based on AM
    bool adaptive_am = false;

    /// Angular momentum threshold above which to use double precision
    /// (only used when mode == Adaptive)
    int am_threshold_for_double = 3;

    /// @brief Create a pure float64 config (default)
    [[nodiscard]] static PrecisionConfig pure_double() noexcept {
        return PrecisionConfig{};
    }

    /// @brief Create a pure float32 config
    [[nodiscard]] static PrecisionConfig pure_float() noexcept {
        PrecisionConfig cfg;
        cfg.compute_precision = Precision::Float32;
        cfg.accumulate_precision = Precision::Float32;
        cfg.mode = MixedPrecisionMode::Pure32;
        return cfg;
    }

    /// @brief Create a mixed precision config (compute float32, accumulate float64)
    [[nodiscard]] static PrecisionConfig mixed() noexcept {
        PrecisionConfig cfg;
        cfg.compute_precision = Precision::Float32;
        cfg.accumulate_precision = Precision::Float64;
        cfg.mode = MixedPrecisionMode::Compute32Accumulate64;
        return cfg;
    }

    /// @brief Create an adaptive precision config
    [[nodiscard]] static PrecisionConfig adaptive(int am_threshold = 3) noexcept {
        PrecisionConfig cfg;
        cfg.compute_precision = Precision::Auto;
        cfg.accumulate_precision = Precision::Float64;
        cfg.mode = MixedPrecisionMode::Adaptive;
        cfg.adaptive_am = true;
        cfg.am_threshold_for_double = am_threshold;
        return cfg;
    }
};

// ============================================================================
// Type Tag for Compile-Time Dispatch
// ============================================================================

/// @brief Type tag for precision dispatch
template<typename T>
struct PrecisionTag {
    using type = T;
};

// ============================================================================
// Runtime-to-Compile-Time Dispatch
// ============================================================================

/// @brief Dispatch a callable based on runtime precision selection
///
/// Converts runtime Precision enum to compile-time type dispatch.
/// The callable receives a PrecisionTag<T> argument where T is float or double.
///
/// @tparam F Callable type accepting PrecisionTag<T>
/// @param precision Runtime precision selection
/// @param f Callable to invoke with the appropriate type tag
/// @return Result of invoking f
template<typename F>
decltype(auto) dispatch_on_precision(Precision precision, F&& f) {
    switch (precision) {
        case Precision::Float32:
            return std::forward<F>(f)(PrecisionTag<float>{});
        case Precision::Float64:
        case Precision::Auto:
        default:
            return std::forward<F>(f)(PrecisionTag<double>{});
    }
}

// ============================================================================
// Precision-Templated Kernel Wrappers
// ============================================================================

/// @brief Compute overlap integrals with precision dispatch
///
/// For float: computes in double precision and converts to float output.
/// For double: directly calls the double-precision kernel.
///
/// @tparam RealType Output precision type (float or double)
/// @param shell_a First shell
/// @param shell_b Second shell
/// @param buffer Output buffer in the target precision
template<typename RealType>
    requires ValidPrecision<RealType>
inline void compute_overlap_typed(const Shell& shell_a, const Shell& shell_b,
                                   OneElectronBuffer<0, RealType>& buffer) {
    if constexpr (std::is_same_v<RealType, double>) {
        kernels::compute_overlap(shell_a, shell_b, buffer);
    } else {
        // Compute in double precision, then convert
        OverlapBuffer double_buf;
        kernels::compute_overlap(shell_a, shell_b, double_buf);
        buffer.copy_from(double_buf);
    }
}

/// @brief Compute kinetic integrals with precision dispatch
template<typename RealType>
    requires ValidPrecision<RealType>
inline void compute_kinetic_typed(const Shell& shell_a, const Shell& shell_b,
                                   OneElectronBuffer<0, RealType>& buffer) {
    if constexpr (std::is_same_v<RealType, double>) {
        kernels::compute_kinetic(shell_a, shell_b, buffer);
    } else {
        KineticBuffer double_buf;
        kernels::compute_kinetic(shell_a, shell_b, double_buf);
        buffer.copy_from(double_buf);
    }
}

/// @brief Compute nuclear attraction integrals with precision dispatch
template<typename RealType>
    requires ValidPrecision<RealType>
inline void compute_nuclear_typed(const Shell& shell_a, const Shell& shell_b,
                                   const PointChargeParams& charges,
                                   OneElectronBuffer<0, RealType>& buffer) {
    if constexpr (std::is_same_v<RealType, double>) {
        kernels::compute_nuclear(shell_a, shell_b, charges, buffer);
    } else {
        NuclearBuffer double_buf;
        kernels::compute_nuclear(shell_a, shell_b, charges, double_buf);
        buffer.copy_from(double_buf);
    }
}

/// @brief Compute ERI integrals with precision dispatch
template<typename RealType>
    requires ValidPrecision<RealType>
inline void compute_eri_typed(const Shell& shell_a, const Shell& shell_b,
                               const Shell& shell_c, const Shell& shell_d,
                               TwoElectronBuffer<0, RealType>& buffer) {
    if constexpr (std::is_same_v<RealType, double>) {
        kernels::compute_eri(shell_a, shell_b, shell_c, shell_d, buffer);
    } else {
        TwoElectronBuffer<0, double> double_buf;
        kernels::compute_eri(shell_a, shell_b, shell_c, shell_d, double_buf);
        buffer.copy_from(double_buf);
    }
}

// ============================================================================
// Precision Selection Heuristics
// ============================================================================

/// @brief Determine optimal compute precision for a shell pair
///
/// For adaptive mode, uses angular momentum to decide precision:
/// - Low AM (s, p, d): float32 is sufficient
/// - High AM (f, g, h, i): use float64 for numerical stability
///
/// @param config Precision configuration
/// @param la Angular momentum of shell A
/// @param lb Angular momentum of shell B
/// @return Recommended compute precision
[[nodiscard]] inline Precision select_precision_1e(
    const PrecisionConfig& config, int la, int lb) noexcept {
    if (config.mode != MixedPrecisionMode::Adaptive) {
        return config.compute_precision;
    }
    // Use double for high AM to maintain accuracy
    if (la >= config.am_threshold_for_double ||
        lb >= config.am_threshold_for_double) {
        return Precision::Float64;
    }
    return Precision::Float32;
}

/// @brief Determine optimal compute precision for a shell quartet
///
/// @param config Precision configuration
/// @param la Angular momentum of shell A
/// @param lb Angular momentum of shell B
/// @param lc Angular momentum of shell C
/// @param ld Angular momentum of shell D
/// @return Recommended compute precision
[[nodiscard]] inline Precision select_precision_2e(
    const PrecisionConfig& config, int la, int lb, int lc, int ld) noexcept {
    if (config.mode != MixedPrecisionMode::Adaptive) {
        return config.compute_precision;
    }
    int max_am = std::max({la, lb, lc, ld});
    if (max_am >= config.am_threshold_for_double) {
        return Precision::Float64;
    }
    return Precision::Float32;
}

}  // namespace libaccint::engine
