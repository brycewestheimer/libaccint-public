// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file precision.hpp
/// @brief Precision infrastructure for single/double precision support
///
/// This file provides the core precision infrastructure enabling LibAccInt
/// to operate in both single-precision (float32) and double-precision (float64)
/// modes. The key components are:
///
/// - `PrecisionTraits<T>`: Type traits for precision-specific constants
/// - `Constants<T>`: Mathematical constants in the appropriate precision
/// - Type aliases for precision-templated types
///
/// Usage:
/// @code
///   using Traits = PrecisionTraits<float>;
///   float threshold = Traits::integral_threshold;
///
///   float pi = Constants<float>::pi;
/// @endcode

#include <cstdint>
#include <limits>
#include <type_traits>
#include <cmath>

namespace libaccint {

// ============================================================================
// Precision Concept
// ============================================================================

/// @brief Concept for valid precision types (float or double)
template<typename T>
concept ValidPrecision = std::is_same_v<T, float> || std::is_same_v<T, double>;

// ============================================================================
// Primary Templates
// ============================================================================

/// @brief Precision traits template (specialized for float and double)
template<typename T>
struct PrecisionTraits;

/// @brief Mathematical constants template (specialized for float and double)
template<typename T>
struct Constants;

// ============================================================================
// Precision Traits - Float (float32)
// ============================================================================

/// @brief Precision traits for single-precision (float32) computations
template<>
struct PrecisionTraits<float> {
    /// The scalar type
    using Scalar = float;

    /// Number of bits in the representation
    static constexpr int bits = 32;

    /// Is this single precision?
    static constexpr bool is_single = true;

    /// Is this double precision?
    static constexpr bool is_double = false;

    /// Machine epsilon
    static constexpr float epsilon = std::numeric_limits<float>::epsilon();

    /// Minimum positive normal value
    static constexpr float min_positive = std::numeric_limits<float>::min();

    /// Maximum finite value
    static constexpr float max_value = std::numeric_limits<float>::max();

    /// Number of significant decimal digits
    static constexpr int digits10 = std::numeric_limits<float>::digits10;

    /// Number of significant binary digits
    static constexpr int digits = std::numeric_limits<float>::digits;

    // ========================================================================
    // Precision-Specific Thresholds
    // ========================================================================

    /// Threshold for negligible integrals
    static constexpr float integral_threshold = 1e-10f;

    /// Threshold for Schwarz screening
    static constexpr float screening_threshold = 1e-8f;

    /// Convergence threshold for iterative methods
    static constexpr float convergence_threshold = 1e-6f;

    /// Threshold for Boys function small-T regime
    static constexpr float boys_small_t = 1e-6f;

    /// Target relative accuracy for Boys function
    static constexpr float boys_accuracy = 1e-6f;

    /// Threshold for asymptotic Boys expansion
    static constexpr float boys_asymptotic_threshold = 25.0f;

    /// Maximum T for Chebyshev interpolation in Boys function
    static constexpr float boys_chebyshev_max = 30.0f;

    /// Threshold for detecting linearly dependent basis functions
    static constexpr float linear_dependency_threshold = 1e-6f;

    /// Threshold for negligible Gaussian overlap
    static constexpr float gaussian_overlap_threshold = 1e-8f;

    // ========================================================================
    // SIMD Configuration
    // ========================================================================

    /// SIMD width for AVX (256-bit / 32-bit = 8)
    static constexpr int simd_width_avx = 8;

    /// SIMD width for AVX-512 (512-bit / 32-bit = 16)
    static constexpr int simd_width_avx512 = 16;
};

// ============================================================================
// Precision Traits - Double (float64)
// ============================================================================

/// @brief Precision traits for double-precision (float64) computations
template<>
struct PrecisionTraits<double> {
    /// The scalar type
    using Scalar = double;

    /// Number of bits in the representation
    static constexpr int bits = 64;

    /// Is this single precision?
    static constexpr bool is_single = false;

    /// Is this double precision?
    static constexpr bool is_double = true;

    /// Machine epsilon
    static constexpr double epsilon = std::numeric_limits<double>::epsilon();

    /// Minimum positive normal value
    static constexpr double min_positive = std::numeric_limits<double>::min();

    /// Maximum finite value
    static constexpr double max_value = std::numeric_limits<double>::max();

    /// Number of significant decimal digits
    static constexpr int digits10 = std::numeric_limits<double>::digits10;

    /// Number of significant binary digits
    static constexpr int digits = std::numeric_limits<double>::digits;

    // ========================================================================
    // Precision-Specific Thresholds
    // ========================================================================

    /// Threshold for negligible integrals
    static constexpr double integral_threshold = 1e-14;

    /// Threshold for Schwarz screening
    static constexpr double screening_threshold = 1e-12;

    /// Convergence threshold for iterative methods
    static constexpr double convergence_threshold = 1e-10;

    /// Threshold for Boys function small-T regime
    static constexpr double boys_small_t = 1e-14;

    /// Target relative accuracy for Boys function
    static constexpr double boys_accuracy = 1e-14;

    /// Threshold for asymptotic Boys expansion
    static constexpr double boys_asymptotic_threshold = 30.0;

    /// Maximum T for Chebyshev interpolation in Boys function
    static constexpr double boys_chebyshev_max = 36.0;

    /// Threshold for detecting linearly dependent basis functions
    static constexpr double linear_dependency_threshold = 1e-10;

    /// Threshold for negligible Gaussian overlap
    static constexpr double gaussian_overlap_threshold = 1e-12;

    // ========================================================================
    // SIMD Configuration
    // ========================================================================

    /// SIMD width for AVX (256-bit / 64-bit = 4)
    static constexpr int simd_width_avx = 4;

    /// SIMD width for AVX-512 (512-bit / 64-bit = 8)
    static constexpr int simd_width_avx512 = 8;
};

// ============================================================================
// Mathematical Constants - Float Specialization
// ============================================================================

/// @brief Mathematical constants in single precision
template<>
struct Constants<float> {
    /// Pi (π)
    static constexpr float pi = 3.14159265f;

    /// Two times Pi (2π)
    static constexpr float two_pi = 6.28318531f;

    /// Pi squared (π²)
    static constexpr float pi_squared = 9.86960440f;

    /// Square root of Pi (√π)
    static constexpr float sqrt_pi = 1.77245385f;

    /// Pi to the 3/2 power (π^(3/2))
    static constexpr float pi_3_2 = 5.56832800f;

    /// One over Pi (1/π)
    static constexpr float one_over_pi = 0.31830989f;

    /// One over sqrt(Pi) (1/√π)
    static constexpr float one_over_sqrt_pi = 0.56418958f;

    /// Natural logarithm of 2 (ln 2)
    static constexpr float ln_2 = 0.69314718f;

    /// Euler's constant (e)
    static constexpr float e = 2.71828183f;

    /// Square root of 2 (√2)
    static constexpr float sqrt_2 = 1.41421356f;

    /// One over sqrt(2) (1/√2)
    static constexpr float one_over_sqrt_2 = 0.70710678f;

    /// Machine epsilon for float
    static constexpr float epsilon = PrecisionTraits<float>::epsilon;

    /// Minimum positive normal value
    static constexpr float min_positive = PrecisionTraits<float>::min_positive;

    /// Maximum finite value
    static constexpr float max_value = PrecisionTraits<float>::max_value;
};

// ============================================================================
// Mathematical Constants - Double Specialization
// ============================================================================

/// @brief Mathematical constants in double precision
template<>
struct Constants<double> {
    /// Pi (π)
    static constexpr double pi = 3.14159265358979323846264338327950288;

    /// Two times Pi (2π)
    static constexpr double two_pi = 6.28318530717958647692528676655900577;

    /// Pi squared (π²)
    static constexpr double pi_squared = 9.86960440108935861883449099987615114;

    /// Square root of Pi (√π)
    static constexpr double sqrt_pi = 1.77245385090551602729816748334114518;

    /// Pi to the 3/2 power (π^(3/2))
    static constexpr double pi_3_2 = 5.56832799683170784528481798212053514;

    /// One over Pi (1/π)
    static constexpr double one_over_pi = 0.31830988618379067153776752674502872;

    /// One over sqrt(Pi) (1/√π)
    static constexpr double one_over_sqrt_pi = 0.56418958354775628694807945156077259;

    /// Natural logarithm of 2 (ln 2)
    static constexpr double ln_2 = 0.69314718055994530941723212145817657;

    /// Euler's constant (e)
    static constexpr double e = 2.71828182845904523536028747135266250;

    /// Square root of 2 (√2)
    static constexpr double sqrt_2 = 1.41421356237309504880168872420969808;

    /// One over sqrt(2) (1/√2)
    static constexpr double one_over_sqrt_2 = 0.70710678118654752440084436210484904;

    /// Machine epsilon for double
    static constexpr double epsilon = PrecisionTraits<double>::epsilon;

    /// Minimum positive normal value
    static constexpr double min_positive = PrecisionTraits<double>::min_positive;

    /// Maximum finite value
    static constexpr double max_value = PrecisionTraits<double>::max_value;
};

// ============================================================================
// Precision Selection Enum
// ============================================================================

/// @brief Runtime precision selection
enum class Precision : std::uint8_t {
    Float32 = 0,  ///< Single precision (float)
    Float64 = 1,  ///< Double precision (double)
    Auto = 2      ///< Automatic selection based on context
};

/// @brief Convert Precision enum to string
[[nodiscard]] inline constexpr const char* precision_to_string(Precision p) noexcept {
    switch (p) {
        case Precision::Float32: return "float32";
        case Precision::Float64: return "float64";
        case Precision::Auto: return "auto";
        default: return "unknown";
    }
}

/// @brief Get size in bytes for a precision type
[[nodiscard]] inline constexpr std::size_t precision_size_bytes(Precision p) noexcept {
    switch (p) {
        case Precision::Float32: return sizeof(float);
        case Precision::Float64: return sizeof(double);
        default: return sizeof(double);  // Default to double
    }
}

// ============================================================================
// Mixed Precision Mode
// ============================================================================

/// @brief Mixed precision computation modes
enum class MixedPrecisionMode : std::uint8_t {
    /// Pure single precision: compute and accumulate in float32
    Pure32 = 0,

    /// Pure double precision: compute and accumulate in float64
    Pure64 = 1,

    /// Mixed: compute in float32, accumulate in float64
    Compute32Accumulate64 = 2,

    /// Adaptive: select precision based on AM and exponent ranges
    Adaptive = 3
};

/// @brief Convert MixedPrecisionMode enum to string
[[nodiscard]] inline constexpr const char* mixed_precision_to_string(MixedPrecisionMode mode) noexcept {
    switch (mode) {
        case MixedPrecisionMode::Pure32: return "pure32";
        case MixedPrecisionMode::Pure64: return "pure64";
        case MixedPrecisionMode::Compute32Accumulate64: return "compute32_accumulate64";
        case MixedPrecisionMode::Adaptive: return "adaptive";
        default: return "unknown";
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// @brief Check if a type is single precision
template<typename T>
[[nodiscard]] inline constexpr bool is_single_precision() noexcept {
    return PrecisionTraits<T>::is_single;
}

/// @brief Check if a type is double precision
template<typename T>
[[nodiscard]] inline constexpr bool is_double_precision() noexcept {
    return PrecisionTraits<T>::is_double;
}

/// @brief Get the appropriate integral threshold for a precision type
template<typename T>
    requires ValidPrecision<T>
[[nodiscard]] inline constexpr T integral_threshold() noexcept {
    return PrecisionTraits<T>::integral_threshold;
}

/// @brief Get the appropriate screening threshold for a precision type
template<typename T>
    requires ValidPrecision<T>
[[nodiscard]] inline constexpr T screening_threshold() noexcept {
    return PrecisionTraits<T>::screening_threshold;
}

/// @brief Convert a value between precisions with appropriate rounding
template<typename To, typename From>
    requires ValidPrecision<To> && ValidPrecision<From>
[[nodiscard]] inline constexpr To precision_cast(From value) noexcept {
    return static_cast<To>(value);
}

// ============================================================================
// Precision-Aware Comparison Utilities
// ============================================================================

/// @brief Compare floating point values with precision-appropriate tolerance
template<typename T>
    requires ValidPrecision<T>
[[nodiscard]] inline bool nearly_equal(T a, T b, T rel_tol = PrecisionTraits<T>::epsilon * T{100}) noexcept {
    if (std::isinf(a) || std::isinf(b)) {
        return a == b;
    }
    if (std::isnan(a) || std::isnan(b)) return false;
    T diff = std::abs(a - b);
    T max_val = std::max(std::abs(a), std::abs(b));
    return diff <= rel_tol * max_val || diff < PrecisionTraits<T>::min_positive;
}

/// @brief Check if a value is negligible for the given precision
template<typename T>
    requires ValidPrecision<T>
[[nodiscard]] inline bool is_negligible(T value, T threshold = PrecisionTraits<T>::integral_threshold) noexcept {
    return std::abs(value) < threshold;
}

}  // namespace libaccint
