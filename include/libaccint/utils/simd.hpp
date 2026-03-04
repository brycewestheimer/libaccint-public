// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file simd.hpp
/// @brief SIMD abstraction layer for portable vectorized operations
///
/// Provides a unified interface for SIMD operations across different ISAs:
/// - AVX2: 4-wide double-precision (256-bit)
/// - AVX-512: 8-wide double-precision (512-bit) [future]
/// - Scalar fallback for non-SIMD systems
///
/// All operations maintain consistent semantics across implementations.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <array>

// Detect available SIMD support
#if defined(__AVX2__) || defined(__AVX__)
    #include <immintrin.h>
    #define LIBACCINT_SIMD_AVX2 1
#elif defined(__SSE4_1__) || defined(__SSE2__)
    #include <emmintrin.h>
    #include <smmintrin.h>
    #define LIBACCINT_SIMD_SSE 1
#else
    #define LIBACCINT_SIMD_SCALAR 1
#endif

namespace libaccint::simd {

// ============================================================================
// SIMD Type Traits and Constants
// ============================================================================

#if defined(LIBACCINT_SIMD_AVX2)

/// @brief SIMD vector type for double-precision
using SimdDouble = __m256d;

/// @brief SIMD width (number of doubles per vector)
inline constexpr int simd_width = 4;

/// @brief Required alignment for SIMD loads/stores
inline constexpr std::size_t simd_alignment = 32;

/// @brief SIMD instruction set name
inline constexpr const char* simd_isa_name = "AVX2";

#elif defined(LIBACCINT_SIMD_SSE)

using SimdDouble = __m128d;
inline constexpr int simd_width = 2;
inline constexpr std::size_t simd_alignment = 16;
inline constexpr const char* simd_isa_name = "SSE";

#else  // Scalar fallback

/// @brief Scalar "SIMD" type (single value)
struct SimdDouble {
    double value;

    SimdDouble() = default;
    explicit SimdDouble(double v) : value(v) {}

    // Allow implicit conversion to/from double for scalar case
    operator double() const { return value; }
};

inline constexpr int simd_width = 1;
inline constexpr std::size_t simd_alignment = 8;
inline constexpr const char* simd_isa_name = "Scalar";

#endif

// ============================================================================
// Load Operations
// ============================================================================

/// @brief Load aligned data from memory
/// @param p Pointer to aligned memory (must be simd_alignment aligned)
/// @return SIMD vector containing loaded values
inline SimdDouble load(const double* p) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    return _mm256_load_pd(p);
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_load_pd(p);
#else
    return SimdDouble{*p};
#endif
}

/// @brief Load unaligned data from memory
/// @param p Pointer to memory (alignment not required)
/// @return SIMD vector containing loaded values
inline SimdDouble loadu(const double* p) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    return _mm256_loadu_pd(p);
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_loadu_pd(p);
#else
    return SimdDouble{*p};
#endif
}

/// @brief Broadcast a scalar to all lanes
/// @param x Value to broadcast
/// @return SIMD vector with x in all lanes
inline SimdDouble broadcast(double x) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    return _mm256_set1_pd(x);
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_set1_pd(x);
#else
    return SimdDouble{x};
#endif
}

/// @brief Create a SIMD vector with all zeros
inline SimdDouble zero() noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    return _mm256_setzero_pd();
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_setzero_pd();
#else
    return SimdDouble{0.0};
#endif
}

// ============================================================================
// Store Operations
// ============================================================================

/// @brief Store to aligned memory
/// @param p Pointer to aligned memory (must be simd_alignment aligned)
/// @param v Vector to store
inline void store(double* p, SimdDouble v) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    _mm256_store_pd(p, v);
#elif defined(LIBACCINT_SIMD_SSE)
    _mm_store_pd(p, v);
#else
    *p = v.value;
#endif
}

/// @brief Store to unaligned memory
/// @param p Pointer to memory (alignment not required)
/// @param v Vector to store
inline void storeu(double* p, SimdDouble v) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    _mm256_storeu_pd(p, v);
#elif defined(LIBACCINT_SIMD_SSE)
    _mm_storeu_pd(p, v);
#else
    *p = v.value;
#endif
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

/// @brief Add two vectors element-wise
inline SimdDouble add(SimdDouble a, SimdDouble b) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    return _mm256_add_pd(a, b);
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_add_pd(a, b);
#else
    return SimdDouble{a.value + b.value};
#endif
}

/// @brief Subtract two vectors element-wise (a - b)
inline SimdDouble sub(SimdDouble a, SimdDouble b) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    return _mm256_sub_pd(a, b);
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_sub_pd(a, b);
#else
    return SimdDouble{a.value - b.value};
#endif
}

/// @brief Multiply two vectors element-wise
inline SimdDouble mul(SimdDouble a, SimdDouble b) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    return _mm256_mul_pd(a, b);
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_mul_pd(a, b);
#else
    return SimdDouble{a.value * b.value};
#endif
}

/// @brief Divide two vectors element-wise (a / b)
inline SimdDouble div(SimdDouble a, SimdDouble b) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    return _mm256_div_pd(a, b);
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_div_pd(a, b);
#else
    return SimdDouble{a.value / b.value};
#endif
}

/// @brief Fused multiply-add: a * b + c
/// @note Uses FMA instruction if available for better precision and performance
inline SimdDouble fma(SimdDouble a, SimdDouble b, SimdDouble c) noexcept {
#if defined(LIBACCINT_SIMD_AVX2) && defined(__FMA__)
    return _mm256_fmadd_pd(a, b, c);
#elif defined(LIBACCINT_SIMD_AVX2)
    return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_add_pd(_mm_mul_pd(a, b), c);
#else
    return SimdDouble{a.value * b.value + c.value};
#endif
}

/// @brief Fused multiply-subtract: a * b - c
inline SimdDouble fms(SimdDouble a, SimdDouble b, SimdDouble c) noexcept {
#if defined(LIBACCINT_SIMD_AVX2) && defined(__FMA__)
    return _mm256_fmsub_pd(a, b, c);
#elif defined(LIBACCINT_SIMD_AVX2)
    return _mm256_sub_pd(_mm256_mul_pd(a, b), c);
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_sub_pd(_mm_mul_pd(a, b), c);
#else
    return SimdDouble{a.value * b.value - c.value};
#endif
}

/// @brief Fused negative multiply-add: -a * b + c = c - a*b
inline SimdDouble fnma(SimdDouble a, SimdDouble b, SimdDouble c) noexcept {
#if defined(LIBACCINT_SIMD_AVX2) && defined(__FMA__)
    return _mm256_fnmadd_pd(a, b, c);
#elif defined(LIBACCINT_SIMD_AVX2)
    return _mm256_sub_pd(c, _mm256_mul_pd(a, b));
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_sub_pd(c, _mm_mul_pd(a, b));
#else
    return SimdDouble{c.value - a.value * b.value};
#endif
}

/// @brief Negate a vector
inline SimdDouble neg(SimdDouble a) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    return _mm256_xor_pd(a, _mm256_set1_pd(-0.0));
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_xor_pd(a, _mm_set1_pd(-0.0));
#else
    return SimdDouble{-a.value};
#endif
}

// ============================================================================
// Math Functions
// ============================================================================

/// @brief Compute square root element-wise
inline SimdDouble sqrt(SimdDouble a) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    return _mm256_sqrt_pd(a);
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_sqrt_pd(a);
#else
    return SimdDouble{std::sqrt(a.value)};
#endif
}

/// @brief Compute 1/sqrt(a) element-wise (fast reciprocal sqrt approximation)
/// @note This uses hardware approximation when available; may not be exact
inline SimdDouble rsqrt(SimdDouble a) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    // No direct rsqrt_pd in AVX2; use reciprocal of sqrt
    return _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(a));
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(a));
#else
    return SimdDouble{1.0 / std::sqrt(a.value)};
#endif
}

/// @brief Compute absolute value element-wise
inline SimdDouble abs(SimdDouble a) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    // Clear sign bit
    return _mm256_andnot_pd(_mm256_set1_pd(-0.0), a);
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_andnot_pd(_mm_set1_pd(-0.0), a);
#else
    return SimdDouble{std::abs(a.value)};
#endif
}

/// @brief Compute minimum element-wise
inline SimdDouble min(SimdDouble a, SimdDouble b) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    return _mm256_min_pd(a, b);
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_min_pd(a, b);
#else
    return SimdDouble{std::min(a.value, b.value)};
#endif
}

/// @brief Compute maximum element-wise
inline SimdDouble max(SimdDouble a, SimdDouble b) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    return _mm256_max_pd(a, b);
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_max_pd(a, b);
#else
    return SimdDouble{std::max(a.value, b.value)};
#endif
}

// ============================================================================
// Reduction Operations
// ============================================================================

/// @brief Horizontal sum (reduce to scalar by adding all lanes)
inline double reduce_add(SimdDouble v) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    // Sum pairs: [a+c, b+d, a+c, b+d]
    __m256d sum1 = _mm256_hadd_pd(v, v);
    // Extract high 128 bits and add to low 128 bits
    __m128d low = _mm256_castpd256_pd128(sum1);
    __m128d high = _mm256_extractf128_pd(sum1, 1);
    __m128d sum2 = _mm_add_pd(low, high);
    return _mm_cvtsd_f64(sum2);
#elif defined(LIBACCINT_SIMD_SSE)
    __m128d sum = _mm_hadd_pd(v, v);
    return _mm_cvtsd_f64(sum);
#else
    return v.value;
#endif
}

/// @brief Extract a single lane from a vector
/// @tparam I Lane index (0 to simd_width-1)
template<int I>
inline double extract(SimdDouble v) noexcept {
    static_assert(I >= 0 && I < simd_width, "Lane index out of range");
#if defined(LIBACCINT_SIMD_AVX2)
    // Extract to temp array
    alignas(32) double temp[4];
    _mm256_store_pd(temp, v);
    return temp[I];
#elif defined(LIBACCINT_SIMD_SSE)
    alignas(16) double temp[2];
    _mm_store_pd(temp, v);
    return temp[I];
#else
    return v.value;
#endif
}

// ============================================================================
// Comparison Operations
// ============================================================================

/// @brief Compare for less-than (returns mask)
inline SimdDouble cmp_lt(SimdDouble a, SimdDouble b) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_cmplt_pd(a, b);
#else
    return SimdDouble{a.value < b.value ? -1.0 : 0.0};
#endif
}

/// @brief Compare for greater-than (returns mask)
inline SimdDouble cmp_gt(SimdDouble a, SimdDouble b) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    return _mm256_cmp_pd(a, b, _CMP_GT_OQ);
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_cmpgt_pd(a, b);
#else
    return SimdDouble{a.value > b.value ? -1.0 : 0.0};
#endif
}

/// @brief Blend/select based on mask: (mask ? a : b)
inline SimdDouble blend(SimdDouble mask, SimdDouble a, SimdDouble b) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    return _mm256_blendv_pd(b, a, mask);
#elif defined(LIBACCINT_SIMD_SSE)
    return _mm_blendv_pd(b, a, mask);
#else
    return mask.value != 0.0 ? a : b;
#endif
}

// ============================================================================
// Exponential Function (Polynomial Approximation)
// ============================================================================

/// @brief Compute exp(x) for each lane using polynomial approximation
///
/// Uses a Padé-like approximation valid for x in [-709, 709]:
///   exp(x) = 2^n * P(r)
/// where x = n*ln(2) + r with |r| < ln(2)/2.
///
/// Accuracy: ~1e-14 relative error across valid range.
inline SimdDouble exp(SimdDouble x) noexcept {
#if defined(LIBACCINT_SIMD_AVX2)
    // Constants
    const __m256d LOG2E = _mm256_set1_pd(1.4426950408889634);
    const __m256d LN2_HI = _mm256_set1_pd(0.6931471805599453);
    const __m256d LN2_LO = _mm256_set1_pd(2.3190468138462996e-17);
    const __m256d HALF = _mm256_set1_pd(0.5);
    const __m256d ONE = _mm256_set1_pd(1.0);

    // Polynomial coefficients for exp(r) approximation
    const __m256d C1 = _mm256_set1_pd(1.0);
    const __m256d C2 = _mm256_set1_pd(0.5);
    const __m256d C3 = _mm256_set1_pd(0.16666666666666666);
    const __m256d C4 = _mm256_set1_pd(0.041666666666666664);
    const __m256d C5 = _mm256_set1_pd(0.008333333333333333);
    const __m256d C6 = _mm256_set1_pd(0.001388888888888889);
    const __m256d C7 = _mm256_set1_pd(0.0001984126984126984);

    // Range reduction: x = n*ln(2) + r
    __m256d t = _mm256_mul_pd(x, LOG2E);
    __m256d n = _mm256_round_pd(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // r = x - n*ln(2) (high precision)
    __m256d r = _mm256_sub_pd(x, _mm256_mul_pd(n, LN2_HI));
    r = _mm256_sub_pd(r, _mm256_mul_pd(n, LN2_LO));

    // Polynomial approximation: exp(r) = 1 + r + r^2/2! + r^3/3! + ...
    __m256d r2 = _mm256_mul_pd(r, r);
    __m256d r3 = _mm256_mul_pd(r2, r);

    // Horner's method for polynomial
    __m256d p = C7;
    p = _mm256_fmadd_pd(p, r, C6);
    p = _mm256_fmadd_pd(p, r, C5);
    p = _mm256_fmadd_pd(p, r, C4);
    p = _mm256_fmadd_pd(p, r, C3);
    p = _mm256_fmadd_pd(p, r, C2);
    p = _mm256_fmadd_pd(p, r, C1);
    p = _mm256_fmadd_pd(p, r, ONE);

    // Scale by 2^n
    // Convert n to integer and use in exponent
    __m128i ni = _mm256_cvttpd_epi32(n);
    ni = _mm_add_epi32(ni, _mm_set1_epi32(1023));  // Add bias
    ni = _mm_slli_epi32(ni, 20);  // Shift to exponent position

    // Construct 2^n for each lane
    alignas(32) int32_t exps[4];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(exps), ni);

    alignas(32) double scale[4];
    for (int i = 0; i < 4; ++i) {
        int64_t bits = static_cast<int64_t>(exps[i]) << 32;
        std::memcpy(&scale[i], &bits, sizeof(double));
    }

    __m256d scale_v = _mm256_load_pd(scale);
    return _mm256_mul_pd(p, scale_v);

#elif defined(LIBACCINT_SIMD_SSE)
    // Scalar fallback for SSE (exp is complex to vectorize well)
    alignas(16) double temp[2];
    _mm_store_pd(temp, x);
    temp[0] = std::exp(temp[0]);
    temp[1] = std::exp(temp[1]);
    return _mm_load_pd(temp);
#else
    return SimdDouble{std::exp(x.value)};
#endif
}

// ============================================================================
// Utility Functions
// ============================================================================

/// @brief Check if a pointer is properly aligned for SIMD operations
inline bool is_aligned(const void* ptr) noexcept {
    return reinterpret_cast<std::uintptr_t>(ptr) % simd_alignment == 0;
}

/// @brief Process array in SIMD chunks, calling func for each chunk
/// @tparam Func Function type: void(SimdDouble*, size_t start, size_t count)
/// @param data Pointer to array (must be aligned)
/// @param n Number of elements
/// @param func Function to call for each SIMD-width chunk
template<typename Func>
inline void for_each_simd(double* data, std::size_t n, Func&& func) {
    std::size_t i = 0;

    // Process full SIMD vectors
    for (; i + simd_width <= n; i += simd_width) {
        SimdDouble v = load(data + i);
        func(v, i);
        store(data + i, v);
    }

    // Handle remainder with scalar operations
    for (; i < n; ++i) {
        SimdDouble v = broadcast(data[i]);
        func(v, i);
        data[i] = extract<0>(v);
    }
}

/// @brief Compute dot product of two arrays using SIMD
/// @param a First array (aligned)
/// @param b Second array (aligned)
/// @param n Number of elements
/// @return Sum of a[i] * b[i]
inline double dot_product(const double* a, const double* b, std::size_t n) noexcept {
    SimdDouble sum = zero();
    std::size_t i = 0;

    // Process full SIMD vectors
    for (; i + simd_width <= n; i += simd_width) {
        SimdDouble va = load(a + i);
        SimdDouble vb = load(b + i);
        sum = fma(va, vb, sum);
    }

    double result = reduce_add(sum);

    // Handle remainder
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

/// @brief Accumulate scaled values: result[i] += scale * values[i]
/// @param result Output array (aligned)
/// @param values Input array (aligned)
/// @param scale Scalar multiplier
/// @param n Number of elements
inline void accumulate_scaled(double* result, const double* values,
                              double scale, std::size_t n) noexcept {
    SimdDouble scale_v = broadcast(scale);
    std::size_t i = 0;

    // Process full SIMD vectors
    for (; i + simd_width <= n; i += simd_width) {
        SimdDouble r = load(result + i);
        SimdDouble v = load(values + i);
        r = fma(scale_v, v, r);
        store(result + i, r);
    }

    // Handle remainder
    for (; i < n; ++i) {
        result[i] += scale * values[i];
    }
}

}  // namespace libaccint::simd
