// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file contraction_strategy.hpp
/// @brief Contraction strategy selection for integral kernel inner loops
///
/// Different contraction degrees (K = number of primitives per shell) benefit
/// from different strategies:
///
/// - **Register**: K <= 4. All primitive products fit in registers. Full loop
///   unrolling with no memory intermediates. Best for STO-3G and small bases.
///
/// - **Cache**: 5 <= K <= 16. Primitive products are computed in blocks that
///   fit in L1 cache. Intermediate results are accumulated in a small buffer.
///   Good for split-valence and correlation-consistent basis sets.
///
/// - **Streaming**: K > 16. Large contraction degrees (e.g., ANO basis sets)
///   stream through primitives with explicit prefetching. Intermediate sums
///   are accumulated in a buffer that may exceed L1 but fits in L2.

#include <libaccint/core/types.hpp>

namespace libaccint::kernels {

/// @brief Strategy for the contraction loop in integral kernels
enum class ContractionStrategy : int {
    Register = 0,   ///< Small K (<=4): register-based accumulation
    Cache = 1,      ///< Medium K (5-16): cache-friendly blocked accumulation
    Streaming = 2,  ///< Large K (>16): streaming with prefetch
};

/// @brief Select the optimal contraction strategy based on primitive count
///
/// For one-electron integrals, K is the product of bra and ket primitive counts.
/// For two-electron integrals, K is the product of all four primitive counts.
///
/// @param K_a Number of primitives in bra shell (or first shell)
/// @param K_b Number of primitives in ket shell (or second shell)
/// @return Optimal contraction strategy for the given primitive counts
[[nodiscard]] constexpr ContractionStrategy select_contraction_strategy(
    int K_a, int K_b) noexcept {
    const int K_product = K_a * K_b;
    if (K_product <= 16) {
        return ContractionStrategy::Register;
    } else if (K_product <= 256) {
        return ContractionStrategy::Cache;
    } else {
        return ContractionStrategy::Streaming;
    }
}

/// @brief Select contraction strategy for two-electron integrals
///
/// @param K_a Primitives in shell A
/// @param K_b Primitives in shell B
/// @param K_c Primitives in shell C
/// @param K_d Primitives in shell D
/// @return Optimal contraction strategy
[[nodiscard]] constexpr ContractionStrategy select_contraction_strategy_2e(
    int K_a, int K_b, int K_c, int K_d) noexcept {
    const int K_bra = K_a * K_b;
    const int K_ket = K_c * K_d;
    const int K_total = K_bra * K_ket;
    if (K_total <= 16) {
        return ContractionStrategy::Register;
    } else if (K_total <= 256) {
        return ContractionStrategy::Cache;
    } else {
        return ContractionStrategy::Streaming;
    }
}

/// @brief Convert ContractionStrategy to string for diagnostics
[[nodiscard]] constexpr const char* to_string(ContractionStrategy s) noexcept {
    switch (s) {
        case ContractionStrategy::Register:  return "Register";
        case ContractionStrategy::Cache:     return "Cache";
        case ContractionStrategy::Streaming: return "Streaming";
    }
    return "Unknown";
}

}  // namespace libaccint::kernels
