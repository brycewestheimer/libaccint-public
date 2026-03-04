// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file kernel_variant.hpp
/// @brief Enumeration of kernel implementation variants for dispatch selection
///
/// Each integral type (overlap, kinetic, nuclear, ERI) can be computed by
/// multiple kernel implementations: hand-written template-specialized, code-generated
/// AM-specialized, fused, or cooperative. The OptimalDispatchTable maps each AM
/// combination to the best-performing variant.

#include <cstdint>
#include <string>
#include <string_view>
#include <stdexcept>

namespace libaccint::kernels {

/// @brief Identifies a specific kernel implementation variant
enum class KernelVariant : uint8_t {
    // One-electron individual kernels
    HandwrittenOverlap,   ///< Hand-written template-specialized overlap kernel
    HandwrittenKinetic,   ///< Hand-written template-specialized kinetic kernel
    HandwrittenNuclear,   ///< Hand-written template-specialized nuclear kernel
    GeneratedOverlap,     ///< Code-generated AM-specialized overlap kernel
    GeneratedKinetic,     ///< Code-generated AM-specialized kinetic kernel
    GeneratedNuclear,     ///< Code-generated AM-specialized nuclear kernel (shared memory prefetch)

    // Fused one-electron kernel
    HandwrittenFused1e,   ///< Fused S+T+V kernel (single launch, 3 outputs)

    // Two-electron kernels
    HandwrittenERI,       ///< Hand-written template-specialized ERI kernel
    GeneratedERI,         ///< Code-generated AM-specialized ERI kernel
    CooperativeERI,       ///< Block-per-quartet cooperative ERI kernel (high AM)
};

/// @brief Convert KernelVariant to a human-readable string
inline constexpr std::string_view to_string(KernelVariant v) noexcept {
    switch (v) {
        case KernelVariant::HandwrittenOverlap:  return "handwritten_overlap";
        case KernelVariant::HandwrittenKinetic:  return "handwritten_kinetic";
        case KernelVariant::HandwrittenNuclear:  return "handwritten_nuclear";
        case KernelVariant::GeneratedOverlap:    return "generated_overlap";
        case KernelVariant::GeneratedKinetic:    return "generated_kinetic";
        case KernelVariant::GeneratedNuclear:    return "generated_nuclear";
        case KernelVariant::HandwrittenFused1e:  return "handwritten_fused_1e";
        case KernelVariant::HandwrittenERI:      return "handwritten_eri";
        case KernelVariant::GeneratedERI:        return "generated_eri";
        case KernelVariant::CooperativeERI:      return "cooperative_eri";
        default:                                 return "unknown";
    }
}

/// @brief Parse a KernelVariant from a string
/// @throws std::invalid_argument if the string doesn't match any variant
inline KernelVariant kernel_variant_from_string(std::string_view s) {
    if (s == "handwritten_overlap")  return KernelVariant::HandwrittenOverlap;
    if (s == "handwritten_kinetic")  return KernelVariant::HandwrittenKinetic;
    if (s == "handwritten_nuclear")  return KernelVariant::HandwrittenNuclear;
    if (s == "generated_overlap")    return KernelVariant::GeneratedOverlap;
    if (s == "generated_kinetic")    return KernelVariant::GeneratedKinetic;
    if (s == "generated_nuclear")    return KernelVariant::GeneratedNuclear;
    if (s == "handwritten_fused_1e") return KernelVariant::HandwrittenFused1e;
    if (s == "handwritten_eri")      return KernelVariant::HandwrittenERI;
    if (s == "generated_eri")        return KernelVariant::GeneratedERI;
    if (s == "cooperative_eri")      return KernelVariant::CooperativeERI;
    // Also accept short-form names used in JSON
    if (s == "handwritten")          return KernelVariant::HandwrittenERI;
    if (s == "generated")            return KernelVariant::GeneratedERI;
    if (s == "cooperative")          return KernelVariant::CooperativeERI;
    throw std::invalid_argument("Unknown KernelVariant: " + std::string(s));
}

}  // namespace libaccint::kernels
