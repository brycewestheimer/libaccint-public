// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file contraction_range.hpp
/// @brief Contraction-degree range enum for K-aware kernel dispatch
///
/// Classifies the contraction degree of a shell quartet so the dispatch
/// system can select different execution strategies for different workloads.
/// This is the single canonical definition used across CPU and GPU backends.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace libaccint::kernels {

/// Number of ContractionRange enum values (SmallK, MediumK, LargeK)
inline constexpr size_t kNumContractionRanges = 3;

/// @brief Contraction-degree range for K-aware dispatch
///
/// Classifies the contraction degree of a shell quartet so the dispatch
/// table can select different execution strategies for different workloads.
/// Mirrors GpuContractionRange from device_resource_tracker.hpp but is
/// defined independently so the dispatch table works without CUDA.
enum class ContractionRange : uint8_t {
    SmallK  = 0,  ///< K <= 3
    MediumK = 1,  ///< 4 <= K <= 6
    LargeK  = 2,  ///< K > 6
};

/// @brief Convert ContractionRange to string
inline constexpr std::string_view to_string(ContractionRange k) noexcept {
    switch (k) {
        case ContractionRange::SmallK:  return "small";
        case ContractionRange::MediumK: return "medium";
        case ContractionRange::LargeK:  return "large";
        default:                        return "unknown";
    }
}

/// @brief Parse a ContractionRange from a string
/// @throws std::invalid_argument if the string doesn't match
inline ContractionRange contraction_range_from_string(std::string_view s) {
    if (s == "small")  return ContractionRange::SmallK;
    if (s == "medium") return ContractionRange::MediumK;
    if (s == "large")  return ContractionRange::LargeK;
    throw std::invalid_argument("Unknown ContractionRange: " + std::string(s));
}

}  // namespace libaccint::kernels
