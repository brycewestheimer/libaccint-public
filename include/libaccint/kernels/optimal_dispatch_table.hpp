// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file optimal_dispatch_table.hpp
/// @brief Dispatch table mapping AM combinations to optimal kernel variants
///
/// The OptimalDispatchTable stores the best-performing kernel variant for each
/// angular momentum combination. It can be populated from built-in defaults
/// (based on RTX 5070 benchmarks), loaded from a JSON file produced by the
/// bench_optimal_dispatch benchmark, or configured for a specific strategy
/// (all-handwritten, all-generated, or optimal).

#include <libaccint/kernels/contraction_range.hpp>
#include <libaccint/kernels/kernel_variant.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace libaccint::kernels {

/// @brief GPU execution strategy for integral kernels
///
/// Mirrors GpuExecutionStrategy from device_resource_tracker.hpp but is
/// defined independently so the dispatch table works without CUDA.
enum class GpuExecutionStrategy : uint8_t {
    ThreadPerQuartet,  ///< One thread per shell quartet
    WarpPerQuartet,    ///< One warp (32 threads) per shell quartet
    BlockPerQuartet,   ///< One thread block per shell quartet (cooperative)
};

/// @brief Convert GpuExecutionStrategy to string
inline constexpr std::string_view to_string(GpuExecutionStrategy s) noexcept {
    switch (s) {
        case GpuExecutionStrategy::ThreadPerQuartet: return "thread_per_quartet";
        case GpuExecutionStrategy::WarpPerQuartet:   return "warp_per_quartet";
        case GpuExecutionStrategy::BlockPerQuartet:  return "block_per_quartet";
        default:                                     return "unknown";
    }
}

/// @brief Parse a GpuExecutionStrategy from a string
/// @throws std::invalid_argument if the string doesn't match
inline GpuExecutionStrategy gpu_execution_strategy_from_string(std::string_view s) {
    if (s == "thread_per_quartet") return GpuExecutionStrategy::ThreadPerQuartet;
    if (s == "warp_per_quartet")   return GpuExecutionStrategy::WarpPerQuartet;
    if (s == "block_per_quartet")  return GpuExecutionStrategy::BlockPerQuartet;
    throw std::invalid_argument("Unknown GpuExecutionStrategy: " + std::string(s));
}

/// @brief One-electron dispatch entry for an AM pair (la, lb)
struct OneElectronEntry {
    KernelVariant overlap = KernelVariant::HandwrittenOverlap;
    KernelVariant kinetic = KernelVariant::HandwrittenKinetic;
    KernelVariant nuclear = KernelVariant::HandwrittenNuclear;
    bool prefer_fused = true;  ///< true = fused S+T+V beats sum of 3 individual
};

/// @brief Two-electron dispatch entry for an AM quartet (la, lb, lc, ld)
///        and contraction range
struct TwoElectronEntry {
    KernelVariant variant = KernelVariant::HandwrittenERI;
    GpuExecutionStrategy gpu_strategy = GpuExecutionStrategy::ThreadPerQuartet;
};

/// @brief Lookup table mapping AM combinations to optimal kernel variants
///
/// The table stores one entry per AM combination:
/// - 1e: indexed by [la * (max_am+1) + lb]
/// - 2e: indexed by [((la*(M+1)+lb)*(M+1)+lc)*(M+1)+ld] where M = max_am
///
/// Usage:
/// @code
///   OptimalDispatchTable table;  // Built-in defaults
///   auto& entry = table.get_1e(1, 2);
///   if (entry.prefer_fused) { ... }
///   auto& eri = table.get_2e(0, 0, 1, 1);
///   if (eri.variant == KernelVariant::GeneratedERI) { ... }
/// @endcode
class OptimalDispatchTable {
public:
    /// @brief Construct with built-in defaults (conservative: handwritten for 1e,
    ///        optimal mix for ERI based on RTX 5070 benchmark data)
    OptimalDispatchTable();

    /// @brief Construct from a strategy name
    /// @param strategy One of "handwritten", "generated", "optimal"
    explicit OptimalDispatchTable(const std::string& strategy);

    /// @brief Load dispatch table from a JSON file
    /// @param path Path to a JSON file produced by bench_optimal_dispatch
    /// @throws std::runtime_error if file cannot be read or parsed
    [[nodiscard]] static OptimalDispatchTable from_json(const std::string& path);

    /// @brief Write dispatch table to a JSON file
    /// @param path Output path for the JSON file
    void to_json(const std::string& path) const;

    /// @brief Get the one-electron entry for AM pair (la, lb)
    /// @param la Angular momentum of bra shell
    /// @param lb Angular momentum of ket shell
    [[nodiscard]] const OneElectronEntry& get_1e(int la, int lb) const;

    /// @brief Get the two-electron entry for AM quartet (la, lb, lc, ld)
    /// @note Defaults to ContractionRange::SmallK for backward compatibility
    [[nodiscard]] const TwoElectronEntry& get_2e(int la, int lb, int lc, int ld) const;

    /// @brief Get the two-electron entry for AM quartet and contraction range
    /// @param la Angular momentum of first bra shell
    /// @param lb Angular momentum of second bra shell
    /// @param lc Angular momentum of first ket shell
    /// @param ld Angular momentum of second ket shell
    /// @param k  Contraction-degree range
    [[nodiscard]] const TwoElectronEntry& get_2e(int la, int lb, int lc, int ld,
                                                  ContractionRange k) const;

    /// @brief Get the maximum AM supported by this table
    [[nodiscard]] int max_am() const noexcept { return max_am_; }

private:
    void populate_defaults();
    void populate_handwritten();
    void populate_generated();

    /// @brief 1e index: la * (max_am_+1) + lb
    [[nodiscard]] size_t index_1e(int la, int lb) const noexcept {
        return static_cast<size_t>(la) * (max_am_ + 1) + lb;
    }

    /// @brief 2e index without K-range (defaults to SmallK)
    [[nodiscard]] size_t index_2e(int la, int lb, int lc, int ld) const noexcept {
        return index_2e(la, lb, lc, ld, ContractionRange::SmallK);
    }

    /// @brief 2e index: ((la*(M+1)+lb)*(M+1)+lc)*(M+1)+ld)*3+k where M = max_am_
    [[nodiscard]] size_t index_2e(int la, int lb, int lc, int ld,
                                   ContractionRange k) const noexcept {
        const size_t M = max_am_ + 1;
        return (((static_cast<size_t>(la) * M + lb) * M + lc) * M + ld)
               * kNumContractionRanges + static_cast<size_t>(k);
    }

    int max_am_ = 4;
    std::vector<OneElectronEntry> table_1e_;
    std::vector<TwoElectronEntry> table_2e_;
};

}  // namespace libaccint::kernels
