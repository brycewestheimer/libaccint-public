// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file optimal_dispatch_table.cpp
/// @brief OptimalDispatchTable implementation with defaults, JSON I/O, and strategy selection

#include <libaccint/kernels/optimal_dispatch_table.hpp>

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace libaccint::kernels {

// =============================================================================
// Constructors
// =============================================================================

OptimalDispatchTable::OptimalDispatchTable() {
    populate_defaults();
}

OptimalDispatchTable::OptimalDispatchTable(const std::string& strategy) {
    if (strategy == "handwritten") {
        populate_handwritten();
    } else if (strategy == "generated") {
        populate_generated();
    } else if (strategy == "optimal") {
        populate_defaults();
    } else {
        throw std::invalid_argument(
            "Unknown kernel dispatch strategy: '" + strategy +
            "'. Valid options: handwritten, generated, optimal");
    }
}

// =============================================================================
// Strategy populators
// =============================================================================

void OptimalDispatchTable::populate_handwritten() {
    const size_t n1e = static_cast<size_t>(max_am_ + 1) * (max_am_ + 1);
    const size_t M = max_am_ + 1;
    const size_t n2e = M * M * M * M * kNumContractionRanges;

    table_1e_.assign(n1e, OneElectronEntry{
        KernelVariant::HandwrittenOverlap,
        KernelVariant::HandwrittenKinetic,
        KernelVariant::HandwrittenNuclear,
        true  // prefer_fused
    });

    table_2e_.assign(n2e, TwoElectronEntry{
        KernelVariant::HandwrittenERI,
        GpuExecutionStrategy::ThreadPerQuartet
    });
}

void OptimalDispatchTable::populate_generated() {
    const size_t n1e = static_cast<size_t>(max_am_ + 1) * (max_am_ + 1);
    const size_t M = max_am_ + 1;
    const size_t n2e = M * M * M * M * kNumContractionRanges;

    table_1e_.assign(n1e, OneElectronEntry{
        KernelVariant::GeneratedOverlap,
        KernelVariant::GeneratedKinetic,
        KernelVariant::GeneratedNuclear,
        false  // individual generated kernels, no fusing
    });

    table_2e_.assign(n2e, TwoElectronEntry{
        KernelVariant::GeneratedERI,
        GpuExecutionStrategy::ThreadPerQuartet
    });

    // Use cooperative ERI for high-AM quartets (total_am >= 5)
    // and set gpu_strategy based on total_am
    for (int la = 0; la <= max_am_; ++la) {
        for (int lb = 0; lb <= max_am_; ++lb) {
            for (int lc = 0; lc <= max_am_; ++lc) {
                for (int ld = 0; ld <= max_am_; ++ld) {
                    const int total_am = la + lb + lc + ld;
                    for (int ki = 0; ki < static_cast<int>(kNumContractionRanges); ++ki) {
                        auto k = static_cast<ContractionRange>(ki);
                        auto& entry = table_2e_[index_2e(la, lb, lc, ld, k)];
                        if (total_am >= 5) {
                            entry.variant = KernelVariant::CooperativeERI;
                            entry.gpu_strategy = GpuExecutionStrategy::BlockPerQuartet;
                        }
                    }
                }
            }
        }
    }
}

void OptimalDispatchTable::populate_defaults() {
    // In the alpha release, generated kernels are not available.
    // Default to all-handwritten dispatch. When generated kernels are
    // included in a future release, this method will incorporate
    // benchmark-driven overrides for mid/high-AM quartets.
    populate_handwritten();
}

// =============================================================================
// Accessors
// =============================================================================

const OneElectronEntry& OptimalDispatchTable::get_1e(int la, int lb) const {
    if (la < 0 || la > max_am_ || lb < 0 || lb > max_am_) {
        static const OneElectronEntry fallback{};
        return fallback;
    }
    return table_1e_[index_1e(la, lb)];
}

const TwoElectronEntry& OptimalDispatchTable::get_2e(int la, int lb, int lc, int ld) const {
    return get_2e(la, lb, lc, ld, ContractionRange::SmallK);
}

const TwoElectronEntry& OptimalDispatchTable::get_2e(int la, int lb, int lc, int ld,
                                                      ContractionRange k) const {
    if (la < 0 || la > max_am_ || lb < 0 || lb > max_am_ ||
        lc < 0 || lc > max_am_ || ld < 0 || ld > max_am_) {
        static const TwoElectronEntry fallback{};
        return fallback;
    }
    return table_2e_[index_2e(la, lb, lc, ld, k)];
}

// =============================================================================
// JSON I/O
// =============================================================================

/// Helper: map 1e variant string to the appropriate KernelVariant for each integral type
static KernelVariant parse_1e_variant(const std::string& s, const std::string& integral_type) {
    if (s == "handwritten") {
        if (integral_type == "overlap") return KernelVariant::HandwrittenOverlap;
        if (integral_type == "kinetic") return KernelVariant::HandwrittenKinetic;
        if (integral_type == "nuclear") return KernelVariant::HandwrittenNuclear;
    } else if (s == "generated") {
        if (integral_type == "overlap") return KernelVariant::GeneratedOverlap;
        if (integral_type == "kinetic") return KernelVariant::GeneratedKinetic;
        if (integral_type == "nuclear") return KernelVariant::GeneratedNuclear;
    }
    return kernel_variant_from_string(s);
}

static std::string variant_to_short_1e(KernelVariant v) {
    switch (v) {
        case KernelVariant::HandwrittenOverlap:
        case KernelVariant::HandwrittenKinetic:
        case KernelVariant::HandwrittenNuclear:
            return "handwritten";
        case KernelVariant::GeneratedOverlap:
        case KernelVariant::GeneratedKinetic:
        case KernelVariant::GeneratedNuclear:
            return "generated";
        default:
            return std::string(to_string(v));
    }
}

static std::string variant_to_short_2e(KernelVariant v) {
    switch (v) {
        case KernelVariant::HandwrittenERI:  return "handwritten";
        case KernelVariant::GeneratedERI:    return "generated";
        case KernelVariant::CooperativeERI:  return "cooperative";
        default: return std::string(to_string(v));
    }
}

OptimalDispatchTable OptimalDispatchTable::from_json(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open dispatch table JSON: " + path);
    }

    nlohmann::json j;
    file >> j;

    OptimalDispatchTable table;

    if (j.contains("max_am")) {
        table.max_am_ = j["max_am"].get<int>();
    }

    // Re-allocate tables for the loaded max_am
    const size_t n1e = static_cast<size_t>(table.max_am_ + 1) * (table.max_am_ + 1);
    const size_t M = table.max_am_ + 1;
    const size_t n2e = M * M * M * M * kNumContractionRanges;
    table.table_1e_.assign(n1e, OneElectronEntry{});
    table.table_2e_.assign(n2e, TwoElectronEntry{});

    // Load 1e entries
    if (j.contains("one_electron")) {
        for (const auto& entry : j["one_electron"]) {
            int la = entry["la"].get<int>();
            int lb = entry["lb"].get<int>();
            if (la > table.max_am_ || lb > table.max_am_) continue;

            auto& e = table.table_1e_[table.index_1e(la, lb)];
            e.overlap = parse_1e_variant(entry["overlap"].get<std::string>(), "overlap");
            e.kinetic = parse_1e_variant(entry["kinetic"].get<std::string>(), "kinetic");
            e.nuclear = parse_1e_variant(entry["nuclear"].get<std::string>(), "nuclear");
            e.prefer_fused = entry.value("prefer_fused", true);
        }
    }

    // Load 2e entries
    if (j.contains("two_electron")) {
        for (const auto& entry : j["two_electron"]) {
            int la = entry["la"].get<int>();
            int lb = entry["lb"].get<int>();
            int lc = entry["lc"].get<int>();
            int ld = entry["ld"].get<int>();
            if (la > table.max_am_ || lb > table.max_am_ ||
                lc > table.max_am_ || ld > table.max_am_) continue;

            std::string variant_str = entry["variant"].get<std::string>();
            KernelVariant variant = kernel_variant_from_string(variant_str);

            // Parse GPU strategy (default: ThreadPerQuartet for backward compat)
            GpuExecutionStrategy gpu_strat = GpuExecutionStrategy::ThreadPerQuartet;
            if (entry.contains("gpu_strategy")) {
                gpu_strat = gpu_execution_strategy_from_string(
                    entry["gpu_strategy"].get<std::string>());
            }

            if (entry.contains("contraction_range")) {
                // New format: entry is specific to one contraction range
                ContractionRange k = contraction_range_from_string(
                    entry["contraction_range"].get<std::string>());
                auto& e = table.table_2e_[table.index_2e(la, lb, lc, ld, k)];
                e.variant = variant;
                e.gpu_strategy = gpu_strat;
            } else {
                // Old format: no contraction_range — apply to all K-ranges
                for (int ki = 0; ki < static_cast<int>(kNumContractionRanges); ++ki) {
                    auto k = static_cast<ContractionRange>(ki);
                    auto& e = table.table_2e_[table.index_2e(la, lb, lc, ld, k)];
                    e.variant = variant;
                    e.gpu_strategy = gpu_strat;
                }
            }
        }
    }

    return table;
}

void OptimalDispatchTable::to_json(const std::string& path) const {
    nlohmann::json j;
    j["max_am"] = max_am_;

    // 1e entries
    auto& one_e = j["one_electron"];
    one_e = nlohmann::json::array();
    for (int la = 0; la <= max_am_; ++la) {
        for (int lb = 0; lb <= max_am_; ++lb) {
            const auto& e = table_1e_[index_1e(la, lb)];
            one_e.push_back({
                {"la", la},
                {"lb", lb},
                {"overlap", variant_to_short_1e(e.overlap)},
                {"kinetic", variant_to_short_1e(e.kinetic)},
                {"nuclear", variant_to_short_1e(e.nuclear)},
                {"prefer_fused", e.prefer_fused}
            });
        }
    }

    // 2e entries (one entry per AM quartet per contraction range)
    auto& two_e = j["two_electron"];
    two_e = nlohmann::json::array();
    for (int la = 0; la <= max_am_; ++la) {
        for (int lb = 0; lb <= max_am_; ++lb) {
            for (int lc = 0; lc <= max_am_; ++lc) {
                for (int ld = 0; ld <= max_am_; ++ld) {
                    for (int ki = 0; ki < static_cast<int>(kNumContractionRanges); ++ki) {
                        auto k = static_cast<ContractionRange>(ki);
                        const auto& e = table_2e_[index_2e(la, lb, lc, ld, k)];
                        two_e.push_back({
                            {"la", la}, {"lb", lb}, {"lc", lc}, {"ld", ld},
                            {"contraction_range", std::string(to_string(k))},
                            {"variant", variant_to_short_2e(e.variant)},
                            {"gpu_strategy", std::string(to_string(e.gpu_strategy))}
                        });
                    }
                }
            }
        }
    }

    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot write dispatch table JSON: " + path);
    }
    file << j.dump(2) << "\n";
}

}  // namespace libaccint::kernels
