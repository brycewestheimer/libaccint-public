// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file work_distribution.hpp
/// @brief Work distribution strategies for multi-GPU execution
///
/// Provides algorithms for partitioning shell quartets across multiple GPUs,
/// including cost estimation and load balancing strategies.

#include <libaccint/config.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/basis/shell_set_quartet_utils.hpp>
#include <libaccint/core/types.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace libaccint::device {

/// @brief Estimated computational cost for a shell quartet
struct QuartetCost {
    Size quartet_index;      ///< Index in the original quartet list
    double cost;             ///< Estimated computational cost
    int device_id = -1;      ///< Assigned device (-1 = unassigned)
};

/// @brief Work partition for a single device
struct DeviceWork {
    int device_id;
    std::vector<Size> quartet_indices;  ///< Indices into the original quartet list
    double total_cost = 0.0;            ///< Sum of quartet costs
};

/// @brief Work distribution strategy enumeration
enum class DistributionStrategy {
    RoundRobin,       ///< Simple round-robin assignment
    CostBased,        ///< Assign based on estimated cost (greedy load balancing)
    ChunkBased,       ///< Divide into contiguous chunks
    WorkStealing      ///< Dynamic work stealing (requires runtime support)
};

/// @brief Configuration for work distribution
struct DistributionConfig {
    DistributionStrategy strategy = DistributionStrategy::CostBased;
    
    /// Minimum quartets per device (to avoid overhead)
    Size min_quartets_per_device = 100;
    
    /// For CostBased: relative compute power per device (indexed by device order)
    std::vector<double> device_weights;
    
    /// For WorkStealing: chunk size for dynamic assignment
    Size steal_chunk_size = 64;
};

/// @brief Estimate computational cost for a shell quartet
///
/// Cost model considers:
/// - Angular momentum (higher AM = more computation)
/// - Primitive count (more primitives = more loops)
/// - Contraction degree
///
/// @param la Angular momentum of shell a
/// @param lb Angular momentum of shell b
/// @param lc Angular momentum of shell c
/// @param ld Angular momentum of shell d
/// @param nprim_a Number of primitives in shell a
/// @param nprim_b Number of primitives in shell b
/// @param nprim_c Number of primitives in shell c
/// @param nprim_d Number of primitives in shell d
/// @return Estimated relative cost (higher = more expensive)
inline double estimate_quartet_cost(
    int la, int lb, int lc, int ld,
    int nprim_a, int nprim_b, int nprim_c, int nprim_d) {
    
    // Number of Cartesian basis functions per shell
    auto n_cart = [](int L) { return (L + 1) * (L + 2) / 2; };
    
    const int na = n_cart(la);
    const int nb = n_cart(lb);
    const int nc = n_cart(lc);
    const int nd = n_cart(ld);
    
    // Total number of integrals
    const double n_integrals = static_cast<double>(na * nb * nc * nd);
    
    // Primitive loops (outer loop cost)
    const double prim_cost = static_cast<double>(nprim_a * nprim_b * nprim_c * nprim_d);
    
    // Boys function evaluations scale with AM
    const int L_total = la + lb + lc + ld;
    const double boys_cost = static_cast<double>(L_total + 1);
    
    // Combined cost model
    // The primitive cost dominates for contracted shells
    // The integral count dominates for high-AM uncontracted shells
    return prim_cost * (n_integrals + boys_cost * 10.0);
}

/// @brief Estimate cost for a ShellSetQuartet
/// @param quartet The shell set quartet
/// @return Estimated relative cost
inline double estimate_quartet_cost(const ShellSetQuartet& quartet) {
    // Reuse the canonical ShellSetQuartet cost model in basis utilities.
    return ::libaccint::estimate_quartet_cost(quartet);
}

/// @brief Work distribution algorithm
///
/// Implements various strategies for distributing shell quartets across
/// multiple GPU devices for load-balanced parallel execution.
class WorkDistributor {
public:
    /// @brief Construct a work distributor
    /// @param config Distribution configuration
    explicit WorkDistributor(DistributionConfig config = {})
        : config_(std::move(config)) {}
    
    /// @brief Partition quartets across devices
    /// @param quartets List of shell set quartets to partition
    /// @param device_ids List of target device IDs
    /// @return Vector of DeviceWork, one per device
    [[nodiscard]] std::vector<DeviceWork> partition(
        const std::vector<ShellSetQuartet>& quartets,
        const std::vector<int>& device_ids) const;
    
    /// @brief Partition pre-computed costs across devices
    /// @param costs Quartet costs (must be same length as quartets)
    /// @param device_ids List of target device IDs
    /// @return Vector of DeviceWork, one per device
    [[nodiscard]] std::vector<DeviceWork> partition_by_cost(
        std::vector<QuartetCost> costs,
        const std::vector<int>& device_ids) const;
    
    /// @brief Get the current configuration
    [[nodiscard]] const DistributionConfig& config() const noexcept {
        return config_;
    }
    
    /// @brief Set configuration
    void set_config(DistributionConfig config) {
        config_ = std::move(config);
    }

private:
    [[nodiscard]] std::vector<DeviceWork> partition_round_robin(
        const std::vector<QuartetCost>& costs,
        const std::vector<int>& device_ids) const;
    
    [[nodiscard]] std::vector<DeviceWork> partition_cost_based(
        std::vector<QuartetCost> costs,
        const std::vector<int>& device_ids) const;
    
    [[nodiscard]] std::vector<DeviceWork> partition_chunk_based(
        const std::vector<QuartetCost>& costs,
        const std::vector<int>& device_ids) const;
    
    DistributionConfig config_;
};

// ============================================================================
// Inline Implementation
// ============================================================================

inline std::vector<DeviceWork> WorkDistributor::partition(
    const std::vector<ShellSetQuartet>& quartets,
    const std::vector<int>& device_ids) const {
    
    // Compute costs for all quartets
    std::vector<QuartetCost> costs;
    costs.reserve(quartets.size());
    
    for (Size i = 0; i < quartets.size(); ++i) {
        costs.push_back(
            QuartetCost{i, ::libaccint::device::estimate_quartet_cost(quartets[i]), -1});
    }
    
    return partition_by_cost(std::move(costs), device_ids);
}

inline std::vector<DeviceWork> WorkDistributor::partition_by_cost(
    std::vector<QuartetCost> costs,
    const std::vector<int>& device_ids) const {
    
    if (device_ids.empty() || costs.empty()) {
        return {};
    }
    
    switch (config_.strategy) {
        case DistributionStrategy::RoundRobin:
            return partition_round_robin(costs, device_ids);
        case DistributionStrategy::CostBased:
            return partition_cost_based(std::move(costs), device_ids);
        case DistributionStrategy::ChunkBased:
            return partition_chunk_based(costs, device_ids);
        case DistributionStrategy::WorkStealing:
            // Work stealing falls back to cost-based for initial assignment
            return partition_cost_based(std::move(costs), device_ids);
    }
    
    return partition_cost_based(std::move(costs), device_ids);
}

inline std::vector<DeviceWork> WorkDistributor::partition_round_robin(
    const std::vector<QuartetCost>& costs,
    const std::vector<int>& device_ids) const {
    
    const auto n_devices = device_ids.size();
    std::vector<DeviceWork> result;
    result.reserve(n_devices);
    
    for (int device_id : device_ids) {
        result.push_back({device_id, {}, 0.0});
    }
    
    for (size_t i = 0; i < costs.size(); ++i) {
        auto& work = result[i % n_devices];
        work.quartet_indices.push_back(costs[i].quartet_index);
        work.total_cost += costs[i].cost;
    }
    
    return result;
}

inline std::vector<DeviceWork> WorkDistributor::partition_cost_based(
    std::vector<QuartetCost> costs,
    const std::vector<int>& device_ids) const {
    
    const auto n_devices = device_ids.size();
    std::vector<DeviceWork> result;
    result.reserve(n_devices);
    
    for (int device_id : device_ids) {
        result.push_back({device_id, {}, 0.0});
    }
    
    // Get device weights (default to equal if not specified)
    std::vector<double> weights = config_.device_weights;
    if (weights.size() != n_devices) {
        weights.assign(n_devices, 1.0);
    }
    
    // Normalize weights
    double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (auto& w : weights) {
        w /= total_weight;
    }
    
    // Sort quartets by cost descending (LPT heuristic)
    std::sort(costs.begin(), costs.end(),
              [](const auto& a, const auto& b) { return a.cost > b.cost; });
    
    // Greedy assignment: assign each quartet to the device with
    // lowest current load relative to its weight
    for (const auto& qc : costs) {
        // Find device with minimum load/weight ratio
        size_t best_device = 0;
        double best_ratio = result[0].total_cost / weights[0];
        
        for (size_t d = 1; d < n_devices; ++d) {
            double ratio = result[d].total_cost / weights[d];
            if (ratio < best_ratio) {
                best_ratio = ratio;
                best_device = d;
            }
        }
        
        result[best_device].quartet_indices.push_back(qc.quartet_index);
        result[best_device].total_cost += qc.cost;
    }
    
    return result;
}

inline std::vector<DeviceWork> WorkDistributor::partition_chunk_based(
    const std::vector<QuartetCost>& costs,
    const std::vector<int>& device_ids) const {
    
    const auto n_devices = device_ids.size();
    const auto n_quartets = costs.size();
    
    std::vector<DeviceWork> result;
    result.reserve(n_devices);
    
    // Divide into roughly equal chunks
    const auto chunk_size = (n_quartets + n_devices - 1) / n_devices;
    
    for (size_t d = 0; d < n_devices; ++d) {
        DeviceWork work{device_ids[d], {}, 0.0};
        
        const auto start = d * chunk_size;
        const auto end = std::min(start + chunk_size, n_quartets);
        
        for (size_t i = start; i < end; ++i) {
            work.quartet_indices.push_back(costs[i].quartet_index);
            work.total_cost += costs[i].cost;
        }
        
        result.push_back(std::move(work));
    }
    
    return result;
}

}  // namespace libaccint::device
