// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file three_center_storage.hpp
/// @brief Memory-efficient blocked three-center integral storage
///
/// ThreeCenterBlockStorage partitions the B tensor by auxiliary shell blocks
/// to avoid the full N_aux x N^2 memory footprint. Blocks can be loaded,
/// computed, and evicted as needed.

#include <libaccint/core/types.hpp>

#include <cstddef>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace libaccint::df {

/// @brief Configuration for blocked three-center storage
struct BlockStorageConfig {
    Size memory_limit_mb{2048};        ///< Maximum memory budget in MB
    Size block_size_aux{0};            ///< Auxiliary functions per block (0 = auto)
    bool use_symmetry{true};           ///< Exploit bra symmetry (ab = ba)
    bool prefetch_next{false};         ///< Prefetch next block while processing current
};

/// @brief Memory-efficient blocked storage for three-center integrals
///
/// Partitions the B tensor B_{ab}^P into blocks along the auxiliary index P.
/// Each block covers a contiguous range of auxiliary functions and stores
/// the full n_orb x n_orb data for those auxiliary functions.
///
/// Features:
///   - LRU eviction when memory limit is reached
///   - Block-based access pattern for J and K contraction
///   - Optional bra symmetry exploitation (stores only upper triangle)
///   - Thread-safe block loading with mutex protection
class ThreeCenterBlockStorage {
public:
    /// @brief Construct block storage
    ///
    /// @param n_orb Number of orbital basis functions
    /// @param n_aux Number of auxiliary basis functions
    /// @param config Configuration options
    ThreeCenterBlockStorage(Size n_orb, Size n_aux,
                            BlockStorageConfig config = {});

    ~ThreeCenterBlockStorage() = default;

    // Non-copyable
    ThreeCenterBlockStorage(const ThreeCenterBlockStorage&) = delete;
    ThreeCenterBlockStorage& operator=(const ThreeCenterBlockStorage&) = delete;

    // Movable
    ThreeCenterBlockStorage(ThreeCenterBlockStorage&&) noexcept = default;
    ThreeCenterBlockStorage& operator=(ThreeCenterBlockStorage&&) noexcept = default;

    // =========================================================================
    // Block Access
    // =========================================================================

    /// @brief Get the number of blocks
    [[nodiscard]] Size n_blocks() const noexcept { return n_blocks_; }

    /// @brief Get auxiliary function range for a block
    /// @param block_idx Block index
    /// @return Pair of (start_aux, end_aux) — half-open range
    [[nodiscard]] std::pair<Size, Size> block_range(Size block_idx) const;

    /// @brief Get number of auxiliary functions in a block
    [[nodiscard]] Size block_size(Size block_idx) const;

    /// @brief Store data for a block
    ///
    /// @param block_idx Block index
    /// @param data Block data (n_orb * n_orb * block_size values)
    void store_block(Size block_idx, const std::vector<Real>& data);

    /// @brief Get read-only access to a block's data
    ///
    /// May trigger LRU eviction if the block is not in cache.
    /// Accessing a non-loaded block throws.
    ///
    /// @param block_idx Block index
    /// @return Span over block data
    [[nodiscard]] std::span<const Real> get_block(Size block_idx) const;

    /// @brief Check if a block is loaded in memory
    [[nodiscard]] bool is_block_loaded(Size block_idx) const;

    /// @brief Evict a block from memory
    void evict_block(Size block_idx);

    /// @brief Evict all blocks
    void clear();

    // =========================================================================
    // Bulk Operations
    // =========================================================================

    /// @brief Store the full B tensor by splitting into blocks
    ///
    /// Takes a full B tensor in (n_orb^2, n_aux) row-major layout
    /// and splits it into blocks.
    ///
    /// @param full_tensor Full B tensor data
    void store_full_tensor(std::span<const Real> full_tensor);

    /// @brief Reconstruct full tensor from blocks
    ///
    /// @return Full B tensor in (n_orb^2, n_aux) row-major layout
    [[nodiscard]] std::vector<Real> reconstruct_full_tensor() const;

    // =========================================================================
    // Diagnostics
    // =========================================================================

    /// @brief Get total memory used by loaded blocks (bytes)
    [[nodiscard]] Size memory_used() const noexcept { return memory_used_; }

    /// @brief Get memory limit (bytes)
    [[nodiscard]] Size memory_limit() const noexcept { return memory_limit_; }

    /// @brief Get number of currently loaded blocks
    [[nodiscard]] Size n_loaded_blocks() const noexcept { return blocks_.size(); }

    /// @brief Get orbital basis size
    [[nodiscard]] Size n_orb() const noexcept { return n_orb_; }

    /// @brief Get auxiliary basis size
    [[nodiscard]] Size n_aux() const noexcept { return n_aux_; }

private:
    Size n_orb_;
    Size n_aux_;
    Size n_blocks_;
    Size block_size_aux_;
    Size memory_limit_;
    mutable Size memory_used_{0};
    BlockStorageConfig config_;

    /// Block ranges: block_ranges_[i] = {start_aux, end_aux}
    std::vector<std::pair<Size, Size>> block_ranges_;

    /// Cached block data: block_idx → data
    mutable std::unordered_map<Size, std::vector<Real>> blocks_;

    /// LRU tracking: most recently used at back
    mutable std::list<Size> lru_order_;
    mutable std::unordered_map<Size, std::list<Size>::iterator> lru_map_;

    mutable std::mutex mutex_;

    /// @brief Ensure memory budget allows loading a block
    void ensure_memory(Size required_bytes) const;

    /// @brief Touch a block in LRU (move to back)
    void touch_lru(Size block_idx) const;
};

}  // namespace libaccint::df
