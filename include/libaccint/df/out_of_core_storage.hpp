// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file out_of_core_storage.hpp
/// @brief Disk-backed out-of-core three-center integral storage
///
/// OutOfCoreThreeCenterStorage extends blocked storage with disk I/O.
/// When the in-memory budget is exceeded, blocks are written to disk
/// and re-read on demand via an LRU cache.

#include <libaccint/core/types.hpp>

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace libaccint::df {

/// @brief Configuration for out-of-core storage
struct OutOfCoreConfig {
    Size memory_limit_mb{1024};        ///< In-core memory budget (MB)
    Size block_size_aux{0};            ///< Auxiliary functions per block (0 = auto)
    std::string scratch_dir{"/tmp"};   ///< Directory for temporary files
    std::string file_prefix{"libaccint_ooc"};  ///< Prefix for scratch files
    bool delete_on_destroy{true};      ///< Delete scratch files on destruction
};

/// @brief Disk-backed chunked three-center integral storage
///
/// Stores blocks of the B tensor on disk, loading them into memory
/// via an LRU cache as needed. Designed for systems where the full
/// B tensor exceeds available RAM.
///
/// Usage:
///   1. Create with n_orb, n_aux, and config
///   2. Write blocks via write_block() during initial setup
///   3. Read blocks via read_block() during contractions
///   4. LRU cache manages in-memory copies automatically
class OutOfCoreThreeCenterStorage {
public:
    /// @brief Construct out-of-core storage
    ///
    /// @param n_orb Number of orbital basis functions
    /// @param n_aux Number of auxiliary basis functions
    /// @param config Configuration options
    OutOfCoreThreeCenterStorage(Size n_orb, Size n_aux,
                                 OutOfCoreConfig config = {});

    ~OutOfCoreThreeCenterStorage();

    // Non-copyable
    OutOfCoreThreeCenterStorage(const OutOfCoreThreeCenterStorage&) = delete;
    OutOfCoreThreeCenterStorage& operator=(const OutOfCoreThreeCenterStorage&) = delete;

    // =========================================================================
    // Block I/O
    // =========================================================================

    /// @brief Write a block to disk
    ///
    /// @param block_idx Block index
    /// @param data Block data (n_orb * n_orb * block_size values)
    void write_block(Size block_idx, std::span<const Real> data);

    /// @brief Read a block, using LRU cache
    ///
    /// @param block_idx Block index
    /// @return Const span over block data
    [[nodiscard]] std::span<const Real> read_block(Size block_idx);

    /// @brief Check if a block has been written to disk
    [[nodiscard]] bool has_block(Size block_idx) const;

    // =========================================================================
    // Block Geometry
    // =========================================================================

    /// @brief Get number of blocks
    [[nodiscard]] Size n_blocks() const noexcept { return n_blocks_; }

    /// @brief Get auxiliary function range for a block
    [[nodiscard]] std::pair<Size, Size> block_range(Size block_idx) const;

    /// @brief Get block size (number of aux functions)
    [[nodiscard]] Size block_size(Size block_idx) const;

    // =========================================================================
    // Diagnostics
    // =========================================================================

    /// @brief Get total disk usage (bytes)
    [[nodiscard]] Size disk_usage() const noexcept { return disk_bytes_; }

    /// @brief Get current in-memory cache usage (bytes)
    [[nodiscard]] Size cache_usage() const noexcept { return cache_bytes_; }

    /// @brief Get cache hit rate (0.0 - 1.0)
    [[nodiscard]] double cache_hit_rate() const noexcept {
        Size total = cache_hits_ + cache_misses_;
        return total > 0 ? static_cast<double>(cache_hits_) / static_cast<double>(total) : 0.0;
    }

    /// @brief Get number of cache hits
    [[nodiscard]] Size cache_hits() const noexcept { return cache_hits_; }

    /// @brief Get number of cache misses
    [[nodiscard]] Size cache_misses() const noexcept { return cache_misses_; }

    /// @brief Flush all cached blocks (free memory)
    void flush_cache();

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
    OutOfCoreConfig config_;

    /// Block ranges
    std::vector<std::pair<Size, Size>> block_ranges_;

    /// File paths for each block
    std::vector<std::filesystem::path> block_files_;

    /// Track which blocks have been written
    std::vector<bool> block_written_;

    /// LRU cache: block_idx → data
    std::unordered_map<Size, std::vector<Real>> cache_;
    std::list<Size> lru_order_;
    std::unordered_map<Size, std::list<Size>::iterator> lru_map_;

    Size cache_bytes_{0};
    Size disk_bytes_{0};
    std::unordered_map<Size, Size> block_disk_bytes_;  ///< Per-block disk usage for accurate accounting
    mutable Size cache_hits_{0};
    mutable Size cache_misses_{0};
    Size instance_id_;  ///< Unique instance ID for scratch file name collision prevention

    std::mutex mutex_;

    /// @brief Evict LRU blocks until space is available
    void ensure_cache_space(Size required_bytes);

    /// @brief Load a block from disk
    std::vector<Real> load_from_disk(Size block_idx);
};

}  // namespace libaccint::df
