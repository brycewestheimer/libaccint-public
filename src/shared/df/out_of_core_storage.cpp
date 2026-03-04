// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file out_of_core_storage.cpp
/// @brief Implementation of disk-backed out-of-core three-center storage

#include <libaccint/df/out_of_core_storage.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace libaccint::df {

// Instance counter for unique file names
static std::atomic<Size> s_instance_counter{0};

// ============================================================================
// Construction / Destruction
// ============================================================================

OutOfCoreThreeCenterStorage::OutOfCoreThreeCenterStorage(
    Size n_orb, Size n_aux, OutOfCoreConfig config)
    : n_orb_(n_orb),
      n_aux_(n_aux),
      config_(std::move(config)),
      instance_id_(s_instance_counter.fetch_add(1)) {

    memory_limit_ = config_.memory_limit_mb * 1024ULL * 1024ULL;

    // Compute block size
    if (config_.block_size_aux > 0) {
        block_size_aux_ = config_.block_size_aux;
    } else {
        Size one_aux_slice = n_orb_ * n_orb_ * sizeof(Real);
        if (one_aux_slice == 0) {
            block_size_aux_ = n_aux_;
        } else {
            Size target = memory_limit_ / 10;
            block_size_aux_ = std::max(Size{1}, target / one_aux_slice);
            block_size_aux_ = std::min(block_size_aux_, n_aux_);
        }
    }

    // Compute block ranges
    n_blocks_ = (n_aux_ + block_size_aux_ - 1) / block_size_aux_;
    block_ranges_.resize(n_blocks_);
    block_files_.resize(n_blocks_);
    block_written_.resize(n_blocks_, false);

    std::filesystem::path scratch(config_.scratch_dir);
    std::filesystem::create_directories(scratch);

    for (Size i = 0; i < n_blocks_; ++i) {
        Size start = i * block_size_aux_;
        Size end = std::min(start + block_size_aux_, n_aux_);
        block_ranges_[i] = {start, end};

        // Include instance ID to prevent file name collisions between instances
        block_files_[i] = scratch /
            (config_.file_prefix + "_" + std::to_string(instance_id_) +
             "_block_" + std::to_string(i) + ".bin");
    }
}

OutOfCoreThreeCenterStorage::~OutOfCoreThreeCenterStorage() {
    if (config_.delete_on_destroy) {
        for (const auto& path : block_files_) {
            std::filesystem::remove(path);
        }
    }
}

// ============================================================================
// Block I/O
// ============================================================================

void OutOfCoreThreeCenterStorage::write_block(
    Size block_idx, std::span<const Real> data) {

    if (block_idx >= n_blocks_) {
        throw InvalidArgumentException("Block index out of range");
    }

    auto [start, end] = block_ranges_[block_idx];
    Size bs = end - start;
    Size expected = n_orb_ * n_orb_ * bs;
    if (data.size() != expected) {
        throw InvalidArgumentException(
            "Block data size mismatch: expected " + std::to_string(expected) +
            ", got " + std::to_string(data.size()));
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Overflow check: ensure write size fits in std::streamsize
    const auto write_bytes = data.size() * sizeof(Real);
    if (write_bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw InvalidStateException(
            "Block write size (" + std::to_string(write_bytes) +
            " bytes) exceeds std::streamsize limit");
    }

    // Write to disk
    std::ofstream ofs(block_files_[block_idx], std::ios::binary);
    if (!ofs.is_open()) {
        throw InvalidStateException(
            "Failed to open scratch file: " + block_files_[block_idx].string());
    }
    ofs.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(write_bytes));

    // Check for I/O errors after writing
    if (!ofs.good()) {
        throw InvalidStateException(
            "Failed to write block data to scratch file: " +
            block_files_[block_idx].string() +
            " (disk full or I/O error)");
    }
    ofs.close();

    // Fix disk accounting: subtract old size on overwrite, then add new size
    Size block_bytes = data.size() * sizeof(Real);
    auto disk_it = block_disk_bytes_.find(block_idx);
    if (disk_it != block_disk_bytes_.end()) {
        disk_bytes_ -= disk_it->second;
    }
    block_disk_bytes_[block_idx] = block_bytes;
    block_written_[block_idx] = true;
    disk_bytes_ += block_bytes;

    // Fix LRU: remove existing entry before inserting new one
    auto lru_it = lru_map_.find(block_idx);
    if (lru_it != lru_map_.end()) {
        lru_order_.erase(lru_it->second);
        lru_map_.erase(lru_it);

        // Account for old cache entry
        auto cache_it = cache_.find(block_idx);
        if (cache_it != cache_.end()) {
            cache_bytes_ -= cache_it->second.size() * sizeof(Real);
            cache_.erase(cache_it);
        }
    }

    // Cache if memory allows
    ensure_cache_space(block_bytes);

    cache_[block_idx] = std::vector<Real>(data.begin(), data.end());
    cache_bytes_ += block_bytes;
    lru_order_.push_back(block_idx);
    lru_map_[block_idx] = std::prev(lru_order_.end());
}

std::span<const Real> OutOfCoreThreeCenterStorage::read_block(Size block_idx) {
    if (block_idx >= n_blocks_) {
        throw InvalidArgumentException("Block index out of range");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Check cache first
    auto it = cache_.find(block_idx);
    if (it != cache_.end()) {
        ++cache_hits_;
        // Touch LRU
        auto lru_it = lru_map_.find(block_idx);
        if (lru_it != lru_map_.end()) {
            lru_order_.erase(lru_it->second);
            lru_order_.push_back(block_idx);
            lru_it->second = std::prev(lru_order_.end());
        }
        return std::span<const Real>(it->second);
    }

    ++cache_misses_;

    // Load from disk
    if (!block_written_[block_idx]) {
        throw InvalidStateException(
            "Block " + std::to_string(block_idx) + " has not been written");
    }

    auto data = load_from_disk(block_idx);
    Size block_bytes = data.size() * sizeof(Real);
    ensure_cache_space(block_bytes);

    cache_[block_idx] = std::move(data);
    cache_bytes_ += block_bytes;
    lru_order_.push_back(block_idx);
    lru_map_[block_idx] = std::prev(lru_order_.end());

    return std::span<const Real>(cache_[block_idx]);
}

bool OutOfCoreThreeCenterStorage::has_block(Size block_idx) const {
    if (block_idx >= n_blocks_) return false;
    return block_written_[block_idx];
}

// ============================================================================
// Block Geometry
// ============================================================================

std::pair<Size, Size> OutOfCoreThreeCenterStorage::block_range(
    Size block_idx) const {
    if (block_idx >= n_blocks_) {
        throw InvalidArgumentException("Block index out of range");
    }
    return block_ranges_[block_idx];
}

Size OutOfCoreThreeCenterStorage::block_size(Size block_idx) const {
    auto [start, end] = block_range(block_idx);
    return end - start;
}

// ============================================================================
// Cache Management
// ============================================================================

void OutOfCoreThreeCenterStorage::flush_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    lru_order_.clear();
    lru_map_.clear();
    cache_bytes_ = 0;
}

void OutOfCoreThreeCenterStorage::ensure_cache_space(Size required_bytes) {
    while (cache_bytes_ + required_bytes > memory_limit_ && !lru_order_.empty()) {
        Size victim = lru_order_.front();
        lru_order_.pop_front();
        lru_map_.erase(victim);

        auto it = cache_.find(victim);
        if (it != cache_.end()) {
            cache_bytes_ -= it->second.size() * sizeof(Real);
            cache_.erase(it);
        }
    }
}

std::vector<Real> OutOfCoreThreeCenterStorage::load_from_disk(Size block_idx) {
    auto [start, end] = block_ranges_[block_idx];
    Size bs = end - start;
    Size n_elements = n_orb_ * n_orb_ * bs;

    std::ifstream ifs(block_files_[block_idx], std::ios::binary);
    if (!ifs.is_open()) {
        throw InvalidStateException(
            "Failed to open scratch file: " + block_files_[block_idx].string());
    }

    // Overflow check: ensure read size fits in std::streamsize
    const auto read_bytes = n_elements * sizeof(Real);
    if (read_bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw InvalidStateException(
            "Block read size (" + std::to_string(read_bytes) +
            " bytes) exceeds std::streamsize limit");
    }

    std::vector<Real> data(n_elements);
    ifs.read(reinterpret_cast<char*>(data.data()),
             static_cast<std::streamsize>(read_bytes));

    if (!ifs.good()) {
        throw InvalidStateException(
            "Failed to read block data from: " + block_files_[block_idx].string());
    }

    return data;
}

}  // namespace libaccint::df
