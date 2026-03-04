// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file three_center_storage.cpp
/// @brief Implementation of blocked three-center integral storage

#include <libaccint/df/three_center_storage.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <algorithm>
#include <cmath>

namespace libaccint::df {

// ============================================================================
// Construction
// ============================================================================

ThreeCenterBlockStorage::ThreeCenterBlockStorage(
    Size n_orb, Size n_aux, BlockStorageConfig config)
    : n_orb_(n_orb),
      n_aux_(n_aux),
      config_(config) {

    memory_limit_ = config.memory_limit_mb * 1024ULL * 1024ULL;

    // Compute block size
    if (config.block_size_aux > 0) {
        block_size_aux_ = config.block_size_aux;
    } else {
        // Auto: choose block size so each block fits in ~10% of memory budget
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
    for (Size i = 0; i < n_blocks_; ++i) {
        Size start = i * block_size_aux_;
        Size end = std::min(start + block_size_aux_, n_aux_);
        block_ranges_[i] = {start, end};
    }
}

// ============================================================================
// Block Access
// ============================================================================

std::pair<Size, Size> ThreeCenterBlockStorage::block_range(Size block_idx) const {
    if (block_idx >= n_blocks_) {
        throw InvalidArgumentException(
            "Block index " + std::to_string(block_idx) +
            " out of range (n_blocks = " + std::to_string(n_blocks_) + ")");
    }
    return block_ranges_[block_idx];
}

Size ThreeCenterBlockStorage::block_size(Size block_idx) const {
    auto [start, end] = block_range(block_idx);
    return end - start;
}

void ThreeCenterBlockStorage::store_block(Size block_idx,
                                           const std::vector<Real>& data) {
    if (block_idx >= n_blocks_) {
        throw InvalidArgumentException("Block index out of range");
    }

    Size bs = block_size(block_idx);
    Size expected = n_orb_ * n_orb_ * bs;
    if (data.size() != expected) {
        throw InvalidArgumentException(
            "Block data size mismatch: expected " + std::to_string(expected) +
            ", got " + std::to_string(data.size()));
    }

    std::lock_guard<std::mutex> lock(mutex_);

    Size block_bytes = data.size() * sizeof(Real);

    // Evict old data for this block if present
    auto it = blocks_.find(block_idx);
    if (it != blocks_.end()) {
        memory_used_ -= it->second.size() * sizeof(Real);
        blocks_.erase(it);
        auto lru_it = lru_map_.find(block_idx);
        if (lru_it != lru_map_.end()) {
            lru_order_.erase(lru_it->second);
            lru_map_.erase(lru_it);
        }
    }

    // Ensure memory
    ensure_memory(block_bytes);

    blocks_[block_idx] = data;
    memory_used_ += block_bytes;

    lru_order_.push_back(block_idx);
    lru_map_[block_idx] = std::prev(lru_order_.end());
}

std::span<const Real> ThreeCenterBlockStorage::get_block(Size block_idx) const {
    if (block_idx >= n_blocks_) {
        throw InvalidArgumentException("Block index out of range");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = blocks_.find(block_idx);
    if (it == blocks_.end()) {
        throw InvalidStateException(
            "Block " + std::to_string(block_idx) + " is not loaded");
    }

    touch_lru(block_idx);
    return std::span<const Real>(it->second);
}

bool ThreeCenterBlockStorage::is_block_loaded(Size block_idx) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return blocks_.find(block_idx) != blocks_.end();
}

void ThreeCenterBlockStorage::evict_block(Size block_idx) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = blocks_.find(block_idx);
    if (it != blocks_.end()) {
        memory_used_ -= it->second.size() * sizeof(Real);
        blocks_.erase(it);
        auto lru_it = lru_map_.find(block_idx);
        if (lru_it != lru_map_.end()) {
            lru_order_.erase(lru_it->second);
            lru_map_.erase(lru_it);
        }
    }
}

void ThreeCenterBlockStorage::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    blocks_.clear();
    lru_order_.clear();
    lru_map_.clear();
    memory_used_ = 0;
}

// ============================================================================
// Bulk Operations
// ============================================================================

void ThreeCenterBlockStorage::store_full_tensor(std::span<const Real> full_tensor) {
    Size expected = n_orb_ * n_orb_ * n_aux_;
    if (full_tensor.size() != expected) {
        throw InvalidArgumentException(
            "Full tensor size mismatch: expected " + std::to_string(expected) +
            ", got " + std::to_string(full_tensor.size()));
    }

    Size n_pairs = n_orb_ * n_orb_;

    for (Size b = 0; b < n_blocks_; ++b) {
        auto [start, end] = block_ranges_[b];
        Size bs = end - start;

        std::vector<Real> block_data(n_pairs * bs);

        // Extract: full_tensor is (n_orb^2, n_aux) row-major
        // block stores (n_orb^2, bs) for aux range [start, end)
        for (Size ab = 0; ab < n_pairs; ++ab) {
            for (Size p = 0; p < bs; ++p) {
                block_data[ab * bs + p] = full_tensor[ab * n_aux_ + start + p];
            }
        }

        store_block(b, block_data);
    }
}

std::vector<Real> ThreeCenterBlockStorage::reconstruct_full_tensor() const {
    Size n_pairs = n_orb_ * n_orb_;
    std::vector<Real> full(n_pairs * n_aux_, 0.0);

    for (Size b = 0; b < n_blocks_; ++b) {
        if (!is_block_loaded(b)) {
            throw InvalidStateException(
                "Cannot reconstruct: block " + std::to_string(b) + " not loaded");
        }

        auto [start, end] = block_ranges_[b];
        Size bs = end - start;
        auto data = get_block(b);

        for (Size ab = 0; ab < n_pairs; ++ab) {
            for (Size p = 0; p < bs; ++p) {
                full[ab * n_aux_ + start + p] = data[ab * bs + p];
            }
        }
    }

    return full;
}

// ============================================================================
// Internal
// ============================================================================

void ThreeCenterBlockStorage::ensure_memory(Size required_bytes) const {
    while (memory_used_ + required_bytes > memory_limit_ && !lru_order_.empty()) {
        Size victim = lru_order_.front();
        lru_order_.pop_front();
        lru_map_.erase(victim);

        auto it = blocks_.find(victim);
        if (it != blocks_.end()) {
            memory_used_ -= it->second.size() * sizeof(Real);
            blocks_.erase(it);
        }
    }
}

void ThreeCenterBlockStorage::touch_lru(Size block_idx) const {
    auto it = lru_map_.find(block_idx);
    if (it != lru_map_.end()) {
        lru_order_.erase(it->second);
        lru_order_.push_back(block_idx);
        it->second = std::prev(lru_order_.end());
    }
}

}  // namespace libaccint::df
