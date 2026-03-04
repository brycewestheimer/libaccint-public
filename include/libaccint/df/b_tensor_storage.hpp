// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file b_tensor_storage.hpp
/// @brief Storage abstraction for DF B tensor access
///
/// Defines a concept-based interface for B tensor storage backends.
/// Both flat-vector (monolithic) and block-partitioned storage satisfy
/// this interface, allowing DFFockBuilder to work with either backend.

#include <libaccint/core/types.hpp>

#include <span>
#include <utility>
#include <vector>

namespace libaccint::df {

/// @brief Concept for B tensor storage backends
///
/// A BTensorStorage provides block-oriented access to the B tensor.
/// The tensor is logically partitioned along the auxiliary index P
/// into contiguous blocks. Each block stores the orbital-pair data
/// for a range of auxiliary functions.
///
/// Block data layout: For a block covering auxiliary range [P_start, P_end),
/// the data is stored as a flat array of size n_orb^2 * block_size, where
/// data[(a * n_orb + b) * block_size + (P - P_start)] gives B_ab^P.
template <typename S>
concept BTensorStorage = requires(const S s, Size block_idx) {
    /// Number of blocks the tensor is partitioned into
    { s.n_blocks() } -> std::same_as<Size>;

    /// Auxiliary function range [start, end) for the given block
    { s.block_range(block_idx) } -> std::same_as<std::pair<Size, Size>>;

    /// Read-only access to block data
    { s.get_block(block_idx) } -> std::same_as<std::span<const Real>>;
};

/// @brief Flat-vector B tensor storage (single block wrapping a vector)
///
/// Wraps an existing flat vector as a single-block BTensorStorage.
/// This is the default storage for small systems where the full B tensor
/// fits comfortably in memory.
class FlatBTensorStorage {
public:
    /// @brief Construct from an existing flat B tensor
    ///
    /// @param data B tensor in (n_orb^2 x n_aux) row-major layout
    /// @param n_aux Number of auxiliary functions
    FlatBTensorStorage(std::span<const Real> data, Size n_aux)
        : data_(data), n_aux_(n_aux) {}

    /// @brief Number of blocks (always 1 for flat storage)
    [[nodiscard]] Size n_blocks() const noexcept { return 1; }

    /// @brief Block range (always [0, n_aux))
    [[nodiscard]] std::pair<Size, Size> block_range([[maybe_unused]] Size block_idx) const {
        return {0, n_aux_};
    }

    /// @brief Get the full tensor data as a single block
    [[nodiscard]] std::span<const Real> get_block([[maybe_unused]] Size block_idx) const {
        return data_;
    }

private:
    std::span<const Real> data_;
    Size n_aux_;
};

// Verify FlatBTensorStorage satisfies the concept
static_assert(BTensorStorage<FlatBTensorStorage>);

}  // namespace libaccint::df
