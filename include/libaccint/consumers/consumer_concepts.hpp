// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file consumer_concepts.hpp
/// @brief C++20 concepts for compile-time consumer interface validation
///
/// Defines concepts that constrain the template parameter `Consumer` in
/// engine template methods (compute_shell_set_quartet, compute_and_consume,
/// etc.).  A static_assert against the appropriate concept gives an
/// immediate, readable diagnostic when a user-defined consumer is missing
/// a required method or has an incompatible signature.
///
/// Concepts defined here:
///   - EriConsumer:            must have `accumulate()`
///   - SymmetryAwareConsumer:  additionally has `accumulate_symmetric()`
///   - ParallelConsumer:       additionally has `prepare_parallel()` / `finalize_parallel()`

#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/core/types.hpp>

#include <concepts>

namespace libaccint {

/// @brief Core consumer concept: must have accumulate()
///
/// Any type passed as the Consumer template parameter to
/// CpuEngine::compute_shell_set_quartet, CudaEngine::compute_shell_set_quartet,
/// or Engine::compute_and_consume must satisfy this concept.
template<typename C>
concept TwoElectronConsumer = requires(C& c, const TwoElectronBuffer<0>& buf,
                               Index fa, Index fb, Index fc, Index fd,
                               int na, int nb, int nc, int nd) {
    c.accumulate(buf, fa, fb, fc, fd, na, nb, nc, nd);
};

/// @brief Backward-compatible alias for TwoElectronConsumer
template<typename C>
concept EriConsumer = TwoElectronConsumer<C>;

/// @brief Symmetry-aware consumer: additionally has accumulate_symmetric()
///
/// Consumers satisfying this concept can be used with canonical 8-fold
/// symmetry iteration, where each unique shell quartet is computed once
/// and the consumer scatters contributions to all permutation-equivalent
/// matrix slots.
template<typename C>
concept SymmetryAwareConsumer = TwoElectronConsumer<C> &&
    requires(C& c, const TwoElectronBuffer<0>& buf,
             Index fa, Index fb, Index fc, Index fd,
             int na, int nb, int nc, int nd,
             bool ij_same, bool kl_same, bool braket_same) {
        c.accumulate_symmetric(buf, fa, fb, fc, fd, na, nb, nc, nd,
                               ij_same, kl_same, braket_same);
    };

/// @brief Parallel-capable consumer: has prepare_parallel() and finalize_parallel()
///
/// Consumers satisfying this concept can be used with OpenMP-parallelized
/// compute paths.  prepare_parallel() is called before the parallel region
/// (to allocate thread-local buffers, etc.) and finalize_parallel() is
/// called after (to reduce thread-local results).
template<typename C>
concept ParallelConsumer = TwoElectronConsumer<C> &&
    requires(C& c, int n_threads) {
        c.prepare_parallel(n_threads);
        c.finalize_parallel();
    };

}  // namespace libaccint
