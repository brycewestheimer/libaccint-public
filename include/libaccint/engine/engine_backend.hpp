// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file engine_backend.hpp
/// @brief C++20 concepts for engine backends and integral consumers

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/operators/operator.hpp>

#include <concepts>
#include <vector>

namespace libaccint {

/// @brief Concept for integral consumer types
///
/// Consumers are used in the compute-and-consume pattern to fuse integral
/// computation with consumption (e.g., Fock matrix build). This avoids
/// storing all integrals in memory.
///
/// A consumer must provide an `accumulate` method that receives computed
/// integrals along with their shell indices and function counts.
template<typename T>
concept IntegralConsumer = requires(T& consumer,
                                     const TwoElectronBuffer<0>& buffer,
                                     Index fa, Index fb, Index fc, Index fd,
                                     int na, int nb, int nc, int nd) {
    { consumer.accumulate(buffer, fa, fb, fc, fd, na, nb, nc, nd) } -> std::same_as<void>;
};

/// @brief Concept for engine backend types
///
/// An EngineBackend provides the core integral computation capabilities.
/// Both CpuEngine and CudaEngine satisfy this concept, enabling compile-time
/// polymorphism without virtual function overhead.
///
/// Required methods:
/// - basis(): Returns a const reference to the BasisSet
/// - compute_1e_shell_pair(): Compute one-electron integrals for a shell pair
/// - compute_2e_shell_quartet(): Compute two-electron integrals for a shell quartet
/// - compute_shell_set_pair(): Compute one-electron integrals for a ShellSetPair
template<typename T>
concept EngineBackend = requires(T& engine,
                                  const Operator& op,
                                  const Shell& shell,
                                  const ShellSetPair& pair,
                                  OneElectronBuffer<0>& buffer_1e,
                                  TwoElectronBuffer<0>& buffer_2e,
                                  std::vector<Real>& result) {
    // Access to the basis set
    { engine.basis() } -> std::same_as<const BasisSet&>;

    // One-electron shell pair computation
    { engine.compute_1e_shell_pair(op, shell, shell, buffer_1e) } -> std::same_as<void>;

    // Two-electron shell quartet computation
    { engine.compute_2e_shell_quartet(op, shell, shell, shell, shell, buffer_2e) } -> std::same_as<void>;

    // ShellSetPair-based one-electron computation
    { engine.compute_shell_set_pair(op, pair, result) } -> std::same_as<void>;
};

}  // namespace libaccint
