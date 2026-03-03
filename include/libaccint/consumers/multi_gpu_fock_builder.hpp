// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file multi_gpu_fock_builder.hpp
/// @brief Multi-GPU Fock matrix builder with parallel accumulation
///
/// Extends the single-GPU FockBuilder to support accumulation across
/// multiple GPU devices with efficient result reduction.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/consumers/fock_builder_gpu.hpp>
#include <libaccint/device/device_manager.hpp>
#include <libaccint/device/multi_device_memory.hpp>
#include <libaccint/core/types.hpp>

#include <cuda_runtime.h>
#include <memory>
#include <span>
#include <vector>

namespace libaccint::consumers {

/// @brief Multi-GPU Fock matrix builder
///
/// MultiGPUFockBuilder maintains per-device Coulomb (J) and exchange (K)
/// accumulation buffers and provides efficient reduction to produce
/// final matrices.
///
/// Usage:
/// @code
///   MultiGPUFockBuilder fock(nbf, {0, 1, 2, 3});
///   fock.set_density(D.data(), nbf);
///   
///   // Use with MultiGPUEngine
///   multi_gpu_engine.compute_all_eri(fock);
///   
///   // Get combined results
///   auto J = fock.get_coulomb_matrix();
///   auto K = fock.get_exchange_matrix();
/// @endcode
class MultiGPUFockBuilder {
public:
    /// @brief Construct a multi-GPU Fock builder
    /// @param nbf Number of basis functions
    /// @param device_ids GPU device IDs to use
    explicit MultiGPUFockBuilder(Size nbf, const std::vector<int>& device_ids);
    
    /// @brief Destructor
    ~MultiGPUFockBuilder();
    
    // Move-only
    MultiGPUFockBuilder(MultiGPUFockBuilder&&) noexcept;
    MultiGPUFockBuilder& operator=(MultiGPUFockBuilder&&) noexcept;
    MultiGPUFockBuilder(const MultiGPUFockBuilder&) = delete;
    MultiGPUFockBuilder& operator=(const MultiGPUFockBuilder&) = delete;
    
    // =========================================================================
    // Configuration
    // =========================================================================
    
    /// @brief Set the density matrix
    /// @param D Pointer to nbf x nbf density matrix (row-major, host memory)
    /// @param nbf Number of basis functions
    /// @note Density is replicated to all devices
    void set_density(const Real* D, Size nbf);
    
    /// @brief Reset J and K matrices on all devices
    void reset();
    
    // =========================================================================
    // Accumulation Interface
    // =========================================================================
    
    /// @brief Accumulate from host-side buffer on specified device
    ///
    /// Called by multi-GPU engine to accumulate integrals computed on
    /// a specific device.
    ///
    /// @param device_id Device to accumulate on
    /// @param buffer Two-electron integral buffer
    /// @param fa, fb, fc, fd Basis function offsets for shells
    /// @param na, nb, nc, nd Number of functions in each shell
    void accumulate(int device_id,
                    const TwoElectronBuffer<0>& buffer,
                    Index fa, Index fb, Index fc, Index fd,
                    int na, int nb, int nc, int nd);

    /// @brief Accumulate from host-side flat ERI data on a specific device
    ///
    /// Used by the alpha multi-GPU/MPI path after copying a device batch back
    /// to host memory.
    void accumulate(int device_id,
                    const double* flat_eri,
                    const ShellSetQuartet& quartet);
    
    /// @brief Accumulate from device-side ERI batch
    ///
    /// For true device-side accumulation without host transfer.
    ///
    /// @param device_id Device where ERIs reside
    /// @param d_eri_batch Device pointer to ERI batch
    /// @param quartet_data Device data describing quartet structure
    /// @param nbf Number of basis functions
    void accumulate_device(int device_id,
                            const double* d_eri_batch,
                            const basis::ShellSetQuartetDeviceData& quartet_data,
                            Size nbf);
    
    // =========================================================================
    // Result Retrieval
    // =========================================================================
    
    /// @brief Reduce per-device J matrices and return combined result
    /// @return Combined J matrix (nbf x nbf, row-major)
    [[nodiscard]] std::vector<Real> get_coulomb_matrix();
    
    /// @brief Reduce per-device K matrices and return combined result
    /// @return Combined K matrix (nbf x nbf, row-major)
    [[nodiscard]] std::vector<Real> get_exchange_matrix();
    
    /// @brief Compute and return the Fock matrix
    /// @param H_core Core Hamiltonian (nbf x nbf, row-major)
    /// @param exchange_fraction Fraction of exact exchange (1.0 for RHF)
    /// @return Fock matrix F = H_core + J - exchange_fraction * K
    [[nodiscard]] std::vector<Real> get_fock_matrix(
        std::span<const Real> H_core,
        Real exchange_fraction = 1.0);
    
    // =========================================================================
    // Accessors
    // =========================================================================
    
    /// @brief Get number of basis functions
    [[nodiscard]] Size nbf() const noexcept { return nbf_; }
    
    /// @brief Get device IDs
    [[nodiscard]] const std::vector<int>& device_ids() const noexcept {
        return device_ids_;
    }
    
    /// @brief Get the per-device Fock builder
    [[nodiscard]] GpuFockBuilder& device_builder(int device_id);
    [[nodiscard]] const GpuFockBuilder& device_builder(int device_id) const;
    
    /// @brief Synchronize all devices
    void synchronize();

private:
    void allocate_device_resources();
    void free_device_resources();
    
    /// @brief Reduce all per-device matrices into the first device
    void reduce_to_primary();
    
    Size nbf_;
    std::vector<int> device_ids_;
    
    // Per-device Fock builders
    std::vector<std::unique_ptr<GpuFockBuilder>> device_builders_;
    
    // Reduction workspace (on primary device)
    double* d_reduction_workspace_ = nullptr;
    
    // State tracking
    bool reduction_done_ = false;
};

}  // namespace libaccint::consumers

#endif  // LIBACCINT_USE_CUDA
