// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file fock_builder_gpu.hpp
/// @brief GPU-accelerated FockBuilder for two-electron integral accumulation
///
/// Implements the compute-and-consume pattern on GPU by accumulating Coulomb (J)
/// and exchange (K) matrix contributions using atomic operations on device memory.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/basis/device_data.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/memory/device_memory.hpp>

#include <cuda_runtime.h>
#include <span>
#include <vector>

namespace libaccint::consumers {

/// @brief GPU-accelerated builder for Coulomb (J) and exchange (K) matrices
///
/// GpuFockBuilder accumulates J and K contributions using device memory and
/// atomic operations:
///   J_mu_nu     += sum_lambda,sigma (mu nu | lambda sigma) * D_lambda_sigma
///   K_mu_lambda += sum_nu,sigma     (mu nu | lambda sigma) * D_nu_sigma
///
/// Usage:
/// @code
///   GpuFockBuilder fock(nbf);
///   fock.set_density(D.data(), nbf);
///
///   // Use with CudaEngine::compute_and_consume_eri
///   cuda_engine.compute_and_consume_eri(fock);
///
///   // Download results
///   auto J = fock.get_coulomb_matrix();
///   auto K = fock.get_exchange_matrix();
/// @endcode
class GpuFockBuilder {
public:
    /// @brief Construct a GpuFockBuilder for a basis with nbf functions
    /// @param nbf Number of basis functions
    /// @param stream CUDA stream for operations (default: null stream)
    explicit GpuFockBuilder(Size nbf, cudaStream_t stream = nullptr);

    /// @brief Destructor - frees device memory
    ~GpuFockBuilder();

    // Move-only semantics
    GpuFockBuilder(GpuFockBuilder&& other) noexcept;
    GpuFockBuilder& operator=(GpuFockBuilder&& other) noexcept;
    GpuFockBuilder(const GpuFockBuilder&) = delete;
    GpuFockBuilder& operator=(const GpuFockBuilder&) = delete;

    /// @brief Set the density matrix for accumulation
    /// @param D Pointer to row-major nbf x nbf density matrix (host memory)
    /// @param nbf Number of basis functions (must match constructor)
    /// @note The density matrix is uploaded to GPU memory
    void set_density(const Real* D, Size nbf);

    /// @brief Accumulate J and K contributions from a buffer of integrals
    ///
    /// For a shell quartet (a b | c d) with basis function offsets
    /// (fa, fb, fc, fd), distributes all integrals into J and K matrices.
    /// The buffer must already be on the host - this method handles the
    /// transfer and accumulation on GPU.
    ///
    /// @param buffer The computed integrals for this quartet
    /// @param fa Starting basis function index for shell a
    /// @param fb Starting basis function index for shell b
    /// @param fc Starting basis function index for shell c
    /// @param fd Starting basis function index for shell d
    /// @param na Number of functions in shell a
    /// @param nb Number of functions in shell b
    /// @param nc Number of functions in shell c
    /// @param nd Number of functions in shell d
    void accumulate(const TwoElectronBuffer<0>& buffer,
                    Index fa, Index fb, Index fc, Index fd,
                    int na, int nb, int nc, int nd);

    /// @brief Get the Coulomb matrix J (downloads from GPU)
    /// @return Vector containing the nbf x nbf J matrix (row-major)
    [[nodiscard]] std::vector<Real> get_coulomb_matrix() const;

    /// @brief Get the exchange matrix K (downloads from GPU)
    /// @return Vector containing the nbf x nbf K matrix (row-major)
    [[nodiscard]] std::vector<Real> get_exchange_matrix() const;

    /// @brief Compute the Fock matrix F = H_core + J - exchange_fraction * K
    /// @param H_core Core Hamiltonian matrix (row-major, nbf x nbf)
    /// @param exchange_fraction Fraction of exact exchange (1.0 for RHF)
    /// @return Vector containing the Fock matrix (row-major, nbf x nbf)
    [[nodiscard]] std::vector<Real> get_fock_matrix(
        std::span<const Real> H_core,
        Real exchange_fraction = 1.0) const;

    /// @brief Reset J and K matrices to zero
    void reset();

    /// @brief Synchronize all pending GPU operations
    void synchronize();

    /// @brief Get the number of basis functions
    [[nodiscard]] Size nbf() const noexcept { return nbf_; }

    /// @brief Get the CUDA stream used for accumulation and downloads
    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

    /// @brief Get device pointer to J matrix
    [[nodiscard]] double* d_J() noexcept { return d_J_; }

    /// @brief Get device pointer to K matrix
    [[nodiscard]] double* d_K() noexcept { return d_K_; }

    /// @brief Get device pointer to density matrix
    [[nodiscard]] const double* d_D() const noexcept { return d_D_; }

    // =========================================================================
    // Phase 4.5: Device-Side Batched Accumulation
    // =========================================================================

    /// @brief Accumulate J and K from device-side ERI batch
    ///
    /// For Phase 4.5 true batched execution: ERIs remain on device and are
    /// accumulated directly into J/K without host transfer.
    ///
    /// @param d_eri_batch Device pointer to batched ERI output
    /// @param quartet_data Device data describing the shell quartet structure
    /// @param nbf Number of basis functions
    void accumulate_device_eri_batch(
        const double* d_eri_batch,
        const basis::ShellSetQuartetDeviceData& quartet_data,
        Size nbf);

    /// @brief Prepare for parallel execution (no-op for GPU, for API compatibility)
    void prepare_parallel([[maybe_unused]] int n_threads = 0) {}

    /// @brief Finalize parallel execution (no-op for GPU, for API compatibility)
    void finalize_parallel() {}

private:
    void allocate_device_memory();
    void free_device_memory();

    Size nbf_;                     ///< Number of basis functions
    cudaStream_t stream_;          ///< CUDA stream for operations
    double* d_J_{nullptr};         ///< Device Coulomb matrix (nbf x nbf)
    double* d_K_{nullptr};         ///< Device exchange matrix (nbf x nbf)
    double* d_D_{nullptr};         ///< Device density matrix (nbf x nbf)
    double* d_eri_buffer_{nullptr}; ///< Device buffer for ERI upload
    size_t eri_buffer_size_{0};    ///< Current ERI buffer size
};

// =============================================================================
// CUDA Kernel Declarations (implementation in .cu file)
// =============================================================================

namespace detail {

/// @brief Launch kernel to accumulate J and K from ERIs
/// @param d_eri Device pointer to ERI values
/// @param d_D Device pointer to density matrix
/// @param d_J Device pointer to Coulomb matrix
/// @param d_K Device pointer to exchange matrix
/// @param fa Starting index for shell a
/// @param fb Starting index for shell b
/// @param fc Starting index for shell c
/// @param fd Starting index for shell d
/// @param na Number of functions in shell a
/// @param nb Number of functions in shell b
/// @param nc Number of functions in shell c
/// @param nd Number of functions in shell d
/// @param nbf Number of basis functions
/// @param stream CUDA stream
void launch_fock_accumulate_kernel(
    const double* d_eri,
    const double* d_D,
    double* d_J,
    double* d_K,
    int fa, int fb, int fc, int fd,
    int na, int nb, int nc, int nd,
    int nbf,
    cudaStream_t stream);

/// @brief Launch kernel to zero a device matrix
void launch_matrix_zero_kernel(double* d_matrix, size_t size, cudaStream_t stream);

/// @brief Launch batched Fock accumulation kernel (Phase 4.5)
///
/// Accumulates J and K contributions from a batch of ERIs computed for a ShellSetQuartet.
/// Each thread processes one ERI value and uses atomic operations for accumulation.
///
/// @param d_eri Device pointer to batched ERI output
/// @param quartet_data Device data for the ShellSetQuartet
/// @param d_D Device pointer to density matrix
/// @param d_J Device pointer to Coulomb matrix
/// @param d_K Device pointer to exchange matrix
/// @param nbf Number of basis functions
/// @param stream CUDA stream
void launch_fock_accumulate_batch_kernel(
    const double* d_eri,
    const basis::ShellSetQuartetDeviceData& quartet_data,
    const double* d_D,
    double* d_J,
    double* d_K,
    int nbf,
    cudaStream_t stream);

}  // namespace detail

}  // namespace libaccint::consumers

#endif  // LIBACCINT_USE_CUDA
