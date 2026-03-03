// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file spherical_transform.cuh
/// @brief CUDA kernels for Cartesian-to-spherical transformation

#ifdef LIBACCINT_CUDA_ENABLED

#include <cuda_runtime.h>

namespace libaccint::device::cuda {

/// Maximum angular momentum for GPU constant memory matrices (stable G-only support)
constexpr int GPU_MAX_L_TRANSFORM = 4;

/**
 * @brief Initialize spherical transformation matrices in GPU constant memory
 *
 * Must be called before any GPU spherical transformations.
 * Thread-safe (uses std::call_once internally).
 */
void initialize_spherical_transform_matrices();

/**
 * @brief Transform 1D Cartesian integrals to spherical on GPU
 *
 * @param L Angular momentum (supported range: 0..GPU_MAX_L_TRANSFORM)
 * @param n_batches Number of integral batches to transform
 * @param d_cartesian Device pointer to Cartesian integrals [n_batches x n_cart(L)]
 * @param d_spherical Device pointer to output spherical integrals [n_batches x n_sph(L)]
 * @param stream CUDA stream for async execution
 */
void transform_1d_gpu(int L, int n_batches,
                      const double* d_cartesian, double* d_spherical,
                      cudaStream_t stream = 0);

/**
 * @brief Transform 2D Cartesian integral blocks to spherical on GPU
 *
 * @param La, Lb Angular momenta (supported range: 0..GPU_MAX_L_TRANSFORM)
 * @param n_batches Number of shell-pair blocks to transform
 * @param d_cartesian Device pointer to Cartesian blocks [n_batches x n_cart_a x n_cart_b]
 * @param d_spherical Device pointer to spherical blocks [n_batches x n_sph_a x n_sph_b]
 * @param stream CUDA stream
 */
void transform_2d_gpu(int La, int Lb, int n_batches,
                      const double* d_cartesian, double* d_spherical,
                      cudaStream_t stream = 0);

/**
 * @brief Transform 4D ERI quartets from Cartesian to spherical on GPU
 *
 * @param La, Lb, Lc, Ld Angular momenta of the quartet (supported range: 0..GPU_MAX_L_TRANSFORM)
 * @param n_batches Number of quartets to transform
 * @param d_cartesian Device pointer to Cartesian ERIs
 * @param d_spherical Device pointer to spherical ERIs
 * @param d_work Device working memory
 * @param stream CUDA stream
 */
void transform_4d_gpu(int La, int Lb, int Lc, int Ld, int n_batches,
                      const double* d_cartesian, double* d_spherical,
                      double* d_work, cudaStream_t stream = 0);

/**
 * @brief Get required device memory size for 4D transformation work buffer
 *
 * @param La, Lb, Lc, Ld Angular momenta
 * @param n_batches Number of batches
 * @return Required work buffer size in bytes
 */
size_t work_size_4d_gpu(int La, int Lb, int Lc, int Ld, int n_batches);

/**
 * @brief GPU Spherical Transformer class
 *
 * Manages GPU resources for batch spherical transformations.
 */
class GPUSphericalTransformer {
public:
    /**
     * @brief Construct with maximum angular momentum and batch size
     * @param max_am Maximum angular momentum to support
     * @param max_batch_size Maximum number of batches per transform call
     */
    GPUSphericalTransformer(int max_am, int max_batch_size);

    ~GPUSphericalTransformer();

    // Non-copyable
    GPUSphericalTransformer(const GPUSphericalTransformer&) = delete;
    GPUSphericalTransformer& operator=(const GPUSphericalTransformer&) = delete;

    // Movable
    GPUSphericalTransformer(GPUSphericalTransformer&&) noexcept;
    GPUSphericalTransformer& operator=(GPUSphericalTransformer&&) noexcept;

    /**
     * @brief Transform 1E integral batch
     */
    void transform_1e(int La, int Lb, int n_batches,
                      const double* d_cartesian, double* d_spherical,
                      cudaStream_t stream = 0);

    /**
     * @brief Transform 2E integral batch
     */
    void transform_2e(int La, int Lb, int Lc, int Ld, int n_batches,
                      const double* d_cartesian, double* d_spherical,
                      cudaStream_t stream = 0);

private:
    int max_am_;
    int max_batch_size_;
    double* d_work_;  ///< Device working memory
};

} // namespace libaccint::device::cuda

#endif // LIBACCINT_CUDA_ENABLED
