// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file gpu_compat.hpp
/// @brief GPU backend compatibility layer for CUDA
///
/// This header provides a unified interface for GPU programming.
/// It defines macros and type aliases that abstract CUDA specifics.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif

namespace libaccint::device {

// ============================================================================
// Platform Detection
// ============================================================================

#if LIBACCINT_USE_CUDA
    #define LIBACCINT_GPU_PLATFORM_CUDA 1
    #define LIBACCINT_GPU_PLATFORM_NAME "CUDA"
#else
    #define LIBACCINT_GPU_PLATFORM_CUDA 0
    #define LIBACCINT_GPU_PLATFORM_NAME "NONE"
#endif

// ============================================================================
// Type Aliases
// ============================================================================

#if LIBACCINT_USE_CUDA
    using gpuStream_t = cudaStream_t;
    using gpuEvent_t = cudaEvent_t;
    using gpuError_t = cudaError_t;
    using gpuDeviceProp_t = cudaDeviceProp;

    constexpr gpuError_t gpuSuccess = cudaSuccess;
    constexpr gpuError_t gpuErrorNotReady = cudaErrorNotReady;
#endif

// ============================================================================
// API Wrapper Macros
// ============================================================================

#if LIBACCINT_USE_CUDA

    // Memory management
    #define gpuMalloc cudaMalloc
    #define gpuFree cudaFree
    #define gpuMallocHost cudaMallocHost
    #define gpuFreeHost cudaFreeHost
    #define gpuMemcpy cudaMemcpy
    #define gpuMemcpyAsync cudaMemcpyAsync
    #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
    #define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
    #define gpuMemset cudaMemset
    #define gpuMemsetAsync cudaMemsetAsync

    // Stream management
    #define gpuStreamCreate cudaStreamCreate
    #define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
    #define gpuStreamDestroy cudaStreamDestroy
    #define gpuStreamSynchronize cudaStreamSynchronize
    #define gpuStreamQuery cudaStreamQuery
    #define gpuStreamWaitEvent cudaStreamWaitEvent
    #define gpuStreamNonBlocking cudaStreamNonBlocking

    // Event management
    #define gpuEventCreate cudaEventCreate
    #define gpuEventCreateWithFlags cudaEventCreateWithFlags
    #define gpuEventDestroy cudaEventDestroy
    #define gpuEventRecord cudaEventRecord
    #define gpuEventSynchronize cudaEventSynchronize
    #define gpuEventQuery cudaEventQuery
    #define gpuEventElapsedTime cudaEventElapsedTime
    #define gpuEventDefault cudaEventDefault
    #define gpuEventDisableTiming cudaEventDisableTiming

    // Device management
    #define gpuGetDeviceCount cudaGetDeviceCount
    #define gpuSetDevice cudaSetDevice
    #define gpuGetDevice cudaGetDevice
    #define gpuGetDeviceProperties cudaGetDeviceProperties
    #define gpuDeviceSynchronize cudaDeviceSynchronize

    // Error handling
    #define gpuGetLastError cudaGetLastError
    #define gpuPeekAtLastError cudaPeekAtLastError
    #define gpuGetErrorString cudaGetErrorString
    #define gpuGetErrorName cudaGetErrorName

#endif

// ============================================================================
// Kernel Launch Macros
// ============================================================================

#if LIBACCINT_USE_CUDA
    #define GPU_LAUNCH_BOUNDS(max_threads) __launch_bounds__(max_threads)
    #define GPU_LAUNCH_BOUNDS2(max_threads, min_blocks) __launch_bounds__(max_threads, min_blocks)
#else
    #define GPU_LAUNCH_BOUNDS(max_threads)
    #define GPU_LAUNCH_BOUNDS2(max_threads, min_blocks)
#endif

// ============================================================================
// Device Function Qualifiers
// ============================================================================

#if LIBACCINT_USE_CUDA
    #define GPU_DEVICE __device__
    #define GPU_HOST __host__
    #define GPU_GLOBAL __global__
    #define GPU_CONSTANT __constant__
    #define GPU_SHARED __shared__
    #define GPU_FORCEINLINE __forceinline__
    #define GPU_HOST_DEVICE __host__ __device__
#else
    #define GPU_DEVICE
    #define GPU_HOST
    #define GPU_GLOBAL
    #define GPU_CONSTANT
    #define GPU_SHARED
    #define GPU_FORCEINLINE inline
    #define GPU_HOST_DEVICE
#endif

// ============================================================================
// Math Functions
// ============================================================================

#if LIBACCINT_USE_CUDA
    #define gpu_exp exp
    #define gpu_sqrt sqrt
    #define gpu_rsqrt rsqrt
    #define gpu_pow pow
    #define gpu_fabs fabs
    #define gpu_copysign copysign
    #define gpu_max max
    #define gpu_min min
#endif

// ============================================================================
// Atomic Operations
// ============================================================================

#if LIBACCINT_USE_CUDA
    #define gpu_atomicAdd atomicAdd
    #define gpu_atomicSub atomicSub
    #define gpu_atomicExch atomicExch
    #define gpu_atomicMin atomicMin
    #define gpu_atomicMax atomicMax
    #define gpu_atomicCAS atomicCAS
    #define gpu_atomicAnd atomicAnd
    #define gpu_atomicOr atomicOr
    #define gpu_atomicXor atomicXor
#endif

// ============================================================================
// Thread Fence Operations
// ============================================================================

#if LIBACCINT_USE_CUDA
    #define gpu_threadfence __threadfence
    #define gpu_threadfence_block __threadfence_block
    #define gpu_threadfence_system __threadfence_system
#endif

// ============================================================================
// Peer Access
// ============================================================================

#if LIBACCINT_USE_CUDA
    #define gpuDeviceCanAccessPeer cudaDeviceCanAccessPeer
    #define gpuDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
    #define gpuDeviceDisablePeerAccess cudaDeviceDisablePeerAccess
    #define gpuMemcpyPeer cudaMemcpyPeer
    #define gpuMemcpyPeerAsync cudaMemcpyPeerAsync
#endif

// ============================================================================
// Warp Operations
// ============================================================================

#if LIBACCINT_USE_CUDA
    #define GPU_WARP_SIZE 32
    #define gpu_syncthreads __syncthreads
    #define gpu_syncwarp __syncwarp
    #define gpu_shfl_sync __shfl_sync
    #define gpu_shfl_down_sync __shfl_down_sync
    #define gpu_shfl_up_sync __shfl_up_sync
    #define gpu_shfl_xor_sync __shfl_xor_sync
    #define gpu_ballot_sync __ballot_sync
    #define gpu_all_sync __all_sync
    #define gpu_any_sync __any_sync
    #define gpu_activemask __activemask
#endif

}  // namespace libaccint::device
