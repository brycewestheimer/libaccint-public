// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file eri_scatter_kernel.cu
/// @brief CUDA kernel to scatter flat ERI output into a 4D tensor on device
///
/// The ERI kernel writes integrals into a flat buffer indexed as:
///   flat[global_id * n_cart + cart_idx]
/// where global_id = ia * nB * nC * nD + ib * nC * nD + ic * nD + id
/// and cart_idx enumerates (a, b, c, d) Cartesian components.
///
/// This scatter kernel reads those values and atomically accumulates them
/// into a dense 4D tensor indexed as:
///   tensor[(fi+a) * nbf^3 + (fj+b) * nbf^2 + (fk+c) * nbf + (fl+d)]
/// using the d_function_offsets arrays to map shell → basis function index.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/kernels/eri_scatter_cuda.hpp>

#include <cuda_runtime.h>

namespace libaccint::kernels::cuda {

/// @brief Scatter kernel: one thread per element in the flat ERI buffer
///
/// Total elements = n_quartets * na * nb * nc * nd
/// Thread decodes which (ia,ib,ic,id,a,b,c,d) it corresponds to, looks up
/// function offsets, and does an atomicAdd into the 4D tensor.
__global__ void eri_scatter_kernel(
    const double* __restrict__ d_eri_flat,
    const int* __restrict__ d_func_a,
    const int* __restrict__ d_func_b,
    const int* __restrict__ d_func_c,
    const int* __restrict__ d_func_d,
    double* __restrict__ d_tensor,
    int nA, int nB, int nC, int nD,
    int na, int nb, int nc, int nd,
    int nbf)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_cart = na * nb * nc * nd;
    const int total_elements = nA * nB * nC * nD * n_cart;

    if (tid >= total_elements) return;

    // Decode: tid = quartet_id * n_cart + cart_id
    const int cart_id = tid % n_cart;
    const int quartet_id = tid / n_cart;

    // Decode shell indices from quartet_id
    // quartet_id = ia * nB*nC*nD + ib * nC*nD + ic * nD + id
    int rem = quartet_id;
    const int id = rem % nD; rem /= nD;
    const int ic = rem % nC; rem /= nC;
    const int ib = rem % nB; rem /= nB;
    const int ia = rem;

    // Decode Cartesian indices from cart_id
    // cart_id = a * nb*nc*nd + b * nc*nd + c * nd + d
    rem = cart_id;
    const int d_idx = rem % nd; rem /= nd;
    const int c_idx = rem % nc; rem /= nc;
    const int b_idx = rem % nb; rem /= nb;
    const int a_idx = rem;

    // Look up basis function offsets
    const int fi = d_func_a[ia];
    const int fj = d_func_b[ib];
    const int fk = d_func_c[ic];
    const int fl = d_func_d[id];

    // Compute destination index in 4D tensor
    const long long nbf_l = static_cast<long long>(nbf);
    const long long dst = static_cast<long long>(fi + a_idx) * nbf_l * nbf_l * nbf_l +
                          static_cast<long long>(fj + b_idx) * nbf_l * nbf_l +
                          static_cast<long long>(fk + c_idx) * nbf_l +
                          static_cast<long long>(fl + d_idx);

    // Read value from flat buffer and accumulate into tensor
    const double val = d_eri_flat[tid];
    atomicAdd(&d_tensor[dst], val);
}

void launch_eri_scatter_kernel(
    const double* d_eri_flat,
    const basis::ShellSetQuartetDeviceData& quartet,
    double* d_tensor,
    int nbf,
    cudaStream_t stream)
{
    const int nA = quartet.a.n_shells;
    const int nB = quartet.b.n_shells;
    const int nC = quartet.c.n_shells;
    const int nD = quartet.d.n_shells;

    const int na = quartet.a.n_functions_per_shell;
    const int nb_f = quartet.b.n_functions_per_shell;
    const int nc = quartet.c.n_functions_per_shell;
    const int nd = quartet.d.n_functions_per_shell;

    const int total_elements = nA * nB * nC * nD * na * nb_f * nc * nd;
    if (total_elements == 0) return;

    constexpr int block_size = 256;
    const int n_blocks = (total_elements + block_size - 1) / block_size;

    eri_scatter_kernel<<<n_blocks, block_size, 0, stream>>>(
        d_eri_flat,
        quartet.a.d_function_offsets,
        quartet.b.d_function_offsets,
        quartet.c.d_function_offsets,
        quartet.d.d_function_offsets,
        d_tensor,
        nA, nB, nC, nD,
        na, nb_f, nc, nd,
        nbf);
}

// =============================================================================
// SoA Scatter Variant
// =============================================================================

/// @brief SoA scatter kernel: reads ERI data in SoA layout
///
/// SoA layout: d_eri_soa[component * n_quartets + quartet_id]
/// where component = a * nb*nc*nd + b * nc*nd + c * nd + d
///
/// Total elements = n_quartets * na * nb * nc * nd (same as AoS)
/// Each thread processes one element, but reads from SoA-transposed layout.
__global__ void eri_scatter_kernel_soa(
    const double* __restrict__ d_eri_soa,
    const int* __restrict__ d_func_a,
    const int* __restrict__ d_func_b,
    const int* __restrict__ d_func_c,
    const int* __restrict__ d_func_d,
    double* __restrict__ d_tensor,
    int nA, int nB, int nC, int nD,
    int na, int nb, int nc, int nd,
    int nbf)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_cart = na * nb * nc * nd;
    const int n_quartets = nA * nB * nC * nD;
    const int total_elements = n_quartets * n_cart;

    if (tid >= total_elements) return;

    // SoA decode: tid = cart_id * n_quartets + quartet_id
    const int quartet_id = tid % n_quartets;
    const int cart_id = tid / n_quartets;

    // Decode shell indices from quartet_id
    int rem = quartet_id;
    const int id = rem % nD; rem /= nD;
    const int ic = rem % nC; rem /= nC;
    const int ib = rem % nB; rem /= nB;
    const int ia = rem;

    // Decode Cartesian indices from cart_id
    rem = cart_id;
    const int d_idx = rem % nd; rem /= nd;
    const int c_idx = rem % nc; rem /= nc;
    const int b_idx = rem % nb; rem /= nb;
    const int a_idx = rem;

    // Look up basis function offsets
    const int fi = d_func_a[ia];
    const int fj = d_func_b[ib];
    const int fk = d_func_c[ic];
    const int fl = d_func_d[id];

    // Compute destination index in 4D tensor
    const long long nbf_l = static_cast<long long>(nbf);
    const long long dst = static_cast<long long>(fi + a_idx) * nbf_l * nbf_l * nbf_l +
                          static_cast<long long>(fj + b_idx) * nbf_l * nbf_l +
                          static_cast<long long>(fk + c_idx) * nbf_l +
                          static_cast<long long>(fl + d_idx);

    // Read value from SoA buffer: d_eri_soa[cart_id * n_quartets + quartet_id]
    const double val = d_eri_soa[tid];
    atomicAdd(&d_tensor[dst], val);
}

void launch_eri_scatter_kernel_soa(
    const double* d_eri_soa,
    const basis::ShellSetQuartetDeviceData& quartet,
    double* d_tensor,
    int nbf,
    cudaStream_t stream)
{
    const int nA = quartet.a.n_shells;
    const int nB = quartet.b.n_shells;
    const int nC = quartet.c.n_shells;
    const int nD = quartet.d.n_shells;

    const int na = quartet.a.n_functions_per_shell;
    const int nb_f = quartet.b.n_functions_per_shell;
    const int nc = quartet.c.n_functions_per_shell;
    const int nd = quartet.d.n_functions_per_shell;

    const int total_elements = nA * nB * nC * nD * na * nb_f * nc * nd;
    if (total_elements == 0) return;

    constexpr int block_size = 256;
    const int n_blocks = (total_elements + block_size - 1) / block_size;

    eri_scatter_kernel_soa<<<n_blocks, block_size, 0, stream>>>(
        d_eri_soa,
        quartet.a.d_function_offsets,
        quartet.b.d_function_offsets,
        quartet.c.d_function_offsets,
        quartet.d.d_function_offsets,
        d_tensor,
        nA, nB, nC, nD,
        na, nb_f, nc, nd,
        nbf);
}

}  // namespace libaccint::kernels::cuda

#endif  // LIBACCINT_USE_CUDA
