// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file fock_builder_gpu.cu
/// @brief GPU-accelerated FockBuilder implementation

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/consumers/fock_builder_gpu.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <cuda_runtime.h>
#include <stdexcept>

namespace libaccint::consumers {

// =============================================================================
// Double-Precision AtomicAdd (for architectures < 6.0)
// =============================================================================

#if __CUDA_ARCH__ < 600
/// @brief Double-precision atomicAdd implementation for older GPUs
///
/// Uses atomicCAS (atomic compare-and-swap) to implement double-precision
/// atomic addition on GPUs with compute capability < 6.0 (Maxwell, etc.)
__device__ double atomicAdd_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}
#else
/// @brief For SM 6.0+, use native atomicAdd
__device__ __forceinline__ double atomicAdd_double(double* address, double val) {
    return atomicAdd(address, val);
}
#endif

// =============================================================================
// CUDA Kernels
// =============================================================================

/// @brief Kernel to accumulate J and K contributions from ERIs
///
/// Each thread processes one ERI value (mu nu | lambda sigma) and atomically
/// adds contributions to J and K matrices.
__global__ void fock_accumulate_kernel(
    const double* __restrict__ d_eri,
    const double* __restrict__ d_D,
    double* __restrict__ d_J,
    double* __restrict__ d_K,
    int fa, int fb, int fc, int fd,
    int na, int nb, int nc, int nd,
    int nbf)
{
    // Calculate total number of integrals in this quartet
    const int total = na * nb * nc * nd;

    // Thread index determines which integral
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    // Decode (a, b, c, d) from linear index
    int tmp = tid;
    const int d_idx = tmp % nd; tmp /= nd;
    const int c_idx = tmp % nc; tmp /= nc;
    const int b_idx = tmp % nb; tmp /= nb;
    const int a_idx = tmp;

    // Calculate basis function indices
    const int mu  = fa + a_idx;
    const int nu  = fb + b_idx;
    const int lam = fc + c_idx;
    const int sig = fd + d_idx;

    // Get the ERI value
    const double eri = d_eri[tid];

    // Get density matrix elements
    const double D_lam_sig = d_D[lam * nbf + sig];
    const double D_nu_sig  = d_D[nu * nbf + sig];

    // Coulomb: J_mu_nu += (mu nu | lam sig) * D_lam_sig
    atomicAdd_double(&d_J[mu * nbf + nu], eri * D_lam_sig);

    // Exchange: K_mu_lam += (mu nu | lam sig) * D_nu_sig
    atomicAdd_double(&d_K[mu * nbf + lam], eri * D_nu_sig);
}

/// @brief Kernel to zero a matrix
__global__ void matrix_zero_kernel(double* d_matrix, size_t size)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        d_matrix[tid] = 0.0;
    }
}

/// @brief Batched Fock accumulation kernel (Phase 4.5)
///
/// Each thread processes one ERI value from the batched output. The thread
/// decodes which shell quartet and local indices correspond to the ERI,
/// then atomically accumulates J and K contributions.
///
/// ERI layout: output[quartet_idx * funcs_per_quartet + local_idx]
/// where quartet_idx = i*(nb*nc*nd) + j*(nc*nd) + k*nd + l
/// and   local_idx = a*(nb_funcs*nc_funcs*nd_funcs) + b*(nc_funcs*nd_funcs) + c*nd_funcs + d
__global__ void fock_accumulate_batch_kernel(
    const double* __restrict__ d_eri,
    const int* __restrict__ d_func_offsets_a,
    const int* __restrict__ d_func_offsets_b,
    const int* __restrict__ d_func_offsets_c,
    const int* __restrict__ d_func_offsets_d,
    int n_shells_a, int n_shells_b, int n_shells_c, int n_shells_d,
    int na_funcs, int nb_funcs, int nc_funcs, int nd_funcs,
    const double* __restrict__ d_D,
    double* __restrict__ d_J,
    double* __restrict__ d_K,
    int nbf)
{
    const size_t funcs_per_quartet = static_cast<size_t>(na_funcs) * nb_funcs * nc_funcs * nd_funcs;
    const size_t n_quartets = static_cast<size_t>(n_shells_a) * n_shells_b * n_shells_c * n_shells_d;
    const size_t total_eris = n_quartets * funcs_per_quartet;

    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (tid >= total_eris) return;

    // Decode quartet index and local index
    const size_t quartet_idx = tid / funcs_per_quartet;
    const size_t local_idx = tid % funcs_per_quartet;

    // Decode shell indices from quartet_idx
    const size_t ncd = static_cast<size_t>(n_shells_c) * n_shells_d;
    const size_t nbcd = static_cast<size_t>(n_shells_b) * ncd;

    const int i = static_cast<int>(quartet_idx / nbcd);
    size_t rem = quartet_idx % nbcd;
    const int j = static_cast<int>(rem / ncd);
    rem = rem % ncd;
    const int k = static_cast<int>(rem / n_shells_d);
    const int l = static_cast<int>(rem % n_shells_d);

    // Decode function indices (a, b, c, d) from local_idx
    const size_t ncd_funcs = static_cast<size_t>(nc_funcs) * nd_funcs;
    const size_t nbcd_funcs = static_cast<size_t>(nb_funcs) * ncd_funcs;

    const int a = static_cast<int>(local_idx / nbcd_funcs);
    size_t rem_funcs = local_idx % nbcd_funcs;
    const int b = static_cast<int>(rem_funcs / ncd_funcs);
    rem_funcs = rem_funcs % ncd_funcs;
    const int c = static_cast<int>(rem_funcs / nd_funcs);
    const int d = static_cast<int>(rem_funcs % nd_funcs);

    // Get basis function indices
    const int mu  = d_func_offsets_a[i] + a;
    const int nu  = d_func_offsets_b[j] + b;
    const int lam = d_func_offsets_c[k] + c;
    const int sig = d_func_offsets_d[l] + d;

    // Get ERI value
    const double eri = d_eri[tid];

    // Get density matrix elements
    const double D_lam_sig = d_D[lam * nbf + sig];
    const double D_nu_sig  = d_D[nu * nbf + sig];

    // Coulomb: J_mu_nu += (mu nu | lam sig) * D_lam_sig
    atomicAdd_double(&d_J[mu * nbf + nu], eri * D_lam_sig);

    // Exchange: K_mu_lam += (mu nu | lam sig) * D_nu_sig
    atomicAdd_double(&d_K[mu * nbf + lam], eri * D_nu_sig);
}

/// @brief SoA batched Fock accumulation kernel
///
/// Reads ERIs in SoA layout: d_eri[component * n_quartets + quartet_idx]
/// where component = a * nb*nc*nd + b * nc*nd + c * nd + d
__global__ void fock_accumulate_batch_kernel_soa(
    const double* __restrict__ d_eri,
    const int* __restrict__ d_func_offsets_a,
    const int* __restrict__ d_func_offsets_b,
    const int* __restrict__ d_func_offsets_c,
    const int* __restrict__ d_func_offsets_d,
    int n_shells_a, int n_shells_b, int n_shells_c, int n_shells_d,
    int na_funcs, int nb_funcs, int nc_funcs, int nd_funcs,
    const double* __restrict__ d_D,
    double* __restrict__ d_J,
    double* __restrict__ d_K,
    int nbf)
{
    const size_t funcs_per_quartet = static_cast<size_t>(na_funcs) * nb_funcs * nc_funcs * nd_funcs;
    const size_t n_quartets = static_cast<size_t>(n_shells_a) * n_shells_b * n_shells_c * n_shells_d;
    const size_t total_eris = n_quartets * funcs_per_quartet;

    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (tid >= total_eris) return;

    // SoA decode: tid = component * n_quartets + quartet_idx
    const size_t quartet_idx = tid % n_quartets;
    const size_t local_idx = tid / n_quartets;

    // Decode shell indices from quartet_idx
    const size_t ncd = static_cast<size_t>(n_shells_c) * n_shells_d;
    const size_t nbcd = static_cast<size_t>(n_shells_b) * ncd;

    const int i = static_cast<int>(quartet_idx / nbcd);
    size_t rem = quartet_idx % nbcd;
    const int j = static_cast<int>(rem / ncd);
    rem = rem % ncd;
    const int k = static_cast<int>(rem / n_shells_d);
    const int l = static_cast<int>(rem % n_shells_d);

    // Decode function indices (a, b, c, d) from local_idx (component index)
    const size_t ncd_funcs = static_cast<size_t>(nc_funcs) * nd_funcs;
    const size_t nbcd_funcs = static_cast<size_t>(nb_funcs) * ncd_funcs;

    const int a = static_cast<int>(local_idx / nbcd_funcs);
    size_t rem_funcs = local_idx % nbcd_funcs;
    const int b = static_cast<int>(rem_funcs / ncd_funcs);
    rem_funcs = rem_funcs % ncd_funcs;
    const int c = static_cast<int>(rem_funcs / nd_funcs);
    const int d = static_cast<int>(rem_funcs % nd_funcs);

    // Get basis function indices
    const int mu  = d_func_offsets_a[i] + a;
    const int nu  = d_func_offsets_b[j] + b;
    const int lam = d_func_offsets_c[k] + c;
    const int sig = d_func_offsets_d[l] + d;

    // Get ERI value from SoA layout
    const double eri = d_eri[tid];

    // Get density matrix elements
    const double D_lam_sig = d_D[lam * nbf + sig];
    const double D_nu_sig  = d_D[nu * nbf + sig];

    // Coulomb: J_mu_nu += (mu nu | lam sig) * D_lam_sig
    atomicAdd_double(&d_J[mu * nbf + nu], eri * D_lam_sig);

    // Exchange: K_mu_lam += (mu nu | lam sig) * D_nu_sig
    atomicAdd_double(&d_K[mu * nbf + lam], eri * D_nu_sig);
}

namespace detail {

void launch_fock_accumulate_kernel(
    const double* d_eri,
    const double* d_D,
    double* d_J,
    double* d_K,
    int fa, int fb, int fc, int fd,
    int na, int nb, int nc, int nd,
    int nbf,
    cudaStream_t stream)
{
    const int total = na * nb * nc * nd;
    if (total == 0) return;

    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    fock_accumulate_kernel<<<grid_size, block_size, 0, stream>>>(
        d_eri, d_D, d_J, d_K,
        fa, fb, fc, fd,
        na, nb, nc, nd,
        nbf
    );
}

void launch_matrix_zero_kernel(double* d_matrix, size_t size, cudaStream_t stream)
{
    if (size == 0) return;

    const int block_size = 256;
    const int grid_size = (static_cast<int>(size) + block_size - 1) / block_size;

    matrix_zero_kernel<<<grid_size, block_size, 0, stream>>>(d_matrix, size);
}

void launch_fock_accumulate_batch_kernel(
    const double* d_eri,
    const basis::ShellSetQuartetDeviceData& quartet_data,
    const double* d_D,
    double* d_J,
    double* d_K,
    int nbf,
    cudaStream_t stream)
{
    const int n_shells_a = quartet_data.a.n_shells;
    const int n_shells_b = quartet_data.b.n_shells;
    const int n_shells_c = quartet_data.c.n_shells;
    const int n_shells_d = quartet_data.d.n_shells;

    const int na_funcs = quartet_data.a.n_functions_per_shell;
    const int nb_funcs = quartet_data.b.n_functions_per_shell;
    const int nc_funcs = quartet_data.c.n_functions_per_shell;
    const int nd_funcs = quartet_data.d.n_functions_per_shell;

    const size_t funcs_per_quartet = static_cast<size_t>(na_funcs) * nb_funcs * nc_funcs * nd_funcs;
    const size_t n_quartets = static_cast<size_t>(n_shells_a) * n_shells_b * n_shells_c * n_shells_d;
    const size_t total_eris = n_quartets * funcs_per_quartet;

    if (total_eris == 0) return;

    const int block_size = 256;
    const size_t grid_size = (total_eris + block_size - 1) / block_size;

    fock_accumulate_batch_kernel<<<static_cast<unsigned int>(grid_size), block_size, 0, stream>>>(
        d_eri,
        quartet_data.a.d_function_offsets,
        quartet_data.b.d_function_offsets,
        quartet_data.c.d_function_offsets,
        quartet_data.d.d_function_offsets,
        n_shells_a, n_shells_b, n_shells_c, n_shells_d,
        na_funcs, nb_funcs, nc_funcs, nd_funcs,
        d_D, d_J, d_K, nbf
    );
}

void launch_fock_accumulate_batch_kernel_soa(
    const double* d_eri,
    const basis::ShellSetQuartetDeviceData& quartet_data,
    const double* d_D,
    double* d_J,
    double* d_K,
    int nbf,
    cudaStream_t stream)
{
    const int n_shells_a = quartet_data.a.n_shells;
    const int n_shells_b = quartet_data.b.n_shells;
    const int n_shells_c = quartet_data.c.n_shells;
    const int n_shells_d = quartet_data.d.n_shells;

    const int na_funcs = quartet_data.a.n_functions_per_shell;
    const int nb_funcs = quartet_data.b.n_functions_per_shell;
    const int nc_funcs = quartet_data.c.n_functions_per_shell;
    const int nd_funcs = quartet_data.d.n_functions_per_shell;

    const size_t funcs_per_quartet = static_cast<size_t>(na_funcs) * nb_funcs * nc_funcs * nd_funcs;
    const size_t n_quartets = static_cast<size_t>(n_shells_a) * n_shells_b * n_shells_c * n_shells_d;
    const size_t total_eris = n_quartets * funcs_per_quartet;

    if (total_eris == 0) return;

    const int block_size = 256;
    const size_t grid_size = (total_eris + block_size - 1) / block_size;

    fock_accumulate_batch_kernel_soa<<<static_cast<unsigned int>(grid_size), block_size, 0, stream>>>(
        d_eri,
        quartet_data.a.d_function_offsets,
        quartet_data.b.d_function_offsets,
        quartet_data.c.d_function_offsets,
        quartet_data.d.d_function_offsets,
        n_shells_a, n_shells_b, n_shells_c, n_shells_d,
        na_funcs, nb_funcs, nc_funcs, nd_funcs,
        d_D, d_J, d_K, nbf
    );
}

}  // namespace detail

// =============================================================================
// GpuFockBuilder Implementation
// =============================================================================

GpuFockBuilder::GpuFockBuilder(Size nbf, cudaStream_t stream)
    : nbf_(nbf)
    , stream_(stream)
{
    allocate_device_memory();
    reset();
}

GpuFockBuilder::~GpuFockBuilder() {
    free_device_memory();
}

GpuFockBuilder::GpuFockBuilder(GpuFockBuilder&& other) noexcept
    : nbf_(other.nbf_)
    , stream_(other.stream_)
    , d_J_(other.d_J_)
    , d_K_(other.d_K_)
    , d_D_(other.d_D_)
    , d_eri_buffer_(other.d_eri_buffer_)
    , eri_buffer_size_(other.eri_buffer_size_)
{
    other.d_J_ = nullptr;
    other.d_K_ = nullptr;
    other.d_D_ = nullptr;
    other.d_eri_buffer_ = nullptr;
    other.eri_buffer_size_ = 0;
}

GpuFockBuilder& GpuFockBuilder::operator=(GpuFockBuilder&& other) noexcept {
    if (this != &other) {
        free_device_memory();
        nbf_ = other.nbf_;
        stream_ = other.stream_;
        d_J_ = other.d_J_;
        d_K_ = other.d_K_;
        d_D_ = other.d_D_;
        d_eri_buffer_ = other.d_eri_buffer_;
        eri_buffer_size_ = other.eri_buffer_size_;

        other.d_J_ = nullptr;
        other.d_K_ = nullptr;
        other.d_D_ = nullptr;
        other.d_eri_buffer_ = nullptr;
        other.eri_buffer_size_ = 0;
    }
    return *this;
}

void GpuFockBuilder::allocate_device_memory() {
    const size_t matrix_size = nbf_ * nbf_ * sizeof(double);

    cudaError_t err;

    err = cudaMalloc(&d_J_, matrix_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device J matrix: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_K_, matrix_size);
    if (err != cudaSuccess) {
        cudaFree(d_J_);
        d_J_ = nullptr;
        throw std::runtime_error("Failed to allocate device K matrix: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_D_, matrix_size);
    if (err != cudaSuccess) {
        cudaFree(d_J_);
        cudaFree(d_K_);
        d_J_ = nullptr;
        d_K_ = nullptr;
        throw std::runtime_error("Failed to allocate device D matrix: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void GpuFockBuilder::free_device_memory() {
    if (d_J_) { cudaFree(d_J_); d_J_ = nullptr; }
    if (d_K_) { cudaFree(d_K_); d_K_ = nullptr; }
    if (d_D_) { cudaFree(d_D_); d_D_ = nullptr; }
    if (d_eri_buffer_) { cudaFree(d_eri_buffer_); d_eri_buffer_ = nullptr; }
    eri_buffer_size_ = 0;
}

void GpuFockBuilder::set_density(const Real* D, Size nbf) {
    if (nbf != nbf_) {
        throw InvalidArgumentException(
            "GpuFockBuilder::set_density: nbf mismatch (expected " +
            std::to_string(nbf_) + ", got " + std::to_string(nbf) + ")");
    }

    const size_t size = nbf * nbf * sizeof(double);
    cudaError_t err = cudaMemcpyAsync(d_D_, D, size, cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to upload density matrix: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

void GpuFockBuilder::accumulate(const TwoElectronBuffer<0>& buffer,
                                 Index fa, Index fb, Index fc, Index fd,
                                 int na, int nb, int nc, int nd)
{
    const size_t n_eri = static_cast<size_t>(na) * nb * nc * nd;
    if (n_eri == 0) return;

    const size_t required_size = n_eri * sizeof(double);

    // Resize ERI buffer if needed
    if (required_size > eri_buffer_size_) {
        if (d_eri_buffer_) {
            cudaFree(d_eri_buffer_);
        }
        cudaError_t err = cudaMalloc(&d_eri_buffer_, required_size);
        if (err != cudaSuccess) {
            d_eri_buffer_ = nullptr;
            eri_buffer_size_ = 0;
            throw std::runtime_error("Failed to allocate ERI buffer: " +
                                     std::string(cudaGetErrorString(err)));
        }
        eri_buffer_size_ = required_size;
    }

    // Pack ERIs into linear array for upload
    std::vector<double> host_eri(n_eri);
    int idx = 0;
    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            for (int c = 0; c < nc; ++c) {
                for (int d = 0; d < nd; ++d) {
                    host_eri[idx++] = buffer(a, b, c, d);
                }
            }
        }
    }

    // Upload ERIs to device
    cudaError_t err = cudaMemcpyAsync(d_eri_buffer_, host_eri.data(),
                                       required_size, cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to upload ERIs: " +
                                 std::string(cudaGetErrorString(err)));
    }

    // Launch accumulation kernel
    detail::launch_fock_accumulate_kernel(
        d_eri_buffer_, d_D_, d_J_, d_K_,
        static_cast<int>(fa), static_cast<int>(fb),
        static_cast<int>(fc), static_cast<int>(fd),
        na, nb, nc, nd,
        static_cast<int>(nbf_),
        stream_
    );
}

std::vector<Real> GpuFockBuilder::get_coulomb_matrix() const {
    std::vector<Real> J(nbf_ * nbf_);
    const size_t size = nbf_ * nbf_ * sizeof(double);

    cudaError_t err = cudaMemcpyAsync(J.data(), d_J_, size,
                                       cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to download J matrix: " +
                                 std::string(cudaGetErrorString(err)));
    }

    // Synchronize to ensure download is complete
    cudaStreamSynchronize(stream_);

    return J;
}

std::vector<Real> GpuFockBuilder::get_exchange_matrix() const {
    std::vector<Real> K(nbf_ * nbf_);
    const size_t size = nbf_ * nbf_ * sizeof(double);

    cudaError_t err = cudaMemcpyAsync(K.data(), d_K_, size,
                                       cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to download K matrix: " +
                                 std::string(cudaGetErrorString(err)));
    }

    cudaStreamSynchronize(stream_);

    return K;
}

std::vector<Real> GpuFockBuilder::get_fock_matrix(
    std::span<const Real> H_core,
    Real exchange_fraction) const
{
    if (H_core.size() != nbf_ * nbf_) {
        throw InvalidArgumentException(
            "GpuFockBuilder::get_fock_matrix: H_core size mismatch");
    }

    // Download J and K
    auto J = get_coulomb_matrix();
    auto K = get_exchange_matrix();

    // Compute F = H_core + J - exchange_fraction * K
    std::vector<Real> F(nbf_ * nbf_);
    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        F[i] = H_core[i] + J[i] - exchange_fraction * K[i];
    }

    return F;
}

void GpuFockBuilder::reset() {
    const size_t matrix_size = nbf_ * nbf_;
    detail::launch_matrix_zero_kernel(d_J_, matrix_size, stream_);
    detail::launch_matrix_zero_kernel(d_K_, matrix_size, stream_);
}

void GpuFockBuilder::synchronize() {
    cudaStreamSynchronize(stream_);
}

// =============================================================================
// Phase 4.5: Device-Side Batched Accumulation
// =============================================================================

void GpuFockBuilder::accumulate_device_eri_batch(
    const double* d_eri_batch,
    const basis::ShellSetQuartetDeviceData& quartet_data,
    Size nbf)
{
    if (nbf != nbf_) {
        throw InvalidArgumentException(
            "GpuFockBuilder::accumulate_device_eri_batch: nbf mismatch (expected " +
            std::to_string(nbf_) + ", got " + std::to_string(nbf) + ")");
    }

    // Launch batched accumulation kernel - ERIs stay on device
    detail::launch_fock_accumulate_batch_kernel(
        d_eri_batch,
        quartet_data,
        d_D_,
        d_J_,
        d_K_,
        static_cast<int>(nbf_),
        stream_
    );
}

}  // namespace libaccint::consumers

#endif  // LIBACCINT_USE_CUDA
