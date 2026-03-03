// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file multi_gpu_fock_builder.cu
/// @brief Multi-GPU Fock builder implementation

#include <libaccint/consumers/multi_gpu_fock_builder.hpp>

#if LIBACCINT_USE_CUDA

#include <algorithm>
#include <numeric>

namespace libaccint::consumers {

// ============================================================================
// Constructor / Destructor
// ============================================================================

MultiGPUFockBuilder::MultiGPUFockBuilder(Size nbf, 
                                           const std::vector<int>& device_ids)
    : nbf_(nbf),
      device_ids_(device_ids) {
    allocate_device_resources();
}

MultiGPUFockBuilder::~MultiGPUFockBuilder() {
    free_device_resources();
}

MultiGPUFockBuilder::MultiGPUFockBuilder(MultiGPUFockBuilder&& other) noexcept
    : nbf_(other.nbf_),
      device_ids_(std::move(other.device_ids_)),
      device_builders_(std::move(other.device_builders_)),
      d_reduction_workspace_(other.d_reduction_workspace_),
      reduction_done_(other.reduction_done_) {
    other.d_reduction_workspace_ = nullptr;
}

MultiGPUFockBuilder& MultiGPUFockBuilder::operator=(
    MultiGPUFockBuilder&& other) noexcept {
    if (this != &other) {
        free_device_resources();
        
        nbf_ = other.nbf_;
        device_ids_ = std::move(other.device_ids_);
        device_builders_ = std::move(other.device_builders_);
        d_reduction_workspace_ = other.d_reduction_workspace_;
        reduction_done_ = other.reduction_done_;
        
        other.d_reduction_workspace_ = nullptr;
    }
    return *this;
}

void MultiGPUFockBuilder::allocate_device_resources() {
    device_builders_.reserve(device_ids_.size());
    
    for (int device_id : device_ids_) {
        device::ScopedDevice guard(device_id);
        
        // Create stream for this device's builder
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        device_builders_.push_back(
            std::make_unique<GpuFockBuilder>(nbf_, stream));
    }
    
    // Allocate reduction workspace on primary device
    if (!device_ids_.empty()) {
        device::ScopedDevice guard(device_ids_[0]);
        cudaMalloc(&d_reduction_workspace_, nbf_ * nbf_ * sizeof(double));
    }
}

void MultiGPUFockBuilder::free_device_resources() {
    // Free reduction workspace
    if (d_reduction_workspace_ != nullptr && !device_ids_.empty()) {
        device::ScopedDevice guard(device_ids_[0]);
        cudaFree(d_reduction_workspace_);
        d_reduction_workspace_ = nullptr;
    }
    
    device_builders_.clear();
}

// ============================================================================
// Configuration
// ============================================================================

void MultiGPUFockBuilder::set_density(const Real* D, Size nbf) {
    if (nbf != nbf_) {
        throw std::invalid_argument("Density matrix size mismatch");
    }
    
    // Replicate density to all devices
    for (size_t i = 0; i < device_ids_.size(); ++i) {
        device::ScopedDevice guard(device_ids_[i]);
        device_builders_[i]->set_density(D, nbf);
    }
    
    reduction_done_ = false;
}

void MultiGPUFockBuilder::reset() {
    for (size_t i = 0; i < device_ids_.size(); ++i) {
        device::ScopedDevice guard(device_ids_[i]);
        device_builders_[i]->reset();
    }
    reduction_done_ = false;
}

// ============================================================================
// Accumulation
// ============================================================================

void MultiGPUFockBuilder::accumulate(
    int device_id,
    const TwoElectronBuffer<0>& buffer,
    Index fa, Index fb, Index fc, Index fd,
    int na, int nb, int nc, int nd) {
    
    // Find device index
    auto it = std::find(device_ids_.begin(), device_ids_.end(), device_id);
    if (it == device_ids_.end()) {
        throw device::DeviceError(device_id, "Device not managed by this builder");
    }
    size_t idx = std::distance(device_ids_.begin(), it);
    
    device::ScopedDevice guard(device_id);
    device_builders_[idx]->accumulate(buffer, fa, fb, fc, fd, na, nb, nc, nd);
    reduction_done_ = false;
}

void MultiGPUFockBuilder::accumulate(
    int device_id,
    const double* flat_eri,
    const ShellSetQuartet& quartet) {

    auto it = std::find(device_ids_.begin(), device_ids_.end(), device_id);
    if (it == device_ids_.end()) {
        throw device::DeviceError(device_id, "Device not managed by this builder");
    }
    size_t idx = std::distance(device_ids_.begin(), it);

    const auto& set_a = quartet.bra_pair().shell_set_a();
    const auto& set_b = quartet.bra_pair().shell_set_b();
    const auto& set_c = quartet.ket_pair().shell_set_a();
    const auto& set_d = quartet.ket_pair().shell_set_b();

    const int na_funcs = n_cartesian(set_a.angular_momentum());
    const int nb_funcs = n_cartesian(set_b.angular_momentum());
    const int nc_funcs = n_cartesian(set_c.angular_momentum());
    const int nd_funcs = n_cartesian(set_d.angular_momentum());
    const Size funcs_per_quartet =
        static_cast<Size>(na_funcs) * nb_funcs * nc_funcs * nd_funcs;

    TwoElectronBuffer<0> buffer;
    buffer.resize(na_funcs, nb_funcs, nc_funcs, nd_funcs);

    device::ScopedDevice guard(device_id);

    size_t flat_idx = 0;
    for (Size i = 0; i < set_a.n_shells(); ++i) {
        const auto& shell_a = set_a.shell(i);
        const Index fi = shell_a.function_index();

        for (Size j = 0; j < set_b.n_shells(); ++j) {
            const auto& shell_b = set_b.shell(j);
            const Index fj = shell_b.function_index();

            for (Size k = 0; k < set_c.n_shells(); ++k) {
                const auto& shell_c = set_c.shell(k);
                const Index fk = shell_c.function_index();

                for (Size l = 0; l < set_d.n_shells(); ++l) {
                    const auto& shell_d = set_d.shell(l);
                    const Index fl = shell_d.function_index();

                    for (int a = 0; a < na_funcs; ++a) {
                        for (int b = 0; b < nb_funcs; ++b) {
                            for (int c = 0; c < nc_funcs; ++c) {
                                for (int d = 0; d < nd_funcs; ++d) {
                                    buffer(a, b, c, d) = flat_eri[
                                        flat_idx +
                                        static_cast<size_t>(a) * nb_funcs * nc_funcs * nd_funcs +
                                        static_cast<size_t>(b) * nc_funcs * nd_funcs +
                                        static_cast<size_t>(c) * nd_funcs + d];
                                }
                            }
                        }
                    }

                    device_builders_[idx]->accumulate(
                        buffer, fi, fj, fk, fl,
                        na_funcs, nb_funcs, nc_funcs, nd_funcs);

                    flat_idx += funcs_per_quartet;
                }
            }
        }
    }

    reduction_done_ = false;
}

void MultiGPUFockBuilder::accumulate_device(
    int device_id,
    const double* d_eri_batch,
    const basis::ShellSetQuartetDeviceData& quartet_data,
    Size nbf) {
    
    auto it = std::find(device_ids_.begin(), device_ids_.end(), device_id);
    if (it == device_ids_.end()) {
        throw device::DeviceError(device_id, "Device not managed by this builder");
    }
    size_t idx = std::distance(device_ids_.begin(), it);
    
    device::ScopedDevice guard(device_id);
    device_builders_[idx]->accumulate_device_eri_batch(d_eri_batch, quartet_data, nbf);
    reduction_done_ = false;
}

// ============================================================================
// Device-side reduction kernel
// ============================================================================
//
// Simple element-wise addition kernel: dst[i] += src[i].
// Used for on-device Fock matrix reduction across GPUs, avoiding the
// previous host-staged round-trip (D2H + CPU add + H2D) which was a
// significant bottleneck for large basis sets.

/// @brief Element-wise in-place addition: dst[i] += src[i]
/// @param dst Destination array (in-place accumulation target)
/// @param src Source array to add
/// @param n   Number of elements
__global__ void device_vector_add_kernel(
    double* __restrict__ dst,
    const double* __restrict__ src,
    size_t n)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < n) {
        dst[tid] += src[tid];
    }
}

/// @brief Launch the device-side vector addition kernel
/// @param dst      Device pointer to destination (accumulation target)
/// @param src      Device pointer to source
/// @param n        Number of double elements
/// @param stream   CUDA stream for async execution
static void launch_device_vector_add(
    double* dst,
    const double* src,
    size_t n,
    cudaStream_t stream)
{
    constexpr int kBlockSize = 256;
    const int n_blocks = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
    device_vector_add_kernel<<<n_blocks, kBlockSize, 0, stream>>>(dst, src, n);
}

// ============================================================================
// Result Retrieval
// ============================================================================

void MultiGPUFockBuilder::reduce_to_primary() {
    if (reduction_done_ || device_ids_.size() <= 1) {
        reduction_done_ = true;
        return;
    }

    auto& dm = device::DeviceManager::instance();
    const int primary_device = device_ids_[0];
    const size_t matrix_size = nbf_ * nbf_;

    // Get primary device J and K pointers
    double* d_J_primary = device_builders_[0]->d_J();
    double* d_K_primary = device_builders_[0]->d_K();

    // Create a stream on the primary device for reduction operations
    cudaStream_t reduction_stream;
    {
        device::ScopedDevice guard(primary_device);
        cudaStreamCreate(&reduction_stream);
    }

    // Reduce from other devices to primary.
    //
    // Two code paths are used:
    //
    // 1. P2P path (preferred): When peer-to-peer access is available
    //    between the primary and source device, we use
    //    cudaMemcpyPeerAsync to transfer data directly into the
    //    reduction workspace on the primary device, then launch the
    //    device_vector_add_kernel to accumulate into J/K.  This avoids
    //    any host-side staging and runs entirely asynchronously on the
    //    GPU.
    //
    // 2. Host-staged fallback: When P2P is not available (e.g.,
    //    different PCIe root complexes, or IOMMU restrictions), we fall
    //    back to the host-staged path: D2H from source, H2D to primary
    //    workspace, then device-side addition.  This is slower but
    //    still uses the device addition kernel rather than CPU-side
    //    element-wise accumulation.

    for (size_t i = 1; i < device_ids_.size(); ++i) {
        int src_device = device_ids_[i];
        double* d_J_src = device_builders_[i]->d_J();
        double* d_K_src = device_builders_[i]->d_K();

        // Synchronise the source device's builder stream so that all
        // accumulations on that device are complete before we read.
        {
            device::ScopedDevice guard(src_device);
            device_builders_[i]->synchronize();
        }

        // ---- Reduce J matrix ----
        {
            device::ScopedDevice guard(primary_device);

            if (dm.can_access_peer(primary_device, src_device)) {
                // P2P path: async copy + device-side add
                cudaMemcpyPeerAsync(d_reduction_workspace_, primary_device,
                                    d_J_src, src_device,
                                    matrix_size * sizeof(double),
                                    reduction_stream);
            } else {
                // Host-staged fallback: sync copy through pinned host buffer
                // TODO(Phase 7.3): Use pinned memory pool for the host
                //   staging buffer to avoid per-reduction allocation overhead
                std::vector<double> host_buffer(matrix_size);
                {
                    device::ScopedDevice src_guard(src_device);
                    cudaMemcpy(host_buffer.data(), d_J_src,
                               matrix_size * sizeof(double),
                               cudaMemcpyDeviceToHost);
                }
                {
                    device::ScopedDevice pri_guard(primary_device);
                    cudaMemcpyAsync(d_reduction_workspace_, host_buffer.data(),
                                    matrix_size * sizeof(double),
                                    cudaMemcpyHostToDevice, reduction_stream);
                    cudaStreamSynchronize(reduction_stream);
                }
            }

            // Device-side element-wise addition: J_primary += workspace
            launch_device_vector_add(d_J_primary, d_reduction_workspace_,
                                     matrix_size, reduction_stream);
        }

        // ---- Reduce K matrix ----
        {
            device::ScopedDevice guard(primary_device);

            if (dm.can_access_peer(primary_device, src_device)) {
                // P2P path: async copy + device-side add
                cudaMemcpyPeerAsync(d_reduction_workspace_, primary_device,
                                    d_K_src, src_device,
                                    matrix_size * sizeof(double),
                                    reduction_stream);
            } else {
                // Host-staged fallback
                std::vector<double> host_buffer(matrix_size);
                {
                    device::ScopedDevice src_guard(src_device);
                    cudaMemcpy(host_buffer.data(), d_K_src,
                               matrix_size * sizeof(double),
                               cudaMemcpyDeviceToHost);
                }
                {
                    device::ScopedDevice pri_guard(primary_device);
                    cudaMemcpyAsync(d_reduction_workspace_, host_buffer.data(),
                                    matrix_size * sizeof(double),
                                    cudaMemcpyHostToDevice, reduction_stream);
                    cudaStreamSynchronize(reduction_stream);
                }
            }

            // Device-side element-wise addition: K_primary += workspace
            launch_device_vector_add(d_K_primary, d_reduction_workspace_,
                                     matrix_size, reduction_stream);
        }
    }

    // Synchronise the reduction stream to ensure all additions are done
    {
        device::ScopedDevice guard(primary_device);
        cudaStreamSynchronize(reduction_stream);
        cudaStreamDestroy(reduction_stream);
    }

    // TODO(Phase 7.3): For 3+ GPUs, implement a tree-reduction pattern
    // where GPUs reduce pair-wise rather than all reducing into the
    // primary.  This halves the serialisation depth for 4+ GPUs.

    reduction_done_ = true;
}

std::vector<Real> MultiGPUFockBuilder::get_coulomb_matrix() {
    reduce_to_primary();
    
    device::ScopedDevice guard(device_ids_[0]);
    return device_builders_[0]->get_coulomb_matrix();
}

std::vector<Real> MultiGPUFockBuilder::get_exchange_matrix() {
    reduce_to_primary();
    
    device::ScopedDevice guard(device_ids_[0]);
    return device_builders_[0]->get_exchange_matrix();
}

std::vector<Real> MultiGPUFockBuilder::get_fock_matrix(
    std::span<const Real> H_core,
    Real exchange_fraction) {
    
    reduce_to_primary();
    
    device::ScopedDevice guard(device_ids_[0]);
    return device_builders_[0]->get_fock_matrix(H_core, exchange_fraction);
}

// ============================================================================
// Accessors
// ============================================================================

GpuFockBuilder& MultiGPUFockBuilder::device_builder(int device_id) {
    auto it = std::find(device_ids_.begin(), device_ids_.end(), device_id);
    if (it == device_ids_.end()) {
        throw device::DeviceError(device_id, "Device not managed by this builder");
    }
    return *device_builders_[std::distance(device_ids_.begin(), it)];
}

const GpuFockBuilder& MultiGPUFockBuilder::device_builder(int device_id) const {
    auto it = std::find(device_ids_.begin(), device_ids_.end(), device_id);
    if (it == device_ids_.end()) {
        throw device::DeviceError(device_id, "Device not managed by this builder");
    }
    return *device_builders_[std::distance(device_ids_.begin(), it)];
}

void MultiGPUFockBuilder::synchronize() {
    for (size_t i = 0; i < device_ids_.size(); ++i) {
        device::ScopedDevice guard(device_ids_[i]);
        device_builders_[i]->synchronize();
    }
}

}  // namespace libaccint::consumers

#endif  // LIBACCINT_USE_CUDA
