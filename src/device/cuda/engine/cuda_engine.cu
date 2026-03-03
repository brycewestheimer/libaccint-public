// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file cuda_engine.cu
/// @brief CUDA GPU backend implementation for molecular integral computation

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/engine/cuda_engine.hpp>
#include <libaccint/engine/cpu_engine.hpp>
#include <libaccint/engine/dispatch_policy.hpp>
#include <libaccint/device/device_resource_tracker.hpp>
#include <libaccint/utils/logging.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/kernels/overlap_kernel_cuda.hpp>
#include <libaccint/kernels/kinetic_kernel_cuda.hpp>
#include <libaccint/kernels/nuclear_kernel_cuda.hpp>
#include <libaccint/kernels/eri_kernel_cuda.hpp>
#include <libaccint/kernels/fused_1e_kernel_cuda.hpp>
#include <libaccint/kernels/generated_kernel_registry_cuda.hpp>
#include <libaccint/kernels/kernel_variant.hpp>
#include <libaccint/core/backend.hpp>

#include <algorithm>
#include <cstdlib>
#include <limits>

#include <cuda_runtime.h>
#include <stdexcept>

// Device Boys function initialization
namespace libaccint::device::math {
    double* boys_device_init(cudaStream_t stream);
    void boys_device_cleanup();
    double* boys_device_get_coeffs();
    bool boys_device_is_initialized();
}

namespace libaccint {

namespace {

constexpr std::size_t kDefaultMaxEriTensorBytes = 1024ull * 1024ull * 1024ull;  // 1 GiB

bool checked_mul(std::size_t a, std::size_t b, std::size_t& out) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

std::size_t max_eri_tensor_bytes() {
    const char* env = std::getenv("LIBACCINT_MAX_ERI_TENSOR_BYTES");
    if (env == nullptr || env[0] == '\0') {
        return kDefaultMaxEriTensorBytes;
    }

    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (end != env && *end == '\0' && parsed > 0) {
        return static_cast<std::size_t>(parsed);
    }
    return kDefaultMaxEriTensorBytes;
}

std::size_t checked_eri_tensor_elements(Size nbf, const char* caller) {
    std::size_t nbf2 = 0;
    std::size_t nbf3 = 0;
    std::size_t nbf4 = 0;
    if (!checked_mul(static_cast<std::size_t>(nbf), static_cast<std::size_t>(nbf), nbf2) ||
        !checked_mul(nbf2, static_cast<std::size_t>(nbf), nbf3) ||
        !checked_mul(nbf3, static_cast<std::size_t>(nbf), nbf4)) {
        throw InvalidArgumentException(
            std::string(caller) + ": nbf^4 overflow for nbf=" + std::to_string(nbf));
    }
    return nbf4;
}

void validate_eri_tensor_limit(Size nbf, std::size_t elements, const char* caller) {
    std::size_t bytes = 0;
    if (!checked_mul(elements, sizeof(double), bytes)) {
        throw InvalidArgumentException(
            std::string(caller) + ": ERI tensor byte-size overflow for nbf=" +
            std::to_string(nbf));
    }

    const std::size_t max_bytes = max_eri_tensor_bytes();
    if (bytes > max_bytes) {
        throw InvalidArgumentException(
            std::string(caller) + ": requested tensor requires " + std::to_string(bytes) +
            " bytes (nbf=" + std::to_string(nbf) + "), exceeding limit " +
            std::to_string(max_bytes) +
            ". Use callback mode/consumer APIs or raise LIBACCINT_MAX_ERI_TENSOR_BYTES.");
    }
}

void validate_gpu_slot_count(Size n_gpu_slots, const char* caller) {
    if (n_gpu_slots == 0) {
        throw InvalidArgumentException(
            std::string(caller) + " requires n_gpu_slots >= 1");
    }
}

void validate_pipeline_slot_count(size_t n_slots, const char* caller) {
    if (n_slots == 0) {
        throw InvalidArgumentException(
            std::string(caller) + " requires config.n_slots >= 1");
    }
}

}  // namespace

// =============================================================================
// Constructor / Destructor
// =============================================================================

CudaEngine::CudaEngine(const BasisSet& basis)
    : basis_(&basis) {
    // Check CUDA availability
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        throw BackendError(BackendType::CUDA, "No CUDA devices available");
    }

    initialize();
}

CudaEngine::~CudaEngine() {
    cleanup();
}

CudaEngine::CudaEngine(CudaEngine&& other) noexcept
    : basis_(other.basis_),
      initialized_(other.initialized_),
      stream_(other.stream_),
      d_boys_coeffs_(other.d_boys_coeffs_),
      shell_cache_(std::move(other.shell_cache_)),
      dispatch_table_(std::move(other.dispatch_table_)),
      tracker_(other.tracker_),
      cpu_fallback_(other.cpu_fallback_),
      min_gpu_batch_size_(other.min_gpu_batch_size_),
      slot_pool_(std::move(other.slot_pool_)),
      n_gpu_slots_(other.n_gpu_slots_) {
    other.basis_ = nullptr;
    other.initialized_ = false;
    other.stream_ = nullptr;
    other.d_boys_coeffs_ = nullptr;
    other.tracker_ = nullptr;
    other.cpu_fallback_ = nullptr;
}

CudaEngine& CudaEngine::operator=(CudaEngine&& other) noexcept {
    if (this != &other) {
        cleanup();
        basis_ = other.basis_;
        initialized_ = other.initialized_;
        stream_ = other.stream_;
        d_boys_coeffs_ = other.d_boys_coeffs_;
        shell_cache_ = std::move(other.shell_cache_);
        dispatch_table_ = std::move(other.dispatch_table_);
        tracker_ = other.tracker_;
        cpu_fallback_ = other.cpu_fallback_;
        min_gpu_batch_size_ = other.min_gpu_batch_size_;
        slot_pool_ = std::move(other.slot_pool_);
        n_gpu_slots_ = other.n_gpu_slots_;
        other.initialized_ = false;
        other.stream_ = nullptr;
        other.d_boys_coeffs_ = nullptr;
        other.tracker_ = nullptr;
        other.cpu_fallback_ = nullptr;
    }
    return *this;
}

void CudaEngine::initialize() {
    if (initialized_) return;

    // Create CUDA stream
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        throw BackendError(BackendType::CUDA,
            std::string("Failed to create CUDA stream: ") + cudaGetErrorString(err));
    }

    // Initialize device Boys function tables
    if (!device::math::boys_device_is_initialized()) {
        d_boys_coeffs_ = device::math::boys_device_init(stream_);
    } else {
        d_boys_coeffs_ = device::math::boys_device_get_coeffs();
    }

    // Initialize shell cache
    shell_cache_ = std::make_unique<basis::ShellSetDeviceCache>();

    // Initialize optimal dispatch table
    // Check for runtime override via environment variable
    const char* dispatch_table_env = std::getenv("LIBACCINT_DISPATCH_TABLE");
    if (dispatch_table_env && dispatch_table_env[0] != '\0') {
        dispatch_table_ = std::make_unique<kernels::OptimalDispatchTable>(
            kernels::OptimalDispatchTable::from_json(dispatch_table_env));
    } else {
        dispatch_table_ = std::make_unique<kernels::OptimalDispatchTable>(
            std::string(LIBACCINT_KERNEL_DISPATCH));
    }

    // Synchronize to ensure Boys tables are uploaded
    cudaStreamSynchronize(stream_);

    // Initialize runtime GPU resource tracker
    tracker_ = &device::DeviceResourceTracker::instance();
    tracker_->refresh_device_properties();

    // Initialize GPU execution slot pool for concurrent access
    validate_gpu_slot_count(n_gpu_slots_, "CudaEngine::initialize");
    slot_pool_ = std::make_unique<memory::GpuSlotPool>(
        static_cast<size_t>(n_gpu_slots_));

    initialized_ = true;
}

void CudaEngine::set_dispatch_config(const DispatchConfig& config) {
    validate_gpu_slot_count(config.n_gpu_slots, "CudaEngine::set_dispatch_config");
    min_gpu_batch_size_ = config.min_gpu_batch_size;

    // Recreate slot pool if the requested count differs
    if (config.n_gpu_slots != n_gpu_slots_ && initialized_) {
        n_gpu_slots_ = config.n_gpu_slots;
        slot_pool_ = std::make_unique<memory::GpuSlotPool>(
            static_cast<size_t>(n_gpu_slots_));
    } else {
        n_gpu_slots_ = config.n_gpu_slots;
    }
}

void CudaEngine::cleanup() {
    if (!initialized_) return;

    // Clear and reset shell cache
    if (shell_cache_) {
        shell_cache_->clear();
        shell_cache_.reset();
    }

    // Destroy slot pool (frees all per-slot device buffers and streams)
    slot_pool_.reset();

    // Destroy primary stream (used for init/cleanup only)
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

    // Note: Don't cleanup Boys tables as they may be shared

    initialized_ = false;
}

// =============================================================================
// Buffer Management (delegated to GpuExecutionSlot)
// =============================================================================

void CudaEngine::scatter_1e_to_matrix(const ShellSetPair& pair,
                                       const std::vector<double>& flat_output,
                                       std::vector<Real>& result) {
    const Size nbf = basis_->n_basis_functions();
    const auto& set_a = pair.shell_set_a();
    const auto& set_b = pair.shell_set_b();

    const int na_funcs = n_cartesian(set_a.angular_momentum());
    const int nb_funcs = n_cartesian(set_b.angular_momentum());
    const Size funcs_per_pair = static_cast<Size>(na_funcs * nb_funcs);

    const bool is_self_pair = (&set_a == &set_b);

    size_t flat_idx = 0;
    for (Size i = 0; i < set_a.n_shells(); ++i) {
        const auto& shell_a = set_a.shell(i);
        const Index fi = shell_a.function_index();

        for (Size j = 0; j < set_b.n_shells(); ++j) {
            const auto& shell_b = set_b.shell(j);
            const Index fj = shell_b.function_index();

            for (int a = 0; a < na_funcs; ++a) {
                for (int b = 0; b < nb_funcs; ++b) {
                    const Real val = flat_output[flat_idx + a * nb_funcs + b];
                    const Size row = static_cast<Size>(fi + a);
                    const Size col = static_cast<Size>(fj + b);
                    result[row * nbf + col] += val;
                    // For cross-pairs, also fill the transpose since
                    // shell_set_pairs only iterates the upper triangle
                    // over shell sets. Self-pairs already iterate both
                    // (i,j) and (j,i) via the full Cartesian product.
                    if (!is_self_pair && row != col) {
                        result[col * nbf + row] += val;
                    }
                }
            }
            flat_idx += funcs_per_pair;
        }
    }
}

void CudaEngine::scatter_2e_soa_to_matrix(
    const ShellSetQuartet& quartet,
    const std::vector<double>& flat_output,
    std::vector<double>& result) {

    const Size nbf = basis_->n_basis_functions();

    const auto& set_a = quartet.bra_pair().shell_set_a();
    const auto& set_b = quartet.bra_pair().shell_set_b();
    const auto& set_c = quartet.ket_pair().shell_set_a();
    const auto& set_d = quartet.ket_pair().shell_set_b();

    const int na_funcs = n_cartesian(set_a.angular_momentum());
    const int nb_funcs = n_cartesian(set_b.angular_momentum());
    const int nc_funcs = n_cartesian(set_c.angular_momentum());
    const int nd_funcs = n_cartesian(set_d.angular_momentum());

    const size_t n_quartets_total = static_cast<size_t>(set_a.n_shells()) *
        set_b.n_shells() * set_c.n_shells() * set_d.n_shells();

    // SoA layout: flat_output[component * n_quartets + quartet_idx]
    size_t q_idx = 0;
    for (Size i = 0; i < set_a.n_shells(); ++i) {
        const Index fi = set_a.shell(i).function_index();
        for (Size j = 0; j < set_b.n_shells(); ++j) {
            const Index fj = set_b.shell(j).function_index();
            for (Size k = 0; k < set_c.n_shells(); ++k) {
                const Index fk = set_c.shell(k).function_index();
                for (Size l = 0; l < set_d.n_shells(); ++l) {
                    const Index fl = set_d.shell(l).function_index();

                    int comp = 0;
                    for (int a = 0; a < na_funcs; ++a) {
                        for (int b = 0; b < nb_funcs; ++b) {
                            for (int c = 0; c < nc_funcs; ++c) {
                                for (int d = 0; d < nd_funcs; ++d) {
                                    const size_t src = static_cast<size_t>(comp) * n_quartets_total + q_idx;
                                    const size_t dst =
                                        static_cast<size_t>(fi + a) * nbf * nbf * nbf +
                                        static_cast<size_t>(fj + b) * nbf * nbf +
                                        static_cast<size_t>(fk + c) * nbf +
                                        static_cast<size_t>(fl + d);
                                    result[dst] += flat_output[src];
                                    ++comp;
                                }
                            }
                        }
                    }

                    ++q_idx;
                }
            }
        }
    }
}

void CudaEngine::synchronize() {
    if (stream_ != nullptr) {
        cudaStreamSynchronize(stream_);
    }
    if (slot_pool_) {
        slot_pool_->synchronize_all();
    }
}

// =============================================================================
// Optimal Dispatch Helpers
// =============================================================================

/// @brief Classify contraction degree into a ContractionRange for K-aware dispatch
///
/// For quartets: max(n_prim_a, n_prim_b, n_prim_c, n_prim_d)
/// For pairs:    max(n_prim_bra, n_prim_ket)  (c/d default to 1)
///
/// Thresholds: <= LIBACCINT_SMALL_K_MAX → SmallK,
///             <= LIBACCINT_MEDIUM_K_MAX → MediumK,
///             else LargeK
static kernels::ContractionRange classify_contraction_range(
    int n_prim_a, int n_prim_b, int n_prim_c = 1, int n_prim_d = 1) noexcept {
    const int max_k = std::max({n_prim_a, n_prim_b, n_prim_c, n_prim_d});
    if (max_k <= LIBACCINT_SMALL_K_MAX) return kernels::ContractionRange::SmallK;
    if (max_k <= LIBACCINT_MEDIUM_K_MAX) return kernels::ContractionRange::MediumK;
    return kernels::ContractionRange::LargeK;
}

/// @brief Convert kernels::ContractionRange to device::GpuContractionRange
static device::GpuContractionRange to_gpu_contraction_range(
    kernels::ContractionRange k) noexcept {
    switch (k) {
        case kernels::ContractionRange::SmallK:  return device::GpuContractionRange::SmallK;
        case kernels::ContractionRange::MediumK: return device::GpuContractionRange::MediumK;
        case kernels::ContractionRange::LargeK:  return device::GpuContractionRange::LargeK;
        default:                                 return device::GpuContractionRange::SmallK;
    }
}

void CudaEngine::dispatch_1e_optimal(OperatorKind op_kind,
                                      const basis::ShellSetPairDeviceData& pair_device,
                                      double* d_output,
                                      cudaStream_t stream,
                                      const operators::DevicePointChargeData* charge_data) {
    const int la = pair_device.bra.angular_momentum;
    const int lb = pair_device.ket.angular_momentum;
    const auto& entry = dispatch_table_->get_1e(la, lb);

    // --- K-range classification (Step 5.3) ---
    const auto k_range = classify_contraction_range(
        pair_device.bra.n_primitives, pair_device.ket.n_primitives);
    const auto gpu_k_range = to_gpu_contraction_range(k_range);

    // --- Resource tracking (Phase 4.2, updated with K-range in Step 5.3) ---
    const int n_work_units = pair_device.n_pairs();

    // Map OperatorKind to GpuIntegralType for the tracker
    device::GpuIntegralType integral_type = device::GpuIntegralType::Overlap;
    switch (op_kind) {
        case OperatorKind::Kinetic:    integral_type = device::GpuIntegralType::Kinetic; break;
        case OperatorKind::Nuclear:
        case OperatorKind::PointCharge: integral_type = device::GpuIntegralType::Nuclear; break;
        default: break;
    }

    // Get recommended batch config and occupancy estimate
    AMQuartet am_quartet = {la, lb, 0, 0};
    device::BatchConfig batch_cfg;
    device::ResourceReservation reservation;
    if (tracker_) {
        batch_cfg = tracker_->recommend_batch_config(
            integral_type, am_quartet,
            gpu_k_range, n_work_units);

        // Estimate occupancy and warn if low
        device::GpuKernelConfig kernel_cfg;
        kernel_cfg.block_size = batch_cfg.block_dim.x;
        kernel_cfg.shared_mem_per_block = batch_cfg.shared_mem_bytes;
        auto occ = tracker_->estimate_occupancy(kernel_cfg, n_work_units);
        if (occ.theoretical_occupancy < 0.5) {
            LIBACCINT_LOG_WARNING("CudaEngine",
                "Low GPU occupancy (" + std::to_string(int(occ.theoretical_occupancy * 100)) +
                "%) for 1e (" + std::to_string(la) + "," + std::to_string(lb) +
                ") — limited by " + occ.limiting_factor);
        }

        // Reserve resources (RAII — released when reservation goes out of scope)
        if (tracker_->can_launch(kernel_cfg, n_work_units)) {
            reservation = tracker_->reserve(kernel_cfg, n_work_units, stream);
        } else if (tracker_->wait_for_resources(kernel_cfg, n_work_units,
                                                 std::chrono::milliseconds{10})) {
            reservation = tracker_->reserve(kernel_cfg, n_work_units, stream);
#ifdef LIBACCINT_TRACE_DISPATCH
            LIBACCINT_LOG_DEBUG("CudaEngine",
                "GPU resources freed after wait for 1e (" + std::to_string(la) + "," +
                std::to_string(lb) + ")");
#endif
        } else {
#ifdef LIBACCINT_TRACE_DISPATCH
            LIBACCINT_LOG_WARNING("CudaEngine",
                "GPU resources unavailable for 1e (" + std::to_string(la) + "," +
                std::to_string(lb) + "), proceeding with launch (best effort)");
#endif
        }
    }

    switch (op_kind) {
        case OperatorKind::Overlap:
            switch (entry.overlap) {
                case kernels::KernelVariant::GeneratedOverlap:
                    if (kernels::cuda::generated::has_generated_overlap_k_aware(la, lb, k_range)) {
#ifdef LIBACCINT_TRACE_DISPATCH
                        LIBACCINT_LOG_DEBUG("CudaEngine",
                            "dispatch 1e overlap (" + std::to_string(la) + "," + std::to_string(lb) +
                            ") K=" + std::string(kernels::to_string(k_range)) + " → generated (K-aware)");
#endif
                        kernels::cuda::generated::launch_generated_overlap_k_aware(
                            pair_device, d_output, k_range, stream,
                            tracker_ ? &batch_cfg : nullptr);
                        return;
                    }
                    [[fallthrough]];
                case kernels::KernelVariant::HandwrittenOverlap:
                default:
#ifdef LIBACCINT_TRACE_DISPATCH
                    LIBACCINT_LOG_DEBUG("CudaEngine",
                        "dispatch 1e overlap (" + std::to_string(la) + "," + std::to_string(lb) +
                        ") K=" + std::string(kernels::to_string(k_range)) + " → handwritten");
#endif
                    kernels::cuda::dispatch_overlap_kernel(pair_device, d_output, stream);
                    return;
            }

        case OperatorKind::Kinetic:
            switch (entry.kinetic) {
                case kernels::KernelVariant::GeneratedKinetic:
                    if (kernels::cuda::generated::has_generated_kinetic_k_aware(la, lb, k_range)) {
#ifdef LIBACCINT_TRACE_DISPATCH
                        LIBACCINT_LOG_DEBUG("CudaEngine",
                            "dispatch 1e kinetic (" + std::to_string(la) + "," + std::to_string(lb) +
                            ") K=" + std::string(kernels::to_string(k_range)) + " → generated (K-aware)");
#endif
                        kernels::cuda::generated::launch_generated_kinetic_k_aware(
                            pair_device, d_output, k_range, stream,
                            tracker_ ? &batch_cfg : nullptr);
                        return;
                    }
                    [[fallthrough]];
                case kernels::KernelVariant::HandwrittenKinetic:
                default:
#ifdef LIBACCINT_TRACE_DISPATCH
                    LIBACCINT_LOG_DEBUG("CudaEngine",
                        "dispatch 1e kinetic (" + std::to_string(la) + "," + std::to_string(lb) +
                        ") K=" + std::string(kernels::to_string(k_range)) + " → handwritten");
#endif
                    kernels::cuda::dispatch_kinetic_kernel(pair_device, d_output, stream);
                    return;
            }

        case OperatorKind::Nuclear:
        case OperatorKind::PointCharge:
            if (!charge_data) {
                throw std::invalid_argument("dispatch_1e_optimal: Nuclear requires charge_data");
            }
            switch (entry.nuclear) {
                case kernels::KernelVariant::GeneratedNuclear:
                    if (kernels::cuda::generated::has_generated_nuclear_k_aware(la, lb, k_range)) {
#ifdef LIBACCINT_TRACE_DISPATCH
                        LIBACCINT_LOG_DEBUG("CudaEngine",
                            "dispatch 1e nuclear (" + std::to_string(la) + "," + std::to_string(lb) +
                            ") K=" + std::string(kernels::to_string(k_range)) + " → generated (K-aware)");
#endif
                        kernels::cuda::generated::launch_generated_nuclear_k_aware(
                            pair_device, *charge_data, d_boys_coeffs_, d_output, k_range, stream,
                            tracker_ ? &batch_cfg : nullptr);
                        return;
                    }
                    [[fallthrough]];
                case kernels::KernelVariant::HandwrittenNuclear:
                default:
#ifdef LIBACCINT_TRACE_DISPATCH
                    LIBACCINT_LOG_DEBUG("CudaEngine",
                        "dispatch 1e nuclear (" + std::to_string(la) + "," + std::to_string(lb) +
                        ") K=" + std::string(kernels::to_string(k_range)) + " → handwritten");
#endif
                    kernels::cuda::dispatch_nuclear_kernel(
                        pair_device, *charge_data, d_boys_coeffs_, d_output, stream);
                    return;
            }

        default:
            throw std::invalid_argument(
                "dispatch_1e_optimal: Unsupported operator kind: " +
                std::string(to_string(op_kind)));
    }
}

void CudaEngine::dispatch_2e_optimal(const basis::ShellSetQuartetDeviceData& quartet_device,
                                      double* d_output,
                                      cudaStream_t stream) {
    const int la = quartet_device.a.angular_momentum;
    const int lb = quartet_device.b.angular_momentum;
    const int lc = quartet_device.c.angular_momentum;
    const int ld = quartet_device.d.angular_momentum;

    // --- K-range classification (Step 5.3) ---
    const auto k_range = classify_contraction_range(
        quartet_device.a.n_primitives, quartet_device.b.n_primitives,
        quartet_device.c.n_primitives, quartet_device.d.n_primitives);
    const auto gpu_k_range = to_gpu_contraction_range(k_range);

    // K-aware dispatch table query — selects variant + GPU strategy per K-range
    const auto& entry = dispatch_table_->get_2e(la, lb, lc, ld, k_range);

    // --- Resource tracking (Phase 4.2, updated with K-range in Step 5.3) ---
    const int n_work_units = static_cast<int>(quartet_device.n_quartets());

    AMQuartet am_quartet = {la, lb, lc, ld};
    device::BatchConfig batch_cfg;
    device::ResourceReservation reservation;
    if (tracker_) {
        batch_cfg = tracker_->recommend_batch_config(
            device::GpuIntegralType::ERI, am_quartet,
            gpu_k_range, n_work_units);

        // Estimate occupancy and warn if low
        device::GpuKernelConfig kernel_cfg;
        kernel_cfg.block_size = batch_cfg.block_dim.x;
        kernel_cfg.shared_mem_per_block = batch_cfg.shared_mem_bytes;
        auto occ = tracker_->estimate_occupancy(kernel_cfg, n_work_units);
        if (occ.theoretical_occupancy < 0.5) {
            LIBACCINT_LOG_WARNING("CudaEngine",
                "Low GPU occupancy (" + std::to_string(int(occ.theoretical_occupancy * 100)) +
                "%) for 2e (" + std::to_string(la) + "," + std::to_string(lb) +
                "," + std::to_string(lc) + "," + std::to_string(ld) +
                ") — limited by " + occ.limiting_factor);
        }

        // Reserve resources (RAII — released when reservation goes out of scope)
        if (tracker_->can_launch(kernel_cfg, n_work_units)) {
            reservation = tracker_->reserve(kernel_cfg, n_work_units, stream);
        } else if (tracker_->wait_for_resources(kernel_cfg, n_work_units,
                                                 std::chrono::milliseconds{10})) {
            reservation = tracker_->reserve(kernel_cfg, n_work_units, stream);
#ifdef LIBACCINT_TRACE_DISPATCH
            LIBACCINT_LOG_DEBUG("CudaEngine",
                "GPU resources freed after wait for 2e (" + std::to_string(la) + "," +
                std::to_string(lb) + "|" + std::to_string(lc) + "," +
                std::to_string(ld) + ")");
#endif
        } else {
#ifdef LIBACCINT_TRACE_DISPATCH
            LIBACCINT_LOG_WARNING("CudaEngine",
                "GPU resources unavailable for 2e (" + std::to_string(la) + "," +
                std::to_string(lb) + "|" + std::to_string(lc) + "," +
                std::to_string(ld) + "), proceeding with launch (best effort)");
#endif
        }
    }

#ifdef LIBACCINT_TRACE_DISPATCH
    LIBACCINT_LOG_DEBUG("CudaEngine",
        "dispatch 2e (" + std::to_string(la) + "," + std::to_string(lb) + "|" +
        std::to_string(lc) + "," + std::to_string(ld) + ") K=" +
        std::string(kernels::to_string(k_range)) + " strategy=" +
        std::string(kernels::to_string(entry.gpu_strategy)));
#endif

    switch (entry.variant) {
        case kernels::KernelVariant::CooperativeERI:
        case kernels::KernelVariant::GeneratedERI:
            if (kernels::cuda::generated::has_generated_eri_k_aware(la, lb, lc, ld, k_range, entry.gpu_strategy)) {
#ifdef LIBACCINT_TRACE_DISPATCH
                LIBACCINT_LOG_DEBUG("CudaEngine",
                    "  → generated ERI (K-aware, strategy=" +
                    std::string(kernels::to_string(entry.gpu_strategy)) + ")");
#endif
                kernels::cuda::generated::launch_generated_eri_k_aware(
                    quartet_device, d_boys_coeffs_, d_output, k_range, entry.gpu_strategy, stream,
                    tracker_ ? &batch_cfg : nullptr);
                return;
            }
            // Fall through to non-K-aware generated, then handwritten
            if (kernels::cuda::generated::has_generated_eri(la, lb, lc, ld)) {
#ifdef LIBACCINT_TRACE_DISPATCH
                LIBACCINT_LOG_DEBUG("CudaEngine",
                    "  → generated ERI (non-K-aware fallback)");
#endif
                kernels::cuda::generated::launch_generated_eri(
                    quartet_device, d_boys_coeffs_, d_output, stream);
                return;
            }
            [[fallthrough]];
        case kernels::KernelVariant::HandwrittenERI:
        default:
#ifdef LIBACCINT_TRACE_DISPATCH
            LIBACCINT_LOG_DEBUG("CudaEngine",
                "  → handwritten ERI fallback");
#endif
            kernels::cuda::dispatch_eri_kernel(quartet_device, d_boys_coeffs_, d_output, stream);
            return;
    }
}

// =============================================================================
// Shell Pair Computation (One-Electron)
// =============================================================================

void CudaEngine::compute_overlap_shell_pair(const Shell& shell_a,
                                            const Shell& shell_b,
                                            OverlapBuffer& buffer) {
    memory::ScopedGpuSlot scoped(*slot_pool_);
    auto& slot = scoped.slot();
    cudaStream_t s = slot.stream.get();

    ShellSet bra_set(shell_a.angular_momentum(), shell_a.n_primitives());
    ShellSet ket_set(shell_b.angular_momentum(), shell_b.n_primitives());
    bra_set.add_shell(shell_a);
    ket_set.add_shell(shell_b);

    basis::ShellSetDeviceData bra_data = basis::upload_shell_set(bra_set, s);
    basis::ShellSetDeviceData ket_data = basis::upload_shell_set(ket_set, s);

    basis::ShellSetPairDeviceData pair;
    pair.bra = bra_data;
    pair.ket = ket_data;

    size_t output_size = kernels::cuda::overlap_output_size(pair);
    memory::DeviceBuffer<double> d_output(output_size);

    dispatch_1e_optimal(OperatorKind::Overlap, pair, d_output.data(), s);

    const int na = shell_a.n_functions();
    const int nb = shell_b.n_functions();
    buffer.resize(na, nb);

    std::vector<double> host_output(output_size);
    d_output.download(host_output.data(), output_size);
    cudaStreamSynchronize(s);

    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            buffer(a, b) = host_output[a * nb + b];
        }
    }

    basis::free_shell_set_device_data(bra_data);
    basis::free_shell_set_device_data(ket_data);
}

void CudaEngine::compute_kinetic_shell_pair(const Shell& shell_a,
                                            const Shell& shell_b,
                                            KineticBuffer& buffer) {
    memory::ScopedGpuSlot scoped(*slot_pool_);
    auto& slot = scoped.slot();
    cudaStream_t s = slot.stream.get();

    ShellSet bra_set(shell_a.angular_momentum(), shell_a.n_primitives());
    ShellSet ket_set(shell_b.angular_momentum(), shell_b.n_primitives());
    bra_set.add_shell(shell_a);
    ket_set.add_shell(shell_b);

    basis::ShellSetDeviceData bra_data = basis::upload_shell_set(bra_set, s);
    basis::ShellSetDeviceData ket_data = basis::upload_shell_set(ket_set, s);

    basis::ShellSetPairDeviceData pair;
    pair.bra = bra_data;
    pair.ket = ket_data;

    size_t output_size = kernels::cuda::kinetic_output_size(pair);
    memory::DeviceBuffer<double> d_output(output_size);

    dispatch_1e_optimal(OperatorKind::Kinetic, pair, d_output.data(), s);

    const int na = shell_a.n_functions();
    const int nb = shell_b.n_functions();
    buffer.resize(na, nb);

    std::vector<double> host_output(output_size);
    d_output.download(host_output.data(), output_size);
    cudaStreamSynchronize(s);

    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            buffer(a, b) = host_output[a * nb + b];
        }
    }

    basis::free_shell_set_device_data(bra_data);
    basis::free_shell_set_device_data(ket_data);
}

void CudaEngine::compute_nuclear_shell_pair(const Shell& shell_a,
                                            const Shell& shell_b,
                                            const PointChargeParams& charges,
                                            NuclearBuffer& buffer) {
    memory::ScopedGpuSlot scoped(*slot_pool_);
    auto& slot = scoped.slot();
    cudaStream_t s = slot.stream.get();

    ShellSet bra_set(shell_a.angular_momentum(), shell_a.n_primitives());
    ShellSet ket_set(shell_b.angular_momentum(), shell_b.n_primitives());
    bra_set.add_shell(shell_a);
    ket_set.add_shell(shell_b);

    basis::ShellSetDeviceData bra_data = basis::upload_shell_set(bra_set, s);
    basis::ShellSetDeviceData ket_data = basis::upload_shell_set(ket_set, s);

    basis::ShellSetPairDeviceData pair;
    pair.bra = bra_data;
    pair.ket = ket_data;

    operators::DevicePointChargeData charge_data =
        operators::upload_point_charges(
            charges.x, charges.y, charges.z, charges.charge, s);

    size_t output_size = kernels::cuda::nuclear_output_size(pair);
    memory::DeviceBuffer<double> d_output(output_size);

    dispatch_1e_optimal(OperatorKind::Nuclear, pair, d_output.data(), s, &charge_data);

    const int na = shell_a.n_functions();
    const int nb = shell_b.n_functions();
    buffer.resize(na, nb);

    std::vector<double> host_output(output_size);
    d_output.download(host_output.data(), output_size);
    cudaStreamSynchronize(s);

    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            buffer(a, b) = host_output[a * nb + b];
        }
    }

    basis::free_shell_set_device_data(bra_data);
    basis::free_shell_set_device_data(ket_data);
    operators::free_point_charge_device_data(charge_data);
}

// =============================================================================
// Shell Quartet Computation (Two-Electron)
// =============================================================================

void CudaEngine::compute_eri_shell_quartet(const Shell& shell_a,
                                           const Shell& shell_b,
                                           const Shell& shell_c,
                                           const Shell& shell_d,
                                           TwoElectronBuffer<0>& buffer) {
    memory::ScopedGpuSlot scoped(*slot_pool_);
    auto& slot = scoped.slot();
    cudaStream_t s = slot.stream.get();

    ShellSet set_a(shell_a.angular_momentum(), shell_a.n_primitives());
    ShellSet set_b(shell_b.angular_momentum(), shell_b.n_primitives());
    ShellSet set_c(shell_c.angular_momentum(), shell_c.n_primitives());
    ShellSet set_d(shell_d.angular_momentum(), shell_d.n_primitives());
    set_a.add_shell(shell_a);
    set_b.add_shell(shell_b);
    set_c.add_shell(shell_c);
    set_d.add_shell(shell_d);

    basis::ShellSetDeviceData data_a = basis::upload_shell_set(set_a, s);
    basis::ShellSetDeviceData data_b = basis::upload_shell_set(set_b, s);
    basis::ShellSetDeviceData data_c = basis::upload_shell_set(set_c, s);
    basis::ShellSetDeviceData data_d = basis::upload_shell_set(set_d, s);

    basis::ShellSetQuartetDeviceData quartet;
    quartet.a = data_a;
    quartet.b = data_b;
    quartet.c = data_c;
    quartet.d = data_d;

    size_t output_size = kernels::cuda::eri_output_size(quartet);
    memory::DeviceBuffer<double> d_output(output_size);

    dispatch_2e_optimal(quartet, d_output.data(), s);

    const int na = shell_a.n_functions();
    const int nb = shell_b.n_functions();
    const int nc = shell_c.n_functions();
    const int nd = shell_d.n_functions();
    buffer.resize(na, nb, nc, nd);

    std::vector<double> host_output(output_size);
    d_output.download(host_output.data(), output_size);
    cudaStreamSynchronize(s);

    int idx = 0;
    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            for (int c = 0; c < nc; ++c) {
                for (int d = 0; d < nd; ++d) {
                    buffer(a, b, c, d) = host_output[idx++];
                }
            }
        }
    }

    basis::free_shell_set_device_data(data_a);
    basis::free_shell_set_device_data(data_b);
    basis::free_shell_set_device_data(data_c);
    basis::free_shell_set_device_data(data_d);
}

// =============================================================================
// Generic Dispatch Methods (for EngineBackend concept)
// =============================================================================

void CudaEngine::compute_1e_shell_pair(const Operator& op,
                                        const Shell& shell_a,
                                        const Shell& shell_b,
                                        OneElectronBuffer<0>& buffer) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    switch (op.kind()) {
        case OperatorKind::Overlap:
            compute_overlap_shell_pair(shell_a, shell_b, buffer);
            break;

        case OperatorKind::Kinetic:
            compute_kinetic_shell_pair(shell_a, shell_b, buffer);
            break;

        case OperatorKind::Nuclear:
        case OperatorKind::PointCharge: {
            const auto& charges = op.params_as<PointChargeParams>();
            compute_nuclear_shell_pair(shell_a, shell_b, charges, buffer);
            break;
        }

        default:
            throw std::invalid_argument(
                "CudaEngine: Unsupported operator kind for one-electron compute: " +
                std::string(to_string(op.kind())));
    }
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
}

void CudaEngine::compute_2e_shell_quartet(const Operator& op,
                                           const Shell& shell_a,
                                           const Shell& shell_b,
                                           const Shell& shell_c,
                                           const Shell& shell_d,
                                           TwoElectronBuffer<0>& buffer) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    switch (op.kind()) {
        case OperatorKind::Coulomb:
            compute_eri_shell_quartet(shell_a, shell_b, shell_c, shell_d, buffer);
            break;

        default:
            throw std::invalid_argument(
                "CudaEngine: Unsupported operator kind for two-electron compute: " +
                std::string(to_string(op.kind())));
    }
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
}

void CudaEngine::compute_shell_set_pair(const Operator& op,
                                         const ShellSetPair& pair,
                                         std::vector<Real>& result) {
    // Phase 4.5: True batched execution
    // Single kernel launch for entire ShellSetPair instead of per-pair launches

    const auto& set_a = pair.shell_set_a();
    const auto& set_b = pair.shell_set_b();

    // Handle empty case
    if (set_a.n_shells() == 0 || set_b.n_shells() == 0) {
        return;
    }

    // --- Small-batch CPU fallback guard (Step 10.2) ---
    const Size n_pairs = set_a.n_shells() * set_b.n_shells();
    if (n_pairs < min_gpu_batch_size_ && cpu_fallback_ != nullptr) {
#ifdef LIBACCINT_TRACE_DISPATCH
        LIBACCINT_LOG_DEBUG("CudaEngine",
            "Small batch (n_pairs=" + std::to_string(n_pairs) +
            " < min=" + std::to_string(min_gpu_batch_size_) +
            "), falling back to CPU");
#endif
        cpu_fallback_->compute_shell_set_pair(op, pair, result);
        // CPU fallback only writes (fi,fj). For cross-pairs, also fill
        // the transpose so callers don't need a separate symmetry pass.
        if (&set_a != &set_b) {
            const Size nbf = basis_->n_basis_functions();
            const int na_funcs = n_cartesian(set_a.angular_momentum());
            const int nb_funcs = n_cartesian(set_b.angular_momentum());
            for (Size si = 0; si < set_a.n_shells(); ++si) {
                const Index fi = set_a.shell(si).function_index();
                for (Size sj = 0; sj < set_b.n_shells(); ++sj) {
                    const Index fj = set_b.shell(sj).function_index();
                    for (int a = 0; a < na_funcs; ++a) {
                        for (int b = 0; b < nb_funcs; ++b) {
                            const Size row = static_cast<Size>(fi + a);
                            const Size col = static_cast<Size>(fj + b);
                            if (row != col) {
                                result[col * nbf + row] = result[row * nbf + col];
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // Acquire an execution slot for this computation
    memory::ScopedGpuSlot scoped(*slot_pool_);
    auto& slot = scoped.slot();
    cudaStream_t s = slot.stream.get();

    // Get cached device data for both ShellSets (avoids redundant uploads)
    const auto& bra_data = shell_cache_->get_or_upload(set_a, s);
    const auto& ket_data = shell_cache_->get_or_upload(set_b, s);

    // Build pair device data
    basis::ShellSetPairDeviceData pair_device;
    pair_device.bra = bra_data;
    pair_device.ket = ket_data;

    // Calculate output buffer size and ensure capacity
    size_t output_size = 0;
    switch (op.kind()) {
        case OperatorKind::Overlap:
            output_size = kernels::cuda::overlap_output_size(pair_device);
            break;
        case OperatorKind::Kinetic:
            output_size = kernels::cuda::kinetic_output_size(pair_device);
            break;
        case OperatorKind::Nuclear:
        case OperatorKind::PointCharge:
            output_size = kernels::cuda::nuclear_output_size(pair_device);
            break;
        default:
            throw std::invalid_argument(
                "CudaEngine::compute_shell_set_pair: Unsupported operator kind: " +
                std::string(to_string(op.kind())));
    }

    slot.ensure_1e_buffer(output_size);

    // Single kernel dispatch for the entire batch via optimal dispatch table
    switch (op.kind()) {
        case OperatorKind::Overlap:
            dispatch_1e_optimal(OperatorKind::Overlap, pair_device, slot.d_1e_output, s);
            break;
        case OperatorKind::Kinetic:
            dispatch_1e_optimal(OperatorKind::Kinetic, pair_device, slot.d_1e_output, s);
            break;
        case OperatorKind::Nuclear:
        case OperatorKind::PointCharge: {
            const auto& charges = op.params_as<PointChargeParams>();
            operators::DevicePointChargeData charge_data =
                operators::upload_point_charges(
                    charges.x, charges.y, charges.z, charges.charge, s);
            dispatch_1e_optimal(OperatorKind::Nuclear, pair_device, slot.d_1e_output, s, &charge_data);
            operators::free_point_charge_device_data(charge_data);
            break;
        }
        default:
            break;  // Already handled above
    }

    // Single download to host staging buffer
    slot.h_1e_staging.resize(output_size);
    cudaMemcpyAsync(slot.h_1e_staging.data(), slot.d_1e_output,
                    output_size * sizeof(double), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    // Scatter from flat buffer to result matrix
    scatter_1e_to_matrix(pair, slot.h_1e_staging, result);
}

// =============================================================================
// Full Matrix Computation
// =============================================================================

void CudaEngine::compute_overlap_matrix(std::vector<Real>& result) {
    const Size nbf = basis_->n_basis_functions();
    result.assign(nbf * nbf, Real{0.0});
    if (basis_->n_shells() == 0) return;

    Operator op = Operator::overlap();
    for (const auto& ssp : basis_->shell_set_pairs()) {
        compute_shell_set_pair(op, ssp, result);
    }
}

void CudaEngine::compute_kinetic_matrix(std::vector<Real>& result) {
    const Size nbf = basis_->n_basis_functions();
    result.assign(nbf * nbf, Real{0.0});
    if (basis_->n_shells() == 0) return;

    Operator op = Operator::kinetic();
    for (const auto& ssp : basis_->shell_set_pairs()) {
        compute_shell_set_pair(op, ssp, result);
    }
}

void CudaEngine::compute_nuclear_matrix(const PointChargeParams& charges,
                                        std::vector<Real>& result) {
    const Size nbf = basis_->n_basis_functions();
    result.assign(nbf * nbf, Real{0.0});
    if (basis_->n_shells() == 0) return;

    Operator op = Operator::nuclear(charges);
    for (const auto& ssp : basis_->shell_set_pairs()) {
        compute_shell_set_pair(op, ssp, result);
    }
}

void CudaEngine::compute_all_1e_fused(const PointChargeParams& charges,
                                       std::vector<Real>& S_result,
                                       std::vector<Real>& T_result,
                                       std::vector<Real>& V_result) {
    const Size nbf = basis_->n_basis_functions();
    S_result.assign(nbf * nbf, Real{0.0});
    T_result.assign(nbf * nbf, Real{0.0});
    V_result.assign(nbf * nbf, Real{0.0});

    if (basis_->n_shells() == 0) return;

    // Upload charges to device once
    operators::DevicePointChargeData charge_data =
        operators::upload_point_charges(
            charges.x, charges.y, charges.z, charges.charge, stream_);

    // Acquire a single slot for the entire fused computation
    // (this method iterates over shell set pairs serially)
    memory::ScopedGpuSlot scoped(*slot_pool_);
    auto& slot = scoped.slot();
    cudaStream_t s = slot.stream.get();

    for (const auto& ssp : basis_->shell_set_pairs()) {
        const auto& set_a = ssp.shell_set_a();
        const auto& set_b = ssp.shell_set_b();

        if (set_a.n_shells() == 0 || set_b.n_shells() == 0) continue;

        // Get cached device data
        const auto& bra_data = shell_cache_->get_or_upload(set_a, s);
        const auto& ket_data = shell_cache_->get_or_upload(set_b, s);

        basis::ShellSetPairDeviceData pair_device;
        pair_device.bra = bra_data;
        pair_device.ket = ket_data;

        // Calculate per-operator output size
        size_t per_op_size = kernels::cuda::fused_1e_output_size(pair_device);
        slot.ensure_fused_1e_buffer(per_op_size);

        // Set up contiguous output pointers: [S_data | T_data | V_data]
        kernels::cuda::Fused1eOutputPointers output;
        output.d_overlap = slot.d_fused_1e_output;
        output.d_kinetic = slot.d_fused_1e_output + per_op_size;
        output.d_nuclear = slot.d_fused_1e_output + 2 * per_op_size;

        // Check dispatch table: fused or 3 individual kernels?
        const int la = set_a.angular_momentum();
        const int lb = set_b.angular_momentum();
        if (dispatch_table_->get_1e(la, lb).prefer_fused) {
            // Single fused kernel launch
            kernels::cuda::dispatch_fused_1e_kernel(pair_device, charge_data, d_boys_coeffs_,
                                                     output, s);
        } else {
            // Three individual optimal kernel launches
            dispatch_1e_optimal(OperatorKind::Overlap, pair_device, output.d_overlap, s);
            dispatch_1e_optimal(OperatorKind::Kinetic, pair_device, output.d_kinetic, s);
            dispatch_1e_optimal(OperatorKind::Nuclear, pair_device, output.d_nuclear, s, &charge_data);
        }

        // Single batched download (all three results contiguous)
        size_t total_size = per_op_size * 3;
        slot.h_1e_staging.resize(total_size);
        cudaMemcpyAsync(slot.h_1e_staging.data(), slot.d_fused_1e_output,
                        total_size * sizeof(double), cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);

        // Scatter from flat buffers to result matrices
        const int na_funcs = n_cartesian(set_a.angular_momentum());
        const int nb_funcs = n_cartesian(set_b.angular_momentum());
        const Size funcs_per_pair = static_cast<Size>(na_funcs * nb_funcs);

        size_t flat_idx = 0;
        for (Size i = 0; i < set_a.n_shells(); ++i) {
            const auto& shell_a = set_a.shell(i);
            const Index fi = shell_a.function_index();

            for (Size j = 0; j < set_b.n_shells(); ++j) {
                const auto& shell_b = set_b.shell(j);
                const Index fj = shell_b.function_index();

                for (int a = 0; a < na_funcs; ++a) {
                    for (int b = 0; b < nb_funcs; ++b) {
                        size_t idx = flat_idx + a * nb_funcs + b;
                        Size mat_idx = static_cast<Size>(fi + a) * nbf + static_cast<Size>(fj + b);
                        S_result[mat_idx] += slot.h_1e_staging[idx];
                        T_result[mat_idx] += slot.h_1e_staging[per_op_size + idx];
                        V_result[mat_idx] += slot.h_1e_staging[2 * per_op_size + idx];
                    }
                }
                flat_idx += funcs_per_pair;
            }
        }
    }

    operators::free_point_charge_device_data(charge_data);

    // Fill symmetric counterparts
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            S_result[j * nbf + i] = S_result[i * nbf + j];
            T_result[j * nbf + i] = T_result[i * nbf + j];
            V_result[j * nbf + i] = V_result[i * nbf + j];
        }
    }
}

void CudaEngine::compute_core_hamiltonian(const PointChargeParams& charges,
                                          std::vector<Real>& result) {
    const Size nbf = basis_->n_basis_functions();

    // Compute T
    compute_kinetic_matrix(result);

    // Compute V and add to result
    std::vector<Real> V;
    compute_nuclear_matrix(charges, V);

    // H = T + V
    for (Size i = 0; i < nbf * nbf; ++i) {
        result[i] += V[i];
    }
}

// =============================================================================
// ShellSetQuartet-Based Computation (Phase 4.5: True Batching)
// =============================================================================

CudaEngine::DeviceEriBatch CudaEngine::compute_eri_batch_device_handle(
    const ShellSetQuartet& quartet) {
    auto scoped = std::make_unique<memory::ScopedGpuSlot>(*slot_pool_);
    double* d_eri_output = nullptr;
    size_t eri_count = 0;
    compute_eri_batch_device(quartet, d_eri_output, eri_count, scoped->slot());
    return DeviceEriBatch(std::move(scoped), d_eri_output, eri_count);
}

void CudaEngine::compute_eri_batch_device(const ShellSetQuartet&,
                                           double*&,
                                           size_t&) {
    throw InvalidStateException(
        "CudaEngine::compute_eri_batch_device raw-pointer overload is unsupported "
        "for alpha because the returned pointer cannot own the execution slot; "
        "use compute_eri_batch_device_handle() instead");
}

void CudaEngine::compute_eri_batch_device(const ShellSetQuartet& quartet,
                                           double*& d_eri_output,
                                           size_t& eri_count,
                                           memory::GpuExecutionSlot& slot) {
    // Get ShellSets from the quartet
    const auto& set_a = quartet.bra_pair().shell_set_a();
    const auto& set_b = quartet.bra_pair().shell_set_b();
    const auto& set_c = quartet.ket_pair().shell_set_a();
    const auto& set_d = quartet.ket_pair().shell_set_b();

    // Handle empty case
    if (set_a.n_shells() == 0 || set_b.n_shells() == 0 ||
        set_c.n_shells() == 0 || set_d.n_shells() == 0) {
        d_eri_output = nullptr;
        eri_count = 0;
        return;
    }

    cudaStream_t s = slot.stream.get();

    // Get cached device data for all four ShellSets
    const auto& data_a = shell_cache_->get_or_upload(set_a, s);
    const auto& data_b = shell_cache_->get_or_upload(set_b, s);
    const auto& data_c = shell_cache_->get_or_upload(set_c, s);
    const auto& data_d = shell_cache_->get_or_upload(set_d, s);

    // Build quartet device data
    basis::ShellSetQuartetDeviceData quartet_device;
    quartet_device.a = data_a;
    quartet_device.b = data_b;
    quartet_device.c = data_c;
    quartet_device.d = data_d;

    // Calculate output buffer size and ensure capacity
    size_t output_size = kernels::cuda::eri_output_size(quartet_device);
    slot.ensure_2e_buffer(output_size);

    // Single kernel launch for entire batch via optimal dispatch - results stay on device
    dispatch_2e_optimal(quartet_device, slot.d_2e_output, s);

    // Return device pointer and count (no download)
    d_eri_output = slot.d_2e_output;
    eri_count = output_size;
}

// =============================================================================
// PipelineSlot Implementation
// =============================================================================

CudaEngine::PipelineSlot::PipelineSlot() = default;

CudaEngine::PipelineSlot::~PipelineSlot() {
    if (d_output != nullptr) {
        cudaFree(d_output);
        d_output = nullptr;
    }
}

CudaEngine::PipelineSlot::PipelineSlot(PipelineSlot&& other) noexcept
    : d_output(other.d_output),
      h_pinned(std::move(other.h_pinned)),
      capacity(other.capacity),
      output_size(other.output_size),
      kernel_done(std::move(other.kernel_done)),
      download_done(std::move(other.download_done)),
      quartet(other.quartet),
      in_flight(other.in_flight) {
    other.d_output = nullptr;
    other.capacity = 0;
    other.output_size = 0;
    other.quartet = nullptr;
    other.in_flight = false;
}

CudaEngine::PipelineSlot& CudaEngine::PipelineSlot::operator=(PipelineSlot&& other) noexcept {
    if (this != &other) {
        if (d_output != nullptr) {
            cudaFree(d_output);
        }
        d_output = other.d_output;
        h_pinned = std::move(other.h_pinned);
        capacity = other.capacity;
        output_size = other.output_size;
        kernel_done = std::move(other.kernel_done);
        download_done = std::move(other.download_done);
        quartet = other.quartet;
        in_flight = other.in_flight;

        other.d_output = nullptr;
        other.capacity = 0;
        other.output_size = 0;
        other.quartet = nullptr;
        other.in_flight = false;
    }
    return *this;
}

void CudaEngine::PipelineSlot::ensure_capacity(size_t required) {
    if (required <= capacity) return;

    // Allocate with 2x headroom to amortize reallocations
    size_t new_capacity = required * 2;

    // Reallocate device buffer
    if (d_output != nullptr) {
        cudaFree(d_output);
    }
    cudaError_t err = cudaMalloc(&d_output, new_capacity * sizeof(double));
    if (err != cudaSuccess) {
        d_output = nullptr;
        capacity = 0;
        throw BackendError(BackendType::CUDA,
            std::string("Failed to allocate pipeline slot device buffer: ") + cudaGetErrorString(err));
    }

    // Reallocate pinned host buffer
    h_pinned.reset(new_capacity);

    capacity = new_capacity;
}

// =============================================================================
// Pipelined ERI Computation
// =============================================================================

void CudaEngine::scatter_eri_slot_to_tensor(const PipelineSlot& slot,
                                             std::vector<double>& result) {
    const auto& q = *slot.quartet;
    const Size nbf = basis_->n_basis_functions();

    const auto& set_a = q.bra_pair().shell_set_a();
    const auto& set_b = q.bra_pair().shell_set_b();
    const auto& set_c = q.ket_pair().shell_set_a();
    const auto& set_d = q.ket_pair().shell_set_b();

    const int na_funcs = n_cartesian(set_a.angular_momentum());
    const int nb_funcs = n_cartesian(set_b.angular_momentum());
    const int nc_funcs = n_cartesian(set_c.angular_momentum());
    const int nd_funcs = n_cartesian(set_d.angular_momentum());

    const size_t funcs_per_quartet = static_cast<size_t>(na_funcs) * nb_funcs * nc_funcs * nd_funcs;

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

                    // Scatter from flat pinned buffer into 4-index result tensor
                    for (int a = 0; a < na_funcs; ++a) {
                        for (int b = 0; b < nb_funcs; ++b) {
                            for (int c = 0; c < nc_funcs; ++c) {
                                for (int d = 0; d < nd_funcs; ++d) {
                                    const size_t src_idx = flat_idx +
                                        a * nb_funcs * nc_funcs * nd_funcs +
                                        b * nc_funcs * nd_funcs + c * nd_funcs + d;
                                    const size_t dst_idx =
                                        static_cast<size_t>(fi + a) * nbf * nbf * nbf +
                                        static_cast<size_t>(fj + b) * nbf * nbf +
                                        static_cast<size_t>(fk + c) * nbf +
                                        static_cast<size_t>(fl + d);
                                    result[dst_idx] += slot.h_pinned.data()[src_idx];
                                }
                            }
                        }
                    }

                    flat_idx += funcs_per_quartet;
                }
            }
        }
    }
}

void CudaEngine::compute_eri_pipelined(std::vector<double>& result,
                                        const EriPipelineConfig& config) {
    const Size nbf = basis_->n_basis_functions();
    const std::size_t tensor_size = checked_eri_tensor_elements(
        nbf, "CudaEngine::compute_eri_pipelined");
    validate_eri_tensor_limit(nbf, tensor_size, "CudaEngine::compute_eri_pipelined");
    result.assign(tensor_size, 0.0);

    compute_eri_pipelined_impl(
        [this, &result](PipelineSlot& slot) {
            scatter_eri_slot_to_tensor(slot, result);
        },
        config);
}

void CudaEngine::compute_eri_pipelined(EriCallback callback,
                                        const EriPipelineConfig& config) {
    compute_eri_pipelined_impl(
        [&callback](PipelineSlot& slot) {
            callback(
                std::span<const double>(slot.h_pinned.data(), slot.output_size),
                *slot.quartet);
        },
        config);
}

void CudaEngine::compute_eri_device_scatter(std::vector<double>& result,
                                             const EriPipelineConfig& config) {
    const Size nbf = basis_->n_basis_functions();
    const std::size_t tensor_size = checked_eri_tensor_elements(
        nbf, "CudaEngine::compute_eri_device_scatter");
    validate_eri_tensor_limit(nbf, tensor_size, "CudaEngine::compute_eri_device_scatter");
    validate_pipeline_slot_count(config.n_slots, "CudaEngine::compute_eri_device_scatter");
    result.assign(tensor_size, 0.0);

    const auto& quartets = basis_->shell_set_quartets();
    if (quartets.empty()) return;

    // Allocate device tensor and zero it
    double* d_tensor = nullptr;
    cudaMalloc(&d_tensor, tensor_size * sizeof(double));
    cudaMemsetAsync(d_tensor, 0, tensor_size * sizeof(double), stream_);
    cudaStreamSynchronize(stream_);

    const size_t n_streams = config.n_slots;
    memory::StreamPool compute_pool(n_streams);

    // Track per-stream ERI output buffers (reused across quartets)
    struct StreamSlot {
        double* d_eri = nullptr;
        size_t capacity = 0;
    };
    std::vector<StreamSlot> stream_slots(n_streams);
    std::vector<memory::StreamHandle*> stream_handles(n_streams, nullptr);
    std::vector<memory::EventHandle> events(n_streams);

    size_t next_stream = 0;
    size_t in_flight = 0;
    size_t oldest = 0;

    for (const auto& q : quartets) {
        const auto& set_a = q.bra_pair().shell_set_a();
        const auto& set_b = q.bra_pair().shell_set_b();
        const auto& set_c = q.ket_pair().shell_set_a();
        const auto& set_d = q.ket_pair().shell_set_b();

        if (set_a.n_shells() == 0 || set_b.n_shells() == 0 ||
            set_c.n_shells() == 0 || set_d.n_shells() == 0) {
            continue;
        }

        // If all streams busy, wait for the oldest
        if (in_flight == n_streams) {
            events[oldest].synchronize();
            if (stream_handles[oldest]) {
                compute_pool.release(*stream_handles[oldest]);
                stream_handles[oldest] = nullptr;
            }
            in_flight--;
            oldest = (oldest + 1) % n_streams;
        }

        auto& slot = stream_slots[next_stream];
        auto& compute_stream = compute_pool.acquire();
        stream_handles[next_stream] = &compute_stream;

        // Upload shell data
        const auto& data_a = shell_cache_->get_or_upload(set_a, compute_stream.get());
        const auto& data_b = shell_cache_->get_or_upload(set_b, compute_stream.get());
        const auto& data_c = shell_cache_->get_or_upload(set_c, compute_stream.get());
        const auto& data_d = shell_cache_->get_or_upload(set_d, compute_stream.get());

        basis::ShellSetQuartetDeviceData quartet_device;
        quartet_device.a = data_a;
        quartet_device.b = data_b;
        quartet_device.c = data_c;
        quartet_device.d = data_d;

        // Calculate output size and ensure buffer capacity
        size_t output_size = kernels::cuda::eri_output_size(quartet_device);
        if (slot.capacity < output_size) {
            if (slot.d_eri) cudaFree(slot.d_eri);
            size_t alloc_size = output_size * 2;  // 2x headroom
            cudaMalloc(&slot.d_eri, alloc_size * sizeof(double));
            slot.capacity = alloc_size;
        }

        // Launch ERI kernel via optimal dispatch
        dispatch_2e_optimal(quartet_device, slot.d_eri, compute_stream.get());

        // Launch scatter kernel on same stream (serialized after ERI kernel)
        kernels::cuda::launch_eri_scatter_kernel(
            slot.d_eri, quartet_device, d_tensor,
            static_cast<int>(nbf), compute_stream.get());

        // Record completion event
        events[next_stream].record(compute_stream.get());

        in_flight++;
        next_stream = (next_stream + 1) % n_streams;
    }

    // Wait for all remaining in-flight work
    for (size_t i = 0; i < in_flight; ++i) {
        events[oldest].synchronize();
        if (stream_handles[oldest]) {
            compute_pool.release(*stream_handles[oldest]);
            stream_handles[oldest] = nullptr;
        }
        oldest = (oldest + 1) % n_streams;
    }

    // Single bulk D2H transfer
    cudaMemcpy(result.data(), d_tensor, tensor_size * sizeof(double),
               cudaMemcpyDeviceToHost);

    // Cleanup
    for (auto& slot : stream_slots) {
        if (slot.d_eri) {
            cudaFree(slot.d_eri);
            slot.d_eri = nullptr;
        }
    }
    cudaFree(d_tensor);
}

}  // namespace libaccint

#endif  // LIBACCINT_USE_CUDA
