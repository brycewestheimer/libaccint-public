// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file generated_kernel_registry_stub.cu
/// @brief Lightweight CUDA generated-registry fallback for constrained builds.
///
/// This keeps the generated CUDA registry API linkable while forcing runtime
/// dispatch to handwritten kernels by reporting no generated-kernel availability.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/kernels/generated_kernel_registry_cuda.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <string>

namespace libaccint::kernels::cuda::generated {
namespace {

[[noreturn]] void throw_registry_disabled(const char* api_name) {
    throw NotImplementedException(
        std::string(api_name) +
        ": CUDA generated registry is disabled "
        "(set LIBACCINT_ENABLE_CUDA_GENERATED_REGISTRY=ON to enable)");
}

}  // namespace

bool has_generated_overlap(int, int) noexcept { return false; }
bool has_generated_kinetic(int, int) noexcept { return false; }
bool has_generated_nuclear(int, int) noexcept { return false; }
bool has_generated_eri(int, int, int, int) noexcept { return false; }
bool has_generated_eri_soa(int, int, int, int) noexcept { return false; }
bool has_generated_eri_cooperative(int, int, int, int) noexcept { return false; }

bool has_generated_overlap_k_aware(int, int, kernels::ContractionRange) noexcept {
    return false;
}

bool has_generated_kinetic_k_aware(int, int, kernels::ContractionRange) noexcept {
    return false;
}

bool has_generated_nuclear_k_aware(int, int, kernels::ContractionRange) noexcept {
    return false;
}

bool has_generated_eri_k_aware(
    int, int, int, int, kernels::ContractionRange, kernels::GpuExecutionStrategy) noexcept {
    return false;
}

void launch_generated_overlap(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t) {
    throw_registry_disabled("launch_generated_overlap");
}

void launch_generated_kinetic(
    const basis::ShellSetPairDeviceData&, double*, cudaStream_t) {
    throw_registry_disabled("launch_generated_kinetic");
}

void launch_generated_nuclear(
    const basis::ShellSetPairDeviceData&,
    const operators::DevicePointChargeData&,
    const double*,
    double*,
    cudaStream_t) {
    throw_registry_disabled("launch_generated_nuclear");
}

void launch_generated_eri(
    const basis::ShellSetQuartetDeviceData&, const double*, double*, cudaStream_t) {
    throw_registry_disabled("launch_generated_eri");
}

void launch_generated_eri_soa(
    const basis::ShellSetQuartetDeviceData&, const double*, double*, cudaStream_t) {
    throw_registry_disabled("launch_generated_eri_soa");
}

void launch_generated_eri_cooperative(
    const basis::ShellSetQuartetDeviceData&, const double*, double*, cudaStream_t) {
    throw_registry_disabled("launch_generated_eri_cooperative");
}

void launch_generated_overlap_k_aware(
    const basis::ShellSetPairDeviceData&,
    double*,
    kernels::ContractionRange,
    cudaStream_t,
    const device::BatchConfig*) {
    throw_registry_disabled("launch_generated_overlap_k_aware");
}

void launch_generated_kinetic_k_aware(
    const basis::ShellSetPairDeviceData&,
    double*,
    kernels::ContractionRange,
    cudaStream_t,
    const device::BatchConfig*) {
    throw_registry_disabled("launch_generated_kinetic_k_aware");
}

void launch_generated_nuclear_k_aware(
    const basis::ShellSetPairDeviceData&,
    const operators::DevicePointChargeData&,
    const double*,
    double*,
    kernels::ContractionRange,
    cudaStream_t,
    const device::BatchConfig*) {
    throw_registry_disabled("launch_generated_nuclear_k_aware");
}

void launch_generated_eri_k_aware(
    const basis::ShellSetQuartetDeviceData&,
    const double*,
    double*,
    kernels::ContractionRange,
    kernels::GpuExecutionStrategy,
    cudaStream_t,
    const device::BatchConfig*) {
    throw_registry_disabled("launch_generated_eri_k_aware");
}

}  // namespace libaccint::kernels::cuda::generated

#endif  // LIBACCINT_USE_CUDA

