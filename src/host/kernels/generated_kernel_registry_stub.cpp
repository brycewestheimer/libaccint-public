// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file generated_kernel_registry_stub.cpp
/// @brief Low-memory fallback implementation of generated CPU kernel registry.
///
/// This stub keeps the public generated-registry API linkable while forcing
/// runtime dispatch to handwritten kernels by returning nullptr for all lookups.

#include <libaccint/kernels/generated_kernel_registry.hpp>

namespace libaccint::kernels::cpu::generated {

OneElectronKernelFn get_generated_overlap(int, int) noexcept {
    return nullptr;
}

OneElectronKernelFn get_generated_kinetic(int, int) noexcept {
    return nullptr;
}

NuclearKernelFn get_generated_nuclear(int, int) noexcept {
    return nullptr;
}

TwoElectronKernelFn get_generated_eri(int, int, int, int) noexcept {
    return nullptr;
}

OneElectronKernelSpanFn get_generated_overlap(int, int, ContractionRange) noexcept {
    return nullptr;
}

OneElectronKernelSpanFn get_generated_kinetic(int, int, ContractionRange) noexcept {
    return nullptr;
}

NuclearKernelSpanFn get_generated_nuclear(int, int, ContractionRange) noexcept {
    return nullptr;
}

TwoElectronKernelSpanFn get_generated_eri(int, int, int, int, ContractionRange) noexcept {
    return nullptr;
}

}  // namespace libaccint::kernels::cpu::generated

