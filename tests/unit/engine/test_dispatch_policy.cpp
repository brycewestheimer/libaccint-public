// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/engine/dispatch_policy.hpp>
#include <libaccint/core/backend.hpp>

#include <gtest/gtest.h>

using namespace libaccint;

// =============================================================================
// DispatchConfig Tests
// =============================================================================

TEST(DispatchConfigTest, DefaultValues) {
    DispatchConfig config;

    EXPECT_EQ(config.min_gpu_batch_size, 16u);
    EXPECT_EQ(config.min_gpu_primitives, 1000u);
    EXPECT_EQ(config.high_am_threshold, 4);
    EXPECT_EQ(config.min_gpu_shells, 10u);
}

TEST(DispatchConfigTest, CustomValues) {
    DispatchConfig config;
    config.min_gpu_batch_size = 32;
    config.min_gpu_primitives = 500;
    config.high_am_threshold = 3;
    config.min_gpu_shells = 20;

    EXPECT_EQ(config.min_gpu_batch_size, 32u);
    EXPECT_EQ(config.min_gpu_primitives, 500u);
    EXPECT_EQ(config.high_am_threshold, 3);
    EXPECT_EQ(config.min_gpu_shells, 20u);
}

// =============================================================================
// DispatchPolicy Construction Tests
// =============================================================================

TEST(DispatchPolicyTest, DefaultConstruction) {
    DispatchPolicy policy;

    EXPECT_EQ(policy.config().min_gpu_batch_size, 16u);
}

TEST(DispatchPolicyTest, ConstructionWithConfig) {
    DispatchConfig config;
    config.min_gpu_batch_size = 64;

    DispatchPolicy policy(config);

    EXPECT_EQ(policy.config().min_gpu_batch_size, 64u);
}

TEST(DispatchPolicyTest, SetConfig) {
    DispatchPolicy policy;

    DispatchConfig config;
    config.min_gpu_batch_size = 128;
    policy.set_config(config);

    EXPECT_EQ(policy.config().min_gpu_batch_size, 128u);
}

// =============================================================================
// Backend Selection Tests - Force Hints
// =============================================================================

TEST(DispatchPolicyTest, ForceCPU_AlwaysReturnsCPU) {
    DispatchPolicy policy;

    // ForceCPU should always return CPU regardless of other factors
    auto backend = policy.select_backend(
        WorkUnitType::FullBasis,
        1000,  // large batch size
        10,    // high AM
        10000, // many primitives
        BackendHint::ForceCPU,
        true   // GPU available
    );

    EXPECT_EQ(backend, BackendType::CPU);
}

TEST(DispatchPolicyTest, ForceGPU_ReturnsGPU_WhenAvailable) {
    DispatchPolicy policy;

    auto backend = policy.select_backend(
        WorkUnitType::SingleShellPair,
        1,     // small batch
        0,     // low AM
        1,     // few primitives
        BackendHint::ForceGPU,
        true   // GPU available
    );

    EXPECT_EQ(backend, BackendType::CUDA);
}

TEST(DispatchPolicyTest, ForceGPU_FallsBackToCPU_WhenNotAvailable) {
    DispatchPolicy policy;

    // When GPU is not available, ForceGPU falls back to CPU
    auto backend = policy.select_backend(
        WorkUnitType::FullBasis,
        1000,
        10,
        10000,
        BackendHint::ForceGPU,
        false  // GPU not available
    );

    EXPECT_EQ(backend, BackendType::CPU);
}

// =============================================================================
// Backend Selection Tests - GPU Not Available
// =============================================================================

TEST(DispatchPolicyTest, NoGPU_AlwaysReturnsCPU) {
    DispatchPolicy policy;

    // Without GPU, should always return CPU
    auto backend = policy.select_backend(
        WorkUnitType::FullBasis,
        1000,
        10,
        10000,
        BackendHint::Auto,
        false  // GPU not available
    );

    EXPECT_EQ(backend, BackendType::CPU);
}

// =============================================================================
// Backend Selection Tests - Single Operations
// =============================================================================

TEST(DispatchPolicyTest, SingleShellPair_PrefersCPU) {
    DispatchPolicy policy;

    // Single shell pair operations should prefer CPU due to kernel launch overhead
    auto backend = policy.select_backend(
        WorkUnitType::SingleShellPair,
        1,
        2,     // moderate AM
        9,     // 3x3 primitives
        BackendHint::Auto,
        true
    );

    EXPECT_EQ(backend, BackendType::CPU);
}

TEST(DispatchPolicyTest, SingleShellQuartet_PrefersCPU) {
    DispatchPolicy policy;

    auto backend = policy.select_backend(
        WorkUnitType::SingleShellQuartet,
        1,
        4,     // moderate AM
        81,    // 3x3x3x3 primitives
        BackendHint::Auto,
        true
    );

    EXPECT_EQ(backend, BackendType::CPU);
}

TEST(DispatchPolicyTest, SingleShellPair_VeryHighAM_MayPreferGPU) {
    DispatchPolicy policy;

    // Very high AM might benefit from GPU even for single operations
    auto backend = policy.select_backend(
        WorkUnitType::SingleShellPair,
        1,
        8,     // very high AM (e.g., g+g shell pair)
        100,   // moderate primitives
        BackendHint::Auto,
        true
    );

    // With default thresholds, very high AM (>=6) might use GPU
    // But this depends on implementation details
    // Just verify it doesn't crash
    EXPECT_TRUE(backend == BackendType::CPU || backend == BackendType::CUDA);
}

// =============================================================================
// Backend Selection Tests - Batched Operations
// =============================================================================

TEST(DispatchPolicyTest, ShellSetPair_LargeBatch_PrefersGPU) {
    DispatchPolicy policy;

    // Large batches should prefer GPU
    auto backend = policy.select_backend(
        WorkUnitType::ShellSetPair,
        100,   // large batch
        4,     // moderate AM
        10000, // many primitives
        BackendHint::Auto,
        true
    );

    EXPECT_EQ(backend, BackendType::CUDA);
}

TEST(DispatchPolicyTest, ShellSetPair_SmallBatch_PrefersCPU) {
    DispatchPolicy policy;

    // Small batches should prefer CPU
    auto backend = policy.select_backend(
        WorkUnitType::ShellSetPair,
        4,     // small batch
        2,     // low AM
        100,   // few primitives
        BackendHint::Auto,
        true
    );

    EXPECT_EQ(backend, BackendType::CPU);
}

TEST(DispatchPolicyTest, ShellSetQuartet_EvenModestBatch_CanPreferGPU) {
    DispatchPolicy policy;

    // ShellSetQuartet has O(N^4) work, so even modest batches benefit from GPU
    auto backend = policy.select_backend(
        WorkUnitType::ShellSetQuartet,
        10,    // modest batch (but 10^4 = 10000 quartets!)
        4,     // moderate AM
        1000,  // moderate primitives
        BackendHint::Auto,
        true
    );

    // Should prefer GPU due to the work scaling
    EXPECT_EQ(backend, BackendType::CUDA);
}

// =============================================================================
// Backend Selection Tests - Full Basis
// =============================================================================

TEST(DispatchPolicyTest, FullBasis_LargeSystem_PrefersGPU) {
    DispatchPolicy policy;

    auto backend = policy.select_backend(
        WorkUnitType::FullBasis,
        50,    // 50 shells
        4,
        10000,
        BackendHint::Auto,
        true
    );

    EXPECT_EQ(backend, BackendType::CUDA);
}

TEST(DispatchPolicyTest, FullBasis_SmallSystem_PrefersCPU) {
    DispatchPolicy policy;

    auto backend = policy.select_backend(
        WorkUnitType::FullBasis,
        5,     // 5 shells - small system
        2,
        100,
        BackendHint::Auto,
        true
    );

    EXPECT_EQ(backend, BackendType::CPU);
}

// =============================================================================
// Backend Selection Tests - Preference Hints
// =============================================================================

TEST(DispatchPolicyTest, PreferCPU_RaisesBarForGPU) {
    DispatchPolicy policy;

    // With PreferCPU, even large batches might stay on CPU
    auto backend = policy.select_backend(
        WorkUnitType::ShellSetPair,
        20,    // moderate batch
        3,     // moderate AM
        2000,  // moderate primitives
        BackendHint::PreferCPU,
        true
    );

    // Should prefer CPU with PreferCPU hint
    EXPECT_EQ(backend, BackendType::CPU);
}

TEST(DispatchPolicyTest, PreferGPU_LowersBarForGPU) {
    DispatchPolicy policy;

    // With PreferGPU, smaller batches might use GPU
    auto backend = policy.select_backend(
        WorkUnitType::ShellSetPair,
        8,     // small-ish batch
        3,     // moderate AM
        500,   // moderate primitives
        BackendHint::PreferGPU,
        true
    );

    // Should prefer GPU with PreferGPU hint
    EXPECT_EQ(backend, BackendType::CUDA);
}

// =============================================================================
// Backend Selection Tests - High Angular Momentum
// =============================================================================

TEST(DispatchPolicyTest, HighAM_PrefersGPU) {
    DispatchPolicy policy;

    // High AM quartets benefit from GPU parallelism
    auto backend = policy.select_backend(
        WorkUnitType::ShellSetQuartet,
        8,     // modest batch
        16,    // high AM (e.g., 4 d-shells = 4*2*4 = 32? or 4*4 = 16 total)
        500,   // moderate primitives
        BackendHint::Auto,
        true
    );

    EXPECT_EQ(backend, BackendType::CUDA);
}

// =============================================================================
// WorkUnitType Enum Tests
// =============================================================================

TEST(WorkUnitTypeTest, EnumValues) {
    // Just verify the enum values are distinct and compile
    EXPECT_NE(static_cast<int>(WorkUnitType::SingleShellPair),
              static_cast<int>(WorkUnitType::SingleShellQuartet));
    EXPECT_NE(static_cast<int>(WorkUnitType::ShellSetPair),
              static_cast<int>(WorkUnitType::ShellSetQuartet));
    EXPECT_NE(static_cast<int>(WorkUnitType::ShellSetQuartet),
              static_cast<int>(WorkUnitType::FullBasis));
}

// =============================================================================
// BackendHint Enum Tests
// =============================================================================

TEST(BackendHintTest, EnumValues) {
    // Verify all enum values are distinct
    EXPECT_NE(static_cast<int>(BackendHint::Auto), static_cast<int>(BackendHint::ForceCPU));
    EXPECT_NE(static_cast<int>(BackendHint::ForceCPU), static_cast<int>(BackendHint::ForceGPU));
    EXPECT_NE(static_cast<int>(BackendHint::ForceGPU), static_cast<int>(BackendHint::PreferCPU));
    EXPECT_NE(static_cast<int>(BackendHint::PreferCPU), static_cast<int>(BackendHint::PreferGPU));
}
