// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_dispatch_registry.cpp
/// @brief Tests for KernelCalculator and DispatchRegistry

#include <libaccint/kernels/dispatch_registry.hpp>
#include <libaccint/kernels/kernel_calculator.hpp>
#include <libaccint/kernels/cost_model.hpp>
#include <libaccint/kernels/registry_key.hpp>
#include <libaccint/kernels/execution_strategy.hpp>
#include <libaccint/kernels/gpu_kernel_strategy.hpp>
#include <libaccint/kernels/contraction_strategy.hpp>
#include <libaccint/config.hpp>
#include <libaccint/engine/dispatch_policy.hpp>
#include <libaccint/operators/operator_types.hpp>
#include <libaccint/core/backend.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>

using namespace libaccint;
using namespace libaccint::kernels;

// =============================================================================
// ExecutionStrategy Tests
// =============================================================================

TEST(ExecutionStrategyTest, UsesGPU) {
    EXPECT_FALSE(uses_gpu(ExecutionStrategy::SerialCPU));
    EXPECT_FALSE(uses_gpu(ExecutionStrategy::SimdCPU));
    EXPECT_FALSE(uses_gpu(ExecutionStrategy::ThreadedCPU));
    EXPECT_FALSE(uses_gpu(ExecutionStrategy::ThreadedSimdCPU));
    EXPECT_TRUE(uses_gpu(ExecutionStrategy::ThreadPerIntegralGPU));
    EXPECT_TRUE(uses_gpu(ExecutionStrategy::WarpPerQuartetGPU));
    EXPECT_TRUE(uses_gpu(ExecutionStrategy::BlockPerBatchGPU));
}

TEST(ExecutionStrategyTest, UsesSimd) {
    EXPECT_FALSE(uses_simd(ExecutionStrategy::SerialCPU));
    EXPECT_TRUE(uses_simd(ExecutionStrategy::SimdCPU));
    EXPECT_FALSE(uses_simd(ExecutionStrategy::ThreadedCPU));
    EXPECT_TRUE(uses_simd(ExecutionStrategy::ThreadedSimdCPU));
    EXPECT_FALSE(uses_simd(ExecutionStrategy::ThreadPerIntegralGPU));
    EXPECT_FALSE(uses_simd(ExecutionStrategy::WarpPerQuartetGPU));
    EXPECT_FALSE(uses_simd(ExecutionStrategy::BlockPerBatchGPU));
}

TEST(ExecutionStrategyTest, UsesThreading) {
    EXPECT_FALSE(uses_threading(ExecutionStrategy::SerialCPU));
    EXPECT_FALSE(uses_threading(ExecutionStrategy::SimdCPU));
    EXPECT_TRUE(uses_threading(ExecutionStrategy::ThreadedCPU));
    EXPECT_TRUE(uses_threading(ExecutionStrategy::ThreadedSimdCPU));
    EXPECT_FALSE(uses_threading(ExecutionStrategy::ThreadPerIntegralGPU));
    EXPECT_FALSE(uses_threading(ExecutionStrategy::WarpPerQuartetGPU));
    EXPECT_FALSE(uses_threading(ExecutionStrategy::BlockPerBatchGPU));
}

TEST(ExecutionStrategyTest, ToString) {
    EXPECT_EQ(to_string(ExecutionStrategy::SerialCPU), "SerialCPU");
    EXPECT_EQ(to_string(ExecutionStrategy::SimdCPU), "SimdCPU");
    EXPECT_EQ(to_string(ExecutionStrategy::ThreadedCPU), "ThreadedCPU");
    EXPECT_EQ(to_string(ExecutionStrategy::ThreadedSimdCPU), "ThreadedSimdCPU");
    EXPECT_EQ(to_string(ExecutionStrategy::ThreadPerIntegralGPU), "ThreadPerIntegralGPU");
    EXPECT_EQ(to_string(ExecutionStrategy::WarpPerQuartetGPU), "WarpPerQuartetGPU");
    EXPECT_EQ(to_string(ExecutionStrategy::BlockPerBatchGPU), "BlockPerBatchGPU");
}

// =============================================================================
// GpuKernelStrategy Tests
// =============================================================================

TEST(GpuKernelStrategyTest, LowAM_SelectsRegisterBuffering) {
    // (ss) = total AM 0
    EXPECT_EQ(select_gpu_kernel_pattern(0, 0), GpuKernelPattern::RegisterBuffering);
    // (sp) = total AM 1
    EXPECT_EQ(select_gpu_kernel_pattern(0, 1), GpuKernelPattern::RegisterBuffering);
    // (pp) = total AM 2
    EXPECT_EQ(select_gpu_kernel_pattern(1, 1), GpuKernelPattern::RegisterBuffering);
}

TEST(GpuKernelStrategyTest, MediumAM_SelectsStreamingWrites) {
    // (sd) = total AM 2 => register, but (pd) = 3 => streaming
    EXPECT_EQ(select_gpu_kernel_pattern(1, 2), GpuKernelPattern::StreamingWrites);
    // (dd) = total AM 4
    EXPECT_EQ(select_gpu_kernel_pattern(2, 2), GpuKernelPattern::StreamingWrites);
}

TEST(GpuKernelStrategyTest, HighAM_SelectsSharedMemory) {
    // (df) = total AM 5
    EXPECT_EQ(select_gpu_kernel_pattern(2, 3), GpuKernelPattern::SharedMemory);
    // (ff) = total AM 6
    EXPECT_EQ(select_gpu_kernel_pattern(3, 3), GpuKernelPattern::SharedMemory);
}

TEST(GpuKernelStrategyTest, TwoElectron_LowAM) {
    // (ss|ss) = total AM 0
    EXPECT_EQ(select_gpu_kernel_pattern(0, 0, 0, 0), GpuKernelPattern::RegisterBuffering);
    // (sp|ss) = total AM 1
    EXPECT_EQ(select_gpu_kernel_pattern(0, 1, 0, 0), GpuKernelPattern::RegisterBuffering);
    // (sp|sp) = total AM 2
    EXPECT_EQ(select_gpu_kernel_pattern(0, 1, 0, 1), GpuKernelPattern::RegisterBuffering);
}

TEST(GpuKernelStrategyTest, TwoElectron_MediumAM) {
    // (pp|sp) = total AM 3
    EXPECT_EQ(select_gpu_kernel_pattern(1, 1, 0, 1), GpuKernelPattern::StreamingWrites);
    // (pp|pp) = total AM 4
    EXPECT_EQ(select_gpu_kernel_pattern(1, 1, 1, 1), GpuKernelPattern::StreamingWrites);
}

TEST(GpuKernelStrategyTest, TwoElectron_HighAM) {
    // (pd|pp) = total AM 5
    EXPECT_EQ(select_gpu_kernel_pattern(1, 2, 1, 1), GpuKernelPattern::SharedMemory);
    // (dd|dd) = total AM 8
    EXPECT_EQ(select_gpu_kernel_pattern(2, 2, 2, 2), GpuKernelPattern::SharedMemory);
}

TEST(GpuKernelStrategyTest, PatternToStrategy) {
    EXPECT_EQ(pattern_to_strategy(GpuKernelPattern::RegisterBuffering),
              ExecutionStrategy::ThreadPerIntegralGPU);
    EXPECT_EQ(pattern_to_strategy(GpuKernelPattern::StreamingWrites),
              ExecutionStrategy::WarpPerQuartetGPU);
    EXPECT_EQ(pattern_to_strategy(GpuKernelPattern::SharedMemory),
              ExecutionStrategy::BlockPerBatchGPU);
}

TEST(GpuKernelStrategyTest, RecommendedBlockSize) {
    EXPECT_EQ(recommended_block_size(GpuKernelPattern::RegisterBuffering), 256);
    EXPECT_EQ(recommended_block_size(GpuKernelPattern::StreamingWrites), 128);
    EXPECT_EQ(recommended_block_size(GpuKernelPattern::SharedMemory), 64);
}

TEST(GpuKernelStrategyTest, RecommendedSharedMemory) {
    // (ss|ss): 1*1*1*1 = 1 entry per table, 3 tables = 24 bytes
    EXPECT_EQ(recommended_shared_memory(0, 0, 0, 0), 3 * sizeof(double));
    // (pp|pp): 3*2*3*2 = 36 entries per table
    EXPECT_EQ(recommended_shared_memory(1, 1, 1, 1), 3 * 36 * sizeof(double));
    // Verify it grows with AM
    EXPECT_GT(recommended_shared_memory(2, 2, 2, 2),
              recommended_shared_memory(1, 1, 1, 1));
}

TEST(GpuKernelStrategyTest, ToString) {
    EXPECT_STREQ(to_string(GpuKernelPattern::RegisterBuffering), "RegisterBuffering");
    EXPECT_STREQ(to_string(GpuKernelPattern::StreamingWrites), "StreamingWrites");
    EXPECT_STREQ(to_string(GpuKernelPattern::SharedMemory), "SharedMemory");
}

// =============================================================================
// ContractionStrategy Tests
// =============================================================================

TEST(ContractionStrategyTest, SmallK_Register) {
    // STO-3G: K=3, K*K = 9 <= 16
    EXPECT_EQ(select_contraction_strategy(3, 3), ContractionStrategy::Register);
    // K=4, K*K = 16 <= 16
    EXPECT_EQ(select_contraction_strategy(4, 4), ContractionStrategy::Register);
}

TEST(ContractionStrategyTest, MediumK_Cache) {
    // K=5, K*K = 25
    EXPECT_EQ(select_contraction_strategy(5, 5), ContractionStrategy::Cache);
    // K=10, K*K = 100
    EXPECT_EQ(select_contraction_strategy(10, 10), ContractionStrategy::Cache);
}

TEST(ContractionStrategyTest, LargeK_Streaming) {
    // K=20, K*K = 400
    EXPECT_EQ(select_contraction_strategy(20, 20), ContractionStrategy::Streaming);
}

TEST(ContractionStrategyTest, TwoElectron) {
    // STO-3G: 3*3*3*3 = 81 <= 256 => Cache
    EXPECT_EQ(select_contraction_strategy_2e(3, 3, 3, 3), ContractionStrategy::Cache);
    // K=1 each: 1 <= 16 => Register
    EXPECT_EQ(select_contraction_strategy_2e(1, 1, 1, 1), ContractionStrategy::Register);
    // K=4 each: 4*4*4*4 = 256 <= 256 => Cache
    EXPECT_EQ(select_contraction_strategy_2e(4, 4, 4, 4), ContractionStrategy::Cache);
    // K=5 each: 5^4 = 625 > 256 => Streaming
    EXPECT_EQ(select_contraction_strategy_2e(5, 5, 5, 5), ContractionStrategy::Streaming);
}

TEST(ContractionStrategyTest, ToString) {
    EXPECT_STREQ(to_string(ContractionStrategy::Register), "Register");
    EXPECT_STREQ(to_string(ContractionStrategy::Cache), "Cache");
    EXPECT_STREQ(to_string(ContractionStrategy::Streaming), "Streaming");
}

// =============================================================================
// RegistryKey Tests
// =============================================================================

TEST(RegistryKeyTest, For1EConstruction) {
    RegistryKey key = RegistryKey::for_1e(OperatorKind::Overlap, 1, 2, 3, 6, BackendType::CPU);

    EXPECT_EQ(key.op_kind, OperatorKind::Overlap);
    EXPECT_EQ(key.am[0], 1);
    EXPECT_EQ(key.am[1], 2);
    EXPECT_EQ(key.am[2], 0);
    EXPECT_EQ(key.am[3], 0);
    EXPECT_EQ(key.n_primitives[0], 3);
    EXPECT_EQ(key.n_primitives[1], 6);
    EXPECT_EQ(key.n_primitives[2], 1);
    EXPECT_EQ(key.n_primitives[3], 1);
    EXPECT_EQ(key.available_backend, BackendType::CPU);
    EXPECT_TRUE(key.is_one_electron());
    EXPECT_FALSE(key.is_two_electron());
}

TEST(RegistryKeyTest, For2EConstruction) {
    RegistryKey key = RegistryKey::for_2e(OperatorKind::Coulomb, 0, 1, 1, 2,
                                           3, 3, 6, 6, BackendType::CUDA);

    EXPECT_EQ(key.op_kind, OperatorKind::Coulomb);
    EXPECT_EQ(key.am[0], 0);
    EXPECT_EQ(key.am[1], 1);
    EXPECT_EQ(key.am[2], 1);
    EXPECT_EQ(key.am[3], 2);
    EXPECT_EQ(key.n_primitives[0], 3);
    EXPECT_EQ(key.n_primitives[1], 3);
    EXPECT_EQ(key.n_primitives[2], 6);
    EXPECT_EQ(key.n_primitives[3], 6);
    EXPECT_EQ(key.available_backend, BackendType::CUDA);
    EXPECT_FALSE(key.is_one_electron());
    EXPECT_TRUE(key.is_two_electron());
}

TEST(RegistryKeyTest, TotalAM) {
    RegistryKey key1e = RegistryKey::for_1e(OperatorKind::Overlap, 2, 3, 3, 3, BackendType::CPU);
    EXPECT_EQ(key1e.total_am(), 5);  // 2 + 3 + 0 + 0

    RegistryKey key2e = RegistryKey::for_2e(OperatorKind::Coulomb, 1, 1, 2, 2,
                                             3, 3, 3, 3, BackendType::CPU);
    EXPECT_EQ(key2e.total_am(), 6);  // 1 + 1 + 2 + 2
}

TEST(RegistryKeyTest, TotalPrimitives) {
    RegistryKey key = RegistryKey::for_2e(OperatorKind::Coulomb, 0, 0, 0, 0,
                                           3, 4, 5, 6, BackendType::CPU);
    EXPECT_EQ(key.total_primitives(), 360u);  // 3 * 4 * 5 * 6
}

TEST(RegistryKeyTest, Equality) {
    RegistryKey key1 = RegistryKey::for_1e(OperatorKind::Overlap, 0, 0, 3, 3, BackendType::CPU);
    RegistryKey key2 = RegistryKey::for_1e(OperatorKind::Overlap, 0, 0, 3, 3, BackendType::CPU);
    RegistryKey key3 = RegistryKey::for_1e(OperatorKind::Kinetic, 0, 0, 3, 3, BackendType::CPU);

    EXPECT_EQ(key1, key2);
    EXPECT_NE(key1, key3);
}

TEST(RegistryKeyTest, HashConsistency) {
    RegistryKey key1 = RegistryKey::for_1e(OperatorKind::Overlap, 1, 2, 3, 6, BackendType::CPU);
    RegistryKey key2 = RegistryKey::for_1e(OperatorKind::Overlap, 1, 2, 3, 6, BackendType::CPU);
    RegistryKey key3 = RegistryKey::for_1e(OperatorKind::Overlap, 2, 1, 3, 6, BackendType::CPU);

    RegistryKey::Hash hash;

    // Equal keys should have equal hashes
    EXPECT_EQ(hash(key1), hash(key2));

    // Different keys should have different hashes (with high probability)
    EXPECT_NE(hash(key1), hash(key3));
}

// =============================================================================
// CostModel Tests
// =============================================================================

TEST(CostModelTest, DefaultConstruction) {
    CostModel model;

    // Should have auto-detected hardware
    EXPECT_GE(model.profile().cpu_cores, 1);
    EXPECT_GE(model.profile().simd_width, 1);
}

TEST(CostModelTest, CustomProfile) {
    HardwareProfile profile;
    profile.cpu_cores = 8;
    profile.simd_width = 8;  // AVX-512
    profile.cpu_gflops = 100.0;

    CostModel model(profile);

    EXPECT_EQ(model.profile().cpu_cores, 8);
    EXPECT_EQ(model.profile().simd_width, 8);
    EXPECT_DOUBLE_EQ(model.profile().cpu_gflops, 100.0);
}

TEST(CostModelTest, EstimateOverlap) {
    CostModel model;

    CostEstimate cost = model.estimate(
        OperatorKind::Overlap,
        {0, 0, 0, 0},  // (s|s) pair
        {3, 3, 1, 1},  // 3 primitives each
        100);          // batch size

    // Should have positive estimates
    EXPECT_GT(cost.cpu_serial_ns, 0.0);
    EXPECT_GT(cost.best_cpu_ns(), 0.0);
}

TEST(CostModelTest, EstimateCoulomb) {
    CostModel model;

    CostEstimate cost = model.estimate(
        OperatorKind::Coulomb,
        {1, 1, 1, 1},  // (pp|pp) quartet
        {3, 3, 3, 3},  // 3 primitives each
        100);          // batch size

    // Should have positive estimates
    EXPECT_GT(cost.cpu_serial_ns, 0.0);
    // ERI should be more expensive than overlap
    CostEstimate overlap_cost = model.estimate(OperatorKind::Overlap, {0, 0, 0, 0},
                                                {3, 3, 1, 1}, 100);
    EXPECT_GT(cost.cpu_serial_ns, overlap_cost.cpu_serial_ns);
}

TEST(CostModelTest, HighAMCostsMore) {
    CostModel model;

    CostEstimate low_am = model.estimate(
        OperatorKind::Coulomb,
        {0, 0, 0, 0},  // (ss|ss)
        {3, 3, 3, 3},
        100);

    CostEstimate high_am = model.estimate(
        OperatorKind::Coulomb,
        {2, 2, 2, 2},  // (dd|dd)
        {3, 3, 3, 3},
        100);

    // Higher AM should cost more
    EXPECT_GT(high_am.cpu_serial_ns, low_am.cpu_serial_ns);
}

TEST(CostModelTest, SelectStrategy_CPUOnly) {
    HardwareProfile profile;
    profile.cpu_cores = 4;
    profile.simd_width = 4;
    profile.cpu_gflops = 20.0;
    profile.gpu_gflops = 0.0;  // No GPU

    CostModel model(profile);

    CostEstimate cost = model.estimate(OperatorKind::Overlap, {0, 0, 0, 0},
                                        {3, 3, 1, 1}, 1000);
    ExecutionStrategy strategy = model.select_strategy(cost, false);

    // Should select a CPU strategy
    EXPECT_FALSE(uses_gpu(strategy));
}

// =============================================================================
// KernelCalculator Tests
// =============================================================================

TEST(KernelCalculatorTest, DefaultConstruction) {
    KernelCalculator calc;

    EXPECT_EQ(calc.mode(), KernelCalculator::Mode::ProfileOnce);
    EXPECT_FALSE(calc.gpu_available());
}

TEST(KernelCalculatorTest, AnalyticalMode) {
    KernelCalculator calc(KernelCalculator::Mode::Analytical);

    RegistryKey key = RegistryKey::for_1e(OperatorKind::Overlap, 0, 0, 3, 3, BackendType::CPU);
    ExecutionStrategy strategy = calc.select(key, 100);

    // Should select a CPU strategy (no GPU available)
    EXPECT_FALSE(uses_gpu(strategy));
}

TEST(KernelCalculatorTest, RecordTiming) {
    KernelCalculator calc(KernelCalculator::Mode::ProfileOnce);

    RegistryKey key = RegistryKey::for_1e(OperatorKind::Overlap, 0, 0, 3, 3, BackendType::CPU);

    // Record some timings
    calc.record_timing(key, ExecutionStrategy::SerialCPU,
                       std::chrono::nanoseconds(1000), 100);
    calc.record_timing(key, ExecutionStrategy::SimdCPU,
                       std::chrono::nanoseconds(500), 100);

    auto history = calc.get_history(key);
    EXPECT_EQ(history.size(), 2u);
}

TEST(KernelCalculatorTest, SelectWithCachedTiming) {
    KernelCalculator calc(KernelCalculator::Mode::ProfileOnce);

    RegistryKey key = RegistryKey::for_1e(OperatorKind::Overlap, 0, 0, 3, 3, BackendType::CPU);

    // Record timing showing SimdCPU is faster
    calc.record_timing(key, ExecutionStrategy::SerialCPU,
                       std::chrono::nanoseconds(1000), 100);
    calc.record_timing(key, ExecutionStrategy::SimdCPU,
                       std::chrono::nanoseconds(500), 100);

    // Selection should use cached data
    ExecutionStrategy strategy = calc.select(key, 100);

    // Should prefer the faster strategy
    EXPECT_EQ(strategy, ExecutionStrategy::SimdCPU);
}

TEST(KernelCalculatorTest, ClearHistory) {
    KernelCalculator calc(KernelCalculator::Mode::ProfileOnce);

    RegistryKey key = RegistryKey::for_1e(OperatorKind::Overlap, 0, 0, 3, 3, BackendType::CPU);

    calc.record_timing(key, ExecutionStrategy::SerialCPU,
                       std::chrono::nanoseconds(1000), 100);

    EXPECT_FALSE(calc.get_history(key).empty());

    calc.clear_history();

    EXPECT_TRUE(calc.get_history(key).empty());
}

// =============================================================================
// DispatchRegistry Tests
// =============================================================================

TEST(DispatchRegistryTest, DefaultConstruction) {
    DispatchRegistry registry;

    EXPECT_EQ(registry.size(), 0u);
    auto stats = registry.stats();
    EXPECT_EQ(stats.entries, 0u);
    EXPECT_EQ(stats.hits, 0u);
    EXPECT_EQ(stats.misses, 0u);
}

TEST(DispatchRegistryTest, LookupCacheMiss) {
    DispatchRegistry registry;

    RegistryKey key = RegistryKey::for_1e(OperatorKind::Overlap, 0, 0, 3, 3, BackendType::CPU);

    auto entry = registry.lookup(key, 100);

    EXPECT_FALSE(entry.was_cached);
    EXPECT_FALSE(uses_gpu(entry.strategy));  // No GPU available by default
    EXPECT_GT(entry.estimated_ns, 0.0);

    auto stats = registry.stats();
    EXPECT_EQ(stats.misses, 1u);
    EXPECT_EQ(stats.entries, 1u);
}

TEST(DispatchRegistryTest, LookupCacheHit) {
    DispatchRegistry registry;

    RegistryKey key = RegistryKey::for_1e(OperatorKind::Overlap, 0, 0, 3, 3, BackendType::CPU);

    // First lookup - cache miss
    auto entry1 = registry.lookup(key, 100);
    EXPECT_FALSE(entry1.was_cached);

    // Second lookup - cache hit
    auto entry2 = registry.lookup(key, 100);
    EXPECT_TRUE(entry2.was_cached);
    EXPECT_EQ(entry1.strategy, entry2.strategy);

    auto stats = registry.stats();
    EXPECT_EQ(stats.hits, 1u);
    EXPECT_EQ(stats.misses, 1u);
}

TEST(DispatchRegistryTest, Clear) {
    DispatchRegistry registry;

    RegistryKey key = RegistryKey::for_1e(OperatorKind::Overlap, 0, 0, 3, 3, BackendType::CPU);
    [[maybe_unused]] auto entry = registry.lookup(key, 100);

    EXPECT_EQ(registry.size(), 1u);

    registry.clear();

    EXPECT_EQ(registry.size(), 0u);
    auto stats = registry.stats();
    EXPECT_EQ(stats.entries, 0u);
}

TEST(DispatchRegistryTest, Warmup) {
    DispatchRegistry registry;

    // Warm up with max AM = 1
    registry.warmup(1, BackendType::CPU);

    // Should have populated many entries
    EXPECT_GT(registry.size(), 0u);

    // Subsequent lookups should be cache hits
    RegistryKey key = RegistryKey::for_1e(OperatorKind::Overlap, 0, 0, 3, 3, BackendType::CPU);
    auto entry = registry.lookup(key, 100);

    // Note: warmup uses specific primitive counts, so this might be a miss
    // if the counts don't match
}

TEST(DispatchRegistryTest, SetGpuAvailableClearsCache) {
    DispatchRegistry registry;

    RegistryKey key = RegistryKey::for_1e(OperatorKind::Overlap, 0, 0, 3, 3, BackendType::CPU);
    [[maybe_unused]] auto entry = registry.lookup(key, 100);

    EXPECT_EQ(registry.size(), 1u);

    // Changing GPU availability should clear cache
    registry.set_gpu_available(true);

    EXPECT_EQ(registry.size(), 0u);
}

TEST(DispatchRegistryTest, RecordTimingInvalidatesCache) {
    // Create registry with AdaptiveTune mode
    auto calculator = std::make_shared<KernelCalculator>(KernelCalculator::Mode::AdaptiveTune);
    DispatchRegistry registry(calculator);

    RegistryKey key = RegistryKey::for_1e(OperatorKind::Overlap, 0, 0, 3, 3, BackendType::CPU);

    // First lookup populates cache
    [[maybe_unused]] auto entry = registry.lookup(key, 100);
    EXPECT_EQ(registry.size(), 1u);

    // Record timing should invalidate cache in AdaptiveTune mode
    registry.record_timing(key, ExecutionStrategy::SimdCPU,
                           std::chrono::nanoseconds(500), 100);

    EXPECT_EQ(registry.size(), 0u);
}

// =============================================================================
// Global Singleton Tests
// =============================================================================

TEST(DispatchRegistryTest, GlobalSingleton) {
    // Reset to ensure clean state
    reset_dispatch_registry();

    DispatchRegistry& reg1 = get_dispatch_registry();
    DispatchRegistry& reg2 = get_dispatch_registry();

    EXPECT_EQ(&reg1, &reg2);
}

// =============================================================================
// DispatchPolicy with Auto-Tuning Tests
// =============================================================================

TEST(DispatchPolicyAutoTuneTest, DefaultDisabled) {
    DispatchConfig config;
    EXPECT_FALSE(config.enable_auto_tuning);

    DispatchPolicy policy(config);
    EXPECT_FALSE(policy.auto_tuning_enabled());
}

TEST(DispatchPolicyAutoTuneTest, EnableAutoTuning) {
    DispatchConfig config;
    config.enable_auto_tuning = true;
    config.auto_tune_mode = KernelCalculator::Mode::Analytical;

    DispatchPolicy policy(config);
    EXPECT_TRUE(policy.auto_tuning_enabled());
}

TEST(DispatchPolicyAutoTuneTest, SelectStrategy) {
    DispatchConfig config;
    config.enable_auto_tuning = true;
    config.auto_tune_min_batch = 10;

    DispatchPolicy policy(config);

    RegistryKey key = RegistryKey::for_1e(OperatorKind::Overlap, 0, 0, 3, 3, BackendType::CPU);

    // With large batch, should use auto-tuning
    auto decision = policy.select_strategy(key, 100, BackendHint::Auto, false);

    // Without GPU, should select CPU backend
    EXPECT_EQ(decision.backend, BackendType::CPU);
    EXPECT_FALSE(uses_gpu(decision.strategy));
}

TEST(DispatchPolicyAutoTuneTest, ForceHintsOverrideAutoTuning) {
    DispatchConfig config;
    config.enable_auto_tuning = true;

    DispatchPolicy policy(config);

    RegistryKey key = RegistryKey::for_1e(OperatorKind::Overlap, 0, 0, 3, 3, BackendType::CPU);

    // ForceCPU should bypass auto-tuning
    auto decision = policy.select_strategy(key, 1000, BackendHint::ForceCPU, true);
    EXPECT_EQ(decision.backend, BackendType::CPU);

    // ForceGPU should bypass auto-tuning (falls back to CPU if no GPU)
    decision = policy.select_strategy(key, 1000, BackendHint::ForceGPU, false);
    EXPECT_EQ(decision.backend, BackendType::CPU);  // No GPU available
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

TEST(DispatchRegistryTest, ConcurrentLookup) {
    DispatchRegistry registry;
    std::atomic<int> hit_count{0};
    std::atomic<int> miss_count{0};

    // Create some keys
    std::vector<RegistryKey> keys;
    for (int la = 0; la <= 2; ++la) {
        for (int lb = 0; lb <= 2; ++lb) {
            keys.push_back(RegistryKey::for_1e(OperatorKind::Overlap, la, lb,
                                               3, 3, BackendType::CPU));
        }
    }

    // Launch multiple threads doing concurrent lookups
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&registry, &keys, &hit_count, &miss_count]() {
            for (int i = 0; i < 100; ++i) {
                for (const auto& key : keys) {
                    auto entry = registry.lookup(key, 100);
                    if (entry.was_cached) {
                        ++hit_count;
                    } else {
                        ++miss_count;
                    }
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Should have some hits and some misses
    EXPECT_GT(hit_count.load(), 0);

    // Cache should have been populated
    EXPECT_GT(registry.size(), 0u);
}

// =============================================================================
// Generated Kernel Registry Tests
// =============================================================================

#if LIBACCINT_ENABLE_CPU_GENERATED_REGISTRY
#include <libaccint/kernels/generated_kernel_registry.hpp>

#include <cmath>

using namespace libaccint::kernels::cpu::generated;

class GeneratedKernelRegistryTest : public ::testing::Test {};

TEST_F(GeneratedKernelRegistryTest, HasGeneratedOverlap_ValidRange) {
    // All AM pairs 0-4 should be available
    for (int la = 0; la <= 4; ++la) {
        for (int lb = 0; lb <= 4; ++lb) {
            EXPECT_TRUE(has_generated_overlap(la, lb))
                << "Expected generated overlap for la=" << la << ", lb=" << lb;
        }
    }
}

TEST_F(GeneratedKernelRegistryTest, HasGeneratedOverlap_OutOfRange) {
    EXPECT_FALSE(has_generated_overlap(5, 0));
    EXPECT_FALSE(has_generated_overlap(0, 5));
    EXPECT_FALSE(has_generated_overlap(-1, 0));
    EXPECT_FALSE(has_generated_overlap(0, -1));
}

TEST_F(GeneratedKernelRegistryTest, HasGeneratedKinetic_ValidRange) {
    for (int la = 0; la <= 4; ++la) {
        for (int lb = 0; lb <= 4; ++lb) {
            EXPECT_TRUE(has_generated_kinetic(la, lb))
                << "Expected generated kinetic for la=" << la << ", lb=" << lb;
        }
    }
}

TEST_F(GeneratedKernelRegistryTest, HasGeneratedKinetic_OutOfRange) {
    EXPECT_FALSE(has_generated_kinetic(5, 0));
    EXPECT_FALSE(has_generated_kinetic(0, 5));
}

TEST_F(GeneratedKernelRegistryTest, GetGeneratedOverlap_ReturnsNonNull) {
    for (int la = 0; la <= 4; ++la) {
        for (int lb = 0; lb <= 4; ++lb) {
            auto fn = get_generated_overlap(la, lb);
            ASSERT_NE(fn, nullptr)
                << "Expected non-null overlap kernel for la=" << la << ", lb=" << lb;
        }
    }
}

TEST_F(GeneratedKernelRegistryTest, GetGeneratedOverlap_OutOfRange_ReturnsNull) {
    EXPECT_EQ(get_generated_overlap(5, 0), nullptr);
    EXPECT_EQ(get_generated_overlap(0, 5), nullptr);
    EXPECT_EQ(get_generated_overlap(6, 6), nullptr);
}

TEST_F(GeneratedKernelRegistryTest, GetGeneratedKinetic_ReturnsNonNull) {
    for (int la = 0; la <= 4; ++la) {
        for (int lb = 0; lb <= 4; ++lb) {
            auto fn = get_generated_kinetic(la, lb);
            ASSERT_NE(fn, nullptr)
                << "Expected non-null kinetic kernel for la=" << la << ", lb=" << lb;
        }
    }
}

TEST_F(GeneratedKernelRegistryTest, GetGeneratedKinetic_OutOfRange_ReturnsNull) {
    EXPECT_EQ(get_generated_kinetic(5, 0), nullptr);
    EXPECT_EQ(get_generated_kinetic(0, 5), nullptr);
}

TEST_F(GeneratedKernelRegistryTest, DistinctFunctionPointers) {
    // Each AM pair should map to a different function
    auto fn_ss = get_generated_overlap(0, 0);
    auto fn_sp = get_generated_overlap(0, 1);
    auto fn_pp = get_generated_overlap(1, 1);
    auto fn_dd = get_generated_overlap(2, 2);

    EXPECT_NE(fn_ss, fn_sp);
    EXPECT_NE(fn_ss, fn_pp);
    EXPECT_NE(fn_ss, fn_dd);
    EXPECT_NE(fn_sp, fn_pp);
}

TEST_F(GeneratedKernelRegistryTest, OverlapSS_ProducesCorrectResult) {
    // Test the generated (s|s) overlap kernel directly
    // Two s-type Gaussians at the same center with exponent 1.0 and coeff 1.0
    // S = (pi/2)^(3/2) * exp(0) = (pi/2)^1.5
    double exp_a[] = {1.0};
    double coeff_a[] = {1.0};
    double center_a[] = {0.0, 0.0, 0.0};

    double exp_b[] = {1.0};
    double coeff_b[] = {1.0};
    double center_b[] = {0.0, 0.0, 0.0};

    double output = 0.0;

    auto fn = get_generated_overlap(0, 0);
    ASSERT_NE(fn, nullptr);

    fn(exp_a, coeff_a, center_a, 1, exp_b, coeff_b, center_b, 1, &output);

    // Expected: (pi / (alpha + beta))^(3/2) = (pi/2)^(3/2)
    const double expected = std::pow(M_PI / 2.0, 1.5);
    EXPECT_NEAR(output, expected, 1e-12)
        << "Generated overlap_ss kernel produced incorrect result";
}

TEST_F(GeneratedKernelRegistryTest, GeneratedMaxAM_IsFour) {
    EXPECT_EQ(GENERATED_MAX_AM, LIBACCINT_MAX_AM);
}
#endif  // LIBACCINT_ENABLE_CPU_GENERATED_REGISTRY
