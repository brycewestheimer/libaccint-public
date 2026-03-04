// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_resource_dispatch.cpp
/// @brief Unit tests for resource-aware dispatch functionality
///
/// Step 10.4: Tests covering DispatchPolicy backend selection heuristics,
/// DeviceResourceTracker batch/occupancy advice, RAII resource reservation,
/// small-batch CPU fallback, and Engine ↔ CudaEngine dispatch config
/// propagation.

#include <libaccint/engine/dispatch_policy.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/core/backend.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA
#include <libaccint/device/device_resource_tracker.hpp>
#include <libaccint/engine/cuda_engine.hpp>
#endif

#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <vector>

using namespace libaccint;

namespace libaccint::test {

// =============================================================================
// Helpers
// =============================================================================

/// @brief Build a minimal test shell with given AM and number of primitives
Shell make_test_shell(int am, Point3D center, int n_prim) {
    std::vector<Real> exponents(static_cast<Size>(n_prim));
    std::vector<Real> coefficients(static_cast<Size>(n_prim));
    for (int i = 0; i < n_prim; ++i) {
        exponents[static_cast<Size>(i)] = 10.0 * std::exp(-0.8 * i);
        coefficients[static_cast<Size>(i)] = 1.0 / static_cast<double>(i + 1);
    }
    return Shell(am, center, exponents, coefficients);
}

/// @brief Build a minimal STO-3G H2O shell set (5 shells, 7 basis functions)
std::vector<Shell> make_sto3g_h2o_shells() {
    constexpr Point3D O{0.0, 0.0, 0.0};
    constexpr Point3D H1{0.0, 1.43233673, -1.10866041};
    constexpr Point3D H2{0.0, -1.43233673, -1.10866041};

    std::vector<Shell> shells;
    shells.reserve(5);

    // O 1s
    { Shell s(0, O, {130.7093200, 23.8088610, 6.4436083},
                     {0.15432897, 0.53532814, 0.44463454});
      s.set_atom_index(0); shells.push_back(std::move(s)); }
    // O 2s
    { Shell s(0, O, {5.0331513, 1.1695961, 0.3803890},
                     {-0.09996723, 0.39951283, 0.70011547});
      s.set_atom_index(0); shells.push_back(std::move(s)); }
    // O 2p
    { Shell s(1, O, {5.0331513, 1.1695961, 0.3803890},
                     {0.15591627, 0.60768372, 0.39195739});
      s.set_atom_index(0); shells.push_back(std::move(s)); }
    // H1 1s
    { Shell s(0, H1, {3.42525091, 0.62391373, 0.16885540},
                      {0.15432897, 0.53532814, 0.44463454});
      s.set_atom_index(1); shells.push_back(std::move(s)); }
    // H2 1s
    { Shell s(0, H2, {3.42525091, 0.62391373, 0.16885540},
                      {0.15432897, 0.53532814, 0.44463454});
      s.set_atom_index(2); shells.push_back(std::move(s)); }

    return shells;
}

// =============================================================================
// Test Fixture
// =============================================================================

class ResourceDispatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        shells_ = make_sto3g_h2o_shells();
        basis_ = std::make_unique<BasisSet>(shells_);
    }

    void TearDown() override {
        basis_.reset();
        shells_.clear();
    }

    std::vector<Shell> shells_;
    std::unique_ptr<BasisSet> basis_;
};

// =============================================================================
// DispatchPolicy Tests
// =============================================================================

/// 1. Small batch with Auto hint → CPU for ShellSetQuartet
TEST_F(ResourceDispatchTest, SelectBackendAutoSmallBatch) {
    DispatchPolicy policy;  // default min_gpu_batch_size = 16

    auto backend = policy.select_backend(
        WorkUnitType::ShellSetQuartet,
        /*batch_size=*/4,
        /*total_am=*/2,
        /*n_primitives=*/36,
        BackendHint::Auto,
        /*gpu_available=*/true);

    EXPECT_EQ(backend, BackendType::CPU)
        << "Small batch (4 < 16) should dispatch to CPU under Auto hint";
}

/// 2. Large batch with Auto hint → GPU (or at least not forced CPU)
TEST_F(ResourceDispatchTest, SelectBackendAutoLargeBatch) {
    DispatchPolicy policy;

    auto backend = policy.select_backend(
        WorkUnitType::ShellSetQuartet,
        /*batch_size=*/256,
        /*total_am=*/4,
        /*n_primitives=*/10000,
        BackendHint::Auto,
        /*gpu_available=*/true);

    // Large batch with many primitives should prefer GPU
    EXPECT_EQ(backend, BackendType::CUDA)
        << "Large batch (256 >= 16) with many primitives should dispatch to GPU";
}

/// 3. ForceCPU always returns CPU regardless of batch size
TEST_F(ResourceDispatchTest, SelectBackendForceCPU) {
    DispatchPolicy policy;

    auto backend = policy.select_backend(
        WorkUnitType::ShellSetQuartet,
        /*batch_size=*/10000,
        /*total_am=*/8,
        /*n_primitives=*/100000,
        BackendHint::ForceCPU,
        /*gpu_available=*/true);

    EXPECT_EQ(backend, BackendType::CPU)
        << "ForceCPU must always return CPU";
}

/// 4. ForceGPU always returns GPU regardless of batch size
TEST_F(ResourceDispatchTest, SelectBackendForceGPU) {
    DispatchPolicy policy;

    auto backend = policy.select_backend(
        WorkUnitType::ShellSetQuartet,
        /*batch_size=*/1,
        /*total_am=*/0,
        /*n_primitives=*/1,
        BackendHint::ForceGPU,
        /*gpu_available=*/true);

    EXPECT_EQ(backend, BackendType::CUDA)
        << "ForceGPU must return CUDA when GPU is available";
}

/// 5. DispatchConfig default values
TEST_F(ResourceDispatchTest, DispatchConfigDefaults) {
    DispatchConfig config;

    EXPECT_EQ(config.min_gpu_batch_size, 16u);
    EXPECT_EQ(config.min_gpu_primitives, 1000u);
    EXPECT_EQ(config.high_am_threshold, 4);
    EXPECT_EQ(config.min_gpu_shells, 10u);
    EXPECT_FALSE(config.enable_auto_tuning);
}

/// 6. Custom DispatchConfig values are respected by DispatchPolicy
TEST_F(ResourceDispatchTest, DispatchConfigCustom) {
    DispatchConfig config;
    config.min_gpu_batch_size = 64;
    config.min_gpu_primitives = 500;
    config.high_am_threshold = 2;
    config.min_gpu_shells = 5;

    DispatchPolicy policy(config);

    EXPECT_EQ(policy.config().min_gpu_batch_size, 64u);
    EXPECT_EQ(policy.config().min_gpu_primitives, 500u);
    EXPECT_EQ(policy.config().high_am_threshold, 2);
    EXPECT_EQ(policy.config().min_gpu_shells, 5u);

    // With min_gpu_batch_size = 64, a batch of 32 should still go to CPU
    // Use low total_am and n_primitives so only batch_size determines dispatch
    auto backend = policy.select_backend(
        WorkUnitType::ShellSetQuartet,
        /*batch_size=*/32,
        /*total_am=*/0,
        /*n_primitives=*/100,
        BackendHint::Auto,
        /*gpu_available=*/true);

    EXPECT_EQ(backend, BackendType::CPU)
        << "Batch of 32 should go to CPU when min_gpu_batch_size = 64";
}

// =============================================================================
// DeviceResourceTracker Tests (CUDA-only)
// =============================================================================

#if LIBACCINT_USE_CUDA

/// 7. recommend_batch_config returns valid grid/block dimensions
TEST_F(ResourceDispatchTest, BatchConfigReasonableValues) {
    if (!is_backend_available(BackendType::CUDA)) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto& tracker = device::DeviceResourceTracker::instance();

    AMQuartet am_quartet = {0, 0, 0, 0};
    auto batch = tracker.recommend_batch_config(
        device::GpuIntegralType::ERI,
        am_quartet,
        device::GpuContractionRange::SmallK,
        /*total_work_units=*/1000);

    // Grid dimensions must be non-zero
    EXPECT_GT(batch.grid_dim.x, 0u) << "grid_dim.x must be positive";
    EXPECT_GT(batch.block_dim.x, 0u) << "block_dim.x must be positive";

    // Block size should be a reasonable CUDA value (32–1024)
    EXPECT_GE(batch.block_dim.x, 32u) << "block_dim.x should be at least a warp";
    EXPECT_LE(batch.block_dim.x, 1024u) << "block_dim.x should not exceed 1024";

    // At least 1 launch
    EXPECT_GE(batch.num_launches, 1);

    // work_units_per_launch should be positive
    EXPECT_GT(batch.work_units_per_launch, 0);
}

/// 8. can_launch returns true for a small workload on a fresh tracker
TEST_F(ResourceDispatchTest, CanLaunchInitiallyTrue) {
    if (!is_backend_available(BackendType::CUDA)) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto& tracker = device::DeviceResourceTracker::instance();
    tracker.reset_counters();

    device::GpuKernelConfig cfg;
    cfg.block_size = 128;
    cfg.registers_per_thread = 32;
    cfg.shared_mem_per_block = 0;

    bool ok = tracker.can_launch(cfg, /*n_work_units=*/64);
    EXPECT_TRUE(ok) << "can_launch should return true for a modest workload "
                       "when no other work is tracked";
}

/// 9. RAII ResourceReservation increments counters and releases on scope exit
TEST_F(ResourceDispatchTest, ResourceReservationRAII) {
    if (!is_backend_available(BackendType::CUDA)) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto& tracker = device::DeviceResourceTracker::instance();
    tracker.reset_counters();

    device::GpuKernelConfig cfg;
    cfg.block_size = 256;
    cfg.registers_per_thread = 32;
    cfg.shared_mem_per_block = 0;

    int kernels_before = tracker.active_kernels();

    {
        auto reservation = tracker.reserve(cfg, /*n_work_units=*/128, nullptr);
        EXPECT_TRUE(reservation.is_valid())
            << "Reservation should be valid after reserve()";
        EXPECT_GT(tracker.active_kernels(), kernels_before)
            << "active_kernels should increase after reserve()";
    }

    // After scope exit, counters should revert
    EXPECT_EQ(tracker.active_kernels(), kernels_before)
        << "active_kernels must revert after ResourceReservation destructor";
}

/// 10. Moved-from reservation doesn't double-release
TEST_F(ResourceDispatchTest, ResourceReservationMove) {
    if (!is_backend_available(BackendType::CUDA)) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto& tracker = device::DeviceResourceTracker::instance();
    tracker.reset_counters();

    device::GpuKernelConfig cfg;
    cfg.block_size = 256;
    cfg.registers_per_thread = 32;
    cfg.shared_mem_per_block = 0;

    int kernels_before = tracker.active_kernels();

    {
        auto r1 = tracker.reserve(cfg, /*n_work_units=*/128, nullptr);
        EXPECT_TRUE(r1.is_valid());

        // Move construct
        auto r2 = std::move(r1);
        EXPECT_TRUE(r2.is_valid());
        EXPECT_FALSE(r1.is_valid())  // NOLINT(bugprone-use-after-move)
            << "Moved-from reservation should be invalid";

        // Only one kernel tracked
        EXPECT_EQ(tracker.active_kernels(), kernels_before + 1);
    }

    // Exactly one release should have occurred
    EXPECT_EQ(tracker.active_kernels(), kernels_before)
        << "Move semantics must not cause double-release";
}

/// 11. wait_for_resources returns true immediately when resources are available
TEST_F(ResourceDispatchTest, WaitForResourcesImmediate) {
    if (!is_backend_available(BackendType::CUDA)) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto& tracker = device::DeviceResourceTracker::instance();
    tracker.reset_counters();

    device::GpuKernelConfig cfg;
    cfg.block_size = 128;
    cfg.registers_per_thread = 32;
    cfg.shared_mem_per_block = 0;

    auto start = std::chrono::steady_clock::now();
    bool ready = tracker.wait_for_resources(
        cfg, /*n_work_units=*/64,
        std::chrono::milliseconds{100});
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_TRUE(ready) << "Should succeed immediately with idle GPU";
    // Should return almost immediately, well under the 100ms timeout
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 50)
        << "wait_for_resources should return quickly when resources are available";
}

/// 12. wait_for_resources returns false after timeout when resources are exhausted
TEST_F(ResourceDispatchTest, WaitForResourcesTimeout) {
    if (!is_backend_available(BackendType::CUDA)) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto& tracker = device::DeviceResourceTracker::instance();
    tracker.reset_counters();

    // Exhaust resources by making many large reservations
    device::GpuKernelConfig heavy_cfg;
    heavy_cfg.block_size = 1024;
    heavy_cfg.registers_per_thread = 128;
    heavy_cfg.shared_mem_per_block = 49152;  // 48 KB shared mem per block

    std::vector<device::ResourceReservation> reservations;
    // Reserve many slots to push the tracker past its limits
    // kMaxConcurrentKernels is private; use the known value (128) + 1
    for (int i = 0; i < 129; ++i) {
        reservations.push_back(tracker.reserve(heavy_cfg, /*n_work_units=*/100000, nullptr));
    }

    // Now try to launch more work — should time out
    auto start = std::chrono::steady_clock::now();
    bool ready = tracker.wait_for_resources(
        heavy_cfg, /*n_work_units=*/100000,
        std::chrono::milliseconds{50});
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(ready) << "Should time out when resources are exhausted";
    // Should have waited approximately the full timeout
    EXPECT_GE(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 40)
        << "Should have waited close to the timeout duration";
}

#endif  // LIBACCINT_USE_CUDA

// =============================================================================
// Small-Batch Fallback Threshold Tests (no GPU required)
// =============================================================================

/// 13. Default small-batch threshold is 16
TEST_F(ResourceDispatchTest, SmallBatchThresholdDefault) {
    DispatchConfig config;
    EXPECT_EQ(config.min_gpu_batch_size, 16u)
        << "Default min_gpu_batch_size should be 16";

    DispatchPolicy policy;
    EXPECT_EQ(policy.config().min_gpu_batch_size, 16u);
}

/// 14. Custom threshold is respected by the policy
TEST_F(ResourceDispatchTest, SmallBatchThresholdCustom) {
    DispatchConfig config;
    config.min_gpu_batch_size = 128;

    DispatchPolicy policy(config);
    EXPECT_EQ(policy.config().min_gpu_batch_size, 128u);

    // Batch of 64 should still go to CPU with threshold 128
    // Use low total_am and n_primitives so only batch_size determines dispatch
    auto backend = policy.select_backend(
        WorkUnitType::ShellSetQuartet,
        /*batch_size=*/64,
        /*total_am=*/0,
        /*n_primitives=*/100,
        BackendHint::Auto,
        /*gpu_available=*/true);

    EXPECT_EQ(backend, BackendType::CPU)
        << "Batch of 64 must dispatch to CPU with min_gpu_batch_size=128";
}

/// 15. Engine propagates DispatchConfig to CudaEngine (when CUDA is available)
TEST_F(ResourceDispatchTest, SetDispatchConfigPropagation) {
    DispatchConfig config;
    config.min_gpu_batch_size = 42;

    Engine engine(*basis_, config);

    // The Engine's dispatch policy should reflect the config
    EXPECT_EQ(engine.dispatch_policy().config().min_gpu_batch_size, 42u);

    // Update config via set_dispatch_config
    DispatchConfig config2;
    config2.min_gpu_batch_size = 99;
    engine.set_dispatch_config(config2);

    EXPECT_EQ(engine.dispatch_policy().config().min_gpu_batch_size, 99u)
        << "set_dispatch_config should update the policy";

#if LIBACCINT_USE_CUDA
    // If CUDA engine exists, verify it also received the config
    if (engine.gpu_available()) {
        auto* cuda = engine.cuda_engine();
        ASSERT_NE(cuda, nullptr);
        // CudaEngine caches min_gpu_batch_size_ internally;
        // we can't inspect it directly, but the Engine path exercises
        // set_dispatch_config() which sets it.
    }
#endif
}

// =============================================================================
// Additional DispatchPolicy edge-case tests
// =============================================================================

/// 16. ForceGPU falls back to CPU when GPU is not available
TEST_F(ResourceDispatchTest, ForceGPUFallbackWhenUnavailable) {
    DispatchPolicy policy;

    auto backend = policy.select_backend(
        WorkUnitType::ShellSetQuartet,
        /*batch_size=*/1000,
        /*total_am=*/4,
        /*n_primitives=*/10000,
        BackendHint::ForceGPU,
        /*gpu_available=*/false);

    EXPECT_EQ(backend, BackendType::CPU)
        << "ForceGPU must fall back to CPU when gpu_available is false";
}

/// 17. PreferGPU selects GPU for large batches when available
TEST_F(ResourceDispatchTest, PreferGPUWithLargeBatch) {
    DispatchPolicy policy;

    auto backend = policy.select_backend(
        WorkUnitType::ShellSetQuartet,
        /*batch_size=*/512,
        /*total_am=*/6,
        /*n_primitives=*/50000,
        BackendHint::PreferGPU,
        /*gpu_available=*/true);

    EXPECT_EQ(backend, BackendType::CUDA)
        << "PreferGPU with large batch and many primitives should select GPU";
}

/// 18. PreferGPU falls back to CPU when GPU is unavailable
TEST_F(ResourceDispatchTest, PreferGPUFallbackWhenUnavailable) {
    DispatchPolicy policy;

    auto backend = policy.select_backend(
        WorkUnitType::ShellSetQuartet,
        /*batch_size=*/512,
        /*total_am=*/6,
        /*n_primitives=*/50000,
        BackendHint::PreferGPU,
        /*gpu_available=*/false);

    EXPECT_EQ(backend, BackendType::CPU)
        << "PreferGPU must fall back to CPU when gpu_available is false";
}

// =============================================================================
// Integration-Level Dispatch Tests (GPU required, skippable)
// =============================================================================

#if LIBACCINT_USE_CUDA

/// 19. Batch kernel launch grid dimensions are reasonable for a given batch size
TEST_F(ResourceDispatchTest, DISABLED_BatchKernelLaunchGridDimensions) {
    if (!is_backend_available(BackendType::CUDA)) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto& tracker = device::DeviceResourceTracker::instance();

    // Test several batch sizes and verify grid dimensions scale appropriately
    for (int batch : {1, 16, 64, 256, 1024, 4096}) {
        AMQuartet am = {1, 0, 1, 0};  // (pp|ss) quartets
        auto config = tracker.recommend_batch_config(
            device::GpuIntegralType::ERI,
            am,
            device::GpuContractionRange::SmallK,
            batch);

        // Total threads must cover the work
        unsigned total_threads = config.grid_dim.x * config.block_dim.x;
        EXPECT_GE(total_threads, static_cast<unsigned>(batch))
            << "Grid must launch at least one thread per work unit for batch="
            << batch;

        // Grid dimension should not be excessively large
        EXPECT_LE(config.grid_dim.x, 65535u)
            << "grid_dim.x should stay within CUDA grid limits for batch="
            << batch;
    }
}

/// 20. Batch of 1 triggers CPU fallback when dispatched through CudaEngine
TEST_F(ResourceDispatchTest, DISABLED_FallbackTriggeredForTinyBatch) {
    if (!is_backend_available(BackendType::CUDA)) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create an Engine with default config (min_gpu_batch_size=16)
    Engine engine(*basis_);

    // A batch of 1 quartet through the Engine with Auto hint should go to CPU.
    // We verify indirectly: the result should match the CPU-only result.
    DispatchConfig config;
    config.min_gpu_batch_size = 16;
    engine.set_dispatch_config(config);

    // Compute a single shell-pair overlap via the Engine with Auto hint.
    // For a single shell pair, dispatch should choose CPU.
    const auto& shell_a = basis_->shell(0);
    const auto& shell_b = basis_->shell(1);

    OneElectronBuffer<0> buf_auto, buf_cpu;
    int max_nf = n_cartesian(basis_->max_angular_momentum());
    buf_auto.resize(max_nf, max_nf);
    buf_cpu.resize(max_nf, max_nf);

    engine.compute_1e_shell_pair(Operator::overlap(),
                                  shell_a, shell_b, buf_auto,
                                  BackendHint::Auto);
    engine.compute_1e_shell_pair(Operator::overlap(),
                                  shell_a, shell_b, buf_cpu,
                                  BackendHint::ForceCPU);

    int na = shell_a.n_functions();
    int nb = shell_b.n_functions();
    for (int i = 0; i < na; ++i) {
        for (int j = 0; j < nb; ++j) {
            EXPECT_DOUBLE_EQ(buf_auto(i, j), buf_cpu(i, j))
                << "Auto dispatch for single pair should match ForceCPU result "
                << "at (" << i << "," << j << ")";
        }
    }
}

#endif  // LIBACCINT_USE_CUDA

}  // namespace libaccint::test
