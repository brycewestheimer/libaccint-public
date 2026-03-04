// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_multi_gpu_engine.cpp
/// @brief Tests for MultiGPUEngine dispatch, partitioning, and correctness
///
/// All tests are guarded with LIBACCINT_USE_CUDA.
/// On CPU-only builds, tests compile but skip at runtime.

#include <libaccint/engine/engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/operators/operator.hpp>

#include <gtest/gtest.h>
#include <vector>

#if LIBACCINT_USE_CUDA
#include <libaccint/engine/multi_gpu_engine.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/device/device_manager.hpp>
#define HAS_GPU_SUPPORT 1
#else
#define HAS_GPU_SUPPORT 0
#endif

using namespace libaccint;

namespace {

constexpr Point3D O_center{0.0, 0.0, 0.0};
constexpr Point3D H1_center{0.0, 1.43233673, -1.10866041};
constexpr Point3D H2_center{0.0, -1.43233673, -1.10866041};

std::vector<Shell> make_sto3g_h2o_shells() {
    std::vector<Shell> shells;
    shells.reserve(5);

    {
        Shell s(0, O_center,
                {130.7093200, 23.8088610, 6.4436083},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    {
        Shell s(0, O_center,
                {5.0331513, 1.1695961, 0.3803890},
                {-0.09996723, 0.39951283, 0.70011547});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    {
        Shell s(1, O_center,
                {5.0331513, 1.1695961, 0.3803890},
                {0.15591627, 0.60768372, 0.39195739});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    {
        Shell s(0, H1_center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(1);
        shells.push_back(std::move(s));
    }
    {
        Shell s(0, H2_center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(2);
        shells.push_back(std::move(s));
    }

    return shells;
}

constexpr Real TIGHT_TOL = 1e-10;

}  // anonymous namespace

// =============================================================================
// MultiGPU Engine Tests (CUDA only)
// =============================================================================

#if HAS_GPU_SUPPORT

TEST(MultiGPUEngine, Construction) {
    auto gpu_count = device::DeviceManager::instance().device_count();
    if (gpu_count == 0) {
        GTEST_SKIP() << "No GPU devices available";
    }

    BasisSet basis(make_sto3g_h2o_shells());

    engine::MultiGPUConfig config;
    config.collect_stats = true;

    engine::MultiGPUEngine mgpu_engine(basis, config);

    EXPECT_EQ(mgpu_engine.device_count(), gpu_count);
}

TEST(MultiGPUEngine, WorkPartitioning) {
    auto gpu_count = device::DeviceManager::instance().device_count();
    if (gpu_count == 0) {
        GTEST_SKIP() << "No GPU devices available";
    }

    BasisSet basis(make_sto3g_h2o_shells());

    engine::MultiGPUConfig config;
    config.collect_stats = true;

    engine::MultiGPUEngine mgpu_engine(basis, config);

    // Use a counting consumer to verify all quartets are processed
    struct CountingConsumer {
        int count = 0;
        void accumulate(const double*, const ShellSetQuartet&) {
            ++count;
        }
        void prepare_parallel(int) {}
        void finalize_parallel() {}
    };

    CountingConsumer counter;
    Operator op = Operator::coulomb();
    mgpu_engine.compute_all_eri(counter);

    // Should have processed all quartets
    const auto& quartets = basis.shell_set_quartets();
    EXPECT_EQ(static_cast<Size>(counter.count), quartets.size());
}

TEST(MultiGPUEngine, Correctness) {
    auto gpu_count = device::DeviceManager::instance().device_count();
    if (gpu_count == 0) {
        GTEST_SKIP() << "No GPU devices available";
    }

    BasisSet basis(make_sto3g_h2o_shells());
    const Size nbf = basis.n_basis_functions();

    // Single-GPU reference via Engine
    Engine single_engine(basis);
    consumers::FockBuilder ref_fock(nbf);
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) D[i * nbf + i] = 1.0;  // Identity density
    ref_fock.set_density(D.data(), nbf);

    Operator op = Operator::coulomb();
    single_engine.compute_and_consume(op, ref_fock);

    // Multi-GPU via MultiGPUFockAdapter
    engine::MultiGPUConfig config;
    config.collect_stats = true;
    engine::MultiGPUEngine mgpu_engine(basis, config);

    consumers::FockBuilder mgpu_fock(nbf);
    mgpu_fock.set_threading_strategy(consumers::FockThreadingStrategy::Atomic);
    mgpu_fock.set_density(D.data(), nbf);

    engine::MultiGPUFockAdapter adapter(mgpu_fock);
    mgpu_engine.compute_all_eri(adapter);

    // Compare Coulomb matrices
    auto ref_J = ref_fock.get_coulomb_matrix();
    auto mgpu_J = mgpu_fock.get_coulomb_matrix();

    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(ref_J[i], mgpu_J[i], TIGHT_TOL)
            << "Coulomb matrix mismatch at index " << i;
    }
}

TEST(MultiGPUEngine, Statistics) {
    auto gpu_count = device::DeviceManager::instance().device_count();
    if (gpu_count == 0) {
        GTEST_SKIP() << "No GPU devices available";
    }

    BasisSet basis(make_sto3g_h2o_shells());

    engine::MultiGPUConfig config;
    config.collect_stats = true;

    engine::MultiGPUEngine mgpu_engine(basis, config);

    struct NullConsumer {
        void accumulate(const double*, const ShellSetQuartet&) {}
        void prepare_parallel(int) {}
        void finalize_parallel() {}
    };

    NullConsumer consumer;
    mgpu_engine.compute_all_eri(consumer);

    const auto& stats = mgpu_engine.stats();

    // At least one device should have processed some quartets
    Size total = 0;
    for (auto count : stats.per_device_quartets) {
        total += count;
    }
    EXPECT_GT(total, 0u);
}

TEST(MultiGPUEngine, SingleDevice) {
    auto gpu_count = device::DeviceManager::instance().device_count();
    if (gpu_count == 0) {
        GTEST_SKIP() << "No GPU devices available";
    }

    BasisSet basis(make_sto3g_h2o_shells());

    // Restrict to single device
    engine::MultiGPUConfig config;
    config.device_ids = {0};

    engine::MultiGPUEngine mgpu_engine(basis, config);
    EXPECT_EQ(mgpu_engine.device_count(), 1u);

    struct NullConsumer {
        void accumulate(const double*, const ShellSetQuartet&) {}
        void prepare_parallel(int) {}
        void finalize_parallel() {}
    };

    NullConsumer consumer;
    EXPECT_NO_THROW(mgpu_engine.compute_all_eri(consumer));
}

#else  // !HAS_GPU_SUPPORT

// CPU-only build: skip all GPU tests
TEST(MultiGPUEngine, NoGPUSkip) {
    GTEST_SKIP() << "MultiGPU tests require CUDA support";
}

#endif
