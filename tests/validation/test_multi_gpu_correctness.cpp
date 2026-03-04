// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_multi_gpu_correctness.cpp
/// @brief Correctness validation for multi-GPU integral computation

#include <gtest/gtest.h>

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/device/device_manager.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/engine/multi_gpu_engine.hpp>
#include <libaccint/operators/operator.hpp>

namespace libaccint::test {

namespace {

constexpr Point3D O_center{0.0, 0.0, 0.0};
constexpr Point3D H1_center{0.0, 1.43233673, -1.10866041};
constexpr Point3D H2_center{0.0, -1.43233673, -1.10866041};

std::vector<Shell> make_sto3g_h2o_shells() {
    std::vector<Shell> shells;
    shells.reserve(5);
    {
        Shell s(0, O_center, {130.7093200, 23.8088610, 6.4436083},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    {
        Shell s(0, O_center, {5.0331513, 1.1695961, 0.3803890},
                {-0.09996723, 0.39951283, 0.70011547});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    {
        Shell s(1, O_center, {5.0331513, 1.1695961, 0.3803890},
                {0.15591627, 0.60768372, 0.39195739});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    {
        Shell s(0, H1_center, {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(1);
        shells.push_back(std::move(s));
    }
    {
        Shell s(0, H2_center, {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(2);
        shells.push_back(std::move(s));
    }
    return shells;
}

constexpr Real TIGHT_TOL = 1e-10;

}  // anonymous namespace

class MultiGPUCorrectnessTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto& dm = device::DeviceManager::instance();
        n_devices_ = dm.device_count();

        if (n_devices_ < 2) {
            GTEST_SKIP() << "Multi-GPU tests require at least 2 GPUs";
        }
    }

    int n_devices_ = 0;
};

/// Test that multi-GPU results match single-GPU exactly
TEST_F(MultiGPUCorrectnessTest, FockBuilderMatchesSingleGPU) {
    BasisSet basis(make_sto3g_h2o_shells());
    const Size nbf = basis.n_basis_functions();

    // Identity density matrix
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) D[i * nbf + i] = 1.0;

    // Single-GPU reference
    Engine single_engine(basis);
    consumers::FockBuilder ref_fock(nbf);
    ref_fock.set_density(D.data(), nbf);
    single_engine.compute_and_consume(Operator::coulomb(), ref_fock);

    // Multi-GPU via adapter
    engine::MultiGPUConfig config;
    engine::MultiGPUEngine mgpu_engine(basis, config);
    consumers::FockBuilder mgpu_fock(nbf);
    mgpu_fock.set_threading_strategy(consumers::FockThreadingStrategy::Atomic);
    mgpu_fock.set_density(D.data(), nbf);
    engine::MultiGPUFockAdapter adapter(mgpu_fock);
    mgpu_engine.compute_all_eri(adapter);

    auto ref_J = ref_fock.get_coulomb_matrix();
    auto mgpu_J = mgpu_fock.get_coulomb_matrix();
    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(ref_J[i], mgpu_J[i], TIGHT_TOL)
            << "Coulomb mismatch at index " << i;
    }
}

/// Test multi-GPU with varying device counts
TEST_F(MultiGPUCorrectnessTest, ConsistentAcrossDeviceCounts) {
    BasisSet basis(make_sto3g_h2o_shells());
    const Size nbf = basis.n_basis_functions();

    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) D[i * nbf + i] = 1.0;

    // Single-device reference
    engine::MultiGPUConfig config1;
    config1.device_ids = {0};
    engine::MultiGPUEngine engine1(basis, config1);
    consumers::FockBuilder fock1(nbf);
    fock1.set_threading_strategy(consumers::FockThreadingStrategy::Atomic);
    fock1.set_density(D.data(), nbf);
    engine::MultiGPUFockAdapter adapter1(fock1);
    engine1.compute_all_eri(adapter1);

    // Multi-device
    engine::MultiGPUConfig config_all;
    engine::MultiGPUEngine engine_all(basis, config_all);
    consumers::FockBuilder fock_all(nbf);
    fock_all.set_threading_strategy(consumers::FockThreadingStrategy::Atomic);
    fock_all.set_density(D.data(), nbf);
    engine::MultiGPUFockAdapter adapter_all(fock_all);
    engine_all.compute_all_eri(adapter_all);

    auto J1 = fock1.get_coulomb_matrix();
    auto J_all = fock_all.get_coulomb_matrix();
    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(J1[i], J_all[i], TIGHT_TOL)
            << "Coulomb mismatch (1-GPU vs all-GPU) at index " << i;
    }
}

/// Test that work distribution strategies produce identical results
TEST_F(MultiGPUCorrectnessTest, DistributionStrategiesConsistent) {
    BasisSet basis(make_sto3g_h2o_shells());
    const Size nbf = basis.n_basis_functions();

    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) D[i * nbf + i] = 1.0;

    // Run with default distribution
    engine::MultiGPUConfig config_default;
    engine::MultiGPUEngine engine_default(basis, config_default);
    consumers::FockBuilder fock_default(nbf);
    fock_default.set_threading_strategy(consumers::FockThreadingStrategy::Atomic);
    fock_default.set_density(D.data(), nbf);
    engine::MultiGPUFockAdapter adapter_default(fock_default);
    engine_default.compute_all_eri(adapter_default);

    // Run with round-robin distribution
    engine::MultiGPUConfig config_rr;
    config_rr.distribution.strategy = device::DistributionStrategy::RoundRobin;
    engine::MultiGPUEngine engine_rr(basis, config_rr);
    consumers::FockBuilder fock_rr(nbf);
    fock_rr.set_threading_strategy(consumers::FockThreadingStrategy::Atomic);
    fock_rr.set_density(D.data(), nbf);
    engine::MultiGPUFockAdapter adapter_rr(fock_rr);
    engine_rr.compute_all_eri(adapter_rr);

    auto J_default = fock_default.get_coulomb_matrix();
    auto J_rr = fock_rr.get_coulomb_matrix();
    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(J_default[i], J_rr[i], TIGHT_TOL)
            << "Coulomb mismatch (default vs round-robin) at index " << i;
    }
}

/// Test device manager functionality
TEST_F(MultiGPUCorrectnessTest, DeviceManagerBasics) {
    auto& dm = device::DeviceManager::instance();

    EXPECT_GE(dm.device_count(), 2);

    // Check device properties
    for (int i = 0; i < dm.device_count(); ++i) {
        const auto& props = dm.get_device_properties(i);
        EXPECT_EQ(props.device_id, i);
        EXPECT_GT(props.total_memory, 0);
        EXPECT_GT(props.multiprocessor_count, 0);
        EXPECT_FALSE(props.name.empty());
    }

    // Test device selection
    dm.set_active_devices({0, 1});
    EXPECT_EQ(dm.active_device_count(), 2);

    dm.set_all_devices();
    EXPECT_EQ(dm.active_device_count(), dm.device_count());
}

/// Test ScopedDevice RAII
TEST_F(MultiGPUCorrectnessTest, ScopedDeviceRestores) {
    auto& dm = device::DeviceManager::instance();
    dm.set_current_device(0);

    EXPECT_EQ(dm.current_device(), 0);

    {
        device::ScopedDevice guard(1);
        EXPECT_EQ(dm.current_device(), 1);
    }

    // Should be restored to 0
    EXPECT_EQ(dm.current_device(), 0);
}

}  // namespace libaccint::test

#endif  // LIBACCINT_USE_CUDA

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
