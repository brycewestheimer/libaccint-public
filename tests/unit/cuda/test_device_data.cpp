// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_device_data.cpp
/// @brief Unit tests for CUDA device data upload utilities

#include <gtest/gtest.h>
#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/basis/device_data.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/operators/device_operator_data.hpp>
#include <libaccint/memory/device_memory.hpp>

#include <cuda_runtime.h>
#include <vector>
#include <cmath>

namespace libaccint {

using memory::DeviceMemoryManager;

// ============================================================================
// Test Fixture
// ============================================================================

class DeviceDataTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

// ============================================================================
// ShellSetDeviceData Tests
// ============================================================================

TEST_F(DeviceDataTest, UploadEmptyShellSet) {
    ShellSet empty_set(0, 1);  // s-type with 1 primitive
    basis::ShellSetDeviceData data = basis::upload_shell_set(empty_set);

    EXPECT_FALSE(data.valid());
    EXPECT_EQ(data.n_shells, 0);
}

TEST_F(DeviceDataTest, UploadSingleShell) {
    // Create a single s-type shell with 3 primitives
    Point3D center{1.0, 2.0, 3.0};
    std::vector<double> exponents = {10.0, 1.0, 0.1};
    std::vector<double> coefficients = {0.1, 0.5, 0.4};

    Shell shell(AngularMomentum::S, center, exponents, coefficients);
    shell.set_shell_index(0);
    shell.set_atom_index(0);
    shell.set_function_index(0);

    ShellSet shell_set(0, 3);
    shell_set.add_shell(shell);

    // Upload to device
    basis::ShellSetDeviceData data = basis::upload_shell_set(shell_set);
    DeviceMemoryManager::synchronize();

    ASSERT_TRUE(data.valid());
    EXPECT_EQ(data.n_shells, 1);
    EXPECT_EQ(data.angular_momentum, 0);
    EXPECT_EQ(data.n_primitives, 3);
    EXPECT_EQ(data.n_functions_per_shell, 1);  // s-type has 1 function

    // Download and verify exponents
    std::vector<double> downloaded_exp(3);
    DeviceMemoryManager::copy_to_host(downloaded_exp.data(), data.d_exponents, 3);
    DeviceMemoryManager::synchronize();

    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(downloaded_exp[i], exponents[i]);
    }

    // Download and verify center
    double center_x, center_y, center_z;
    DeviceMemoryManager::copy_to_host(&center_x, data.d_centers_x, 1);
    DeviceMemoryManager::copy_to_host(&center_y, data.d_centers_y, 1);
    DeviceMemoryManager::copy_to_host(&center_z, data.d_centers_z, 1);
    DeviceMemoryManager::synchronize();

    EXPECT_DOUBLE_EQ(center_x, 1.0);
    EXPECT_DOUBLE_EQ(center_y, 2.0);
    EXPECT_DOUBLE_EQ(center_z, 3.0);

    // Clean up
    basis::free_shell_set_device_data(data);
    EXPECT_FALSE(data.valid());
}

TEST_F(DeviceDataTest, UploadMultipleShells) {
    // Create multiple p-type shells with 2 primitives each
    Point3D center1{0.0, 0.0, 0.0};
    Point3D center2{1.5, 0.0, 0.0};
    std::vector<double> exponents = {5.0, 0.5};
    std::vector<double> coefficients = {0.6, 0.4};

    Shell shell1(AngularMomentum::P, center1, exponents, coefficients);
    Shell shell2(AngularMomentum::P, center2, exponents, coefficients);
    shell1.set_shell_index(0);
    shell1.set_atom_index(0);
    shell1.set_function_index(0);
    shell2.set_shell_index(1);
    shell2.set_atom_index(1);
    shell2.set_function_index(3);

    ShellSet shell_set(1, 2);
    shell_set.add_shell(shell1);
    shell_set.add_shell(shell2);

    // Upload to device
    basis::ShellSetDeviceData data = basis::upload_shell_set(shell_set);
    DeviceMemoryManager::synchronize();

    ASSERT_TRUE(data.valid());
    EXPECT_EQ(data.n_shells, 2);
    EXPECT_EQ(data.angular_momentum, 1);
    EXPECT_EQ(data.n_primitives, 2);
    EXPECT_EQ(data.n_functions_per_shell, 3);  // p-type has 3 functions

    // Total primitives = 2 shells * 2 primitives = 4
    EXPECT_EQ(data.total_primitives(), 4);

    // Verify function offsets
    std::vector<int> offsets(2);
    DeviceMemoryManager::copy_to_host(offsets.data(), data.d_function_offsets, 2);
    DeviceMemoryManager::synchronize();

    EXPECT_EQ(offsets[0], 0);
    EXPECT_EQ(offsets[1], 3);

    basis::free_shell_set_device_data(data);
}

TEST_F(DeviceDataTest, ShellSetDeviceHandle) {
    Point3D center{0.0, 0.0, 0.0};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell(AngularMomentum::S, center, exponents, coefficients);
    shell.set_shell_index(0);
    shell.set_atom_index(0);
    shell.set_function_index(0);

    ShellSet shell_set(0, 1);
    shell_set.add_shell(shell);

    {
        basis::ShellSetDeviceHandle handle(shell_set);
        DeviceMemoryManager::synchronize();

        EXPECT_TRUE(handle.valid());
        EXPECT_EQ(handle.data().n_shells, 1);
        EXPECT_EQ(handle.key().angular_momentum, 0);
        EXPECT_EQ(handle.key().n_primitives, 1);
    }

    // Handle destroyed - memory freed
    DeviceMemoryManager::synchronize();
    SUCCEED();  // No crash means RAII worked
}

TEST_F(DeviceDataTest, ShellSetDeviceHandleMove) {
    Point3D center{0.0, 0.0, 0.0};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell(AngularMomentum::S, center, exponents, coefficients);
    shell.set_shell_index(0);
    shell.set_atom_index(0);
    shell.set_function_index(0);

    ShellSet shell_set(0, 1);
    shell_set.add_shell(shell);

    basis::ShellSetDeviceHandle handle1(shell_set);
    DeviceMemoryManager::synchronize();
    ASSERT_TRUE(handle1.valid());

    // Move construct
    basis::ShellSetDeviceHandle handle2(std::move(handle1));
    EXPECT_FALSE(handle1.valid());
    EXPECT_TRUE(handle2.valid());

    // Move assign
    basis::ShellSetDeviceHandle handle3;
    handle3 = std::move(handle2);
    EXPECT_FALSE(handle2.valid());
    EXPECT_TRUE(handle3.valid());
}

TEST_F(DeviceDataTest, ShellSetDeviceCache) {
    Point3D center{0.0, 0.0, 0.0};
    std::vector<double> exponents = {1.0, 0.5};
    std::vector<double> coefficients = {0.6, 0.4};

    Shell shell(AngularMomentum::D, center, exponents, coefficients);
    shell.set_shell_index(0);
    shell.set_atom_index(0);
    shell.set_function_index(0);

    ShellSet shell_set(2, 2);
    shell_set.add_shell(shell);

    basis::ShellSetDeviceCache cache;

    EXPECT_EQ(cache.size(), 0);
    EXPECT_FALSE(cache.contains(shell_set.key()));

    // First upload
    const basis::ShellSetDeviceData& data1 = cache.get_or_upload(shell_set);
    DeviceMemoryManager::synchronize();

    EXPECT_TRUE(data1.valid());
    EXPECT_EQ(cache.size(), 1);
    EXPECT_TRUE(cache.contains(shell_set.key()));

    // Second call should return same data (cached)
    const basis::ShellSetDeviceData& data2 = cache.get_or_upload(shell_set);
    EXPECT_EQ(&data1, &data2);  // Same object
    EXPECT_EQ(cache.size(), 1);

    // Clear cache
    cache.clear();
    EXPECT_EQ(cache.size(), 0);
}

// ============================================================================
// PointChargeDeviceData Tests
// ============================================================================

TEST_F(DeviceDataTest, UploadEmptyPointCharges) {
    std::vector<double> empty;
    operators::DevicePointChargeData data =
        operators::upload_point_charges(empty, empty, empty, empty);

    EXPECT_FALSE(data.valid());
    EXPECT_EQ(data.n_charges, 0);
}

TEST_F(DeviceDataTest, UploadSinglePointCharge) {
    std::vector<double> x = {1.0};
    std::vector<double> y = {2.0};
    std::vector<double> z = {3.0};
    std::vector<double> charges = {8.0};  // Oxygen

    operators::DevicePointChargeData data =
        operators::upload_point_charges(x, y, z, charges);
    DeviceMemoryManager::synchronize();

    ASSERT_TRUE(data.valid());
    EXPECT_EQ(data.n_charges, 1);

    // Download and verify
    double dl_x, dl_y, dl_z, dl_q;
    DeviceMemoryManager::copy_to_host(&dl_x, data.d_x, 1);
    DeviceMemoryManager::copy_to_host(&dl_y, data.d_y, 1);
    DeviceMemoryManager::copy_to_host(&dl_z, data.d_z, 1);
    DeviceMemoryManager::copy_to_host(&dl_q, data.d_charges, 1);
    DeviceMemoryManager::synchronize();

    EXPECT_DOUBLE_EQ(dl_x, 1.0);
    EXPECT_DOUBLE_EQ(dl_y, 2.0);
    EXPECT_DOUBLE_EQ(dl_z, 3.0);
    EXPECT_DOUBLE_EQ(dl_q, 8.0);

    operators::free_point_charge_device_data(data);
    EXPECT_FALSE(data.valid());
}

TEST_F(DeviceDataTest, UploadMultiplePointCharges) {
    // Water molecule: O at origin, two H atoms
    std::vector<double> x = {0.0, 1.43, -1.43};
    std::vector<double> y = {0.0, 0.0, 0.0};
    std::vector<double> z = {0.0, 1.11, 1.11};
    std::vector<double> charges = {8.0, 1.0, 1.0};

    operators::DevicePointChargeData data =
        operators::upload_point_charges(x, y, z, charges);
    DeviceMemoryManager::synchronize();

    ASSERT_TRUE(data.valid());
    EXPECT_EQ(data.n_charges, 3);

    // Download and verify all charges
    std::vector<double> dl_charges(3);
    DeviceMemoryManager::copy_to_host(dl_charges.data(), data.d_charges, 3);
    DeviceMemoryManager::synchronize();

    EXPECT_DOUBLE_EQ(dl_charges[0], 8.0);
    EXPECT_DOUBLE_EQ(dl_charges[1], 1.0);
    EXPECT_DOUBLE_EQ(dl_charges[2], 1.0);

    operators::free_point_charge_device_data(data);
}

TEST_F(DeviceDataTest, UploadPointChargesMismatchedSizes) {
    std::vector<double> x = {0.0, 1.0};
    std::vector<double> y = {0.0};  // Wrong size!
    std::vector<double> z = {0.0, 1.0};
    std::vector<double> charges = {8.0, 1.0};

    EXPECT_THROW(
        operators::upload_point_charges(x, y, z, charges),
        InvalidArgumentException);
}

TEST_F(DeviceDataTest, PointChargeDeviceHandle) {
    std::vector<double> x = {0.0};
    std::vector<double> y = {0.0};
    std::vector<double> z = {0.0};
    std::vector<double> charges = {6.0};  // Carbon

    {
        operators::PointChargeDeviceHandle handle(x, y, z, charges);
        DeviceMemoryManager::synchronize();

        EXPECT_TRUE(handle.valid());
        EXPECT_EQ(handle.n_charges(), 1);
        EXPECT_EQ(handle.data().n_charges, 1);
    }

    DeviceMemoryManager::synchronize();
    SUCCEED();  // No crash means RAII worked
}

TEST_F(DeviceDataTest, PointChargeDeviceHandleMove) {
    std::vector<double> x = {0.0};
    std::vector<double> y = {0.0};
    std::vector<double> z = {0.0};
    std::vector<double> charges = {1.0};

    operators::PointChargeDeviceHandle handle1(x, y, z, charges);
    DeviceMemoryManager::synchronize();
    ASSERT_TRUE(handle1.valid());

    // Move construct
    operators::PointChargeDeviceHandle handle2(std::move(handle1));
    EXPECT_FALSE(handle1.valid());
    EXPECT_TRUE(handle2.valid());

    // Move assign
    operators::PointChargeDeviceHandle handle3;
    handle3 = std::move(handle2);
    EXPECT_FALSE(handle2.valid());
    EXPECT_TRUE(handle3.valid());
}

TEST_F(DeviceDataTest, PointChargeDeviceHandleEmpty) {
    operators::PointChargeDeviceHandle handle;
    EXPECT_FALSE(handle.valid());
    EXPECT_EQ(handle.n_charges(), 0);
}

// ============================================================================
// ShellSetPairDeviceData Tests
// ============================================================================

TEST_F(DeviceDataTest, ShellSetPairDeviceDataConstruction) {
    // Create two shell sets
    Point3D center1{0.0, 0.0, 0.0};
    Point3D center2{1.0, 0.0, 0.0};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell1(AngularMomentum::S, center1, exponents, coefficients);
    Shell shell2(AngularMomentum::P, center2, exponents, coefficients);
    shell1.set_shell_index(0);
    shell1.set_atom_index(0);
    shell1.set_function_index(0);
    shell2.set_shell_index(1);
    shell2.set_atom_index(1);
    shell2.set_function_index(1);

    ShellSet s_set(0, 1);
    ShellSet p_set(1, 1);
    s_set.add_shell(shell1);
    p_set.add_shell(shell2);

    // Upload both
    basis::ShellSetDeviceData bra_data = basis::upload_shell_set(s_set);
    basis::ShellSetDeviceData ket_data = basis::upload_shell_set(p_set);
    DeviceMemoryManager::synchronize();

    // Create pair
    basis::ShellSetPairDeviceData pair;
    pair.bra = bra_data;
    pair.ket = ket_data;

    EXPECT_TRUE(pair.valid());
    EXPECT_EQ(pair.n_pairs(), 1);

    // Clean up (be careful not to double-free)
    // We need to clear the pair's pointers before freeing
    basis::free_shell_set_device_data(bra_data);
    basis::free_shell_set_device_data(ket_data);
}

}  // namespace libaccint

#else  // LIBACCINT_USE_CUDA

TEST(DeviceDataTest, CudaNotEnabled) {
    GTEST_SKIP() << "CUDA support not enabled";
}

#endif  // LIBACCINT_USE_CUDA
