// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_overlap_kernel_cuda.cpp
/// @brief Unit tests for CUDA overlap integral kernel

#include <gtest/gtest.h>
#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/kernels/overlap_kernel_cuda.hpp>
#include <libaccint/kernels/overlap_kernel.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/device_data.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/memory/device_memory.hpp>

#include <cuda_runtime.h>
#include <vector>
#include <cmath>

namespace libaccint {

using memory::DeviceMemoryManager;
using memory::DeviceBuffer;
using kernels::cuda::dispatch_overlap_kernel;
using kernels::cuda::overlap_output_size;

// ============================================================================
// Test Fixture
// ============================================================================

class OverlapKernelCudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }

    /// Compare GPU results with CPU reference implementation
    void compare_with_cpu(const Shell& shell_a, const Shell& shell_b,
                          double tolerance = 1e-12) {
        // Create ShellSets
        ShellSet bra_set(shell_a.angular_momentum(), shell_a.n_primitives());
        ShellSet ket_set(shell_b.angular_momentum(), shell_b.n_primitives());
        bra_set.add_shell(shell_a);
        ket_set.add_shell(shell_b);

        // Upload to device
        basis::ShellSetDeviceData bra_data = basis::upload_shell_set(bra_set);
        basis::ShellSetDeviceData ket_data = basis::upload_shell_set(ket_set);
        DeviceMemoryManager::synchronize();

        // Create pair
        basis::ShellSetPairDeviceData pair;
        pair.bra = bra_data;
        pair.ket = ket_data;

        // Allocate output buffer
        size_t output_size = overlap_output_size(pair);
        DeviceBuffer<double> d_output(output_size);

        // Launch GPU kernel
        dispatch_overlap_kernel(pair, d_output.data());
        DeviceMemoryManager::synchronize();

        // Download results
        std::vector<double> gpu_results(output_size);
        d_output.download(gpu_results.data(), output_size);
        DeviceMemoryManager::synchronize();

        // Compute CPU reference
        OverlapBuffer cpu_buffer;
        kernels::compute_overlap(shell_a, shell_b, cpu_buffer);

        // Compare results
        const int na = shell_a.n_functions();
        const int nb = shell_b.n_functions();
        for (int a = 0; a < na; ++a) {
            for (int b = 0; b < nb; ++b) {
                const double gpu_val = gpu_results[a * nb + b];
                const double cpu_val = cpu_buffer(a, b);
                EXPECT_NEAR(gpu_val, cpu_val, tolerance)
                    << "Mismatch at (" << a << ", " << b << "): "
                    << "GPU=" << gpu_val << ", CPU=" << cpu_val;
            }
        }

        // Clean up
        basis::free_shell_set_device_data(bra_data);
        basis::free_shell_set_device_data(ket_data);
    }
};

// ============================================================================
// (s|s) Tests
// ============================================================================

TEST_F(OverlapKernelCudaTest, SsOverlap_SameCenter) {
    Point3D center{0.0, 0.0, 0.0};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::S, center, exponents, coefficients);
    Shell shell_b(AngularMomentum::S, center, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_a.set_atom_index(0);
    shell_a.set_function_index(0);
    shell_b.set_shell_index(1);
    shell_b.set_atom_index(0);
    shell_b.set_function_index(1);

    compare_with_cpu(shell_a, shell_b);
}

TEST_F(OverlapKernelCudaTest, SsOverlap_DifferentCenters) {
    Point3D center_a{0.0, 0.0, 0.0};
    Point3D center_b{1.5, 0.0, 0.0};
    std::vector<double> exponents = {0.5};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::S, center_a, exponents, coefficients);
    Shell shell_b(AngularMomentum::S, center_b, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_a.set_atom_index(0);
    shell_a.set_function_index(0);
    shell_b.set_shell_index(1);
    shell_b.set_atom_index(1);
    shell_b.set_function_index(1);

    compare_with_cpu(shell_a, shell_b);
}

TEST_F(OverlapKernelCudaTest, SsOverlap_ContractedBasis) {
    Point3D center_a{0.0, 0.0, 0.0};
    Point3D center_b{1.0, 0.5, 0.3};
    std::vector<double> exp_a = {10.0, 1.0, 0.1};
    std::vector<double> coef_a = {0.1, 0.5, 0.4};
    std::vector<double> exp_b = {5.0, 0.5};
    std::vector<double> coef_b = {0.6, 0.4};

    Shell shell_a(AngularMomentum::S, center_a, exp_a, coef_a);
    Shell shell_b(AngularMomentum::S, center_b, exp_b, coef_b);
    shell_a.set_shell_index(0);
    shell_a.set_atom_index(0);
    shell_a.set_function_index(0);
    shell_b.set_shell_index(1);
    shell_b.set_atom_index(1);
    shell_b.set_function_index(1);

    compare_with_cpu(shell_a, shell_b);
}

// ============================================================================
// (s|p) and (p|s) Tests
// ============================================================================

TEST_F(OverlapKernelCudaTest, SpOverlap) {
    Point3D center_a{0.0, 0.0, 0.0};
    Point3D center_b{1.0, 0.0, 0.0};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::S, center_a, exponents, coefficients);
    Shell shell_b(AngularMomentum::P, center_b, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_a.set_atom_index(0);
    shell_a.set_function_index(0);
    shell_b.set_shell_index(1);
    shell_b.set_atom_index(1);
    shell_b.set_function_index(1);

    compare_with_cpu(shell_a, shell_b);
}

TEST_F(OverlapKernelCudaTest, PsOverlap) {
    Point3D center_a{0.5, 0.5, 0.5};
    Point3D center_b{-0.5, 0.0, 0.0};
    std::vector<double> exponents = {0.8};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::P, center_a, exponents, coefficients);
    Shell shell_b(AngularMomentum::S, center_b, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_a.set_atom_index(0);
    shell_a.set_function_index(0);
    shell_b.set_shell_index(1);
    shell_b.set_atom_index(1);
    shell_b.set_function_index(3);

    compare_with_cpu(shell_a, shell_b);
}

// ============================================================================
// (p|p) Tests
// ============================================================================

TEST_F(OverlapKernelCudaTest, PpOverlap_SameCenter) {
    Point3D center{0.0, 0.0, 0.0};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::P, center, exponents, coefficients);
    Shell shell_b(AngularMomentum::P, center, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_a.set_atom_index(0);
    shell_a.set_function_index(0);
    shell_b.set_shell_index(1);
    shell_b.set_atom_index(0);
    shell_b.set_function_index(3);

    compare_with_cpu(shell_a, shell_b);
}

TEST_F(OverlapKernelCudaTest, PpOverlap_DifferentCenters) {
    Point3D center_a{0.0, 0.0, 0.0};
    Point3D center_b{1.5, 1.0, 0.5};
    std::vector<double> exp_a = {2.0, 0.5};
    std::vector<double> coef_a = {0.7, 0.3};

    Shell shell_a(AngularMomentum::P, center_a, exp_a, coef_a);
    Shell shell_b(AngularMomentum::P, center_b, exp_a, coef_a);
    shell_a.set_shell_index(0);
    shell_a.set_atom_index(0);
    shell_a.set_function_index(0);
    shell_b.set_shell_index(1);
    shell_b.set_atom_index(1);
    shell_b.set_function_index(3);

    compare_with_cpu(shell_a, shell_b);
}

// ============================================================================
// (s|d) and (d|s) Tests
// ============================================================================

TEST_F(OverlapKernelCudaTest, SdOverlap) {
    Point3D center_a{0.0, 0.0, 0.0};
    Point3D center_b{1.0, 0.0, 0.0};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::S, center_a, exponents, coefficients);
    Shell shell_b(AngularMomentum::D, center_b, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_a.set_atom_index(0);
    shell_a.set_function_index(0);
    shell_b.set_shell_index(1);
    shell_b.set_atom_index(1);
    shell_b.set_function_index(1);

    compare_with_cpu(shell_a, shell_b);
}

TEST_F(OverlapKernelCudaTest, DsOverlap) {
    Point3D center_a{0.5, 0.5, 0.5};
    Point3D center_b{-0.5, 0.0, 0.0};
    std::vector<double> exponents = {0.8};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::D, center_a, exponents, coefficients);
    Shell shell_b(AngularMomentum::S, center_b, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_a.set_atom_index(0);
    shell_a.set_function_index(0);
    shell_b.set_shell_index(1);
    shell_b.set_atom_index(1);
    shell_b.set_function_index(6);

    compare_with_cpu(shell_a, shell_b);
}

// ============================================================================
// (p|d) and (d|p) Tests
// ============================================================================

TEST_F(OverlapKernelCudaTest, PdOverlap) {
    Point3D center_a{0.0, 0.0, 0.0};
    Point3D center_b{1.5, 0.5, 0.0};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::P, center_a, exponents, coefficients);
    Shell shell_b(AngularMomentum::D, center_b, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_a.set_atom_index(0);
    shell_a.set_function_index(0);
    shell_b.set_shell_index(1);
    shell_b.set_atom_index(1);
    shell_b.set_function_index(3);

    compare_with_cpu(shell_a, shell_b);
}

TEST_F(OverlapKernelCudaTest, DpOverlap) {
    Point3D center_a{0.5, 0.5, 0.5};
    Point3D center_b{-0.5, 0.2, 0.1};
    std::vector<double> exp_a = {3.0, 0.5};
    std::vector<double> coef_a = {0.4, 0.6};

    Shell shell_a(AngularMomentum::D, center_a, exp_a, coef_a);
    Shell shell_b(AngularMomentum::P, center_b, exp_a, coef_a);
    shell_a.set_shell_index(0);
    shell_a.set_atom_index(0);
    shell_a.set_function_index(0);
    shell_b.set_shell_index(1);
    shell_b.set_atom_index(1);
    shell_b.set_function_index(6);

    compare_with_cpu(shell_a, shell_b);
}

// ============================================================================
// (d|d) Tests
// ============================================================================

TEST_F(OverlapKernelCudaTest, DdOverlap_SameCenter) {
    Point3D center{0.0, 0.0, 0.0};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::D, center, exponents, coefficients);
    Shell shell_b(AngularMomentum::D, center, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_a.set_atom_index(0);
    shell_a.set_function_index(0);
    shell_b.set_shell_index(1);
    shell_b.set_atom_index(0);
    shell_b.set_function_index(6);

    compare_with_cpu(shell_a, shell_b);
}

TEST_F(OverlapKernelCudaTest, DdOverlap_DifferentCenters) {
    Point3D center_a{0.0, 0.0, 0.0};
    Point3D center_b{1.0, 1.0, 1.0};
    std::vector<double> exp_a = {2.0, 0.5, 0.1};
    std::vector<double> coef_a = {0.3, 0.5, 0.2};

    Shell shell_a(AngularMomentum::D, center_a, exp_a, coef_a);
    Shell shell_b(AngularMomentum::D, center_b, exp_a, coef_a);
    shell_a.set_shell_index(0);
    shell_a.set_atom_index(0);
    shell_a.set_function_index(0);
    shell_b.set_shell_index(1);
    shell_b.set_atom_index(1);
    shell_b.set_function_index(6);

    compare_with_cpu(shell_a, shell_b);
}

// ============================================================================
// Multiple Shell Pairs Test
// ============================================================================

TEST_F(OverlapKernelCudaTest, MultipleShellPairs) {
    // Create two s-type shells with different centers
    Point3D center1{0.0, 0.0, 0.0};
    Point3D center2{1.5, 0.0, 0.0};
    Point3D center3{0.0, 1.5, 0.0};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell1(AngularMomentum::S, center1, exponents, coefficients);
    Shell shell2(AngularMomentum::S, center2, exponents, coefficients);
    Shell shell3(AngularMomentum::S, center3, exponents, coefficients);
    shell1.set_shell_index(0);
    shell1.set_atom_index(0);
    shell1.set_function_index(0);
    shell2.set_shell_index(1);
    shell2.set_atom_index(1);
    shell2.set_function_index(1);
    shell3.set_shell_index(2);
    shell3.set_atom_index(2);
    shell3.set_function_index(2);

    // Create ShellSets with multiple shells
    ShellSet bra_set(0, 1);  // s-type, 1 primitive
    ShellSet ket_set(0, 1);
    bra_set.add_shell(shell1);
    bra_set.add_shell(shell2);
    ket_set.add_shell(shell1);
    ket_set.add_shell(shell3);

    // Upload to device
    basis::ShellSetDeviceData bra_data = basis::upload_shell_set(bra_set);
    basis::ShellSetDeviceData ket_data = basis::upload_shell_set(ket_set);
    DeviceMemoryManager::synchronize();

    // Create pair
    basis::ShellSetPairDeviceData pair;
    pair.bra = bra_data;
    pair.ket = ket_data;

    // Allocate output buffer
    size_t output_size = overlap_output_size(pair);
    ASSERT_EQ(output_size, 4u);  // 2x2 shell pairs, 1 function each

    DeviceBuffer<double> d_output(output_size);

    // Launch GPU kernel
    dispatch_overlap_kernel(pair, d_output.data());
    DeviceMemoryManager::synchronize();

    // Download results
    std::vector<double> gpu_results(output_size);
    d_output.download(gpu_results.data(), output_size);
    DeviceMemoryManager::synchronize();

    // Compute CPU references for each shell pair
    OverlapBuffer cpu_buffer;
    auto bra_shells = bra_set.shells();
    auto ket_shells = ket_set.shells();

    for (size_t i = 0; i < bra_shells.size(); ++i) {
        for (size_t j = 0; j < ket_shells.size(); ++j) {
            kernels::compute_overlap(bra_shells[i], ket_shells[j], cpu_buffer);
            size_t idx = i * ket_shells.size() + j;
            EXPECT_NEAR(gpu_results[idx], cpu_buffer(0, 0), 1e-12)
                << "Mismatch at shell pair (" << i << ", " << j << ")";
        }
    }

    // Clean up
    basis::free_shell_set_device_data(bra_data);
    basis::free_shell_set_device_data(ket_data);
}

}  // namespace libaccint

#else  // LIBACCINT_USE_CUDA

TEST(OverlapKernelCudaTest, CudaNotEnabled) {
    GTEST_SKIP() << "CUDA support not enabled";
}

#endif  // LIBACCINT_USE_CUDA
