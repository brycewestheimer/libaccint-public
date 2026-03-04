// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_eri_kernel_cuda.cpp
/// @brief Unit tests for CUDA electron repulsion integral kernel

#include <gtest/gtest.h>
#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/kernels/eri_kernel_cuda.hpp>
#include <libaccint/kernels/eri_kernel.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/device_data.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/memory/device_memory.hpp>

#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Include device Boys function for initialization
namespace libaccint::device::math {
    double* boys_device_init(cudaStream_t stream);
    void boys_device_cleanup();
    double* boys_device_get_coeffs();
    bool boys_device_is_initialized();
}

namespace libaccint {

using memory::DeviceMemoryManager;
using memory::DeviceBuffer;
using kernels::cuda::dispatch_eri_kernel;
using kernels::cuda::eri_output_size;

// ============================================================================
// Test Fixture
// ============================================================================

class EriKernelCudaTest : public ::testing::Test {
protected:
    double* d_boys_coeffs_ = nullptr;

    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        // Initialize device Boys function tables
        if (!device::math::boys_device_is_initialized()) {
            d_boys_coeffs_ = device::math::boys_device_init(nullptr);
        } else {
            d_boys_coeffs_ = device::math::boys_device_get_coeffs();
        }
        DeviceMemoryManager::synchronize();
    }

    void TearDown() override {
        // Don't cleanup Boys tables - they're shared across tests
    }

    /// Compare GPU results with CPU reference implementation
    void compare_with_cpu(const Shell& shell_a, const Shell& shell_b,
                          const Shell& shell_c, const Shell& shell_d,
                          double tolerance = 1e-10,
                          bool use_relative = false) {
        // Create ShellSets
        ShellSet set_a(shell_a.angular_momentum(), shell_a.n_primitives());
        ShellSet set_b(shell_b.angular_momentum(), shell_b.n_primitives());
        ShellSet set_c(shell_c.angular_momentum(), shell_c.n_primitives());
        ShellSet set_d(shell_d.angular_momentum(), shell_d.n_primitives());
        set_a.add_shell(shell_a);
        set_b.add_shell(shell_b);
        set_c.add_shell(shell_c);
        set_d.add_shell(shell_d);

        // Upload shells to device
        basis::ShellSetDeviceData data_a = basis::upload_shell_set(set_a);
        basis::ShellSetDeviceData data_b = basis::upload_shell_set(set_b);
        basis::ShellSetDeviceData data_c = basis::upload_shell_set(set_c);
        basis::ShellSetDeviceData data_d = basis::upload_shell_set(set_d);
        DeviceMemoryManager::synchronize();

        // Create quartet
        basis::ShellSetQuartetDeviceData quartet;
        quartet.a = data_a;
        quartet.b = data_b;
        quartet.c = data_c;
        quartet.d = data_d;

        // Allocate output buffer
        size_t output_size = eri_output_size(quartet);
        DeviceBuffer<double> d_output(output_size);

        // Launch GPU kernel
        dispatch_eri_kernel(quartet, d_boys_coeffs_, d_output.data());
        DeviceMemoryManager::synchronize();

        // Download results
        std::vector<double> gpu_results(output_size);
        d_output.download(gpu_results.data(), output_size);
        DeviceMemoryManager::synchronize();

        // Compute CPU reference
        TwoElectronBuffer<0> cpu_buffer;
        kernels::compute_eri(shell_a, shell_b, shell_c, shell_d, cpu_buffer);

        // Compare results
        const int na = shell_a.n_functions();
        const int nb = shell_b.n_functions();
        const int nc = shell_c.n_functions();
        const int nd = shell_d.n_functions();

        int idx = 0;
        for (int a = 0; a < na; ++a) {
            for (int b = 0; b < nb; ++b) {
                for (int c = 0; c < nc; ++c) {
                    for (int d = 0; d < nd; ++d) {
                        const double gpu_val = gpu_results[idx];
                        const double cpu_val = cpu_buffer(a, b, c, d);
                        if (use_relative && std::abs(cpu_val) > 1e-10) {
                            const double rel_err = std::abs((gpu_val - cpu_val) / cpu_val);
                            EXPECT_LT(rel_err, tolerance)
                                << "Relative error at (" << a << ", " << b << ", "
                                << c << ", " << d << "): "
                                << "GPU=" << gpu_val << ", CPU=" << cpu_val
                                << ", rel_err=" << rel_err;
                        } else {
                            EXPECT_NEAR(gpu_val, cpu_val, tolerance)
                                << "Mismatch at (" << a << ", " << b << ", "
                                << c << ", " << d << "): "
                                << "GPU=" << gpu_val << ", CPU=" << cpu_val;
                        }
                        idx++;
                    }
                }
            }
        }

        // Clean up
        basis::free_shell_set_device_data(data_a);
        basis::free_shell_set_device_data(data_b);
        basis::free_shell_set_device_data(data_c);
        basis::free_shell_set_device_data(data_d);
    }
};

// ============================================================================
// (ss|ss) Tests
// ============================================================================

TEST_F(EriKernelCudaTest, SsssEri_SameCenter) {
    Point3D center{0.0, 0.0, 0.0};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::S, center, exponents, coefficients);
    Shell shell_b(AngularMomentum::S, center, exponents, coefficients);
    Shell shell_c(AngularMomentum::S, center, exponents, coefficients);
    Shell shell_d(AngularMomentum::S, center, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_b.set_shell_index(1);
    shell_c.set_shell_index(2);
    shell_d.set_shell_index(3);
    shell_a.set_function_index(0);
    shell_b.set_function_index(1);
    shell_c.set_function_index(2);
    shell_d.set_function_index(3);

    compare_with_cpu(shell_a, shell_b, shell_c, shell_d);
}

TEST_F(EriKernelCudaTest, SsssEri_DifferentCenters) {
    Point3D center_a{0.0, 0.0, 0.0};
    Point3D center_b{1.0, 0.0, 0.0};
    Point3D center_c{0.0, 1.0, 0.0};
    Point3D center_d{0.0, 0.0, 1.0};
    std::vector<double> exponents = {0.5};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::S, center_a, exponents, coefficients);
    Shell shell_b(AngularMomentum::S, center_b, exponents, coefficients);
    Shell shell_c(AngularMomentum::S, center_c, exponents, coefficients);
    Shell shell_d(AngularMomentum::S, center_d, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_b.set_shell_index(1);
    shell_c.set_shell_index(2);
    shell_d.set_shell_index(3);
    shell_a.set_function_index(0);
    shell_b.set_function_index(1);
    shell_c.set_function_index(2);
    shell_d.set_function_index(3);

    compare_with_cpu(shell_a, shell_b, shell_c, shell_d);
}

TEST_F(EriKernelCudaTest, SsssEri_Contracted) {
    Point3D center_a{0.0, 0.0, 0.0};
    Point3D center_b{1.5, 0.0, 0.0};
    std::vector<double> exponents = {2.0, 0.5, 0.1};
    std::vector<double> coefficients = {0.3, 0.5, 0.2};

    Shell shell_a(AngularMomentum::S, center_a, exponents, coefficients);
    Shell shell_b(AngularMomentum::S, center_b, exponents, coefficients);
    Shell shell_c(AngularMomentum::S, center_a, exponents, coefficients);
    Shell shell_d(AngularMomentum::S, center_b, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_b.set_shell_index(1);
    shell_c.set_shell_index(2);
    shell_d.set_shell_index(3);
    shell_a.set_function_index(0);
    shell_b.set_function_index(1);
    shell_c.set_function_index(2);
    shell_d.set_function_index(3);

    compare_with_cpu(shell_a, shell_b, shell_c, shell_d);
}

// ============================================================================
// (ss|sp) and related Tests
// ============================================================================

TEST_F(EriKernelCudaTest, SsspEri) {
    Point3D center_a{0.0, 0.0, 0.0};
    Point3D center_b{1.0, 0.0, 0.0};
    Point3D center_c{0.0, 1.0, 0.0};
    Point3D center_d{0.5, 0.5, 0.5};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::S, center_a, exponents, coefficients);
    Shell shell_b(AngularMomentum::S, center_b, exponents, coefficients);
    Shell shell_c(AngularMomentum::S, center_c, exponents, coefficients);
    Shell shell_d(AngularMomentum::P, center_d, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_b.set_shell_index(1);
    shell_c.set_shell_index(2);
    shell_d.set_shell_index(3);
    shell_a.set_function_index(0);
    shell_b.set_function_index(1);
    shell_c.set_function_index(2);
    shell_d.set_function_index(3);

    compare_with_cpu(shell_a, shell_b, shell_c, shell_d);
}

TEST_F(EriKernelCudaTest, SsppEri) {
    Point3D center_a{0.0, 0.0, 0.0};
    Point3D center_b{1.0, 0.0, 0.0};
    Point3D center_c{0.0, 1.0, 0.0};
    Point3D center_d{0.5, 0.5, 0.5};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::S, center_a, exponents, coefficients);
    Shell shell_b(AngularMomentum::S, center_b, exponents, coefficients);
    Shell shell_c(AngularMomentum::P, center_c, exponents, coefficients);
    Shell shell_d(AngularMomentum::P, center_d, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_b.set_shell_index(1);
    shell_c.set_shell_index(2);
    shell_d.set_shell_index(3);
    shell_a.set_function_index(0);
    shell_b.set_function_index(1);
    shell_c.set_function_index(2);
    shell_d.set_function_index(5);

    compare_with_cpu(shell_a, shell_b, shell_c, shell_d);
}

// ============================================================================
// (sp|sp) Tests
// ============================================================================

TEST_F(EriKernelCudaTest, SpspEri) {
    Point3D center_a{0.0, 0.0, 0.0};
    Point3D center_b{1.0, 0.0, 0.0};
    Point3D center_c{0.0, 1.0, 0.0};
    Point3D center_d{0.0, 0.0, 1.0};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::S, center_a, exponents, coefficients);
    Shell shell_b(AngularMomentum::P, center_b, exponents, coefficients);
    Shell shell_c(AngularMomentum::S, center_c, exponents, coefficients);
    Shell shell_d(AngularMomentum::P, center_d, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_b.set_shell_index(1);
    shell_c.set_shell_index(2);
    shell_d.set_shell_index(3);
    shell_a.set_function_index(0);
    shell_b.set_function_index(1);
    shell_c.set_function_index(4);
    shell_d.set_function_index(5);

    compare_with_cpu(shell_a, shell_b, shell_c, shell_d);
}

// ============================================================================
// (pp|pp) Tests
// ============================================================================

TEST_F(EriKernelCudaTest, PpppEri_SameCenter) {
    Point3D center{0.0, 0.0, 0.0};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::P, center, exponents, coefficients);
    Shell shell_b(AngularMomentum::P, center, exponents, coefficients);
    Shell shell_c(AngularMomentum::P, center, exponents, coefficients);
    Shell shell_d(AngularMomentum::P, center, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_b.set_shell_index(1);
    shell_c.set_shell_index(2);
    shell_d.set_shell_index(3);
    shell_a.set_function_index(0);
    shell_b.set_function_index(3);
    shell_c.set_function_index(6);
    shell_d.set_function_index(9);

    compare_with_cpu(shell_a, shell_b, shell_c, shell_d);
}

TEST_F(EriKernelCudaTest, PpppEri_DifferentCenters) {
    Point3D center_a{0.0, 0.0, 0.0};
    Point3D center_b{1.0, 0.0, 0.0};
    Point3D center_c{0.0, 1.0, 0.0};
    Point3D center_d{0.0, 0.0, 1.0};
    std::vector<double> exponents = {0.8};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::P, center_a, exponents, coefficients);
    Shell shell_b(AngularMomentum::P, center_b, exponents, coefficients);
    Shell shell_c(AngularMomentum::P, center_c, exponents, coefficients);
    Shell shell_d(AngularMomentum::P, center_d, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_b.set_shell_index(1);
    shell_c.set_shell_index(2);
    shell_d.set_shell_index(3);
    shell_a.set_function_index(0);
    shell_b.set_function_index(3);
    shell_c.set_function_index(6);
    shell_d.set_function_index(9);

    compare_with_cpu(shell_a, shell_b, shell_c, shell_d);
}

// ============================================================================
// (ss|dd) Tests
// ============================================================================

TEST_F(EriKernelCudaTest, SsddEri) {
    Point3D center_a{0.0, 0.0, 0.0};
    Point3D center_b{1.0, 0.0, 0.0};
    Point3D center_c{0.0, 1.0, 0.0};
    Point3D center_d{0.0, 0.0, 1.0};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::S, center_a, exponents, coefficients);
    Shell shell_b(AngularMomentum::S, center_b, exponents, coefficients);
    Shell shell_c(AngularMomentum::D, center_c, exponents, coefficients);
    Shell shell_d(AngularMomentum::D, center_d, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_b.set_shell_index(1);
    shell_c.set_shell_index(2);
    shell_d.set_shell_index(3);
    shell_a.set_function_index(0);
    shell_b.set_function_index(1);
    shell_c.set_function_index(2);
    shell_d.set_function_index(8);

    compare_with_cpu(shell_a, shell_b, shell_c, shell_d);
}

// ============================================================================
// (dd|dd) Tests
// ============================================================================

TEST_F(EriKernelCudaTest, DdddEri_SameCenter) {
    Point3D center{0.0, 0.0, 0.0};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::D, center, exponents, coefficients);
    Shell shell_b(AngularMomentum::D, center, exponents, coefficients);
    Shell shell_c(AngularMomentum::D, center, exponents, coefficients);
    Shell shell_d(AngularMomentum::D, center, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_b.set_shell_index(1);
    shell_c.set_shell_index(2);
    shell_d.set_shell_index(3);
    shell_a.set_function_index(0);
    shell_b.set_function_index(6);
    shell_c.set_function_index(12);
    shell_d.set_function_index(18);

    compare_with_cpu(shell_a, shell_b, shell_c, shell_d);
}

}  // namespace libaccint

#else  // LIBACCINT_USE_CUDA

TEST(EriKernelCudaTest, CudaNotEnabled) {
    GTEST_SKIP() << "CUDA support not enabled";
}

#endif  // LIBACCINT_USE_CUDA
