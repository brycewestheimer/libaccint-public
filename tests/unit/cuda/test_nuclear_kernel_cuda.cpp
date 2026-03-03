// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_nuclear_kernel_cuda.cpp
/// @brief Unit tests for CUDA nuclear attraction integral kernel

#include <gtest/gtest.h>
#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/kernels/nuclear_kernel_cuda.hpp>
#include <libaccint/kernels/nuclear_kernel.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/device_data.hpp>
#include <libaccint/operators/device_operator_data.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
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
using kernels::cuda::dispatch_nuclear_kernel;
using kernels::cuda::nuclear_output_size;

// ============================================================================
// Test Fixture
// ============================================================================

class NuclearKernelCudaTest : public ::testing::Test {
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

    /// Compare GPU results with CPU reference implementation.
    /// The device Rys quadrature uses the Chebyshev+QL algorithm with Boys
    /// function moments. Boys function tables are uploaded to device memory
    /// in SetUp(), providing accurate moments. The Chebyshev+QL solver
    /// achieves near machine precision for n_roots <= 5, so tight absolute
    /// tolerances (1e-10) are appropriate for all supported angular momenta.
    void compare_with_cpu(const Shell& shell_a, const Shell& shell_b,
                          const PointChargeParams& charges,
                          double tolerance = 1e-10,
                          bool use_relative = false) {
        // Create ShellSets
        ShellSet bra_set(shell_a.angular_momentum(), shell_a.n_primitives());
        ShellSet ket_set(shell_b.angular_momentum(), shell_b.n_primitives());
        bra_set.add_shell(shell_a);
        ket_set.add_shell(shell_b);

        // Upload shells to device
        basis::ShellSetDeviceData bra_data = basis::upload_shell_set(bra_set);
        basis::ShellSetDeviceData ket_data = basis::upload_shell_set(ket_set);
        DeviceMemoryManager::synchronize();

        // Create pair
        basis::ShellSetPairDeviceData pair;
        pair.bra = bra_data;
        pair.ket = ket_data;

        // Upload point charges to device
        operators::DevicePointChargeData charge_data =
            operators::upload_point_charges(charges.x, charges.y, charges.z, charges.charge);
        DeviceMemoryManager::synchronize();

        // Allocate output buffer
        size_t output_size = nuclear_output_size(pair);
        DeviceBuffer<double> d_output(output_size);

        // Launch GPU kernel
        dispatch_nuclear_kernel(pair, charge_data, d_boys_coeffs_, d_output.data());
        DeviceMemoryManager::synchronize();

        // Download results
        std::vector<double> gpu_results(output_size);
        d_output.download(gpu_results.data(), output_size);
        DeviceMemoryManager::synchronize();

        // Compute CPU reference
        NuclearBuffer cpu_buffer;
        kernels::compute_nuclear(shell_a, shell_b, charges, cpu_buffer);

        // Compare results
        const int na = shell_a.n_functions();
        const int nb = shell_b.n_functions();
        for (int a = 0; a < na; ++a) {
            for (int b = 0; b < nb; ++b) {
                const double gpu_val = gpu_results[a * nb + b];
                const double cpu_val = cpu_buffer(a, b);
                if (use_relative && std::abs(cpu_val) > 1e-10) {
                    const double rel_err = std::abs((gpu_val - cpu_val) / cpu_val);
                    EXPECT_LT(rel_err, tolerance)
                        << "Relative error at (" << a << ", " << b << "): "
                        << "GPU=" << gpu_val << ", CPU=" << cpu_val
                        << ", rel_err=" << rel_err;
                } else {
                    EXPECT_NEAR(gpu_val, cpu_val, tolerance)
                        << "Mismatch at (" << a << ", " << b << "): "
                        << "GPU=" << gpu_val << ", CPU=" << cpu_val;
                }
            }
        }

        // Clean up
        basis::free_shell_set_device_data(bra_data);
        basis::free_shell_set_device_data(ket_data);
        operators::free_point_charge_device_data(charge_data);
    }
};

// ============================================================================
// (s|s) Tests
// ============================================================================

TEST_F(NuclearKernelCudaTest, SsNuclear_SameCenter) {
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

    // Single nucleus at the shell center
    PointChargeParams charges;
    charges.x = {0.0};
    charges.y = {0.0};
    charges.z = {0.0};
    charges.charge = {1.0};  // Unit charge

    compare_with_cpu(shell_a, shell_b, charges);
}

TEST_F(NuclearKernelCudaTest, SsNuclear_DifferentCenters) {
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

    // Single nucleus between the shells
    PointChargeParams charges;
    charges.x = {0.75};
    charges.y = {0.0};
    charges.z = {0.0};
    charges.charge = {8.0};  // Oxygen-like

    compare_with_cpu(shell_a, shell_b, charges);
}

TEST_F(NuclearKernelCudaTest, SsNuclear_MultipleNuclei) {
    Point3D center_a{0.0, 0.0, 0.0};
    Point3D center_b{1.0, 0.0, 0.0};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::S, center_a, exponents, coefficients);
    Shell shell_b(AngularMomentum::S, center_b, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_a.set_atom_index(0);
    shell_a.set_function_index(0);
    shell_b.set_shell_index(1);
    shell_b.set_atom_index(1);
    shell_b.set_function_index(1);

    // Water-like: O at center, two H atoms
    PointChargeParams charges;
    charges.x = {0.0, 0.757, -0.757};
    charges.y = {0.0, 0.587, 0.587};
    charges.z = {0.0, 0.0, 0.0};
    charges.charge = {8.0, 1.0, 1.0};

    compare_with_cpu(shell_a, shell_b, charges);
}

// ============================================================================
// (s|p) and (p|s) Tests
// ============================================================================

TEST_F(NuclearKernelCudaTest, SpNuclear) {
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

    PointChargeParams charges;
    charges.x = {0.5};
    charges.y = {0.5};
    charges.z = {0.0};
    charges.charge = {6.0};  // Carbon-like

    compare_with_cpu(shell_a, shell_b, charges);
}

TEST_F(NuclearKernelCudaTest, PsNuclear) {
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

    PointChargeParams charges;
    charges.x = {0.0, 1.0};
    charges.y = {0.0, 0.0};
    charges.z = {0.0, 0.0};
    charges.charge = {7.0, 1.0};  // N + H

    compare_with_cpu(shell_a, shell_b, charges);
}

// ============================================================================
// (p|p) Tests
// ============================================================================

TEST_F(NuclearKernelCudaTest, PpNuclear_SameCenter) {
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

    PointChargeParams charges;
    charges.x = {0.0};
    charges.y = {0.0};
    charges.z = {0.0};
    charges.charge = {6.0};

    compare_with_cpu(shell_a, shell_b, charges);
}

TEST_F(NuclearKernelCudaTest, PpNuclear_DifferentCenters) {
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

    PointChargeParams charges;
    charges.x = {0.75, 0.0, 1.5};
    charges.y = {0.5, 0.0, 1.0};
    charges.z = {0.25, 0.0, 0.5};
    charges.charge = {8.0, 1.0, 1.0};

    // 2-root Rys quadrature (n_roots = (1+1)/2 + 1 = 2).
    // The device Chebyshev+QL Rys solver achieves near machine precision for
    // n_roots <= 5 when Boys moments are accurate (table-based initialization
    // is done in the fixture SetUp). Use tight absolute tolerance.
    compare_with_cpu(shell_a, shell_b, charges, 1e-10);
}

// ============================================================================
// (s|d) and (d|s) Tests
// ============================================================================

TEST_F(NuclearKernelCudaTest, SdNuclear) {
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

    PointChargeParams charges;
    charges.x = {0.5};
    charges.y = {0.0};
    charges.z = {0.0};
    charges.charge = {8.0};

    compare_with_cpu(shell_a, shell_b, charges);
}

TEST_F(NuclearKernelCudaTest, DsNuclear) {
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

    PointChargeParams charges;
    charges.x = {0.0};
    charges.y = {0.25};
    charges.z = {0.25};
    charges.charge = {6.0};

    compare_with_cpu(shell_a, shell_b, charges);
}

// ============================================================================
// (d|d) Tests
// ============================================================================

TEST_F(NuclearKernelCudaTest, DdNuclear_SameCenter) {
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

    PointChargeParams charges;
    charges.x = {0.0};
    charges.y = {0.0};
    charges.z = {0.0};
    charges.charge = {8.0};

    // 3-root Rys quadrature (n_roots = (2+2)/2 + 1 = 3).
    // The device Chebyshev+QL Rys solver achieves near machine precision for
    // n_roots <= 5 when Boys moments are accurate (table-based initialization
    // is done in the fixture SetUp). Use tight absolute tolerance.
    compare_with_cpu(shell_a, shell_b, charges, 1e-10);
}

TEST_F(NuclearKernelCudaTest, DdNuclear_DifferentCenters) {
    Point3D center_a{0.0, 0.0, 0.0};
    Point3D center_b{1.0, 1.0, 1.0};
    std::vector<double> exp_a = {2.0, 0.5};
    std::vector<double> coef_a = {0.4, 0.6};

    Shell shell_a(AngularMomentum::D, center_a, exp_a, coef_a);
    Shell shell_b(AngularMomentum::D, center_b, exp_a, coef_a);
    shell_a.set_shell_index(0);
    shell_a.set_atom_index(0);
    shell_a.set_function_index(0);
    shell_b.set_shell_index(1);
    shell_b.set_atom_index(1);
    shell_b.set_function_index(6);

    // Multiple nuclei
    PointChargeParams charges;
    charges.x = {0.5, 0.0, 1.0};
    charges.y = {0.5, 0.0, 1.0};
    charges.z = {0.5, 0.0, 1.0};
    charges.charge = {8.0, 1.0, 1.0};

    // 3-root Rys quadrature (n_roots = (2+2)/2 + 1 = 3).
    // Tight absolute tolerance: Boys function tables are uploaded to device
    // in the fixture SetUp, and the Chebyshev+QL Rys solver achieves near
    // machine precision for n_roots <= 5.
    compare_with_cpu(shell_a, shell_b, charges, 1e-10);
}

}  // namespace libaccint

#else  // LIBACCINT_USE_CUDA

TEST(NuclearKernelCudaTest, CudaNotEnabled) {
    GTEST_SKIP() << "CUDA support not enabled";
}

#endif  // LIBACCINT_USE_CUDA
