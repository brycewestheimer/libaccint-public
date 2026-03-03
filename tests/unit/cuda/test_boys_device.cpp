// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_boys_device.cpp
/// @brief Unit tests for CUDA device-side Boys function

#include <gtest/gtest.h>
#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/memory/device_memory.hpp>
#include <libaccint/math/boys_function.hpp>

#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Forward declaration of device test function
namespace libaccint::device::math {
    double* boys_device_init(cudaStream_t stream = nullptr);
    void boys_device_cleanup();
    double* boys_device_get_coeffs();
    bool boys_device_is_initialized();
    void boys_device_test(int n_max, const double* d_T_values, int n_T_values,
                          double* d_results, cudaStream_t stream);
}

namespace libaccint {

using device::math::boys_device_init;
using device::math::boys_device_cleanup;
using device::math::boys_device_get_coeffs;
using device::math::boys_device_is_initialized;
using device::math::boys_device_test;

namespace {  // anonymous namespace for tests

// ============================================================================
// Test Fixture
// ============================================================================

class BoysDeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        // Initialize device Boys tables
        boys_device_init();
    }

    void TearDown() override {
        boys_device_cleanup();
    }
};

// ============================================================================
// Initialization Tests
// ============================================================================

TEST_F(BoysDeviceTest, InitializationSucceeds) {
    EXPECT_TRUE(boys_device_is_initialized());
    EXPECT_NE(boys_device_get_coeffs(), nullptr);
}

TEST_F(BoysDeviceTest, CleanupAndReinitialize) {
    EXPECT_TRUE(boys_device_is_initialized());

    boys_device_cleanup();
    EXPECT_FALSE(boys_device_is_initialized());

    boys_device_init();
    EXPECT_TRUE(boys_device_is_initialized());
}

// ============================================================================
// Basic Value Tests (compare GPU vs CPU)
// ============================================================================

TEST_F(BoysDeviceTest, SingleValueT0) {
    // Test T = 0: F_n(0) = 1/(2n+1)
    std::vector<double> T_values = {0.0};

    memory::DeviceBuffer<double> d_T(1);
    d_T.upload(T_values.data(), 1);

    constexpr int n_max = 10;
    memory::DeviceBuffer<double> d_results(n_max + 1);

    memory::DeviceMemoryManager::synchronize();
    boys_device_test(n_max, d_T.data(), 1, d_results.data(), nullptr);
    memory::DeviceMemoryManager::synchronize();

    std::vector<double> gpu_results(n_max + 1);
    d_results.download(gpu_results.data(), n_max + 1);
    memory::DeviceMemoryManager::synchronize();

    for (int n = 0; n <= n_max; ++n) {
        double expected = 1.0 / (2 * n + 1);
        EXPECT_DOUBLE_EQ(gpu_results[n], expected)
            << "Mismatch at n=" << n << " for T=0";
    }
}

TEST_F(BoysDeviceTest, SingleValueSmallT) {
    // Test very small T (Taylor regime)
    std::vector<double> T_values = {1e-16};

    memory::DeviceBuffer<double> d_T(1);
    d_T.upload(T_values.data(), 1);

    constexpr int n_max = 5;
    memory::DeviceBuffer<double> d_results(n_max + 1);

    memory::DeviceMemoryManager::synchronize();
    boys_device_test(n_max, d_T.data(), 1, d_results.data(), nullptr);
    memory::DeviceMemoryManager::synchronize();

    std::vector<double> gpu_results(n_max + 1);
    d_results.download(gpu_results.data(), n_max + 1);
    memory::DeviceMemoryManager::synchronize();

    // Compare with CPU
    std::vector<double> cpu_results(n_max + 1);
    math::boys_evaluate_array(n_max, T_values[0], cpu_results.data());

    for (int n = 0; n <= n_max; ++n) {
        EXPECT_NEAR(gpu_results[n], cpu_results[n], 1e-14)
            << "Mismatch at n=" << n << " for T=" << T_values[0];
    }
}

TEST_F(BoysDeviceTest, SingleValueMediumT) {
    // Test medium T (Chebyshev regime)
    std::vector<double> T_values = {5.0};

    memory::DeviceBuffer<double> d_T(1);
    d_T.upload(T_values.data(), 1);

    constexpr int n_max = 15;
    memory::DeviceBuffer<double> d_results(n_max + 1);

    memory::DeviceMemoryManager::synchronize();
    boys_device_test(n_max, d_T.data(), 1, d_results.data(), nullptr);
    memory::DeviceMemoryManager::synchronize();

    std::vector<double> gpu_results(n_max + 1);
    d_results.download(gpu_results.data(), n_max + 1);
    memory::DeviceMemoryManager::synchronize();

    // Compare with CPU
    std::vector<double> cpu_results(n_max + 1);
    math::boys_evaluate_array(n_max, T_values[0], cpu_results.data());

    for (int n = 0; n <= n_max; ++n) {
        double rel_error = std::abs(gpu_results[n] - cpu_results[n]) / std::abs(cpu_results[n]);
        EXPECT_LT(rel_error, 1e-13)
            << "Relative error " << rel_error << " at n=" << n << " for T=" << T_values[0]
            << "\n  GPU: " << gpu_results[n] << "\n  CPU: " << cpu_results[n];
    }
}

TEST_F(BoysDeviceTest, SingleValueLargeT) {
    // Test large T (asymptotic regime)
    std::vector<double> T_values = {50.0};

    memory::DeviceBuffer<double> d_T(1);
    d_T.upload(T_values.data(), 1);

    constexpr int n_max = 10;
    memory::DeviceBuffer<double> d_results(n_max + 1);

    memory::DeviceMemoryManager::synchronize();
    boys_device_test(n_max, d_T.data(), 1, d_results.data(), nullptr);
    memory::DeviceMemoryManager::synchronize();

    std::vector<double> gpu_results(n_max + 1);
    d_results.download(gpu_results.data(), n_max + 1);
    memory::DeviceMemoryManager::synchronize();

    // Compare with CPU
    std::vector<double> cpu_results(n_max + 1);
    math::boys_evaluate_array(n_max, T_values[0], cpu_results.data());

    for (int n = 0; n <= n_max; ++n) {
        double rel_error = std::abs(gpu_results[n] - cpu_results[n]) / std::abs(cpu_results[n]);
        EXPECT_LT(rel_error, 1e-13)
            << "Relative error " << rel_error << " at n=" << n << " for T=" << T_values[0]
            << "\n  GPU: " << gpu_results[n] << "\n  CPU: " << cpu_results[n];
    }
}

// ============================================================================
// Batch Tests
// ============================================================================

TEST_F(BoysDeviceTest, BatchEvaluationChebyshevRegime) {
    // Test multiple T values in Chebyshev regime
    constexpr int n_T = 100;
    std::vector<double> T_values(n_T);
    for (int i = 0; i < n_T; ++i) {
        T_values[i] = 0.1 + i * 0.35;  // 0.1 to 34.75
    }

    memory::DeviceBuffer<double> d_T(n_T);
    d_T.upload(T_values.data(), n_T);

    constexpr int n_max = 20;
    const int result_size = n_T * (n_max + 1);
    memory::DeviceBuffer<double> d_results(result_size);

    memory::DeviceMemoryManager::synchronize();
    boys_device_test(n_max, d_T.data(), n_T, d_results.data(), nullptr);
    memory::DeviceMemoryManager::synchronize();

    std::vector<double> gpu_results(result_size);
    d_results.download(gpu_results.data(), result_size);
    memory::DeviceMemoryManager::synchronize();

    // Compare with CPU
    std::vector<double> cpu_results(n_max + 1);
    int max_rel_error_n = -1;
    int max_rel_error_i = -1;
    double max_rel_error = 0.0;

    for (int i = 0; i < n_T; ++i) {
        math::boys_evaluate_array(n_max, T_values[i], cpu_results.data());

        for (int n = 0; n <= n_max; ++n) {
            double gpu_val = gpu_results[i * (n_max + 1) + n];
            double cpu_val = cpu_results[n];
            double rel_error = std::abs(gpu_val - cpu_val) / std::abs(cpu_val);

            if (rel_error > max_rel_error) {
                max_rel_error = rel_error;
                max_rel_error_n = n;
                max_rel_error_i = i;
            }

            EXPECT_LT(rel_error, 1e-13)
                << "Relative error " << rel_error << " at n=" << n << ", T=" << T_values[i]
                << "\n  GPU: " << gpu_val << "\n  CPU: " << cpu_val;
        }
    }

    // Report worst case
    if (max_rel_error > 0) {
        EXPECT_LT(max_rel_error, 1e-13)
            << "Maximum relative error " << max_rel_error
            << " at n=" << max_rel_error_n << ", T=" << T_values[max_rel_error_i];
    }
}

TEST_F(BoysDeviceTest, BatchEvaluationAllRegimes) {
    // Test T values spanning all regimes
    std::vector<double> T_values = {
        0.0,           // T = 0
        1e-16, 1e-15,  // Taylor regime
        0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 35.0, 35.9,  // Chebyshev regime
        36.0, 40.0, 50.0, 100.0, 200.0  // Asymptotic regime
    };
    const int n_T = static_cast<int>(T_values.size());

    memory::DeviceBuffer<double> d_T(n_T);
    d_T.upload(T_values.data(), n_T);

    constexpr int n_max = 15;
    const int result_size = n_T * (n_max + 1);
    memory::DeviceBuffer<double> d_results(result_size);

    memory::DeviceMemoryManager::synchronize();
    boys_device_test(n_max, d_T.data(), n_T, d_results.data(), nullptr);
    memory::DeviceMemoryManager::synchronize();

    std::vector<double> gpu_results(result_size);
    d_results.download(gpu_results.data(), result_size);
    memory::DeviceMemoryManager::synchronize();

    // Compare with CPU
    std::vector<double> cpu_results(n_max + 1);

    for (int i = 0; i < n_T; ++i) {
        math::boys_evaluate_array(n_max, T_values[i], cpu_results.data());

        for (int n = 0; n <= n_max; ++n) {
            double gpu_val = gpu_results[i * (n_max + 1) + n];
            double cpu_val = cpu_results[n];

            // For T = 0, use exact comparison
            if (T_values[i] == 0.0) {
                EXPECT_DOUBLE_EQ(gpu_val, cpu_val)
                    << "Mismatch at n=" << n << " for T=0";
            } else {
                double rel_error = std::abs(gpu_val - cpu_val) / std::abs(cpu_val);
                EXPECT_LT(rel_error, 1e-13)
                    << "Relative error " << rel_error << " at n=" << n << ", T=" << T_values[i]
                    << "\n  GPU: " << gpu_val << "\n  CPU: " << cpu_val;
            }
        }
    }
}

// ============================================================================
// High Order Tests
// ============================================================================

TEST_F(BoysDeviceTest, HighOrderN) {
    // Test high n values (up to n = 30)
    std::vector<double> T_values = {1.0, 10.0, 30.0};
    const int n_T = static_cast<int>(T_values.size());

    memory::DeviceBuffer<double> d_T(n_T);
    d_T.upload(T_values.data(), n_T);

    constexpr int n_max = 30;
    const int result_size = n_T * (n_max + 1);
    memory::DeviceBuffer<double> d_results(result_size);

    memory::DeviceMemoryManager::synchronize();
    boys_device_test(n_max, d_T.data(), n_T, d_results.data(), nullptr);
    memory::DeviceMemoryManager::synchronize();

    std::vector<double> gpu_results(result_size);
    d_results.download(gpu_results.data(), result_size);
    memory::DeviceMemoryManager::synchronize();

    // Compare with CPU
    std::vector<double> cpu_results(n_max + 1);

    for (int i = 0; i < n_T; ++i) {
        math::boys_evaluate_array(n_max, T_values[i], cpu_results.data());

        for (int n = 0; n <= n_max; ++n) {
            double gpu_val = gpu_results[i * (n_max + 1) + n];
            double cpu_val = cpu_results[n];
            double rel_error = std::abs(gpu_val - cpu_val) / std::abs(cpu_val);

            EXPECT_LT(rel_error, 1e-13)
                << "Relative error " << rel_error << " at n=" << n << ", T=" << T_values[i]
                << "\n  GPU: " << gpu_val << "\n  CPU: " << cpu_val;
        }
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(BoysDeviceTest, LargeBatch) {
    // Test large batch for performance/correctness
    constexpr int n_T = 10000;
    std::vector<double> T_values(n_T);

    // Random-ish T values spanning all regimes
    for (int i = 0; i < n_T; ++i) {
        T_values[i] = (i * 0.01) + ((i * 17) % 100) * 0.5;
        if (T_values[i] > 200.0) T_values[i] = 200.0;
    }

    memory::DeviceBuffer<double> d_T(n_T);
    d_T.upload(T_values.data(), n_T);

    constexpr int n_max = 10;
    const int result_size = n_T * (n_max + 1);
    memory::DeviceBuffer<double> d_results(result_size);

    memory::DeviceMemoryManager::synchronize();
    boys_device_test(n_max, d_T.data(), n_T, d_results.data(), nullptr);
    memory::DeviceMemoryManager::synchronize();

    std::vector<double> gpu_results(result_size);
    d_results.download(gpu_results.data(), result_size);
    memory::DeviceMemoryManager::synchronize();

    // Spot-check a few values
    std::vector<double> cpu_results(n_max + 1);
    int check_indices[] = {0, 100, 500, 1000, 5000, 9999};

    for (int idx : check_indices) {
        math::boys_evaluate_array(n_max, T_values[idx], cpu_results.data());

        for (int n = 0; n <= n_max; ++n) {
            double gpu_val = gpu_results[idx * (n_max + 1) + n];
            double cpu_val = cpu_results[n];

            if (cpu_val != 0.0) {
                double rel_error = std::abs(gpu_val - cpu_val) / std::abs(cpu_val);
                EXPECT_LT(rel_error, 1e-13)
                    << "Relative error " << rel_error << " at idx=" << idx
                    << ", n=" << n << ", T=" << T_values[idx];
            }
        }
    }
}

}  // anonymous namespace
}  // namespace libaccint

#else  // LIBACCINT_USE_CUDA

TEST(BoysDeviceTest, CudaNotEnabled) {
    GTEST_SKIP() << "CUDA support not enabled";
}

#endif  // LIBACCINT_USE_CUDA
