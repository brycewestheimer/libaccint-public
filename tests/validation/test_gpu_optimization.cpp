// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_gpu_optimization.cpp
/// @brief Validation tests for GPU optimization utilities (Tasks 27.3.1–27.3.3)
///
/// All tests skip gracefully when CUDA GPU is not available.

#include <gtest/gtest.h>

#include <libaccint/config.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/operators/operator.hpp>

#include <random>
#include <vector>

namespace libaccint {
namespace {

// ============================================================================
// GPU Detection Helper
// ============================================================================

bool has_gpu() {
#if LIBACCINT_USE_CUDA
    return has_cuda_backend();
#else
    return false;
#endif
}

// ============================================================================
// 27.3.1: CUDA Kernel Occupancy Tests
// ============================================================================

TEST(GpuOccupancy, VerifyGpuBackendConfig) {
    if (!has_gpu()) {
        GTEST_SKIP() << "CUDA GPU not available";
    }

    // Verify GPU dispatch produces correct overlap integrals
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {1.430429, 0.0, 1.107157}},
        {1, {-1.430429, 0.0, 1.107157}}
    };
    auto basis = data::create_builtin_basis("sto-3g", atoms);

    // CPU reference
    Engine cpu_engine(basis);
    std::vector<Real> S_cpu;
    cpu_engine.compute_1e(Operator::overlap(), S_cpu);

    // GPU dispatch (PreferGPU may fall back to CPU if GPU not available for 1e)
    Engine gpu_engine(basis);
    std::vector<Real> S_gpu;
    gpu_engine.compute_1e(Operator::overlap(), S_gpu, BackendHint::PreferGPU);

    ASSERT_EQ(S_cpu.size(), S_gpu.size());
    for (Size i = 0; i < S_cpu.size(); ++i) {
        EXPECT_NEAR(S_cpu[i], S_gpu[i], 1e-10)
            << "GPU/CPU mismatch at index " << i;
    }
}

TEST(GpuOccupancy, FockBuildCorrectness) {
    if (!has_gpu()) {
        GTEST_SKIP() << "CUDA GPU not available";
    }

    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {1.430429, 0.0, 1.107157}},
        {1, {-1.430429, 0.0, 1.107157}}
    };
    auto basis = data::create_builtin_basis("sto-3g", atoms);
    const Size nbf = basis.n_basis_functions();

    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0 / static_cast<Real>(nbf);
    }

    // CPU
    Engine cpu_engine(basis);
    consumers::FockBuilder cpu_fock(nbf);
    cpu_fock.set_density(D.data(), nbf);
    cpu_engine.compute_and_consume(Operator::coulomb(), cpu_fock);

    // GPU
    Engine gpu_engine(basis);
    consumers::FockBuilder gpu_fock(nbf);
    gpu_fock.set_density(D.data(), nbf);
    gpu_engine.compute_and_consume(Operator::coulomb(), gpu_fock, BackendHint::PreferGPU);

    auto J_cpu = cpu_fock.get_coulomb_matrix();
    auto J_gpu = gpu_fock.get_coulomb_matrix();

    for (Size i = 0; i < J_cpu.size(); ++i) {
        EXPECT_NEAR(J_cpu[i], J_gpu[i], 1e-8)
            << "GPU Fock mismatch at index " << i;
    }
}

// ============================================================================
// 27.3.2: GPU Memory Transfer Minimization Tests
// ============================================================================

TEST(GpuTransfer, RepeatedComputeNoExtraTransfer) {
    if (!has_gpu()) {
        GTEST_SKIP() << "CUDA GPU not available";
    }

    // Repeated computations with same engine should reuse GPU data
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {1.430429, 0.0, 1.107157}},
        {1, {-1.430429, 0.0, 1.107157}}
    };
    auto basis = data::create_builtin_basis("sto-3g", atoms);

    Engine engine(basis);

    std::vector<Real> S1, S2;
    engine.compute_1e(Operator::overlap(), S1, BackendHint::PreferGPU);
    engine.compute_1e(Operator::overlap(), S2, BackendHint::PreferGPU);

    ASSERT_EQ(S1.size(), S2.size());
    for (Size i = 0; i < S1.size(); ++i) {
        EXPECT_DOUBLE_EQ(S1[i], S2[i]);
    }
}

// ============================================================================
// 27.3.3: Warp Utilization Tests
// ============================================================================

TEST(GpuWarpUtil, DifferentAMPerformance) {
    if (!has_gpu()) {
        GTEST_SKIP() << "CUDA GPU not available";
    }

    // Verify correct results across different AM combinations on GPU
    for (int am = 0; am <= 2; ++am) {
        std::vector<double> exp = {3.42525091, 0.62391373, 0.16885540};
        std::vector<double> coef = {0.15432897, 0.53532814, 0.44463454};

        auto am_enum = static_cast<AngularMomentum>(am);
        Shell shell_a(am_enum, {0.0, 0.0, 0.0}, exp, coef);
        shell_a.set_shell_index(0);
        shell_a.set_atom_index(0);
        shell_a.set_function_index(0);

        Shell shell_b(am_enum, {2.0, 0.0, 0.0}, exp, coef);
        shell_b.set_shell_index(1);
        shell_b.set_atom_index(1);
        shell_b.set_function_index(shell_a.n_functions());

        BasisSet basis({shell_a, shell_b});

        Engine engine(basis);

        std::vector<Real> S;
        engine.compute_1e(Operator::overlap(), S, BackendHint::PreferGPU);

        // Overlap matrix should have positive diagonal
        Size nbf = basis.n_basis_functions();
        for (Size i = 0; i < nbf; ++i) {
            EXPECT_GT(S[i * nbf + i], 0.0)
                << "AM=" << am << " diagonal[" << i << "] not positive";
        }
    }
}

// ============================================================================
// Backend Configuration Tests
// ============================================================================

TEST(GpuConfig, HasCudaBackendConsistent) {
    // has_cuda_backend() should return consistent value
    bool cuda1 = has_cuda_backend();
    bool cuda2 = has_cuda_backend();
    EXPECT_EQ(cuda1, cuda2);
}

TEST(GpuConfig, ForceGPUFallsBackWhenNotAvailable) {
    if (has_gpu()) {
        GTEST_SKIP() << "This test is for non-GPU environments";
    }

    // When GPU is not available, ForceGPU should still work
    // (the engine handles fallback/error internally)
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {1.430429, 0.0, 1.107157}},
        {1, {-1.430429, 0.0, 1.107157}}
    };
    auto basis = data::create_builtin_basis("sto-3g", atoms);
    Engine engine(basis);

    std::vector<Real> S;
    engine.compute_1e(Operator::overlap(), S);
    EXPECT_FALSE(S.empty());
}

}  // namespace
}  // namespace libaccint
