// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_fock_builder_gpu.cpp
/// @brief Unit tests for GPU FockBuilder

#include <gtest/gtest.h>
#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/consumers/fock_builder_gpu.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/engine/cuda_engine.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/operators/operator.hpp>

#include <vector>
#include <cmath>
#include <random>

// Suppress deprecation warnings for legacy single-shell CudaEngine API tests.
// These tests intentionally validate the deprecated code paths.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace libaccint::consumers {

// ============================================================================
// Test Fixture
// ============================================================================

class GpuFockBuilderTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }

    /// Create a simple test basis set (H2-like)
    std::vector<Shell> create_h2_sto3g_basis() {
        // STO-3G hydrogen basis (1s)
        std::vector<double> h_exp = {3.42525091, 0.62391373, 0.16885540};
        std::vector<double> h_coef = {0.15432897, 0.53532814, 0.44463454};

        // H2 geometry (1.4 bohr separation along z)
        Point3D H1{0.0, 0.0, 0.0};
        Point3D H2{0.0, 0.0, 1.4};

        std::vector<Shell> shells;

        Shell h1_1s(AngularMomentum::S, H1, h_exp, h_coef);
        h1_1s.set_shell_index(0);
        h1_1s.set_atom_index(0);
        h1_1s.set_function_index(0);
        shells.push_back(h1_1s);

        Shell h2_1s(AngularMomentum::S, H2, h_exp, h_coef);
        h2_1s.set_shell_index(1);
        h2_1s.set_atom_index(1);
        h2_1s.set_function_index(1);
        shells.push_back(h2_1s);

        return shells;
    }

    /// Create H2O STO-3G basis
    std::vector<Shell> create_h2o_sto3g_basis() {
        // STO-3G hydrogen basis (1s)
        std::vector<double> h_exp = {3.42525091, 0.62391373, 0.16885540};
        std::vector<double> h_coef = {0.15432897, 0.53532814, 0.44463454};

        // STO-3G oxygen basis
        std::vector<double> o_s_exp = {130.70932, 23.808861, 6.4436083};
        std::vector<double> o_s_coef = {0.15432897, 0.53532814, 0.44463454};

        std::vector<double> o_sp_exp = {5.0331513, 1.1695961, 0.38038896};
        std::vector<double> o_2s_coef = {-0.09996723, 0.39951283, 0.70011547};
        std::vector<double> o_2p_coef = {0.15591627, 0.60768372, 0.39195739};

        // H2O geometry (bohr)
        Point3D O{0.0, 0.0, 0.0};
        Point3D H1{1.430429, 0.0, 1.107157};
        Point3D H2{-1.430429, 0.0, 1.107157};

        std::vector<Shell> shells;

        // Oxygen 1s
        Shell o_1s(AngularMomentum::S, O, o_s_exp, o_s_coef);
        o_1s.set_shell_index(0);
        o_1s.set_atom_index(0);
        o_1s.set_function_index(0);
        shells.push_back(o_1s);

        // Oxygen 2s
        Shell o_2s(AngularMomentum::S, O, o_sp_exp, o_2s_coef);
        o_2s.set_shell_index(1);
        o_2s.set_atom_index(0);
        o_2s.set_function_index(1);
        shells.push_back(o_2s);

        // Oxygen 2p
        Shell o_2p(AngularMomentum::P, O, o_sp_exp, o_2p_coef);
        o_2p.set_shell_index(2);
        o_2p.set_atom_index(0);
        o_2p.set_function_index(2);
        shells.push_back(o_2p);

        // Hydrogen 1 - 1s
        Shell h1_1s(AngularMomentum::S, H1, h_exp, h_coef);
        h1_1s.set_shell_index(3);
        h1_1s.set_atom_index(1);
        h1_1s.set_function_index(5);
        shells.push_back(h1_1s);

        // Hydrogen 2 - 1s
        Shell h2_1s(AngularMomentum::S, H2, h_exp, h_coef);
        h2_1s.set_shell_index(4);
        h2_1s.set_atom_index(2);
        h2_1s.set_function_index(6);
        shells.push_back(h2_1s);

        return shells;
    }

    /// Create a random symmetric density matrix
    std::vector<Real> create_random_density(Size nbf, unsigned seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        std::vector<Real> D(nbf * nbf);
        for (Size i = 0; i < nbf; ++i) {
            for (Size j = i; j < nbf; ++j) {
                double val = dist(gen);
                D[i * nbf + j] = val;
                D[j * nbf + i] = val;
            }
        }
        return D;
    }

    /// Compare matrices with tolerance
    void compare_matrices(const std::vector<Real>& gpu,
                          const std::vector<Real>& cpu,
                          double tolerance,
                          const std::string& name) {
        ASSERT_EQ(gpu.size(), cpu.size()) << name << " size mismatch";
        double max_err = 0.0;
        for (size_t i = 0; i < gpu.size(); ++i) {
            double err = std::abs(gpu[i] - cpu[i]);
            max_err = std::max(max_err, err);
            EXPECT_NEAR(gpu[i], cpu[i], tolerance)
                << name << " mismatch at index " << i;
        }
        std::cout << name << " max error: " << max_err << std::endl;
    }
};

// ============================================================================
// Basic Construction Tests
// ============================================================================

TEST_F(GpuFockBuilderTest, ConstructAndDestroy) {
    Size nbf = 10;
    GpuFockBuilder fock(nbf);
    EXPECT_EQ(fock.nbf(), nbf);
}

TEST_F(GpuFockBuilderTest, MoveConstruction) {
    Size nbf = 10;
    GpuFockBuilder fock1(nbf);
    EXPECT_EQ(fock1.nbf(), nbf);

    GpuFockBuilder fock2(std::move(fock1));
    EXPECT_EQ(fock2.nbf(), nbf);
}

TEST_F(GpuFockBuilderTest, SetDensity) {
    Size nbf = 4;
    GpuFockBuilder fock(nbf);

    std::vector<Real> D = create_random_density(nbf);
    EXPECT_NO_THROW(fock.set_density(D.data(), nbf));
}

// ============================================================================
// Comparison with CPU FockBuilder
// ============================================================================

TEST_F(GpuFockBuilderTest, CompareWithCPU_H2) {
    auto shells = create_h2_sto3g_basis();
    BasisSet basis(shells);
    const Size nbf = basis.n_basis_functions();

    // Create random density matrix
    auto D = create_random_density(nbf);

    // GPU computation
    CudaEngine cuda_engine(basis);
    GpuFockBuilder gpu_fock(nbf, cuda_engine.stream());
    gpu_fock.set_density(D.data(), nbf);
    cuda_engine.compute_and_consume_eri(gpu_fock);
    gpu_fock.synchronize();

    auto J_gpu = gpu_fock.get_coulomb_matrix();
    auto K_gpu = gpu_fock.get_exchange_matrix();

    // CPU computation
    Engine cpu_engine(basis);
    FockBuilder cpu_fock(nbf);
    cpu_fock.set_density(D.data(), nbf);
    cpu_engine.compute_and_consume(Operator::coulomb(), cpu_fock);

    auto J_cpu = std::vector<Real>(cpu_fock.get_coulomb_matrix().begin(),
                                    cpu_fock.get_coulomb_matrix().end());
    auto K_cpu = std::vector<Real>(cpu_fock.get_exchange_matrix().begin(),
                                    cpu_fock.get_exchange_matrix().end());

    // Compare with relaxed tolerance due to atomic accumulation order
    compare_matrices(J_gpu, J_cpu, 1e-8, "Coulomb J");
    compare_matrices(K_gpu, K_cpu, 1e-8, "Exchange K");
}

TEST_F(GpuFockBuilderTest, CompareWithCPU_H2O) {
    auto shells = create_h2o_sto3g_basis();
    BasisSet basis(shells);
    const Size nbf = basis.n_basis_functions();

    // Create random density matrix
    auto D = create_random_density(nbf);

    // GPU computation
    CudaEngine cuda_engine(basis);
    GpuFockBuilder gpu_fock(nbf, cuda_engine.stream());
    gpu_fock.set_density(D.data(), nbf);
    cuda_engine.compute_and_consume_eri(gpu_fock);
    gpu_fock.synchronize();

    auto J_gpu = gpu_fock.get_coulomb_matrix();
    auto K_gpu = gpu_fock.get_exchange_matrix();

    // CPU computation
    Engine cpu_engine(basis);
    FockBuilder cpu_fock(nbf);
    cpu_fock.set_density(D.data(), nbf);
    cpu_engine.compute_and_consume(Operator::coulomb(), cpu_fock);

    auto J_cpu = std::vector<Real>(cpu_fock.get_coulomb_matrix().begin(),
                                    cpu_fock.get_coulomb_matrix().end());
    auto K_cpu = std::vector<Real>(cpu_fock.get_exchange_matrix().begin(),
                                    cpu_fock.get_exchange_matrix().end());

    compare_matrices(J_gpu, J_cpu, 1e-10, "Coulomb J");
    compare_matrices(K_gpu, K_cpu, 1e-10, "Exchange K");
}

// ============================================================================
// Fock Matrix Tests
// ============================================================================

TEST_F(GpuFockBuilderTest, GetFockMatrix) {
    Size nbf = 4;
    GpuFockBuilder fock(nbf);

    // Create test H_core
    std::vector<Real> H_core(nbf * nbf, 0.1);
    for (Size i = 0; i < nbf; ++i) {
        H_core[i * nbf + i] = -1.0;
    }

    // Create density and set it
    auto D = create_random_density(nbf);
    fock.set_density(D.data(), nbf);

    // Get Fock matrix (even without ERIs, should return H_core)
    auto F = fock.get_fock_matrix(H_core, 1.0);
    EXPECT_EQ(F.size(), nbf * nbf);
}

TEST_F(GpuFockBuilderTest, ResetClearsMatrices) {
    Size nbf = 4;
    GpuFockBuilder fock(nbf);

    // After reset, J and K should be zero
    fock.reset();
    fock.synchronize();

    auto J = fock.get_coulomb_matrix();
    auto K = fock.get_exchange_matrix();

    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(J[i], 0.0, 1e-15);
        EXPECT_NEAR(K[i], 0.0, 1e-15);
    }
}

}  // namespace libaccint::consumers

#else  // LIBACCINT_USE_CUDA

TEST(GpuFockBuilderTest, CudaNotEnabled) {
    GTEST_SKIP() << "CUDA support not enabled";
}

#endif  // LIBACCINT_USE_CUDA

#pragma GCC diagnostic pop
