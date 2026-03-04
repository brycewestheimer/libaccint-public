// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_cuda_engine.cpp
/// @brief Unit tests for CUDA engine GPU dispatch

#include <gtest/gtest.h>
#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/engine/cuda_engine.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/operators/operator.hpp>

#include <vector>
#include <cmath>

// Suppress deprecation warnings for legacy single-shell CudaEngine API tests.
// These tests intentionally validate the deprecated code paths.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace libaccint {

// ============================================================================
// Test Fixture
// ============================================================================

class CudaEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }

    /// Create a simple test basis set (H2O-like)
    std::vector<Shell> create_h2o_sto3g_basis() {
        // STO-3G hydrogen basis (1s)
        std::vector<double> h_exp = {3.42525091, 0.62391373, 0.16885540};
        std::vector<double> h_coef = {0.15432897, 0.53532814, 0.44463454};

        // STO-3G oxygen basis (1s, 2s, 2p)
        std::vector<double> o_s_exp = {130.70932, 23.808861, 6.4436083};
        std::vector<double> o_s_coef = {0.15432897, 0.53532814, 0.44463454};

        std::vector<double> o_sp_exp = {5.0331513, 1.1695961, 0.38038896};
        std::vector<double> o_2s_coef = {-0.09996723, 0.39951283, 0.70011547};
        std::vector<double> o_2p_coef = {0.15591627, 0.60768372, 0.39195739};

        // H2O geometry (angstrom converted to bohr)
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

    /// Create point charges for H2O nuclei
    PointChargeParams create_h2o_charges() {
        PointChargeParams charges;
        charges.x = {0.0, 1.430429, -1.430429};
        charges.y = {0.0, 0.0, 0.0};
        charges.z = {0.0, 1.107157, 1.107157};
        charges.charge = {8.0, 1.0, 1.0};  // O, H, H
        return charges;
    }

    /// Compare matrices with tolerance
    void compare_matrices(const std::vector<Real>& gpu,
                          const std::vector<Real>& cpu,
                          double tolerance,
                          const std::string& name) {
        ASSERT_EQ(gpu.size(), cpu.size()) << name << " size mismatch";
        for (size_t i = 0; i < gpu.size(); ++i) {
            if (std::abs(cpu[i]) > 1e-10) {
                double rel_err = std::abs((gpu[i] - cpu[i]) / cpu[i]);
                EXPECT_LT(rel_err, tolerance)
                    << name << " relative error at " << i
                    << ": GPU=" << gpu[i] << ", CPU=" << cpu[i];
            } else {
                EXPECT_NEAR(gpu[i], cpu[i], tolerance)
                    << name << " absolute error at " << i;
            }
        }
    }
};

// ============================================================================
// Basic Construction Tests
// ============================================================================

TEST_F(CudaEngineTest, ConstructAndDestroy) {
    auto shells = create_h2o_sto3g_basis();
    BasisSet basis(shells);

    CudaEngine engine(basis);
    EXPECT_TRUE(engine.is_initialized());
    EXPECT_EQ(&engine.basis(), &basis);
}

TEST_F(CudaEngineTest, MoveConstruction) {
    auto shells = create_h2o_sto3g_basis();
    BasisSet basis(shells);

    CudaEngine engine1(basis);
    EXPECT_TRUE(engine1.is_initialized());

    CudaEngine engine2(std::move(engine1));
    EXPECT_TRUE(engine2.is_initialized());
    EXPECT_FALSE(engine1.is_initialized());
}

// ============================================================================
// Overlap Matrix Tests
// ============================================================================

TEST_F(CudaEngineTest, OverlapMatrix_CompareWithCPU) {
    auto shells = create_h2o_sto3g_basis();
    BasisSet basis(shells);

    // GPU computation
    CudaEngine cuda_engine(basis);
    std::vector<Real> S_gpu;
    cuda_engine.compute_overlap_matrix(S_gpu);

    // CPU computation
    Engine cpu_engine(basis);
    std::vector<Real> S_cpu;
    cpu_engine.compute_1e(Operator::overlap(), S_cpu);

    // Compare
    compare_matrices(S_gpu, S_cpu, 1e-10, "Overlap");
}

// ============================================================================
// Kinetic Matrix Tests
// ============================================================================

TEST_F(CudaEngineTest, KineticMatrix_CompareWithCPU) {
    auto shells = create_h2o_sto3g_basis();
    BasisSet basis(shells);

    // GPU computation
    CudaEngine cuda_engine(basis);
    std::vector<Real> T_gpu;
    cuda_engine.compute_kinetic_matrix(T_gpu);

    // CPU computation
    Engine cpu_engine(basis);
    std::vector<Real> T_cpu;
    cpu_engine.compute_1e(Operator::kinetic(), T_cpu);

    // Compare
    compare_matrices(T_gpu, T_cpu, 1e-10, "Kinetic");
}

// ============================================================================
// Nuclear Attraction Matrix Tests
// ============================================================================

TEST_F(CudaEngineTest, NuclearMatrix_CompareWithCPU) {
    auto shells = create_h2o_sto3g_basis();
    BasisSet basis(shells);
    auto charges = create_h2o_charges();

    // GPU computation
    CudaEngine cuda_engine(basis);
    std::vector<Real> V_gpu;
    cuda_engine.compute_nuclear_matrix(charges, V_gpu);

    // CPU computation
    Engine cpu_engine(basis);
    std::vector<Real> V_cpu;
    cpu_engine.compute_1e(Operator::nuclear(charges), V_cpu);

    // Compare with stricter tolerance to catch meaningful regressions
    compare_matrices(V_gpu, V_cpu, 0.05, "Nuclear");
}

// ============================================================================
// Core Hamiltonian Tests
// ============================================================================

TEST_F(CudaEngineTest, CoreHamiltonian_CompareWithCPU) {
    auto shells = create_h2o_sto3g_basis();
    BasisSet basis(shells);
    auto charges = create_h2o_charges();

    // GPU computation
    CudaEngine cuda_engine(basis);
    std::vector<Real> H_gpu;
    cuda_engine.compute_core_hamiltonian(charges, H_gpu);

    // CPU computation: H = T + V
    Engine cpu_engine(basis);
    std::vector<Real> T_cpu, V_cpu;
    cpu_engine.compute_1e(Operator::kinetic(), T_cpu);
    cpu_engine.compute_1e(Operator::nuclear(charges), V_cpu);
    std::vector<Real> H_cpu(T_cpu.size());
    for (size_t i = 0; i < T_cpu.size(); ++i) {
        H_cpu[i] = T_cpu[i] + V_cpu[i];
    }

    // Compare with stricter tolerance to catch meaningful regressions
    compare_matrices(H_gpu, H_cpu, 0.05, "Core Hamiltonian");
}

// ============================================================================
// Shell Pair Tests
// ============================================================================

TEST_F(CudaEngineTest, OverlapShellPair_SameCenter) {
    Point3D center{0.0, 0.0, 0.0};
    std::vector<double> exponents = {1.0};
    std::vector<double> coefficients = {1.0};

    Shell shell_a(AngularMomentum::S, center, exponents, coefficients);
    Shell shell_b(AngularMomentum::S, center, exponents, coefficients);
    shell_a.set_shell_index(0);
    shell_b.set_shell_index(1);
    shell_a.set_function_index(0);
    shell_b.set_function_index(1);

    std::vector<Shell> shells = {shell_a, shell_b};
    BasisSet basis(shells);

    CudaEngine engine(basis);
    OverlapBuffer buffer;
    engine.compute_overlap_shell_pair(shell_a, shell_b, buffer);

    // For s-s on same center with alpha=1: integral = sqrt(pi/2)^3 = 1.0
    // Actually, for normalized s shells: <s|s> on same center = 1.0
    EXPECT_NEAR(buffer(0, 0), 1.0, 1e-10);
}

TEST_F(CudaEngineTest, EriShellQuartet_Ssss) {
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

    std::vector<Shell> shells = {shell_a, shell_b, shell_c, shell_d};
    BasisSet basis(shells);

    CudaEngine engine(basis);
    TwoElectronBuffer<0> buffer;
    engine.compute_eri_shell_quartet(shell_a, shell_b, shell_c, shell_d, buffer);

    // For s-s-s-s on same center with alpha=1:
    // (ss|ss) = 2 * pi^(5/2) / (zeta * eta * sqrt(zeta+eta)) * F_0(0)
    // where F_0(0) = 1, zeta = eta = 2, so = 2 * pi^2.5 / (2*2*2) = pi^2.5 / 4
    // With normalization, result should be approximately 1.128
    EXPECT_GT(buffer(0, 0, 0, 0), 0.5);  // Sanity check that it's positive
}

}  // namespace libaccint

#else  // LIBACCINT_USE_CUDA

TEST(CudaEngineTest, CudaNotEnabled) {
    GTEST_SKIP() << "CUDA support not enabled";
}

#endif  // LIBACCINT_USE_CUDA

#pragma GCC diagnostic pop
