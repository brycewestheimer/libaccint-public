// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_engine_gpu_routing.cpp
/// @brief Phase 0.5 Engine-level GPU routing regression and performance-smoke tests
///
/// Validates that Engine::compute() correctly routes to GPU when
/// BackendHint::PreferGPU is used, and that GPU results match CPU results
/// end-to-end through the unified Engine API.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/engine/engine.hpp>
#include <libaccint/engine/dispatch_policy.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>
#include <span>
#include <vector>
#include <cmath>

using namespace libaccint;

namespace {

// =============================================================================
// Test Data
// =============================================================================

/// Water geometry in bohr
constexpr Point3D O_center{0.0, 0.0, 0.0};
constexpr Point3D H1_center{0.0, 1.43233673, -1.10866041};
constexpr Point3D H2_center{0.0, -1.43233673, -1.10866041};

/// Build STO-3G H2O shells
std::vector<Shell> make_sto3g_h2o_shells() {
    std::vector<Shell> shells;
    shells.reserve(5);

    // O 1s (L=0, K=3, atom 0)
    {
        Shell s(0, O_center,
                {130.7093200, 23.8088610, 6.4436083},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }

    // O 2s (L=0, K=3, atom 0)
    {
        Shell s(0, O_center,
                {5.0331513, 1.1695961, 0.3803890},
                {-0.09996723, 0.39951283, 0.70011547});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }

    // O 2p (L=1, K=3, atom 0)
    {
        Shell s(1, O_center,
                {5.0331513, 1.1695961, 0.3803890},
                {0.15591627, 0.60768372, 0.39195739});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }

    // H1 1s (L=0, K=3, atom 1)
    {
        Shell s(0, H1_center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(1);
        shells.push_back(std::move(s));
    }

    // H2 1s (L=0, K=3, atom 2)
    {
        Shell s(0, H2_center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(2);
        shells.push_back(std::move(s));
    }

    return shells;
}

/// Tolerance for GPU vs CPU 1e comparison
constexpr double TOLERANCE_1E = 1e-10;

/// Tolerance for GPU vs CPU 2e comparison (Fock matrices)
constexpr double TOLERANCE_2E = 1e-8;

/// Compare containers element-wise
template<typename ContainerA, typename ContainerB>
bool matrices_near(const ContainerA& a, const ContainerB& b,
                   double tol, double* max_diff = nullptr) {
    if (a.size() != b.size()) return false;
    double max_d = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = std::abs(a[i] - b[i]);
        if (diff > max_d) max_d = diff;
    }
    if (max_diff) *max_diff = max_d;
    return max_d <= tol;
}

}  // namespace

// =============================================================================
// Test Class
// =============================================================================

class EngineGpuRoutingTest : public ::testing::Test {
protected:
    void SetUp() override {
        basis_ = std::make_unique<BasisSet>(make_sto3g_h2o_shells());
        engine_ = std::make_unique<Engine>(*basis_);
        nbf_ = basis_->n_basis_functions();
    }

    std::unique_ptr<BasisSet> basis_;
    std::unique_ptr<Engine> engine_;
    Size nbf_{0};
};

// =============================================================================
// GPU Availability Check
// =============================================================================

TEST_F(EngineGpuRoutingTest, GpuAvailableCheck) {
    // In a CUDA-enabled build, the Engine should detect the GPU.
    // If no GPU is present at runtime, other tests will GTEST_SKIP().
    if (!engine_->gpu_available()) {
        GTEST_SKIP() << "CUDA GPU not available at runtime";
    }
    EXPECT_TRUE(engine_->gpu_available());
}

// =============================================================================
// 1e Integral GPU Routing Tests
// =============================================================================

TEST_F(EngineGpuRoutingTest, OverlapPreferGPU) {
    if (!engine_->gpu_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Compute overlap via GPU path
    std::vector<Real> S_gpu;
    engine_->compute_overlap_matrix(S_gpu, BackendHint::PreferGPU);

    // Compute overlap via CPU path
    std::vector<Real> S_cpu;
    engine_->compute_overlap_matrix(S_cpu, BackendHint::ForceCPU);

    ASSERT_EQ(S_gpu.size(), nbf_ * nbf_);
    ASSERT_EQ(S_cpu.size(), nbf_ * nbf_);

    // Verify diagonal (self-overlap = 1.0)
    for (Size i = 0; i < nbf_; ++i) {
        EXPECT_NEAR(S_gpu[i * nbf_ + i], 1.0, TOLERANCE_1E)
            << "GPU overlap diagonal[" << i << "] should be 1.0";
    }

    // Compare GPU vs CPU
    double max_diff = 0.0;
    EXPECT_TRUE(matrices_near(S_gpu, S_cpu, TOLERANCE_1E, &max_diff))
        << "GPU overlap differs from CPU by " << max_diff;
}

TEST_F(EngineGpuRoutingTest, KineticPreferGPU) {
    if (!engine_->gpu_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Compute kinetic via GPU path
    std::vector<Real> T_gpu;
    engine_->compute_kinetic_matrix(T_gpu, BackendHint::PreferGPU);

    // Compute kinetic via CPU path
    std::vector<Real> T_cpu;
    engine_->compute_kinetic_matrix(T_cpu, BackendHint::ForceCPU);

    ASSERT_EQ(T_gpu.size(), nbf_ * nbf_);
    ASSERT_EQ(T_cpu.size(), nbf_ * nbf_);

    // Verify diagonal (kinetic energy should be positive)
    for (Size i = 0; i < nbf_; ++i) {
        EXPECT_GT(T_gpu[i * nbf_ + i], 0.0)
            << "GPU kinetic diagonal[" << i << "] should be positive";
    }

    // Compare GPU vs CPU
    double max_diff = 0.0;
    EXPECT_TRUE(matrices_near(T_gpu, T_cpu, TOLERANCE_1E, &max_diff))
        << "GPU kinetic differs from CPU by " << max_diff;
}

// =============================================================================
// 2e Integral GPU Routing Tests (Full-Basis Fock Build)
// =============================================================================

TEST_F(EngineGpuRoutingTest, FockPreferGPU) {
    if (!engine_->gpu_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Build identity density matrix
    std::vector<Real> D(nbf_ * nbf_, 0.0);
    for (Size i = 0; i < nbf_; ++i) {
        D[i * nbf_ + i] = 1.0;
    }

    // GPU path: full-basis Fock build with PreferGPU
    consumers::FockBuilder fock_gpu(nbf_);
    fock_gpu.set_density(D.data(), nbf_);
    engine_->compute(Operator::coulomb(), fock_gpu, BackendHint::PreferGPU);

    auto J_gpu = fock_gpu.get_coulomb_matrix();
    auto K_gpu = fock_gpu.get_exchange_matrix();

    // CPU path: full-basis Fock build with ForceCPU
    consumers::FockBuilder fock_cpu(nbf_);
    fock_cpu.set_density(D.data(), nbf_);
    engine_->compute(Operator::coulomb(), fock_cpu, BackendHint::ForceCPU);

    auto J_cpu = fock_cpu.get_coulomb_matrix();
    auto K_cpu = fock_cpu.get_exchange_matrix();

    ASSERT_EQ(J_gpu.size(), nbf_ * nbf_);
    ASSERT_EQ(K_gpu.size(), nbf_ * nbf_);

    // Verify J is non-zero
    bool j_nonzero = false;
    for (Size i = 0; i < J_gpu.size(); ++i) {
        if (std::abs(J_gpu[i]) > 1e-15) {
            j_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(j_nonzero) << "GPU Coulomb matrix J should be non-zero";

    // Compare Coulomb matrices
    double max_diff_J = 0.0;
    EXPECT_TRUE(matrices_near(J_gpu, J_cpu, TOLERANCE_2E, &max_diff_J))
        << "GPU Coulomb (J) differs from CPU by " << max_diff_J;

    // Compare Exchange matrices
    double max_diff_K = 0.0;
    EXPECT_TRUE(matrices_near(K_gpu, K_cpu, TOLERANCE_2E, &max_diff_K))
        << "GPU Exchange (K) differs from CPU by " << max_diff_K;
}

// =============================================================================
// 2e Integral GPU Routing Tests (Worklist Path)
// =============================================================================

TEST_F(EngineGpuRoutingTest, WorklistPreferGPU) {
    if (!engine_->gpu_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Build identity density matrix
    std::vector<Real> D(nbf_ * nbf_, 0.0);
    for (Size i = 0; i < nbf_; ++i) {
        D[i * nbf_ + i] = 1.0;
    }

    // Get all ShellSetQuartets from the basis
    const auto& quartets = basis_->shell_set_quartets();
    ASSERT_GT(quartets.size(), 0u) << "Basis should have shell set quartets";

    // GPU path: worklist-based Fock build with PreferGPU
    consumers::FockBuilder fock_gpu(nbf_);
    fock_gpu.set_density(D.data(), nbf_);
    engine_->compute(Operator::coulomb(),
                     std::span<const ShellSetQuartet>(quartets),
                     fock_gpu,
                     BackendHint::PreferGPU);

    auto J_gpu = fock_gpu.get_coulomb_matrix();
    auto K_gpu = fock_gpu.get_exchange_matrix();

    // CPU path: worklist-based Fock build with ForceCPU
    consumers::FockBuilder fock_cpu(nbf_);
    fock_cpu.set_density(D.data(), nbf_);
    engine_->compute(Operator::coulomb(),
                     std::span<const ShellSetQuartet>(quartets),
                     fock_cpu,
                     BackendHint::ForceCPU);

    auto J_cpu = fock_cpu.get_coulomb_matrix();
    auto K_cpu = fock_cpu.get_exchange_matrix();

    ASSERT_EQ(J_gpu.size(), nbf_ * nbf_);
    ASSERT_EQ(K_gpu.size(), nbf_ * nbf_);

    // Verify J is non-zero
    bool j_nonzero = false;
    for (Size i = 0; i < J_gpu.size(); ++i) {
        if (std::abs(J_gpu[i]) > 1e-15) {
            j_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(j_nonzero) << "GPU worklist Coulomb matrix J should be non-zero";

    // Compare Coulomb matrices
    double max_diff_J = 0.0;
    EXPECT_TRUE(matrices_near(J_gpu, J_cpu, TOLERANCE_2E, &max_diff_J))
        << "GPU worklist Coulomb (J) differs from CPU by " << max_diff_J;

    // Compare Exchange matrices
    double max_diff_K = 0.0;
    EXPECT_TRUE(matrices_near(K_gpu, K_cpu, TOLERANCE_2E, &max_diff_K))
        << "GPU worklist Exchange (K) differs from CPU by " << max_diff_K;
}

#else  // !LIBACCINT_USE_CUDA

#include <gtest/gtest.h>

TEST(EngineGpuRoutingTest, CudaNotEnabled) {
    GTEST_SKIP() << "CUDA not enabled in this build";
}

#endif  // LIBACCINT_USE_CUDA
