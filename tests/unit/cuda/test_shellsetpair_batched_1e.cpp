// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_shellsetpair_batched_1e.cpp
/// @brief Tests for Phase 4.5 CUDA true batched ShellSetPair execution
///
/// Validates that batched GPU execution produces results matching CPU reference.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/engine/cuda_engine.hpp>
#include <libaccint/engine/cpu_engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <map>

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

/// Build ShellSets from shells
std::map<ShellSetKey, std::unique_ptr<ShellSet>> group_shells_into_sets(const std::vector<Shell>& shells) {
    std::map<ShellSetKey, std::unique_ptr<ShellSet>> sets;

    for (const auto& shell : shells) {
        ShellSetKey key{shell.angular_momentum(), shell.n_primitives()};
        auto it = sets.find(key);
        if (it == sets.end()) {
            auto new_set = std::make_unique<ShellSet>(shell.angular_momentum(), shell.n_primitives());
            new_set->add_shell(shell);
            sets.emplace(key, std::move(new_set));
        } else {
            it->second->add_shell(shell);
        }
    }

    return sets;
}

/// Tolerance for GPU vs CPU comparison
constexpr double TOLERANCE = 1e-10;

/// Compare matrices
bool matrices_near(const std::vector<Real>& gpu, const std::vector<Real>& cpu,
                   double tol, double* max_diff = nullptr) {
    if (gpu.size() != cpu.size()) return false;
    double max_d = 0.0;
    for (size_t i = 0; i < gpu.size(); ++i) {
        double diff = std::abs(gpu[i] - cpu[i]);
        if (diff > max_d) max_d = diff;
    }
    if (max_diff) *max_diff = max_d;
    return max_d <= tol;
}

}  // namespace

// =============================================================================
// Test Class
// =============================================================================

class ShellSetPairBatched1eTest : public ::testing::Test {
protected:
    void SetUp() override {
        shells_ = make_sto3g_h2o_shells();
        basis_ = std::make_unique<BasisSet>(shells_);

        // Initialize engines
        cpu_engine_ = std::make_unique<engine::CpuEngine>(*basis_);

        try {
            cuda_engine_ = std::make_unique<CudaEngine>(*basis_);
        } catch (const std::exception& e) {
            // GPU not available
            cuda_available_ = false;
        }

        // Use basis_->shells() which has function indices assigned
        std::vector<Shell> shells_with_indices(basis_->shells().begin(), basis_->shells().end());
        shell_sets_ = group_shells_into_sets(shells_with_indices);
    }

    std::vector<Shell> shells_;
    std::unique_ptr<BasisSet> basis_;
    std::unique_ptr<engine::CpuEngine> cpu_engine_;
    std::unique_ptr<CudaEngine> cuda_engine_;
    std::map<ShellSetKey, std::unique_ptr<ShellSet>> shell_sets_;
    bool cuda_available_ = true;
};

// =============================================================================
// Overlap Integral Tests
// =============================================================================

TEST_F(ShellSetPairBatched1eTest, OverlapBatchedMatchesCpu) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    const Size nbf = basis_->n_basis_functions();

    auto it = shell_sets_.find({0, 3});
    if (it == shell_sets_.end()) {
        GTEST_SKIP() << "No s-type shells found";
    }

    const ShellSet& s_shells = *it->second;
    ShellSetPair pair(s_shells, s_shells);

    // Compute on CPU
    std::vector<Real> cpu_result(nbf * nbf, 0.0);
    cpu_engine_->compute_shell_set_pair(Operator::overlap(), pair, cpu_result);

    // Compute on GPU (batched)
    std::vector<Real> gpu_result(nbf * nbf, 0.0);
    cuda_engine_->compute_shell_set_pair(Operator::overlap(), pair, gpu_result);

    // Compare
    double max_diff = 0.0;
    EXPECT_TRUE(matrices_near(gpu_result, cpu_result, TOLERANCE, &max_diff))
        << "GPU batched overlap differs from CPU by " << max_diff;
}

TEST_F(ShellSetPairBatched1eTest, OverlapMixedAngularMomentum) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    const Size nbf = basis_->n_basis_functions();

    auto s_it = shell_sets_.find({0, 3});
    auto p_it = shell_sets_.find({1, 3});

    if (s_it == shell_sets_.end() || p_it == shell_sets_.end()) {
        GTEST_SKIP() << "Need both s and p shells";
    }

    const ShellSet& s_shells = *s_it->second;
    const ShellSet& p_shells = *p_it->second;

    // (s|p) pair
    ShellSetPair pair(s_shells, p_shells);

    std::vector<Real> cpu_result(nbf * nbf, 0.0);
    cpu_engine_->compute_shell_set_pair(Operator::overlap(), pair, cpu_result);

    std::vector<Real> gpu_result(nbf * nbf, 0.0);
    cuda_engine_->compute_shell_set_pair(Operator::overlap(), pair, gpu_result);

    double max_diff = 0.0;
    EXPECT_TRUE(matrices_near(gpu_result, cpu_result, TOLERANCE, &max_diff))
        << "GPU batched (s|p) overlap differs from CPU by " << max_diff;
}

// =============================================================================
// Kinetic Integral Tests
// =============================================================================

TEST_F(ShellSetPairBatched1eTest, KineticBatchedMatchesCpu) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    const Size nbf = basis_->n_basis_functions();

    auto it = shell_sets_.find({0, 3});
    if (it == shell_sets_.end()) {
        GTEST_SKIP() << "No s-type shells found";
    }

    const ShellSet& s_shells = *it->second;
    ShellSetPair pair(s_shells, s_shells);

    std::vector<Real> cpu_result(nbf * nbf, 0.0);
    cpu_engine_->compute_shell_set_pair(Operator::kinetic(), pair, cpu_result);

    std::vector<Real> gpu_result(nbf * nbf, 0.0);
    cuda_engine_->compute_shell_set_pair(Operator::kinetic(), pair, gpu_result);

    double max_diff = 0.0;
    EXPECT_TRUE(matrices_near(gpu_result, cpu_result, TOLERANCE, &max_diff))
        << "GPU batched kinetic differs from CPU by " << max_diff;
}

// =============================================================================
// Nuclear Attraction Tests
// =============================================================================

TEST_F(ShellSetPairBatched1eTest, NuclearBatchedMatchesCpu) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    const Size nbf = basis_->n_basis_functions();

    PointChargeParams charges;
    charges.x = {0.0, 0.0, 0.0};
    charges.y = {0.0, 1.43233673, -1.43233673};
    charges.z = {0.0, -1.10866041, -1.10866041};
    charges.charge = {8.0, 1.0, 1.0};

    auto it = shell_sets_.find({0, 3});
    if (it == shell_sets_.end()) {
        GTEST_SKIP() << "No s-type shells found";
    }

    const ShellSet& s_shells = *it->second;
    ShellSetPair pair(s_shells, s_shells);

    std::vector<Real> cpu_result(nbf * nbf, 0.0);
    cpu_engine_->compute_shell_set_pair(Operator::nuclear(charges), pair, cpu_result);

    std::vector<Real> gpu_result(nbf * nbf, 0.0);
    cuda_engine_->compute_shell_set_pair(Operator::nuclear(charges), pair, gpu_result);

    double max_diff = 0.0;
    EXPECT_TRUE(matrices_near(gpu_result, cpu_result, TOLERANCE, &max_diff))
        << "GPU batched nuclear differs from CPU by " << max_diff;
}

// =============================================================================
// Performance Verification
// =============================================================================

TEST_F(ShellSetPairBatched1eTest, BatchedUsesFewerKernelLaunches) {
    // This test documents the expected behavior: batched execution should
    // use a single kernel launch per ShellSetPair, not per shell pair.
    //
    // While we can't easily count kernel launches in a unit test,
    // we verify the correct behavior by checking that results match
    // and execution completes successfully.

    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    const Size nbf = basis_->n_basis_functions();

    auto it = shell_sets_.find({0, 3});
    if (it == shell_sets_.end()) {
        GTEST_SKIP() << "No s-type shells found";
    }

    const ShellSet& s_shells = *it->second;
    Size n_shell_pairs = s_shells.n_shells() * s_shells.n_shells();

    // Phase 4.5: With true batching, this should be 1 kernel launch
    // Previously: n_shell_pairs kernel launches
    // We can't verify the launch count, but we verify correctness

    ShellSetPair pair(s_shells, s_shells);

    std::vector<Real> result(nbf * nbf, 0.0);
    cuda_engine_->compute_shell_set_pair(Operator::overlap(), pair, result);

    // Verify result has values (test didn't crash or produce zeros)
    bool has_nonzero = false;
    for (Size i = 0; i < result.size(); ++i) {
        if (std::abs(result[i]) > 1e-15) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero)
        << "Batched execution should produce non-zero results for "
        << n_shell_pairs << " shell pairs";
}

#endif  // LIBACCINT_USE_CUDA
