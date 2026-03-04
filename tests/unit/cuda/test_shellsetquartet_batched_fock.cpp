// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_shellsetquartet_batched_fock.cpp
/// @brief Tests for Phase 4.5 CUDA true batched ShellSetQuartet execution
///
/// Validates that batched GPU ERI computation with device-side Fock accumulation
/// produces results matching CPU reference.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/engine/cuda_engine.hpp>
#include <libaccint/engine/cpu_engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/consumers/fock_builder_gpu.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <map>
#include <memory>
#include <span>

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

    // O 1s
    {
        Shell s(0, O_center,
                {130.7093200, 23.8088610, 6.4436083},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }

    // O 2s
    {
        Shell s(0, O_center,
                {5.0331513, 1.1695961, 0.3803890},
                {-0.09996723, 0.39951283, 0.70011547});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }

    // O 2p
    {
        Shell s(1, O_center,
                {5.0331513, 1.1695961, 0.3803890},
                {0.15591627, 0.60768372, 0.39195739});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }

    // H1 1s
    {
        Shell s(0, H1_center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(1);
        shells.push_back(std::move(s));
    }

    // H2 1s
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
        ShellSetKey key{shell.angular_momentum(), static_cast<int>(shell.n_primitives())};
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

/// Compare matrices (accepts both vectors and spans)
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

class ShellSetQuartetBatchedFockTest : public ::testing::Test {
protected:
    void SetUp() override {
        shells_ = make_sto3g_h2o_shells();
        basis_ = std::make_unique<BasisSet>(shells_);

        cpu_engine_ = std::make_unique<engine::CpuEngine>(*basis_);

        try {
            cuda_engine_ = std::make_unique<CudaEngine>(*basis_);
        } catch (const std::exception& e) {
            cuda_available_ = false;
        }

        // Use basis_->shells() which has function indices assigned
        std::vector<Shell> shells_with_indices(basis_->shells().begin(), basis_->shells().end());
        shell_sets_ = group_shells_into_sets(shells_with_indices);

        // Create identity-like density matrix
        const Size nbf = basis_->n_basis_functions();
        D_.resize(nbf * nbf, 0.0);
        for (Size i = 0; i < nbf; ++i) {
            D_[i * nbf + i] = 1.0;
        }
    }

    std::vector<Shell> shells_;
    std::unique_ptr<BasisSet> basis_;
    std::unique_ptr<engine::CpuEngine> cpu_engine_;
    std::unique_ptr<CudaEngine> cuda_engine_;
    std::map<ShellSetKey, std::unique_ptr<ShellSet>> shell_sets_;
    std::vector<Real> D_;
    bool cuda_available_ = true;
};

// =============================================================================
// Device-Side Fock Build Tests
// =============================================================================

TEST_F(ShellSetQuartetBatchedFockTest, GpuFockBuilderBatchedMatchesCpu) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    const Size nbf = basis_->n_basis_functions();

    auto it = shell_sets_.find({0, 3});
    if (it == shell_sets_.end()) {
        GTEST_SKIP() << "No s-type shells found";
    }

    const ShellSet& s_shells = *it->second;

    ShellSetPair bra(s_shells, s_shells);
    ShellSetPair ket(s_shells, s_shells);
    ShellSetQuartet quartet(bra, ket);

    // CPU reference using FockBuilder with Atomic strategy
    consumers::FockBuilder cpu_fock(nbf);
    cpu_fock.set_threading_strategy(consumers::FockThreadingStrategy::Atomic);
    cpu_fock.set_density(D_.data(), nbf);
    cpu_engine_->compute_shell_set_quartet(Operator::coulomb(), quartet, cpu_fock);
    auto cpu_J = cpu_fock.get_coulomb_matrix();
    auto cpu_K = cpu_fock.get_exchange_matrix();

    // GPU using GpuFockBuilder with device-side accumulation
    consumers::GpuFockBuilder gpu_fock(nbf, cuda_engine_->stream());
    gpu_fock.set_density(D_.data(), nbf);
    cuda_engine_->compute_shell_set_quartet(Operator::coulomb(), quartet, gpu_fock);
    auto gpu_J = gpu_fock.get_coulomb_matrix();
    auto gpu_K = gpu_fock.get_exchange_matrix();

    // Compare Coulomb matrices
    double max_diff_J = 0.0;
    EXPECT_TRUE(matrices_near(gpu_J, cpu_J, TOLERANCE, &max_diff_J))
        << "GPU batched Coulomb (J) differs from CPU by " << max_diff_J;

    // Compare Exchange matrices
    double max_diff_K = 0.0;
    EXPECT_TRUE(matrices_near(gpu_K, cpu_K, TOLERANCE, &max_diff_K))
        << "GPU batched Exchange (K) differs from CPU by " << max_diff_K;
}

TEST_F(ShellSetQuartetBatchedFockTest, DeviceSideAccumulationNonZeroResults) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    const Size nbf = basis_->n_basis_functions();

    auto it = shell_sets_.find({0, 3});
    if (it == shell_sets_.end()) {
        GTEST_SKIP() << "No s-type shells found";
    }

    const ShellSet& s_shells = *it->second;

    ShellSetPair bra(s_shells, s_shells);
    ShellSetPair ket(s_shells, s_shells);
    ShellSetQuartet quartet(bra, ket);

    consumers::GpuFockBuilder gpu_fock(nbf, cuda_engine_->stream());
    gpu_fock.set_density(D_.data(), nbf);
    cuda_engine_->compute_shell_set_quartet(Operator::coulomb(), quartet, gpu_fock);

    auto J = gpu_fock.get_coulomb_matrix();
    auto K = gpu_fock.get_exchange_matrix();

    // Verify non-zero results
    bool j_nonzero = false, k_nonzero = false;
    for (Size i = 0; i < nbf * nbf; ++i) {
        if (std::abs(J[i]) > 1e-15) j_nonzero = true;
        if (std::abs(K[i]) > 1e-15) k_nonzero = true;
    }

    EXPECT_TRUE(j_nonzero) << "Device-side accumulated J should be non-zero";
    EXPECT_TRUE(k_nonzero) << "Device-side accumulated K should be non-zero";
}

// =============================================================================
// compute_eri_batch_device Tests
// =============================================================================

TEST_F(ShellSetQuartetBatchedFockTest, EriBatchDeviceOutputSize) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto it = shell_sets_.find({0, 3});
    if (it == shell_sets_.end()) {
        GTEST_SKIP() << "No s-type shells found";
    }

    const ShellSet& s_shells = *it->second;

    ShellSetPair bra(s_shells, s_shells);
    ShellSetPair ket(s_shells, s_shells);
    ShellSetQuartet quartet(bra, ket);

    auto batch = cuda_engine_->compute_eri_batch_device_handle(quartet);

    // Expected: n_quartets * funcs_per_quartet
    // s-type shells: 1 function each
    // n_quartets = n_s^4 (e.g., 4^4 = 256 for 4 s-shells)
    Size expected_quartets = s_shells.n_shells() * s_shells.n_shells() *
                             s_shells.n_shells() * s_shells.n_shells();
    Size expected_funcs_per_quartet = 1 * 1 * 1 * 1;  // (ss|ss)
    Size expected_count = expected_quartets * expected_funcs_per_quartet;

    EXPECT_EQ(batch.size(), expected_count)
        << "ERI batch should have " << expected_count << " values";
    EXPECT_NE(batch.data(), nullptr)
        << "Device ERI output pointer should be non-null";
}

TEST_F(ShellSetQuartetBatchedFockTest, RawPointerBatchApiThrows) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto it = shell_sets_.find({0, 3});
    if (it == shell_sets_.end()) {
        GTEST_SKIP() << "No s-type shells found";
    }

    const ShellSet& s_shells = *it->second;
    ShellSetPair bra(s_shells, s_shells);
    ShellSetPair ket(s_shells, s_shells);
    ShellSetQuartet quartet(bra, ket);

    double* d_eri_output = nullptr;
    size_t eri_count = 0;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    EXPECT_THROW(
        cuda_engine_->compute_eri_batch_device(quartet, d_eri_output, eri_count),
        InvalidStateException);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
}

// =============================================================================
// CPU FockBuilder Fallback Tests
// =============================================================================

TEST_F(ShellSetQuartetBatchedFockTest, CpuFockBuilderFallbackWorks) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    const Size nbf = basis_->n_basis_functions();

    auto it = shell_sets_.find({0, 3});
    if (it == shell_sets_.end()) {
        GTEST_SKIP() << "No s-type shells found";
    }

    const ShellSet& s_shells = *it->second;

    ShellSetPair bra(s_shells, s_shells);
    ShellSetPair ket(s_shells, s_shells);
    ShellSetQuartet quartet(bra, ket);

    // Use CPU FockBuilder with CUDA engine
    // This should use the fallback path (download + CPU accumulate)
    consumers::FockBuilder cpu_fock(nbf);
    cpu_fock.set_threading_strategy(consumers::FockThreadingStrategy::Sequential);
    cpu_fock.set_density(D_.data(), nbf);

    cuda_engine_->compute_shell_set_quartet(Operator::coulomb(), quartet, cpu_fock);

    auto J = cpu_fock.get_coulomb_matrix();

    bool j_nonzero = false;
    for (Size i = 0; i < nbf * nbf; ++i) {
        if (std::abs(J[i]) > 1e-15) {
            j_nonzero = true;
            break;
        }
    }

    EXPECT_TRUE(j_nonzero)
        << "CPU FockBuilder fallback should produce non-zero Coulomb matrix";
}

// =============================================================================
// Multiple ShellSetQuartet Tests
// =============================================================================

TEST_F(ShellSetQuartetBatchedFockTest, MultipleQuartetsAccumulate) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    const Size nbf = basis_->n_basis_functions();

    auto s_it = shell_sets_.find({0, 3});
    auto p_it = shell_sets_.find({1, 3});

    if (s_it == shell_sets_.end()) {
        GTEST_SKIP() << "No s-type shells found";
    }

    const ShellSet& s_shells = *s_it->second;

    // Create GPU FockBuilder
    consumers::GpuFockBuilder gpu_fock(nbf, cuda_engine_->stream());
    gpu_fock.set_density(D_.data(), nbf);

    // Process (ss|ss) quartet
    {
        ShellSetPair bra(s_shells, s_shells);
        ShellSetPair ket(s_shells, s_shells);
        ShellSetQuartet quartet(bra, ket);
        cuda_engine_->compute_shell_set_quartet(Operator::coulomb(), quartet, gpu_fock);
    }

    // If p shells exist, process (ss|pp) quartet too
    if (p_it != shell_sets_.end()) {
        const ShellSet& p_shells = *p_it->second;
        ShellSetPair bra(s_shells, s_shells);
        ShellSetPair ket(p_shells, p_shells);
        ShellSetQuartet quartet(bra, ket);
        cuda_engine_->compute_shell_set_quartet(Operator::coulomb(), quartet, gpu_fock);
    }

    // Results should have accumulated from multiple quartets
    auto J = gpu_fock.get_coulomb_matrix();

    bool j_nonzero = false;
    for (Size i = 0; i < nbf * nbf; ++i) {
        if (std::abs(J[i]) > 1e-15) {
            j_nonzero = true;
            break;
        }
    }

    EXPECT_TRUE(j_nonzero)
        << "Multiple quartets should accumulate into non-zero J matrix";
}

#endif  // LIBACCINT_USE_CUDA
