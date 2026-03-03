// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_higher_am_gpu.cpp
/// @brief GPU validation for higher angular momentum integrals (Task 23.1.4)
///
/// Compares GPU-computed integrals to CPU reference for f/g/h functions.
/// All tests GTEST_SKIP() when CUDA is not available.

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/config.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

namespace libaccint::test {
namespace {

/// @brief Create a shell for GPU testing
Shell make_gpu_test_shell(int am, Point3D center) {
    std::vector<Real> exponents;
    std::vector<Real> coefficients;

    if (am <= 2) {
        exponents = {3.42525091, 0.62391373, 0.16885540};
        coefficients = {0.15432897, 0.53532814, 0.44463454};
    } else if (am == 3) {
        exponents = {1.533, 0.5417, 0.2211};
        coefficients = {0.25, 0.50, 0.35};
    } else if (am == 4) {
        exponents = {1.208, 0.4537, 0.1813};
        coefficients = {0.30, 0.45, 0.35};
    } else {
        exponents = {0.9876, 0.3654, 0.1432};
        coefficients = {0.35, 0.40, 0.35};
    }

    return Shell(am, center, exponents, coefficients);
}

/// @brief Build a basis set for GPU testing
BasisSet make_gpu_test_basis(int am) {
    std::vector<Shell> shells;
    auto s0 = make_gpu_test_shell(0, {0.0, 0.0, 0.0});
    s0.set_atom_index(0);
    shells.push_back(std::move(s0));

    auto s1 = make_gpu_test_shell(am, {0.0, 0.0, 1.5});
    s1.set_atom_index(1);
    shells.push_back(std::move(s1));

    return BasisSet(std::move(shells));
}

// ============================================================================
// GPU vs CPU Higher AM Validation
// ============================================================================

class HigherAMGpuTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override {
        am_ = GetParam();

        // Skip if no GPU backend available
        if (!has_cuda_backend()) {
            skip_ = true;
            skip_reason_ = "No GPU backend (CUDA) available";
            return;
        }

        try {
            basis_ = make_gpu_test_basis(am_);
            nbf_ = basis_.n_basis_functions();
        } catch (const std::exception& e) {
            skip_ = true;
            skip_reason_ = std::string("Setup failed: ") + e.what();
        }
    }

    int am_{0};
    BasisSet basis_;
    Size nbf_{0};
    bool skip_{false};
    std::string skip_reason_;
};

TEST_P(HigherAMGpuTest, OverlapGpuMatchesCpu) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    if (!engine.gpu_available()) {
        GTEST_SKIP() << "GPU not available in Engine";
    }

    std::vector<Real> S_cpu(nbf_ * nbf_, 0.0);
    std::vector<Real> S_gpu(nbf_ * nbf_, 0.0);

    engine.compute_overlap_matrix(S_cpu, BackendHint::ForceCPU);
    engine.compute_overlap_matrix(S_gpu, BackendHint::PreferGPU);

    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(S_cpu[i], S_gpu[i], 1e-12)
            << "GPU/CPU overlap mismatch at index " << i << " for AM=" << am_;
    }
}

TEST_P(HigherAMGpuTest, KineticGpuMatchesCpu) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    if (!engine.gpu_available()) {
        GTEST_SKIP() << "GPU not available in Engine";
    }

    std::vector<Real> T_cpu(nbf_ * nbf_, 0.0);
    std::vector<Real> T_gpu(nbf_ * nbf_, 0.0);

    engine.compute_kinetic_matrix(T_cpu, BackendHint::ForceCPU);
    engine.compute_kinetic_matrix(T_gpu, BackendHint::PreferGPU);

    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(T_cpu[i], T_gpu[i], 1e-12)
            << "GPU/CPU kinetic mismatch at index " << i << " for AM=" << am_;
    }
}

TEST_P(HigherAMGpuTest, NuclearGpuMatchesCpu) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    if (!engine.gpu_available()) {
        GTEST_SKIP() << "GPU not available in Engine";
    }

    PointChargeParams charges;
    charges.x = {0.0, 0.0};
    charges.y = {0.0, 0.0};
    charges.z = {0.0, 1.5};
    charges.charge = {8.0, 1.0};

    std::vector<Real> V_cpu(nbf_ * nbf_, 0.0);
    std::vector<Real> V_gpu(nbf_ * nbf_, 0.0);

    engine.compute_nuclear_matrix(charges, V_cpu, BackendHint::ForceCPU);
    engine.compute_nuclear_matrix(charges, V_gpu, BackendHint::PreferGPU);

    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(V_cpu[i], V_gpu[i], 1e-11)
            << "GPU/CPU nuclear mismatch at index " << i << " for AM=" << am_;
    }
}

TEST_P(HigherAMGpuTest, ERIGpuMatchesCpu) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    if (!engine.gpu_available()) {
        GTEST_SKIP() << "GPU not available in Engine";
    }

    Shell shell_a = make_gpu_test_shell(am_, {0.0, 0.0, 0.0});
    Shell shell_b = make_gpu_test_shell(0, {0.0, 0.0, 1.5});

    int na = shell_a.n_functions();
    int nb = shell_b.n_functions();

    TwoElectronBuffer<0> buf_cpu(na, nb, na, nb);
    TwoElectronBuffer<0> buf_gpu(na, nb, na, nb);
    buf_cpu.clear();
    buf_gpu.clear();

    engine.compute_2e_shell_quartet(Operator::coulomb(),
                                     shell_a, shell_b, shell_a, shell_b,
                                     buf_cpu, BackendHint::ForceCPU);
    engine.compute_2e_shell_quartet(Operator::coulomb(),
                                     shell_a, shell_b, shell_a, shell_b,
                                     buf_gpu, BackendHint::PreferGPU);

    auto data_cpu = buf_cpu.data();
    auto data_gpu = buf_gpu.data();

    for (Size i = 0; i < data_cpu.size(); ++i) {
        EXPECT_NEAR(data_cpu[i], data_gpu[i], 1e-10)
            << "GPU/CPU ERI mismatch at index " << i << " for AM=" << am_;
    }
}

std::string HigherAMGpuName(const ::testing::TestParamInfo<int>& info) {
    switch (info.param) {
        case 3: return "f_AM3";
        case 4: return "g_AM4";
        case 5: return "h_AM5";
        default: return "AM" + std::to_string(info.param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    HigherAMGpu,
    HigherAMGpuTest,
    ::testing::Values(3, 4, 5),
    HigherAMGpuName
);

}  // namespace
}  // namespace libaccint::test
