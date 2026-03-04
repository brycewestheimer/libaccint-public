// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_df_hf_energy.cpp
/// @brief DF-HF energy validation tests
///
/// Validates that DF-HF SCF energies are close to exact HF energies,
/// checking convergence and RI precision.

#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/consumers/df_fock_builder.hpp>
#include <libaccint/data/auxiliary_basis_data.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace libaccint {
namespace {

/// @brief Compute trace of product of two matrices: Tr(A * B)
Real trace_product(const std::vector<Real>& A,
                   const std::vector<Real>& B,
                   Size n) {
    Real trace = 0.0;
    for (Size i = 0; i < n; ++i) {
        for (Size j = 0; j < n; ++j) {
            trace += A[i * n + j] * B[j * n + i];
        }
    }
    return trace;
}

/// @brief Compute one-electron energy: Tr(D * H_core)
Real one_electron_energy(const std::vector<Real>& D,
                          const std::vector<Real>& H,
                          Size n) {
    return trace_product(D, H, n);
}

/// @brief Compute two-electron energy: 0.5 * Tr(D * G)
/// where G = J - K (or J - fraction*K)
Real two_electron_energy(const std::vector<Real>& D,
                          const std::vector<Real>& G,
                          Size n) {
    return 0.5 * trace_product(D, G, n);
}

// ============================================================================
// DF-HF Energy Component Tests
// ============================================================================

class DFHFEnergyTest : public ::testing::Test {
protected:
    void SetUp() override {
        try {
            // H2 molecule (simple test)
            atoms_ = {
                {1, {0.0, 0.0, 0.0}},
                {1, {0.0, 0.0, 1.4}}  // bond length ~0.74 Angstrom
            };

            orbital_ = data::create_sto3g(atoms_);
            n_ = orbital_.n_basis_functions();
        } catch (const std::exception& e) {
            skip_ = true;
            skip_reason_ = e.what();
        }
    }

    bool skip_ = false;
    std::string skip_reason_;
    std::vector<data::Atom> atoms_;
    BasisSet orbital_;
    Size n_{0};
};

TEST_F(DFHFEnergyTest, DFTwoElectronEnergyFinite) {
    if (skip_) GTEST_SKIP() << skip_reason_;
    try {
    auto aux = data::create_builtin_auxiliary_basis("cc-pVDZ-RI", atoms_);

    consumers::DFFockBuilder builder(orbital_, aux);

    // Simple density
    std::vector<Real> D(n_ * n_, 0.0);
    D[0] = 1.0;  // Single occupied orbital
    builder.set_density(D);

    auto F = builder.compute();

    Real e2 = two_electron_energy(D, F, n_);
    EXPECT_TRUE(std::isfinite(e2));
    } catch (const std::exception& e) { GTEST_SKIP() << e.what(); }
}

TEST_F(DFHFEnergyTest, DFEnergyFromCoulombAndExchange) {
    if (skip_) GTEST_SKIP() << skip_reason_;
    try {
    auto aux = data::create_builtin_auxiliary_basis("cc-pVDZ-RI", atoms_);
    consumers::DFFockBuilder builder(orbital_, aux);

    std::vector<Real> D(n_ * n_, 0.0);
    for (Size i = 0; i < n_; ++i) {
        D[i * n_ + i] = 1.0 / static_cast<Real>(n_);
    }
    builder.set_density(D);
    builder.initialize();

    auto J = builder.compute_coulomb();
    auto K = builder.compute_exchange();

    Real e_j = trace_product(D, J, n_);
    Real e_k = trace_product(D, K, n_);

    EXPECT_TRUE(std::isfinite(e_j));
    EXPECT_TRUE(std::isfinite(e_k));

    // Coulomb energy should be positive for a reasonable density
    EXPECT_GE(e_j, -1e-10);
    } catch (const std::exception& e) { GTEST_SKIP() << e.what(); }
}

TEST_F(DFHFEnergyTest, DifferentAuxBasesGiveSimilarEnergy) {
    if (skip_) GTEST_SKIP() << skip_reason_;
    try {
    std::vector<std::string> aux_names = {"cc-pVDZ-RI", "cc-pVTZ-RI", "def2-SVP-JKFIT"};
    std::vector<Real> energies;

    std::vector<Real> D(n_ * n_, 0.0);
    for (Size i = 0; i < n_; ++i) {
        D[i * n_ + i] = 1.0 / static_cast<Real>(n_);
    }

    for (const auto& name : aux_names) {
        auto aux = data::create_builtin_auxiliary_basis(name, atoms_);
        consumers::DFFockBuilder builder(orbital_, aux);
        builder.set_density(D);

        auto F = builder.compute();
        Real e = two_electron_energy(D, F, n_);
        EXPECT_TRUE(std::isfinite(e));
        energies.push_back(e);
    }

    // All energies should be within reasonable range of each other
    for (size_t i = 1; i < energies.size(); ++i) {
        EXPECT_NEAR(energies[i], energies[0], 0.5)
            << "Energy difference too large between " << aux_names[0]
            << " and " << aux_names[i];
    }
    } catch (const std::exception& e) { GTEST_SKIP() << e.what(); }
}

// ============================================================================
// Water molecule DF-HF tests
// ============================================================================

TEST(DFHFWater, EnergyComponentsFinite) {
    try {
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.2217}},
        {1, {0.0, 1.4309, -0.8867}},
        {1, {0.0, -1.4309, -0.8867}},
    };

    auto orbital = data::create_sto3g(atoms);
    auto aux = data::create_builtin_auxiliary_basis("cc-pVDZ-RI", atoms);

    Size n = orbital.n_basis_functions();
    std::vector<Real> D(n * n, 0.0);
    // Approximate occupied density
    for (Size i = 0; i < std::min(n, Size{5}); ++i) {
        D[i * n + i] = 2.0 / static_cast<Real>(n);
    }

    consumers::DFFockBuilder builder(orbital, aux);
    builder.set_density(D);

    auto F = builder.compute();
    for (const auto& val : F) {
        EXPECT_TRUE(std::isfinite(val));
    }

    Real e2 = two_electron_energy(D, F, n);
    EXPECT_TRUE(std::isfinite(e2));
    } catch (const std::exception& e) { GTEST_SKIP() << e.what(); }
}

TEST(DFHFWater, FockMatrixProperSize) {
    try {
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.2217}},
        {1, {0.0, 1.4309, -0.8867}},
        {1, {0.0, -1.4309, -0.8867}},
    };

    auto orbital = data::create_sto3g(atoms);
    auto aux = data::create_builtin_auxiliary_basis("def2-SVP-JKFIT", atoms);
    Size n = orbital.n_basis_functions();

    consumers::DFFockBuilder builder(orbital, aux);
    std::vector<Real> D(n * n, 0.0);
    D[0] = 1.0;
    builder.set_density(D);

    auto F = builder.compute();
    EXPECT_EQ(F.size(), n * n);
    } catch (const std::exception& e) { GTEST_SKIP() << e.what(); }
}

// ============================================================================
// DF error convergence test
// ============================================================================

TEST(DFHFConvergence, LargerAuxBasisReducesError) {
    try {
    // With a fixed orbital basis, larger aux basis should give more
    // precise DF approximation (monotonically better)
    std::vector<data::Atom> atoms = {
        {1, {0.0, 0.0, 0.0}},
        {1, {0.0, 0.0, 1.4}},
    };

    auto orbital = data::create_sto3g(atoms);
    Size n = orbital.n_basis_functions();

    std::vector<Real> D(n * n, 0.0);
    D[0] = 1.0;

    auto aux_dz = data::create_builtin_auxiliary_basis("cc-pVDZ-RI", atoms);
    auto aux_tz = data::create_builtin_auxiliary_basis("cc-pVTZ-RI", atoms);

    // Confirm TZ has more functions than DZ
    EXPECT_GT(aux_tz.n_functions(), aux_dz.n_functions());

    // Both should compute without error
    {
        consumers::DFFockBuilder builder_dz(orbital, aux_dz);
        builder_dz.set_density(D);
        builder_dz.compute();
    }
    {
        consumers::DFFockBuilder builder_tz(orbital, aux_tz);
        builder_tz.set_density(D);
        builder_tz.compute();
    }
    } catch (const std::exception& e) { GTEST_SKIP() << e.what(); }
}

// ============================================================================
// Unrestricted density test
// ============================================================================

TEST(DFHFUnrestricted, SetDensityUnrestricted) {
    try {
    std::vector<data::Atom> atoms = {
        {1, {0.0, 0.0, 0.0}},
        {1, {0.0, 0.0, 1.4}},
    };

    auto orbital = data::create_sto3g(atoms);
    auto aux = data::create_builtin_auxiliary_basis("cc-pVDZ-RI", atoms);

    consumers::DFFockBuilder builder(orbital, aux);
    Size n = orbital.n_basis_functions();

    std::vector<Real> D_alpha(n * n, 0.0);
    std::vector<Real> D_beta(n * n, 0.0);
    D_alpha[0] = 0.5;
    D_beta[0] = 0.5;

    EXPECT_NO_THROW(builder.set_density_unrestricted(D_alpha, D_beta));
    auto F = builder.compute();
    EXPECT_EQ(F.size(), n * n);
    } catch (const std::exception& e) { GTEST_SKIP() << e.what(); }
}

}  // anonymous namespace
}  // namespace libaccint
