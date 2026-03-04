// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_high_am_integrals.cpp
/// @brief Integration tests exercising G-function (AM=4) and higher angular
///        momentum kernels with real basis sets (cc-pVQZ, aug-cc-pVTZ).
///
/// Step 13.5: Validates that high angular momentum integrals are computed
/// correctly using cc-pVQZ (which includes G-functions) on Neon and water.

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/data/basis_parser.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/engine/dispatch_policy.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace libaccint::test {
namespace {

// =============================================================================
// Test fixture
// =============================================================================

class HighAMIntegrals : public ::testing::Test {
protected:
    void SetUp() override {
        // Neon (Z=10) — single atom, simplest possible G-function test
        atoms_ne_ = {{10, {0.0, 0.0, 0.0}}};

        // Water — multi-atom with G-functions
        atoms_h2o_ = {
            {8, {0.0, 0.0, 0.0}},
            {1, {0.0, 1.43233673, -1.10866041}},
            {1, {0.0, -1.43233673, -1.10866041}},
        };
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// @brief Build nuclear point charges from an atom list
    PointChargeParams make_charges(const std::vector<data::Atom>& atoms) {
        PointChargeParams params;
        for (const auto& atom : atoms) {
            params.x.push_back(atom.position.x);
            params.y.push_back(atom.position.y);
            params.z.push_back(atom.position.z);
            params.charge.push_back(static_cast<Real>(atom.atomic_number));
        }
        return params;
    }

    /// @brief Assert that a row-major square matrix is symmetric
    void expect_symmetric(const std::vector<Real>& matrix, Size n,
                          Real tol, const std::string& label) {
        for (Size i = 0; i < n; ++i) {
            for (Size j = i + 1; j < n; ++j) {
                EXPECT_NEAR(matrix[i * n + j], matrix[j * n + i], tol)
                    << label << "[" << i << "," << j << "] vs ["
                    << j << "," << i << "]";
            }
        }
    }

    /// @brief Assert diagonal elements of the overlap matrix are 1.0
    void expect_unit_diagonal(const std::vector<Real>& matrix, Size n,
                              Real tol) {
        for (Size i = 0; i < n; ++i) {
            EXPECT_NEAR(matrix[i * n + i], 1.0, tol)
                << "diagonal[" << i << "]";
        }
    }

    // -------------------------------------------------------------------------
    // Fixture members
    // -------------------------------------------------------------------------

    std::vector<data::Atom> atoms_ne_;
    std::vector<data::Atom> atoms_h2o_;
};

// =============================================================================
// Basis loading and validation
// =============================================================================

TEST_F(HighAMIntegrals, LoadCcpVQZ_Water) {
    BasisSet basis;
    try {
        basis = data::load_basis_set("cc-pvqz", atoms_h2o_);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "cc-pVQZ basis loading failed: " << e.what();
    }

    EXPECT_GE(basis.max_angular_momentum(), 4)
        << "cc-pVQZ should contain G-functions (AM >= 4)";
}

TEST_F(HighAMIntegrals, LoadCcpVQZ_Neon) {
    BasisSet basis;
    try {
        basis = data::load_basis_set("cc-pvqz", atoms_ne_);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "cc-pVQZ basis loading failed: " << e.what();
    }

    EXPECT_GE(basis.max_angular_momentum(), 4)
        << "cc-pVQZ for Neon should contain G-functions (AM >= 4)";

    // Verify at least one shell has AM == 4
    bool has_g_shell = false;
    for (Size i = 0; i < basis.n_shells(); ++i) {
        if (basis.shell(i).angular_momentum() == 4) {
            has_g_shell = true;
            break;
        }
    }
    EXPECT_TRUE(has_g_shell)
        << "Neon cc-pVQZ should have at least one G-function shell";
}

TEST_F(HighAMIntegrals, CcpVQZ_ShellStructure) {
    BasisSet basis;
    try {
        basis = data::load_basis_set("cc-pvqz", atoms_h2o_);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "cc-pVQZ basis loading failed: " << e.what();
    }

    // cc-pVQZ water: O has ~55 contracted functions, each H ~30
    // Total should be roughly 115 basis functions
    EXPECT_GT(basis.n_basis_functions(), 80u)
        << "cc-pVQZ water should have a substantial basis";
    EXPECT_LT(basis.n_basis_functions(), 200u)
        << "cc-pVQZ water should not exceed ~200 basis functions";

    EXPECT_GT(basis.n_shells(), 20u)
        << "cc-pVQZ water should have many shells";

    EXPECT_GE(basis.n_shell_sets(), 3u)
        << "cc-pVQZ water should have several distinct ShellSet groups";
}

TEST_F(HighAMIntegrals, CcpVQZ_GFunctionShellSets) {
    BasisSet basis;
    try {
        basis = data::load_basis_set("cc-pvqz", atoms_h2o_);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "cc-pVQZ basis loading failed: " << e.what();
    }

    auto g_sets = basis.shell_sets_with_am(4);
    EXPECT_GE(g_sets.size(), 1u)
        << "cc-pVQZ water should have at least one ShellSet with AM == 4";

    // Each G-function ShellSet should contain at least one shell
    for (const auto* ss : g_sets) {
        EXPECT_GT(ss->n_shells(), 0u)
            << "G-function ShellSet should not be empty";
        EXPECT_EQ(ss->angular_momentum(), 4);
    }
}

// =============================================================================
// One-electron integrals with G-functions
// =============================================================================

TEST_F(HighAMIntegrals, OverlapMatrixCcpVQZ) {
    BasisSet basis;
    try {
        basis = data::load_basis_set("cc-pvqz", atoms_h2o_);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "cc-pVQZ basis loading failed: " << e.what();
    }

    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();

    std::vector<Real> S;
    ASSERT_NO_THROW(engine.compute_overlap_matrix(S));
    ASSERT_EQ(S.size(), nbf * nbf);

    // Overlap diagonal must be 1.0 (normalized basis functions)
    expect_unit_diagonal(S, nbf, 1e-12);

    // Overlap must be symmetric
    expect_symmetric(S, nbf, 1e-13, "S");

    // Off-diagonal elements should be bounded: |S_ij| <= 1.0
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            EXPECT_LE(std::abs(S[i * nbf + j]), 1.0 + 1e-10)
                << "Overlap S[" << i << "," << j << "] out of bounds";
        }
    }
}

TEST_F(HighAMIntegrals, KineticMatrixCcpVQZ) {
    BasisSet basis;
    try {
        basis = data::load_basis_set("cc-pvqz", atoms_h2o_);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "cc-pVQZ basis loading failed: " << e.what();
    }

    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();

    std::vector<Real> T;
    ASSERT_NO_THROW(engine.compute_kinetic_matrix(T));
    ASSERT_EQ(T.size(), nbf * nbf);

    // Kinetic energy diagonal must be strictly positive
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GT(T[i * nbf + i], 0.0)
            << "Kinetic T[" << i << "," << i << "] should be positive";
    }

    // Kinetic must be symmetric
    expect_symmetric(T, nbf, 1e-12, "T");
}

TEST_F(HighAMIntegrals, NuclearMatrixCcpVQZ) {
    BasisSet basis;
    try {
        basis = data::load_basis_set("cc-pvqz", atoms_h2o_);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "cc-pVQZ basis loading failed: " << e.what();
    }

    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();
    auto charges = make_charges(atoms_h2o_);

    std::vector<Real> V;
    ASSERT_NO_THROW(engine.compute_nuclear_matrix(charges, V));
    ASSERT_EQ(V.size(), nbf * nbf);

    // Nuclear attraction diagonal should be negative (attractive potential)
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_LT(V[i * nbf + i], 0.0)
            << "Nuclear V[" << i << "," << i << "] should be negative";
    }

    // Nuclear must be symmetric
    expect_symmetric(V, nbf, 1e-12, "V");
}

// =============================================================================
// Two-electron integrals with G-functions
// =============================================================================

TEST_F(HighAMIntegrals, ERICcpVQZ_SmallMolecule) {
    // Use Neon (single atom) to keep ERI computation manageable
    BasisSet basis;
    try {
        basis = data::load_basis_set("cc-pvqz", atoms_ne_);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "cc-pVQZ basis loading failed: " << e.what();
    }

    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();

    // Identity density matrix
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0;
    }

    consumers::FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    Operator coulomb = Operator::coulomb();
    ASSERT_NO_THROW(engine.compute_and_consume(coulomb, fock));

    auto J = fock.get_coulomb_matrix();
    ASSERT_EQ(J.size(), nbf * nbf);

    // J should be symmetric
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(J[i * nbf + j], J[j * nbf + i], 1e-10)
                << "Coulomb J[" << i << "," << j << "] not symmetric";
        }
    }

    // With identity density, J diagonal should be non-negative
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GE(J[i * nbf + i], -1e-10)
            << "J diagonal should be non-negative with identity density";
    }
}

TEST_F(HighAMIntegrals, ERIMixedAM_GFunctions) {
    BasisSet basis;
    try {
        basis = data::load_basis_set("cc-pvqz", atoms_ne_);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "cc-pVQZ basis loading failed: " << e.what();
    }

    // Verify that ShellSetQuartets involving G-functions are generated
    const auto& quartets = basis.shell_set_quartets();

    bool has_g_quartet = false;
    int max_am_seen = 0;
    for (const auto& q : quartets) {
        int qmax = std::max({q.La(), q.Lb(), q.Lc(), q.Ld()});
        max_am_seen = std::max(max_am_seen, qmax);
        if (q.La() == 4 || q.Lb() == 4 || q.Lc() == 4 || q.Ld() == 4) {
            has_g_quartet = true;
        }
    }

    EXPECT_TRUE(has_g_quartet)
        << "cc-pVQZ should generate ShellSetQuartets with AM == 4 (G-functions)";
    EXPECT_GE(max_am_seen, 4)
        << "Maximum AM seen in quartets should be >= 4";
}

// =============================================================================
// Generated vs generic kernel comparison
// =============================================================================

TEST_F(HighAMIntegrals, GeneratedVsGenericOverlap_GFunction) {
    BasisSet basis;
    try {
        basis = data::load_basis_set("cc-pvqz", atoms_ne_);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "cc-pVQZ basis loading failed: " << e.what();
    }

    const Size nbf = basis.n_basis_functions();

    // Compute overlap with default dispatch (may use generated kernels)
    Engine engine_default(basis);
    std::vector<Real> S_default;
    engine_default.compute_overlap_matrix(S_default);

    // Compute overlap forcing CPU (uses generic fallback path)
    Engine engine_cpu(basis);
    std::vector<Real> S_cpu;
    engine_cpu.compute_overlap_matrix(S_cpu, BackendHint::ForceCPU);

    ASSERT_EQ(S_default.size(), S_cpu.size());
    ASSERT_EQ(S_default.size(), nbf * nbf);

    // Results must agree to high precision
    for (Size i = 0; i < S_default.size(); ++i) {
        EXPECT_NEAR(S_default[i], S_cpu[i], 1e-12)
            << "Overlap mismatch at flat index " << i;
    }
}

TEST_F(HighAMIntegrals, GeneratedVsGenericERI_GFunction) {
    // Use Neon for manageable ERI computation
    BasisSet basis;
    try {
        basis = data::load_basis_set("cc-pvqz", atoms_ne_);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "cc-pVQZ basis loading failed: " << e.what();
    }

    const Size nbf = basis.n_basis_functions();

    // Identity density
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0;
    }

    Operator coulomb = Operator::coulomb();

    // Default dispatch
    Engine engine_default(basis);
    consumers::FockBuilder fock_default(nbf);
    fock_default.set_density(D.data(), nbf);
    engine_default.compute_and_consume(coulomb, fock_default);
    auto J_default = fock_default.get_coulomb_matrix();

    // Force CPU dispatch
    Engine engine_cpu(basis);
    consumers::FockBuilder fock_cpu(nbf);
    fock_cpu.set_density(D.data(), nbf);
    engine_cpu.compute_and_consume(coulomb, fock_cpu, BackendHint::ForceCPU);
    auto J_cpu = fock_cpu.get_coulomb_matrix();

    ASSERT_EQ(J_default.size(), J_cpu.size());

    for (Size i = 0; i < J_default.size(); ++i) {
        EXPECT_NEAR(J_default[i], J_cpu[i], 1e-10)
            << "ERI/FockBuilder J mismatch at flat index " << i;
    }
}

// =============================================================================
// GPU dispatch for high AM
// =============================================================================

TEST_F(HighAMIntegrals, GPUDispatchHighAM) {
    BasisSet basis;
    try {
        basis = data::load_basis_set("cc-pvqz", atoms_ne_);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "cc-pVQZ basis loading failed: " << e.what();
    }

    Engine engine(basis);
    if (!engine.gpu_available()) {
        GTEST_SKIP() << "GPU not available — skipping GPU dispatch test";
    }

    // Verify dispatch policy routes high-AM work to GPU
    DispatchConfig config;
    DispatchPolicy policy(config);

    // A batch with total AM >= high_am_threshold should prefer GPU
    BackendType selected = policy.select_backend(
        WorkUnitType::ShellSetQuartet,
        /*batch_size=*/100,
        /*total_am=*/config.high_am_threshold * 4,  // e.g., (gggg) total AM=16
        /*n_primitives=*/config.min_gpu_primitives,
        BackendHint::Auto,
        /*gpu_available=*/true);

    EXPECT_EQ(selected, BackendType::CUDA)
        << "High-AM quartets should be routed to GPU when available";
}

TEST_F(HighAMIntegrals, GPUvsCPU_CcpVQZ_Overlap) {
    BasisSet basis;
    try {
        basis = data::load_basis_set("cc-pvqz", atoms_h2o_);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "cc-pVQZ basis loading failed: " << e.what();
    }

    Engine engine(basis);
    if (!engine.gpu_available()) {
        GTEST_SKIP() << "GPU not available — skipping GPU vs CPU comparison";
    }

    const Size nbf = basis.n_basis_functions();

    std::vector<Real> S_cpu;
    engine.compute_overlap_matrix(S_cpu, BackendHint::ForceCPU);

    std::vector<Real> S_gpu;
    engine.compute_overlap_matrix(S_gpu, BackendHint::ForceGPU);

    ASSERT_EQ(S_cpu.size(), S_gpu.size());

    for (Size i = 0; i < S_cpu.size(); ++i) {
        EXPECT_NEAR(S_cpu[i], S_gpu[i], 1e-10)
            << "GPU vs CPU overlap mismatch at flat index " << i;
    }
}

// =============================================================================
// Core Hamiltonian
// =============================================================================

TEST_F(HighAMIntegrals, CoreHamiltonianCcpVQZ) {
    BasisSet basis;
    try {
        basis = data::load_basis_set("cc-pvqz", atoms_h2o_);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "cc-pVQZ basis loading failed: " << e.what();
    }

    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();
    auto charges = make_charges(atoms_h2o_);

    // Compute T and V separately
    std::vector<Real> T, V;
    engine.compute_kinetic_matrix(T);
    engine.compute_nuclear_matrix(charges, V);

    // Compute H_core = T + V via the combined method
    std::vector<Real> H_core;
    engine.compute_core_hamiltonian(charges, H_core);

    ASSERT_EQ(T.size(), nbf * nbf);
    ASSERT_EQ(V.size(), nbf * nbf);
    ASSERT_EQ(H_core.size(), nbf * nbf);

    // H_core should equal T + V
    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(H_core[i], T[i] + V[i], 1e-12)
            << "H_core != T + V at flat index " << i;
    }

    // Core Hamiltonian should be symmetric
    expect_symmetric(H_core, nbf, 1e-12, "H_core");
}

// =============================================================================
// aug-cc-pVTZ tests
// =============================================================================

TEST_F(HighAMIntegrals, LoadAugCcpVTZ_Water) {
    BasisSet basis;
    try {
        basis = data::load_basis_set("aug-cc-pvtz", atoms_h2o_);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "aug-cc-pVTZ basis loading failed: " << e.what();
    }

    // aug-cc-pVTZ should have at least d-functions (AM >= 2)
    EXPECT_GE(basis.max_angular_momentum(), 2)
        << "aug-cc-pVTZ should have at least d-functions";

    // aug-cc-pVTZ for water should have ~92 basis functions
    EXPECT_GT(basis.n_basis_functions(), 50u)
        << "aug-cc-pVTZ water should have a reasonable number of basis functions";

    // Verify overlap sanity as a quick smoke test
    Engine engine(basis);
    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(S.size(), nbf * nbf);
    expect_unit_diagonal(S, nbf, 1e-12);
}

}  // anonymous namespace
}  // namespace libaccint::test
