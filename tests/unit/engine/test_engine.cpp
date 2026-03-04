// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/engine/engine.hpp>
#include <libaccint/engine/dispatch_policy.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/core/backend.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace libaccint;

namespace {

// =============================================================================
// STO-3G H2O Test Data
// =============================================================================

/// Water geometry in bohr
constexpr Point3D O_center{0.0, 0.0, 0.0};
constexpr Point3D H1_center{0.0, 1.43233673, -1.10866041};
constexpr Point3D H2_center{0.0, -1.43233673, -1.10866041};

/// Build STO-3G H2O shells (5 shells, 7 basis functions)
/// Shell order: O 1s, O 2s, O 2p, H1 1s, H2 1s
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

}  // anonymous namespace

// =============================================================================
// Construction Tests
// =============================================================================

TEST(EngineTest, DefaultConstruction) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    // Legacy backend() returns CPU for backward compatibility
    EXPECT_EQ(engine.backend(), BackendType::CPU);
    EXPECT_EQ(engine.max_angular_momentum(), 1);
}

TEST(EngineTest, ConstructionWithDispatchConfig) {
    BasisSet basis(make_sto3g_h2o_shells());
    DispatchConfig config;
    config.min_gpu_batch_size = 32;
    config.high_am_threshold = 5;

    Engine engine(basis, config);

    EXPECT_EQ(engine.dispatch_policy().config().min_gpu_batch_size, 32u);
    EXPECT_EQ(engine.dispatch_policy().config().high_am_threshold, 5);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST(EngineTest, BackendAccessor) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    // Legacy backend() returns CPU
    EXPECT_EQ(engine.backend(), BackendType::CPU);
}

TEST(EngineTest, BasisAccessor) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    // Verify basis() returns the same basis set
    const BasisSet& retrieved_basis = engine.basis();
    EXPECT_EQ(&retrieved_basis, &basis);
    EXPECT_EQ(retrieved_basis.n_shells(), 5u);
    EXPECT_EQ(retrieved_basis.n_basis_functions(), 7u);
}

TEST(EngineTest, MaxAngularMomentumAccessor) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    // STO-3G H2O has max_am = 1 (P-shell)
    EXPECT_EQ(engine.max_angular_momentum(), 1);
}

TEST(EngineTest, GpuAvailableAccessor) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    // gpu_available() reflects actual GPU availability
#if LIBACCINT_USE_CUDA
    // GPU might or might not be available depending on hardware
    // Just verify the method exists and returns a bool
    [[maybe_unused]] bool has_gpu = engine.gpu_available();
#else
    // Without CUDA support, GPU should never be available
    EXPECT_FALSE(engine.gpu_available());
#endif
}

TEST(EngineTest, CpuEngineAccessor) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    // cpu_engine() should always return a valid reference
    engine::CpuEngine& cpu = engine.cpu_engine();
    EXPECT_EQ(&cpu.basis(), &basis);
}

// =============================================================================
// Empty Basis Tests
// =============================================================================

TEST(EngineTest, EmptyBasisConstruction) {
    BasisSet empty_basis;
    Engine engine(empty_basis);

    EXPECT_EQ(engine.backend(), BackendType::CPU);
    EXPECT_EQ(engine.max_angular_momentum(), 0);
    EXPECT_EQ(engine.basis().n_shells(), 0u);
}

// =============================================================================
// STO-3G H2O Tests
// =============================================================================

TEST(EngineTest, STO3G_H2O_MaxAngularMomentum) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    // STO-3G H2O: max_am = 1 (P-shell on oxygen)
    EXPECT_EQ(engine.max_angular_momentum(), 1);
}

TEST(EngineTest, STO3G_H2O_BasisReference) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    // Verify the basis() reference is stable
    const BasisSet& ref1 = engine.basis();
    const BasisSet& ref2 = engine.basis();
    EXPECT_EQ(&ref1, &ref2);
    EXPECT_EQ(&ref1, &basis);
}

// =============================================================================
// Two-Electron Integral Tests
// =============================================================================

TEST(EngineTest, Compute2EShellQuartet) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    Operator op = Operator::coulomb();
    TwoElectronBuffer<0> buffer;

    // compute_2e_shell_quartet should work for Coulomb operator
    const Shell& shell_a = basis.shell(0);
    EXPECT_NO_THROW(engine.compute_2e_shell_quartet(op, shell_a, shell_a,
                                                     shell_a, shell_a, buffer));

    // Should throw for unsupported operator kinds
    Operator overlap_op = Operator::overlap();
    EXPECT_THROW(engine.compute_2e_shell_quartet(overlap_op, shell_a, shell_a,
                                                  shell_a, shell_a, buffer),
                 InvalidArgumentException);
}

TEST(EngineTest, ComputeAndConsumeRejectsOneElectronOp) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    Operator op = Operator::overlap();

    // A minimal consumer struct for testing
    struct DummyConsumer {
        void accumulate(const TwoElectronBuffer<0>&,
                        Index, Index, Index, Index,
                        int, int, int, int) {}
        void prepare_parallel(int) {}
        void finalize_parallel() {}
    } consumer;

    // compute_and_consume should throw for one-electron operators
    EXPECT_THROW(engine.compute_and_consume(op, consumer), InvalidArgumentException);
}

// =============================================================================
// One-Electron Integral Computation Tests
// =============================================================================

namespace {

/// Tolerance for floating-point comparisons
constexpr Real TIGHT_TOL = 1e-10;

/// Build PointChargeParams for H2O nuclear charges
PointChargeParams make_h2o_charges() {
    PointChargeParams charges;
    charges.x = {0.0, 0.0, 0.0};
    charges.y = {0.0, 1.43233673, -1.43233673};
    charges.z = {0.0, -1.10866041, -1.10866041};
    charges.charge = {8.0, 1.0, 1.0};
    return charges;
}

/// Check that a flat N x N matrix is symmetric
void expect_symmetric(const std::vector<Real>& matrix, Size n,
                      Real tol, const std::string& label) {
    for (Size i = 0; i < n; ++i) {
        for (Size j = i + 1; j < n; ++j) {
            EXPECT_NEAR(matrix[i * n + j], matrix[j * n + i], tol)
                << label << ": element (" << i << "," << j
                << ") != (" << j << "," << i << ")";
        }
    }
}

}  // anonymous namespace

TEST(EngineTest, Compute1E_OverlapMatrix) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    std::vector<Real> S;
    OneElectronOperator op = Operator::overlap();
    engine.compute_1e(op, S);

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(nbf, 7u);
    ASSERT_EQ(S.size(), 49u);  // 7 x 7

    // Diagonal elements should be 1.0 (normalized shells)
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_NEAR(S[i * nbf + i], 1.0, TIGHT_TOL)
            << "S(" << i << "," << i << ") should be 1.0";
    }

    // Matrix should be symmetric
    expect_symmetric(S, nbf, TIGHT_TOL, "Overlap");
}

TEST(EngineTest, Compute1E_KineticMatrix) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    std::vector<Real> T;
    OneElectronOperator op = Operator::kinetic();
    engine.compute_1e(op, T);

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(nbf, 7u);
    ASSERT_EQ(T.size(), 49u);

    // Diagonal elements should be positive (kinetic energy >= 0)
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GT(T[i * nbf + i], 0.0)
            << "T(" << i << "," << i << ") should be positive";
    }

    // Matrix should be symmetric
    expect_symmetric(T, nbf, TIGHT_TOL, "Kinetic");
}

TEST(EngineTest, Compute1E_NuclearMatrix) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    auto charges = make_h2o_charges();
    std::vector<Real> V;
    OneElectronOperator op = Operator::nuclear(charges);
    engine.compute_1e(op, V);

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(nbf, 7u);
    ASSERT_EQ(V.size(), 49u);

    // Diagonal elements should be negative (attractive potential)
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_LT(V[i * nbf + i], 0.0)
            << "V(" << i << "," << i << ") should be negative";
    }

    // Matrix should be symmetric
    expect_symmetric(V, nbf, TIGHT_TOL, "Nuclear");
}

TEST(EngineTest, Compute1E_ComposedOperator_Hcore) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    auto charges = make_h2o_charges();
    const Size nbf = basis.n_basis_functions();

    // Compute T and V separately
    std::vector<Real> T, V;
    engine.compute_1e(OneElectronOperator(Operator::kinetic()), T);
    engine.compute_1e(OneElectronOperator(Operator::nuclear(charges)), V);

    // Compute H_core = T + V using composed operator
    OneElectronOperator h_core = Operator::kinetic();
    h_core.add(Operator::nuclear(charges));
    std::vector<Real> H;
    engine.compute_1e(h_core, H);

    ASSERT_EQ(H.size(), nbf * nbf);

    // H_core should be T + V
    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(H[i], T[i] + V[i], TIGHT_TOL)
            << "H_core[" << i << "] should equal T + V";
    }
}

TEST(EngineTest, Compute1E_ScaleFactor) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Size nbf = basis.n_basis_functions();

    // Compute S
    std::vector<Real> S;
    engine.compute_1e(OneElectronOperator(Operator::overlap()), S);

    // Compute 2*S using scale factor
    OneElectronOperator scaled_op = 2.0 * OneElectronOperator(Operator::overlap());
    std::vector<Real> S2;
    engine.compute_1e(scaled_op, S2);

    ASSERT_EQ(S2.size(), nbf * nbf);

    // 2*S should be exactly twice S
    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(S2[i], 2.0 * S[i], TIGHT_TOL)
            << "2*S[" << i << "] should equal 2 * S";
    }
}

TEST(EngineTest, Compute1E_ShellPairOverlap) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    // Compute overlap for shell pair (0, 0) — O 1s with itself
    const Shell& shell_a = basis.shell(0);
    OneElectronBuffer<0> buffer;
    Operator op = Operator::overlap();

    engine.compute_1e_shell_pair(op, shell_a, shell_a, buffer);

    // O 1s is an s-shell: 1 function. Self-overlap = 1.0
    ASSERT_EQ(buffer.na(), 1);
    ASSERT_EQ(buffer.nb(), 1);
    EXPECT_NEAR(buffer(0, 0), 1.0, TIGHT_TOL)
        << "Self-overlap of O 1s should be 1.0";
}

TEST(EngineTest, Compute1E_ShellPairKinetic) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    // Compute kinetic for shell pair (0, 1) — O 1s with O 2s
    const Shell& shell_0 = basis.shell(0);
    const Shell& shell_1 = basis.shell(1);
    OneElectronBuffer<0> buffer;
    Operator op = Operator::kinetic();

    engine.compute_1e_shell_pair(op, shell_0, shell_1, buffer);

    // Both are s-shells: 1x1 buffer
    ASSERT_EQ(buffer.na(), 1);
    ASSERT_EQ(buffer.nb(), 1);

    // The kinetic integral between O 1s and O 2s should be non-zero
    EXPECT_NE(buffer(0, 0), 0.0)
        << "Kinetic integral between O 1s and O 2s should be non-zero";
}

TEST(EngineTest, Compute1E_ShellPairNuclear) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    auto charges = make_h2o_charges();
    const Shell& shell_0 = basis.shell(0);
    OneElectronBuffer<0> buffer;
    Operator op = Operator::nuclear(charges);

    engine.compute_1e_shell_pair(op, shell_0, shell_0, buffer);

    // O 1s self: 1x1 buffer, should be negative (attractive)
    ASSERT_EQ(buffer.na(), 1);
    ASSERT_EQ(buffer.nb(), 1);
    EXPECT_LT(buffer(0, 0), 0.0)
        << "Nuclear attraction of O 1s with itself should be negative";
}

TEST(EngineTest, Compute1E_ShellPairPShell) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    // Compute overlap for shell pair (2, 2) — O 2p with itself
    const Shell& shell_p = basis.shell(2);
    OneElectronBuffer<0> buffer;
    Operator op = Operator::overlap();

    engine.compute_1e_shell_pair(op, shell_p, shell_p, buffer);

    // O 2p is a p-shell: 3 functions. Self-overlap diagonal = 1.0
    ASSERT_EQ(buffer.na(), 3);
    ASSERT_EQ(buffer.nb(), 3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(buffer(i, i), 1.0, TIGHT_TOL)
            << "Self-overlap of O 2p(" << i << "," << i << ") should be 1.0";
    }

    // Off-diagonal should be 0 (orthogonal p components)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i != j) {
                EXPECT_NEAR(buffer(i, j), 0.0, TIGHT_TOL)
                    << "Self-overlap of O 2p(" << i << "," << j << ") should be 0.0";
            }
        }
    }
}

TEST(EngineTest, Compute1E_EmptyBasis) {
    BasisSet empty_basis;
    Engine engine(empty_basis);

    std::vector<Real> result;
    OneElectronOperator op = Operator::overlap();
    engine.compute_1e(op, result);

    // Empty basis: result should be empty
    EXPECT_EQ(result.size(), 0u);
}

TEST(EngineTest, Compute1E_UnsupportedOperatorThrows) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Shell& shell_a = basis.shell(0);
    const Shell& shell_b = basis.shell(1);
    OneElectronBuffer<0> buffer;

    // Coulomb is a two-electron operator, should throw InvalidArgumentException
    Operator op = Operator::coulomb();
    EXPECT_THROW(engine.compute_1e_shell_pair(op, shell_a, shell_b, buffer),
                 InvalidArgumentException);
}

TEST(EngineTest, Compute1E_OverlapConsistencyWithShellPair) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Size nbf = basis.n_basis_functions();

    // Compute full overlap matrix via compute_1e
    std::vector<Real> S;
    engine.compute_1e(OneElectronOperator(Operator::overlap()), S);

    // Compute individual shell pair and verify consistency
    // Shell pair (0, 3): O 1s with H1 1s
    const Shell& shell_0 = basis.shell(0);
    const Shell& shell_3 = basis.shell(3);
    OneElectronBuffer<0> buffer;
    engine.compute_1e_shell_pair(Operator::overlap(), shell_0, shell_3, buffer);

    // O 1s starts at function_index 0, H1 1s starts at function_index 4
    // (O 1s: 1 func, O 2s: 1 func, O 2p: 3 funcs = offset 4 for shell 3)
    Index fi = shell_0.function_index();
    Index fj = shell_3.function_index();

    ASSERT_EQ(buffer.na(), 1);
    ASSERT_EQ(buffer.nb(), 1);

    EXPECT_NEAR(buffer(0, 0),
                S[static_cast<Size>(fi) * nbf + static_cast<Size>(fj)],
                TIGHT_TOL)
        << "Shell pair overlap should match full matrix element";
}

// =============================================================================
// BackendHint Tests
// =============================================================================

TEST(EngineTest, BackendHint_ForceCPU) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    std::vector<Real> S;
    OneElectronOperator op = Operator::overlap();

    // ForceCPU hint should work regardless of GPU availability
    EXPECT_NO_THROW(engine.compute_1e(op, S, BackendHint::ForceCPU));

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(S.size(), nbf * nbf);
}

TEST(EngineTest, BackendHint_Auto) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    std::vector<Real> S;
    OneElectronOperator op = Operator::overlap();

    // Auto hint should work
    EXPECT_NO_THROW(engine.compute_1e(op, S, BackendHint::Auto));

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(S.size(), nbf * nbf);
}

// =============================================================================
// Convenience Method Tests
// =============================================================================

TEST(EngineTest, ComputeOverlapMatrix) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    std::vector<Real> S;
    engine.compute_overlap_matrix(S);

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(S.size(), nbf * nbf);

    // Diagonal should be 1.0
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_NEAR(S[i * nbf + i], 1.0, TIGHT_TOL);
    }
}

TEST(EngineTest, ComputeKineticMatrix) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    std::vector<Real> T;
    engine.compute_kinetic_matrix(T);

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(T.size(), nbf * nbf);

    // Diagonal should be positive
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GT(T[i * nbf + i], 0.0);
    }
}

TEST(EngineTest, ComputeNuclearMatrix) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    auto charges = make_h2o_charges();
    std::vector<Real> V;
    engine.compute_nuclear_matrix(charges, V);

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(V.size(), nbf * nbf);

    // Diagonal should be negative
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_LT(V[i * nbf + i], 0.0);
    }
}

TEST(EngineTest, ComputeCoreHamiltonian) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    auto charges = make_h2o_charges();
    const Size nbf = basis.n_basis_functions();

    // Compute T and V separately
    std::vector<Real> T, V;
    engine.compute_kinetic_matrix(T);
    engine.compute_nuclear_matrix(charges, V);

    // Compute core Hamiltonian
    std::vector<Real> H;
    engine.compute_core_hamiltonian(charges, H);

    ASSERT_EQ(H.size(), nbf * nbf);

    // H should equal T + V
    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(H[i], T[i] + V[i], TIGHT_TOL);
    }
}

// =============================================================================
// Interface Compilation Tests
// =============================================================================

TEST(EngineTest, InterfaceCompilation) {
    // This test verifies that all method signatures compile correctly
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    // Verify all accessors compile and are marked [[nodiscard]]
    [[maybe_unused]] auto backend = engine.backend();
    [[maybe_unused]] const BasisSet& basis_ref = engine.basis();
    [[maybe_unused]] auto max_am = engine.max_angular_momentum();
    [[maybe_unused]] bool gpu = engine.gpu_available();
    [[maybe_unused]] const DispatchPolicy& policy = engine.dispatch_policy();
    [[maybe_unused]] engine::CpuEngine& cpu = engine.cpu_engine();

    // Verify method signatures compile
    std::vector<Real> result;
    OneElectronOperator op1e = Operator::overlap();
    Operator op2e = Operator::coulomb();
    OneElectronBuffer<0> buffer;
    const Shell& shell_a = basis.shell(0);
    const Shell& shell_b = basis.shell(1);

    // compute_1e and compute_1e_shell_pair are now implemented
    engine.compute_1e(op1e, result);
    engine.compute_1e_shell_pair(Operator::overlap(), shell_a, shell_b, buffer);

    // Two-electron dispatch — testing compilation
    TwoElectronBuffer<0> buffer_2e;
    engine.compute_2e_shell_quartet(op2e, shell_a, shell_b, shell_a, shell_b, buffer_2e);

    // compute_and_consume with a minimal consumer
    struct TestConsumer {
        void accumulate(const TwoElectronBuffer<0>&,
                        Index, Index, Index, Index,
                        int, int, int, int) {}
    } consumer;
    // Don't call compute_and_consume here since it does a full O(N^4) loop
    // Just verify the interface compiles
    (void)consumer;
}

// =============================================================================
// Multiple Backend Tests (when available)
// =============================================================================

#if LIBACCINT_USE_CUDA
TEST(EngineTest, CudaEngineAccessor) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    // cuda_engine() returns pointer (nullptr if unavailable)
    CudaEngine* cuda = engine.cuda_engine();
    if (engine.gpu_available()) {
        EXPECT_NE(cuda, nullptr);
        EXPECT_EQ(&cuda->basis(), &basis);
    } else {
        EXPECT_EQ(cuda, nullptr);
    }
}
#endif

// =============================================================================
// Basis Lifetime Tests
// =============================================================================

TEST(EngineTest, BasisMustRemainValid) {
    // This test documents that the Engine stores a pointer to the BasisSet,
    // so the BasisSet must remain valid for the Engine's lifetime.

    Engine* engine_ptr = nullptr;
    {
        BasisSet basis(make_sto3g_h2o_shells());
        engine_ptr = new Engine(basis);

        // Engine is valid while basis is in scope
        EXPECT_EQ(engine_ptr->max_angular_momentum(), 1);
    }
    // After this point, basis is destroyed and engine_ptr->basis() would be dangling
    // Clean up
    delete engine_ptr;
}
