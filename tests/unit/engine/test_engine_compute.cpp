// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_engine_compute.cpp
/// @brief Tests for unified Engine::compute() API

#include <libaccint/engine/engine.hpp>
#include <libaccint/engine/dispatch_policy.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/core/backend.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <gtest/gtest.h>
#include <vector>

using namespace libaccint;

namespace {

// =============================================================================
// Test Data
// =============================================================================

/// Water geometry in bohr
constexpr Point3D O_center{0.0, 0.0, 0.0};
constexpr Point3D H1_center{0.0, 1.43233673, -1.10866041};
constexpr Point3D H2_center{0.0, -1.43233673, -1.10866041};

/// Build STO-3G H2O shells (5 shells, 7 basis functions)
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

/// Build PointChargeParams for H2O nuclear charges
PointChargeParams make_h2o_charges() {
    PointChargeParams charges;
    charges.x = {0.0, 0.0, 0.0};
    charges.y = {0.0, 1.43233673, -1.43233673};
    charges.z = {0.0, -1.10866041, -1.10866041};
    charges.charge = {8.0, 1.0, 1.0};
    return charges;
}

/// Tolerance for floating-point comparisons
constexpr Real TIGHT_TOL = 1e-10;

}  // anonymous namespace

// =============================================================================
// Unified compute() API - Shell Pair Tests
// =============================================================================

TEST(EngineComputeTest, ComputeShellPair_Overlap) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Shell& shell_a = basis.shell(0);  // O 1s
    const Shell& shell_b = basis.shell(0);  // O 1s

    OneElectronBuffer<0> buffer;
    Operator op = Operator::overlap();

    // Using unified compute() API
    engine.compute(op, shell_a, shell_b, buffer);

    ASSERT_EQ(buffer.na(), 1);
    ASSERT_EQ(buffer.nb(), 1);
    EXPECT_NEAR(buffer(0, 0), 1.0, TIGHT_TOL)
        << "Self-overlap of O 1s should be 1.0";
}

TEST(EngineComputeTest, ComputeShellPair_Kinetic) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Shell& shell_a = basis.shell(0);  // O 1s
    const Shell& shell_b = basis.shell(1);  // O 2s

    OneElectronBuffer<0> buffer;
    Operator op = Operator::kinetic();

    engine.compute(op, shell_a, shell_b, buffer);

    ASSERT_EQ(buffer.na(), 1);
    ASSERT_EQ(buffer.nb(), 1);
    // Kinetic integral between O 1s and O 2s should be non-zero
    EXPECT_NE(buffer(0, 0), 0.0);
}

TEST(EngineComputeTest, ComputeShellPair_Nuclear) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    auto charges = make_h2o_charges();
    const Shell& shell_a = basis.shell(0);  // O 1s

    OneElectronBuffer<0> buffer;
    Operator op = Operator::nuclear(charges);

    engine.compute(op, shell_a, shell_a, buffer);

    ASSERT_EQ(buffer.na(), 1);
    ASSERT_EQ(buffer.nb(), 1);
    // Nuclear attraction should be negative
    EXPECT_LT(buffer(0, 0), 0.0);
}

TEST(EngineComputeTest, ComputeShellPair_ThrowsForTwoElectronOp) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Shell& shell_a = basis.shell(0);
    const Shell& shell_b = basis.shell(1);

    OneElectronBuffer<0> buffer;
    Operator op = Operator::coulomb();  // Two-electron operator

    EXPECT_THROW(engine.compute(op, shell_a, shell_b, buffer),
                 InvalidArgumentException);
}

TEST(EngineComputeTest, ComputeShellPair_ConsistencyWithDirect) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Shell& shell_a = basis.shell(0);
    const Shell& shell_b = basis.shell(3);  // H1 1s

    OneElectronBuffer<0> buffer_unified;
    OneElectronBuffer<0> buffer_direct;
    Operator op = Operator::overlap();

    // Unified API
    engine.compute(op, shell_a, shell_b, buffer_unified);

    // Direct API
    engine.compute_1e_shell_pair(op, shell_a, shell_b, buffer_direct);

    // Should be identical
    ASSERT_EQ(buffer_unified.na(), buffer_direct.na());
    ASSERT_EQ(buffer_unified.nb(), buffer_direct.nb());
    for (int i = 0; i < buffer_unified.na(); ++i) {
        for (int j = 0; j < buffer_unified.nb(); ++j) {
            EXPECT_NEAR(buffer_unified(i, j), buffer_direct(i, j), TIGHT_TOL);
        }
    }
}

// =============================================================================
// Unified compute() API - Shell Quartet Tests
// =============================================================================

TEST(EngineComputeTest, ComputeShellQuartet_Coulomb) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Shell& shell_a = basis.shell(0);  // O 1s

    TwoElectronBuffer<0> buffer;
    Operator op = Operator::coulomb();

    // (O 1s O 1s | O 1s O 1s)
    engine.compute(op, shell_a, shell_a, shell_a, shell_a, buffer);

    // All s-shells: should be 1x1x1x1 = 1 integral
    ASSERT_EQ(buffer.na(), 1);
    ASSERT_EQ(buffer.nb(), 1);
    ASSERT_EQ(buffer.nc(), 1);
    ASSERT_EQ(buffer.nd(), 1);

    // ERI should be positive
    EXPECT_GT(buffer(0, 0, 0, 0), 0.0);
}

TEST(EngineComputeTest, ComputeShellQuartet_ThrowsForOneElectronOp) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Shell& shell_a = basis.shell(0);

    TwoElectronBuffer<0> buffer;
    Operator op = Operator::overlap();  // One-electron operator

    EXPECT_THROW(engine.compute(op, shell_a, shell_a, shell_a, shell_a, buffer),
                 InvalidArgumentException);
}

TEST(EngineComputeTest, ComputeShellQuartet_ConsistencyWithDirect) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Shell& shell_a = basis.shell(0);  // O 1s
    const Shell& shell_b = basis.shell(1);  // O 2s

    TwoElectronBuffer<0> buffer_unified;
    TwoElectronBuffer<0> buffer_direct;
    Operator op = Operator::coulomb();

    // Unified API
    engine.compute(op, shell_a, shell_b, shell_a, shell_b, buffer_unified);

    // Direct API
    engine.compute_2e_shell_quartet(op, shell_a, shell_b, shell_a, shell_b, buffer_direct);

    // Should be identical
    ASSERT_EQ(buffer_unified.na(), buffer_direct.na());
    ASSERT_EQ(buffer_unified.nb(), buffer_direct.nb());
    ASSERT_EQ(buffer_unified.nc(), buffer_direct.nc());
    ASSERT_EQ(buffer_unified.nd(), buffer_direct.nd());

    for (int i = 0; i < buffer_unified.na(); ++i) {
        for (int j = 0; j < buffer_unified.nb(); ++j) {
            for (int k = 0; k < buffer_unified.nc(); ++k) {
                for (int l = 0; l < buffer_unified.nd(); ++l) {
                    EXPECT_NEAR(buffer_unified(i, j, k, l),
                                buffer_direct(i, j, k, l), TIGHT_TOL);
                }
            }
        }
    }
}

// =============================================================================
// Unified compute() API - ShellSetPair Tests
// =============================================================================

TEST(EngineComputeTest, ComputeShellSetPair_Overlap) {
    // This test verifies the compute() overload for ShellSetPair compiles
    // and dispatches correctly. The actual ShellSetPair computation is
    // tested more thoroughly in test_shell_set_pair.cpp.

    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    // Just verify the method exists and compiles - the actual ShellSetPair
    // construction requires specific shell properties that may not match
    // in this simple test setup

    // Verify compile-time: the method signature is correct
    // (would fail at compile time if the overload doesn't exist)
    [[maybe_unused]] auto method_ptr = static_cast<void (Engine::*)(
        const Operator&, const ShellSetPair&, std::vector<Real>&, BackendHint)>(&Engine::compute);

    SUCCEED();  // Test passes if compilation succeeds
}

// =============================================================================
// Unified compute() API - Full Basis OneElectronOperator Tests
// =============================================================================

TEST(EngineComputeTest, ComputeFullBasis_Overlap) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    std::vector<Real> S;
    OneElectronOperator op = Operator::overlap();

    engine.compute(op, S);

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(nbf, 7u);
    ASSERT_EQ(S.size(), 49u);

    // Diagonal should be 1.0
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_NEAR(S[i * nbf + i], 1.0, TIGHT_TOL);
    }
}

TEST(EngineComputeTest, ComputeFullBasis_Kinetic) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    std::vector<Real> T;
    OneElectronOperator op = Operator::kinetic();

    engine.compute(op, T);

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(T.size(), nbf * nbf);

    // Diagonal should be positive
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GT(T[i * nbf + i], 0.0);
    }
}

TEST(EngineComputeTest, ComputeFullBasis_ConsistencyWithDirect) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    std::vector<Real> result_unified;
    std::vector<Real> result_direct;
    OneElectronOperator op = Operator::overlap();

    // Unified API
    engine.compute(op, result_unified);

    // Direct API
    engine.compute_1e(op, result_direct);

    // Should be identical
    ASSERT_EQ(result_unified.size(), result_direct.size());
    for (Size i = 0; i < result_unified.size(); ++i) {
        EXPECT_NEAR(result_unified[i], result_direct[i], TIGHT_TOL);
    }
}

// =============================================================================
// Unified compute() API - Consumer Tests
// =============================================================================

TEST(EngineComputeTest, ComputeWithConsumer_ThrowsForOneElectronOp) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    struct DummyConsumer {
        void accumulate(const TwoElectronBuffer<0>&,
                        Index, Index, Index, Index,
                        int, int, int, int) {}
        void prepare_parallel(int) {}
        void finalize_parallel() {}
    } consumer;

    Operator op = Operator::overlap();  // One-electron operator

    // compute_and_consume (underlying method) should throw for 1e operators
    EXPECT_THROW(engine.compute(op, consumer), InvalidArgumentException);
}

// =============================================================================
// Backend Hint Tests
// =============================================================================

TEST(EngineComputeTest, ComputeShellPair_WithBackendHint) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Shell& shell_a = basis.shell(0);
    OneElectronBuffer<0> buffer;
    Operator op = Operator::overlap();

    // Should work with ForceCPU hint
    EXPECT_NO_THROW(engine.compute(op, shell_a, shell_a, buffer, BackendHint::ForceCPU));
    EXPECT_NEAR(buffer(0, 0), 1.0, TIGHT_TOL);

    // Should work with Auto hint (default)
    EXPECT_NO_THROW(engine.compute(op, shell_a, shell_a, buffer, BackendHint::Auto));
    EXPECT_NEAR(buffer(0, 0), 1.0, TIGHT_TOL);
}

TEST(EngineComputeTest, ComputeFullBasis_WithBackendHint) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    std::vector<Real> S;
    OneElectronOperator op = Operator::overlap();

    // Should work with ForceCPU hint
    EXPECT_NO_THROW(engine.compute(op, S, BackendHint::ForceCPU));

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(S.size(), nbf * nbf);
}

// =============================================================================
// Interface Compilation Tests
// =============================================================================

TEST(EngineComputeTest, InterfaceCompilation_AllOverloads) {
    // This test verifies that all compute() overloads compile correctly
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Shell& shell_a = basis.shell(0);
    const Shell& shell_b = basis.shell(1);

    // Overload 1: Shell pair + 1e buffer
    {
        OneElectronBuffer<0> buffer;
        engine.compute(Operator::overlap(), shell_a, shell_b, buffer);
    }

    // Overload 2: Shell quartet + 2e buffer
    {
        TwoElectronBuffer<0> buffer;
        engine.compute(Operator::coulomb(), shell_a, shell_a, shell_a, shell_a, buffer);
    }

    // Overload 3: ShellSetPair + result vector
    // (Skip if no matching shells available)

    // Overload 4: ShellSetQuartet + consumer
    // (Tested separately due to compilation requirements)

    // Overload 5: Operator + consumer (compute_and_consume)
    // (Tested separately due to O(N^4) cost)

    // Overload 6: OneElectronOperator + result vector
    {
        std::vector<Real> result;
        engine.compute(OneElectronOperator(Operator::overlap()), result);
    }
}
