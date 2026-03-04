// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_two_electron.cpp
/// @brief Integration tests for two-electron integrals, FockBuilder, and fused path

#include <libaccint/engine/engine.hpp>
#include <libaccint/kernels/eri_kernel.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/utils/constants.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace libaccint;
using namespace libaccint::data;
using namespace libaccint::kernels;
using namespace libaccint::consumers;

// =============================================================================
// H2O/STO-3G Test Fixture
// =============================================================================

namespace {

constexpr Real ERI_TOL = 1e-10;

/// H2O geometry in Bohr (same as Phase 1 integration tests)
BasisSet make_h2o_sto3g() {
    std::vector<Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},                    // O
        {1, {0.0, 1.43233673, -1.10866041}},      // H1
        {1, {0.0, -1.43233673, -1.10866041}},     // H2
    };
    return create_sto3g(atoms);
}

/// H2 geometry in Bohr
BasisSet make_h2_sto3g() {
    std::vector<Atom> atoms = {
        {1, {0.0, 0.0, 0.0}},
        {1, {0.0, 0.0, 1.39839733}},
    };
    return create_sto3g(atoms);
}

}  // anonymous namespace

// =============================================================================
// Primitive ERI Tests (ss|ss)
// =============================================================================

TEST(ERIKernelTest, SsSsSameCenter) {
    // Two s-type hydrogen shells at the same center
    // (1s 1s | 1s 1s) = single diagonal ERI
    Shell s1(0, {0.0, 0.0, 0.0}, {3.42525091, 0.62391373, 0.16885540},
                                   {0.15432897, 0.53532814, 0.44463454});
    Shell s2(0, {0.0, 0.0, 0.0}, {3.42525091, 0.62391373, 0.16885540},
                                   {0.15432897, 0.53532814, 0.44463454});

    TwoElectronBuffer<0> buffer;
    compute_eri(s1, s1, s1, s1, buffer);

    // PySCF reference: H at origin, (0,0|0,0) = 0.7746059439198978
    EXPECT_NEAR(buffer(0, 0, 0, 0), 0.7746059439198978, ERI_TOL);
}

TEST(ERIKernelTest, SsSsDifferentCenters) {
    // H2 molecule: two 1s shells at different centers
    auto basis = make_h2_sto3g();
    Engine engine(basis);

    TwoElectronBuffer<0> buffer;
    const auto& s0 = basis.shell(0);
    const auto& s1 = basis.shell(1);

    // (0 0 | 0 0)
    compute_eri(s0, s0, s0, s0, buffer);
    EXPECT_NEAR(buffer(0, 0, 0, 0), 0.7746059439198978, ERI_TOL);

    // (0 0 | 1 1)
    compute_eri(s0, s0, s1, s1, buffer);
    EXPECT_NEAR(buffer(0, 0, 0, 0), 0.5699948826767758, ERI_TOL);

    // (0 1 | 0 1)
    compute_eri(s0, s1, s0, s1, buffer);
    EXPECT_NEAR(buffer(0, 0, 0, 0), 0.2975905517135221, ERI_TOL);

    // (1 1 | 1 1)
    compute_eri(s1, s1, s1, s1, buffer);
    EXPECT_NEAR(buffer(0, 0, 0, 0), 0.7746059439198978, ERI_TOL);
}

// =============================================================================
// H2O/STO-3G ERI Validation against PySCF
// =============================================================================

class H2OERI : public ::testing::Test {
protected:
    void SetUp() override {
        basis = make_h2o_sto3g();
        ASSERT_EQ(basis.n_basis_functions(), 7u);
    }

    /// Compute a single ERI element (mu nu | lambda sigma) using individual shells
    Real compute_element(int mu, int nu, int lam, int sig) {
        // Find which shell each index belongs to
        auto find_shell = [this](int idx) -> std::pair<Size, int> {
            for (Size s = 0; s < basis.n_shells(); ++s) {
                const auto& shell = basis.shell(s);
                int fi = static_cast<int>(shell.function_index());
                int nf = shell.n_functions();
                if (idx >= fi && idx < fi + nf) {
                    return {s, idx - fi};
                }
            }
            return {0, 0};
        };

        auto [si, ai] = find_shell(mu);
        auto [sj, bj] = find_shell(nu);
        auto [sk, ck] = find_shell(lam);
        auto [sl, dl] = find_shell(sig);

        TwoElectronBuffer<0> buffer;
        compute_eri(basis.shell(si), basis.shell(sj),
                    basis.shell(sk), basis.shell(sl), buffer);
        return buffer(ai, bj, ck, dl);
    }

    BasisSet basis;
};

TEST_F(H2OERI, DiagonalElements) {
    // PySCF reference values for H2O/STO-3G (Bohr geometry)
    EXPECT_NEAR(compute_element(0, 0, 0, 0), 4.785065404705503e+00, ERI_TOL);
    EXPECT_NEAR(compute_element(1, 1, 0, 0), 1.118946866342470e+00, ERI_TOL);
    EXPECT_NEAR(compute_element(0, 0, 1, 1), 1.118946866342470e+00, ERI_TOL);
    EXPECT_NEAR(compute_element(2, 2, 2, 2), 8.801590933750454e-01, ERI_TOL);
    EXPECT_NEAR(compute_element(5, 5, 5, 5), 7.746059439198978e-01, ERI_TOL);
    EXPECT_NEAR(compute_element(6, 6, 6, 6), 7.746059439198978e-01, ERI_TOL);
}

TEST_F(H2OERI, OffDiagonalElements) {
    EXPECT_NEAR(compute_element(0, 1, 0, 1), 1.368733853543883e-01, ERI_TOL);
    EXPECT_NEAR(compute_element(0, 0, 6, 6), 5.313797098289044e-01, ERI_TOL);
    EXPECT_NEAR(compute_element(5, 6, 5, 6), 3.562963018430980e-02, ERI_TOL);
    EXPECT_NEAR(compute_element(0, 2, 0, 2), 2.447741225809927e-02, ERI_TOL);
    EXPECT_NEAR(compute_element(1, 2, 1, 2), 1.805183921046321e-01, ERI_TOL);
    EXPECT_NEAR(compute_element(5, 5, 6, 6), 3.425329442119288e-01, ERI_TOL);
}

TEST_F(H2OERI, EightFoldSymmetry) {
    // Test 8-fold permutation symmetry:
    // (mu nu | lam sig) = (nu mu | lam sig) = (mu nu | sig lam) = (nu mu | sig lam)
    //                    = (lam sig | mu nu) = (sig lam | mu nu) = (lam sig | nu mu)
    //                    = (sig lam | nu mu)

    // Use non-trivial indices from different shells
    // Shell 0: O 1s (idx 0), Shell 1: O 2s (idx 1), Shell 2: O 2p (idx 2,3,4)
    // Shell 3: H1 1s (idx 5), Shell 4: H2 1s (idx 6)

    struct TestCase {
        int mu, nu, lam, sig;
    };

    std::vector<TestCase> cases = {
        {0, 1, 5, 6},
        {0, 5, 1, 6},
        {1, 5, 0, 6},
        {2, 3, 5, 6},
    };

    for (const auto& tc : cases) {
        const Real ref = compute_element(tc.mu, tc.nu, tc.lam, tc.sig);

        // Test all 8 permutations
        EXPECT_NEAR(compute_element(tc.nu, tc.mu, tc.lam, tc.sig), ref, ERI_TOL)
            << "Failed (nu mu | lam sig) for (" << tc.mu << " " << tc.nu
            << " | " << tc.lam << " " << tc.sig << ")";
        EXPECT_NEAR(compute_element(tc.mu, tc.nu, tc.sig, tc.lam), ref, ERI_TOL)
            << "Failed (mu nu | sig lam)";
        EXPECT_NEAR(compute_element(tc.nu, tc.mu, tc.sig, tc.lam), ref, ERI_TOL)
            << "Failed (nu mu | sig lam)";
        EXPECT_NEAR(compute_element(tc.lam, tc.sig, tc.mu, tc.nu), ref, ERI_TOL)
            << "Failed (lam sig | mu nu)";
        EXPECT_NEAR(compute_element(tc.sig, tc.lam, tc.mu, tc.nu), ref, ERI_TOL)
            << "Failed (sig lam | mu nu)";
        EXPECT_NEAR(compute_element(tc.lam, tc.sig, tc.nu, tc.mu), ref, ERI_TOL)
            << "Failed (lam sig | nu mu)";
        EXPECT_NEAR(compute_element(tc.sig, tc.lam, tc.nu, tc.mu), ref, ERI_TOL)
            << "Failed (sig lam | nu mu)";
    }
}

TEST_F(H2OERI, SchwarzInequality) {
    // |(mu nu | lam sig)| <= sqrt((mu nu | mu nu)) * sqrt((lam sig | lam sig))
    // Test for a sampling of quartets

    // Compute diagonal ERIs
    std::vector<std::vector<Real>> diag(7, std::vector<Real>(7));
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            diag[i][j] = compute_element(i, j, i, j);
        }
    }

    // Check Schwarz for all quartets in a subset
    for (int mu = 0; mu < 7; ++mu) {
        for (int nu = 0; nu < 7; ++nu) {
            for (int lam = 0; lam < 7; ++lam) {
                for (int sig = 0; sig < 7; ++sig) {
                    Real eri = compute_element(mu, nu, lam, sig);
                    Real bound = std::sqrt(std::abs(diag[mu][nu])) *
                                 std::sqrt(std::abs(diag[lam][sig]));
                    EXPECT_LE(std::abs(eri), bound + 1e-12)
                        << "Schwarz violated for (" << mu << " " << nu
                        << " | " << lam << " " << sig << ")";
                }
            }
        }
    }
}

// =============================================================================
// FockBuilder Tests
// =============================================================================

TEST(FockBuilderTest, H2FockMatrix) {
    auto basis = make_h2_sto3g();
    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(nbf, 2u);

    Engine engine(basis);
    Operator coulomb = Operator::coulomb();

    // Compute one-electron integrals for density matrix setup
    // Use identity density for a simple test
    std::vector<Real> D(nbf * nbf, 0.0);
    D[0] = 1.0;  // Just occupy orbital 0
    D[3] = 1.0;  // D(1,1) = 1

    FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    // Use compute_and_consume to build J and K
    engine.compute_and_consume(coulomb, fock);

    // J and K should be non-zero
    auto J = fock.get_coulomb_matrix();
    auto K = fock.get_exchange_matrix();

    EXPECT_GT(std::abs(J[0]), 0.0) << "J(0,0) should be non-zero";
    EXPECT_GT(std::abs(K[0]), 0.0) << "K(0,0) should be non-zero";

    // J should be symmetric
    EXPECT_NEAR(J[0 * nbf + 1], J[1 * nbf + 0], 1e-12) << "J not symmetric";
    // K should be symmetric
    EXPECT_NEAR(K[0 * nbf + 1], K[1 * nbf + 0], 1e-12) << "K not symmetric";
}

// =============================================================================
// Engine Two-Electron Tests
// =============================================================================

TEST(EngineTest, ComputeAndConsumeBasic) {
    auto basis = make_h2_sto3g();
    Engine engine(basis);
    Operator coulomb = Operator::coulomb();

    const Size nbf = basis.n_basis_functions();

    // Use a simple density matrix
    std::vector<Real> D(nbf * nbf, 0.0);
    D[0] = 1.0;
    D[nbf + 1] = 1.0;

    FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    EXPECT_NO_THROW(engine.compute_and_consume(coulomb, fock));

    // Verify J matrix has reasonable values
    auto J = fock.get_coulomb_matrix();
    // J(0,0) = sum_lam,sig (0 0 | lam sig) * D(lam,sig)
    // With our diagonal density, J(0,0) = (00|00)*D(0,0) + (00|11)*D(1,1)
    Real expected_J00 = 0.7746059439198978 * 1.0 + 0.5699948826767758 * 1.0;
    EXPECT_NEAR(J[0], expected_J00, 1e-8);
}

TEST(EngineTest, Compute2EShellQuartetDispatch) {
    auto basis = make_h2o_sto3g();
    Engine engine(basis);

    TwoElectronBuffer<0> buffer;

    // Test Coulomb dispatch
    Operator coulomb = Operator::coulomb();
    EXPECT_NO_THROW(engine.compute_2e_shell_quartet(
        coulomb, basis.shell(0), basis.shell(0),
        basis.shell(0), basis.shell(0), buffer));

    // (1s_O 1s_O | 1s_O 1s_O) = known value
    EXPECT_NEAR(buffer(0, 0, 0, 0), 4.785065404705503e+00, ERI_TOL);
}

// =============================================================================
// Built-in Basis Set Tests
// =============================================================================

TEST(BuiltinBasisTest, STO3GH2OShellCount) {
    auto basis = make_h2o_sto3g();
    // H2O/STO-3G: O has 3 shells (1s, 2s, 2p), each H has 1 shell (1s)
    EXPECT_EQ(basis.n_shells(), 5u);
    EXPECT_EQ(basis.n_basis_functions(), 7u);
}

TEST(BuiltinBasisTest, STO3GCH4ShellCount) {
    std::vector<Atom> atoms = {
        {6, {0.0, 0.0, 0.0}},
        {1, {1.18585, 1.18585, 1.18585}},
        {1, {1.18585, -1.18585, -1.18585}},
        {1, {-1.18585, 1.18585, -1.18585}},
        {1, {-1.18585, -1.18585, 1.18585}},
    };
    auto basis = create_sto3g(atoms);
    // CH4/STO-3G: C has 3 shells (1s, 2s, 2p), each H has 1 shell (1s)
    EXPECT_EQ(basis.n_shells(), 7u);
    // C: 1+1+3=5, 4*H: 4*1=4, total: 9
    EXPECT_EQ(basis.n_basis_functions(), 9u);
}

TEST(BuiltinBasisTest, CreateBuiltinBasis) {
    std::vector<Atom> atoms = {{1, {0.0, 0.0, 0.0}}, {1, {0.0, 0.0, 1.4}}};

    auto basis = create_builtin_basis("sto-3g", atoms);
    EXPECT_EQ(basis.n_shells(), 2u);
    EXPECT_EQ(basis.n_basis_functions(), 2u);

    // Case insensitive
    auto basis2 = create_builtin_basis("STO-3G", atoms);
    EXPECT_EQ(basis2.n_shells(), 2u);

    // Unknown basis should throw
    EXPECT_THROW(create_builtin_basis("unknown-basis", atoms), InvalidArgumentException);
}
