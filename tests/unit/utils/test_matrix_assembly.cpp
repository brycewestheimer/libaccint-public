// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/utils/matrix_assembly.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/utils/error_handling.hpp>
#include <gtest/gtest.h>

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

// =============================================================================
// Vector Overload Tests
// =============================================================================

TEST(MatrixAssemblyTest, OverlapMatrixVectorOverload) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    OneElectronOperator op = Operator::overlap();
    std::vector<Real> S = assemble_one_electron_matrix(engine, op);

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

TEST(MatrixAssemblyTest, KineticMatrixVectorOverload) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    OneElectronOperator op = Operator::kinetic();
    std::vector<Real> T = assemble_one_electron_matrix(engine, op);

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

TEST(MatrixAssemblyTest, NuclearMatrixVectorOverload) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    auto charges = make_h2o_charges();
    OneElectronOperator op = Operator::nuclear(charges);
    std::vector<Real> V = assemble_one_electron_matrix(engine, op);

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

// =============================================================================
// Span Overload Tests
// =============================================================================

TEST(MatrixAssemblyTest, OverlapMatrixSpanOverload) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Size nbf = basis.n_basis_functions();
    std::vector<Real> S(nbf * nbf);

    OneElectronOperator op = Operator::overlap();
    assemble_one_electron_matrix(engine, op, std::span(S));

    ASSERT_EQ(S.size(), 49u);  // 7 x 7

    // Diagonal elements should be 1.0 (normalized shells)
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_NEAR(S[i * nbf + i], 1.0, TIGHT_TOL)
            << "S(" << i << "," << i << ") should be 1.0";
    }

    // Matrix should be symmetric
    expect_symmetric(S, nbf, TIGHT_TOL, "Overlap");
}

TEST(MatrixAssemblyTest, KineticMatrixSpanOverload) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Size nbf = basis.n_basis_functions();
    std::vector<Real> T(nbf * nbf);

    OneElectronOperator op = Operator::kinetic();
    assemble_one_electron_matrix(engine, op, std::span(T));

    ASSERT_EQ(T.size(), 49u);

    // Diagonal elements should be positive
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GT(T[i * nbf + i], 0.0)
            << "T(" << i << "," << i << ") should be positive";
    }

    // Matrix should be symmetric
    expect_symmetric(T, nbf, TIGHT_TOL, "Kinetic");
}

// =============================================================================
// Property Tests
// =============================================================================

TEST(MatrixAssemblyTest, KineticMatrixPositiveDiagonal) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    OneElectronOperator op = Operator::kinetic();
    std::vector<Real> T = assemble_one_electron_matrix(engine, op);

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(nbf, 7u);

    // All diagonal elements must be positive
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GT(T[i * nbf + i], 0.0)
            << "Kinetic diagonal T(" << i << "," << i << ") must be positive";
    }

    // Matrix must be symmetric
    expect_symmetric(T, nbf, TIGHT_TOL, "Kinetic");
}

TEST(MatrixAssemblyTest, NuclearMatrixNegativeDiagonal) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    auto charges = make_h2o_charges();
    OneElectronOperator op = Operator::nuclear(charges);
    std::vector<Real> V = assemble_one_electron_matrix(engine, op);

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(nbf, 7u);

    // All diagonal elements must be negative (attractive)
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_LT(V[i * nbf + i], 0.0)
            << "Nuclear diagonal V(" << i << "," << i << ") must be negative";
    }

    // Matrix must be symmetric
    expect_symmetric(V, nbf, TIGHT_TOL, "Nuclear");
}

TEST(MatrixAssemblyTest, ComposedOperatorHcore) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    auto charges = make_h2o_charges();
    const Size nbf = basis.n_basis_functions();

    // Compute T and V separately
    std::vector<Real> T = assemble_one_electron_matrix(engine, Operator::kinetic());
    std::vector<Real> V = assemble_one_electron_matrix(engine, Operator::nuclear(charges));

    // Compute H_core = T + V using composed operator
    OneElectronOperator h_core = Operator::kinetic();
    h_core.add(Operator::nuclear(charges));
    std::vector<Real> H = assemble_one_electron_matrix(engine, h_core);

    ASSERT_EQ(H.size(), nbf * nbf);

    // H_core should be T + V
    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(H[i], T[i] + V[i], TIGHT_TOL)
            << "H_core[" << i << "] should equal T + V";
    }
}

TEST(MatrixAssemblyTest, SymmetryVerification) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Size nbf = basis.n_basis_functions();

    // Test overlap
    {
        std::vector<Real> S = assemble_one_electron_matrix(engine, Operator::overlap());
        expect_symmetric(S, nbf, TIGHT_TOL, "Overlap");
    }

    // Test kinetic
    {
        std::vector<Real> T = assemble_one_electron_matrix(engine, Operator::kinetic());
        expect_symmetric(T, nbf, TIGHT_TOL, "Kinetic");
    }

    // Test nuclear
    {
        auto charges = make_h2o_charges();
        std::vector<Real> V = assemble_one_electron_matrix(engine, Operator::nuclear(charges));
        expect_symmetric(V, nbf, TIGHT_TOL, "Nuclear");
    }
}

TEST(MatrixAssemblyTest, ScaleFactorWorks) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Size nbf = basis.n_basis_functions();

    // Compute S
    std::vector<Real> S = assemble_one_electron_matrix(engine, Operator::overlap());

    // Compute 2*S using scale factor
    OneElectronOperator scaled_op = 2.0 * OneElectronOperator(Operator::overlap());
    std::vector<Real> S2 = assemble_one_electron_matrix(engine, scaled_op);

    ASSERT_EQ(S2.size(), nbf * nbf);

    // 2*S should be exactly twice S
    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(S2[i], 2.0 * S[i], TIGHT_TOL)
            << "2*S[" << i << "] should equal 2 * S";
    }
}

// =============================================================================
// Validation Tests
// =============================================================================

TEST(MatrixAssemblyTest, SpanSizeValidation) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(nbf, 7u);

    // Correct size (49) should work
    {
        std::vector<Real> correct_matrix(nbf * nbf);
        EXPECT_NO_THROW(
            assemble_one_electron_matrix(engine, Operator::overlap(), std::span(correct_matrix))
        );
    }

    // Wrong size (too small) should throw
    {
        std::vector<Real> too_small(nbf * nbf - 1);
        EXPECT_THROW(
            assemble_one_electron_matrix(engine, Operator::overlap(), std::span(too_small)),
            InvalidArgumentException
        );
    }

    // Wrong size (too large) should throw
    {
        std::vector<Real> too_large(nbf * nbf + 1);
        EXPECT_THROW(
            assemble_one_electron_matrix(engine, Operator::overlap(), std::span(too_large)),
            InvalidArgumentException
        );
    }

    // Empty span should throw (for non-empty basis)
    {
        std::vector<Real> empty;
        EXPECT_THROW(
            assemble_one_electron_matrix(engine, Operator::overlap(), std::span(empty)),
            InvalidArgumentException
        );
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST(MatrixAssemblyTest, SingleShellBasis) {
    // Create a single s-shell basis (1 basis function)
    std::vector<Shell> shells;
    Shell s(0, Point3D{0.0, 0.0, 0.0},
            {1.0}, {1.0});
    s.set_atom_index(0);
    shells.push_back(std::move(s));

    BasisSet basis(std::move(shells));
    Engine engine(basis);

    ASSERT_EQ(basis.n_basis_functions(), 1u);

    // Test vector overload
    {
        std::vector<Real> S = assemble_one_electron_matrix(engine, Operator::overlap());
        ASSERT_EQ(S.size(), 1u);
        EXPECT_NEAR(S[0], 1.0, TIGHT_TOL)
            << "Single shell self-overlap should be 1.0";
    }

    // Test span overload
    {
        std::vector<Real> S(1);
        assemble_one_electron_matrix(engine, Operator::overlap(), std::span(S));
        EXPECT_NEAR(S[0], 1.0, TIGHT_TOL)
            << "Single shell self-overlap should be 1.0";
    }
}

TEST(MatrixAssemblyTest, EmptyBasis) {
    BasisSet empty_basis;
    Engine engine(empty_basis);

    ASSERT_EQ(engine.basis().n_basis_functions(), 0u);

    // Test vector overload
    {
        std::vector<Real> result = assemble_one_electron_matrix(engine, Operator::overlap());
        EXPECT_EQ(result.size(), 0u)
            << "Empty basis should give 0-sized result";
    }

    // Test span overload (empty span is valid for empty basis)
    {
        std::vector<Real> result;
        EXPECT_NO_THROW(
            assemble_one_electron_matrix(engine, Operator::overlap(), std::span(result))
        );
    }
}

// =============================================================================
// Consistency Tests
// =============================================================================

TEST(MatrixAssemblyTest, VectorAndSpanOverloadsMatch) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Size nbf = basis.n_basis_functions();

    OneElectronOperator op = Operator::overlap();

    // Compute using vector overload
    std::vector<Real> S_vector = assemble_one_electron_matrix(engine, op);

    // Compute using span overload
    std::vector<Real> S_span(nbf * nbf);
    assemble_one_electron_matrix(engine, op, std::span(S_span));

    // Results should be identical
    ASSERT_EQ(S_vector.size(), S_span.size());
    for (Size i = 0; i < S_vector.size(); ++i) {
        EXPECT_DOUBLE_EQ(S_vector[i], S_span[i])
            << "Vector and span overloads should give identical results at index " << i;
    }
}

TEST(MatrixAssemblyTest, ConsistencyWithEngineCompute1e) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);

    const Size nbf = basis.n_basis_functions();

    OneElectronOperator op = Operator::kinetic();

    // Compute using matrix_assembly utility
    std::vector<Real> T_util = assemble_one_electron_matrix(engine, op);

    // Compute using Engine::compute_1e directly
    std::vector<Real> T_engine;
    engine.compute_1e(op, T_engine);

    // Results should be identical
    ASSERT_EQ(T_util.size(), T_engine.size());
    for (Size i = 0; i < T_util.size(); ++i) {
        EXPECT_DOUBLE_EQ(T_util[i], T_engine[i])
            << "Utility and engine results should be identical at index " << i;
    }
}
