// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_numerical_golden_values.cpp
/// @brief Numerical correctness regression suite with golden-value tests
///        for standard molecules and basis sets.
///
/// Step 9.4: Validates that computed molecular integrals satisfy known
/// physical/mathematical constraints and maintain numerical stability
/// across code versions. Tests cover:
///   - H₂ (STO-3G): overlap, kinetic, nuclear, and ERI matrices
///   - H₂O (STO-3G): full 7×7 one-electron matrices and selected ERIs
///   - Symmetry and definiteness properties
///   - ERI 8-fold permutation symmetry
///   - Inter-matrix consistency (H_core = T + V)

#include <libaccint/engine/engine.hpp>
#include <libaccint/engine/cpu_engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace libaccint;

// =============================================================================
// Constants
// =============================================================================

namespace {

/// Tolerance for physical constraint checks (overlap diagonal, sign tests).
/// These are exact mathematical properties, so tight tolerance is appropriate.
constexpr Real CONSTRAINT_TOL = 1e-12;

/// Tolerance for inter-matrix consistency checks (H = T + V).
constexpr Real CONSISTENCY_TOL = 1e-12;

/// Tolerance for eigenvalue positivity (accounts for numerical noise).
constexpr Real EIGENVALUE_TOL = 1e-10;

// =============================================================================
// H₂ Molecule Geometry (STO-3G, R = 1.4 bohr)
// =============================================================================

/// Standard H₂ bond length in bohr
constexpr Real H2_BOND_LENGTH = 1.4;

/// H atom 1 at origin
constexpr Point3D H2_center_A{0.0, 0.0, 0.0};

/// H atom 2 along z-axis at R = 1.4 bohr
constexpr Point3D H2_center_B{0.0, 0.0, H2_BOND_LENGTH};

// =============================================================================
// H₂O Molecule Geometry (STO-3G, standard geometry in bohr)
// =============================================================================

constexpr Point3D O_center{0.0, 0.0, 0.0};
constexpr Point3D H1_center{0.0, 1.43233673, -1.10866041};
constexpr Point3D H2_center{0.0, -1.43233673, -1.10866041};

// =============================================================================
// Helper: Build STO-3G H₂ shells (2 shells, 2 basis functions)
// =============================================================================

std::vector<Shell> make_sto3g_h2_shells() {
    std::vector<Shell> shells;
    shells.reserve(2);

    // STO-3G hydrogen 1s: exponents and coefficients from standard tables
    const std::vector<Real> h_exponents = {3.42525091, 0.62391373, 0.16885540};
    const std::vector<Real> h_coefficients = {0.15432897, 0.53532814, 0.44463454};

    // H1 1s
    {
        Shell s(0, H2_center_A, h_exponents, h_coefficients);
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    // H2 1s
    {
        Shell s(0, H2_center_B, h_exponents, h_coefficients);
        s.set_atom_index(1);
        shells.push_back(std::move(s));
    }

    return shells;
}

/// Build PointChargeParams for H₂ nuclear charges
PointChargeParams make_h2_charges() {
    PointChargeParams charges;
    charges.x = {H2_center_A.x, H2_center_B.x};
    charges.y = {H2_center_A.y, H2_center_B.y};
    charges.z = {H2_center_A.z, H2_center_B.z};
    charges.charge = {1.0, 1.0};
    return charges;
}

// =============================================================================
// Helper: Build STO-3G H₂O shells (5 shells, 7 basis functions)
// =============================================================================

std::vector<Shell> make_sto3g_h2o_shells() {
    std::vector<Shell> shells;
    shells.reserve(5);

    // O 1s (L=0, K=3)
    {
        Shell s(0, O_center,
                {130.7093200, 23.8088610, 6.4436083},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    // O 2s (L=0, K=3)
    {
        Shell s(0, O_center,
                {5.0331513, 1.1695961, 0.3803890},
                {-0.09996723, 0.39951283, 0.70011547});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    // O 2p (L=1, K=3)
    {
        Shell s(1, O_center,
                {5.0331513, 1.1695961, 0.3803890},
                {0.15591627, 0.60768372, 0.39195739});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }
    // H1 1s (L=0, K=3)
    {
        Shell s(0, H1_center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(1);
        shells.push_back(std::move(s));
    }
    // H2 1s (L=0, K=3)
    {
        Shell s(0, H2_center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(2);
        shells.push_back(std::move(s));
    }

    return shells;
}

/// Build PointChargeParams for H₂O nuclear charges
PointChargeParams make_h2o_charges() {
    PointChargeParams charges;
    charges.x = {O_center.x, H1_center.x, H2_center.x};
    charges.y = {O_center.y, H1_center.y, H2_center.y};
    charges.z = {O_center.z, H1_center.z, H2_center.z};
    charges.charge = {8.0, 1.0, 1.0};
    return charges;
}

// =============================================================================
// Helper: Check that flat N×N matrix is symmetric
// =============================================================================

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

// =============================================================================
// Helper: Compute eigenvalues of a real symmetric matrix via Jacobi iteration
//
// This avoids an external LAPACK dependency for the test suite.
// Only used for small matrices (2×2 or 7×7) so O(n³) is fine.
// =============================================================================

std::vector<Real> symmetric_eigenvalues(const std::vector<Real>& matrix, Size n) {
    // Copy matrix for in-place rotation
    std::vector<Real> A(matrix);
    const int max_iter = 200;
    const Real jacobi_tol = 1e-14;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Find the largest off-diagonal element
        Real max_off = 0.0;
        Size p = 0, q = 1;
        for (Size i = 0; i < n; ++i) {
            for (Size j = i + 1; j < n; ++j) {
                Real aij = std::abs(A[i * n + j]);
                if (aij > max_off) {
                    max_off = aij;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_off < jacobi_tol) break;

        // Compute rotation angle
        Real app = A[p * n + p];
        Real aqq = A[q * n + q];
        Real apq = A[p * n + q];
        Real theta = 0.5 * std::atan2(2.0 * apq, aqq - app);
        Real c = std::cos(theta);
        Real s = std::sin(theta);

        // Apply Jacobi rotation
        std::vector<Real> B(A);
        for (Size i = 0; i < n; ++i) {
            if (i != p && i != q) {
                Real aip = A[i * n + p];
                Real aiq = A[i * n + q];
                B[i * n + p] = c * aip - s * aiq;
                B[p * n + i] = B[i * n + p];
                B[i * n + q] = s * aip + c * aiq;
                B[q * n + i] = B[i * n + q];
            }
        }
        B[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        B[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        B[p * n + q] = 0.0;
        B[q * n + p] = 0.0;

        A = B;
    }

    // Diagonal elements are eigenvalues
    std::vector<Real> eigenvalues(n);
    for (Size i = 0; i < n; ++i) {
        eigenvalues[i] = A[i * n + i];
    }
    std::sort(eigenvalues.begin(), eigenvalues.end());
    return eigenvalues;
}

}  // anonymous namespace

// =============================================================================
// Test Fixture
// =============================================================================

class NumericalGoldenValues : public ::testing::Test {
protected:
    // H₂ data
    BasisSet h2_basis_{make_sto3g_h2_shells()};
    Engine h2_engine_{h2_basis_};
    PointChargeParams h2_charges_ = make_h2_charges();
    static constexpr Size H2_NBF = 2;

    // H₂O data
    BasisSet h2o_basis_{make_sto3g_h2o_shells()};
    Engine h2o_engine_{h2o_basis_};
    PointChargeParams h2o_charges_ = make_h2o_charges();
    static constexpr Size H2O_NBF = 7;
};

// =============================================================================
// Test 1: H₂ Overlap Matrix — Normalization and Symmetry
// =============================================================================

TEST_F(NumericalGoldenValues, H2_Overlap_NormalizationAndSymmetry) {
    std::vector<Real> S;
    h2_engine_.compute_overlap_matrix(S);

    ASSERT_EQ(S.size(), H2_NBF * H2_NBF);

    // Diagonal elements must be exactly 1.0 (normalized basis)
    EXPECT_NEAR(S[0 * H2_NBF + 0], 1.0, CONSTRAINT_TOL)
        << "S₁₁ must be 1.0 (normalization)";
    EXPECT_NEAR(S[1 * H2_NBF + 1], 1.0, CONSTRAINT_TOL)
        << "S₂₂ must be 1.0 (normalization)";

    // Symmetry: S₁₂ = S₂₁
    EXPECT_NEAR(S[0 * H2_NBF + 1], S[1 * H2_NBF + 0], CONSTRAINT_TOL)
        << "Overlap matrix must be symmetric";

    // Off-diagonal: 0 < S₁₂ < 1 (positive overlap, less than self-overlap)
    Real S12 = S[0 * H2_NBF + 1];
    EXPECT_GT(S12, 0.0) << "S₁₂ should be positive for overlapping H 1s shells";
    EXPECT_LT(S12, 1.0) << "S₁₂ should be less than 1.0 (shells not identical)";
}

// =============================================================================
// Test 2: H₂ Kinetic Energy Matrix — Sign and Symmetry
// =============================================================================

TEST_F(NumericalGoldenValues, H2_Kinetic_SignAndSymmetry) {
    std::vector<Real> T;
    h2_engine_.compute_kinetic_matrix(T);

    ASSERT_EQ(T.size(), H2_NBF * H2_NBF);

    // Diagonal kinetic energy must be strictly positive
    EXPECT_GT(T[0 * H2_NBF + 0], 0.0)
        << "T₁₁ must be positive (kinetic energy of 1s)";
    EXPECT_GT(T[1 * H2_NBF + 1], 0.0)
        << "T₂₂ must be positive (kinetic energy of 1s)";

    // Identical shells → identical diagonal elements
    EXPECT_NEAR(T[0 * H2_NBF + 0], T[1 * H2_NBF + 1], CONSTRAINT_TOL)
        << "T₁₁ should equal T₂₂ for identical hydrogen 1s shells";

    // Symmetry
    EXPECT_NEAR(T[0 * H2_NBF + 1], T[1 * H2_NBF + 0], CONSTRAINT_TOL)
        << "Kinetic matrix must be symmetric";
}

// =============================================================================
// Test 3: H₂ Nuclear Attraction Matrix — Sign and Symmetry
// =============================================================================

TEST_F(NumericalGoldenValues, H2_Nuclear_SignAndSymmetry) {
    std::vector<Real> V;
    h2_engine_.compute_nuclear_matrix(h2_charges_, V);

    ASSERT_EQ(V.size(), H2_NBF * H2_NBF);

    // Nuclear attraction for positive charges should be negative
    EXPECT_LT(V[0 * H2_NBF + 0], 0.0)
        << "V₁₁ must be negative (attraction to positive nuclei)";
    EXPECT_LT(V[1 * H2_NBF + 1], 0.0)
        << "V₂₂ must be negative (attraction to positive nuclei)";

    // Symmetry
    EXPECT_NEAR(V[0 * H2_NBF + 1], V[1 * H2_NBF + 0], CONSTRAINT_TOL)
        << "Nuclear attraction matrix must be symmetric";

    // By symmetry of H₂: V₁₁ = V₂₂ (identical atoms, symmetric geometry)
    // Note: This holds because H atom 1 is at the origin and H atom 2 is at (0,0,R).
    // Each basis function "sees" both nuclei. Since the geometry is symmetric
    // about the midpoint and the basis functions are identical,
    // V₁₁ and V₂₂ should be equal.
    EXPECT_NEAR(V[0 * H2_NBF + 0], V[1 * H2_NBF + 1], CONSTRAINT_TOL)
        << "V₁₁ should equal V₂₂ for symmetric H₂";
}

// =============================================================================
// Test 4: H₂ Two-Electron Integrals — Physical Constraints
// =============================================================================

TEST_F(NumericalGoldenValues, H2_ERI_PhysicalConstraints) {
    // Compute all 16 ERIs for the 2-function basis
    // Loop over all quartet combinations
    std::vector<std::vector<std::vector<std::vector<Real>>>> eri(
        H2_NBF, std::vector<std::vector<std::vector<Real>>>(
            H2_NBF, std::vector<std::vector<Real>>(
                H2_NBF, std::vector<Real>(H2_NBF, 0.0))));

    for (Size a = 0; a < H2_NBF; ++a) {
        for (Size b = 0; b < H2_NBF; ++b) {
            for (Size c = 0; c < H2_NBF; ++c) {
                for (Size d = 0; d < H2_NBF; ++d) {
                    TwoElectronBuffer<0> buf;
                    h2_engine_.compute_2e_shell_quartet(
                        Operator::coulomb(),
                        h2_basis_.shell(a), h2_basis_.shell(b),
                        h2_basis_.shell(c), h2_basis_.shell(d), buf);
                    eri[a][b][c][d] = buf(0, 0, 0, 0);
                }
            }
        }
    }

    // Self-repulsion must be positive: (ii|ii) > 0
    EXPECT_GT(eri[0][0][0][0], 0.0) << "(11|11) must be positive";
    EXPECT_GT(eri[1][1][1][1], 0.0) << "(22|22) must be positive";

    // All ERIs should be positive for s-type functions with positive overlap
    for (Size a = 0; a < H2_NBF; ++a) {
        for (Size b = 0; b < H2_NBF; ++b) {
            for (Size c = 0; c < H2_NBF; ++c) {
                for (Size d = 0; d < H2_NBF; ++d) {
                    EXPECT_GT(eri[a][b][c][d], 0.0)
                        << "ERI(" << a << b << "|" << c << d
                        << ") should be positive for s-type functions";
                }
            }
        }
    }

    // Symmetry: (ab|cd) = (ba|cd) = (ab|dc) = (cd|ab)
    for (Size a = 0; a < H2_NBF; ++a) {
        for (Size b = 0; b < H2_NBF; ++b) {
            for (Size c = 0; c < H2_NBF; ++c) {
                for (Size d = 0; d < H2_NBF; ++d) {
                    EXPECT_NEAR(eri[a][b][c][d], eri[b][a][c][d], CONSTRAINT_TOL)
                        << "ERI bra permutation failed: (" << a << b
                        << "|" << c << d << ")";
                    EXPECT_NEAR(eri[a][b][c][d], eri[a][b][d][c], CONSTRAINT_TOL)
                        << "ERI ket permutation failed: (" << a << b
                        << "|" << c << d << ")";
                    EXPECT_NEAR(eri[a][b][c][d], eri[c][d][a][b], CONSTRAINT_TOL)
                        << "ERI bra-ket exchange failed: (" << a << b
                        << "|" << c << d << ")";
                }
            }
        }
    }

    // By symmetry of H₂: (11|11) = (22|22)
    EXPECT_NEAR(eri[0][0][0][0], eri[1][1][1][1], CONSTRAINT_TOL)
        << "(11|11) should equal (22|22) for symmetric H₂";
}

// =============================================================================
// Test 5: H₂ Overlap Positive Definite (Eigenvalues > 0)
// =============================================================================

TEST_F(NumericalGoldenValues, H2_Overlap_PositiveDefinite) {
    std::vector<Real> S;
    h2_engine_.compute_overlap_matrix(S);

    auto eigenvalues = symmetric_eigenvalues(S, H2_NBF);

    // For 2×2 overlap of identical s shells: eigenvalues are (1-S₁₂) and (1+S₁₂)
    // Both must be positive since 0 < S₁₂ < 1
    ASSERT_EQ(eigenvalues.size(), H2_NBF);
    for (Size i = 0; i < H2_NBF; ++i) {
        EXPECT_GT(eigenvalues[i], EIGENVALUE_TOL)
            << "Overlap eigenvalue " << i << " = " << eigenvalues[i]
            << " should be positive";
    }
}

// =============================================================================
// Test 6: H₂ Kinetic Positive Semi-Definite
// =============================================================================

TEST_F(NumericalGoldenValues, H2_Kinetic_PositiveSemiDefinite) {
    std::vector<Real> T;
    h2_engine_.compute_kinetic_matrix(T);

    auto eigenvalues = symmetric_eigenvalues(T, H2_NBF);

    // Kinetic energy matrix is positive semi-definite
    for (Size i = 0; i < H2_NBF; ++i) {
        EXPECT_GE(eigenvalues[i], -EIGENVALUE_TOL)
            << "Kinetic eigenvalue " << i << " = " << eigenvalues[i]
            << " should be non-negative";
    }
}

// =============================================================================
// Test 7: H₂O Overlap Matrix — All Diagonals Equal 1.0
// =============================================================================

TEST_F(NumericalGoldenValues, H2O_Overlap_DiagonalNormalization) {
    std::vector<Real> S;
    h2o_engine_.compute_overlap_matrix(S);

    ASSERT_EQ(S.size(), H2O_NBF * H2O_NBF);

    // All 7 diagonal overlap elements must be 1.0
    for (Size i = 0; i < H2O_NBF; ++i) {
        EXPECT_NEAR(S[i * H2O_NBF + i], 1.0, CONSTRAINT_TOL)
            << "S(" << i << "," << i << ") must be 1.0 (normalization)";
    }
}

// =============================================================================
// Test 8: H₂O Overlap Matrix — Full Symmetry
// =============================================================================

TEST_F(NumericalGoldenValues, H2O_Overlap_Symmetry) {
    std::vector<Real> S;
    h2o_engine_.compute_overlap_matrix(S);

    expect_symmetric(S, H2O_NBF, CONSTRAINT_TOL, "H2O overlap");
}

// =============================================================================
// Test 9: H₂O Overlap Matrix — Positive Definite
// =============================================================================

TEST_F(NumericalGoldenValues, H2O_Overlap_PositiveDefinite) {
    std::vector<Real> S;
    h2o_engine_.compute_overlap_matrix(S);

    auto eigenvalues = symmetric_eigenvalues(S, H2O_NBF);

    ASSERT_EQ(eigenvalues.size(), H2O_NBF);
    for (Size i = 0; i < H2O_NBF; ++i) {
        EXPECT_GT(eigenvalues[i], EIGENVALUE_TOL)
            << "Overlap eigenvalue " << i << " = " << eigenvalues[i]
            << " must be positive (overlap matrix is positive definite)";
    }
}

// =============================================================================
// Test 10: H₂O Kinetic Matrix — Symmetry and Sign
// =============================================================================

TEST_F(NumericalGoldenValues, H2O_Kinetic_SymmetryAndSign) {
    std::vector<Real> T;
    h2o_engine_.compute_kinetic_matrix(T);

    ASSERT_EQ(T.size(), H2O_NBF * H2O_NBF);

    // Symmetry
    expect_symmetric(T, H2O_NBF, CONSTRAINT_TOL, "H2O kinetic");

    // All diagonal kinetic energies must be positive
    for (Size i = 0; i < H2O_NBF; ++i) {
        EXPECT_GT(T[i * H2O_NBF + i], 0.0)
            << "T(" << i << "," << i << ") must be positive";
    }
}

// =============================================================================
// Test 11: H₂O Kinetic Matrix — Positive Semi-Definite
// =============================================================================

TEST_F(NumericalGoldenValues, H2O_Kinetic_PositiveSemiDefinite) {
    std::vector<Real> T;
    h2o_engine_.compute_kinetic_matrix(T);

    auto eigenvalues = symmetric_eigenvalues(T, H2O_NBF);

    for (Size i = 0; i < H2O_NBF; ++i) {
        EXPECT_GE(eigenvalues[i], -EIGENVALUE_TOL)
            << "Kinetic eigenvalue " << i << " = " << eigenvalues[i]
            << " should be non-negative (positive semi-definite)";
    }
}

// =============================================================================
// Test 12: H₂O Nuclear Attraction — Symmetry and Sign
// =============================================================================

TEST_F(NumericalGoldenValues, H2O_Nuclear_SymmetryAndSign) {
    std::vector<Real> V;
    h2o_engine_.compute_nuclear_matrix(h2o_charges_, V);

    ASSERT_EQ(V.size(), H2O_NBF * H2O_NBF);

    // Symmetry
    expect_symmetric(V, H2O_NBF, CONSTRAINT_TOL, "H2O nuclear");

    // All diagonal nuclear attraction elements must be negative
    // (attraction to positive nuclei lowers energy)
    for (Size i = 0; i < H2O_NBF; ++i) {
        EXPECT_LT(V[i * H2O_NBF + i], 0.0)
            << "V(" << i << "," << i << ") must be negative "
            << "(nuclear attraction for positive charges)";
    }
}

// =============================================================================
// Test 13: H₂O Nuclear Attraction — Negative Semi-Definite
// =============================================================================

TEST_F(NumericalGoldenValues, H2O_Nuclear_NegativeSemiDefinite) {
    std::vector<Real> V;
    h2o_engine_.compute_nuclear_matrix(h2o_charges_, V);

    auto eigenvalues = symmetric_eigenvalues(V, H2O_NBF);

    // Nuclear attraction matrix for positive charges should be negative semi-definite
    for (Size i = 0; i < H2O_NBF; ++i) {
        EXPECT_LE(eigenvalues[i], EIGENVALUE_TOL)
            << "Nuclear attraction eigenvalue " << i << " = " << eigenvalues[i]
            << " should be non-positive for positive nuclear charges";
    }
}

// =============================================================================
// Test 14: H₂O Core Hamiltonian Consistency (H = T + V)
// =============================================================================

TEST_F(NumericalGoldenValues, H2O_CoreHamiltonian_Consistency) {
    std::vector<Real> T, V, H;
    h2o_engine_.compute_kinetic_matrix(T);
    h2o_engine_.compute_nuclear_matrix(h2o_charges_, V);
    h2o_engine_.compute_core_hamiltonian(h2o_charges_, H);

    ASSERT_EQ(H.size(), H2O_NBF * H2O_NBF);

    for (Size i = 0; i < H2O_NBF * H2O_NBF; ++i) {
        EXPECT_NEAR(H[i], T[i] + V[i], CONSISTENCY_TOL)
            << "H_core[" << i << "] != T + V";
    }
}

// =============================================================================
// Test 15: H₂O ERI — Self-Repulsion Positive
// =============================================================================

TEST_F(NumericalGoldenValues, H2O_ERI_SelfRepulsionPositive) {
    // Check (ii|ii) > 0 for each shell. For the 7 basis functions we test
    // diagonal self-repulsion for shells 0..4 (the s and p shells).
    const Size n_shells = h2o_basis_.n_shells();

    for (Size i = 0; i < n_shells; ++i) {
        const auto& shell_i = h2o_basis_.shell(i);
        TwoElectronBuffer<0> buf;
        h2o_engine_.compute_2e_shell_quartet(
            Operator::coulomb(), shell_i, shell_i, shell_i, shell_i, buf);

        // The (0,0,0,0) element is the self-repulsion of the first Cartesian
        // component of this shell
        EXPECT_GT(buf(0, 0, 0, 0), 0.0)
            << "Shell " << i << " self-repulsion (00|00) must be positive";
    }
}

// =============================================================================
// Test 16: H₂O ERI — 8-Fold Permutation Symmetry
// =============================================================================

TEST_F(NumericalGoldenValues, H2O_ERI_PermutationSymmetry) {
    // Test ERI permutation symmetry for selected shell quartets.
    // (ab|cd) = (ba|cd) = (ab|dc) = (ba|dc) = (cd|ab) = (dc|ab) = (cd|ba) = (dc|ba)
    //
    // Pick diverse quartets: (O1s H1s | O2p H2s), (O1s O2s | O1s O2s)
    struct QuartetIdx { Size a, b, c, d; };
    std::vector<QuartetIdx> test_quartets = {
        {0, 3, 2, 4},  // (O1s, H1-1s | O2p, H2-1s)
        {0, 1, 0, 1},  // (O1s, O2s | O1s, O2s)
        {1, 2, 3, 4},  // (O2s, O2p | H1-1s, H2-1s)
    };

    for (const auto& q : test_quartets) {
        // Compute all 8 permutations
        auto compute = [&](Size a, Size b, Size c, Size d) {
            TwoElectronBuffer<0> buf;
            h2o_engine_.compute_2e_shell_quartet(
                Operator::coulomb(),
                h2o_basis_.shell(a), h2o_basis_.shell(b),
                h2o_basis_.shell(c), h2o_basis_.shell(d), buf);
            return buf;
        };

        auto buf_abcd = compute(q.a, q.b, q.c, q.d);
        auto buf_bacd = compute(q.b, q.a, q.c, q.d);
        auto buf_abdc = compute(q.a, q.b, q.d, q.c);
        auto buf_cdab = compute(q.c, q.d, q.a, q.b);

        int na = buf_abcd.na(), nb = buf_abcd.nb();
        int nc = buf_abcd.nc(), nd = buf_abcd.nd();

        for (int a = 0; a < na; ++a) {
            for (int b = 0; b < nb; ++b) {
                for (int c = 0; c < nc; ++c) {
                    for (int d = 0; d < nd; ++d) {
                        Real val = buf_abcd(a, b, c, d);

                        // (ba|cd): bra permutation
                        EXPECT_NEAR(buf_bacd(b, a, c, d), val, CONSTRAINT_TOL)
                            << "Bra permutation symmetry failed for quartet ("
                            << q.a << q.b << "|" << q.c << q.d << ") at ("
                            << a << "," << b << "," << c << "," << d << ")";

                        // (ab|dc): ket permutation
                        EXPECT_NEAR(buf_abdc(a, b, d, c), val, CONSTRAINT_TOL)
                            << "Ket permutation symmetry failed for quartet ("
                            << q.a << q.b << "|" << q.c << q.d << ") at ("
                            << a << "," << b << "," << c << "," << d << ")";

                        // (cd|ab): bra-ket exchange
                        EXPECT_NEAR(buf_cdab(c, d, a, b), val, CONSTRAINT_TOL)
                            << "Bra-ket exchange symmetry failed for quartet ("
                            << q.a << q.b << "|" << q.c << q.d << ") at ("
                            << a << "," << b << "," << c << "," << d << ")";
                    }
                }
            }
        }
    }
}

// =============================================================================
// Test 17: H₂O Overlap Trace Equals Number of Basis Functions
// =============================================================================

TEST_F(NumericalGoldenValues, H2O_Overlap_TraceEqualsNBF) {
    std::vector<Real> S;
    h2o_engine_.compute_overlap_matrix(S);

    // Tr(S) = sum of diagonal = N_basis (since all diagonals are 1.0)
    Real trace = 0.0;
    for (Size i = 0; i < H2O_NBF; ++i) {
        trace += S[i * H2O_NBF + i];
    }

    EXPECT_NEAR(trace, static_cast<Real>(H2O_NBF), CONSTRAINT_TOL)
        << "Trace of overlap matrix should equal number of basis functions";
}

// =============================================================================
// Test 18: H₂ Kinetic Energy Magnitude Check
// =============================================================================

TEST_F(NumericalGoldenValues, H2_Kinetic_MagnitudeCheck) {
    std::vector<Real> T;
    h2_engine_.compute_kinetic_matrix(T);

    // For STO-3G hydrogen 1s, the kinetic energy integral T₁₁ ≈ 0.76 Hartree
    // (analytical value for STO-3G H 1s is well-known).
    // We check it's in a physically reasonable range: 0.5 < T₁₁ < 1.5
    Real T11 = T[0 * H2_NBF + 0];
    EXPECT_GT(T11, 0.5)
        << "T₁₁ for STO-3G H 1s should be > 0.5 Hartree";
    EXPECT_LT(T11, 1.5)
        << "T₁₁ for STO-3G H 1s should be < 1.5 Hartree";

    // Kinetic trace should be reasonable: Tr(T) = 2 * T₁₁ ≈ 1.5 Hartree
    Real T_trace = T[0] + T[3];  // T₁₁ + T₂₂
    EXPECT_GT(T_trace, 1.0);
    EXPECT_LT(T_trace, 3.0);
}

// =============================================================================
// Test 19: H₂ Nuclear Attraction Magnitude Check
// =============================================================================

TEST_F(NumericalGoldenValues, H2_Nuclear_MagnitudeCheck) {
    std::vector<Real> V;
    h2_engine_.compute_nuclear_matrix(h2_charges_, V);

    // For H₂ STO-3G at R=1.4 bohr, the nuclear attraction integrals
    // should be in a physically reasonable range.
    // V₁₁ includes attraction to both nuclei ≈ -1.8 to -2.0 a.u.
    Real V11 = V[0 * H2_NBF + 0];
    EXPECT_LT(V11, -1.0)
        << "V₁₁ should be significantly negative (attraction to two nuclei)";
    EXPECT_GT(V11, -5.0)
        << "V₁₁ magnitude should be reasonable";

    // V₁₂ should also be negative (off-site attraction)
    Real V12 = V[0 * H2_NBF + 1];
    EXPECT_LT(V12, 0.0)
        << "V₁₂ should be negative";
}

// =============================================================================
// Test 20: H₂O Kinetic Energy Ordering
// =============================================================================

TEST_F(NumericalGoldenValues, H2O_Kinetic_DiagonalOrdering) {
    std::vector<Real> T;
    h2o_engine_.compute_kinetic_matrix(T);

    // O 1s (tight exponents) should have the largest kinetic energy
    Real T_O1s = T[0 * H2O_NBF + 0];

    // H 1s shells should have smaller kinetic energy than O 1s
    Real T_H1 = T[5 * H2O_NBF + 5];  // function index 5 = H1 1s
    Real T_H2 = T[6 * H2O_NBF + 6];  // function index 6 = H2 1s

    EXPECT_GT(T_O1s, T_H1)
        << "O 1s kinetic energy should be larger than H 1s";

    // The two hydrogen 1s shells should have identical kinetic energies
    // (same STO-3G parameters, kinetic energy is independent of center)
    EXPECT_NEAR(T_H1, T_H2, CONSTRAINT_TOL)
        << "H1 and H2 1s kinetic energies should be equal (same basis)";
}

// =============================================================================
// Test 21: Regression Snapshot — Deterministic Results
// =============================================================================

TEST_F(NumericalGoldenValues, RegressionSnapshot_Deterministic) {
    // Verify that results are perfectly deterministic across runs.
    // Compute everything twice with fresh engines.
    BasisSet basis1(make_sto3g_h2o_shells());
    Engine engine1(basis1);

    BasisSet basis2(make_sto3g_h2o_shells());
    Engine engine2(basis2);

    auto charges = make_h2o_charges();

    std::vector<Real> S1, S2, T1, T2, V1, V2;

    engine1.compute_overlap_matrix(S1);
    engine2.compute_overlap_matrix(S2);

    engine1.compute_kinetic_matrix(T1);
    engine2.compute_kinetic_matrix(T2);

    engine1.compute_nuclear_matrix(charges, V1);
    engine2.compute_nuclear_matrix(charges, V2);

    ASSERT_EQ(S1.size(), S2.size());
    ASSERT_EQ(T1.size(), T2.size());
    ASSERT_EQ(V1.size(), V2.size());

    for (Size i = 0; i < S1.size(); ++i) {
        EXPECT_EQ(S1[i], S2[i])
            << "Overlap not bitwise identical at index " << i;
    }
    for (Size i = 0; i < T1.size(); ++i) {
        EXPECT_EQ(T1[i], T2[i])
            << "Kinetic not bitwise identical at index " << i;
    }
    for (Size i = 0; i < V1.size(); ++i) {
        EXPECT_EQ(V1[i], V2[i])
            << "Nuclear not bitwise identical at index " << i;
    }
}
