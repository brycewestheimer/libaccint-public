// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_one_electron.cpp
/// @brief End-to-end integration tests for one-electron integrals (S, T, V)
///        validated against PySCF reference values.

#include <libaccint/engine/engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

using namespace libaccint;

// =============================================================================
// PySCF Reference Data
// =============================================================================
//
// Generated with PySCF using the following molecules and basis:
//   H2/STO-3G:  H at (0,0,0), H at (0,0,1.4) bohr
//   H2O/STO-3G: O at origin, H at (0,1.43233673,-1.10866041),
//                              H at (0,-1.43233673,-1.10866041) bohr
//
// Shell ordering: O_1s, O_2s, O_2p(x,y,z), H1_1s, H2_1s
// PySCF Cartesian p-shell order: px, py, pz (same as libaccint canonical)

namespace {

// =============================================================================
// Tolerance
// =============================================================================

/// Tolerance for element-by-element comparison against PySCF reference
constexpr Real REF_TOL = 1e-12;

/// Loose tolerance for symmetry checks (numerical noise at machine epsilon)
constexpr Real SYM_TOL = 1e-14;

// =============================================================================
// H2/STO-3G Reference Values (2x2 matrices)
// =============================================================================

// clang-format off

/// H2/STO-3G overlap matrix (row-major, 2x2)
constexpr Real H2_S_REF[4] = {
    1.0000000000000002,  0.6593182061348640,
    0.6593182061348640,  1.0000000000000002
};

/// H2/STO-3G kinetic energy matrix (row-major, 2x2)
constexpr Real H2_T_REF[4] = {
    0.7600318835666091,  0.2364546559796740,
    0.2364546559796740,  0.7600318835666091
};

/// H2/STO-3G nuclear attraction matrix (row-major, 2x2)
constexpr Real H2_V_REF[4] = {
    -1.8804408924734295, -1.1948346203692910,
    -1.1948346203692910, -1.8804408924734295
};

// =============================================================================
// H2O/STO-3G Reference Values (7x7 matrices)
// =============================================================================

/// H2O/STO-3G overlap matrix (row-major, 7x7)
constexpr Real H2O_S_REF[49] = {
    // Row 0: O 1s
     1.0000000000000000,  0.2367039365108476,  0.0000000000000000,  0.0000000000000000,  0.0000000000000000,  0.0538132217391523,  0.0538132217391523,
    // Row 1: O 2s
     0.2367039365108476,  1.0000000000000000,  0.0000000000000000,  0.0000000000000000,  0.0000000000000000,  0.4739635050688644,  0.4739635050688644,
    // Row 2: O 2px
     0.0000000000000000,  0.0000000000000000,  1.0000000000000000,  0.0000000000000000,  0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
    // Row 3: O 2py
     0.0000000000000000,  0.0000000000000000,  0.0000000000000000,  1.0000000000000000,  0.0000000000000000,  0.3107838556244360, -0.3107838556244360,
    // Row 4: O 2pz
     0.0000000000000000,  0.0000000000000000,  0.0000000000000000,  0.0000000000000000,  1.0000000000000000, -0.2405535999889969, -0.2405535999889969,
    // Row 5: H1 1s
     0.0538132217391523,  0.4739635050688644,  0.0000000000000000,  0.3107838556244360, -0.2405535999889970,  1.0000000000000002,  0.2509870880078567,
    // Row 6: H2 1s
     0.0538132217391523,  0.4739635050688644,  0.0000000000000000, -0.3107838556244360, -0.2405535999889970,  0.2509870880078567,  1.0000000000000002
};

/// H2O/STO-3G kinetic energy matrix (row-major, 7x7)
constexpr Real H2O_T_REF[49] = {
    // Row 0: O 1s
     2.9003199945539567e+01, -1.6801093931649205e-01,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00, -2.6014296930086927e-03, -2.6014296930086927e-03,
    // Row 1: O 2s
    -1.6801093931649186e-01,  8.0812795493034717e-01,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  1.2799347218268392e-01,  1.2799347218268392e-01,
    // Row 2: O 2px
     0.0000000000000000e+00,  0.0000000000000000e+00,  2.5287311981947633e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,
    // Row 3: O 2py
     0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  2.5287311981947633e+00,  0.0000000000000000e+00,  2.2402119406970927e-01, -2.2402119406970927e-01,
    // Row 4: O 2pz
     0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  2.5287311981947633e+00, -1.7339737483797782e-01, -1.7339737483797782e-01,
    // Row 5: H1 1s
    -2.6014296930087014e-03,  1.2799347218268389e-01,  0.0000000000000000e+00,  2.2402119406970933e-01, -1.7339737483797782e-01,  7.6003188356660911e-01,  8.3215971782915881e-03,
    // Row 6: H2 1s
    -2.6014296930087014e-03,  1.2799347218268389e-01,  0.0000000000000000e+00, -2.2402119406970933e-01, -1.7339737483797782e-01,  8.3215971782915881e-03,  7.6003188356660911e-01
};

/// H2O/STO-3G nuclear attraction matrix (row-major, 7x7)
constexpr Real H2O_V_REF[49] = {
    // Row 0: O 1s
    -6.1722649584460683e+01, -7.4444466718942088e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  1.8952661667439041e-02, -1.7415712587102528e+00, -1.7415712587102528e+00,
    // Row 1: O 2s
    -7.4444466718942115e+00, -1.0141609588390519e+01,  0.0000000000000000e+00,  0.0000000000000000e+00,  2.2303577583430578e-01, -3.8611015296291202e+00, -3.8611015296291202e+00,
    // Row 2: O 2px
     0.0000000000000000e+00,  0.0000000000000000e+00, -9.9852946107476654e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00,
    // Row 3: O 2py
     0.0000000000000000e+00,  0.0000000000000000e+00,  0.0000000000000000e+00, -1.0141154304519693e+01,  0.0000000000000000e+00, -2.2510741842099091e+00,  2.2510741842099091e+00,
    // Row 4: O 2pz
     1.8952661667439037e-02,  2.2303577583430578e-01,  0.0000000000000000e+00,  0.0000000000000000e+00, -1.0078671766096500e+01,  1.8150889435713657e+00,  1.8150889435713657e+00,
    // Row 5: H1 1s
    -1.7415712587102525e+00, -3.8611015296291202e+00,  0.0000000000000000e+00, -2.2510741842099091e+00,  1.8150889435713660e+00, -5.8313810166054090e+00, -1.6108272751186510e+00,
    // Row 6: H2 1s
    -1.7415712587102525e+00, -3.8611015296291202e+00,  0.0000000000000000e+00,  2.2510741842099091e+00,  1.8150889435713660e+00, -1.6108272751186505e+00, -5.8313810166054099e+00
};

// clang-format on

// =============================================================================
// Helper Functions
// =============================================================================

/// @brief Create STO-3G H2 basis set (2 basis functions)
BasisSet make_h2_sto3g() {
    Point3D H1{0.0, 0.0, 0.0};
    Point3D H2{0.0, 0.0, 1.4};

    std::vector<Real> H_exp = {3.42525091, 0.62391373, 0.16885540};
    std::vector<Real> H_coeff = {0.15432897, 0.53532814, 0.44463454};

    Shell s1(0, H1, H_exp, H_coeff);
    s1.set_atom_index(0);
    Shell s2(0, H2, H_exp, H_coeff);
    s2.set_atom_index(1);

    return BasisSet({s1, s2});
}

/// @brief Create point charges for H2 (two protons)
PointChargeParams make_h2_charges() {
    PointChargeParams charges;
    charges.x = {0.0, 0.0};
    charges.y = {0.0, 0.0};
    charges.z = {0.0, 1.4};
    charges.charge = {1.0, 1.0};
    return charges;
}

/// @brief Create STO-3G H2O basis set (7 basis functions)
BasisSet make_h2o_sto3g() {
    Point3D O_center{0.0, 0.0, 0.0};
    Point3D H1_center{0.0, 1.43233673, -1.10866041};
    Point3D H2_center{0.0, -1.43233673, -1.10866041};

    // O 1s
    Shell O_1s(0, O_center, {130.7093200, 23.8088610, 6.4436083},
                             {0.15432897, 0.53532814, 0.44463454});
    O_1s.set_atom_index(0);

    // O 2s
    Shell O_2s(0, O_center, {5.0331513, 1.1695961, 0.3803890},
                             {-0.09996723, 0.39951283, 0.70011547});
    O_2s.set_atom_index(0);

    // O 2p
    Shell O_2p(1, O_center, {5.0331513, 1.1695961, 0.3803890},
                             {0.15591627, 0.60768372, 0.39195739});
    O_2p.set_atom_index(0);

    // H1 1s
    Shell H1_1s(0, H1_center, {3.42525091, 0.62391373, 0.16885540},
                                {0.15432897, 0.53532814, 0.44463454});
    H1_1s.set_atom_index(1);

    // H2 1s
    Shell H2_1s(0, H2_center, {3.42525091, 0.62391373, 0.16885540},
                                {0.15432897, 0.53532814, 0.44463454});
    H2_1s.set_atom_index(2);

    return BasisSet({O_1s, O_2s, O_2p, H1_1s, H2_1s});
}

/// @brief Create point charges for H2O (O=8, H=1, H=1)
PointChargeParams make_h2o_charges() {
    PointChargeParams charges;
    charges.x = {0.0, 0.0, 0.0};
    charges.y = {0.0, 1.43233673, -1.43233673};
    charges.z = {0.0, -1.10866041, -1.10866041};
    charges.charge = {8.0, 1.0, 1.0};
    return charges;
}

/// @brief Check that a row-major N x N matrix is symmetric
void check_symmetry(const std::vector<Real>& mat, int n, Real tol = SYM_TOL) {
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            EXPECT_NEAR(mat[i * n + j], mat[j * n + i], tol)
                << "Asymmetry at (" << i << "," << j << ")";
        }
    }
}

/// @brief Compare computed matrix against reference values element-by-element
void compare_matrices(const std::vector<Real>& computed,
                      const Real* reference,
                      int n, Real tol, const std::string& name) {
    ASSERT_EQ(computed.size(), static_cast<Size>(n * n))
        << name << " size mismatch: expected " << (n * n);
    Real max_err = 0.0;
    int max_i = 0, max_j = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Size idx = static_cast<Size>(i * n + j);
            Real err = std::abs(computed[idx] - reference[idx]);
            if (err > max_err) {
                max_err = err;
                max_i = i;
                max_j = j;
            }
            EXPECT_NEAR(computed[idx], reference[idx], tol)
                << name << " mismatch at (" << i << "," << j << "): "
                << "computed=" << computed[idx]
                << " reference=" << reference[idx]
                << " err=" << err;
        }
    }
    EXPECT_LT(max_err, tol)
        << name << " max absolute error " << max_err
        << " at (" << max_i << "," << max_j << ")";
}

}  // anonymous namespace

// =============================================================================
// Test Fixture for H2/STO-3G
// =============================================================================

class H2STO3GTest : public ::testing::Test {
protected:
    void SetUp() override {
        basis_ = make_h2_sto3g();
        engine_ = std::make_unique<Engine>(basis_);
        nbf_ = static_cast<int>(basis_.n_basis_functions());
    }

    BasisSet basis_;
    std::unique_ptr<Engine> engine_;
    int nbf_{0};
};

// =============================================================================
// Test Fixture for H2O/STO-3G
// =============================================================================

class H2OSTO3GTest : public ::testing::Test {
protected:
    void SetUp() override {
        basis_ = make_h2o_sto3g();
        engine_ = std::make_unique<Engine>(basis_);
        nbf_ = static_cast<int>(basis_.n_basis_functions());
    }

    BasisSet basis_;
    std::unique_ptr<Engine> engine_;
    int nbf_{0};
};

// =============================================================================
// H2/STO-3G Tests
// =============================================================================

TEST_F(H2STO3GTest, BasisSetSize) {
    EXPECT_EQ(nbf_, 2);
    EXPECT_EQ(basis_.n_shells(), 2u);
}

TEST_F(H2STO3GTest, OverlapAgainstPySCF) {
    std::vector<Real> S;
    engine_->compute_1e(OneElectronOperator(Operator::overlap()), S);

    ASSERT_EQ(S.size(), 4u);
    compare_matrices(S, H2_S_REF, nbf_, REF_TOL, "H2 Overlap");
}

TEST_F(H2STO3GTest, OverlapSymmetry) {
    std::vector<Real> S;
    engine_->compute_1e(OneElectronOperator(Operator::overlap()), S);
    check_symmetry(S, nbf_);
}

TEST_F(H2STO3GTest, OverlapDiagonalUnity) {
    std::vector<Real> S;
    engine_->compute_1e(OneElectronOperator(Operator::overlap()), S);

    for (int i = 0; i < nbf_; ++i) {
        EXPECT_NEAR(S[i * nbf_ + i], 1.0, REF_TOL)
            << "S(" << i << "," << i << ") should be 1.0";
    }
}

TEST_F(H2STO3GTest, KineticAgainstPySCF) {
    std::vector<Real> T;
    engine_->compute_1e(OneElectronOperator(Operator::kinetic()), T);

    ASSERT_EQ(T.size(), 4u);
    compare_matrices(T, H2_T_REF, nbf_, REF_TOL, "H2 Kinetic");
}

TEST_F(H2STO3GTest, KineticSymmetry) {
    std::vector<Real> T;
    engine_->compute_1e(OneElectronOperator(Operator::kinetic()), T);
    check_symmetry(T, nbf_);
}

TEST_F(H2STO3GTest, NuclearAgainstPySCF) {
    auto charges = make_h2_charges();
    std::vector<Real> V;
    engine_->compute_1e(OneElectronOperator(Operator::nuclear(charges)), V);

    ASSERT_EQ(V.size(), 4u);
    compare_matrices(V, H2_V_REF, nbf_, REF_TOL, "H2 Nuclear");
}

TEST_F(H2STO3GTest, NuclearSymmetry) {
    auto charges = make_h2_charges();
    std::vector<Real> V;
    engine_->compute_1e(OneElectronOperator(Operator::nuclear(charges)), V);
    check_symmetry(V, nbf_);
}

// =============================================================================
// H2O/STO-3G Tests
// =============================================================================

TEST_F(H2OSTO3GTest, BasisSetSize) {
    EXPECT_EQ(nbf_, 7);
    EXPECT_EQ(basis_.n_shells(), 5u);
    EXPECT_EQ(basis_.max_angular_momentum(), 1);
}

TEST_F(H2OSTO3GTest, OverlapAgainstPySCF) {
    std::vector<Real> S;
    engine_->compute_1e(OneElectronOperator(Operator::overlap()), S);

    ASSERT_EQ(S.size(), 49u);
    compare_matrices(S, H2O_S_REF, nbf_, REF_TOL, "H2O Overlap");
}

TEST_F(H2OSTO3GTest, OverlapSymmetry) {
    std::vector<Real> S;
    engine_->compute_1e(OneElectronOperator(Operator::overlap()), S);
    check_symmetry(S, nbf_);
}

TEST_F(H2OSTO3GTest, OverlapDiagonalUnity) {
    std::vector<Real> S;
    engine_->compute_1e(OneElectronOperator(Operator::overlap()), S);

    for (int i = 0; i < nbf_; ++i) {
        EXPECT_NEAR(S[i * nbf_ + i], 1.0, REF_TOL)
            << "S(" << i << "," << i << ") should be 1.0";
    }
}

TEST_F(H2OSTO3GTest, KineticAgainstPySCF) {
    std::vector<Real> T;
    engine_->compute_1e(OneElectronOperator(Operator::kinetic()), T);

    ASSERT_EQ(T.size(), 49u);
    compare_matrices(T, H2O_T_REF, nbf_, REF_TOL, "H2O Kinetic");
}

TEST_F(H2OSTO3GTest, KineticSymmetry) {
    std::vector<Real> T;
    engine_->compute_1e(OneElectronOperator(Operator::kinetic()), T);
    check_symmetry(T, nbf_);
}

TEST_F(H2OSTO3GTest, KineticDiagonalPositive) {
    std::vector<Real> T;
    engine_->compute_1e(OneElectronOperator(Operator::kinetic()), T);

    for (int i = 0; i < nbf_; ++i) {
        EXPECT_GT(T[i * nbf_ + i], 0.0)
            << "T(" << i << "," << i << ") should be positive";
    }
}

TEST_F(H2OSTO3GTest, NuclearAgainstPySCF) {
    auto charges = make_h2o_charges();
    std::vector<Real> V;
    engine_->compute_1e(OneElectronOperator(Operator::nuclear(charges)), V);

    ASSERT_EQ(V.size(), 49u);
    compare_matrices(V, H2O_V_REF, nbf_, REF_TOL, "H2O Nuclear");
}

TEST_F(H2OSTO3GTest, NuclearSymmetry) {
    auto charges = make_h2o_charges();
    std::vector<Real> V;
    engine_->compute_1e(OneElectronOperator(Operator::nuclear(charges)), V);
    check_symmetry(V, nbf_);
}

TEST_F(H2OSTO3GTest, NuclearDiagonalNegative) {
    auto charges = make_h2o_charges();
    std::vector<Real> V;
    engine_->compute_1e(OneElectronOperator(Operator::nuclear(charges)), V);

    for (int i = 0; i < nbf_; ++i) {
        EXPECT_LT(V[i * nbf_ + i], 0.0)
            << "V(" << i << "," << i << ") should be negative";
    }
}

// =============================================================================
// Composed Operator Tests (H2O/STO-3G)
// =============================================================================

TEST_F(H2OSTO3GTest, HcoreEqualsTPlusV) {
    auto charges = make_h2o_charges();

    // Compute T and V separately
    std::vector<Real> T, V;
    engine_->compute_1e(OneElectronOperator(Operator::kinetic()), T);
    engine_->compute_1e(OneElectronOperator(Operator::nuclear(charges)), V);

    // Compute H_core = T + V using composed operator
    OneElectronOperator h_core = Operator::kinetic();
    h_core.add(Operator::nuclear(charges));
    std::vector<Real> H;
    engine_->compute_1e(h_core, H);

    ASSERT_EQ(H.size(), static_cast<Size>(nbf_ * nbf_));

    for (int i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(H[i], T[i] + V[i], REF_TOL)
            << "H_core[" << i << "] = " << H[i]
            << " should equal T+V = " << (T[i] + V[i]);
    }
}

TEST_F(H2OSTO3GTest, HcoreSymmetry) {
    auto charges = make_h2o_charges();

    OneElectronOperator h_core = Operator::kinetic();
    h_core.add(Operator::nuclear(charges));
    std::vector<Real> H;
    engine_->compute_1e(h_core, H);

    check_symmetry(H, nbf_);
}

TEST_F(H2OSTO3GTest, HcoreAgainstPySCFSum) {
    auto charges = make_h2o_charges();

    // Compute H_core via composed operator
    OneElectronOperator h_core = Operator::kinetic();
    h_core.add(Operator::nuclear(charges));
    std::vector<Real> H;
    engine_->compute_1e(h_core, H);

    ASSERT_EQ(H.size(), 49u);

    // Compare against PySCF T_ref + V_ref
    for (int i = 0; i < 49; ++i) {
        Real ref_hcore = H2O_T_REF[i] + H2O_V_REF[i];
        EXPECT_NEAR(H[i], ref_hcore, REF_TOL)
            << "H_core[" << (i / nbf_) << "," << (i % nbf_) << "]: "
            << "computed=" << H[i] << " ref(T+V)=" << ref_hcore;
    }
}

TEST_F(H2OSTO3GTest, ComposedOperatorViaAddition) {
    auto charges = make_h2o_charges();

    // Build H_core using operator+ on OneElectronOperator objects
    OneElectronOperator op_T(Operator::kinetic());
    OneElectronOperator op_V(Operator::nuclear(charges));
    OneElectronOperator h_core = op_T + op_V;

    std::vector<Real> H;
    engine_->compute_1e(h_core, H);

    // Compare T and V computed separately
    std::vector<Real> T, V;
    engine_->compute_1e(op_T, T);
    engine_->compute_1e(op_V, V);

    ASSERT_EQ(H.size(), static_cast<Size>(nbf_ * nbf_));

    for (int i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_NEAR(H[i], T[i] + V[i], REF_TOL)
            << "operator+ composed H_core[" << i << "] mismatch";
    }
}

// =============================================================================
// Matrix Property Tests
// =============================================================================

TEST_F(H2OSTO3GTest, OverlapPositiveDefiniteApprox) {
    // Approximation: if all diagonal elements are 1.0 and off-diagonal < 1.0,
    // the overlap matrix is likely positive definite. A full eigenvalue check
    // is beyond the scope here.
    std::vector<Real> S;
    engine_->compute_1e(OneElectronOperator(Operator::overlap()), S);

    for (int i = 0; i < nbf_; ++i) {
        EXPECT_GT(S[i * nbf_ + i], 0.0)
            << "S diagonal should be positive for positive definiteness";
    }

    // Off-diagonal magnitudes should be strictly less than 1.0
    for (int i = 0; i < nbf_; ++i) {
        for (int j = 0; j < nbf_; ++j) {
            if (i != j) {
                EXPECT_LT(std::abs(S[i * nbf_ + j]), 1.0)
                    << "Off-diagonal |S(" << i << "," << j
                    << ")| should be < 1.0";
            }
        }
    }
}

TEST_F(H2OSTO3GTest, OverlapTraceEqualsNBF) {
    // Trace of the overlap matrix should equal the number of basis functions
    // because each normalized shell has self-overlap = 1.0
    std::vector<Real> S;
    engine_->compute_1e(OneElectronOperator(Operator::overlap()), S);

    Real trace = 0.0;
    for (int i = 0; i < nbf_; ++i) {
        trace += S[i * nbf_ + i];
    }
    EXPECT_NEAR(trace, static_cast<Real>(nbf_), REF_TOL);
}

TEST_F(H2OSTO3GTest, KineticPositiveSemiDefinite) {
    // All diagonal elements of kinetic matrix should be positive
    std::vector<Real> T;
    engine_->compute_1e(OneElectronOperator(Operator::kinetic()), T);

    for (int i = 0; i < nbf_; ++i) {
        EXPECT_GT(T[i * nbf_ + i], 0.0)
            << "T(" << i << "," << i << ") should be > 0";
    }
}

TEST_F(H2OSTO3GTest, NuclearAllDiagonalNegative) {
    // For a molecule with only positive nuclear charges,
    // all diagonal V elements should be negative
    auto charges = make_h2o_charges();
    std::vector<Real> V;
    engine_->compute_1e(OneElectronOperator(Operator::nuclear(charges)), V);

    for (int i = 0; i < nbf_; ++i) {
        EXPECT_LT(V[i * nbf_ + i], 0.0)
            << "V(" << i << "," << i << ") should be < 0 (attractive)";
    }
}

// =============================================================================
// Cross-Validation: H2 Composed Operator
// =============================================================================

TEST_F(H2STO3GTest, HcoreEqualsTPlusV) {
    auto charges = make_h2_charges();

    // Compute T and V separately
    std::vector<Real> T, V;
    engine_->compute_1e(OneElectronOperator(Operator::kinetic()), T);
    engine_->compute_1e(OneElectronOperator(Operator::nuclear(charges)), V);

    // Compute H_core via composed operator
    OneElectronOperator h_core = Operator::kinetic();
    h_core.add(Operator::nuclear(charges));
    std::vector<Real> H;
    engine_->compute_1e(h_core, H);

    ASSERT_EQ(H.size(), 4u);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(H[i], T[i] + V[i], REF_TOL)
            << "H2 H_core[" << i << "] should equal T + V";
    }
}

TEST_F(H2STO3GTest, HcoreAgainstPySCFSum) {
    auto charges = make_h2_charges();

    OneElectronOperator h_core = Operator::kinetic();
    h_core.add(Operator::nuclear(charges));
    std::vector<Real> H;
    engine_->compute_1e(h_core, H);

    ASSERT_EQ(H.size(), 4u);

    for (int i = 0; i < 4; ++i) {
        Real ref_hcore = H2_T_REF[i] + H2_V_REF[i];
        EXPECT_NEAR(H[i], ref_hcore, REF_TOL)
            << "H2 H_core[" << (i / 2) << "," << (i % 2) << "]: "
            << "computed=" << H[i] << " ref(T+V)=" << ref_hcore;
    }
}
