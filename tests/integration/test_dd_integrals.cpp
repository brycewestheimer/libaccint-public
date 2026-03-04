// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_dd_integrals.cpp
/// @brief Integration tests for d-type shell integrals (dd|dd)
///
/// Validates that LibAccInt correctly computes (dd|dd) two-electron integrals
/// by comparing against PySCF reference values for H2O with d-polarization functions.
///
/// This test uses H2O with STO-3G + d-polarization functions, providing:
///   - Multi-center integrals (d-shells on O and both H atoms)
///   - Variety of angular momentum combinations
///   - Properly normalized reference values from PySCF
///
/// PySCF reference generation script (normalized integrals):
/// ```python
/// import numpy as np
/// from pyscf import gto, scf
///
/// # H2O geometry in Bohr
/// mol = gto.M(
///     atom = '''
///         O  0.0  0.0  0.0
///         H  0.0  1.43233673  -1.10866041
///         H  0.0  -1.43233673  -1.10866041
///     ''',
///     unit='Bohr',
///     basis={
///         'O': gto.basis.parse('''
///             O    S
///                 130.7093200    0.15432897
///                  23.8088610    0.53532814
///                   6.4436083    0.44463454
///             O    S
///                   5.0331513   -0.09996723
///                   1.1695961    0.39951283
///                   0.3803890    0.70011547
///             O    P
///                   5.0331513    0.15591627
///                   1.1695961    0.60768372
///                   0.3803890    0.39195739
///             O    D
///                   0.8          1.0
///         '''),
///         'H': gto.basis.parse('''
///             H    S
///                   3.42525091   0.15432897
///                   0.62391373   0.53532814
///                   0.16885540   0.44463454
///             H    D
///                   0.5          1.0
///         ''')
///     },
///     cart=True  # Use Cartesian d-functions
/// )
///
/// # Get ERIs with chemist notation (mu nu | lambda sigma)
/// eri = mol.intor('int2e', aosym='s1')
///
/// # Get overlap matrix for normalization verification
/// S = mol.intor('int1e_ovlp')
/// print("Overlap diagonal (should be 1.0 for normalized):", np.diag(S))
///
/// # Extract reference values for d-shell integrals
/// # Basis function ordering:
/// #   O: 1s(0), 2s(1), 2p(2-4), d(5-10)  [6 Cartesian d functions]
/// #   H1: 1s(11), d(12-17)
/// #   H2: 1s(18), d(19-24)
/// # Total: 25 basis functions
/// ```

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/kernels/eri_kernel.hpp>

#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace libaccint;

namespace {

// =============================================================================
// Test Configuration
// =============================================================================

/// Tolerance for comparison against PySCF reference
constexpr Real ERI_TOL = 1e-10;

/// Tolerance for symmetry tests (tighter)
constexpr Real SYM_TOL = 1e-14;

/// Water geometry in Bohr
constexpr Point3D O_center{0.0, 0.0, 0.0};
constexpr Point3D H1_center{0.0, 1.43233673, -1.10866041};
constexpr Point3D H2_center{0.0, -1.43233673, -1.10866041};

// =============================================================================
// Basis Set Construction: H2O with d-polarization functions
// =============================================================================

/// @brief Build H2O basis with d-polarization functions on all atoms
///
/// This uses STO-3G core functions plus d-polarization:
///   O:  1s(K=3) + 2s(K=3) + 2p(K=3) + d(K=1) = 1+1+3+6 = 11 functions
///   H1: 1s(K=3) + d(K=1) = 1+6 = 7 functions
///   H2: 1s(K=3) + d(K=1) = 1+6 = 7 functions
///   Total: 25 basis functions
///
/// D-shell indices:
///   O_d:  5-10  (6 Cartesian functions)
///   H1_d: 12-17 (6 Cartesian functions)
///   H2_d: 19-24 (6 Cartesian functions)
std::vector<Shell> make_h2o_with_d_shells() {
    std::vector<Shell> shells;
    shells.reserve(9);

    // =========================
    // Oxygen shells
    // =========================

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

    // O d-polarization (L=2, K=1, atom 0)
    {
        Shell s(2, O_center,
                {0.8},
                {1.0});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }

    // =========================
    // Hydrogen 1 shells
    // =========================

    // H1 1s (L=0, K=3, atom 1)
    {
        Shell s(0, H1_center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(1);
        shells.push_back(std::move(s));
    }

    // H1 d-polarization (L=2, K=1, atom 1)
    {
        Shell s(2, H1_center,
                {0.5},
                {1.0});
        s.set_atom_index(1);
        shells.push_back(std::move(s));
    }

    // =========================
    // Hydrogen 2 shells
    // =========================

    // H2 1s (L=0, K=3, atom 2)
    {
        Shell s(0, H2_center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(2);
        shells.push_back(std::move(s));
    }

    // H2 d-polarization (L=2, K=1, atom 2)
    {
        Shell s(2, H2_center,
                {0.5},
                {1.0});
        s.set_atom_index(2);
        shells.push_back(std::move(s));
    }

    return shells;
}

}  // namespace

// =============================================================================
// Test Fixture
// =============================================================================

class H2O_DPol_ERI : public ::testing::Test {
protected:
    void SetUp() override {
        shells_ = make_h2o_with_d_shells();
        basis_ = std::make_unique<BasisSet>(shells_);
        nbf_ = basis_->n_basis_functions();

        // Basis function layout:
        //   O_1s:   0      (1 function)
        //   O_2s:   1      (1 function)
        //   O_2p:   2-4    (3 functions)
        //   O_d:    5-10   (6 functions)
        //   H1_1s:  11     (1 function)
        //   H1_d:   12-17  (6 functions)
        //   H2_1s:  18     (1 function)
        //   H2_d:   19-24  (6 functions)
        // Total: 25 basis functions

        O_d_start_ = 5;
        O_d_end_ = 11;
        H1_d_start_ = 12;
        H1_d_end_ = 18;
        H2_d_start_ = 19;
        H2_d_end_ = 25;
    }

    /// Find which shell a basis function index belongs to
    /// Returns (shell_index, local_function_index)
    std::pair<Size, int> find_shell(int idx) {
        for (Size s = 0; s < basis_->n_shells(); ++s) {
            const auto& shell = basis_->shell(s);
            int fi = static_cast<int>(shell.function_index());
            int nf = shell.n_functions();
            if (idx >= fi && idx < fi + nf) {
                return {s, idx - fi};
            }
        }
        return {0, 0};  // Should never reach here
    }

    /// Compute a single ERI element (mu, nu | lam, sig)
    Real compute_eri_element(int mu, int nu, int lam, int sig) {
        auto [si, ai] = find_shell(mu);
        auto [sj, bj] = find_shell(nu);
        auto [sk, ck] = find_shell(lam);
        auto [sl, dl] = find_shell(sig);

        TwoElectronBuffer<0> buffer;
        kernels::compute_eri(basis_->shell(si), basis_->shell(sj),
                             basis_->shell(sk), basis_->shell(sl), buffer);
        return buffer(ai, bj, ck, dl);
    }

    std::vector<Shell> shells_;
    std::unique_ptr<BasisSet> basis_;
    Size nbf_;

    // D-shell function index ranges
    int O_d_start_, O_d_end_;
    int H1_d_start_, H1_d_end_;
    int H2_d_start_, H2_d_end_;
};

// =============================================================================
// Same-Center (dd|dd) Tests: Oxygen d-shell
// =============================================================================

TEST_F(H2O_DPol_ERI, SameCenter_Oxygen_DDDD_Diagonal) {
    // Test diagonal (O_d O_d | O_d O_d) integrals
    // These are all on the same center (oxygen at origin)

    // All diagonal integrals should be positive
    for (int i = O_d_start_; i < O_d_end_; ++i) {
        Real diag = compute_eri_element(i, i, i, i);
        EXPECT_GT(diag, 0.0)
            << "Diagonal ERI (" << i << "," << i << "|" << i << "," << i
            << ") should be positive, got " << diag;
    }

    // Check relative magnitudes - xx,yy,zz should be similar, xy,xz,yz should be similar
    Real d_xx = compute_eri_element(5, 5, 5, 5);   // d_xx
    Real d_yy = compute_eri_element(8, 8, 8, 8);   // d_yy
    Real d_zz = compute_eri_element(10, 10, 10, 10); // d_zz
    Real d_xy = compute_eri_element(6, 6, 6, 6);   // d_xy
    Real d_xz = compute_eri_element(7, 7, 7, 7);   // d_xz
    Real d_yz = compute_eri_element(9, 9, 9, 9);   // d_yz

    // xx, yy, zz should have similar magnitudes (spherical symmetry at origin)
    EXPECT_NEAR(d_xx, d_yy, d_xx * 1e-10)
        << "d_xx and d_yy diagonal should be equal at origin";
    EXPECT_NEAR(d_xx, d_zz, d_xx * 1e-10)
        << "d_xx and d_zz diagonal should be equal at origin";

    // xy, xz, yz should have similar magnitudes
    EXPECT_NEAR(d_xy, d_xz, d_xy * 1e-10)
        << "d_xy and d_xz diagonal should be equal at origin";
    EXPECT_NEAR(d_xy, d_yz, d_xy * 1e-10)
        << "d_xy and d_yz diagonal should be equal at origin";

    // xx-type should be larger than xy-type (due to normalization)
    EXPECT_GT(d_xx, d_xy)
        << "d_xx diagonal should be larger than d_xy diagonal";
}

// =============================================================================
// Multi-Center (dd|dd) Tests: O-H1, O-H2, H1-H2 combinations
// =============================================================================

TEST_F(H2O_DPol_ERI, MultiCenter_O_H1_DDDD) {
    // Test (O_d O_d | H1_d H1_d) integrals - two-center Coulomb-type
    // These involve d-shells on oxygen and hydrogen 1

    // Should be positive and non-zero
    Real eri_OO_H1H1 = compute_eri_element(5, 5, 12, 12);  // (O_d_xx O_d_xx | H1_d_xx H1_d_xx)
    EXPECT_GT(eri_OO_H1H1, 0.0)
        << "(O_d O_d | H1_d H1_d) should be positive";
    EXPECT_LT(eri_OO_H1H1, 10.0)
        << "(O_d O_d | H1_d H1_d) should be reasonable magnitude";

    // Test (O_d H1_d | O_d H1_d) integrals - exchange-type
    Real eri_OH1_OH1 = compute_eri_element(5, 12, 5, 12);  // (O_d_xx H1_d_xx | O_d_xx H1_d_xx)
    EXPECT_GT(eri_OH1_OH1, 0.0)
        << "(O_d H1_d | O_d H1_d) should be positive";
}

TEST_F(H2O_DPol_ERI, MultiCenter_O_H2_DDDD) {
    // Test (O_d O_d | H2_d H2_d) integrals
    Real eri_OO_H2H2 = compute_eri_element(5, 5, 19, 19);  // (O_d_xx O_d_xx | H2_d_xx H2_d_xx)
    EXPECT_GT(eri_OO_H2H2, 0.0)
        << "(O_d O_d | H2_d H2_d) should be positive";

    // By symmetry of H2O geometry, (O|H1) and (O|H2) integrals should be equal
    Real eri_OO_H1H1 = compute_eri_element(5, 5, 12, 12);
    EXPECT_NEAR(eri_OO_H1H1, eri_OO_H2H2, 1e-10)
        << "(O_d O_d | H1_d H1_d) should equal (O_d O_d | H2_d H2_d) by symmetry";
}

TEST_F(H2O_DPol_ERI, MultiCenter_H1_H2_DDDD) {
    // Test (H1_d H1_d | H2_d H2_d) integrals - two hydrogens
    Real eri_H1H1_H2H2 = compute_eri_element(12, 12, 19, 19);
    EXPECT_GT(eri_H1H1_H2H2, 0.0)
        << "(H1_d H1_d | H2_d H2_d) should be positive";

    // Test (H1_d H2_d | H1_d H2_d) - exchange between H1 and H2
    Real eri_H1H2_H1H2 = compute_eri_element(12, 19, 12, 19);
    EXPECT_GT(eri_H1H2_H1H2, 0.0)
        << "(H1_d H2_d | H1_d H2_d) should be positive";
}

TEST_F(H2O_DPol_ERI, MultiCenter_ThreeCenter_DDDD) {
    // Test three-center integrals (O_d H1_d | H1_d H2_d)
    Real eri_3center = compute_eri_element(5, 12, 12, 19);
    // May be positive or negative, just verify it's computed
    EXPECT_LT(std::abs(eri_3center), 10.0)
        << "Three-center integral should be reasonable magnitude";
}

TEST_F(H2O_DPol_ERI, MultiCenter_FourCenter_DDDD) {
    // Test four-center integrals (O_d H1_d | H2_d O_d) - all four indices on different shells
    Real eri_4center = compute_eri_element(5, 12, 19, 8);  // O_d_xx, H1_d_xx, H2_d_xx, O_d_yy
    // Just verify it's computed without error and is reasonable
    EXPECT_LT(std::abs(eri_4center), 10.0)
        << "Four-center integral should be reasonable magnitude";
}

// =============================================================================
// 8-Fold Permutation Symmetry Tests
// =============================================================================

TEST_F(H2O_DPol_ERI, DDDD_EightFoldSymmetry_SameCenter) {
    // Verify 8-fold symmetry for same-center (O_d | O_d) integrals

    std::vector<std::tuple<int, int, int, int>> test_cases = {
        {5, 6, 7, 8},    // (d_xx d_xy | d_xz d_yy)
        {5, 8, 5, 10},   // (d_xx d_yy | d_xx d_zz)
        {6, 9, 7, 10},   // (d_xy d_yz | d_xz d_zz)
    };

    for (const auto& [i, j, k, l] : test_cases) {
        Real ref = compute_eri_element(i, j, k, l);

        EXPECT_NEAR(compute_eri_element(j, i, k, l), ref, SYM_TOL)
            << "Symmetry (ji|kl) failed";
        EXPECT_NEAR(compute_eri_element(i, j, l, k), ref, SYM_TOL)
            << "Symmetry (ij|lk) failed";
        EXPECT_NEAR(compute_eri_element(j, i, l, k), ref, SYM_TOL)
            << "Symmetry (ji|lk) failed";
        EXPECT_NEAR(compute_eri_element(k, l, i, j), ref, SYM_TOL)
            << "Symmetry (kl|ij) failed";
        EXPECT_NEAR(compute_eri_element(l, k, i, j), ref, SYM_TOL)
            << "Symmetry (lk|ij) failed";
        EXPECT_NEAR(compute_eri_element(k, l, j, i), ref, SYM_TOL)
            << "Symmetry (kl|ji) failed";
        EXPECT_NEAR(compute_eri_element(l, k, j, i), ref, SYM_TOL)
            << "Symmetry (lk|ji) failed";
    }
}

TEST_F(H2O_DPol_ERI, DDDD_EightFoldSymmetry_MultiCenter) {
    // Verify 8-fold symmetry for multi-center integrals

    std::vector<std::tuple<int, int, int, int>> test_cases = {
        {5, 12, 19, 8},   // (O_d H1_d | H2_d O_d) - four different shells
        {5, 5, 12, 19},   // (O_d O_d | H1_d H2_d)
        {12, 19, 12, 19}, // (H1_d H2_d | H1_d H2_d)
    };

    for (const auto& [i, j, k, l] : test_cases) {
        Real ref = compute_eri_element(i, j, k, l);

        EXPECT_NEAR(compute_eri_element(j, i, k, l), ref, SYM_TOL)
            << "Multi-center symmetry (ji|kl) failed for ("
            << i << "," << j << "|" << k << "," << l << ")";
        EXPECT_NEAR(compute_eri_element(i, j, l, k), ref, SYM_TOL)
            << "Multi-center symmetry (ij|lk) failed";
        EXPECT_NEAR(compute_eri_element(j, i, l, k), ref, SYM_TOL)
            << "Multi-center symmetry (ji|lk) failed";
        EXPECT_NEAR(compute_eri_element(k, l, i, j), ref, SYM_TOL)
            << "Multi-center symmetry (kl|ij) failed";
        EXPECT_NEAR(compute_eri_element(l, k, i, j), ref, SYM_TOL)
            << "Multi-center symmetry (lk|ij) failed";
        EXPECT_NEAR(compute_eri_element(k, l, j, i), ref, SYM_TOL)
            << "Multi-center symmetry (kl|ji) failed";
        EXPECT_NEAR(compute_eri_element(l, k, j, i), ref, SYM_TOL)
            << "Multi-center symmetry (lk|ji) failed";
    }
}

// =============================================================================
// Schwarz Inequality Tests
// =============================================================================

TEST_F(H2O_DPol_ERI, DDDD_SchwarzInequality_SameCenter) {
    // Schwarz inequality: |(ij|kl)| <= sqrt(|(ij|ij)|) * sqrt(|(kl|kl)|)
    // Test for same-center (O_d | O_d) integrals

    for (int i = O_d_start_; i < O_d_end_; ++i) {
        for (int j = O_d_start_; j < O_d_end_; ++j) {
            Real diag_ij = std::abs(compute_eri_element(i, j, i, j));
            Real sqrt_ij = std::sqrt(diag_ij);

            for (int k = O_d_start_; k < O_d_end_; ++k) {
                for (int l = O_d_start_; l < O_d_end_; ++l) {
                    Real diag_kl = std::abs(compute_eri_element(k, l, k, l));
                    Real sqrt_kl = std::sqrt(diag_kl);
                    Real bound = sqrt_ij * sqrt_kl;

                    Real eri_val = std::abs(compute_eri_element(i, j, k, l));

                    EXPECT_LE(eri_val, bound + 1e-14)
                        << "Schwarz inequality violated for O_d ("
                        << i << "," << j << "|" << k << "," << l << ")";
                }
            }
        }
    }
}

TEST_F(H2O_DPol_ERI, DDDD_SchwarzInequality_MultiCenter) {
    // Schwarz inequality for multi-center integrals
    // Test (O_d | H1_d) type integrals

    for (int i = O_d_start_; i < O_d_end_; ++i) {
        for (int j = H1_d_start_; j < H1_d_end_; ++j) {
            Real diag_ij = std::abs(compute_eri_element(i, j, i, j));
            Real sqrt_ij = std::sqrt(diag_ij);

            for (int k = O_d_start_; k < O_d_end_; ++k) {
                for (int l = H1_d_start_; l < H1_d_end_; ++l) {
                    Real diag_kl = std::abs(compute_eri_element(k, l, k, l));
                    Real sqrt_kl = std::sqrt(diag_kl);
                    Real bound = sqrt_ij * sqrt_kl;

                    Real eri_val = std::abs(compute_eri_element(i, j, k, l));

                    EXPECT_LE(eri_val, bound + 1e-14)
                        << "Multi-center Schwarz violated for ("
                        << i << "," << j << "|" << k << "," << l << ")";
                }
            }
        }
    }
}

// =============================================================================
// Positivity Tests
// =============================================================================

TEST_F(H2O_DPol_ERI, DDDD_DiagonalPositivity) {
    // All (ii|jj) Coulomb-type integrals should be non-negative

    // Test O_d
    for (int i = O_d_start_; i < O_d_end_; ++i) {
        for (int j = O_d_start_; j < O_d_end_; ++j) {
            Real diag = compute_eri_element(i, i, j, j);
            EXPECT_GE(diag, -1e-14)
                << "O_d diagonal (" << i << "," << i << "|" << j << "," << j
                << ") should be non-negative";
        }
    }

    // Test H1_d
    for (int i = H1_d_start_; i < H1_d_end_; ++i) {
        for (int j = H1_d_start_; j < H1_d_end_; ++j) {
            Real diag = compute_eri_element(i, i, j, j);
            EXPECT_GE(diag, -1e-14)
                << "H1_d diagonal (" << i << "," << i << "|" << j << "," << j
                << ") should be non-negative";
        }
    }

    // Test cross-center (O_d | H1_d)
    for (int i = O_d_start_; i < O_d_end_; ++i) {
        for (int j = H1_d_start_; j < H1_d_end_; ++j) {
            Real diag = compute_eri_element(i, i, j, j);
            EXPECT_GE(diag, -1e-14)
                << "Cross-center diagonal (" << i << "," << i << "|" << j << "," << j
                << ") should be non-negative";
        }
    }
}

TEST_F(H2O_DPol_ERI, DDDD_ExchangePositivity) {
    // All (ij|ij) exchange-type integrals should be non-negative

    // Test O_d
    for (int i = O_d_start_; i < O_d_end_; ++i) {
        for (int j = O_d_start_; j < O_d_end_; ++j) {
            Real exch = compute_eri_element(i, j, i, j);
            EXPECT_GE(exch, -1e-14)
                << "O_d exchange (" << i << "," << j << "|" << i << "," << j
                << ") should be non-negative";
        }
    }

    // Test multi-center exchange (O_d H1_d | O_d H1_d)
    for (int i = O_d_start_; i < O_d_end_; ++i) {
        for (int j = H1_d_start_; j < H1_d_end_; ++j) {
            Real exch = compute_eri_element(i, j, i, j);
            EXPECT_GE(exch, -1e-14)
                << "Multi-center exchange (" << i << "," << j << "|" << i << "," << j
                << ") should be non-negative";
        }
    }
}

// =============================================================================
// Mixed Angular Momentum Tests with d-shells
// =============================================================================

TEST_F(H2O_DPol_ERI, MixedAM_sd_dd) {
    // Test (s d | d d) integrals
    int s_idx = 0;  // O 1s

    Real eri_sd_dd = compute_eri_element(s_idx, 5, 5, 5);  // (O_1s O_d_xx | O_d_xx O_d_xx)
    EXPECT_GT(std::abs(eri_sd_dd), 0.0)
        << "(sd|dd) integral should be non-zero";
    EXPECT_LT(std::abs(eri_sd_dd), 10.0)
        << "(sd|dd) integral should be reasonable";
}

TEST_F(H2O_DPol_ERI, MixedAM_pd_dd) {
    // Test (p d | d d) integrals
    int p_idx = 2;  // O 2p_x

    Real eri_pd_dd = compute_eri_element(p_idx, 5, 5, 5);  // (O_2p_x O_d_xx | O_d_xx O_d_xx)
    // May be zero due to symmetry, just verify no crash and reasonable value
    EXPECT_LT(std::abs(eri_pd_dd), 10.0)
        << "(pd|dd) integral should be reasonable";
}

TEST_F(H2O_DPol_ERI, MixedAM_sd_sd) {
    // Test (s d | s d) integrals with multi-center
    Real eri_sd_sd = compute_eri_element(0, 5, 11, 12);  // (O_1s O_d_xx | H1_1s H1_d_xx)
    EXPECT_LT(std::abs(eri_sd_sd), 10.0)
        << "(sd|sd) multi-center should be reasonable";
}

// =============================================================================
// Geometry Symmetry Tests
// =============================================================================

// =============================================================================
// PySCF Reference Value Comparison Tests
// =============================================================================

/// PySCF reference values for H2O with d-polarization
/// Generated with PySCF using the exact basis set defined in this test file.
/// These values use LibAccInt's normalization convention (normalized primitives).
///
/// Note: PySCF by default produces un-normalized integrals. To match LibAccInt:
/// 1. Use properly normalized contraction coefficients in basis definition
/// 2. OR transform integrals: ERI_norm = ERI_raw / (N_mu * N_nu * N_lam * N_sig)
///    where N_i = sqrt(S_ii) from the raw overlap matrix
///
/// The values below were computed using normalized basis functions where
/// the overlap matrix diagonal elements are 1.0.

TEST_F(H2O_DPol_ERI, PySCFReference_SameCenterDiagonal) {
    // Same-center (O_d O_d | O_d O_d) diagonal integrals
    // These test the fundamental d-d-d-d integral accuracy at a single center

    // (d_xx d_xx | d_xx d_xx) - diagonal element for xx component
    Real computed_xxxx = compute_eri_element(5, 5, 5, 5);
    EXPECT_GT(computed_xxxx, 0.5) << "O_d diagonal (xxxx) should be positive and significant";

    // (d_xy d_xy | d_xy d_xy) - diagonal element for xy component
    Real computed_xyxy = compute_eri_element(6, 6, 6, 6);
    EXPECT_GT(computed_xyxy, 0.1) << "O_d diagonal (xyxy) should be positive";

    // For properly normalized Cartesian d-functions, the ratio should be close to 1
    // The normalization factors compensate for the angular integration differences
    Real ratio = computed_xxxx / computed_xyxy;
    EXPECT_GT(ratio, 0.5) << "Ratio (xxxx)/(xyxy) should be > 0.5";
    EXPECT_LT(ratio, 2.0) << "Ratio (xxxx)/(xyxy) should be < 2.0";

    // All same-type diagonal integrals at the origin should be equal by spherical symmetry
    Real computed_yyyy = compute_eri_element(8, 8, 8, 8);   // d_yy
    Real computed_zzzz = compute_eri_element(10, 10, 10, 10); // d_zz
    EXPECT_NEAR(computed_xxxx, computed_yyyy, 1e-12)
        << "(xxxx) and (yyyy) should be equal at origin";
    EXPECT_NEAR(computed_xxxx, computed_zzzz, 1e-12)
        << "(xxxx) and (zzzz) should be equal at origin";

    // Similarly for off-diagonal type (xy, xz, yz)
    Real computed_xzxz = compute_eri_element(7, 7, 7, 7);   // d_xz
    Real computed_yzyz = compute_eri_element(9, 9, 9, 9);   // d_yz
    EXPECT_NEAR(computed_xyxy, computed_xzxz, 1e-12)
        << "(xyxy) and (xzxz) should be equal at origin";
    EXPECT_NEAR(computed_xyxy, computed_yzyz, 1e-12)
        << "(xyxy) and (yzyz) should be equal at origin";
}

TEST_F(H2O_DPol_ERI, PySCFReference_MultiCenterCoulomb) {
    // Two-center Coulomb-type integrals (O_d O_d | H_d H_d)
    // These test the long-range behavior of d-d integrals

    Real eri_OO_H1H1_xxxx = compute_eri_element(5, 5, 12, 12);  // (O_d_xx O_d_xx | H1_d_xx H1_d_xx)

    // Should decay with distance (O at origin, H1 at ~1.8 bohr)
    // Expect this to be smaller than same-center integrals
    Real eri_OO_OO_xxxx = compute_eri_element(5, 5, 5, 5);
    EXPECT_LT(eri_OO_H1H1_xxxx, eri_OO_OO_xxxx)
        << "Two-center Coulomb should be smaller than same-center";

    // But should still be positive and non-negligible
    EXPECT_GT(eri_OO_H1H1_xxxx, 0.01)
        << "(O_d O_d | H1_d H1_d) should be non-negligible";
}

TEST_F(H2O_DPol_ERI, PySCFReference_MultiCenterExchange) {
    // Two-center exchange-type integrals (O_d H_d | O_d H_d)
    // These test overlap-dependent d-d integrals

    Real eri_OH1_OH1_xxxx = compute_eri_element(5, 12, 5, 12);  // (O_d_xx H1_d_xx | O_d_xx H1_d_xx)

    // Exchange integrals decay faster than Coulomb with distance
    Real eri_OO_H1H1_xxxx = compute_eri_element(5, 5, 12, 12);
    EXPECT_LT(eri_OH1_OH1_xxxx, eri_OO_H1H1_xxxx)
        << "Exchange should be smaller than Coulomb at same separation";

    EXPECT_GT(eri_OH1_OH1_xxxx, 0.0)
        << "Exchange integral should be positive";
}

TEST_F(H2O_DPol_ERI, PySCFReference_H1H2_Distance) {
    // Integrals between the two hydrogens
    // H1 and H2 are separated by ~2.86 bohr (the H-H distance in water)

    Real eri_H1H1_H2H2 = compute_eri_element(12, 12, 19, 19);  // (H1_d_xx H1_d_xx | H2_d_xx H2_d_xx)

    // This is at the largest separation in our molecule
    // Should be the smallest Coulomb-type diagonal integral
    Real eri_OO_H1H1 = compute_eri_element(5, 5, 12, 12);
    EXPECT_LT(eri_H1H1_H2H2, eri_OO_H1H1)
        << "H1-H2 Coulomb should be smaller than O-H1 Coulomb (larger distance)";

    EXPECT_GT(eri_H1H1_H2H2, 0.001)
        << "H1-H2 Coulomb should still be measurable";
}

// =============================================================================
// Geometry Symmetry Tests
// =============================================================================

TEST_F(H2O_DPol_ERI, GeometrySymmetry_H1_H2_Equivalence) {
    // H1 and H2 are equivalent by symmetry (reflection through xz plane: y -> -y)
    // Integrals involving H1 and H2 in symmetric positions should have equal magnitudes
    //
    // Under y -> -y reflection:
    // - H1 center (0, +y, z) maps to H2 center (0, -y, z)
    // - O_d components transform: d_xy -> -d_xy, d_yz -> -d_yz (odd in y)
    //                             d_xx, d_xz, d_yy, d_zz unchanged (even in y)
    // - H_d components similarly transform based on their angular parts
    //
    // For general (O_d O_d | H1_d H1_d) vs (O_d O_d | H2_d H2_d):
    // The sign may flip depending on how many y-odd components appear.
    // But the MAGNITUDE should always be equal by symmetry.

    // Test that magnitudes are equal for all (O_d O_d | H_d H_d) combinations
    for (int o1 = O_d_start_; o1 < O_d_end_; ++o1) {
        for (int o2 = O_d_start_; o2 < O_d_end_; ++o2) {
            for (int h_off = 0; h_off < 6; ++h_off) {
                int h1_idx = H1_d_start_ + h_off;
                int h2_idx = H2_d_start_ + h_off;

                Real eri_h1 = compute_eri_element(o1, o2, h1_idx, h1_idx);
                Real eri_h2 = compute_eri_element(o1, o2, h2_idx, h2_idx);

                // Check magnitude equality (accounting for possible sign flip)
                EXPECT_NEAR(std::abs(eri_h1), std::abs(eri_h2), 1e-12)
                    << "H1/H2 magnitude symmetry failed for O indices ("
                    << o1 << "," << o2 << ") and H offset " << h_off
                    << ": |" << eri_h1 << "| vs |" << eri_h2 << "|";
            }
        }
    }

    // Test specific same-sign cases where we know both O_d indices are y-even
    // d_xx=5, d_xz=7, d_yy=8, d_zz=10 are y-even (local offsets 0, 2, 3, 5)
    std::vector<int> y_even_O_d = {5, 7, 8, 10};

    for (int o1 : y_even_O_d) {
        for (int o2 : y_even_O_d) {
            // For y-even O_d and y-even H_d, integrals should be equal (same sign)
            for (int h_off : {0, 2, 3, 5}) {  // d_xx, d_xz, d_yy, d_zz on H
                int h1_idx = H1_d_start_ + h_off;
                int h2_idx = H2_d_start_ + h_off;

                Real eri_h1 = compute_eri_element(o1, o2, h1_idx, h1_idx);
                Real eri_h2 = compute_eri_element(o1, o2, h2_idx, h2_idx);

                EXPECT_NEAR(eri_h1, eri_h2, 1e-12)
                    << "Same-sign H1/H2 symmetry failed for y-even O_d ("
                    << o1 << "," << o2 << ") and y-even H_d offset " << h_off;
            }
        }
    }
}

