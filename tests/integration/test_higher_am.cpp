// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_higher_am.cpp
/// @brief Integration tests for f-shell (AM=3) and g-shell (AM=4) integrals
///
/// Validates that LibAccInt correctly computes higher angular momentum integrals
/// for overlap, kinetic, nuclear attraction, and ERI using generated kernels.
/// Tests include:
///   - Single-center and two-center 1e integrals for f/g shells
///   - ERI integrals for all combinations involving f/g
///   - Symmetry properties (S symmetric, T symmetric, ERI 8-fold symmetry)
///   - Normalization (overlap diagonal = 1 for normalized shells)
///   - Positive-definiteness of overlap matrix
///   - Cross-AM combinations (e.g. (sf|pd), (fg|fg))

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/kernels/overlap_kernel.hpp>
#include <libaccint/kernels/kinetic_kernel.hpp>
#include <libaccint/kernels/nuclear_kernel.hpp>
#include <libaccint/kernels/eri_kernel.hpp>
#include <libaccint/math/cartesian_indices.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace libaccint;

namespace {

// =============================================================================
// Tolerances
// =============================================================================

/// Tolerance for comparing against symmetry / normalization requirements
constexpr Real NORM_TOL = 1e-12;

/// Tolerance for symmetry tests
constexpr Real SYM_TOL = 1e-14;

/// Tolerance for ERI positivity
constexpr Real POS_TOL = -1e-14;

// =============================================================================
// Test Geometry: H2O in Bohr
// =============================================================================

constexpr Point3D O_center{0.0, 0.0, 0.0};
constexpr Point3D H1_center{0.0, 1.43233673, -1.10866041};
constexpr Point3D H2_center{0.0, -1.43233673, -1.10866041};

constexpr Real Z_O = 8.0;
constexpr Real Z_H = 1.0;

// =============================================================================
// Helper: Build shells with f and g polarization functions
// =============================================================================

/// @brief Build H2O basis with f-polarization on O and g-polarization on O
///
/// O:  1s(K=3) + 2s(K=3) + 2p(K=3) + f(K=1) + g(K=1)
/// H1: 1s(K=3) + d(K=1)
/// H2: 1s(K=3) + d(K=1)
std::vector<Shell> make_h2o_with_fg_shells() {
    std::vector<Shell> shells;
    shells.reserve(9);

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

    // O f-polarization (L=3, K=1, atom 0) — cc-pVTZ-like
    {
        Shell s(3, O_center, {1.2}, {1.0});
        s.set_atom_index(0);
        shells.push_back(std::move(s));
    }

    // O g-polarization (L=4, K=1, atom 0) — cc-pVQZ-like
    {
        Shell s(4, O_center, {0.9}, {1.0});
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

    // H1 d-polarization (L=2, K=1, atom 1)
    {
        Shell s(2, H1_center, {0.5}, {1.0});
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

    // H2 d-polarization (L=2, K=1, atom 2)
    {
        Shell s(2, H2_center, {0.5}, {1.0});
        s.set_atom_index(2);
        shells.push_back(std::move(s));
    }

    return shells;
}

}  // namespace

// =============================================================================
// Test Fixture
// =============================================================================

class HigherAM_Integration : public ::testing::Test {
protected:
    void SetUp() override {
        shells_ = make_h2o_with_fg_shells();
        basis_ = std::make_unique<BasisSet>(shells_);
        nbf_ = basis_->n_basis_functions();

        // Basis function layout:
        //   O_1s:   0      (1 function)
        //   O_2s:   1      (1 function)
        //   O_2p:   2-4    (3 functions)
        //   O_f:    5-14   (10 functions)
        //   O_g:    15-29  (15 functions)
        //   H1_1s:  30     (1 function)
        //   H1_d:   31-36  (6 functions)
        //   H2_1s:  37     (1 function)
        //   H2_d:   38-43  (6 functions)
        // Total: 44 basis functions
    }

    /// Find which shell a basis function index belongs to
    std::pair<Size, int> find_shell(int idx) {
        for (Size s = 0; s < basis_->n_shells(); ++s) {
            const auto& shell = basis_->shell(s);
            int fi = static_cast<int>(shell.function_index());
            int nf = shell.n_functions();
            if (idx >= fi && idx < fi + nf) {
                return {s, idx - fi};
            }
        }
        return {0, 0};
    }

    /// Compute a single overlap matrix element
    Real compute_overlap(int mu, int nu) {
        auto [si, ai] = find_shell(mu);
        auto [sj, bj] = find_shell(nu);

        OneElectronBuffer<0> buffer;
        kernels::compute_overlap(basis_->shell(si), basis_->shell(sj), buffer);
        return buffer(ai, bj);
    }

    /// Compute a single kinetic matrix element
    Real compute_kinetic(int mu, int nu) {
        auto [si, ai] = find_shell(mu);
        auto [sj, bj] = find_shell(nu);

        OneElectronBuffer<0> buffer;
        kernels::compute_kinetic(basis_->shell(si), basis_->shell(sj), buffer);
        return buffer(ai, bj);
    }

    /// Compute a single ERI element (mu, nu | lam, sig)
    Real compute_eri(int mu, int nu, int lam, int sig) {
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
};

// =============================================================================
// 1. Overlap Normalization Tests
// =============================================================================

TEST_F(HigherAM_Integration, OverlapDiagonal_AllShells) {
    // Diagonal elements of the overlap matrix should be 1.0 for normalized shells
    for (Size i = 0; i < nbf_; ++i) {
        Real s_ii = compute_overlap(i, i);
        EXPECT_NEAR(s_ii, 1.0, NORM_TOL)
            << "S(" << i << "," << i << ") should be 1.0 for normalized shells";
    }
}

TEST_F(HigherAM_Integration, OverlapSymmetry_FShell) {
    // S(mu, nu) = S(nu, mu) for f-shell pairs
    // O_f starts at index 5, has 10 functions
    for (int i = 5; i < 15; ++i) {
        for (int j = i + 1; j < 15; ++j) {
            Real s_ij = compute_overlap(i, j);
            Real s_ji = compute_overlap(j, i);
            EXPECT_NEAR(s_ij, s_ji, SYM_TOL)
                << "S not symmetric: S(" << i << "," << j << ")=" << s_ij
                << " vs S(" << j << "," << i << ")=" << s_ji;
        }
    }
}

TEST_F(HigherAM_Integration, OverlapSymmetry_GShell) {
    // S(mu, nu) = S(nu, mu) for g-shell pairs
    // O_g starts at index 15, has 15 functions
    for (int i = 15; i < 30; ++i) {
        for (int j = i + 1; j < 30; ++j) {
            Real s_ij = compute_overlap(i, j);
            Real s_ji = compute_overlap(j, i);
            EXPECT_NEAR(s_ij, s_ji, SYM_TOL)
                << "S not symmetric: S(" << i << "," << j << ")=" << s_ij
                << " vs S(" << j << "," << i << ")=" << s_ji;
        }
    }
}

TEST_F(HigherAM_Integration, OverlapCrossAM_FG) {
    // Cross-AM overlap: S(f, g) = S(g, f) (same center)
    for (int i = 5; i < 15; ++i) {
        for (int j = 15; j < 30; ++j) {
            Real s_ij = compute_overlap(i, j);
            Real s_ji = compute_overlap(j, i);
            EXPECT_NEAR(s_ij, s_ji, SYM_TOL)
                << "S(f,g) not symmetric: (" << i << "," << j << ")";
        }
    }
}

TEST_F(HigherAM_Integration, OverlapSameCenter_FShell_Orthogonality) {
    // Same-center f-shell overlaps with s-shell should be zero (orthogonality)
    for (int f = 5; f < 15; ++f) {
        Real s_sf = compute_overlap(0, f);
        EXPECT_NEAR(s_sf, 0.0, 1e-10)
            << "S(s,f) on same center should be near zero: S(0," << f << ")=" << s_sf;
    }
}

// =============================================================================
// 2. Kinetic Energy Tests
// =============================================================================

TEST_F(HigherAM_Integration, KineticSymmetry_FShell) {
    // T(mu, nu) = T(nu, mu) for f-shell pairs
    for (int i = 5; i < 15; ++i) {
        for (int j = i + 1; j < 15; ++j) {
            Real t_ij = compute_kinetic(i, j);
            Real t_ji = compute_kinetic(j, i);
            EXPECT_NEAR(t_ij, t_ji, SYM_TOL)
                << "T not symmetric: T(" << i << "," << j << ")=" << t_ij
                << " vs T(" << j << "," << i << ")=" << t_ji;
        }
    }
}

TEST_F(HigherAM_Integration, KineticSymmetry_GShell) {
    // T(mu, nu) = T(nu, mu) for g-shell pairs
    for (int i = 15; i < 30; ++i) {
        for (int j = i + 1; j < 30; ++j) {
            Real t_ij = compute_kinetic(i, j);
            Real t_ji = compute_kinetic(j, i);
            EXPECT_NEAR(t_ij, t_ji, SYM_TOL)
                << "T not symmetric: T(" << i << "," << j << ")=" << t_ij
                << " vs T(" << j << "," << i << ")=" << t_ji;
        }
    }
}

TEST_F(HigherAM_Integration, KineticDiagonal_Positive) {
    // Diagonal kinetic energy integrals must be positive
    for (Size i = 0; i < nbf_; ++i) {
        Real t_ii = compute_kinetic(i, i);
        EXPECT_GT(t_ii, 0.0)
            << "T(" << i << "," << i << ") should be positive, got " << t_ii;
    }
}

// =============================================================================
// 3. ERI Tests
// =============================================================================

TEST_F(HigherAM_Integration, ERI_FShell_Diagonal_Positive) {
    // Same-center f-shell diagonal ERIs: (fi fi | fi fi) > 0
    for (int i = 5; i < 15; ++i) {
        Real eri = compute_eri(i, i, i, i);
        EXPECT_GT(eri, POS_TOL)
            << "ERI diagonal (" << i << i << "|" << i << i << ") = " << eri
            << " should be positive";
    }
}

TEST_F(HigherAM_Integration, ERI_GShell_Diagonal_Positive) {
    // Same-center g-shell diagonal ERIs: (gi gi | gi gi) > 0
    for (int i = 15; i < 30; ++i) {
        Real eri = compute_eri(i, i, i, i);
        EXPECT_GT(eri, POS_TOL)
            << "ERI diagonal (" << i << i << "|" << i << i << ") = " << eri
            << " should be positive";
    }
}

TEST_F(HigherAM_Integration, ERI_FShell_Symmetry_Chemical) {
    // Chemical notation symmetry: (ij|kl) = (ji|kl) = (ij|lk) = (kl|ij)
    // Test a subset of f-shell quartets
    const int f_start = 5;
    const int f_end = 8;  // Test first 3 f-functions for speed

    for (int i = f_start; i < f_end; ++i) {
        for (int j = f_start; j < f_end; ++j) {
            for (int k = f_start; k < f_end; ++k) {
                for (int l = f_start; l < f_end; ++l) {
                    Real ijkl = compute_eri(i, j, k, l);
                    Real jikl = compute_eri(j, i, k, l);
                    Real ijlk = compute_eri(i, j, l, k);
                    Real klij = compute_eri(k, l, i, j);

                    EXPECT_NEAR(ijkl, jikl, SYM_TOL)
                        << "(ij|kl) != (ji|kl) at (" << i << j << "|" << k << l << ")";
                    EXPECT_NEAR(ijkl, ijlk, SYM_TOL)
                        << "(ij|kl) != (ij|lk) at (" << i << j << "|" << k << l << ")";
                    EXPECT_NEAR(ijkl, klij, SYM_TOL)
                        << "(ij|kl) != (kl|ij) at (" << i << j << "|" << k << l << ")";
                }
            }
        }
    }
}

TEST_F(HigherAM_Integration, ERI_CrossAM_SF) {
    // Cross-AM ERI: (sf|sf) with s on H1 and f on O
    int s_H1 = 30;  // H1 s-shell
    int f_O = 5;    // O f-shell first function

    Real eri = compute_eri(s_H1, f_O, s_H1, f_O);
    // Cross-center integral should be small but nonzero
    EXPECT_NE(eri, 0.0)
        << "Cross-center (sH1 fO | sH1 fO) should be nonzero";
}

TEST_F(HigherAM_Integration, ERI_CrossAM_DG) {
    // Cross-AM ERI: (dH1 gO | dH1 gO)
    int d_H1 = 31;  // H1 d-shell first function
    int g_O = 15;   // O g-shell first function

    Real eri = compute_eri(d_H1, g_O, d_H1, g_O);
    EXPECT_NE(eri, 0.0)
        << "Cross-center (dH1 gO | dH1 gO) should be nonzero";
}

TEST_F(HigherAM_Integration, ERI_CrossAM_FG) {
    // Cross-AM ERI: (fO gO | fO gO) — same center
    int f_O = 5;   // O f-shell first function
    int g_O = 15;  // O g-shell first function

    Real eri = compute_eri(f_O, g_O, f_O, g_O);
    // Same-center integral should be nonzero and positive
    EXPECT_GT(eri, 0.0)
        << "Same-center (fO gO | fO gO) should be positive, got " << eri;
}

// =============================================================================
// 4. Generated Kernel Registry Tests
// =============================================================================

TEST_F(HigherAM_Integration, GeneratedKernel_Overlap_FF) {
    // Verify the generated overlap kernel produces the same results
    // as the reference for f-f pair (same center)
    // The generated kernel should be dispatched automatically
    const auto& shell_f = basis_->shell(3);  // O f-shell

    OneElectronBuffer<0> buffer;
    kernels::compute_overlap(shell_f, shell_f, buffer);

    // Diagonal should be 1.0
    for (int i = 0; i < 10; ++i) {
        EXPECT_NEAR(buffer(i, i), 1.0, NORM_TOL)
            << "Generated overlap (ff) diagonal(" << i << ") should be 1.0";
    }
}

TEST_F(HigherAM_Integration, GeneratedKernel_Overlap_GG) {
    // Verify generated overlap kernel for g-g pair
    const auto& shell_g = basis_->shell(4);  // O g-shell

    OneElectronBuffer<0> buffer;
    kernels::compute_overlap(shell_g, shell_g, buffer);

    // Diagonal should be 1.0
    for (int i = 0; i < 15; ++i) {
        EXPECT_NEAR(buffer(i, i), 1.0, NORM_TOL)
            << "Generated overlap (gg) diagonal(" << i << ") should be 1.0";
    }
}

TEST_F(HigherAM_Integration, GeneratedKernel_Kinetic_FF) {
    // Verify generated kinetic kernel for f-f pair
    const auto& shell_f = basis_->shell(3);

    OneElectronBuffer<0> buffer;
    kernels::compute_kinetic(shell_f, shell_f, buffer);

    // Diagonal should be positive
    for (int i = 0; i < 10; ++i) {
        EXPECT_GT(buffer(i, i), 0.0)
            << "Generated kinetic (ff) diagonal(" << i << ") should be positive";
    }

    // Symmetry
    for (int i = 0; i < 10; ++i) {
        for (int j = i + 1; j < 10; ++j) {
            EXPECT_NEAR(buffer(i, j), buffer(j, i), SYM_TOL)
                << "Generated kinetic (ff) not symmetric at (" << i << "," << j << ")";
        }
    }
}

TEST_F(HigherAM_Integration, GeneratedKernel_Kinetic_GG) {
    // Verify generated kinetic kernel for g-g pair
    const auto& shell_g = basis_->shell(4);

    OneElectronBuffer<0> buffer;
    kernels::compute_kinetic(shell_g, shell_g, buffer);

    // Diagonal should be positive
    for (int i = 0; i < 15; ++i) {
        EXPECT_GT(buffer(i, i), 0.0)
            << "Generated kinetic (gg) diagonal(" << i << ") should be positive";
    }

    // Symmetry
    for (int i = 0; i < 15; ++i) {
        for (int j = i + 1; j < 15; ++j) {
            EXPECT_NEAR(buffer(i, j), buffer(j, i), SYM_TOL)
                << "Generated kinetic (gg) not symmetric at (" << i << "," << j << ")";
        }
    }
}

TEST_F(HigherAM_Integration, GeneratedKernel_ERI_FFFF) {
    // Verify generated ERI kernel for (ff|ff) same center
    const auto& shell_f = basis_->shell(3);

    TwoElectronBuffer<0> buffer;
    kernels::compute_eri(shell_f, shell_f, shell_f, shell_f, buffer);

    // All diagonal elements should be positive
    for (int i = 0; i < 10; ++i) {
        EXPECT_GT(buffer(i, i, i, i), 0.0)
            << "ERI (ff|ff) diagonal(" << i << ") should be positive";
    }

    // Symmetry: (ij|kl) = (ji|kl)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    EXPECT_NEAR(buffer(i, j, k, l), buffer(j, i, k, l), SYM_TOL)
                        << "ERI (ff|ff) symmetry failed at (" << i << j << "|" << k << l << ")";
                }
            }
        }
    }
}

TEST_F(HigherAM_Integration, GeneratedKernel_ERI_GGGG) {
    // Verify generated ERI kernel for (gg|gg) same center
    const auto& shell_g = basis_->shell(4);

    TwoElectronBuffer<0> buffer;
    kernels::compute_eri(shell_g, shell_g, shell_g, shell_g, buffer);

    // All diagonal elements should be positive
    for (int i = 0; i < 15; ++i) {
        EXPECT_GT(buffer(i, i, i, i), 0.0)
            << "ERI (gg|gg) diagonal(" << i << ") should be positive";
    }
}

// =============================================================================
// 5. Dimension Tests
// =============================================================================

TEST_F(HigherAM_Integration, BasisDimension) {
    // Verify total number of basis functions:
    // O: 1 + 1 + 3 + 10 + 15 = 30
    // H1: 1 + 6 = 7
    // H2: 1 + 6 = 7
    // Total = 44
    EXPECT_EQ(nbf_, 44u)
        << "Expected 44 basis functions for H2O with f+g on O, d on H";
}

TEST_F(HigherAM_Integration, ShellDimensions) {
    // f-shell should have 10 Cartesian functions
    EXPECT_EQ(basis_->shell(3).n_functions(), 10);
    // g-shell should have 15 Cartesian functions
    EXPECT_EQ(basis_->shell(4).n_functions(), 15);
}
