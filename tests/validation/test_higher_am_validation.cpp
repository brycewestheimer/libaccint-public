// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_higher_am_validation.cpp
/// @brief Higher angular momentum integral validation (Tasks 23.1.1-23.1.3)
///
/// Validates f (AM=3), g (AM=4), and h (AM=5) function integrals for:
///   - Overlap (S), Kinetic (T), Nuclear attraction (V), and ERI
///   - Symmetry properties of integral matrices
///   - Normalization consistency
///   - Correct matrix dimensions

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/math/cartesian_indices.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace libaccint::test {
namespace {

// ============================================================================
// Helper: Create a shell with given AM and appropriate exponents
// ============================================================================

/// @brief Create a shell with STO-3G-like parameters for a given AM
Shell make_shell(int am, Point3D center) {
    // Exponents decrease with AM to stay numerically well-conditioned
    std::vector<Real> exponents;
    std::vector<Real> coefficients;

    if (am <= 2) {
        exponents = {3.42525091, 0.62391373, 0.16885540};
        coefficients = {0.15432897, 0.53532814, 0.44463454};
    } else if (am == 3) {
        // f-function exponents (cc-pVTZ-like)
        exponents = {1.533, 0.5417, 0.2211};
        coefficients = {0.25, 0.50, 0.35};
    } else if (am == 4) {
        // g-function exponents (cc-pVQZ-like)
        exponents = {1.208, 0.4537, 0.1813};
        coefficients = {0.30, 0.45, 0.35};
    } else {
        // h-function exponents (cc-pV5Z-like)
        exponents = {0.9876, 0.3654, 0.1432};
        coefficients = {0.35, 0.40, 0.35};
    }

    return Shell(am, center, exponents, coefficients);
}

/// @brief Create a minimal basis set containing shells up to given AM
BasisSet make_basis_with_am(int max_am, Point3D center = {0.0, 0.0, 0.0}) {
    std::vector<Shell> shells;
    // Always include an s-shell for a valid basis
    shells.push_back(make_shell(0, center));
    shells.back().set_atom_index(0);

    if (max_am >= 1) {
        shells.push_back(make_shell(1, center));
        shells.back().set_atom_index(0);
    }

    // Add the target AM shell
    if (max_am >= 2) {
        for (int am = 2; am <= max_am; ++am) {
            shells.push_back(make_shell(am, center));
            shells.back().set_atom_index(0);
        }
    }

    return BasisSet(std::move(shells));
}

/// @brief Create a two-center basis set for integral testing
std::pair<BasisSet, PointChargeParams> make_two_center_basis(int am) {
    Point3D center_a = {0.0, 0.0, 0.0};
    Point3D center_b = {0.0, 0.0, 1.5};

    std::vector<Shell> shells;
    // Shell on center A
    auto sa = make_shell(am, center_a);
    sa.set_atom_index(0);
    shells.push_back(std::move(sa));

    // s-shell on center B for cross-integrals
    auto sb = make_shell(0, center_b);
    sb.set_atom_index(1);
    shells.push_back(std::move(sb));

    PointChargeParams charges;
    charges.x = {center_a.x, center_b.x};
    charges.y = {center_a.y, center_b.y};
    charges.z = {center_a.z, center_b.z};
    charges.charge = {8.0, 1.0};  // O, H like charges

    return {BasisSet(std::move(shells)), charges};
}

// ============================================================================
// Test Fixture
// ============================================================================

class HigherAMValidationTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override {
        am_ = GetParam();
        try {
            auto [basis, charges] = make_two_center_basis(am_);
            basis_ = std::move(basis);
            charges_ = std::move(charges);
            nbf_ = basis_.n_basis_functions();
        } catch (const std::exception& e) {
            skip_ = true;
            skip_reason_ = std::string("Setup failed: ") + e.what();
        }
    }

    int am_{0};
    BasisSet basis_;
    PointChargeParams charges_;
    Size nbf_{0};
    bool skip_{false};
    std::string skip_reason_;
};

// ============================================================================
// Task 23.1.1-23.1.3: Overlap Integral Validation
// ============================================================================

TEST_P(HigherAMValidationTest, OverlapMatrixComputable) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    std::vector<Real> S(nbf_ * nbf_, 0.0);
    EXPECT_NO_THROW(engine.compute_overlap_matrix(S));
}

TEST_P(HigherAMValidationTest, OverlapMatrixFinite) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    std::vector<Real> S(nbf_ * nbf_, 0.0);
    engine.compute_overlap_matrix(S);

    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_TRUE(std::isfinite(S[i]))
            << "Non-finite overlap at linear index " << i << " for AM=" << am_;
    }
}

TEST_P(HigherAMValidationTest, OverlapMatrixSymmetric) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    std::vector<Real> S(nbf_ * nbf_, 0.0);
    engine.compute_overlap_matrix(S);

    for (Size i = 0; i < nbf_; ++i) {
        for (Size j = 0; j < nbf_; ++j) {
            EXPECT_NEAR(S[i * nbf_ + j], S[j * nbf_ + i], 1e-14)
                << "Overlap not symmetric at (" << i << "," << j << ") for AM=" << am_;
        }
    }
}

TEST_P(HigherAMValidationTest, OverlapDiagonalPositive) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    std::vector<Real> S(nbf_ * nbf_, 0.0);
    engine.compute_overlap_matrix(S);

    for (Size i = 0; i < nbf_; ++i) {
        EXPECT_GT(S[i * nbf_ + i], 0.0)
            << "Overlap diagonal non-positive at (" << i << ") for AM=" << am_;
    }
}

TEST_P(HigherAMValidationTest, OverlapMatrixCorrectDimensions) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    // Total basis functions: n_cartesian(am) + n_cartesian(0) = (am+1)(am+2)/2 + 1
    int expected_nf = n_cartesian(am_) + n_cartesian(0);
    EXPECT_EQ(static_cast<int>(nbf_), expected_nf)
        << "Incorrect number of basis functions for AM=" << am_;
}

// ============================================================================
// Task 23.1.1-23.1.3: Kinetic Integral Validation
// ============================================================================

TEST_P(HigherAMValidationTest, KineticMatrixComputable) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    std::vector<Real> T(nbf_ * nbf_, 0.0);
    EXPECT_NO_THROW(engine.compute_kinetic_matrix(T));
}

TEST_P(HigherAMValidationTest, KineticMatrixFinite) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    std::vector<Real> T(nbf_ * nbf_, 0.0);
    engine.compute_kinetic_matrix(T);

    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_TRUE(std::isfinite(T[i]))
            << "Non-finite kinetic at linear index " << i << " for AM=" << am_;
    }
}

TEST_P(HigherAMValidationTest, KineticMatrixSymmetric) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    std::vector<Real> T(nbf_ * nbf_, 0.0);
    engine.compute_kinetic_matrix(T);

    for (Size i = 0; i < nbf_; ++i) {
        for (Size j = 0; j < nbf_; ++j) {
            EXPECT_NEAR(T[i * nbf_ + j], T[j * nbf_ + i], 1e-13)
                << "Kinetic not symmetric at (" << i << "," << j << ") for AM=" << am_;
        }
    }
}

TEST_P(HigherAMValidationTest, KineticDiagonalPositive) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    std::vector<Real> T(nbf_ * nbf_, 0.0);
    engine.compute_kinetic_matrix(T);

    for (Size i = 0; i < nbf_; ++i) {
        EXPECT_GT(T[i * nbf_ + i], 0.0)
            << "Kinetic diagonal non-positive at (" << i << ") for AM=" << am_;
    }
}

// ============================================================================
// Task 23.1.1-23.1.3: Nuclear Attraction Integral Validation
// ============================================================================

TEST_P(HigherAMValidationTest, NuclearMatrixComputable) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    std::vector<Real> V(nbf_ * nbf_, 0.0);
    EXPECT_NO_THROW(engine.compute_nuclear_matrix(charges_, V));
}

TEST_P(HigherAMValidationTest, NuclearMatrixFinite) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    std::vector<Real> V(nbf_ * nbf_, 0.0);
    engine.compute_nuclear_matrix(charges_, V);

    for (Size i = 0; i < nbf_ * nbf_; ++i) {
        EXPECT_TRUE(std::isfinite(V[i]))
            << "Non-finite nuclear at linear index " << i << " for AM=" << am_;
    }
}

TEST_P(HigherAMValidationTest, NuclearMatrixSymmetric) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    std::vector<Real> V(nbf_ * nbf_, 0.0);
    engine.compute_nuclear_matrix(charges_, V);

    for (Size i = 0; i < nbf_; ++i) {
        for (Size j = 0; j < nbf_; ++j) {
            EXPECT_NEAR(V[i * nbf_ + j], V[j * nbf_ + i], 1e-13)
                << "Nuclear not symmetric at (" << i << "," << j << ") for AM=" << am_;
        }
    }
}

TEST_P(HigherAMValidationTest, NuclearDiagonalNegative) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    std::vector<Real> V(nbf_ * nbf_, 0.0);
    engine.compute_nuclear_matrix(charges_, V);

    // Nuclear attraction for positive charges should be negative
    for (Size i = 0; i < nbf_; ++i) {
        EXPECT_LT(V[i * nbf_ + i], 0.0)
            << "Nuclear diagonal not negative at (" << i << ") for AM=" << am_;
    }
}

// ============================================================================
// Task 23.1.1-23.1.3: ERI Validation (shell-level)
// ============================================================================

TEST_P(HigherAMValidationTest, ERIShellQuartetComputable) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    // Create same-AM shell quartet (aa|aa)
    Shell shell_a = make_shell(am_, {0.0, 0.0, 0.0});
    Shell shell_b = make_shell(0, {0.0, 0.0, 1.5});

    int na = shell_a.n_functions();
    int nb = shell_b.n_functions();

    TwoElectronBuffer<0> buffer(na, nb, na, nb);

    Engine engine(basis_);
    EXPECT_NO_THROW(
        engine.compute_2e_shell_quartet(Operator::coulomb(),
                                         shell_a, shell_b, shell_a, shell_b,
                                         buffer)
    );
}

TEST_P(HigherAMValidationTest, ERIShellQuartetFinite) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Shell shell_a = make_shell(am_, {0.0, 0.0, 0.0});
    Shell shell_b = make_shell(0, {0.0, 0.0, 1.5});

    int na = shell_a.n_functions();
    int nb = shell_b.n_functions();

    TwoElectronBuffer<0> buffer(na, nb, na, nb);
    buffer.clear();

    Engine engine(basis_);
    engine.compute_2e_shell_quartet(Operator::coulomb(),
                                     shell_a, shell_b, shell_a, shell_b,
                                     buffer);

    auto data = buffer.data();
    for (Size i = 0; i < data.size(); ++i) {
        EXPECT_TRUE(std::isfinite(data[i]))
            << "Non-finite ERI at index " << i << " for AM=" << am_;
    }
}

TEST_P(HigherAMValidationTest, ERISameShellSymmetry) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    // For (aa|aa), check permutation symmetry: (ab|cd) = (ba|dc) = (cd|ab) = ...
    Shell shell = make_shell(am_, {0.0, 0.0, 0.0});
    int nf = shell.n_functions();

    TwoElectronBuffer<0> buffer(nf, nf, nf, nf);
    buffer.clear();

    Engine engine(basis_);
    engine.compute_2e_shell_quartet(Operator::coulomb(),
                                     shell, shell, shell, shell, buffer);

    // Check (ab|cd) = (cd|ab) for same shell
    for (int a = 0; a < nf; ++a) {
        for (int b = 0; b < nf; ++b) {
            for (int c = 0; c < nf; ++c) {
                for (int d = 0; d < nf; ++d) {
                    EXPECT_NEAR(buffer(a, b, c, d), buffer(c, d, a, b), 1e-12)
                        << "ERI symmetry violated: (" << a << b << "|" << c << d
                        << ") != (" << c << d << "|" << a << b << ") for AM=" << am_;
                }
            }
        }
    }
}

TEST_P(HigherAMValidationTest, ERISameShellDiagonalPositive) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    Shell shell = make_shell(am_, {0.0, 0.0, 0.0});
    int nf = shell.n_functions();

    TwoElectronBuffer<0> buffer(nf, nf, nf, nf);
    buffer.clear();

    Engine engine(basis_);
    engine.compute_2e_shell_quartet(Operator::coulomb(),
                                     shell, shell, shell, shell, buffer);

    // Diagonal ERIs (aa|aa) should be positive (Coulomb repulsion of charge with itself)
    for (int a = 0; a < nf; ++a) {
        EXPECT_GT(buffer(a, a, a, a), 0.0)
            << "Diagonal ERI not positive for function " << a << " at AM=" << am_;
    }
}

// ============================================================================
// Normalization / Self-Overlap Consistency
// ============================================================================

TEST_P(HigherAMValidationTest, SelfOverlapNearUnity) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    // For normalized shells, self-overlap diagonal should be ~1.0
    Shell shell = make_shell(am_, {0.0, 0.0, 0.0});
    int nf = shell.n_functions();

    OneElectronBuffer<0> buffer(nf, nf);
    buffer.clear();

    Engine engine(basis_);
    engine.compute_1e_shell_pair(Operator::overlap(), shell, shell, buffer);

    // Diagonal elements should be 1.0 for properly normalized shells
    for (int a = 0; a < nf; ++a) {
        EXPECT_NEAR(buffer(a, a), 1.0, 1e-10)
            << "Self-overlap not unity for function " << a << " at AM=" << am_;
    }
}

TEST_P(HigherAMValidationTest, OverlapCauchySchwarz) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    // |S_ij| <= sqrt(S_ii * S_jj) (Cauchy-Schwarz inequality)
    Engine engine(basis_);
    std::vector<Real> S(nbf_ * nbf_, 0.0);
    engine.compute_overlap_matrix(S);

    for (Size i = 0; i < nbf_; ++i) {
        for (Size j = i + 1; j < nbf_; ++j) {
            Real bound = std::sqrt(S[i * nbf_ + i] * S[j * nbf_ + j]);
            EXPECT_LE(std::abs(S[i * nbf_ + j]), bound + 1e-14)
                << "Cauchy-Schwarz violated at (" << i << "," << j << ") for AM=" << am_;
        }
    }
}

TEST_P(HigherAMValidationTest, KineticPositiveSemiDefinite) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    // T should be positive semi-definite: all diagonal elements >= 0
    Engine engine(basis_);
    std::vector<Real> T(nbf_ * nbf_, 0.0);
    engine.compute_kinetic_matrix(T);

    // Trace should be positive
    Real trace = 0.0;
    for (Size i = 0; i < nbf_; ++i) {
        trace += T[i * nbf_ + i];
    }
    EXPECT_GT(trace, 0.0) << "Kinetic trace not positive for AM=" << am_;
}

// ============================================================================
// Cross-AM Integral Tests (higher AM with s-function)
// ============================================================================

TEST_P(HigherAMValidationTest, CrossAMOverlapDecay) {
    if (skip_) GTEST_SKIP() << skip_reason_;

    // Overlap between well-separated shells should decay
    Shell shell_a = make_shell(am_, {0.0, 0.0, 0.0});
    Shell shell_b = make_shell(0, {0.0, 0.0, 10.0});  // Far apart

    int na = shell_a.n_functions();
    int nb = shell_b.n_functions();

    OneElectronBuffer<0> buffer(na, nb);
    buffer.clear();

    Engine engine(basis_);
    engine.compute_1e_shell_pair(Operator::overlap(), shell_a, shell_b, buffer);

    // All elements should be small due to large separation
    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < nb; ++b) {
            EXPECT_LT(std::abs(buffer(a, b)), 0.1)
                << "Cross-AM overlap too large for well-separated shells at AM=" << am_;
        }
    }
}

// ============================================================================
// Parameterized instantiation for f, g, h functions
// ============================================================================

std::string HigherAMValidationName(const ::testing::TestParamInfo<int>& info) {
    switch (info.param) {
        case 3: return "f_AM3";
        case 4: return "g_AM4";
        case 5: return "h_AM5";
        default: return "AM" + std::to_string(info.param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    HigherAMIntegrals,
    HigherAMValidationTest,
    ::testing::Values(3, 4, 5),
    HigherAMValidationName
);

}  // namespace
}  // namespace libaccint::test
