// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_eri_erfc_coulomb.cpp
/// @brief Unit tests for erfc-attenuated Coulomb ERI kernel
///
/// Tests the CPU implementation of erfc(ω*r₁₂)/r₁₂ integrals and verifies
/// the fundamental identity: erf + erfc = full Coulomb.

#include <libaccint/kernels/eri_erfc_coulomb_kernel.hpp>
#include <libaccint/kernels/eri_erf_coulomb_kernel.hpp>
#include <libaccint/kernels/eri_kernel.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <gtest/gtest.h>
#include <cmath>

using namespace libaccint;

constexpr Real TIGHT_TOL = 1e-12;
constexpr Real IDENTITY_TOL = 1e-10;
constexpr Real SYMMETRY_TOL = 1e-13;
constexpr Real CONVERGENCE_TOL = 1e-6;

namespace {

Shell make_s_shell(Point3D center, Real alpha = 1.0) {
    return Shell(0, center, {alpha}, {1.0});
}

Shell make_p_shell(Point3D center, Real alpha = 1.0) {
    return Shell(1, center, {alpha}, {1.0});
}

}  // anonymous namespace

// ============================================================================
// Basic erfc-Coulomb evaluation
// ============================================================================

TEST(ERIErfcCoulombTest, SsssPositive) {
    auto s = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    TwoElectronBuffer<0> buffer;
    kernels::compute_eri_erfc_coulomb(s, s, s, s, 1.0, buffer);

    EXPECT_GT(buffer(0, 0, 0, 0), 0.0)
        << "erfc-Coulomb (ss|ss) must be positive";
}

TEST(ERIErfcCoulombTest, SsssFinite) {
    auto sa = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto sb = make_s_shell({2.0, 0.0, 0.0}, 1.0);

    TwoElectronBuffer<0> buffer;
    kernels::compute_eri_erfc_coulomb(sa, sb, sa, sb, 0.5, buffer);

    EXPECT_TRUE(std::isfinite(buffer(0, 0, 0, 0)));
}

// ============================================================================
// Symmetry
// ============================================================================

TEST(ERIErfcCoulombTest, BraKetSymmetry) {
    auto sa = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto pb = make_p_shell({1.0, 0.0, 0.0}, 0.8);

    TwoElectronBuffer<0> buf_abcd, buf_cdab;
    kernels::compute_eri_erfc_coulomb(sa, pb, sa, pb, 0.5, buf_abcd);

    // (ab|ab) should be symmetric under bra-ket exchange
    for (int ia = 0; ia < buf_abcd.na(); ++ia) {
        for (int ib = 0; ib < buf_abcd.nb(); ++ib) {
            for (int ic = 0; ic < buf_abcd.nc(); ++ic) {
                for (int id = 0; id < buf_abcd.nd(); ++id) {
                    EXPECT_NEAR(buf_abcd(ia, ib, ic, id),
                                buf_abcd(ic, id, ia, ib), SYMMETRY_TOL);
                }
            }
        }
    }
}

// ============================================================================
// CRITICAL: erf + erfc = full Coulomb identity
// ============================================================================

TEST(ERIErfcCoulombTest, ErfPlusErfcEqualsFull_Ssss) {
    auto sa = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto sb = make_s_shell({1.5, 0.0, 0.0}, 1.0);
    auto sc = make_s_shell({0.0, 1.5, 0.0}, 1.0);
    auto sd = make_s_shell({0.0, 0.0, 1.5}, 1.0);

    Real omega = 0.8;
    TwoElectronBuffer<0> buf_full, buf_erf, buf_erfc;
    kernels::compute_eri(sa, sb, sc, sd, buf_full);
    kernels::compute_eri_erf_coulomb(sa, sb, sc, sd, omega, buf_erf);
    kernels::compute_eri_erfc_coulomb(sa, sb, sc, sd, omega, buf_erfc);

    Real sum = buf_erf(0, 0, 0, 0) + buf_erfc(0, 0, 0, 0);
    EXPECT_NEAR(sum, buf_full(0, 0, 0, 0), IDENTITY_TOL)
        << "erf + erfc should equal full Coulomb for (ss|ss)";
}

TEST(ERIErfcCoulombTest, ErfPlusErfcEqualsFull_Ppss) {
    auto pa = make_p_shell({0.0, 0.0, 0.0}, 1.0);
    auto pb = make_p_shell({1.0, 0.0, 0.0}, 0.8);
    auto sc = make_s_shell({0.0, 1.0, 0.0}, 1.2);
    auto sd = make_s_shell({0.0, 0.0, 1.0}, 0.9);

    Real omega = 0.5;
    TwoElectronBuffer<0> buf_full, buf_erf, buf_erfc;
    kernels::compute_eri(pa, pb, sc, sd, buf_full);
    kernels::compute_eri_erf_coulomb(pa, pb, sc, sd, omega, buf_erf);
    kernels::compute_eri_erfc_coulomb(pa, pb, sc, sd, omega, buf_erfc);

    for (int ia = 0; ia < buf_full.na(); ++ia) {
        for (int ib = 0; ib < buf_full.nb(); ++ib) {
            Real sum = buf_erf(ia, ib, 0, 0) + buf_erfc(ia, ib, 0, 0);
            EXPECT_NEAR(sum, buf_full(ia, ib, 0, 0), IDENTITY_TOL)
                << "erf + erfc identity failed at (" << ia << "," << ib << ")";
        }
    }
}

TEST(ERIErfcCoulombTest, ErfPlusErfcEqualsFull_MultipleOmega) {
    // Verify identity for several omega values
    auto s = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto s2 = make_s_shell({1.0, 0.0, 0.0}, 1.0);

    TwoElectronBuffer<0> buf_full;
    kernels::compute_eri(s, s2, s, s2, buf_full);
    const Real full_val = buf_full(0, 0, 0, 0);

    for (Real omega : {0.1, 0.3, 0.5, 1.0, 2.0, 5.0}) {
        TwoElectronBuffer<0> buf_erf, buf_erfc;
        kernels::compute_eri_erf_coulomb(s, s2, s, s2, omega, buf_erf);
        kernels::compute_eri_erfc_coulomb(s, s2, s, s2, omega, buf_erfc);

        Real sum = buf_erf(0, 0, 0, 0) + buf_erfc(0, 0, 0, 0);
        EXPECT_NEAR(sum, full_val, IDENTITY_TOL)
            << "Identity failed at omega=" << omega;
    }
}

// ============================================================================
// Limit behavior
// ============================================================================

TEST(ERIErfcCoulombTest, SmallOmegaApproachesFullCoulomb) {
    // erfc(ω*r)/r → 1/r as ω → 0 (since erfc(0)=1)
    auto s = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto s2 = make_s_shell({1.0, 0.0, 0.0}, 1.0);

    TwoElectronBuffer<0> buf_full, buf_erfc;
    kernels::compute_eri(s, s2, s, s2, buf_full);
    kernels::compute_eri_erfc_coulomb(s, s2, s, s2, 0.001, buf_erfc);

    EXPECT_NEAR(buf_erfc(0, 0, 0, 0), buf_full(0, 0, 0, 0), 1e-3)
        << "erfc-Coulomb should approach full Coulomb for small omega";
}

TEST(ERIErfcCoulombTest, LargeOmegaApproachesZero) {
    // erfc(ω*r)/r → 0 as ω → ∞
    auto s = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto s2 = make_s_shell({1.0, 0.0, 0.0}, 1.0);

    TwoElectronBuffer<0> buf_erfc;
    kernels::compute_eri_erfc_coulomb(s, s2, s, s2, 100.0, buf_erfc);

    EXPECT_LT(std::abs(buf_erfc(0, 0, 0, 0)), 0.01)
        << "erfc-Coulomb should be near zero for large omega";
}
