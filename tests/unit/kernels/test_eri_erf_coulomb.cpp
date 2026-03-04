// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_eri_erf_coulomb.cpp
/// @brief Unit tests for erf-attenuated Coulomb ERI kernel
///
/// Tests the CPU implementation of erf(ω*r₁₂)/r₁₂ integrals.

#include <libaccint/kernels/eri_erf_coulomb_kernel.hpp>
#include <libaccint/kernels/eri_kernel.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <gtest/gtest.h>
#include <cmath>

using namespace libaccint;

constexpr Real TIGHT_TOL = 1e-12;
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
// Basic erf-Coulomb evaluation
// ============================================================================

TEST(ERIErfCoulombTest, SsssPositive) {
    auto s = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    TwoElectronBuffer<0> buffer;
    kernels::compute_eri_erf_coulomb(s, s, s, s, 1.0, buffer);

    EXPECT_GT(buffer(0, 0, 0, 0), 0.0)
        << "erf-Coulomb (ss|ss) must be positive";
}

TEST(ERIErfCoulombTest, SsssFinite) {
    auto sa = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto sb = make_s_shell({2.0, 0.0, 0.0}, 1.0);
    auto sc = make_s_shell({0.0, 2.0, 0.0}, 1.0);
    auto sd = make_s_shell({0.0, 0.0, 2.0}, 1.0);

    TwoElectronBuffer<0> buffer;
    kernels::compute_eri_erf_coulomb(sa, sb, sc, sd, 0.5, buffer);

    EXPECT_TRUE(std::isfinite(buffer(0, 0, 0, 0)));
}

// ============================================================================
// erf-Coulomb symmetry
// ============================================================================

TEST(ERIErfCoulombTest, BraKetSymmetry) {
    auto sa = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto pb = make_p_shell({1.0, 0.0, 0.0}, 0.8);
    auto sc = make_s_shell({0.0, 1.0, 0.0}, 1.2);
    auto pd = make_p_shell({0.0, 0.0, 1.0}, 0.9);

    TwoElectronBuffer<0> buf_abcd, buf_cdab;
    kernels::compute_eri_erf_coulomb(sa, pb, sc, pd, 0.5, buf_abcd);
    kernels::compute_eri_erf_coulomb(sc, pd, sa, pb, 0.5, buf_cdab);

    for (int ia = 0; ia < buf_abcd.na(); ++ia) {
        for (int ib = 0; ib < buf_abcd.nb(); ++ib) {
            for (int ic = 0; ic < buf_abcd.nc(); ++ic) {
                for (int id = 0; id < buf_abcd.nd(); ++id) {
                    EXPECT_NEAR(buf_abcd(ia, ib, ic, id),
                                buf_cdab(ic, id, ia, ib), SYMMETRY_TOL)
                        << "erf-Coulomb bra-ket symmetry violated";
                }
            }
        }
    }
}

// ============================================================================
// Large omega limit: erf(ω*r)/r → 1/r (full Coulomb)
// ============================================================================

TEST(ERIErfCoulombTest, LargeOmegaApproachesFullCoulomb) {
    auto sa = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto sb = make_s_shell({1.0, 0.0, 0.0}, 1.0);
    auto sc = make_s_shell({0.0, 1.0, 0.0}, 1.0);
    auto sd = make_s_shell({0.0, 0.0, 1.0}, 1.0);

    TwoElectronBuffer<0> buf_full, buf_erf;
    kernels::compute_eri(sa, sb, sc, sd, buf_full);
    kernels::compute_eri_erf_coulomb(sa, sb, sc, sd, 100.0, buf_erf);

    // At large omega, erf(100*r)/r ≈ 1/r
    // Convergence depends on omega and exponents; use relative tolerance
    EXPECT_NEAR(buf_erf(0, 0, 0, 0), buf_full(0, 0, 0, 0), 1e-4)
        << "erf-Coulomb should approach full Coulomb for large omega";
}

// ============================================================================
// Small omega limit: erf(ω*r)/r → 0
// ============================================================================

TEST(ERIErfCoulombTest, SmallOmegaApproachesZero) {
    auto sa = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto sb = make_s_shell({1.0, 0.0, 0.0}, 1.0);

    TwoElectronBuffer<0> buf_small;
    kernels::compute_eri_erf_coulomb(sa, sb, sa, sb, 0.001, buf_small);

    // At very small omega, erf(0.001*r)/r → 0
    EXPECT_LT(std::abs(buf_small(0, 0, 0, 0)), 0.01)
        << "erf-Coulomb should be near zero for very small omega";
}

// ============================================================================
// Monotonicity in omega
// ============================================================================

TEST(ERIErfCoulombTest, MonotonicallyIncreasingWithOmega) {
    // erf(ω*r)/r is monotonically increasing in ω
    auto s = make_s_shell({0.0, 0.0, 0.0}, 1.0);

    TwoElectronBuffer<0> buf1, buf2, buf3;
    kernels::compute_eri_erf_coulomb(s, s, s, s, 0.5, buf1);
    kernels::compute_eri_erf_coulomb(s, s, s, s, 1.0, buf2);
    kernels::compute_eri_erf_coulomb(s, s, s, s, 2.0, buf3);

    EXPECT_LT(buf1(0, 0, 0, 0), buf2(0, 0, 0, 0));
    EXPECT_LT(buf2(0, 0, 0, 0), buf3(0, 0, 0, 0));
}
