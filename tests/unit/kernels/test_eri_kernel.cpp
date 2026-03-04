// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_eri_kernel.cpp
/// @brief Unit tests for 4-center electron repulsion integral (ERI) kernel
///
/// Tests the CPU implementation of (ab|cd) integrals using Rys quadrature.
/// Covers: basic evaluation, permutation symmetry, Schwarz inequality,
/// higher angular momentum, and buffer dimensions.

#include <libaccint/kernels/eri_kernel.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/utils/constants.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace libaccint;

// Tolerances
constexpr Real TIGHT_TOL = 1e-12;
constexpr Real LOOSE_TOL = 1e-8;
constexpr Real SYMMETRY_TOL = 1e-13;

namespace {

Shell make_s_shell(Point3D center, Real alpha = 1.0, Real coeff = 1.0) {
    return Shell(0, center, {alpha}, {coeff});
}

Shell make_p_shell(Point3D center, Real alpha = 1.0, Real coeff = 1.0) {
    return Shell(1, center, {alpha}, {coeff});
}

Shell make_d_shell(Point3D center, Real alpha = 0.5, Real coeff = 1.0) {
    return Shell(2, center, {alpha}, {coeff});
}

Shell make_f_shell(Point3D center, Real alpha = 0.3, Real coeff = 1.0) {
    return Shell(3, center, {alpha}, {coeff});
}

Shell make_contracted_s_shell(Point3D center) {
    return Shell(0, center,
                 {3.425250914, 0.6239137298, 0.168855404},
                 {0.1543289673, 0.5353281423, 0.4446345422});
}

}  // anonymous namespace

// ============================================================================
// Basic (ss|ss) evaluation
// ============================================================================

TEST(ERIKernelTest, SsssBufferDimensions) {
    auto sa = make_s_shell({0.0, 0.0, 0.0});
    auto sb = make_s_shell({1.0, 0.0, 0.0});
    auto sc = make_s_shell({0.0, 1.0, 0.0});
    auto sd = make_s_shell({0.0, 0.0, 1.0});

    TwoElectronBuffer<0> buffer;
    kernels::compute_eri(sa, sb, sc, sd, buffer);

    EXPECT_EQ(buffer.na(), 1);
    EXPECT_EQ(buffer.nb(), 1);
    EXPECT_EQ(buffer.nc(), 1);
    EXPECT_EQ(buffer.nd(), 1);
}

TEST(ERIKernelTest, SsssSameCenter) {
    // (ss|ss) with all shells at origin — should be positive
    auto s = make_s_shell({0.0, 0.0, 0.0}, 1.0);

    TwoElectronBuffer<0> buffer;
    kernels::compute_eri(s, s, s, s, buffer);

    EXPECT_GT(buffer(0, 0, 0, 0), 0.0)
        << "Self-repulsion integral (ss|ss) must be positive";
}

TEST(ERIKernelTest, SsssFinite) {
    // (ss|ss) at well-separated centers
    auto sa = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto sb = make_s_shell({2.0, 0.0, 0.0}, 1.0);
    auto sc = make_s_shell({0.0, 2.0, 0.0}, 1.0);
    auto sd = make_s_shell({0.0, 0.0, 2.0}, 1.0);

    TwoElectronBuffer<0> buffer;
    kernels::compute_eri(sa, sb, sc, sd, buffer);

    Real val = buffer(0, 0, 0, 0);
    EXPECT_TRUE(std::isfinite(val)) << "ERI must be finite, got: " << val;
}

TEST(ERIKernelTest, SsssDecaysWithDistance) {
    // Coulomb repulsion decreases with distance between charge distributions
    auto sa = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto sb = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto sc_near = make_s_shell({1.0, 0.0, 0.0}, 1.0);
    auto sc_far  = make_s_shell({5.0, 0.0, 0.0}, 1.0);
    auto sd = make_s_shell({0.0, 0.0, 0.0}, 1.0);

    TwoElectronBuffer<0> buf_near, buf_far;
    kernels::compute_eri(sa, sb, sc_near, sd, buf_near);
    kernels::compute_eri(sa, sb, sc_far, sd, buf_far);

    // Near should have larger magnitude than far
    EXPECT_GT(std::abs(buf_near(0, 0, 0, 0)), std::abs(buf_far(0, 0, 0, 0)))
        << "ERI should decrease with separation";
}

TEST(ERIKernelTest, SsssContracted) {
    // Contracted s-shells should also produce positive/finite results
    auto sa = make_contracted_s_shell({0.0, 0.0, 0.0});
    auto sb = make_contracted_s_shell({1.4, 0.0, 0.0});  // ~H2 bond length

    TwoElectronBuffer<0> buffer;
    kernels::compute_eri(sa, sb, sa, sb, buffer);

    Real val = buffer(0, 0, 0, 0);
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_GT(val, 0.0) << "Contracted (ss|ss) must be positive";
}

// ============================================================================
// Permutation symmetry
// ============================================================================

TEST(ERIKernelTest, BraExchangeSymmetry) {
    // (ab|cd) = (ba|cd)
    auto sa = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto sb = make_p_shell({1.0, 0.0, 0.0}, 0.8);
    auto sc = make_s_shell({0.0, 1.0, 0.0}, 1.2);
    auto sd = make_s_shell({0.0, 0.0, 1.0}, 0.9);

    TwoElectronBuffer<0> buf_ab, buf_ba;
    kernels::compute_eri(sa, sb, sc, sd, buf_ab);
    kernels::compute_eri(sb, sa, sc, sd, buf_ba);

    // (ab|cd)[ia,ib,ic,id] = (ba|cd)[ib,ia,ic,id]
    for (int ia = 0; ia < buf_ab.na(); ++ia) {
        for (int ib = 0; ib < buf_ab.nb(); ++ib) {
            for (int ic = 0; ic < buf_ab.nc(); ++ic) {
                for (int id = 0; id < buf_ab.nd(); ++id) {
                    EXPECT_NEAR(buf_ab(ia, ib, ic, id),
                                buf_ba(ib, ia, ic, id), SYMMETRY_TOL)
                        << "Bra exchange symmetry violated at ("
                        << ia << "," << ib << "," << ic << "," << id << ")";
                }
            }
        }
    }
}

TEST(ERIKernelTest, KetExchangeSymmetry) {
    // (ab|cd) = (ab|dc)
    auto sa = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto sb = make_s_shell({1.0, 0.0, 0.0}, 1.0);
    auto sc = make_s_shell({0.0, 1.0, 0.0}, 1.2);
    auto sd = make_p_shell({0.0, 0.0, 1.0}, 0.9);

    TwoElectronBuffer<0> buf_cd, buf_dc;
    kernels::compute_eri(sa, sb, sc, sd, buf_cd);
    kernels::compute_eri(sa, sb, sd, sc, buf_dc);

    for (int ia = 0; ia < buf_cd.na(); ++ia) {
        for (int ib = 0; ib < buf_cd.nb(); ++ib) {
            for (int ic = 0; ic < buf_cd.nc(); ++ic) {
                for (int id = 0; id < buf_cd.nd(); ++id) {
                    EXPECT_NEAR(buf_cd(ia, ib, ic, id),
                                buf_dc(ia, ib, id, ic), SYMMETRY_TOL)
                        << "Ket exchange symmetry violated";
                }
            }
        }
    }
}

TEST(ERIKernelTest, BraKetExchangeSymmetry) {
    // (ab|cd) = (cd|ab)
    auto sa = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto sb = make_p_shell({1.0, 0.0, 0.0}, 0.8);
    auto sc = make_s_shell({0.0, 1.0, 0.0}, 1.2);
    auto sd = make_p_shell({0.0, 0.0, 1.0}, 0.9);

    TwoElectronBuffer<0> buf_abcd, buf_cdab;
    kernels::compute_eri(sa, sb, sc, sd, buf_abcd);
    kernels::compute_eri(sc, sd, sa, sb, buf_cdab);

    for (int ia = 0; ia < buf_abcd.na(); ++ia) {
        for (int ib = 0; ib < buf_abcd.nb(); ++ib) {
            for (int ic = 0; ic < buf_abcd.nc(); ++ic) {
                for (int id = 0; id < buf_abcd.nd(); ++id) {
                    EXPECT_NEAR(buf_abcd(ia, ib, ic, id),
                                buf_cdab(ic, id, ia, ib), SYMMETRY_TOL)
                        << "Bra-ket exchange symmetry violated";
                }
            }
        }
    }
}

// ============================================================================
// Schwarz inequality
// ============================================================================

TEST(ERIKernelTest, SchwarzInequality) {
    // |(ab|cd)|^2 <= (ab|ab) * (cd|cd)
    auto sa = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto sb = make_p_shell({1.0, 0.0, 0.0}, 0.8);
    auto sc = make_s_shell({0.0, 1.5, 0.0}, 1.2);
    auto sd = make_p_shell({0.0, 0.0, 1.5}, 0.9);

    TwoElectronBuffer<0> buf_abcd, buf_abab, buf_cdcd;
    kernels::compute_eri(sa, sb, sc, sd, buf_abcd);
    kernels::compute_eri(sa, sb, sa, sb, buf_abab);
    kernels::compute_eri(sc, sd, sc, sd, buf_cdcd);

    // Find max diagonal elements
    Real max_abab = 0.0;
    for (int ia = 0; ia < buf_abab.na(); ++ia) {
        for (int ib = 0; ib < buf_abab.nb(); ++ib) {
            max_abab = std::max(max_abab, buf_abab(ia, ib, ia, ib));
        }
    }

    Real max_cdcd = 0.0;
    for (int ic = 0; ic < buf_cdcd.nc(); ++ic) {
        for (int id = 0; id < buf_cdcd.nd(); ++id) {
            max_cdcd = std::max(max_cdcd, buf_cdcd(ic, id, ic, id));
        }
    }

    // For any element: |val|^2 <= max_abab * max_cdcd
    // (This is a simplified Schwarz check using max diag elements)
    Real schwarz_bound = std::sqrt(max_abab * max_cdcd);

    for (int ia = 0; ia < buf_abcd.na(); ++ia) {
        for (int ib = 0; ib < buf_abcd.nb(); ++ib) {
            for (int ic = 0; ic < buf_abcd.nc(); ++ic) {
                for (int id = 0; id < buf_abcd.nd(); ++id) {
                    EXPECT_LE(std::abs(buf_abcd(ia, ib, ic, id)),
                              schwarz_bound * 1.01)  // small tolerance
                        << "Schwarz inequality violated at ("
                        << ia << "," << ib << "," << ic << "," << id << ")";
                }
            }
        }
    }
}

// ============================================================================
// Higher angular momentum
// ============================================================================

TEST(ERIKernelTest, PpssBufferDimensions) {
    auto pa = make_p_shell({0.0, 0.0, 0.0});
    auto pb = make_p_shell({1.0, 0.0, 0.0});
    auto sc = make_s_shell({0.0, 1.0, 0.0});
    auto sd = make_s_shell({0.0, 0.0, 1.0});

    TwoElectronBuffer<0> buffer;
    kernels::compute_eri(pa, pb, sc, sd, buffer);

    EXPECT_EQ(buffer.na(), 3);
    EXPECT_EQ(buffer.nb(), 3);
    EXPECT_EQ(buffer.nc(), 1);
    EXPECT_EQ(buffer.nd(), 1);
}

TEST(ERIKernelTest, PpppSymmetry) {
    // Higher AM: (pp|pp) should still respect permutation symmetry
    auto pa = make_p_shell({0.0, 0.0, 0.0}, 1.0);
    auto pb = make_p_shell({1.0, 0.0, 0.0}, 0.8);
    auto pc = make_p_shell({0.0, 1.0, 0.0}, 1.2);
    auto pd = make_p_shell({0.0, 0.0, 1.0}, 0.9);

    TwoElectronBuffer<0> buf_abcd, buf_cdab;
    kernels::compute_eri(pa, pb, pc, pd, buf_abcd);
    kernels::compute_eri(pc, pd, pa, pb, buf_cdab);

    for (int ia = 0; ia < 3; ++ia) {
        for (int ib = 0; ib < 3; ++ib) {
            for (int ic = 0; ic < 3; ++ic) {
                for (int id = 0; id < 3; ++id) {
                    EXPECT_NEAR(buf_abcd(ia, ib, ic, id),
                                buf_cdab(ic, id, ia, ib), SYMMETRY_TOL)
                        << "(pp|pp) bra-ket symmetry at ("
                        << ia << "," << ib << "," << ic << "," << id << ")";
                }
            }
        }
    }
}

TEST(ERIKernelTest, DdssFinite) {
    auto da = make_d_shell({0.0, 0.0, 0.0});
    auto db = make_d_shell({1.5, 0.0, 0.0});
    auto sc = make_s_shell({0.0, 1.0, 0.0});
    auto sd = make_s_shell({0.0, 0.0, 1.0});

    TwoElectronBuffer<0> buffer;
    kernels::compute_eri(da, db, sc, sd, buffer);

    EXPECT_EQ(buffer.na(), 6);
    EXPECT_EQ(buffer.nb(), 6);

    for (int ia = 0; ia < 6; ++ia) {
        for (int ib = 0; ib < 6; ++ib) {
            EXPECT_TRUE(std::isfinite(buffer(ia, ib, 0, 0)))
                << "(dd|ss) element (" << ia << "," << ib << ") not finite";
        }
    }
}

TEST(ERIKernelTest, SpspSymmetry) {
    // Mixed AM: (sp|sp) bra-ket exchange
    auto sa = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto pb = make_p_shell({1.0, 0.0, 0.0}, 0.8);

    TwoElectronBuffer<0> buf1, buf2;
    kernels::compute_eri(sa, pb, sa, pb, buf1);
    kernels::compute_eri(sa, pb, sa, pb, buf2);

    // Self-repulsion (sp|sp) should be bra-ket symmetric
    for (int ia = 0; ia < buf1.na(); ++ia) {
        for (int ib = 0; ib < buf1.nb(); ++ib) {
            for (int ic = 0; ic < buf1.nc(); ++ic) {
                for (int id = 0; id < buf1.nd(); ++id) {
                    EXPECT_NEAR(buf1(ia, ib, ic, id),
                                buf1(ic, id, ia, ib), SYMMETRY_TOL)
                        << "(sp|sp) bra-ket symmetry";
                }
            }
        }
    }
}

TEST(ERIKernelTest, FsssFinite) {
    // F-type shell in bra
    auto fa = make_f_shell({0.0, 0.0, 0.0});
    auto sb = make_s_shell({1.0, 0.0, 0.0});
    auto sc = make_s_shell({0.0, 1.0, 0.0});
    auto sd = make_s_shell({0.0, 0.0, 1.0});

    TwoElectronBuffer<0> buffer;
    kernels::compute_eri(fa, sb, sc, sd, buffer);

    EXPECT_EQ(buffer.na(), 10);  // 10 Cartesian f-components

    for (int ia = 0; ia < 10; ++ia) {
        EXPECT_TRUE(std::isfinite(buffer(ia, 0, 0, 0)))
            << "(fs|ss) element " << ia << " not finite";
    }
}

// ============================================================================
// Coulomb diagonal positivity
// ============================================================================

TEST(ERIKernelTest, DiagonalPositivity) {
    // (ab|ab) diagonal elements (ia=ic, ib=id) should be non-negative
    auto sa = make_s_shell({0.0, 0.0, 0.0}, 1.0);
    auto pb = make_p_shell({1.0, 0.0, 0.0}, 0.8);

    TwoElectronBuffer<0> buffer;
    kernels::compute_eri(sa, pb, sa, pb, buffer);

    for (int ia = 0; ia < buffer.na(); ++ia) {
        for (int ib = 0; ib < buffer.nb(); ++ib) {
            EXPECT_GE(buffer(ia, ib, ia, ib), -TIGHT_TOL)
                << "Diagonal element (" << ia << "," << ib << "|"
                << ia << "," << ib << ") should be non-negative";
        }
    }
}

// ============================================================================
// Consistency: same-center shells
// ============================================================================

TEST(ERIKernelTest, SameCenterAllSShells) {
    // All shells at the same center: should give a well-defined positive value
    auto s = make_s_shell({0.0, 0.0, 0.0}, 1.5);

    TwoElectronBuffer<0> buffer;
    kernels::compute_eri(s, s, s, s, buffer);

    Real val = buffer(0, 0, 0, 0);
    EXPECT_GT(val, 0.0);
    EXPECT_TRUE(std::isfinite(val));
}
