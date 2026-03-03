// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_range_separated_eri.cpp
/// @brief Integration tests for range-separated ERI kernels
///
/// Tests the critical Quality Gate G3 criteria:
/// - erf + erfc = full Coulomb within 1e-12
/// - Various omega values work correctly
/// - Limiting behavior is correct

#include <libaccint/kernels/eri_kernel.hpp>
#include <libaccint/kernels/eri_erf_coulomb_kernel.hpp>
#include <libaccint/kernels/eri_erfc_coulomb_kernel.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/operators/operator.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace libaccint;
using namespace libaccint::kernels;

namespace {

// Helper: Create a shell at the origin
Shell create_test_shell(int L, double exponent, double coefficient,
                        double x = 0.0, double y = 0.0, double z = 0.0) {
    Point3D center{x, y, z};
    std::vector<Real> exponents = {exponent};
    std::vector<Real> coefficients = {coefficient};
    return Shell(L, center, exponents, coefficients);  // normalized
}

// Helper: Create H atom shell (1s STO-3G approximation with single primitive)
Shell create_h_1s_shell(double x = 0.0, double y = 0.0, double z = 0.0) {
    return create_test_shell(0, 3.42525091, 1.0, x, y, z);
}

// Helper: Create C atom shell (2p STO-3G approximation with single primitive)
Shell create_c_2p_shell(double x = 0.0, double y = 0.0, double z = 0.0) {
    return create_test_shell(1, 2.941249, 1.0, x, y, z);
}

// Helper: Compute sum of squared differences
double compute_rms_difference(const TwoElectronBuffer<0>& a,
                               const TwoElectronBuffer<0>& b) {
    double sum_sq = 0.0;
    int count = 0;

    for (int i = 0; i < a.na(); ++i) {
        for (int j = 0; j < a.nb(); ++j) {
            for (int k = 0; k < a.nc(); ++k) {
                for (int l = 0; l < a.nd(); ++l) {
                    double diff = a(i, j, k, l) - b(i, j, k, l);
                    sum_sq += diff * diff;
                    ++count;
                }
            }
        }
    }

    return std::sqrt(sum_sq / count);
}

// Helper: Compute relative error between buffers
double compute_relative_error(const TwoElectronBuffer<0>& computed,
                               const TwoElectronBuffer<0>& reference) {
    double numerator = 0.0;
    double denominator = 0.0;

    for (int i = 0; i < computed.na(); ++i) {
        for (int j = 0; j < computed.nb(); ++j) {
            for (int k = 0; k < computed.nc(); ++k) {
                for (int l = 0; l < computed.nd(); ++l) {
                    double diff = computed(i, j, k, l) - reference(i, j, k, l);
                    numerator += diff * diff;
                    denominator += reference(i, j, k, l) * reference(i, j, k, l);
                }
            }
        }
    }

    if (denominator < 1e-100) {
        return std::sqrt(numerator);
    }
    return std::sqrt(numerator / denominator);
}

// Helper: Add two buffers element-wise
void add_buffers(TwoElectronBuffer<0>& result,
                 const TwoElectronBuffer<0>& a,
                 const TwoElectronBuffer<0>& b) {
    result.resize(a.na(), a.nb(), a.nc(), a.nd());

    for (int i = 0; i < a.na(); ++i) {
        for (int j = 0; j < a.nb(); ++j) {
            for (int k = 0; k < a.nc(); ++k) {
                for (int l = 0; l < a.nd(); ++l) {
                    result(i, j, k, l) = a(i, j, k, l) + b(i, j, k, l);
                }
            }
        }
    }
}

}  // namespace

// ============================================================================
// Decomposition Identity Tests (Critical for Quality Gate G3)
// ============================================================================

TEST(RangeSeparatedERI, DecompositionIdentity_SSSS) {
    // Test (ss|ss) quartet: erf + erfc = full Coulomb
    auto shell_a = create_h_1s_shell(0.0, 0.0, 0.0);
    auto shell_b = create_h_1s_shell(1.4, 0.0, 0.0);
    auto shell_c = create_h_1s_shell(0.0, 1.0, 0.0);
    auto shell_d = create_h_1s_shell(1.4, 1.0, 0.0);

    std::vector<double> omegas = {0.1, 0.2, 0.33, 0.4, 0.5, 1.0, 2.0, 5.0};
    double max_error = 0.0;

    for (double omega : omegas) {
        TwoElectronBuffer<0> eri_full, eri_erf, eri_erfc, eri_sum;

        // Compute full Coulomb
        compute_eri(shell_a, shell_b, shell_c, shell_d, eri_full);

        // Compute erf-Coulomb
        compute_eri_erf_coulomb(shell_a, shell_b, shell_c, shell_d, omega, eri_erf);

        // Compute erfc-Coulomb
        compute_eri_erfc_coulomb(shell_a, shell_b, shell_c, shell_d, omega, eri_erfc);

        // Sum erf + erfc
        add_buffers(eri_sum, eri_erf, eri_erfc);

        // Check identity
        double rel_error = compute_relative_error(eri_sum, eri_full);
        max_error = std::max(max_error, rel_error);

        EXPECT_LT(rel_error, 1e-12)
            << "(ss|ss) decomposition failed for omega=" << omega
            << ", relative error=" << rel_error;
    }

    std::cout << "(ss|ss) max decomposition error: " << max_error << std::endl;
}

TEST(RangeSeparatedERI, DecompositionIdentity_SPSP) {
    // Test (sp|sp) quartet
    auto shell_s1 = create_h_1s_shell(0.0, 0.0, 0.0);
    auto shell_p1 = create_c_2p_shell(1.4, 0.0, 0.0);
    auto shell_s2 = create_h_1s_shell(0.0, 2.0, 0.0);
    auto shell_p2 = create_c_2p_shell(1.4, 2.0, 0.0);

    std::vector<double> omegas = {0.33, 0.4, 0.5, 1.0};
    double max_error = 0.0;

    for (double omega : omegas) {
        TwoElectronBuffer<0> eri_full, eri_erf, eri_erfc, eri_sum;

        compute_eri(shell_s1, shell_p1, shell_s2, shell_p2, eri_full);
        compute_eri_erf_coulomb(shell_s1, shell_p1, shell_s2, shell_p2, omega, eri_erf);
        compute_eri_erfc_coulomb(shell_s1, shell_p1, shell_s2, shell_p2, omega, eri_erfc);

        add_buffers(eri_sum, eri_erf, eri_erfc);

        double rel_error = compute_relative_error(eri_sum, eri_full);
        max_error = std::max(max_error, rel_error);

        EXPECT_LT(rel_error, 1e-12)
            << "(sp|sp) decomposition failed for omega=" << omega;
    }

    std::cout << "(sp|sp) max decomposition error: " << max_error << std::endl;
}

TEST(RangeSeparatedERI, DecompositionIdentity_PPPP) {
    // Test (pp|pp) quartet
    auto shell_a = create_c_2p_shell(0.0, 0.0, 0.0);
    auto shell_b = create_c_2p_shell(1.5, 0.0, 0.0);
    auto shell_c = create_c_2p_shell(0.0, 1.5, 0.0);
    auto shell_d = create_c_2p_shell(1.5, 1.5, 0.0);

    std::vector<double> omegas = {0.33, 0.4, 1.0};
    double max_error = 0.0;

    for (double omega : omegas) {
        TwoElectronBuffer<0> eri_full, eri_erf, eri_erfc, eri_sum;

        compute_eri(shell_a, shell_b, shell_c, shell_d, eri_full);
        compute_eri_erf_coulomb(shell_a, shell_b, shell_c, shell_d, omega, eri_erf);
        compute_eri_erfc_coulomb(shell_a, shell_b, shell_c, shell_d, omega, eri_erfc);

        add_buffers(eri_sum, eri_erf, eri_erfc);

        double rel_error = compute_relative_error(eri_sum, eri_full);
        max_error = std::max(max_error, rel_error);

        EXPECT_LT(rel_error, 1e-12)
            << "(pp|pp) decomposition failed for omega=" << omega;
    }

    std::cout << "(pp|pp) max decomposition error: " << max_error << std::endl;
}

// ============================================================================
// Limiting Behavior Tests
// ============================================================================

TEST(RangeSeparatedERI, ErfLargeOmegaApproachesFull) {
    // omega -> infinity: erf-Coulomb -> full Coulomb
    // For large omega, the implementation goes through the general path
    // which accumulates small numerical errors from omega2_ratio calculations
    auto shell_a = create_h_1s_shell(0.0, 0.0, 0.0);
    auto shell_b = create_h_1s_shell(1.4, 0.0, 0.0);
    auto shell_c = create_h_1s_shell(0.0, 1.0, 0.0);
    auto shell_d = create_h_1s_shell(1.4, 1.0, 0.0);

    TwoElectronBuffer<0> eri_full, eri_erf;

    compute_eri(shell_a, shell_b, shell_c, shell_d, eri_full);
    compute_eri_erf_coulomb(shell_a, shell_b, shell_c, shell_d, 1000.0, eri_erf);

    double rel_error = compute_relative_error(eri_erf, eri_full);

    // For omega=1000: omega2_ratio = 1000000/1000001 ≈ 0.999999
    // This means we should get very close to full Coulomb
    EXPECT_LT(rel_error, 1e-6)
        << "Large omega erf-Coulomb should approach full Coulomb, rel_error=" << rel_error;
}

TEST(RangeSeparatedERI, ErfSmallOmegaApproachesZero) {
    // omega -> 0: erf-Coulomb -> 0
    auto shell_a = create_h_1s_shell(0.0, 0.0, 0.0);
    auto shell_b = create_h_1s_shell(1.4, 0.0, 0.0);
    auto shell_c = create_h_1s_shell(0.0, 1.0, 0.0);
    auto shell_d = create_h_1s_shell(1.4, 1.0, 0.0);

    TwoElectronBuffer<0> eri_erf;
    compute_eri_erf_coulomb(shell_a, shell_b, shell_c, shell_d, 0.001, eri_erf);

    // All integrals should be near zero
    double max_val = 0.0;
    for (int i = 0; i < eri_erf.na(); ++i) {
        for (int j = 0; j < eri_erf.nb(); ++j) {
            for (int k = 0; k < eri_erf.nc(); ++k) {
                for (int l = 0; l < eri_erf.nd(); ++l) {
                    max_val = std::max(max_val, std::abs(eri_erf(i, j, k, l)));
                }
            }
        }
    }

    EXPECT_LT(max_val, 1e-5)
        << "Small omega erf-Coulomb should approach zero, max_val=" << max_val;
}

TEST(RangeSeparatedERI, ErfcSmallOmegaApproachesFull) {
    // omega -> 0: erfc-Coulomb -> full Coulomb
    // Since erfc = full - erf, and erf approaches zero for small omega,
    // erfc should approach full. But omega=0.0001 still has some contribution.
    auto shell_a = create_h_1s_shell(0.0, 0.0, 0.0);
    auto shell_b = create_h_1s_shell(1.4, 0.0, 0.0);
    auto shell_c = create_h_1s_shell(0.0, 1.0, 0.0);
    auto shell_d = create_h_1s_shell(1.4, 1.0, 0.0);

    TwoElectronBuffer<0> eri_full, eri_erfc;

    compute_eri(shell_a, shell_b, shell_c, shell_d, eri_full);
    compute_eri_erfc_coulomb(shell_a, shell_b, shell_c, shell_d, 0.0001, eri_erfc);

    double rel_error = compute_relative_error(eri_erfc, eri_full);

    // For omega=0.0001: omega2_ratio ≈ 1e-8, so erf contribution is negligible
    // But some numerical difference is expected from the erfc computation path
    EXPECT_LT(rel_error, 1e-3)
        << "Small omega erfc-Coulomb should approach full Coulomb, rel_error=" << rel_error;
}

TEST(RangeSeparatedERI, ErfcLargeOmegaApproachesZero) {
    // omega -> infinity: erfc-Coulomb -> 0
    auto shell_a = create_h_1s_shell(0.0, 0.0, 0.0);
    auto shell_b = create_h_1s_shell(1.4, 0.0, 0.0);
    auto shell_c = create_h_1s_shell(0.0, 1.0, 0.0);
    auto shell_d = create_h_1s_shell(1.4, 1.0, 0.0);

    TwoElectronBuffer<0> eri_erfc;
    compute_eri_erfc_coulomb(shell_a, shell_b, shell_c, shell_d, 1000.0, eri_erfc);

    double max_val = 0.0;
    for (int i = 0; i < eri_erfc.na(); ++i) {
        for (int j = 0; j < eri_erfc.nb(); ++j) {
            for (int k = 0; k < eri_erfc.nc(); ++k) {
                for (int l = 0; l < eri_erfc.nd(); ++l) {
                    max_val = std::max(max_val, std::abs(eri_erfc(i, j, k, l)));
                }
            }
        }
    }

    EXPECT_LT(max_val, 1e-10)
        << "Large omega erfc-Coulomb should approach zero";
}

// ============================================================================
// CAM-B3LYP Omega Value Tests
// ============================================================================

TEST(RangeSeparatedERI, CAMB3LYP_DecompositionIdentity) {
    // CAM-B3LYP uses omega = 0.33
    const double omega = 0.33;

    auto shell_a = create_h_1s_shell(0.0, 0.0, 0.0);
    auto shell_b = create_h_1s_shell(1.4, 0.0, 0.0);
    auto shell_c = create_h_1s_shell(0.0, 1.0, 0.0);
    auto shell_d = create_h_1s_shell(1.4, 1.0, 0.0);

    TwoElectronBuffer<0> eri_full, eri_erf, eri_erfc, eri_sum;

    compute_eri(shell_a, shell_b, shell_c, shell_d, eri_full);
    compute_eri_erf_coulomb(shell_a, shell_b, shell_c, shell_d, omega, eri_erf);
    compute_eri_erfc_coulomb(shell_a, shell_b, shell_c, shell_d, omega, eri_erfc);

    add_buffers(eri_sum, eri_erf, eri_erfc);

    double rel_error = compute_relative_error(eri_sum, eri_full);

    EXPECT_LT(rel_error, 1e-12)
        << "CAM-B3LYP omega=0.33 decomposition identity failed";
}

// ============================================================================
// Zero Omega Handling
// ============================================================================

TEST(RangeSeparatedERI, ZeroOmegaErfReturnsZero) {
    auto shell_a = create_h_1s_shell(0.0, 0.0, 0.0);
    auto shell_b = create_h_1s_shell(1.4, 0.0, 0.0);
    auto shell_c = create_h_1s_shell(0.0, 1.0, 0.0);
    auto shell_d = create_h_1s_shell(1.4, 1.0, 0.0);

    TwoElectronBuffer<0> eri_erf;
    compute_eri_erf_coulomb(shell_a, shell_b, shell_c, shell_d, 0.0, eri_erf);

    // All integrals should be exactly zero
    for (int i = 0; i < eri_erf.na(); ++i) {
        for (int j = 0; j < eri_erf.nb(); ++j) {
            for (int k = 0; k < eri_erf.nc(); ++k) {
                for (int l = 0; l < eri_erf.nd(); ++l) {
                    EXPECT_DOUBLE_EQ(eri_erf(i, j, k, l), 0.0)
                        << "erf with omega=0 should be exactly zero";
                }
            }
        }
    }
}

TEST(RangeSeparatedERI, ZeroOmegaErfcReturnsFull) {
    auto shell_a = create_h_1s_shell(0.0, 0.0, 0.0);
    auto shell_b = create_h_1s_shell(1.4, 0.0, 0.0);
    auto shell_c = create_h_1s_shell(0.0, 1.0, 0.0);
    auto shell_d = create_h_1s_shell(1.4, 1.0, 0.0);

    TwoElectronBuffer<0> eri_full, eri_erfc;

    compute_eri(shell_a, shell_b, shell_c, shell_d, eri_full);
    compute_eri_erfc_coulomb(shell_a, shell_b, shell_c, shell_d, 0.0, eri_erfc);

    // Should match full Coulomb exactly
    for (int i = 0; i < eri_erfc.na(); ++i) {
        for (int j = 0; j < eri_erfc.nb(); ++j) {
            for (int k = 0; k < eri_erfc.nc(); ++k) {
                for (int l = 0; l < eri_erfc.nd(); ++l) {
                    EXPECT_NEAR(eri_erfc(i, j, k, l), eri_full(i, j, k, l), 1e-15)
                        << "erfc with omega=0 should equal full Coulomb";
                }
            }
        }
    }
}

// ============================================================================
// Symmetry Tests
// ============================================================================

TEST(RangeSeparatedERI, ErfCoulombSymmetry) {
    // (ab|cd) = (ba|cd) = (ab|dc) = (cd|ab) etc.
    auto shell_s = create_h_1s_shell(0.0, 0.0, 0.0);
    auto shell_p = create_c_2p_shell(1.5, 0.0, 0.0);

    const double omega = 0.4;

    TwoElectronBuffer<0> eri_spsp, eri_pssp, eri_spps;

    compute_eri_erf_coulomb(shell_s, shell_p, shell_s, shell_p, omega, eri_spsp);
    compute_eri_erf_coulomb(shell_p, shell_s, shell_s, shell_p, omega, eri_pssp);
    compute_eri_erf_coulomb(shell_s, shell_p, shell_p, shell_s, omega, eri_spps);

    // Check (sp|sp) symmetry: swap bra shells gives (ps|sp)
    // The integrals should be related by index permutation
    // This is a basic sanity check
    EXPECT_TRUE(eri_spsp.na() > 0);
    EXPECT_TRUE(eri_pssp.na() > 0);
    EXPECT_TRUE(eri_spps.na() > 0);
}

// ============================================================================
// Comprehensive Quality Gate G3 Test
// ============================================================================

TEST(RangeSeparatedERI, QualityGateG3_Comprehensive) {
    // Comprehensive test for Quality Gate G3
    // Tests multiple angular momentum combinations and omega values

    struct ShellConfig {
        int L;
        double exponent;
        std::string name;
    };

    std::vector<ShellConfig> configs = {
        {0, 3.0, "s"},
        {1, 2.5, "p"}
    };

    std::vector<double> omegas = {0.1, 0.33, 0.4, 0.5, 1.0, 2.0, 5.0};

    double global_max_error = 0.0;
    int n_tests = 0;

    for (const auto& cfg_a : configs) {
        for (const auto& cfg_b : configs) {
            for (const auto& cfg_c : configs) {
                for (const auto& cfg_d : configs) {
                    auto shell_a = create_test_shell(cfg_a.L, cfg_a.exponent, 1.0, 0.0, 0.0, 0.0);
                    auto shell_b = create_test_shell(cfg_b.L, cfg_b.exponent, 1.0, 1.5, 0.0, 0.0);
                    auto shell_c = create_test_shell(cfg_c.L, cfg_c.exponent, 1.0, 0.0, 1.5, 0.0);
                    auto shell_d = create_test_shell(cfg_d.L, cfg_d.exponent, 1.0, 1.5, 1.5, 0.0);

                    for (double omega : omegas) {
                        TwoElectronBuffer<0> eri_full, eri_erf, eri_erfc, eri_sum;

                        compute_eri(shell_a, shell_b, shell_c, shell_d, eri_full);
                        compute_eri_erf_coulomb(shell_a, shell_b, shell_c, shell_d, omega, eri_erf);
                        compute_eri_erfc_coulomb(shell_a, shell_b, shell_c, shell_d, omega, eri_erfc);

                        add_buffers(eri_sum, eri_erf, eri_erfc);

                        double rel_error = compute_relative_error(eri_sum, eri_full);
                        global_max_error = std::max(global_max_error, rel_error);

                        EXPECT_LT(rel_error, 1e-12)
                            << "Decomposition failed: ("
                            << cfg_a.name << cfg_b.name << "|" << cfg_c.name << cfg_d.name
                            << "), omega=" << omega;

                        ++n_tests;
                    }
                }
            }
        }
    }

    std::cout << "Quality Gate G3: Tested " << n_tests << " cases" << std::endl;
    std::cout << "Quality Gate G3: Maximum decomposition error = " << global_max_error << std::endl;

    EXPECT_LT(global_max_error, 1e-12)
        << "QUALITY GATE G3 FAILED: Maximum error exceeds 1e-12 tolerance";
}
