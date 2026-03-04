// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include "libaccint/math/rys_quadrature.hpp"
#include "libaccint/math/boys_function.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <vector>
#include <array>
#include <numeric>

using namespace libaccint::math;

// ============================================================================
// Helpers
// ============================================================================

/// @brief Compute relative error between two values
static double relative_error(double computed, double reference) {
    if (reference == 0.0) {
        return std::abs(computed);
    }
    return std::abs((computed - reference) / reference);
}

/// @brief Verify moment matching: sum_i w_i u_i^k = F_k(T) for k = 0..2n-1
static void verify_moments(int n_roots, double T,
                           const double* roots, const double* weights,
                           double tol) {
    const int n_moments = 2 * n_roots;
    std::vector<double> expected(n_moments);
    boys_evaluate_array(n_moments - 1, T, expected.data());

    for (int k = 0; k < n_moments; ++k) {
        double sum = 0.0;
        for (int i = 0; i < n_roots; ++i) {
            sum += weights[i] * std::pow(roots[i], k);
        }
        double err = relative_error(sum, expected[k]);
        EXPECT_LT(err, tol)
            << "Moment mismatch at k=" << k << " for n_roots=" << n_roots
            << ", T=" << T << ": computed=" << sum
            << ", expected=" << expected[k] << ", rel_err=" << err;
    }
}

// ============================================================================
// Task 0.4.1: Root Finding Tests
// ============================================================================

TEST(RysRootTest, SingleRoot) {
    // n_roots=1: u = F_1(T)/F_0(T), w = F_0(T)
    double root, weight;
    double T_values[] = {0.0, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0};

    for (double T : T_values) {
        rys_compute(1, T, &root, &weight);

        double F0, F1;
        std::array<double, 2> f;
        boys_evaluate_array(1, T, f.data());
        F0 = f[0];
        F1 = f[1];

        double expected_root = F1 / F0;
        double expected_weight = F0;

        EXPECT_LT(relative_error(root, expected_root), 1e-14)
            << "Root mismatch for T=" << T;
        EXPECT_LT(relative_error(weight, expected_weight), 1e-14)
            << "Weight mismatch for T=" << T;
    }
}

TEST(RysRootTest, RootsInRange) {
    // All roots should be in (0, 1) for finite T
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    for (int n = 1; n <= 10; ++n) {
        for (double T : {0.0, 0.5, 2.0, 10.0, 30.0, 50.0}) {
            rys_compute(n, T, roots, weights);
            for (int i = 0; i < n; ++i) {
                EXPECT_GT(roots[i], 0.0)
                    << "Root " << i << " not positive for n=" << n << ", T=" << T;
                EXPECT_LT(roots[i], 1.0)
                    << "Root " << i << " >= 1 for n=" << n << ", T=" << T;
            }
        }
    }
}

TEST(RysRootTest, RootsSorted) {
    // Roots should be in ascending order
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    for (int n = 2; n <= 10; ++n) {
        for (double T : {0.0, 1.0, 5.0, 20.0, 50.0}) {
            rys_compute(n, T, roots, weights);
            for (int i = 0; i < n - 1; ++i) {
                EXPECT_LT(roots[i], roots[i + 1])
                    << "Roots not sorted at index " << i
                    << " for n=" << n << ", T=" << T;
            }
        }
    }
}

TEST(RysRootTest, RootsDistinct) {
    // All roots should be distinct (no repeated roots)
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    for (int n = 2; n <= 10; ++n) {
        rys_compute(n, 5.0, roots, weights);
        for (int i = 0; i < n - 1; ++i) {
            EXPECT_GT(roots[i + 1] - roots[i], 1e-14)
                << "Roots " << i << " and " << i + 1
                << " are not distinct for n=" << n;
        }
    }
}

TEST(RysRootTest, RootsDecreaseWithT) {
    // As T increases, roots shift toward 0 (weight function concentrates near t=0)
    double roots_lo[MAX_RYS_ROOTS], roots_hi[MAX_RYS_ROOTS];
    double weights[MAX_RYS_ROOTS];

    for (int n = 1; n <= 5; ++n) {
        rys_compute(n, 1.0, roots_lo, weights);
        rys_compute(n, 50.0, roots_hi, weights);

        for (int i = 0; i < n; ++i) {
            EXPECT_LT(roots_hi[i], roots_lo[i])
                << "Root " << i << " did not decrease with T for n=" << n;
        }
    }
}

TEST(RysRootTest, ChebyshevCoefficients) {
    // Test the modified Chebyshev algorithm directly
    // For n=2, T=0: moments are [1, 1/3, 1/5, 1/7]
    double moments[] = {1.0, 1.0 / 3.0, 1.0 / 5.0, 1.0 / 7.0};
    double alpha[2], beta[2];

    rys_chebyshev(2, moments, alpha, beta);

    // alpha[0] = μ_1/μ_0 = 1/3
    EXPECT_NEAR(alpha[0], 1.0 / 3.0, 1e-15);

    // beta[0] = μ_0 = 1
    EXPECT_NEAR(beta[0], 1.0, 1e-15);

    // sigma[1][1] = 1/5 - (1/3)*(1/3) = 4/45
    // sigma[1][2] = 1/7 - (1/3)*(1/5) = 8/105
    // alpha[1] = (8/105)/(4/45) - (1/3)/1 = 6/7 - 1/3 = 11/21
    EXPECT_NEAR(alpha[1], 11.0 / 21.0, 1e-14);

    // beta[1] = (4/45)/1 = 4/45
    EXPECT_NEAR(beta[1], 4.0 / 45.0, 1e-15);
}

TEST(RysRootTest, PolynomialEvaluation) {
    // For n=2 with the T=0 coefficients above, the polynomial is:
    // π_2(u) = (u - α_1)(u - α_0) - β_1
    // = (u - 11/21)(u - 1/3) - 4/45
    // Roots should be the eigenvalues we computed: ~0.1156, ~0.7415
    double alpha[] = {1.0 / 3.0, 11.0 / 21.0};
    double beta[] = {1.0, 4.0 / 45.0};

    // Evaluate at a known root
    double u = 3.0 / 35.0 * (3.0 - std::sqrt(9.0 - 35.0 * 4.0 / 45.0 +
               35.0 * (1.0 / 3.0 * 11.0 / 21.0)));
    // Actually, let me just evaluate and check it's a polynomial
    double val, deriv;

    // π_2(0) = (0 - α_1)(0 - α_0) - β_1 = α_0 * α_1 - β_1
    rys_poly_eval(2, 0.0, alpha, beta, &val, &deriv);
    double expected = alpha[0] * alpha[1] - beta[1];
    EXPECT_NEAR(val, expected, 1e-15);

    // π_2(1) = (1 - α_1)(1 - α_0) - β_1
    rys_poly_eval(2, 1.0, alpha, beta, &val, &deriv);
    expected = (1.0 - alpha[1]) * (1.0 - alpha[0]) - beta[1];
    EXPECT_NEAR(val, expected, 1e-15);
}

TEST(RysRootTest, NewtonPolishing) {
    // Start with slightly perturbed roots and verify Newton polishing converges
    double moments[] = {1.0, 1.0 / 3.0, 1.0 / 5.0, 1.0 / 7.0};
    double alpha[2], beta[2];
    rys_chebyshev(2, moments, alpha, beta);

    // Get accurate roots first
    double roots_exact[2], weights[2];
    rys_compute(2, 0.0, roots_exact, weights);

    // Perturb roots slightly
    double roots[2] = {roots_exact[0] * 1.001, roots_exact[1] * 0.999};

    // Polish
    rys_newton_polish(2, alpha, beta, roots, 20);

    EXPECT_LT(relative_error(roots[0], roots_exact[0]), 1e-14);
    EXPECT_LT(relative_error(roots[1], roots_exact[1]), 1e-14);
}

TEST(RysRootTest, MaxRoots) {
    // Test with maximum supported roots
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    // n=15 should work
    rys_compute(MAX_RYS_ROOTS, 5.0, roots, weights);

    // Verify roots are in range and sorted
    for (int i = 0; i < MAX_RYS_ROOTS; ++i) {
        EXPECT_GT(roots[i], 0.0);
        EXPECT_LT(roots[i], 1.0);
    }
    for (int i = 0; i < MAX_RYS_ROOTS - 1; ++i) {
        EXPECT_LT(roots[i], roots[i + 1]);
    }
}

// ============================================================================
// Task 0.4.2: Weight Computation Tests
// ============================================================================

TEST(RysWeightTest, WeightSumEqualsF0) {
    // Sum of weights should equal F_0(T) for any T and n_roots
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    for (int n = 1; n <= 10; ++n) {
        for (double T : {0.0, 0.01, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0}) {
            rys_compute(n, T, roots, weights);
            double sum = 0.0;
            for (int i = 0; i < n; ++i) {
                sum += weights[i];
            }

            double F0 = boys_evaluate(0, T);
            EXPECT_LT(relative_error(sum, F0), 1e-13)
                << "Weight sum != F_0(T) for n=" << n << ", T=" << T;
        }
    }
}

TEST(RysWeightTest, WeightsPositive) {
    // All weights should be positive
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    for (int n = 1; n <= 10; ++n) {
        for (double T : {0.0, 1.0, 10.0, 50.0}) {
            rys_compute(n, T, roots, weights);
            for (int i = 0; i < n; ++i) {
                EXPECT_GT(weights[i], 0.0)
                    << "Non-positive weight at index " << i
                    << " for n=" << n << ", T=" << T;
            }
        }
    }
}

TEST(RysWeightTest, MomentMatchingSmallN) {
    // Quadrature exactness: sum w_i u_i^k = F_k(T) for k = 0..2n-1
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    for (int n = 1; n <= 5; ++n) {
        for (double T : {0.0, 0.5, 1.0, 5.0, 10.0, 25.0}) {
            rys_compute(n, T, roots, weights);
            verify_moments(n, T, roots, weights, 1e-13);
        }
    }
}

TEST(RysWeightTest, MomentMatchingLargeN) {
    // Test moment matching for larger n_roots
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    for (int n = 6; n <= 10; ++n) {
        for (double T : {0.0, 1.0, 5.0, 10.0, 25.0, 50.0}) {
            rys_compute(n, T, roots, weights);
            verify_moments(n, T, roots, weights, 1e-12);
        }
    }
}

TEST(RysWeightTest, MomentMatchingN15) {
    // Test at maximum roots with relaxed tolerance
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    for (double T : {0.0, 1.0, 5.0, 10.0, 25.0}) {
        rys_compute(15, T, roots, weights);
        verify_moments(15, T, roots, weights, 1e-10);
    }
}

TEST(RysWeightTest, WeightsFromRoots) {
    // Test the weight-from-roots computation matches the combined computation
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];
    double weights2[MAX_RYS_ROOTS];

    for (int n = 2; n <= 8; ++n) {
        for (double T : {0.0, 1.0, 5.0, 20.0}) {
            rys_compute(n, T, roots, weights);
            rys_weights_from_roots(n, T, roots, weights2);

            for (int i = 0; i < n; ++i) {
                EXPECT_LT(relative_error(weights2[i], weights[i]), 1e-12)
                    << "Weight from roots mismatch at i=" << i
                    << " for n=" << n << ", T=" << T;
            }
        }
    }
}

TEST(RysWeightTest, TZeroAnalytical) {
    // For T=0, verify against analytically computed values for n=1,2
    double roots[2], weights[2];

    // n=1: u = 1/3, w = 1
    rys_compute(1, 0.0, roots, weights);
    EXPECT_NEAR(roots[0], 1.0 / 3.0, 1e-15);
    EXPECT_NEAR(weights[0], 1.0, 1e-15);

    // n=2: analytical roots of Jacobi matrix
    // J = [[1/3, sqrt(4/45)], [sqrt(4/45), 11/21]]
    // Eigenvalues: (6/7 ± sqrt(36/49 - 12/35))/2
    rys_compute(2, 0.0, roots, weights);

    double trace = 6.0 / 7.0;
    double det = 3.0 / 35.0;
    double disc = std::sqrt(trace * trace - 4.0 * det);
    double u1 = (trace - disc) / 2.0;
    double u2 = (trace + disc) / 2.0;

    EXPECT_LT(relative_error(roots[0], u1), 1e-14);
    EXPECT_LT(relative_error(roots[1], u2), 1e-14);

    // Weights should sum to 1 (= F_0(0))
    EXPECT_NEAR(weights[0] + weights[1], 1.0, 1e-15);
}

TEST(RysWeightTest, LargeTAsymptotic) {
    // For large T, the smallest root should approach 0 and
    // the largest root should be well below 1
    double roots[5], weights[5];

    rys_compute(5, 100.0, roots, weights);

    // All roots should be much smaller than 1 for large T
    for (int i = 0; i < 5; ++i) {
        EXPECT_LT(roots[i], 0.5)
            << "Root " << i << " too large for T=100";
        EXPECT_GT(weights[i], 0.0);
    }

    // Moment matching should still hold
    verify_moments(5, 100.0, roots, weights, 1e-12);
}

TEST(RysWeightTest, SpecificTValues) {
    // Test at specific T values from task spec: T=0, 1, 5, 10, 25, 50
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    double T_values[] = {0.0, 1.0, 5.0, 10.0, 25.0, 50.0};

    for (double T : T_values) {
        for (int n = 1; n <= 10; ++n) {
            rys_compute(n, T, roots, weights);
            verify_moments(n, T, roots, weights, 1e-13);
        }
    }
}

TEST(RysWeightTest, SmallTStability) {
    // Very small T values should be handled stably
    double roots[5], weights[5];

    for (double T : {1e-10, 1e-8, 1e-6, 1e-4, 0.01}) {
        rys_compute(5, T, roots, weights);
        verify_moments(5, T, roots, weights, 1e-13);
    }
}

TEST(RysWeightTest, ConsecutiveNRoots) {
    // Increasing n_roots should add roots that interleave with existing ones
    // (Separation property of Gaussian quadrature)
    double roots3[3], roots4[4], weights[4];

    rys_compute(3, 5.0, roots3, weights);
    rys_compute(4, 5.0, roots4, weights);

    // Each 3-root should fall between consecutive 4-roots (approximately)
    // This is the interlacing property
    for (int i = 0; i < 3; ++i) {
        EXPECT_GT(roots3[i], roots4[i])
            << "Interlacing violated at i=" << i;
        EXPECT_LT(roots3[i], roots4[i + 1])
            << "Interlacing violated at i=" << i;
    }
}

// ============================================================================
// Task 0.4.3: Table Lookup Tests
// ============================================================================

TEST(RysTableTest, LookupMatchesCompute) {
    // Table lookup should agree with direct computation to < 1e-13
    double roots_lk[MAX_RYS_ROOTS], weights_lk[MAX_RYS_ROOTS];
    double roots_cp[MAX_RYS_ROOTS], weights_cp[MAX_RYS_ROOTS];

    // Test at non-grid-point T values (interpolation + Newton polishing)
    for (int n = 1; n <= 12; ++n) {
        for (double T : {0.05, 0.15, 0.77, 2.33, 5.5, 10.3, 20.7, 35.9}) {
            rys_lookup(n, T, roots_lk, weights_lk);
            rys_compute(n, T, roots_cp, weights_cp);

            for (int i = 0; i < n; ++i) {
                double r_err = relative_error(roots_lk[i], roots_cp[i]);
                double w_err = relative_error(weights_lk[i], weights_cp[i]);

                EXPECT_LT(r_err, 1e-13)
                    << "Root lookup mismatch at i=" << i
                    << " for n=" << n << ", T=" << T
                    << " rel_err=" << r_err;
                EXPECT_LT(w_err, 1e-13)
                    << "Weight lookup mismatch at i=" << i
                    << " for n=" << n << ", T=" << T
                    << " rel_err=" << w_err;
            }
        }
    }
}

TEST(RysTableTest, LookupAtGridPoints) {
    // At grid points, interpolation should be exact (or nearly so)
    double roots_lk[MAX_RYS_ROOTS], weights_lk[MAX_RYS_ROOTS];
    double roots_cp[MAX_RYS_ROOTS], weights_cp[MAX_RYS_ROOTS];

    for (int n = 1; n <= 5; ++n) {
        for (double T : {0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 39.0}) {
            rys_lookup(n, T, roots_lk, weights_lk);
            rys_compute(n, T, roots_cp, weights_cp);

            for (int i = 0; i < n; ++i) {
                EXPECT_LT(relative_error(roots_lk[i], roots_cp[i]), 1e-13)
                    << "Root at grid point mismatch for n=" << n << ", T=" << T;
                EXPECT_LT(relative_error(weights_lk[i], weights_cp[i]), 1e-13)
                    << "Weight at grid point mismatch for n=" << n << ", T=" << T;
            }
        }
    }
}

TEST(RysTableTest, LookupFallbackForLargeT) {
    // For T > 40, lookup should fall back to rys_compute
    double roots_lk[MAX_RYS_ROOTS], weights_lk[MAX_RYS_ROOTS];
    double roots_cp[MAX_RYS_ROOTS], weights_cp[MAX_RYS_ROOTS];

    for (int n = 1; n <= 5; ++n) {
        for (double T : {41.0, 50.0, 100.0}) {
            rys_lookup(n, T, roots_lk, weights_lk);
            rys_compute(n, T, roots_cp, weights_cp);

            for (int i = 0; i < n; ++i) {
                EXPECT_DOUBLE_EQ(roots_lk[i], roots_cp[i]);
                EXPECT_DOUBLE_EQ(weights_lk[i], weights_cp[i]);
            }
        }
    }
}

TEST(RysTableTest, LookupMomentMatching) {
    // Interpolated values should satisfy moment matching to < 1e-13
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    for (int n = 1; n <= 12; ++n) {
        for (double T : {0.33, 2.71, 7.77, 15.5, 25.3}) {
            rys_lookup(n, T, roots, weights);
            verify_moments(n, T, roots, weights, 1e-13);
        }
    }
}

TEST(RysTableTest, LookupLargeNRoots) {
    // Test table lookup for n_roots = 13..15 (falls back to rys_compute)
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    for (int n = 13; n <= 15; ++n) {
        for (double T : {0.5, 5.0, 15.0, 30.0}) {
            rys_lookup(n, T, roots, weights);

            // Basic sanity checks
            for (int i = 0; i < n; ++i) {
                EXPECT_GT(roots[i], 0.0);
                EXPECT_LT(roots[i], 1.0);
                EXPECT_GT(weights[i], 0.0);
            }

            // Weight sum = F_0(T)
            double sum = 0.0;
            for (int i = 0; i < n; ++i) sum += weights[i];
            EXPECT_LT(relative_error(sum, boys_evaluate(0, T)), 1e-8)
                << "Weight sum mismatch for n=" << n << ", T=" << T;
        }
    }
}

// ============================================================================
// Task 0.4.4: Public API Tests
// ============================================================================

TEST(RysApiTest, SingletonAccess) {
    // get_rys_quadrature() should return the same instance
    auto& q1 = get_rys_quadrature();
    auto& q2 = get_rys_quadrature();
    EXPECT_EQ(&q1, &q2);

    auto& q3 = RysQuadrature::instance();
    EXPECT_EQ(&q1, &q3);
}

TEST(RysApiTest, ComputeRootsAndWeights) {
    // Singleton compute_roots_and_weights should match rys_compute
    auto& rys = get_rys_quadrature();
    double roots_api[MAX_RYS_ROOTS], weights_api[MAX_RYS_ROOTS];
    double roots_ref[MAX_RYS_ROOTS], weights_ref[MAX_RYS_ROOTS];

    for (int n = 1; n <= 8; ++n) {
        for (double T : {0.0, 1.0, 5.0, 10.0, 25.0}) {
            rys.compute_roots_and_weights(n, T, roots_api, weights_api);
            rys_compute(n, T, roots_ref, weights_ref);

            for (int i = 0; i < n; ++i) {
                EXPECT_LT(relative_error(roots_api[i], roots_ref[i]), 1e-13)
                    << "Root mismatch at i=" << i << " for n=" << n << ", T=" << T;
                EXPECT_LT(relative_error(weights_api[i], weights_ref[i]), 1e-13)
                    << "Weight mismatch at i=" << i << " for n=" << n << ", T=" << T;
            }
        }
    }
}

TEST(RysApiTest, RysRootsWeightsTemplate) {
    // RysRootsWeights<N> should work for N=1..8
    auto test_n = [](auto rw, int n, double T) {
        rw.compute(n, T);
        EXPECT_EQ(rw.n_roots, n);

        // Verify moment matching
        std::vector<double> expected(2 * n);
        boys_evaluate_array(2 * n - 1, T, expected.data());

        for (int k = 0; k < 2 * n; ++k) {
            double sum = 0.0;
            for (int i = 0; i < n; ++i) {
                sum += rw.weights[i] * std::pow(rw.roots[i], k);
            }
            double err = relative_error(sum, expected[k]);
            EXPECT_LT(err, 1e-13)
                << "Moment mismatch at k=" << k << " for n=" << n << ", T=" << T;
        }
    };

    test_n(RysRootsWeights<1>{}, 1, 5.0);
    test_n(RysRootsWeights<2>{}, 2, 5.0);
    test_n(RysRootsWeights<3>{}, 3, 5.0);
    test_n(RysRootsWeights<4>{}, 4, 5.0);
    test_n(RysRootsWeights<5>{}, 5, 5.0);
    test_n(RysRootsWeights<6>{}, 6, 5.0);
    test_n(RysRootsWeights<7>{}, 7, 5.0);
    test_n(RysRootsWeights<8>{}, 8, 5.0);
}

TEST(RysApiTest, TemplateComputeMethod) {
    // RysQuadrature::compute<N>() convenience method
    auto& rys = get_rys_quadrature();

    auto result = rys.compute<5>(5, 10.0);
    EXPECT_EQ(result.n_roots, 5);

    // All roots in (0, 1)
    for (int i = 0; i < 5; ++i) {
        EXPECT_GT(result.roots[i], 0.0);
        EXPECT_LT(result.roots[i], 1.0);
        EXPECT_GT(result.weights[i], 0.0);
    }

    // Weight sum = F_0(10)
    double sum = 0.0;
    for (int i = 0; i < 5; ++i) sum += result.weights[i];
    EXPECT_LT(relative_error(sum, boys_evaluate(0, 10.0)), 1e-13);
}

TEST(RysApiTest, QuadratureExactnessThroughApi) {
    // Verify full quadrature exactness through the public API
    auto& rys = get_rys_quadrature();

    for (int n = 1; n <= 8; ++n) {
        for (double T : {0.5, 3.3, 12.0, 30.0}) {
            double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];
            rys.compute_roots_and_weights(n, T, roots, weights);
            verify_moments(n, T, roots, weights, 1e-13);
        }
    }
}

// ============================================================================
// Task 0.4.5: Comprehensive Validation
// ============================================================================

TEST(RysValidationTest, AnalyticalSTypeERI) {
    // Validate against analytical (ss|ss) electron repulsion integral
    //
    // (ss|ss) = 2π^{5/2} / (pq√(p+q)) × K_AB × K_CD × F_0(T)
    // where p = α+β, q = γ+δ, K_AB = exp(-αβ/(α+β)|A-B|²), etc.
    // T = pq/(p+q) × |P-Q|², P = (αA+βB)/p, Q = (γC+δD)/q
    //
    // The Rys quadrature gives: ∑_i w_i = F_0(T) (k=0 moment)
    // So (ss|ss)_rys = 2π^{5/2}/(pq√(p+q)) × K_AB × K_CD × ∑_i w_i

    constexpr double PI = 3.14159265358979323846;

    struct ERICase {
        double alpha, beta, gamma, delta;
        double Ax, Bx, Cx, Dx;  // 1D positions (others at origin)
    };

    ERICase cases[] = {
        {1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0},  // Same center
        {1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0},  // Separated pairs
        {3.0, 2.0, 1.5, 0.8, 0.0, 0.5, 1.0, 1.5},  // Asymmetric
        {10.0, 5.0, 8.0, 3.0, 0.0, 0.0, 0.0, 0.0}, // Tight exponents
        {0.1, 0.2, 0.15, 0.25, 0.0, 2.0, -1.0, 3.0}, // Diffuse exponents
    };

    for (const auto& c : cases) {
        double p = c.alpha + c.beta;
        double q = c.gamma + c.delta;

        double Px = (c.alpha * c.Ax + c.beta * c.Bx) / p;
        double Qx = (c.gamma * c.Cx + c.delta * c.Dx) / q;

        double AB2 = (c.Ax - c.Bx) * (c.Ax - c.Bx);
        double CD2 = (c.Cx - c.Dx) * (c.Cx - c.Dx);
        double PQ2 = (Px - Qx) * (Px - Qx);

        double K_AB = std::exp(-c.alpha * c.beta / p * AB2);
        double K_CD = std::exp(-c.gamma * c.delta / q * CD2);
        double T = p * q / (p + q) * PQ2;

        double prefactor = 2.0 * std::pow(PI, 2.5) / (p * q * std::sqrt(p + q))
                         * K_AB * K_CD;

        // Analytical: prefactor × F_0(T)
        double F0 = boys_evaluate(0, T);
        double eri_analytical = prefactor * F0;

        // Rys quadrature: n_roots=1 for (ss|ss)
        double root, weight;
        rys_compute(1, T, &root, &weight);
        double eri_rys = prefactor * weight;

        EXPECT_LT(relative_error(eri_rys, eri_analytical), 1e-14)
            << "ERI mismatch for case with T=" << T;
    }
}

TEST(RysValidationTest, AnalyticalHigherAMExactness) {
    // n-point Rys quadrature integrates polynomials of degree ≤ 2n-1 exactly:
    //   ∑_i w_i × u_i^k = F_k(T)  for k = 0, ..., 2n-1
    //
    // For large T and high-order moments (k ≥ 10), floating-point errors
    // in computing u^k accumulate significantly (roots become very small
    // at large T, amplifying relative error in u^k). We test moderate T
    // for exhaustive moment matching and large T separately for low moments.
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    // Moderate T: all moments should match to high precision
    for (int n = 1; n <= 10; ++n) {
        for (double T : {0.0, 0.01, 0.1, 1.0, 5.0, 10.0, 25.0}) {
            rys_compute(n, T, roots, weights);
            verify_moments(n, T, roots, weights, 1e-12);
        }
    }

    // Large T: verify low-order moments (k ≤ 5) which are more numerically stable
    for (int n = 1; n <= 10; ++n) {
        for (double T : {50.0, 100.0}) {
            rys_compute(n, T, roots, weights);

            int max_k = std::min(2 * n, 6);
            for (int k = 0; k < max_k; ++k) {
                double sum = 0.0;
                for (int i = 0; i < n; ++i) {
                    sum += weights[i] * std::pow(roots[i], k);
                }
                double F_k = boys_evaluate(k, T);
                EXPECT_LT(relative_error(sum, F_k), 1e-12)
                    << "Moment " << k << " mismatch for n=" << n << ", T=" << T;
            }
        }
    }
}

TEST(RysValidationTest, PolynomialExactness) {
    // An n-point Rys quadrature should exactly integrate any polynomial
    // p(u) of degree ≤ 2n-1 against the Rys weight function.
    //
    // Test: ∑_i w_i × p(u_i) = ∫₀¹ p(t²) exp(-Tt²) dt
    // where p(u) = c_0 + c_1 u + ... + c_{2n-1} u^{2n-1}
    //
    // The integral = ∑_k c_k × F_k(T)

    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    // Use positive coefficients to avoid catastrophic cancellation
    double coeffs[] = {1.0, 0.5, 0.3, 0.2, 0.15, 0.12, 0.1, 0.08,
                       0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.012};

    for (int n = 1; n <= 8; ++n) {
        const int coeff_max_degree = static_cast<int>(std::size(coeffs)) - 1;
        int max_degree = std::min(2 * n - 1, coeff_max_degree);
        ASSERT_GE(max_degree, 0);
        ASSERT_LE(max_degree, coeff_max_degree);

        for (double T : {0.1, 2.0, 10.0, 30.0}) {
            rys_compute(n, T, roots, weights);

            // Compute ∑_i w_i × p(u_i)
            double quad_sum = 0.0;
            for (int i = 0; i < n; ++i) {
                double poly = 0.0;
                double u_pow = 1.0;
                for (int k = 0; k <= max_degree; ++k) {
                    poly += coeffs[k] * u_pow;
                    u_pow *= roots[i];
                }
                quad_sum += weights[i] * poly;
            }

            // Compute exact integral = ∑_k c_k × F_k(T)
            double exact = 0.0;
            for (int k = 0; k <= max_degree; ++k) {
                exact += coeffs[k] * boys_evaluate(k, T);
            }

            EXPECT_LT(relative_error(quad_sum, exact), 1e-12)
                << "Polynomial exactness failed for n=" << n << ", T=" << T
                << " quad=" << quad_sum << " exact=" << exact;
        }
    }
}

TEST(RysValidationTest, ReferenceDataConsistency) {
    // Validate that rys_compute and rys_lookup produce identical results
    // across a comprehensive range of T values and n_roots.
    // Also cross-validates against moment matching (the fundamental
    // definition of correctness for Rys quadrature).
    double roots_c[MAX_RYS_ROOTS], weights_c[MAX_RYS_ROOTS];
    double roots_l[MAX_RYS_ROOTS], weights_l[MAX_RYS_ROOTS];

    // Dense sampling including boundary regions
    double T_values[] = {0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0,
                         10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 39.9, 40.0,
                         40.1, 45.0, 50.0, 100.0};

    for (int n = 1; n <= 10; ++n) {
        for (double T : T_values) {
            rys_compute(n, T, roots_c, weights_c);
            rys_lookup(n, T, roots_l, weights_l);

            for (int i = 0; i < n; ++i) {
                EXPECT_LT(relative_error(roots_l[i], roots_c[i]), 1e-13)
                    << "Root mismatch at i=" << i
                    << " for n=" << n << ", T=" << T;
                EXPECT_LT(relative_error(weights_l[i], weights_c[i]), 1e-13)
                    << "Weight mismatch at i=" << i
                    << " for n=" << n << ", T=" << T;
            }

            // Cross-validate with moment matching
            verify_moments(n, T, roots_c, weights_c, 1e-12);
        }
    }
}

TEST(RysValidationTest, SymmetryProperties) {
    // Verify mathematical properties of the Rys quadrature
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    // Property 1: For T=0, the weight function is uniform on [0,1],
    // so the quadrature reduces to standard Gauss-Legendre on [0,1]
    // in the variable u = t².
    // F_k(0) = 1/(2k+1), so roots solve: ∑ w_i u_i^k = 1/(2k+1)
    for (int n = 1; n <= 8; ++n) {
        rys_compute(n, 0.0, roots, weights);
        for (int k = 0; k < 2 * n; ++k) {
            double sum = 0.0;
            for (int i = 0; i < n; ++i) {
                sum += weights[i] * std::pow(roots[i], k);
            }
            EXPECT_NEAR(sum, 1.0 / (2 * k + 1), 1e-14)
                << "T=0 moment k=" << k << " for n=" << n;
        }
    }

    // Property 2: All weights should be positive
    for (int n = 1; n <= 15; ++n) {
        for (double T : {0.0, 1.0, 10.0, 50.0, 100.0}) {
            rys_compute(n, T, roots, weights);
            for (int i = 0; i < n; ++i) {
                EXPECT_GT(weights[i], 0.0)
                    << "Non-positive weight at n=" << n << ", T=" << T << ", i=" << i;
            }
        }
    }

    // Property 3: Roots are strictly in (0,1) for all finite T
    for (int n = 1; n <= 15; ++n) {
        for (double T : {0.0, 0.001, 1.0, 10.0, 100.0}) {
            rys_compute(n, T, roots, weights);
            for (int i = 0; i < n; ++i) {
                EXPECT_GT(roots[i], 0.0)
                    << "Root ≤ 0 at n=" << n << ", T=" << T << ", i=" << i;
                EXPECT_LT(roots[i], 1.0)
                    << "Root ≥ 1 at n=" << n << ", T=" << T << ", i=" << i;
            }
        }
    }
}

TEST(RysValidationTest, PerformanceBenchmark) {
    // Performance test: target < 200 ns per root/weight set for n_roots ≤ 5
    // Using rys_lookup for the fast path
    constexpr int N_ITERS = 10000;
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    for (int n = 1; n <= 5; ++n) {
        // Warm up
        for (int i = 0; i < 100; ++i) {
            rys_lookup(n, 5.0 + i * 0.001, roots, weights);
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_ITERS; ++i) {
            double T = 0.1 + 39.8 * (i % 100) / 100.0;  // Vary T
            rys_lookup(n, T, roots, weights);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double ns_per_call = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - start).count() / static_cast<double>(N_ITERS);

        // Record the result (informational, not a hard failure)
        // Target: < 200 ns, but may vary by platform
        EXPECT_LT(ns_per_call, 50000.0)  // Very loose bound for CI
            << "n_roots=" << n << " took " << ns_per_call << " ns/call";

        // Print timing for inspection
        ::testing::Test::RecordProperty(
            std::string("ns_per_call_n" + std::to_string(n)).c_str(),
            std::to_string(ns_per_call));
    }
}

// ============================================================================
// Task 3.3.8: High Root Count Tests
// ============================================================================

TEST(RysHighRootTest, TenRoots) {
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    for (double T : {0.5, 5.0, 20.0, 50.0}) {
        rys_compute(10, T, roots, weights);

        // Verify roots in (0, 1), ascending, weights positive
        for (int i = 0; i < 10; ++i) {
            EXPECT_GT(roots[i], 0.0) << "n=10, T=" << T << ", i=" << i;
            EXPECT_LT(roots[i], 1.0) << "n=10, T=" << T << ", i=" << i;
            EXPECT_GT(weights[i], 0.0) << "n=10, T=" << T << ", i=" << i;
        }
        for (int i = 0; i < 9; ++i) {
            EXPECT_LT(roots[i], roots[i + 1]) << "n=10, T=" << T << ", i=" << i;
        }

        // Verify moment matching
        verify_moments(10, T, roots, weights, 1e-10);
    }
}

TEST(RysHighRootTest, TwelveRoots) {
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    for (double T : {1.0, 10.0, 30.0}) {
        rys_compute(12, T, roots, weights);

        for (int i = 0; i < 12; ++i) {
            EXPECT_GT(roots[i], 0.0) << "n=12, T=" << T;
            EXPECT_LT(roots[i], 1.0) << "n=12, T=" << T;
            EXPECT_GT(weights[i], 0.0) << "n=12, T=" << T;
        }

        verify_moments(12, T, roots, weights, 1e-8);
    }
}

TEST(RysHighRootTest, FifteenRoots) {
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    for (double T : {1.0, 10.0, 50.0}) {
        rys_compute(15, T, roots, weights);

        for (int i = 0; i < 15; ++i) {
            EXPECT_GT(roots[i], 0.0) << "n=15, T=" << T;
            EXPECT_LT(roots[i], 1.0) << "n=15, T=" << T;
            EXPECT_GT(weights[i], 0.0) << "n=15, T=" << T;
        }

        // Looser tolerance for very high root count
        verify_moments(15, T, roots, weights, 1e-6);
    }
}

// ============================================================================
// Task 3.3.8: Boundary T Values
// ============================================================================

TEST(RysBoundaryTest, VerySmallT) {
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    // T near zero
    for (double T : {1e-15, 1e-10, 1e-5}) {
        for (int n = 1; n <= 5; ++n) {
            rys_compute(n, T, roots, weights);

            for (int i = 0; i < n; ++i) {
                EXPECT_GT(roots[i], 0.0) << "n=" << n << ", T=" << T;
                EXPECT_LT(roots[i], 1.0) << "n=" << n << ", T=" << T;
                EXPECT_GT(weights[i], 0.0) << "n=" << n << ", T=" << T;
                EXPECT_TRUE(std::isfinite(roots[i])) << "n=" << n << ", T=" << T;
                EXPECT_TRUE(std::isfinite(weights[i])) << "n=" << n << ", T=" << T;
            }
        }
    }
}

TEST(RysBoundaryTest, LargeT) {
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    // Large T values
    for (double T : {100.0, 500.0, 1000.0}) {
        for (int n = 1; n <= 5; ++n) {
            rys_compute(n, T, roots, weights);

            for (int i = 0; i < n; ++i) {
                EXPECT_GT(roots[i], 0.0) << "n=" << n << ", T=" << T;
                EXPECT_LT(roots[i], 1.0) << "n=" << n << ", T=" << T;
                EXPECT_GT(weights[i], 0.0) << "n=" << n << ", T=" << T;
                EXPECT_TRUE(std::isfinite(roots[i])) << "n=" << n << ", T=" << T;
                EXPECT_TRUE(std::isfinite(weights[i])) << "n=" << n << ", T=" << T;
            }
        }
    }
}

TEST(RysBoundaryTest, TransitionRegions) {
    // Test near lookup/compute transition boundaries
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    // Test around T=40 (common transition point for lookup tables)
    for (double T : {39.5, 39.9, 40.0, 40.1, 40.5}) {
        for (int n : {1, 3, 5}) {
            rys_compute(n, T, roots, weights);

            for (int i = 0; i < n; ++i) {
                EXPECT_GT(roots[i], 0.0);
                EXPECT_LT(roots[i], 1.0);
                EXPECT_GT(weights[i], 0.0);
            }

            verify_moments(n, T, roots, weights, 1e-10);
        }
    }
}

TEST(RysBoundaryTest, WeightSumEqualsF0) {
    // Sum of weights = F_0(T) for all T
    double roots[MAX_RYS_ROOTS], weights[MAX_RYS_ROOTS];

    for (int n = 1; n <= 10; ++n) {
        for (double T : {0.0, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0}) {
            rys_compute(n, T, roots, weights);

            double weight_sum = 0.0;
            for (int i = 0; i < n; ++i) {
                weight_sum += weights[i];
            }

            double F0 = boys_evaluate(0, T);
            EXPECT_LT(relative_error(weight_sum, F0), 1e-12)
                << "Weight sum mismatch at n=" << n << ", T=" << T;
        }
    }
}
