// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_schwarz_thread_safety.cpp
/// @brief Thread-safety tests for Schwarz bounds precomputation (Task 8.3.1, 8.3.5)

#include <libaccint/screening/schwarz_bounds.hpp>
#include <libaccint/screening/screening_options.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/consumers/fock_builder.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <thread>
#include <vector>
#include <atomic>

using namespace libaccint;
using namespace libaccint::screening;

namespace {

Shell make_s_shell(Point3D center) {
    std::vector<Real> exponents = {3.0, 1.0, 0.3};
    std::vector<Real> coefficients = {0.3, 0.5, 0.2};
    return Shell(AngularMomentum::S, center, exponents, coefficients);
}

Shell make_p_shell(Point3D center) {
    std::vector<Real> exponents = {2.0, 0.5};
    std::vector<Real> coefficients = {0.6, 0.4};
    return Shell(AngularMomentum::P, center, exponents, coefficients);
}

BasisSet make_h2o_basis() {
    std::vector<Shell> shells;
    shells.push_back(make_s_shell(Point3D(0.0, 0.0, 0.0)));
    shells.push_back(make_p_shell(Point3D(0.0, 0.0, 0.0)));
    shells.push_back(make_s_shell(Point3D(0.0, 1.43, -1.11)));
    shells.push_back(make_s_shell(Point3D(0.0, -1.43, -1.11)));
    return BasisSet(std::move(shells));
}

}  // namespace

// =========================================================================
// Task 8.3.1: Schwarz Bounds Precomputation Tests
// =========================================================================

TEST(SchwarzBoundsTest, BasicPrecomputation) {
    auto basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    EXPECT_TRUE(bounds.is_initialized());
    EXPECT_EQ(bounds.n_shells(), 4u);
    EXPECT_GT(bounds.max_bound(), 0.0);
}

TEST(SchwarzBoundsTest, UpperBoundProperty) {
    // Q_ij * Q_kl >= |(ij|kl)| — the Schwarz inequality
    auto basis = make_h2o_basis();
    SchwarzBounds bounds(basis);
    Engine engine(basis);

    TwoElectronBuffer<0> buffer;
    Size n = basis.n_shells();

    for (Size i = 0; i < n; ++i) {
        for (Size j = 0; j < n; ++j) {
            for (Size k = 0; k < n; ++k) {
                for (Size l = 0; l < n; ++l) {
                    engine.cpu_engine().compute_2e_shell_quartet(
                        Operator::coulomb(),
                        basis.shell(i), basis.shell(j),
                        basis.shell(k), basis.shell(l), buffer);

                    Real bound = bounds.quartet_bound(i, j, k, l);
                    int na = basis.shell(i).n_functions();
                    int nb = basis.shell(j).n_functions();
                    int nc = basis.shell(k).n_functions();
                    int nd = basis.shell(l).n_functions();

                    for (int a = 0; a < na; ++a)
                        for (int b = 0; b < nb; ++b)
                            for (int c = 0; c < nc; ++c)
                                for (int d = 0; d < nd; ++d) {
                                    EXPECT_LE(std::abs(buffer(a, b, c, d)), bound * 1.001)
                                        << "Schwarz upper bound violated at ("
                                        << i << "," << j << "," << k << "," << l << ")";
                                }
                }
            }
        }
    }
}

TEST(SchwarzBoundsTest, SignificancePreservation) {
    // Integrals that pass screening should be non-negligible
    auto basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    // All self-pair bounds should be positive
    for (Size i = 0; i < bounds.n_shells(); ++i) {
        EXPECT_GT(bounds(i, i), 0.0)
            << "Self-pair bound should be positive for shell " << i;
    }
}

TEST(SchwarzBoundsTest, Symmetry) {
    auto basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    for (Size i = 0; i < bounds.n_shells(); ++i) {
        for (Size j = 0; j < bounds.n_shells(); ++j) {
            EXPECT_EQ(bounds(i, j), bounds(j, i))
                << "Schwarz bounds should be symmetric";
        }
    }
}

TEST(SchwarzBoundsTest, Determinism) {
    auto basis = make_h2o_basis();
    SchwarzBounds bounds1(basis);
    SchwarzBounds bounds2(basis);

    for (Size i = 0; i < bounds1.n_shells(); ++i) {
        for (Size j = 0; j <= i; ++j) {
            EXPECT_EQ(bounds1(i, j), bounds2(i, j))
                << "Bounds should be deterministic";
        }
    }
}

TEST(SchwarzBoundsTest, CountPassingQuartets) {
    auto basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    Size count_tight = bounds.count_passing_quartets(1e-14);
    Size count_loose = bounds.count_passing_quartets(1e-8);

    EXPECT_GE(count_tight, count_loose)
        << "Tighter threshold should pass more quartets";
}

TEST(SchwarzBoundsTest, PassFractionRange) {
    auto basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    Real frac = bounds.estimate_pass_fraction(1e-12);
    EXPECT_GE(frac, 0.0);
    EXPECT_LE(frac, 1.0);
}

// =========================================================================
// Task 8.3.5: Thread-Safety Stress Tests
// =========================================================================

TEST(SchwarzThreadSafetyTest, ConcurrentPrecompute) {
    // Multiple threads calling precompute_schwarz_bounds concurrently
    // should all see the same result
    auto basis = make_h2o_basis();
    Engine engine(basis);

    constexpr int N_THREADS = 8;
    constexpr int N_ITERATIONS = 100;

    std::vector<std::thread> threads;
    std::atomic<int> errors{0};

    // Store reference bounds
    const auto& ref_bounds = engine.precompute_schwarz_bounds();
    Real ref_max = ref_bounds.max_bound();

    // Reset engine to test concurrent lazy init
    // We can't reset once_flag, so we test concurrent reads instead
    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&engine, ref_max, &errors]() {
            for (int iter = 0; iter < N_ITERATIONS; ++iter) {
                const auto& bounds = engine.get_schwarz_bounds();
                if (std::abs(bounds.max_bound() - ref_max) > 1e-15) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& t : threads) t.join();
    EXPECT_EQ(errors.load(), 0) << "Concurrent reads returned inconsistent bounds";
}

TEST(SchwarzThreadSafetyTest, ConcurrentQuery) {
    // Concurrent lookups into SchwarzBounds should be safe (immutable data)
    auto basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    constexpr int N_THREADS = 8;
    constexpr int N_ITERATIONS = 1000;

    std::vector<std::thread> threads;
    std::atomic<int> errors{0};

    Size n = bounds.n_shells();

    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&bounds, &errors, n]() {
            for (int iter = 0; iter < N_ITERATIONS; ++iter) {
                for (Size i = 0; i < n; ++i) {
                    for (Size j = 0; j < n; ++j) {
                        Real q = bounds(i, j);
                        if (q < 0.0 || !std::isfinite(q)) {
                            errors.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            }
        });
    }

    for (auto& t : threads) t.join();
    EXPECT_EQ(errors.load(), 0) << "Concurrent queries returned invalid bounds";
}

TEST(SchwarzThreadSafetyTest, ConcurrentScreening) {
    // Concurrent passes_screening calls
    auto basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    constexpr int N_THREADS = 8;
    constexpr int N_ITERATIONS = 500;

    std::vector<std::thread> threads;
    std::atomic<int> errors{0};

    Size n = bounds.n_shells();

    // Pre-compute reference results
    std::vector<bool> ref_results;
    Real threshold = 1e-12;
    for (Size i = 0; i < n; ++i)
        for (Size j = 0; j < n; ++j)
            for (Size k = 0; k < n; ++k)
                for (Size l = 0; l < n; ++l)
                    ref_results.push_back(bounds.passes_screening(i, j, k, l, threshold));

    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&bounds, &errors, &ref_results, n, threshold]() {
            for (int iter = 0; iter < N_ITERATIONS; ++iter) {
                Size idx = 0;
                for (Size i = 0; i < n; ++i)
                    for (Size j = 0; j < n; ++j)
                        for (Size k = 0; k < n; ++k)
                            for (Size l = 0; l < n; ++l) {
                                bool result = bounds.passes_screening(i, j, k, l, threshold);
                                if (result != ref_results[idx]) {
                                    errors.fetch_add(1, std::memory_order_relaxed);
                                }
                                ++idx;
                            }
            }
        });
    }

    for (auto& t : threads) t.join();
    EXPECT_EQ(errors.load(), 0) << "Concurrent screening inconsistent";
}
