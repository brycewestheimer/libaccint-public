// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_screened_iterator.cpp
/// @brief Unit tests for ScreenedQuartetIterator

#include <libaccint/screening/screened_quartet_iterator.hpp>
#include <libaccint/screening/schwarz_bounds.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/utils/error_handling.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace libaccint;
using namespace libaccint::screening;

namespace {

/// Helper: create an S-shell (L=0) with given center
Shell make_s_shell(Point3D center) {
    std::vector<Real> exponents = {3.0, 1.0, 0.3};
    std::vector<Real> coefficients = {0.3, 0.5, 0.2};
    return Shell(AngularMomentum::S, center, exponents, coefficients);
}

/// Helper: create a P-shell (L=1) with given center
Shell make_p_shell(Point3D center) {
    std::vector<Real> exponents = {2.0, 0.5};
    std::vector<Real> coefficients = {0.6, 0.4};
    return Shell(AngularMomentum::P, center, exponents, coefficients);
}

/// Helper: create a simple H2O-like basis set
BasisSet make_h2o_basis() {
    std::vector<Shell> shells;

    // Oxygen (at origin)
    shells.push_back(make_s_shell(Point3D(0.0, 0.0, 0.0)));
    shells.push_back(make_p_shell(Point3D(0.0, 0.0, 0.0)));

    // Hydrogen 1
    shells.push_back(make_s_shell(Point3D(0.0, 1.43, -1.11)));

    // Hydrogen 2
    shells.push_back(make_s_shell(Point3D(0.0, -1.43, -1.11)));

    return BasisSet(std::move(shells));
}

}  // anonymous namespace

// =============================================================================
// ScreenedQuartetIterator Basic Tests
// =============================================================================

TEST(ScreenedQuartetIteratorTest, ConstructWithThreshold) {
    BasisSet basis = make_h2o_basis();

    ScreenedQuartetIterator iter(basis, 1e-12);

    EXPECT_DOUBLE_EQ(iter.threshold(), 1e-12);
    EXPECT_TRUE(iter.has_more());
}

TEST(ScreenedQuartetIteratorTest, ConstructWithOptions) {
    BasisSet basis = make_h2o_basis();
    ScreeningOptions opts = ScreeningOptions::tight();

    ScreenedQuartetIterator iter(basis, opts);

    EXPECT_DOUBLE_EQ(iter.threshold(), 1e-14);
}

TEST(ScreenedQuartetIteratorTest, NegativeThresholdThrows) {
    BasisSet basis = make_h2o_basis();

    EXPECT_THROW(ScreenedQuartetIterator(basis, -1e-12), InvalidArgumentException);
}

TEST(ScreenedQuartetIteratorTest, IterateYieldsQuartets) {
    BasisSet basis = make_h2o_basis();
    ScreenedQuartetIterator iter(basis, 0.0);  // No screening

    Size count = 0;
    while (auto batch = iter.next_batch(100)) {
        count += batch->size();
    }

    // With 4 shells and 8-fold symmetry:
    // n_pairs = 4 * 5 / 2 = 10
    // n_unique_quartets = 10 * 11 / 2 = 55
    EXPECT_EQ(count, 55u);
}

TEST(ScreenedQuartetIteratorTest, TotalUniqueQuartets) {
    BasisSet basis = make_h2o_basis();
    ScreenedQuartetIterator iter(basis, 0.0);

    // 4 shells: 10 pairs, 55 unique quartets
    EXPECT_EQ(iter.total_unique_quartets(), 55u);
}

TEST(ScreenedQuartetIteratorTest, ScreeningReducesQuartets) {
    BasisSet basis = make_h2o_basis();

    // Count without screening
    ScreenedQuartetIterator iter_none(basis, 0.0);
    Size count_none = 0;
    while (auto batch = iter_none.next_batch(100)) {
        count_none += batch->size();
    }

    // Count with screening
    ScreenedQuartetIterator iter_screened(basis, 1e-10);
    Size count_screened = 0;
    while (auto batch = iter_screened.next_batch(100)) {
        count_screened += batch->size();
    }

    // With screening, may get fewer quartets (or same if all significant)
    EXPECT_LE(count_screened, count_none);
}

TEST(ScreenedQuartetIteratorTest, QuartetBoundsAboveThreshold) {
    BasisSet basis = make_h2o_basis();
    Real threshold = 1e-10;

    ScreenedQuartetIterator iter(basis, threshold);

    while (auto batch = iter.next_batch(100)) {
        for (const auto& q : *batch) {
            // All yielded quartets should have bound >= threshold
            EXPECT_GE(q.schwarz_bound, threshold);
        }
    }
}

TEST(ScreenedQuartetIteratorTest, StatisticsTracked) {
    BasisSet basis = make_h2o_basis();
    ScreenedQuartetIterator iter(basis, 1e-10);

    // Consume all quartets
    while (auto batch = iter.next_batch(100)) {
        // Just iterate
    }

    const auto& stats = iter.statistics();
    EXPECT_EQ(stats.total_quartets, 55u);  // Total unique quartets
    EXPECT_EQ(stats.computed_quartets + stats.skipped_quartets, stats.total_quartets);
}

TEST(ScreenedQuartetIteratorTest, Reset) {
    BasisSet basis = make_h2o_basis();
    ScreenedQuartetIterator iter(basis, 1e-10);

    // Consume all
    while (auto batch = iter.next_batch(100)) {}
    EXPECT_FALSE(iter.has_more());

    // Reset
    iter.reset();
    EXPECT_TRUE(iter.has_more());

    // Can iterate again
    Size count = 0;
    while (auto batch = iter.next_batch(100)) {
        count += batch->size();
    }
    EXPECT_GT(count, 0u);
}

TEST(ScreenedQuartetIteratorTest, BatchSizeRespected) {
    BasisSet basis = make_h2o_basis();
    ScreenedQuartetIterator iter(basis, 0.0);  // No screening

    Size batch_size = 10;
    bool checked_middle_batch = false;

    while (auto batch = iter.next_batch(batch_size)) {
        // All batches except possibly the last should have batch_size elements
        if (iter.has_more()) {
            EXPECT_EQ(batch->size(), batch_size);
            checked_middle_batch = true;
        } else {
            // Last batch can be smaller
            EXPECT_LE(batch->size(), batch_size);
        }
    }

    EXPECT_TRUE(checked_middle_batch);  // We had at least one full batch
}

// =============================================================================
// SchwarzBounds Storage Tests
// =============================================================================

TEST(SchwarzBoundsTest, ConstructFromBasis) {
    BasisSet basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    EXPECT_EQ(bounds.n_shells(), 4u);
    EXPECT_EQ(bounds.storage_size(), 10u);  // 4 * 5 / 2 = 10 pairs
    EXPECT_TRUE(bounds.is_initialized());
}

TEST(SchwarzBoundsTest, AllBoundsPositive) {
    BasisSet basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    for (Size i = 0; i < basis.n_shells(); ++i) {
        for (Size j = 0; j < basis.n_shells(); ++j) {
            EXPECT_GT(bounds(i, j), 0.0);
        }
    }
}

TEST(SchwarzBoundsTest, Symmetric) {
    BasisSet basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    for (Size i = 0; i < basis.n_shells(); ++i) {
        for (Size j = 0; j < basis.n_shells(); ++j) {
            EXPECT_DOUBLE_EQ(bounds(i, j), bounds(j, i));
        }
    }
}

TEST(SchwarzBoundsTest, QuartetBound) {
    BasisSet basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    Real Q_ij = bounds(0, 1);
    Real Q_kl = bounds(2, 3);

    EXPECT_DOUBLE_EQ(bounds.quartet_bound(0, 1, 2, 3), Q_ij * Q_kl);
}

TEST(SchwarzBoundsTest, PassesScreening) {
    BasisSet basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    Real max_bound = bounds.max_bound();
    Real threshold = max_bound * max_bound * 0.5;  // Below max possible

    // Find at least one quartet that passes
    bool found_passing = false;
    for (Size i = 0; i < basis.n_shells() && !found_passing; ++i) {
        for (Size j = i; j < basis.n_shells() && !found_passing; ++j) {
            for (Size k = i; k < basis.n_shells() && !found_passing; ++k) {
                Size l_start = (k == i) ? j : k;
                for (Size l = l_start; l < basis.n_shells(); ++l) {
                    if (bounds.passes_screening(i, j, k, l, threshold)) {
                        found_passing = true;
                        break;
                    }
                }
            }
        }
    }

    EXPECT_TRUE(found_passing);
}

TEST(SchwarzBoundsTest, CountPassingQuartets) {
    BasisSet basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    // All quartets pass with threshold = 0
    EXPECT_EQ(bounds.count_passing_quartets(0.0), 55u);

    // Some quartets pass with reasonable threshold
    Size passing = bounds.count_passing_quartets(1e-10);
    EXPECT_LE(passing, 55u);
}

TEST(SchwarzBoundsTest, EstimatePassFraction) {
    BasisSet basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    EXPECT_DOUBLE_EQ(bounds.estimate_pass_fraction(0.0), 1.0);
    EXPECT_LE(bounds.estimate_pass_fraction(1e-10), 1.0);
    EXPECT_GE(bounds.estimate_pass_fraction(1e-10), 0.0);
}

TEST(SchwarzBoundsTest, IteratorWithExternalBounds) {
    BasisSet basis = make_h2o_basis();
    SchwarzBounds bounds(basis);

    // Use iterator with external bounds
    ScreenedQuartetIterator iter(basis, bounds, 1e-10);

    Size count = 0;
    while (auto batch = iter.next_batch(100)) {
        count += batch->size();
    }

    EXPECT_GT(count, 0u);
}

