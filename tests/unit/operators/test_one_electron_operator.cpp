// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/utils/error_handling.hpp>
#include <gtest/gtest.h>

using namespace libaccint;

// ============================================================================
// Construction Tests
// ============================================================================

TEST(OneElectronOperatorTest, ImplicitConstructionFromOperator) {
    // Test that we can implicitly construct from a single Operator
    auto s = Operator::overlap();
    OneElectronOperator op = s;  // Implicit conversion

    EXPECT_EQ(op.n_contributions(), 1);
    auto contribs = op.contributions();
    EXPECT_EQ(contribs.size(), 1);
    EXPECT_EQ(contribs[0].op.kind(), OperatorKind::Overlap);
    EXPECT_DOUBLE_EQ(contribs[0].scale, 1.0);
}

TEST(OneElectronOperatorTest, ExplicitConstructionFromOperator) {
    auto t = Operator::kinetic();
    OneElectronOperator op(t);

    EXPECT_EQ(op.n_contributions(), 1);
    auto contribs = op.contributions();
    EXPECT_EQ(contribs.size(), 1);
    EXPECT_EQ(contribs[0].op.kind(), OperatorKind::Kinetic);
    EXPECT_DOUBLE_EQ(contribs[0].scale, 1.0);
}

TEST(OneElectronOperatorTest, RejectTwoElectronOperator) {
    auto coulomb = Operator::coulomb();
    EXPECT_THROW(
        OneElectronOperator op(coulomb),
        InvalidArgumentException
    );
}

// ============================================================================
// Add Method Tests
// ============================================================================

TEST(OneElectronOperatorTest, AddContribution) {
    OneElectronOperator op = Operator::overlap();
    op.add(Operator::kinetic());

    EXPECT_EQ(op.n_contributions(), 2);
    auto contribs = op.contributions();
    EXPECT_EQ(contribs[0].op.kind(), OperatorKind::Overlap);
    EXPECT_EQ(contribs[1].op.kind(), OperatorKind::Kinetic);
}

TEST(OneElectronOperatorTest, AddContributionWithScale) {
    OneElectronOperator op = Operator::overlap();
    op.add(Operator::kinetic(), 2.5);

    EXPECT_EQ(op.n_contributions(), 2);
    auto contribs = op.contributions();
    EXPECT_DOUBLE_EQ(contribs[0].scale, 1.0);
    EXPECT_DOUBLE_EQ(contribs[1].scale, 2.5);
}

TEST(OneElectronOperatorTest, AddRejectsTwoElectronOperator) {
    OneElectronOperator op = Operator::overlap();
    EXPECT_THROW(
        op.add(Operator::coulomb()),
        InvalidArgumentException
    );
}

// ============================================================================
// Operator+ Tests
// ============================================================================

TEST(OneElectronOperatorTest, AdditionOperator) {
    auto s = Operator::overlap();
    auto t = Operator::kinetic();

    OneElectronOperator op = OneElectronOperator(s) + t;

    EXPECT_EQ(op.n_contributions(), 2);
    auto contribs = op.contributions();
    EXPECT_EQ(contribs[0].op.kind(), OperatorKind::Overlap);
    EXPECT_EQ(contribs[1].op.kind(), OperatorKind::Kinetic);
    EXPECT_DOUBLE_EQ(contribs[0].scale, 1.0);
    EXPECT_DOUBLE_EQ(contribs[1].scale, 1.0);
}

TEST(OneElectronOperatorTest, ChainedAddition) {
    // Create a dummy nuclear operator (we need point charges)
    PointChargeParams charges;
    charges.x = {0.0};
    charges.y = {0.0};
    charges.z = {0.0};
    charges.charge = {1.0};

    auto s = Operator::overlap();
    auto t = Operator::kinetic();
    auto v = Operator::nuclear(charges);

    OneElectronOperator op = OneElectronOperator(s) + t + v;

    EXPECT_EQ(op.n_contributions(), 3);
    auto contribs = op.contributions();
    EXPECT_EQ(contribs[0].op.kind(), OperatorKind::Overlap);
    EXPECT_EQ(contribs[1].op.kind(), OperatorKind::Kinetic);
    EXPECT_EQ(contribs[2].op.kind(), OperatorKind::Nuclear);
    EXPECT_DOUBLE_EQ(contribs[0].scale, 1.0);
    EXPECT_DOUBLE_EQ(contribs[1].scale, 1.0);
    EXPECT_DOUBLE_EQ(contribs[2].scale, 1.0);
}

// ============================================================================
// Operator* Tests
// ============================================================================

TEST(OneElectronOperatorTest, RightScaling) {
    auto s = Operator::overlap();
    auto t = Operator::kinetic();

    OneElectronOperator op = (OneElectronOperator(s) + t) * 2.0;

    EXPECT_EQ(op.n_contributions(), 2);
    auto contribs = op.contributions();
    EXPECT_DOUBLE_EQ(contribs[0].scale, 2.0);
    EXPECT_DOUBLE_EQ(contribs[1].scale, 2.0);
}

TEST(OneElectronOperatorTest, LeftScaling) {
    PointChargeParams charges;
    charges.x = {0.0};
    charges.y = {0.0};
    charges.z = {0.0};
    charges.charge = {1.0};

    auto v = Operator::nuclear(charges);
    OneElectronOperator op = 0.5 * OneElectronOperator(v);

    EXPECT_EQ(op.n_contributions(), 1);
    auto contribs = op.contributions();
    EXPECT_DOUBLE_EQ(contribs[0].scale, 0.5);
}

TEST(OneElectronOperatorTest, CombinedScalingAndAddition) {
    auto s = Operator::overlap();
    auto t = Operator::kinetic();

    // Create (S + T) * 3.0
    OneElectronOperator op = (OneElectronOperator(s) + t) * 3.0;

    EXPECT_EQ(op.n_contributions(), 2);
    auto contribs = op.contributions();
    EXPECT_DOUBLE_EQ(contribs[0].scale, 3.0);
    EXPECT_DOUBLE_EQ(contribs[1].scale, 3.0);
}

TEST(OneElectronOperatorTest, ScalingPreservesOperatorKinds) {
    auto s = Operator::overlap();
    auto t = Operator::kinetic();

    OneElectronOperator op = 2.5 * (OneElectronOperator(s) + t);

    auto contribs = op.contributions();
    EXPECT_EQ(contribs[0].op.kind(), OperatorKind::Overlap);
    EXPECT_EQ(contribs[1].op.kind(), OperatorKind::Kinetic);
}

// ============================================================================
// Query Methods Tests
// ============================================================================

TEST(OneElectronOperatorTest, HasProjectionTermsFalseForCommonOperators) {
    auto s = Operator::overlap();
    auto t = Operator::kinetic();

    OneElectronOperator op = OneElectronOperator(s) + t;

    EXPECT_FALSE(op.has_projection_terms());
}

TEST(OneElectronOperatorTest, HasPotentialTermsTrueForNuclear) {
    PointChargeParams charges;
    charges.x = {0.0};
    charges.y = {0.0};
    charges.z = {0.0};
    charges.charge = {1.0};

    auto v = Operator::nuclear(charges);
    OneElectronOperator op = v;

    EXPECT_TRUE(op.has_potential_terms());
}

TEST(OneElectronOperatorTest, HasPotentialTermsTrueForPointCharge) {
    PointChargeParams charges;
    charges.x = {0.0, 1.0};
    charges.y = {0.0, 0.0};
    charges.z = {0.0, 0.0};
    charges.charge = {1.0, -1.0};

    auto pc = Operator::point_charges(charges);
    OneElectronOperator op = pc;

    EXPECT_TRUE(op.has_potential_terms());
}

TEST(OneElectronOperatorTest, HasPotentialTermsFalseForOverlapKinetic) {
    auto s = Operator::overlap();
    auto t = Operator::kinetic();

    OneElectronOperator op = OneElectronOperator(s) + t;

    EXPECT_FALSE(op.has_potential_terms());
}

TEST(OneElectronOperatorTest, HasPotentialTermsTrueWhenMixed) {
    PointChargeParams charges;
    charges.x = {0.0};
    charges.y = {0.0};
    charges.z = {0.0};
    charges.charge = {1.0};

    auto s = Operator::overlap();
    auto t = Operator::kinetic();
    auto v = Operator::nuclear(charges);

    OneElectronOperator op = OneElectronOperator(s) + t + v;

    EXPECT_TRUE(op.has_potential_terms());
}

// ============================================================================
// Contributions Access Tests
// ============================================================================

TEST(OneElectronOperatorTest, ContributionsSpanMatchesSize) {
    auto s = Operator::overlap();
    auto t = Operator::kinetic();

    OneElectronOperator op = OneElectronOperator(s) + t;

    auto contribs = op.contributions();
    EXPECT_EQ(contribs.size(), op.n_contributions());
    EXPECT_EQ(contribs.size(), 2);
}

TEST(OneElectronOperatorTest, ContributionsAreConst) {
    auto s = Operator::overlap();
    OneElectronOperator op = s;

    // This should compile (returning const span)
    std::span<const OneElectronOperator::Contribution> contribs = op.contributions();
    EXPECT_EQ(contribs.size(), 1);
}

// ============================================================================
// Complex Composition Tests
// ============================================================================

TEST(OneElectronOperatorTest, ComplexComposition) {
    PointChargeParams charges;
    charges.x = {0.0, 0.0};
    charges.y = {0.0, 0.0};
    charges.z = {0.0, 1.5};
    charges.charge = {1.0, 1.0};

    auto s = Operator::overlap();
    auto t = Operator::kinetic();
    auto v = Operator::nuclear(charges);

    // Build H = T + V - 0.1*S (for example, generalized eigenvalue problem)
    OneElectronOperator h = OneElectronOperator(t) + v + (-0.1 * OneElectronOperator(s));

    EXPECT_EQ(h.n_contributions(), 3);
    auto contribs = h.contributions();

    EXPECT_EQ(contribs[0].op.kind(), OperatorKind::Kinetic);
    EXPECT_DOUBLE_EQ(contribs[0].scale, 1.0);

    EXPECT_EQ(contribs[1].op.kind(), OperatorKind::Nuclear);
    EXPECT_DOUBLE_EQ(contribs[1].scale, 1.0);

    EXPECT_EQ(contribs[2].op.kind(), OperatorKind::Overlap);
    EXPECT_DOUBLE_EQ(contribs[2].scale, -0.1);

    EXPECT_TRUE(h.has_potential_terms());
}

TEST(OneElectronOperatorTest, MultipleScalingOperations) {
    auto s = Operator::overlap();

    OneElectronOperator op = s;
    op = op * 2.0;
    op = 3.0 * op;

    EXPECT_EQ(op.n_contributions(), 1);
    auto contribs = op.contributions();
    EXPECT_DOUBLE_EQ(contribs[0].scale, 6.0);  // 2.0 * 3.0
}

TEST(OneElectronOperatorTest, AddingPreScaledOperators) {
    auto s = Operator::overlap();
    auto t = Operator::kinetic();

    OneElectronOperator s2 = 2.0 * OneElectronOperator(s);
    OneElectronOperator t3 = 3.0 * OneElectronOperator(t);

    OneElectronOperator combined = s2 + t3;

    EXPECT_EQ(combined.n_contributions(), 2);
    auto contribs = combined.contributions();
    EXPECT_DOUBLE_EQ(contribs[0].scale, 2.0);
    EXPECT_DOUBLE_EQ(contribs[1].scale, 3.0);
}
