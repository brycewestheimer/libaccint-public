// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/utils/error_handling.hpp>
#include <gtest/gtest.h>

#include <functional>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace libaccint;

namespace {

/// Helper: create an S-shell (L=0) with given center and K primitives
Shell make_s_shell(Point3D center, Size k = 3) {
    std::vector<Real> exponents;
    std::vector<Real> coefficients;
    exponents.reserve(k);
    coefficients.reserve(k);
    for (Size i = 0; i < k; ++i) {
        exponents.push_back(3.0 / static_cast<Real>(i + 1));
        coefficients.push_back(1.0 / static_cast<Real>(k));
    }
    return Shell(AngularMomentum::S, center, std::move(exponents), std::move(coefficients));
}

/// Helper: create a P-shell (L=1) with given center and K primitives
Shell make_p_shell(Point3D center, Size k = 3) {
    std::vector<Real> exponents;
    std::vector<Real> coefficients;
    exponents.reserve(k);
    coefficients.reserve(k);
    for (Size i = 0; i < k; ++i) {
        exponents.push_back(2.0 / static_cast<Real>(i + 1));
        coefficients.push_back(1.0 / static_cast<Real>(k));
    }
    return Shell(AngularMomentum::P, center, std::move(exponents), std::move(coefficients));
}

/// Helper: create a D-shell (L=2) with given center and K primitives
Shell make_d_shell(Point3D center, Size k = 1) {
    std::vector<Real> exponents;
    std::vector<Real> coefficients;
    exponents.reserve(k);
    coefficients.reserve(k);
    for (Size i = 0; i < k; ++i) {
        exponents.push_back(1.5 / static_cast<Real>(i + 1));
        coefficients.push_back(1.0 / static_cast<Real>(k));
    }
    return Shell(AngularMomentum::D, center, std::move(exponents), std::move(coefficients));
}

}  // anonymous namespace

// =============================================================================
// ShellSetKey Tests
// =============================================================================

TEST(ShellSetKeyTest, DefaultConstruction) {
    ShellSetKey key;
    EXPECT_EQ(key.angular_momentum, 0);
    EXPECT_EQ(key.n_primitives, 0);
}

TEST(ShellSetKeyTest, ValueConstruction) {
    ShellSetKey key(2, 5);
    EXPECT_EQ(key.angular_momentum, 2);
    EXPECT_EQ(key.n_primitives, 5);
}

TEST(ShellSetKeyTest, Equality) {
    ShellSetKey a(1, 3);
    ShellSetKey b(1, 3);
    ShellSetKey c(1, 4);
    ShellSetKey d(2, 3);

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    EXPECT_NE(a, d);
    EXPECT_NE(c, d);
}

TEST(ShellSetKeyTest, HashConsistency) {
    ShellSetKey a(1, 3);
    ShellSetKey b(1, 3);

    std::hash<ShellSetKey> hasher;
    EXPECT_EQ(hasher(a), hasher(b));
}

TEST(ShellSetKeyTest, HashDistribution) {
    // Different keys should (with high probability) produce different hashes
    std::hash<ShellSetKey> hasher;

    ShellSetKey k1(0, 1);
    ShellSetKey k2(0, 3);
    ShellSetKey k3(1, 1);
    ShellSetKey k4(2, 5);

    // We don't require all hashes to be different (hash collisions are valid),
    // but at least some should differ for a reasonable hash function
    std::size_t h1 = hasher(k1);
    std::size_t h2 = hasher(k2);
    std::size_t h3 = hasher(k3);
    std::size_t h4 = hasher(k4);

    int distinct_count = 1;
    if (h2 != h1) ++distinct_count;
    if (h3 != h1 && h3 != h2) ++distinct_count;
    if (h4 != h1 && h4 != h2 && h4 != h3) ++distinct_count;

    // At least 3 out of 4 hashes should be distinct
    EXPECT_GE(distinct_count, 3)
        << "Hash function produces too many collisions";
}

TEST(ShellSetKeyTest, UsableInUnorderedMap) {
    std::unordered_map<ShellSetKey, int> map;
    map[ShellSetKey(0, 3)] = 10;
    map[ShellSetKey(1, 3)] = 20;
    map[ShellSetKey(2, 1)] = 30;

    EXPECT_EQ(map.size(), 3u);
    EXPECT_EQ(map[ShellSetKey(0, 3)], 10);
    EXPECT_EQ(map[ShellSetKey(1, 3)], 20);
    EXPECT_EQ(map[ShellSetKey(2, 1)], 30);

    // Overwrite existing key
    map[ShellSetKey(0, 3)] = 42;
    EXPECT_EQ(map[ShellSetKey(0, 3)], 42);
    EXPECT_EQ(map.size(), 3u);
}

// =============================================================================
// Construction from Vector of Shell References
// =============================================================================

TEST(ShellSetTest, ConstructFromShellReferences) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));
    Shell s3 = make_s_shell(Point3D(0.0, 1.0, 0.0));

    std::vector<std::reference_wrapper<const Shell>> refs = {
        std::cref(s1), std::cref(s2), std::cref(s3)
    };

    ShellSet set(refs);

    EXPECT_EQ(set.angular_momentum(), 0);
    EXPECT_EQ(set.n_primitives_per_shell(), 3);
    EXPECT_EQ(set.n_shells(), 3u);
    EXPECT_FALSE(set.empty());
}

TEST(ShellSetTest, ConstructFromSingleShell) {
    Shell s = make_p_shell(Point3D(1.0, 2.0, 3.0));

    std::vector<std::reference_wrapper<const Shell>> refs = {std::cref(s)};
    ShellSet set(refs);

    EXPECT_EQ(set.angular_momentum(), 1);
    EXPECT_EQ(set.n_primitives_per_shell(), 3);
    EXPECT_EQ(set.n_shells(), 1u);
    EXPECT_EQ(set.n_functions_per_shell(), 3);  // P-shell: 3 Cartesian functions
    EXPECT_EQ(set.n_total_functions(), 3u);
}

TEST(ShellSetTest, ConstructFromEmptyThrows) {
    std::vector<std::reference_wrapper<const Shell>> empty;

    EXPECT_THROW({
        ShellSet set(empty);
    }, InvalidArgumentException);
}

TEST(ShellSetTest, ConstructMismatchedAMThrows) {
    Shell s_shell = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell p_shell = make_p_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<std::reference_wrapper<const Shell>> refs = {
        std::cref(s_shell), std::cref(p_shell)
    };

    EXPECT_THROW({
        ShellSet set(refs);
    }, InvalidArgumentException);
}

TEST(ShellSetTest, ConstructMismatchedPrimitivesThrows) {
    Shell s3 = make_s_shell(Point3D(0.0, 0.0, 0.0), 3);  // 3 primitives
    Shell s1 = make_s_shell(Point3D(1.0, 0.0, 0.0), 1);  // 1 primitive

    std::vector<std::reference_wrapper<const Shell>> refs = {
        std::cref(s3), std::cref(s1)
    };

    EXPECT_THROW({
        ShellSet set(refs);
    }, InvalidArgumentException);
}

// =============================================================================
// Construction with AM and n_primitives
// =============================================================================

TEST(ShellSetTest, ConstructWithAMAndPrimitives) {
    ShellSet set(2, 4);

    EXPECT_EQ(set.angular_momentum(), 2);
    EXPECT_EQ(set.angular_momentum_enum(), AngularMomentum::D);
    EXPECT_EQ(set.n_primitives_per_shell(), 4);
    EXPECT_EQ(set.n_shells(), 0u);
    EXPECT_TRUE(set.empty());
    EXPECT_EQ(set.n_functions_per_shell(), 6);  // D-shell: 6 Cartesian
    EXPECT_EQ(set.n_total_functions(), 0u);
}

TEST(ShellSetTest, ConstructInvalidAMThrows) {
    EXPECT_THROW({
        ShellSet set(-1, 3);
    }, InvalidArgumentException);

    EXPECT_THROW({
        ShellSet set(MAX_ANGULAR_MOMENTUM + 1, 3);
    }, InvalidArgumentException);
}

TEST(ShellSetTest, ConstructInvalidPrimitivesThrows) {
    EXPECT_THROW({
        ShellSet set(0, 0);
    }, InvalidArgumentException);

    EXPECT_THROW({
        ShellSet set(0, -1);
    }, InvalidArgumentException);
}

// =============================================================================
// add_shell Tests
// =============================================================================

TEST(ShellSetTest, AddShell) {
    ShellSet set(0, 3);

    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    set.add_shell(s1);
    EXPECT_EQ(set.n_shells(), 1u);

    set.add_shell(s2);
    EXPECT_EQ(set.n_shells(), 2u);
}

TEST(ShellSetTest, AddShellMismatchedAMThrows) {
    ShellSet set(0, 3);  // S-shells with 3 primitives
    Shell p_shell = make_p_shell(Point3D(0.0, 0.0, 0.0));  // P-shell

    EXPECT_THROW({
        set.add_shell(p_shell);
    }, InvalidArgumentException);
}

TEST(ShellSetTest, AddShellMismatchedPrimitivesThrows) {
    ShellSet set(0, 3);  // expects 3 primitives
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0), 1);  // 1 primitive

    EXPECT_THROW({
        set.add_shell(s1);
    }, InvalidArgumentException);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST(ShellSetTest, ShellAccessor) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 2.0, 3.0));

    std::vector<std::reference_wrapper<const Shell>> refs = {
        std::cref(s1), std::cref(s2)
    };
    ShellSet set(refs);

    EXPECT_DOUBLE_EQ(set.shell(0).center().x, 0.0);
    EXPECT_DOUBLE_EQ(set.shell(1).center().x, 1.0);
    EXPECT_DOUBLE_EQ(set.shell(1).center().y, 2.0);
    EXPECT_DOUBLE_EQ(set.shell(1).center().z, 3.0);
}

TEST(ShellSetTest, ShellAccessorOutOfBoundsThrows) {
    Shell s = make_s_shell(Point3D(0.0, 0.0, 0.0));
    std::vector<std::reference_wrapper<const Shell>> refs = {std::cref(s)};
    ShellSet set(refs);

    EXPECT_THROW(set.shell(1), InvalidArgumentException);
}

TEST(ShellSetTest, ShellsSpan) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));

    std::vector<std::reference_wrapper<const Shell>> refs = {
        std::cref(s1), std::cref(s2)
    };
    ShellSet set(refs);

    auto span = set.shells();
    EXPECT_EQ(span.size(), 2u);
    EXPECT_DOUBLE_EQ(span[0].center().x, 0.0);
    EXPECT_DOUBLE_EQ(span[1].center().x, 1.0);
}

TEST(ShellSetTest, Key) {
    ShellSet set(1, 3);
    ShellSetKey key = set.key();
    EXPECT_EQ(key.angular_momentum, 1);
    EXPECT_EQ(key.n_primitives, 3);
}

TEST(ShellSetTest, DefaultConstructor) {
    ShellSet set;
    EXPECT_EQ(set.angular_momentum(), 0);
    EXPECT_EQ(set.n_primitives_per_shell(), 0);
    EXPECT_EQ(set.n_shells(), 0u);
    EXPECT_TRUE(set.empty());
    EXPECT_FALSE(set.soa_ready());
}

// =============================================================================
// SoA Data Layout Correctness
// =============================================================================

TEST(ShellSetTest, SoADataCenters) {
    Shell s1 = make_s_shell(Point3D(1.0, 2.0, 3.0));
    Shell s2 = make_s_shell(Point3D(4.0, 5.0, 6.0));
    Shell s3 = make_s_shell(Point3D(7.0, 8.0, 9.0));

    std::vector<std::reference_wrapper<const Shell>> refs = {
        std::cref(s1), std::cref(s2), std::cref(s3)
    };
    ShellSet set(refs);

    const auto& soa = set.soa_data();

    EXPECT_EQ(soa.n_shells(), 3u);

    EXPECT_DOUBLE_EQ(soa.center_x[0], 1.0);
    EXPECT_DOUBLE_EQ(soa.center_y[0], 2.0);
    EXPECT_DOUBLE_EQ(soa.center_z[0], 3.0);

    EXPECT_DOUBLE_EQ(soa.center_x[1], 4.0);
    EXPECT_DOUBLE_EQ(soa.center_y[1], 5.0);
    EXPECT_DOUBLE_EQ(soa.center_z[1], 6.0);

    EXPECT_DOUBLE_EQ(soa.center_x[2], 7.0);
    EXPECT_DOUBLE_EQ(soa.center_y[2], 8.0);
    EXPECT_DOUBLE_EQ(soa.center_z[2], 9.0);
}

TEST(ShellSetTest, SoADataPrimitives) {
    // Use pre-normalized shells so coefficients are exactly as specified
    Shell s1(pre_normalized, AngularMomentum::S, Point3D(0.0, 0.0, 0.0),
             {3.0, 2.0, 1.0}, {0.5, 0.3, 0.2});
    Shell s2(pre_normalized, AngularMomentum::S, Point3D(1.0, 0.0, 0.0),
             {6.0, 4.0, 2.0}, {0.1, 0.6, 0.3});

    std::vector<std::reference_wrapper<const Shell>> refs = {
        std::cref(s1), std::cref(s2)
    };
    ShellSet set(refs);

    const auto& soa = set.soa_data();

    // Should have 2 shells * 3 primitives = 6 total primitives
    EXPECT_EQ(soa.n_total_primitives(), 6u);

    // First shell primitives: indices 0, 1, 2
    EXPECT_DOUBLE_EQ(soa.exponents[0], 3.0);
    EXPECT_DOUBLE_EQ(soa.exponents[1], 2.0);
    EXPECT_DOUBLE_EQ(soa.exponents[2], 1.0);
    EXPECT_DOUBLE_EQ(soa.coefficients[0], 0.5);
    EXPECT_DOUBLE_EQ(soa.coefficients[1], 0.3);
    EXPECT_DOUBLE_EQ(soa.coefficients[2], 0.2);

    // Second shell primitives: indices 3, 4, 5
    EXPECT_DOUBLE_EQ(soa.exponents[3], 6.0);
    EXPECT_DOUBLE_EQ(soa.exponents[4], 4.0);
    EXPECT_DOUBLE_EQ(soa.exponents[5], 2.0);
    EXPECT_DOUBLE_EQ(soa.coefficients[3], 0.1);
    EXPECT_DOUBLE_EQ(soa.coefficients[4], 0.6);
    EXPECT_DOUBLE_EQ(soa.coefficients[5], 0.3);
}

TEST(ShellSetTest, SoADataTrackingIndices) {
    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    s1.set_atom_index(0);
    s1.set_shell_index(0);
    s1.set_function_index(0);

    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));
    s2.set_atom_index(1);
    s2.set_shell_index(3);
    s2.set_function_index(7);

    Shell s3 = make_s_shell(Point3D(2.0, 0.0, 0.0));
    s3.set_atom_index(1);
    s3.set_shell_index(5);
    s3.set_function_index(10);

    std::vector<std::reference_wrapper<const Shell>> refs = {
        std::cref(s1), std::cref(s2), std::cref(s3)
    };
    ShellSet set(refs);

    const auto& soa = set.soa_data();

    // Shell indices
    EXPECT_EQ(soa.shell_indices[0], 0);
    EXPECT_EQ(soa.shell_indices[1], 3);
    EXPECT_EQ(soa.shell_indices[2], 5);

    // Atom indices
    EXPECT_EQ(soa.atom_indices[0], 0);
    EXPECT_EQ(soa.atom_indices[1], 1);
    EXPECT_EQ(soa.atom_indices[2], 1);

    // Function offsets
    EXPECT_EQ(soa.function_offsets[0], 0);
    EXPECT_EQ(soa.function_offsets[1], 7);
    EXPECT_EQ(soa.function_offsets[2], 10);
}

TEST(ShellSetTest, SoADataMatchesSourceShells) {
    // Verify all SoA data matches the original shells exactly
    Shell s1(pre_normalized, AngularMomentum::P, Point3D(1.5, -0.5, 2.0),
             {5.0, 1.0}, {0.7, 0.3});
    s1.set_atom_index(2);
    s1.set_shell_index(4);
    s1.set_function_index(12);

    Shell s2(pre_normalized, AngularMomentum::P, Point3D(-1.0, 3.0, 0.0),
             {8.0, 3.0}, {0.4, 0.6});
    s2.set_atom_index(5);
    s2.set_shell_index(9);
    s2.set_function_index(27);

    std::vector<std::reference_wrapper<const Shell>> refs = {
        std::cref(s1), std::cref(s2)
    };
    ShellSet set(refs);

    const auto& soa = set.soa_data();
    const int k = set.n_primitives_per_shell();

    for (Size i = 0; i < set.n_shells(); ++i) {
        const Shell& shell = set.shell(i);

        // Centers
        EXPECT_DOUBLE_EQ(soa.center_x[i], shell.center().x);
        EXPECT_DOUBLE_EQ(soa.center_y[i], shell.center().y);
        EXPECT_DOUBLE_EQ(soa.center_z[i], shell.center().z);

        // Tracking indices
        EXPECT_EQ(soa.shell_indices[i], shell.shell_index());
        EXPECT_EQ(soa.atom_indices[i], shell.atom_index());
        EXPECT_EQ(soa.function_offsets[i], shell.function_index());

        // Primitive data
        auto exps = shell.exponents();
        auto coeffs = shell.coefficients();
        for (int j = 0; j < k; ++j) {
            EXPECT_DOUBLE_EQ(soa.exponents[i * static_cast<Size>(k) + static_cast<Size>(j)], exps[static_cast<Size>(j)]);
            EXPECT_DOUBLE_EQ(soa.coefficients[i * static_cast<Size>(k) + static_cast<Size>(j)], coeffs[static_cast<Size>(j)]);
        }
    }
}

// =============================================================================
// Lazy Initialization
// =============================================================================

TEST(ShellSetTest, LazyInitializationNotReadyBeforeAccess) {
    Shell s = make_s_shell(Point3D(0.0, 0.0, 0.0));
    std::vector<std::reference_wrapper<const Shell>> refs = {std::cref(s)};
    ShellSet set(refs);

    // SoA should not be ready before first access
    EXPECT_FALSE(set.soa_ready());
}

TEST(ShellSetTest, LazyInitializationReadyAfterAccess) {
    Shell s = make_s_shell(Point3D(0.0, 0.0, 0.0));
    std::vector<std::reference_wrapper<const Shell>> refs = {std::cref(s)};
    ShellSet set(refs);

    // Trigger lazy init
    const auto& soa = set.soa_data();
    (void)soa;

    EXPECT_TRUE(set.soa_ready());
}

TEST(ShellSetTest, LazyInitializationConsistentResults) {
    Shell s1 = make_s_shell(Point3D(1.0, 2.0, 3.0));
    Shell s2 = make_s_shell(Point3D(4.0, 5.0, 6.0));

    std::vector<std::reference_wrapper<const Shell>> refs = {
        std::cref(s1), std::cref(s2)
    };
    ShellSet set(refs);

    // Multiple calls should return the same data
    const auto& soa1 = set.soa_data();
    const auto& soa2 = set.soa_data();

    // Should be the same object (same address)
    EXPECT_EQ(&soa1, &soa2);

    // And same contents
    EXPECT_EQ(soa1.n_shells(), soa2.n_shells());
    EXPECT_DOUBLE_EQ(soa1.center_x[0], soa2.center_x[0]);
    EXPECT_DOUBLE_EQ(soa1.center_x[1], soa2.center_x[1]);
}

TEST(ShellSetTest, AddShellInvalidatesSoA) {
    ShellSet set(0, 3);

    Shell s1 = make_s_shell(Point3D(0.0, 0.0, 0.0));
    set.add_shell(s1);

    // Trigger SoA build
    const auto& soa1 = set.soa_data();
    EXPECT_EQ(soa1.n_shells(), 1u);
    EXPECT_TRUE(set.soa_ready());

    // Add another shell - should invalidate SoA
    Shell s2 = make_s_shell(Point3D(1.0, 0.0, 0.0));
    set.add_shell(s2);
    EXPECT_FALSE(set.soa_ready());

    // Re-trigger SoA build - should now include both shells
    const auto& soa2 = set.soa_data();
    EXPECT_EQ(soa2.n_shells(), 2u);
    EXPECT_TRUE(set.soa_ready());
}

// =============================================================================
// Thread Safety
// =============================================================================

TEST(ShellSetTest, ConcurrentSoAAccess) {
    // Create a ShellSet with several shells
    constexpr Size n_shells = 10;
    std::vector<Shell> shells;
    shells.reserve(n_shells);
    for (Size i = 0; i < n_shells; ++i) {
        shells.push_back(make_s_shell(
            Point3D(static_cast<Real>(i), 0.0, 0.0)));
    }

    std::vector<std::reference_wrapper<const Shell>> refs;
    refs.reserve(n_shells);
    for (const auto& s : shells) {
        refs.push_back(std::cref(s));
    }

    ShellSet set(refs);

    // Concurrently access soa_data from multiple threads
    constexpr int n_threads = 8;
    std::vector<std::thread> threads;
    std::vector<const ShellSetDataSoA*> results(n_threads, nullptr);
    std::vector<Size> result_n_shells(n_threads, 0);
    std::vector<std::vector<Real>> result_center_x(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&set, &results, &result_n_shells, &result_center_x, t]() {
            const auto& soa = set.soa_data();
            results[t] = &soa;
            result_n_shells[t] = soa.n_shells();
            result_center_x[t] = soa.center_x;  // copy for comparison
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // All threads should get the same pointer
    for (int t = 1; t < n_threads; ++t) {
        EXPECT_EQ(results[t], results[0])
            << "Thread " << t << " got a different SoA address";
    }

    // All threads should see the same data
    for (int t = 0; t < n_threads; ++t) {
        EXPECT_EQ(result_n_shells[t], n_shells)
            << "Thread " << t << " saw wrong n_shells";
        ASSERT_EQ(result_center_x[t].size(), n_shells)
            << "Thread " << t << " saw wrong center_x size";
        for (Size i = 0; i < n_shells; ++i) {
            EXPECT_DOUBLE_EQ(result_center_x[t][i], static_cast<Real>(i))
                << "Thread " << t << " saw wrong center_x[" << i << "]";
        }
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(ShellSetTest, SingleShellSoA) {
    Shell s(pre_normalized, AngularMomentum::D, Point3D(1.0, -2.0, 3.5),
            {0.8}, {1.0});
    s.set_atom_index(7);
    s.set_shell_index(12);
    s.set_function_index(42);

    std::vector<std::reference_wrapper<const Shell>> refs = {std::cref(s)};
    ShellSet set(refs);

    const auto& soa = set.soa_data();
    EXPECT_EQ(soa.n_shells(), 1u);
    EXPECT_EQ(soa.n_total_primitives(), 1u);

    EXPECT_DOUBLE_EQ(soa.center_x[0], 1.0);
    EXPECT_DOUBLE_EQ(soa.center_y[0], -2.0);
    EXPECT_DOUBLE_EQ(soa.center_z[0], 3.5);

    EXPECT_DOUBLE_EQ(soa.exponents[0], 0.8);
    EXPECT_DOUBLE_EQ(soa.coefficients[0], 1.0);

    EXPECT_EQ(soa.atom_indices[0], 7);
    EXPECT_EQ(soa.shell_indices[0], 12);
    EXPECT_EQ(soa.function_offsets[0], 42);
}

TEST(ShellSetTest, NFunctionsPerShell) {
    // S: 1, P: 3, D: 6, F: 10
    {
        ShellSet set(0, 1);
        EXPECT_EQ(set.n_functions_per_shell(), 1);
    }
    {
        ShellSet set(1, 1);
        EXPECT_EQ(set.n_functions_per_shell(), 3);
    }
    {
        ShellSet set(2, 1);
        EXPECT_EQ(set.n_functions_per_shell(), 6);
    }
    {
        ShellSet set(3, 1);
        EXPECT_EQ(set.n_functions_per_shell(), 10);
    }
}

TEST(ShellSetTest, NTotalFunctions) {
    // 4 P-shells -> 4 * 3 = 12 total functions
    Shell s1 = make_p_shell(Point3D(0.0, 0.0, 0.0));
    Shell s2 = make_p_shell(Point3D(1.0, 0.0, 0.0));
    Shell s3 = make_p_shell(Point3D(2.0, 0.0, 0.0));
    Shell s4 = make_p_shell(Point3D(3.0, 0.0, 0.0));

    std::vector<std::reference_wrapper<const Shell>> refs = {
        std::cref(s1), std::cref(s2), std::cref(s3), std::cref(s4)
    };
    ShellSet set(refs);

    EXPECT_EQ(set.n_total_functions(), 12u);
}

TEST(ShellSetTest, HighAngularMomentumConstruction) {
    // Construct ShellSet at max AM
    ShellSet set(MAX_ANGULAR_MOMENTUM, 2);
    EXPECT_EQ(set.angular_momentum(), MAX_ANGULAR_MOMENTUM);
    EXPECT_EQ(set.n_primitives_per_shell(), 2);
    EXPECT_EQ(set.n_functions_per_shell(), n_cartesian(MAX_ANGULAR_MOMENTUM));
}

TEST(ShellSetTest, ManyShellsPerformance) {
    // Create a ShellSet with many shells and verify SoA data
    constexpr Size n_shells = 100;
    std::vector<Shell> shells;
    shells.reserve(n_shells);
    for (Size i = 0; i < n_shells; ++i) {
        Real x = static_cast<Real>(i) * 0.1;
        shells.push_back(make_s_shell(Point3D(x, 0.0, 0.0)));
    }

    std::vector<std::reference_wrapper<const Shell>> refs;
    refs.reserve(n_shells);
    for (const auto& s : shells) {
        refs.push_back(std::cref(s));
    }

    ShellSet set(refs);

    const auto& soa = set.soa_data();
    EXPECT_EQ(soa.n_shells(), n_shells);
    EXPECT_EQ(soa.n_total_primitives(), n_shells * 3);  // 3 primitives each

    // Spot check a few entries
    EXPECT_DOUBLE_EQ(soa.center_x[0], 0.0);
    EXPECT_DOUBLE_EQ(soa.center_x[50], 5.0);
    EXPECT_DOUBLE_EQ(soa.center_x[99], 9.9);
}
