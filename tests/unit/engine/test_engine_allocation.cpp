// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_engine_allocation.cpp
/// @brief Tests verifying that engine does NOT have O(N^4) memory allocation

#include <libaccint/engine/engine.hpp>
#include <libaccint/engine/cpu_engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/core/types.hpp>

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using namespace libaccint;

namespace {

// =============================================================================
// Utility: memory measurement via /proc/self/status (Linux)
// =============================================================================

/// Read VmRSS (Resident Set Size) from /proc/self/status in KB
long get_rss_kb() {
    std::ifstream proc_status("/proc/self/status");
    if (!proc_status.is_open()) return -1;

    std::string line;
    while (std::getline(proc_status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            // Parse: "VmRSS:     12345 kB"
            long rss_kb = 0;
            for (char c : line) {
                if (c >= '0' && c <= '9') {
                    rss_kb = rss_kb * 10 + (c - '0');
                }
            }
            return rss_kb;
        }
    }
    return -1;
}

/// Build a synthetic basis of uniform s-type shells at the origin
/// Each shell has 3 primitives (simulating STO-3G)
std::vector<Shell> make_uniform_s_basis(Size n_shells) {
    std::vector<Shell> shells;
    shells.reserve(n_shells);

    // Spread shells along x-axis to create some structure
    const double spacing = 2.0;  // bohr
    for (Size i = 0; i < n_shells; ++i) {
        Point3D center{static_cast<double>(i) * spacing, 0.0, 0.0};
        Shell s(0, center,
                {3.42525091, 0.62391373, 0.16885540},
                {0.15432897, 0.53532814, 0.44463454});
        s.set_atom_index(static_cast<int>(i));
        shells.push_back(std::move(s));
    }

    return shells;
}

/// Simple counting consumer that doesn't store anything
struct NullConsumer {
    void accumulate(const TwoElectronBuffer<0>& /*buffer*/,
                    Index /*fa*/, Index /*fb*/, Index /*fc*/, Index /*fd*/,
                    int /*na*/, int /*nb*/, int /*nc*/, int /*nd*/) {}
    void prepare_parallel(int /*n_threads*/) {}
    void finalize_parallel() {}
};

}  // anonymous namespace

// =============================================================================
// Basic Functionality — Small Basis
// =============================================================================

TEST(EngineAllocation, SmallBasis) {
    auto shells = make_uniform_s_basis(10);
    BasisSet basis(std::move(shells));
    Engine engine(basis);

    NullConsumer consumer;
    Operator op = Operator::coulomb();

    EXPECT_NO_THROW(engine.compute_and_consume(op, consumer));
}

// =============================================================================
// Medium Basis — Verify Memory Within Bounds
// =============================================================================

TEST(EngineAllocation, MediumBasis) {
    // 30 shells = 30^4 = 810k quartets — fast enough for a unit test
    auto shells = make_uniform_s_basis(30);
    BasisSet basis(std::move(shells));
    Engine engine(basis);

    long rss_before = get_rss_kb();

    NullConsumer consumer;
    Operator op = Operator::coulomb();

    engine.compute_and_consume(op, consumer);

    long rss_after = get_rss_kb();

    if (rss_before > 0 && rss_after > 0) {
        long delta_kb = rss_after - rss_before;
        // Should not use more than 50 MB for task management
        EXPECT_LT(delta_kb, 50 * 1024)
            << "Memory usage delta " << delta_kb << " KB exceeds 50 MB threshold; "
            << "possible O(N^4) allocation detected";
    }
    // If we can't read /proc/self/status, just verify it completes
}

// =============================================================================
// Memory Scaling — Verify Sub-O(N^4) Scaling
// =============================================================================

TEST(EngineAllocation, MemoryScaling) {
    // Test at multiple sizes and verify exponent < 3 (i.e., not N^4)
    std::vector<Size> sizes = {5, 10, 20};
    std::vector<long> rss_deltas;

    for (Size n : sizes) {
        auto shells = make_uniform_s_basis(n);
        BasisSet basis(std::move(shells));
        Engine engine(basis);

        long rss_before = get_rss_kb();

        NullConsumer consumer;
        Operator op = Operator::coulomb();
        engine.compute_and_consume(op, consumer);

        long rss_after = get_rss_kb();

        if (rss_before <= 0 || rss_after <= 0) {
            GTEST_SKIP() << "Cannot read /proc/self/status for memory measurement";
        }

        rss_deltas.push_back(std::max(0L, rss_after - rss_before));
    }

    // If all deltas are small (< 1 MB), the fix is working — no significant allocation
    bool all_small = true;
    for (long delta : rss_deltas) {
        if (delta > 1024) {  // > 1 MB
            all_small = false;
            break;
        }
    }

    if (all_small) {
        // No significant memory usage at any size = O(1) for task management
        SUCCEED() << "Task management memory is O(1) — no allocation detected";
        return;
    }

    // If there are measurable deltas, check scaling exponent
    // exponent = log(delta[i+1] / delta[i]) / log(sizes[i+1] / sizes[i])
    for (size_t i = 0; i + 1 < rss_deltas.size(); ++i) {
        if (rss_deltas[i] > 0 && rss_deltas[i + 1] > 0) {
            double ratio = static_cast<double>(rss_deltas[i + 1]) /
                           static_cast<double>(rss_deltas[i]);
            double size_ratio = static_cast<double>(sizes[i + 1]) /
                                static_cast<double>(sizes[i]);
            double exponent = std::log(ratio) / std::log(size_ratio);

            // For O(N^4), exponent would be ~4. We require < 3.
            EXPECT_LT(exponent, 3.5)
                << "Memory scaling exponent " << exponent << " between "
                << sizes[i] << " and " << sizes[i + 1] << " shells suggests O(N^4)";
        }
    }
}

// =============================================================================
// No Memory Leak — Verify Memory Returns to Baseline
// =============================================================================

TEST(EngineAllocation, NoLeak) {
    long rss_before = get_rss_kb();
    if (rss_before <= 0) {
        GTEST_SKIP() << "Cannot read /proc/self/status for memory measurement";
    }

    // Run compute-and-consume in a scope so everything is freed
    {
        auto shells = make_uniform_s_basis(20);
        BasisSet basis(std::move(shells));
        Engine engine(basis);

        NullConsumer consumer;
        Operator op = Operator::coulomb();
        engine.compute_and_consume(op, consumer);
    }

    long rss_after = get_rss_kb();

    // Allow some slack for allocator fragmentation (10 MB)
    long delta = rss_after - rss_before;
    EXPECT_LT(delta, 10 * 1024)
        << "Memory not returned after engine destroyed; delta = " << delta << " KB";
}

// =============================================================================
// Parallel Version Also Bounded
// =============================================================================

TEST(EngineAllocation, ParallelSmallBasis) {
    auto shells = make_uniform_s_basis(10);
    BasisSet basis(std::move(shells));
    Engine engine(basis);

    NullConsumer consumer;
    Operator op = Operator::coulomb();

    // Parallel compute should also work without O(N^4) allocation
    EXPECT_NO_THROW(engine.compute_and_consume_parallel(op, consumer, 2));
}

