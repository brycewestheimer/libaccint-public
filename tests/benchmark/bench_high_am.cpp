// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_high_am.cpp
/// @brief Lightweight high-AM performance validation using GoogleTest.
///
/// Step 13.6: Informational timing tests for high angular momentum integrals
/// using single-atom Neon with cc-pVQZ.  All tests SUCCEED() unconditionally;
/// the purpose is to print timing data, not enforce thresholds.

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/data/basis_parser.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/engine/dispatch_policy.hpp>
#include <libaccint/operators/operator.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace libaccint::test {
namespace {

using Clock = std::chrono::high_resolution_clock;

// =============================================================================
// Test fixture
// =============================================================================

class BenchHighAM : public ::testing::Test {
protected:
    void SetUp() override {
        atoms_ne_ = {{10, {0.0, 0.0, 0.0}}};

        try {
            basis_ = data::load_basis_set("cc-pvqz", atoms_ne_);
            basis_loaded_ = true;
        } catch (const std::exception& e) {
            skip_reason_ = std::string("cc-pVQZ basis loading failed: ") + e.what();
            basis_loaded_ = false;
        }
    }

    /// @brief Helper: format a duration as milliseconds with 3 decimal places
    static std::string fmt_ms(std::chrono::nanoseconds ns) {
        double ms = std::chrono::duration<double, std::milli>(ns).count();
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.3f ms", ms);
        return buf;
    }

    /// @brief Helper: create identity density matrix
    std::vector<Real> make_identity_density(Size nbf) {
        std::vector<Real> D(nbf * nbf, 0.0);
        for (Size i = 0; i < nbf; ++i) {
            D[i * nbf + i] = 1.0;
        }
        return D;
    }

    // -------------------------------------------------------------------------
    // Fixture members
    // -------------------------------------------------------------------------

    std::vector<data::Atom> atoms_ne_;
    BasisSet basis_;
    bool basis_loaded_ = false;
    std::string skip_reason_;
};

// =============================================================================
// 1. Overlap throughput
// =============================================================================

TEST_F(BenchHighAM, OverlapThroughput) {
    if (!basis_loaded_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    const Size nbf = basis_.n_basis_functions();

    std::vector<Real> S;
    auto t0 = Clock::now();
    engine.compute_overlap_matrix(S);
    auto t1 = Clock::now();

    auto elapsed = t1 - t0;
    std::cout << "[  TIMING ] Overlap matrix (" << nbf << " x " << nbf
              << "): " << fmt_ms(elapsed) << std::endl;

    SUCCEED();
}

// =============================================================================
// 2. Kinetic throughput
// =============================================================================

TEST_F(BenchHighAM, KineticThroughput) {
    if (!basis_loaded_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    const Size nbf = basis_.n_basis_functions();

    std::vector<Real> T;
    auto t0 = Clock::now();
    engine.compute_kinetic_matrix(T);
    auto t1 = Clock::now();

    auto elapsed = t1 - t0;
    std::cout << "[  TIMING ] Kinetic matrix (" << nbf << " x " << nbf
              << "): " << fmt_ms(elapsed) << std::endl;

    SUCCEED();
}

// =============================================================================
// 3. ERI throughput via FockBuilder with identity density
// =============================================================================

TEST_F(BenchHighAM, ERIThroughput) {
    if (!basis_loaded_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    const Size nbf = basis_.n_basis_functions();

    auto D = make_identity_density(nbf);
    consumers::FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    Operator coulomb = Operator::coulomb();

    auto t0 = Clock::now();
    engine.compute_and_consume(coulomb, fock);
    auto t1 = Clock::now();

    auto elapsed = t1 - t0;
    std::cout << "[  TIMING ] ERI / FockBuilder (" << nbf << " bf): "
              << fmt_ms(elapsed) << std::endl;

    SUCCEED();
}

// =============================================================================
// 4. AM class breakdown — classify quartets by (La,Lb,Lc,Ld)
// =============================================================================

TEST_F(BenchHighAM, AMClassBreakdown) {
    if (!basis_loaded_) GTEST_SKIP() << skip_reason_;

    const auto& quartets = basis_.shell_set_quartets();

    // Key: "La Lb Lc Ld" → count
    std::map<std::string, int> class_counts;
    int max_am = 0;

    for (const auto& q : quartets) {
        int la = q.La(), lb = q.Lb(), lc = q.Lc(), ld = q.Ld();
        max_am = std::max({max_am, la, lb, lc, ld});

        char key[32];
        std::snprintf(key, sizeof(key), "(%d,%d|%d,%d)", la, lb, lc, ld);
        class_counts[key]++;
    }

    std::cout << "[   INFO  ] Total ShellSetQuartets: " << quartets.size()
              << "  (max AM = " << max_am << ")" << std::endl;
    std::cout << "[   INFO  ] AM class breakdown:" << std::endl;

    for (const auto& [cls, count] : class_counts) {
        std::cout << "[   INFO  ]   " << std::setw(16) << cls
                  << " : " << count << std::endl;
    }

    SUCCEED();
}

// =============================================================================
// 5. Generated vs generic (ForceCPU) timing comparison
// =============================================================================

TEST_F(BenchHighAM, GeneratedVsGenericTiming) {
    if (!basis_loaded_) GTEST_SKIP() << skip_reason_;

    Engine engine(basis_);
    const Size nbf = basis_.n_basis_functions();

    // --- Default dispatch (may use generated kernels) ---
    std::vector<Real> S_default;
    auto t0_default = Clock::now();
    engine.compute_overlap_matrix(S_default);
    auto t1_default = Clock::now();
    auto dt_default = t1_default - t0_default;

    // --- ForceCPU dispatch (generic fallback path) ---
    std::vector<Real> S_cpu;
    auto t0_cpu = Clock::now();
    engine.compute_overlap_matrix(S_cpu, BackendHint::ForceCPU);
    auto t1_cpu = Clock::now();
    auto dt_cpu = t1_cpu - t0_cpu;

    double ms_default = std::chrono::duration<double, std::milli>(dt_default).count();
    double ms_cpu     = std::chrono::duration<double, std::milli>(dt_cpu).count();
    double ratio      = (ms_cpu > 0.0) ? ms_default / ms_cpu : 0.0;

    std::cout << "[  TIMING ] Default dispatch : " << fmt_ms(dt_default) << std::endl;
    std::cout << "[  TIMING ] ForceCPU dispatch: " << fmt_ms(dt_cpu) << std::endl;
    std::cout << "[  TIMING ] Ratio (default / ForceCPU): "
              << std::fixed << std::setprecision(3) << ratio << std::endl;

    SUCCEED();
}

}  // anonymous namespace
}  // namespace libaccint::test
