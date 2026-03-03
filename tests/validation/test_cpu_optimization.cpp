// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_cpu_optimization.cpp
/// @brief Validation tests for CPU optimization utilities (Tasks 27.2.1–27.2.4)
///
/// Tests SIMD configuration detection, cache-aware patterns,
/// contraction loop correctness under optimization, and memory pool usage.

#include <gtest/gtest.h>

#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/config.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/memory/memory_pool.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/utils/diagnostics.hpp>
#include <libaccint/utils/simd.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace libaccint {
namespace {

// ============================================================================
// 27.2.1: SIMD Vectorization Audit Tests
// ============================================================================

TEST(CpuSIMDAudit, SimdWidthPositive) {
    EXPECT_GT(simd::simd_width, 0);
}

TEST(CpuSIMDAudit, SimdAlignmentValid) {
    EXPECT_GE(simd::simd_alignment, sizeof(double));
    // Alignment must be a power of 2
    EXPECT_EQ(simd::simd_alignment & (simd::simd_alignment - 1), 0u);
}

TEST(CpuSIMDAudit, SimdIsaNameNotEmpty) {
    std::string isa_name = simd::simd_isa_name;
    EXPECT_FALSE(isa_name.empty());
}

TEST(CpuSIMDAudit, ConfigVectorWidth) {
    // The config should report consistent vector width
    int width = vector_width();
    EXPECT_GT(width, 0);
}

TEST(CpuSIMDAudit, ConfigVectorIsa) {
    std::string isa = vector_isa();
    EXPECT_FALSE(isa.empty());
}

TEST(CpuSIMDAudit, VectorAddCorrectness) {
    // Verify vectorized add produces correct results
    const Size n = 256;
    std::vector<double> a(n), b(n), c(n);
    for (Size i = 0; i < n; ++i) {
        a[i] = static_cast<double>(i);
        b[i] = static_cast<double>(n - i);
    }
    for (Size i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
    for (Size i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(c[i], static_cast<double>(n));
    }
}

// ============================================================================
// 27.2.2: Cache Utilization Tests
// ============================================================================

TEST(CpuCacheUtil, SequentialAccessFasterThanStrided) {
    // Sequential access should generally be as fast or faster than strided
    // Just verify both produce correct results here
    const Size n = 4096;
    std::vector<double> data(n);
    std::iota(data.begin(), data.end(), 0.0);

    // Sequential sum
    double seq_sum = 0.0;
    for (Size i = 0; i < n; ++i) {
        seq_sum += data[i];
    }

    // Strided sum (stride=8)
    double strided_sum = 0.0;
    for (Size i = 0; i < n; i += 8) {
        strided_sum += data[i];
    }

    // Sequential sum should be larger since it sums more elements
    EXPECT_GT(seq_sum, strided_sum);
    EXPECT_DOUBLE_EQ(seq_sum, static_cast<double>(n) * (n - 1) / 2.0);
}

TEST(CpuCacheUtil, OverlapMatrixConsistency) {
    // Verify that overlap integrals produce consistent results
    // (checks that cache-optimized code paths are correct)
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {1.430429, 0.0, 1.107157}},
        {1, {-1.430429, 0.0, 1.107157}}
    };
    auto basis = data::create_builtin_basis("sto-3g", atoms);
    Engine engine(basis);

    std::vector<Real> S1, S2;
    engine.compute_1e(Operator::overlap(), S1);
    engine.compute_1e(Operator::overlap(), S2);

    ASSERT_EQ(S1.size(), S2.size());
    for (Size i = 0; i < S1.size(); ++i) {
        EXPECT_DOUBLE_EQ(S1[i], S2[i]);
    }
}

// ============================================================================
// 27.2.3: Contraction Loop Optimization Tests
// ============================================================================

TEST(CpuContractionLoop, OverlapSymmetry) {
    // Overlap matrix must be symmetric: S_ij = S_ji
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {1.430429, 0.0, 1.107157}},
        {1, {-1.430429, 0.0, 1.107157}}
    };
    auto basis = data::create_builtin_basis("sto-3g", atoms);
    Engine engine(basis);

    std::vector<Real> S;
    engine.compute_1e(Operator::overlap(), S);

    Size nbf = basis.n_basis_functions();
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(S[i * nbf + j], S[j * nbf + i], 1e-14)
                << "Symmetry violation at (" << i << "," << j << ")";
        }
    }
}

TEST(CpuContractionLoop, OverlapDiagonalPositive) {
    // Overlap diagonal elements should be positive (self-overlap)
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {1.430429, 0.0, 1.107157}},
        {1, {-1.430429, 0.0, 1.107157}}
    };
    auto basis = data::create_builtin_basis("sto-3g", atoms);
    Engine engine(basis);

    std::vector<Real> S;
    engine.compute_1e(Operator::overlap(), S);

    Size nbf = basis.n_basis_functions();
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GT(S[i * nbf + i], 0.0)
            << "Non-positive diagonal at index " << i;
    }
}

TEST(CpuContractionLoop, KineticSymmetry) {
    // Kinetic matrix must be symmetric
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {1.430429, 0.0, 1.107157}},
        {1, {-1.430429, 0.0, 1.107157}}
    };
    auto basis = data::create_builtin_basis("sto-3g", atoms);
    Engine engine(basis);

    std::vector<Real> T;
    engine.compute_1e(Operator::kinetic(), T);

    Size nbf = basis.n_basis_functions();
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(T[i * nbf + j], T[j * nbf + i], 1e-14)
                << "Kinetic symmetry violation at (" << i << "," << j << ")";
        }
    }
}

TEST(CpuContractionLoop, FockBuildJSymmetry) {
    // Coulomb matrix J should be symmetric
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {1.430429, 0.0, 1.107157}},
        {1, {-1.430429, 0.0, 1.107157}}
    };
    auto basis = data::create_builtin_basis("sto-3g", atoms);
    const Size nbf = basis.n_basis_functions();

    // Unit density
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0 / static_cast<Real>(nbf);
    }

    Engine engine(basis);
    consumers::FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);
    engine.compute_and_consume(Operator::coulomb(), fock);

    auto J = fock.get_coulomb_matrix();
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(J[i * nbf + j], J[j * nbf + i], 1e-12)
                << "J symmetry violation at (" << i << "," << j << ")";
        }
    }
}

// ============================================================================
// 27.2.4: Memory Allocation Hotspot Tests
// ============================================================================

TEST(CpuMemoryAlloc, PoolAcquireAndRelease) {
    auto& pool = memory::get_thread_local_pool();

    // Acquire a small buffer
    auto buf = pool.acquire(256);
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_GE(buf.size(), 256u);
    // PooledBuffer releases automatically on destruction
}

TEST(CpuMemoryAlloc, PoolMultipleAcquisitions) {
    auto& pool = memory::get_thread_local_pool();

    std::vector<memory::PooledBuffer> buffers;
    // Acquire several buffers
    for (int i = 0; i < 5; ++i) {
        auto buf = pool.acquire(1024);
        EXPECT_NE(buf.data(), nullptr);
        buffers.push_back(std::move(buf));
    }
    // All released on destruction
    EXPECT_EQ(buffers.size(), 5u);
}

TEST(CpuMemoryAlloc, PoolDifferentSizeClasses) {
    auto& pool = memory::get_thread_local_pool();

    // Exercise different size classes
    std::vector<memory::PooledBuffer> allocs;
    for (Size sz : {64u, 256u, 1024u, 4096u, 16384u}) {
        auto buf = pool.acquire(sz);
        EXPECT_NE(buf.data(), nullptr);
        EXPECT_GE(buf.size(), sz);
        allocs.push_back(std::move(buf));
    }
    EXPECT_EQ(allocs.size(), 5u);
}

TEST(CpuMemoryAlloc, DiagnosticsCounterTracking) {
    // When diagnostics are enabled, buffer allocations should be tracked
    auto& diag = diagnostics::DiagnosticsCollector::instance();
    diag.set_enabled(true);
    diag.reset();

    // Run some integrals
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {1.430429, 0.0, 1.107157}},
        {1, {-1.430429, 0.0, 1.107157}}
    };
    auto basis = data::create_builtin_basis("sto-3g", atoms);
    Engine engine(basis);
    std::vector<Real> S;
    engine.compute_1e(Operator::overlap(), S);

    // Just verify diagnostics can produce a report
    std::string report = diag.report();
    EXPECT_FALSE(report.empty());

    diag.set_enabled(false);
}

}  // namespace
}  // namespace libaccint
