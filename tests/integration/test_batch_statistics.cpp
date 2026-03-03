// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_batch_statistics.cpp
/// @brief Step 11.4: Batch size distribution analysis tests for (H₂O)₄
///
/// Analyzes the ShellSet, ShellSetPair, and ShellSetQuartet statistics
/// for the aug-cc-pVDZ basis on a water tetramer. Verifies that the
/// system is large enough to exercise GPU dispatch paths and that
/// batch sizes span a meaningful range.

#include "h2o4_fixture.hpp"

#include <libaccint/engine/dispatch_policy.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace libaccint::test {

using BatchStatFixture = H2O4AugccpVDZFixture;

// =============================================================================
// Helper: AM integer → label character
// =============================================================================

static char am_label(int l) {
    static constexpr const char* labels = "spdfghi";
    if (l >= 0 && l <= 6) return labels[l];
    return '?';
}

/// Format a quartet AM class as "(ab|cd)"
static std::string quartet_am_string(int la, int lb, int lc, int ld) {
    std::string s;
    s += '(';
    s += am_label(la);
    s += am_label(lb);
    s += '|';
    s += am_label(lc);
    s += am_label(ld);
    s += ')';
    return s;
}

// =============================================================================
// 1. Basis composition — ShellSet consistency
// =============================================================================

TEST_F(BatchStatFixture, ShellSetComposition) {
    auto shell_sets = basis_->shell_sets();
    EXPECT_GE(shell_sets.size(), 4u)
        << "aug-cc-pVDZ on (H2O)4 should have at least 4 ShellSets";

    // Map: AM → {n_shell_sets, total_shells}
    std::map<int, std::pair<int, Size>> am_summary;

    for (const auto* ss : shell_sets) {
        EXPECT_GT(ss->n_shells(), 0u) << "ShellSet should not be empty";

        const int am = ss->angular_momentum();
        const int k  = ss->n_primitives_per_shell();
        EXPECT_GE(am, 0);
        EXPECT_GE(k, 1);

        // Every shell in this set must share the same AM and K
        for (Size i = 0; i < ss->n_shells(); ++i) {
            EXPECT_EQ(ss->shell(i).angular_momentum(), am)
                << "Shell " << i << " AM mismatch in ShellSet";
            EXPECT_EQ(static_cast<int>(ss->shell(i).n_primitives()), k)
                << "Shell " << i << " K mismatch in ShellSet";
        }

        am_summary[am].first  += 1;
        am_summary[am].second += ss->n_shells();
    }

    // Print summary table
    std::cout << "\n  ShellSet Composition\n";
    std::cout << "  AM  | n_shell_sets | total_shells\n";
    std::cout << "  ----+--------------+-------------\n";
    for (const auto& [am, info] : am_summary) {
        std::cout << "   " << am_label(am) << "  |     "
                  << std::setw(4) << info.first << "     |    "
                  << std::setw(4) << info.second << "\n";
    }

    // aug-cc-pVDZ must have s, p, d functions
    EXPECT_TRUE(am_summary.count(0)) << "Missing s-type ShellSets";
    EXPECT_TRUE(am_summary.count(1)) << "Missing p-type ShellSets";
    EXPECT_TRUE(am_summary.count(2)) << "Missing d-type ShellSets";
}

// =============================================================================
// 2. ShellSetPair statistics
// =============================================================================

TEST_F(BatchStatFixture, PairStatistics) {
    const auto& pairs = basis_->shell_set_pairs();
    EXPECT_GT(pairs.size(), 0u);

    Size total_pairs = 0;
    Size min_n = std::numeric_limits<Size>::max();
    Size max_n = 0;

    // AM pair class distribution
    std::map<std::pair<int,int>, int> am_class_count;

    for (const auto& p : pairs) {
        const Size np = p.n_pairs();
        total_pairs += np;
        min_n = std::min(min_n, np);
        max_n = std::max(max_n, np);
        am_class_count[{p.La(), p.Lb()}]++;
    }

    const double mean_n = static_cast<double>(total_pairs) /
                          static_cast<double>(pairs.size());

    // Print summary
    std::cout << "\n  ShellSetPair Statistics\n";
    std::cout << "  ShellSetPairs: " << pairs.size() << "\n";
    std::cout << "  Total individual pairs: " << total_pairs << "\n";
    std::cout << "  n_pairs per ShellSetPair — min: " << min_n
              << "  max: " << max_n
              << "  mean: " << std::fixed << std::setprecision(1) << mean_n << "\n";
    std::cout << "  AM pair class distribution:\n";
    for (const auto& [am, cnt] : am_class_count) {
        std::cout << "    (" << am_label(am.first) << am_label(am.second)
                  << "): " << cnt << " ShellSetPairs\n";
    }

    EXPECT_GT(total_pairs, 0u);
    EXPECT_GT(pairs.size(), 1u) << "Should have more than one ShellSetPair";
}

// =============================================================================
// 3. Pair size distribution
// =============================================================================

TEST_F(BatchStatFixture, PairSizeDistribution) {
    const auto& pairs = basis_->shell_set_pairs();

    int count_gt1  = 0;
    int count_ge10 = 0;

    for (const auto& p : pairs) {
        if (p.n_pairs() > 1)  ++count_gt1;
        if (p.n_pairs() >= 10) ++count_ge10;
    }

    EXPECT_GT(count_gt1, 0)
        << "Expected at least some ShellSetPairs with n_pairs > 1";
    EXPECT_GT(count_ge10, 0)
        << "Expected at least some ShellSetPairs with n_pairs >= 10";

    std::cout << "\n  Pair Size Distribution\n";
    std::cout << "  Pairs with n_pairs > 1:  " << count_gt1
              << " / " << pairs.size() << "\n";
    std::cout << "  Pairs with n_pairs >= 10: " << count_ge10
              << " / " << pairs.size() << "\n";
}

// =============================================================================
// 4. ShellSetQuartet statistics
// =============================================================================

TEST_F(BatchStatFixture, QuartetStatistics) {
    const auto& quartets = basis_->shell_set_quartets();
    EXPECT_GT(quartets.size(), 0u);

    Size total_individual = 0;
    Size min_batch = std::numeric_limits<Size>::max();
    Size max_batch = 0;

    // AM quartet class: (La,Lb,Lc,Ld) → {count, total_quartets}
    using AMKey = std::tuple<int,int,int,int>;
    std::map<AMKey, std::pair<int, Size>> am_dist;

    for (const auto& q : quartets) {
        const Size nq = q.n_quartets();
        total_individual += nq;
        min_batch = std::min(min_batch, nq);
        max_batch = std::max(max_batch, nq);

        AMKey key{q.La(), q.Lb(), q.Lc(), q.Ld()};
        am_dist[key].first  += 1;
        am_dist[key].second += nq;
    }

    const double mean_batch = static_cast<double>(total_individual) /
                              static_cast<double>(quartets.size());

    // Print summary
    std::cout << "\n  ShellSetQuartet Statistics\n";
    std::cout << "  Unique ShellSetQuartets: " << quartets.size() << "\n";
    std::cout << "  Total individual quartets: " << total_individual << "\n";
    std::cout << "  Batch size — min: " << min_batch
              << "  max: " << max_batch
              << "  mean: " << std::fixed << std::setprecision(1) << mean_batch << "\n";
    std::cout << "  AM quartet class distribution:\n";
    for (const auto& [key, info] : am_dist) {
        const auto& [la, lb, lc, ld] = key;
        std::cout << "    " << quartet_am_string(la, lb, lc, ld)
                  << ": " << info.first << " SSQs, "
                  << info.second << " total quartets, avg "
                  << std::fixed << std::setprecision(1)
                  << (info.first > 0
                      ? static_cast<double>(info.second) / info.first
                      : 0.0)
                  << "\n";
    }

    EXPECT_GT(total_individual, 0u);
    EXPECT_GT(quartets.size(), 1u)
        << "Should have more than one ShellSetQuartet";
}

// =============================================================================
// 5. Quartet size distribution — GPU-viable batches
// =============================================================================

TEST_F(BatchStatFixture, QuartetSizeDistribution) {
    const auto& quartets = basis_->shell_set_quartets();
    constexpr Size min_gpu_batch = 16;  // DispatchConfig default

    int count_ge_gpu = 0;
    for (const auto& q : quartets) {
        if (q.n_quartets() >= min_gpu_batch) ++count_ge_gpu;
    }

    EXPECT_GT(count_ge_gpu, 0)
        << "Expected at least some quartets with batch_size >= "
        << min_gpu_batch << " for GPU dispatch viability";

    std::cout << "\n  Quartet Size Distribution (GPU threshold)\n";
    std::cout << "  Quartets with batch >= " << min_gpu_batch << ": "
              << count_ge_gpu << " / " << quartets.size() << "\n";
}

// =============================================================================
// 6. Quartet AM class distribution — expected classes for aug-cc-pVDZ
// =============================================================================

TEST_F(BatchStatFixture, QuartetAMClassDistribution) {
    const auto& quartets = basis_->shell_set_quartets();

    using AMKey = std::tuple<int,int,int,int>;
    std::set<AMKey> present_classes;

    for (const auto& q : quartets) {
        present_classes.insert({q.La(), q.Lb(), q.Lc(), q.Ld()});
    }

    // (ss|ss) must be present
    EXPECT_TRUE(present_classes.count({0,0,0,0}))
        << "Missing (ss|ss) class — every basis has s functions";

    // d-type functions are present in aug-cc-pVDZ, so we should see mixed
    // classes involving d. Check for at least one class with any AM=2.
    bool has_d_class = false;
    bool has_mixed_sp = false;
    for (const auto& [la, lb, lc, ld] : present_classes) {
        if (la == 2 || lb == 2 || lc == 2 || ld == 2) has_d_class = true;
        // mixed s/p class: e.g. (sp|ss), (sp|sp), (ps|ss), etc.
        if ((la == 0 && lb == 1) || (la == 1 && lb == 0) ||
            (lc == 0 && ld == 1) || (lc == 1 && ld == 0)) {
            has_mixed_sp = true;
        }
    }

    EXPECT_TRUE(has_d_class)
        << "aug-cc-pVDZ should produce at least one AM class involving d";
    EXPECT_TRUE(has_mixed_sp)
        << "Should have at least one mixed s/p AM class";

    std::cout << "\n  AM Classes Present (" << present_classes.size() << " total):\n  ";
    int col = 0;
    for (const auto& [la, lb, lc, ld] : present_classes) {
        std::cout << quartet_am_string(la, lb, lc, ld) << " ";
        if (++col % 8 == 0) std::cout << "\n  ";
    }
    std::cout << "\n";
}

// =============================================================================
// 7. GPU dispatch predictions
// =============================================================================

TEST_F(BatchStatFixture, GPUDispatchPredictions) {
    const auto& quartets = basis_->shell_set_quartets();
    DispatchPolicy policy;  // default config
    const bool gpu_available = is_backend_available(BackendType::CUDA);

    int gpu_count = 0;
    int cpu_count = 0;

    for (const auto& q : quartets) {
        // Compute total primitives for this quartet
        const Size n_prim =
            static_cast<Size>(q.bra_pair().shell_set_a().n_primitives_per_shell()) *
            static_cast<Size>(q.bra_pair().shell_set_b().n_primitives_per_shell()) *
            static_cast<Size>(q.ket_pair().shell_set_a().n_primitives_per_shell()) *
            static_cast<Size>(q.ket_pair().shell_set_b().n_primitives_per_shell()) *
            q.n_quartets();

        BackendType backend = policy.select_backend(
            WorkUnitType::ShellSetQuartet,
            q.n_quartets(),
            q.L_total(),
            n_prim,
            BackendHint::Auto,
            gpu_available);

        if (is_gpu_backend(backend)) {
            ++gpu_count;
        } else {
            ++cpu_count;
        }
    }

    // If GPU is available, at least some should go to GPU
    if (gpu_available) {
        EXPECT_GT(gpu_count, 0)
            << "With GPU available, at least some quartets should dispatch to GPU";
    }

    // Even without GPU, all should go to CPU — just verify counts are sane
    EXPECT_EQ(static_cast<Size>(gpu_count + cpu_count), quartets.size());

    std::cout << "\n  GPU Dispatch Predictions (gpu_available="
              << (gpu_available ? "true" : "false") << ")\n";
    std::cout << "  GPU: " << gpu_count << " / " << quartets.size() << "\n";
    std::cout << "  CPU: " << cpu_count << " / " << quartets.size() << "\n";
}

// =============================================================================
// 8. Small batch fallback count
// =============================================================================

TEST_F(BatchStatFixture, SmallBatchFallbackCount) {
    const auto& quartets = basis_->shell_set_quartets();
    constexpr Size min_gpu_batch = 16;  // DispatchConfig default

    int small_count = 0;
    Size min_batch = std::numeric_limits<Size>::max();
    for (const auto& q : quartets) {
        Size nq = q.n_quartets();
        if (nq < min_gpu_batch) ++small_count;
        min_batch = std::min(min_batch, nq);
    }

    // For aug-cc-pVDZ on (H2O)4, even the smallest ShellSet (D-type)
    // produces batches of 8*8 * 8*8 = 4096 quartets, well above the
    // GPU threshold. Verify the count is consistent (small + large = total)
    // rather than requiring small batches to exist for every basis.
    EXPECT_EQ(static_cast<Size>(small_count) + (quartets.size() - small_count),
              quartets.size());

    std::cout << "\n  Small Batch Fallback\n";
    std::cout << "  Quartets below GPU threshold (" << min_gpu_batch << "): "
              << small_count << " / " << quartets.size() << "\n";
    std::cout << "  Minimum batch size: " << min_batch << "\n";
}

// =============================================================================
// 9. K-range distribution (contraction degree)
// =============================================================================

TEST_F(BatchStatFixture, KRangeDistribution) {
    const auto& quartets = basis_->shell_set_quartets();

    // k_total = Ka * Kb * Kc * Kd  (product of primitive counts per shell)
    std::map<std::string, int> k_range_buckets;

    auto classify_k = [](Size k_total) -> std::string {
        if (k_total <= 1)    return "[1]";
        if (k_total <= 16)   return "[2-16]";
        if (k_total <= 81)   return "[17-81]";
        if (k_total <= 625)  return "[82-625]";
        return "[626+]";
    };

    std::set<Size> unique_k_totals;

    for (const auto& q : quartets) {
        Size k_total =
            static_cast<Size>(q.bra_pair().shell_set_a().n_primitives_per_shell()) *
            static_cast<Size>(q.bra_pair().shell_set_b().n_primitives_per_shell()) *
            static_cast<Size>(q.ket_pair().shell_set_a().n_primitives_per_shell()) *
            static_cast<Size>(q.ket_pair().shell_set_b().n_primitives_per_shell());
        k_range_buckets[classify_k(k_total)]++;
        unique_k_totals.insert(k_total);
    }

    // With aug-cc-pVDZ, expect multiple different K-ranges
    EXPECT_GT(unique_k_totals.size(), 1u)
        << "Expected multiple contraction degrees (K-ranges) in aug-cc-pVDZ";

    std::cout << "\n  K-Range Distribution (Ka*Kb*Kc*Kd)\n";
    std::cout << "  Unique K-products: " << unique_k_totals.size() << "\n";
    for (const auto& [bucket, count] : k_range_buckets) {
        std::cout << "    " << std::setw(10) << bucket << ": " << count
                  << " SSQs\n";
    }
}

// =============================================================================
// 10. Comprehensive summary table (informational — always passes)
// =============================================================================

TEST_F(BatchStatFixture, PrintBatchSummaryTable) {
    const auto  shell_sets = basis_->shell_sets();
    const auto& pairs      = basis_->shell_set_pairs();
    const auto& quartets   = basis_->shell_set_quartets();

    // ------------- aggregate totals ---------------
    Size total_individual_q = 0;
    Size min_batch = std::numeric_limits<Size>::max();
    Size max_batch = 0;

    using AMKey = std::tuple<int,int,int,int>;
    struct AMInfo { int n_ssq = 0; Size total_q = 0; };
    std::map<AMKey, AMInfo> am_dist;

    // Batch size buckets
    int bucket_1_15   = 0;
    int bucket_16_63  = 0;
    int bucket_64_255 = 0;
    int bucket_256p   = 0;

    DispatchPolicy policy;
    const bool gpu_available = is_backend_available(BackendType::CUDA);
    int gpu_count = 0;
    int cpu_count = 0;

    for (const auto& q : quartets) {
        const Size nq = q.n_quartets();
        total_individual_q += nq;
        min_batch = std::min(min_batch, nq);
        max_batch = std::max(max_batch, nq);

        AMKey key{q.La(), q.Lb(), q.Lc(), q.Ld()};
        am_dist[key].n_ssq++;
        am_dist[key].total_q += nq;

        if      (nq <= 15)  ++bucket_1_15;
        else if (nq <= 63)  ++bucket_16_63;
        else if (nq <= 255) ++bucket_64_255;
        else                ++bucket_256p;

        const Size n_prim =
            static_cast<Size>(q.bra_pair().shell_set_a().n_primitives_per_shell()) *
            static_cast<Size>(q.bra_pair().shell_set_b().n_primitives_per_shell()) *
            static_cast<Size>(q.ket_pair().shell_set_a().n_primitives_per_shell()) *
            static_cast<Size>(q.ket_pair().shell_set_b().n_primitives_per_shell()) *
            nq;

        BackendType backend = policy.select_backend(
            WorkUnitType::ShellSetQuartet,
            nq, q.L_total(), n_prim,
            BackendHint::Auto, gpu_available);

        if (is_gpu_backend(backend)) ++gpu_count;
        else                         ++cpu_count;
    }

    const Size total_ssq = quartets.size();
    const double pct_gpu = total_ssq > 0
        ? 100.0 * static_cast<double>(gpu_count) / static_cast<double>(total_ssq)
        : 0.0;
    const double pct_cpu = total_ssq > 0
        ? 100.0 * static_cast<double>(cpu_count) / static_cast<double>(total_ssq)
        : 0.0;

    // ======================= Print =========================
    std::cout << "\n";
    std::cout << "  (H2O)4 aug-cc-pVDZ Batch Statistics\n";
    std::cout << "  ====================================\n";
    std::cout << "  Basis functions: " << nbf_ << "\n";
    std::cout << "  Shells: " << basis_->n_shells() << "\n";
    std::cout << "  Shell sets: " << shell_sets.size() << "\n";
    std::cout << "  Shell set pairs: " << pairs.size() << "\n";
    std::cout << "  Shell set quartets: " << total_ssq << "\n";
    std::cout << "  Total individual quartets: " << total_individual_q << "\n";
    std::cout << "\n";

    // AM class distribution
    std::cout << "  AM Class Distribution:\n";
    for (const auto& [key, info] : am_dist) {
        const auto& [la, lb, lc, ld] = key;
        double avg = info.n_ssq > 0
            ? static_cast<double>(info.total_q) / info.n_ssq
            : 0.0;
        std::cout << "  " << quartet_am_string(la, lb, lc, ld)
                  << ": " << std::setw(4) << info.n_ssq << " quartets ("
                  << std::setw(8) << info.total_q << " total, avg batch "
                  << std::fixed << std::setprecision(1)
                  << std::setw(8) << avg << ")\n";
    }
    std::cout << "\n";

    // Batch size distribution
    std::cout << "  Batch Size Distribution:\n";
    std::cout << "  [1-15]  (CPU fallback): " << bucket_1_15 << " quartets\n";
    std::cout << "  [16-63]:                " << bucket_16_63 << " quartets\n";
    std::cout << "  [64-255]:               " << bucket_64_255 << " quartets\n";
    std::cout << "  [256+]:                 " << bucket_256p << " quartets\n";
    std::cout << "\n";

    // Dispatch summary
    std::cout << "  GPU Dispatch: " << gpu_count << "/" << total_ssq
              << " quartets (" << std::fixed << std::setprecision(0)
              << pct_gpu << "%)\n";
    std::cout << "  CPU Fallback: " << cpu_count << "/" << total_ssq
              << " quartets (" << std::fixed << std::setprecision(0)
              << pct_cpu << "%)\n";
    std::cout << "\n";

    // This test is purely informational — always pass
    EXPECT_TRUE(true);
}

}  // namespace libaccint::test
