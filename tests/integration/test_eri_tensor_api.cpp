// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_eri_tensor_api.cpp
/// @brief Integration tests for non-consumer ERI API: compute() overloads,
///        compute_eri_tensor(), compute_eri_block(), screening, parallel
///
/// Step 14.7: Validates all Phase 14 additions to the Engine API.

#include <libaccint/engine/engine.hpp>
#include <libaccint/engine/integral_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/screening/screening_options.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <span>
#include <vector>

using namespace libaccint;
using namespace libaccint::data;
using namespace libaccint::engine;

namespace {

constexpr Real ERI_TOL = 1e-10;
constexpr Real PARALLEL_TOL = 1e-12;

// H2/STO-3G: 2 shells, 2 basis functions
BasisSet make_h2_sto3g() {
    std::vector<Atom> atoms = {
        {1, {0.0, 0.0, 0.0}},
        {1, {0.0, 0.0, 1.39839733}},
    };
    return create_sto3g(atoms);
}

// H2O/STO-3G: 5 shells, 7 basis functions
BasisSet make_h2o_sto3g() {
    std::vector<Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},
        {1, {0.0, 1.43233673, -1.10866041}},
        {1, {0.0, -1.43233673, -1.10866041}},
    };
    return create_sto3g(atoms);
}

BasisSet make_h_chain_sto3g(int n_atoms) {
    std::vector<Atom> atoms;
    atoms.reserve(static_cast<Size>(n_atoms));
    for (int i = 0; i < n_atoms; ++i) {
        atoms.push_back({1, {static_cast<Real>(i) * 2.0, 0.0, 0.0}});
    }
    return create_sto3g(atoms);
}

}  // anonymous namespace

// =============================================================================
// compute(op, ShellSetQuartet) → IntegralBuffer
// =============================================================================

TEST(EriTensorApi, ComputeShellSetQuartetReturnsIntegralBuffer) {
    auto basis = make_h2_sto3g();
    Engine engine(basis);
    Operator coulomb = Operator::coulomb();

    const auto& quartets = basis.shell_set_quartets();
    ASSERT_FALSE(quartets.empty());

    auto buf = engine.compute(coulomb, quartets[0]);
    EXPECT_GT(buf.n_shell_quartets(), 0u);

    // Every quartet in the buffer should have a non-empty data span
    for (Size q = 0; q < buf.n_shell_quartets(); ++q) {
        auto data = buf.quartet_data(q);
        EXPECT_FALSE(data.empty());
    }
}

TEST(EriTensorApi, ComputeShellSetQuartetMatchesExplicit) {
    auto basis = make_h2_sto3g();
    Engine engine(basis);
    Operator coulomb = Operator::coulomb();

    const auto& quartets = basis.shell_set_quartets();
    ASSERT_FALSE(quartets.empty());

    for (const auto& ssq : quartets) {
        auto buf = engine.compute(coulomb, ssq);

        // Cross-check each sub-quartet against direct shell-level compute
        const auto& bra = ssq.bra_pair();
        const auto& ket = ssq.ket_pair();
        const auto& set_a = bra.shell_set_a();
        const auto& set_b = bra.shell_set_b();
        const auto& set_c = ket.shell_set_a();
        const auto& set_d = ket.shell_set_b();

        TwoElectronBuffer<0> ref_buffer;
        Size buf_idx = 0;

        for (Size ia = 0; ia < set_a.n_shells(); ++ia) {
            for (Size ib = 0; ib < set_b.n_shells(); ++ib) {
                for (Size ic = 0; ic < set_c.n_shells(); ++ic) {
                    for (Size id = 0; id < set_d.n_shells(); ++id) {
                        engine.compute(coulomb,
                                       set_a.shell(ia), set_b.shell(ib),
                                       set_c.shell(ic), set_d.shell(id),
                                       ref_buffer);

                        ASSERT_LT(buf_idx, buf.n_shell_quartets());
                        auto computed = buf.quartet_data(buf_idx);

                        const int na = set_a.shell(ia).n_functions();
                        const int nb = set_b.shell(ib).n_functions();
                        const int nc = set_c.shell(ic).n_functions();
                        const int nd = set_d.shell(id).n_functions();

                        Size flat = 0;
                        for (int a = 0; a < na; ++a) {
                            for (int b = 0; b < nb; ++b) {
                                for (int c = 0; c < nc; ++c) {
                                    for (int d = 0; d < nd; ++d) {
                                        EXPECT_NEAR(computed[flat],
                                                    ref_buffer(a, b, c, d),
                                                    ERI_TOL)
                                            << "Mismatch at quartet " << buf_idx
                                            << " [" << a << "," << b
                                            << "," << c << "," << d << "]";
                                        ++flat;
                                    }
                                }
                            }
                        }
                        ++buf_idx;
                    }
                }
            }
        }
    }
}

// =============================================================================
// compute(op, span<ShellSetQuartet>) → vector<IntegralBuffer>
// =============================================================================

TEST(EriTensorApi, ComputeSpanOfQuartetsMatchesIndividual) {
    auto basis = make_h2o_sto3g();
    Engine engine(basis);
    Operator coulomb = Operator::coulomb();

    const auto& quartets = basis.shell_set_quartets();

    auto all_bufs = engine.compute(coulomb, std::span<const ShellSetQuartet>(quartets));
    ASSERT_EQ(all_bufs.size(), quartets.size());

    for (Size i = 0; i < quartets.size(); ++i) {
        auto individual = engine.compute(coulomb, quartets[i]);
        ASSERT_EQ(all_bufs[i].n_shell_quartets(), individual.n_shell_quartets())
            << "Mismatch at quartet " << i;

        for (Size q = 0; q < individual.n_shell_quartets(); ++q) {
            auto d1 = all_bufs[i].quartet_data(q);
            auto d2 = individual.quartet_data(q);
            ASSERT_EQ(d1.size(), d2.size());
            for (Size k = 0; k < d1.size(); ++k) {
                EXPECT_NEAR(d1[k], d2[k], ERI_TOL);
            }
        }
    }
}

// =============================================================================
// compute(op, ShellSetPair) → IntegralBuffer (1e)
// =============================================================================

TEST(EriTensorApi, ComputeShellSetPairReturnsIntegralBuffer) {
    auto basis = make_h2_sto3g();
    Engine engine(basis);
    Operator overlap = Operator::overlap();

    const auto& pairs = basis.shell_set_pairs();
    ASSERT_FALSE(pairs.empty());

    auto buf = engine.compute(overlap, pairs[0]);
    // 1e IntegralBuffer uses n_shell_pairs()
    EXPECT_GT(buf.n_shell_pairs(), 0u);
}

TEST(EriTensorApi, ComputeShellSetPairMatchesMatrix) {
    auto basis = make_h2o_sto3g();
    Engine engine(basis);
    Operator overlap_op = Operator::overlap();

    // Full overlap matrix via existing API
    std::vector<Real> S_full(basis.n_basis_functions() * basis.n_basis_functions(), 0.0);
    engine.compute_overlap_matrix(S_full);

    // Per-pair IntegralBuffer
    const auto& pairs = basis.shell_set_pairs();
    for (const auto& pair : pairs) {
        auto buf = engine.compute(overlap_op, pair);

        // Each pair_data should match the corresponding element in the full matrix
        for (Size p = 0; p < buf.n_shell_pairs(); ++p) {
            const auto& meta = buf.pair_meta(p);
            auto data = buf.pair_data(p);

            Size idx = 0;
            for (int a = 0; a < meta.na; ++a) {
                const Size i = meta.fi + a;
                for (int b = 0; b < meta.nb; ++b) {
                    const Size j = meta.fj + b;
                    EXPECT_NEAR(data[idx],
                                S_full[i * basis.n_basis_functions() + j],
                                ERI_TOL)
                        << "Pair mismatch at (" << i << "," << j << ")";
                    ++idx;
                }
            }
        }
    }
}

// =============================================================================
// compute_eri_tensor()
// =============================================================================

TEST(EriTensorApi, ComputeEriTensorH2) {
    auto basis = make_h2_sto3g();
    Engine engine(basis);

    auto tensor = engine.compute_eri_tensor();

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(nbf, 2u);
    ASSERT_EQ(tensor.size(), 16u);  // 2^4 = 16

    // PySCF reference values for H2/STO-3G
    // (00|00)
    EXPECT_NEAR(tensor[0 * 8 + 0 * 4 + 0 * 2 + 0], 0.7746059439198978, ERI_TOL);
    // (00|11)
    EXPECT_NEAR(tensor[0 * 8 + 0 * 4 + 1 * 2 + 1], 0.5699948826767758, ERI_TOL);
    // (01|01)
    EXPECT_NEAR(tensor[0 * 8 + 1 * 4 + 0 * 2 + 1], 0.2975905517135221, ERI_TOL);
    // (11|11)
    EXPECT_NEAR(tensor[1 * 8 + 1 * 4 + 1 * 2 + 1], 0.7746059439198978, ERI_TOL);
}

TEST(EriTensorApi, ComputeEriTensorSymmetry) {
    auto basis = make_h2_sto3g();
    Engine engine(basis);

    auto tensor = engine.compute_eri_tensor();
    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(nbf, 2u);

    // Verify 8-fold permutation symmetry:
    // (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji)
    auto idx = [nbf](Size i, Size j, Size k, Size l) -> Size {
        return i * nbf * nbf * nbf + j * nbf * nbf + k * nbf + l;
    };

    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            for (Size k = 0; k < nbf; ++k) {
                for (Size l = 0; l < nbf; ++l) {
                    Real val = tensor[idx(i, j, k, l)];
                    EXPECT_NEAR(tensor[idx(j, i, k, l)], val, ERI_TOL);
                    EXPECT_NEAR(tensor[idx(i, j, l, k)], val, ERI_TOL);
                    EXPECT_NEAR(tensor[idx(j, i, l, k)], val, ERI_TOL);
                    EXPECT_NEAR(tensor[idx(k, l, i, j)], val, ERI_TOL);
                    EXPECT_NEAR(tensor[idx(l, k, i, j)], val, ERI_TOL);
                    EXPECT_NEAR(tensor[idx(k, l, j, i)], val, ERI_TOL);
                    EXPECT_NEAR(tensor[idx(l, k, j, i)], val, ERI_TOL);
                }
            }
        }
    }
}

TEST(EriTensorApi, ComputeEriTensorThrowsForOversizedAllocation) {
    auto basis = make_h_chain_sto3g(150);
    Engine engine(basis);

    EXPECT_THROW(
        (void)engine.compute_eri_tensor(),
        InvalidArgumentException);
}

// =============================================================================
// compute_eri_block()
// =============================================================================

TEST(EriTensorApi, ComputeEriBlockMatchesTensor) {
    auto basis = make_h2_sto3g();
    Engine engine(basis);
    Operator coulomb = Operator::coulomb();

    auto tensor = engine.compute_eri_tensor();
    const Size nbf = basis.n_basis_functions();

    const auto& quartets = basis.shell_set_quartets();
    ASSERT_FALSE(quartets.empty());

    for (const auto& ssq : quartets) {
        auto block = engine.compute_eri_block(coulomb, ssq);

        // Scatter block back to tensor indexing and compare
        const auto& set_a = ssq.bra_pair().shell_set_a();
        const auto& set_b = ssq.bra_pair().shell_set_b();
        const auto& set_c = ssq.ket_pair().shell_set_a();
        const auto& set_d = ssq.ket_pair().shell_set_b();

        // Count total functions per shell set
        Size nf_a = 0, nf_b = 0, nf_c = 0, nf_d = 0;
        for (Size s = 0; s < set_a.n_shells(); ++s) nf_a += set_a.shell(s).n_functions();
        for (Size s = 0; s < set_b.n_shells(); ++s) nf_b += set_b.shell(s).n_functions();
        for (Size s = 0; s < set_c.n_shells(); ++s) nf_c += set_c.shell(s).n_functions();
        for (Size s = 0; s < set_d.n_shells(); ++s) nf_d += set_d.shell(s).n_functions();

        ASSERT_EQ(block.size(), nf_a * nf_b * nf_c * nf_d);

        const Index base_a = set_a.shell(0).function_index();
        const Index base_b = set_b.shell(0).function_index();
        const Index base_c = set_c.shell(0).function_index();
        const Index base_d = set_d.shell(0).function_index();

        for (Size a = 0; a < nf_a; ++a) {
            for (Size b = 0; b < nf_b; ++b) {
                for (Size c = 0; c < nf_c; ++c) {
                    for (Size d = 0; d < nf_d; ++d) {
                        Real block_val = block[a * nf_b * nf_c * nf_d +
                                               b * nf_c * nf_d +
                                               c * nf_d + d];
                        Size gi = base_a + a;
                        Size gj = base_b + b;
                        Size gk = base_c + c;
                        Size gl = base_d + d;
                        Real tensor_val = tensor[gi * nbf * nbf * nbf +
                                                 gj * nbf * nbf +
                                                 gk * nbf + gl];
                        EXPECT_NEAR(block_val, tensor_val, ERI_TOL)
                            << "Block/tensor mismatch at local ("
                            << a << "," << b << "," << c << "," << d
                            << ") global (" << gi << "," << gj
                            << "," << gk << "," << gl << ")";
                    }
                }
            }
        }
    }
}

// =============================================================================
// Screening
// =============================================================================

TEST(EriTensorApi, ScreeningNoneReturnsAllNonEmpty) {
    auto basis = make_h2o_sto3g();
    Engine engine(basis);
    Operator coulomb = Operator::coulomb();

    const auto& quartets = basis.shell_set_quartets();
    auto results = engine.compute(coulomb, quartets, BackendHint::Auto,
                                  screening::ScreeningOptions::none());

    ASSERT_EQ(results.size(), quartets.size());

    // With no screening, every IntegralBuffer should be non-empty
    for (Size i = 0; i < results.size(); ++i) {
        EXPECT_GT(results[i].n_shell_quartets(), 0u)
            << "Expected non-empty buffer at index " << i;
    }
}

TEST(EriTensorApi, ScreeningStatisticsConsistency) {
    auto basis = make_h2o_sto3g();
    Engine engine(basis);
    Operator coulomb = Operator::coulomb();

    const auto& quartets = basis.shell_set_quartets();

    // No screening
    auto results_none = engine.compute(coulomb, quartets, BackendHint::Auto,
                                       screening::ScreeningOptions::none());
    auto stats_none = Engine::compute_screening_statistics(results_none);

    EXPECT_EQ(stats_none.total_quartets, quartets.size());
    EXPECT_EQ(stats_none.computed_quartets, quartets.size());
    EXPECT_EQ(stats_none.skipped_quartets, 0u);

    // With screening (loose)
    auto results_screened = engine.compute(coulomb, quartets, BackendHint::Auto,
                                           screening::ScreeningOptions::loose());
    auto stats_screened = Engine::compute_screening_statistics(results_screened);

    EXPECT_EQ(stats_screened.total_quartets, quartets.size());
    EXPECT_EQ(stats_screened.computed_quartets + stats_screened.skipped_quartets,
              stats_screened.total_quartets);
    // Skipped should be >= 0 (may be 0 for small basis)
    EXPECT_GE(stats_screened.skipped_quartets, 0u);
}

// =============================================================================
// Parallel consistency
// =============================================================================

TEST(EriTensorApi, ParallelMatchesSerial) {
    auto basis = make_h2o_sto3g();
    Engine engine(basis);
    Operator coulomb = Operator::coulomb();

    auto serial = engine.compute_all_2e(coulomb);
    auto parallel = engine.compute_all_2e_parallel(coulomb, 2);

    ASSERT_EQ(serial.size(), parallel.size());

    for (Size i = 0; i < serial.size(); ++i) {
        ASSERT_EQ(serial[i].n_shell_quartets(), parallel[i].n_shell_quartets())
            << "Quartet count mismatch at index " << i;

        for (Size q = 0; q < serial[i].n_shell_quartets(); ++q) {
            auto d_ser = serial[i].quartet_data(q);
            auto d_par = parallel[i].quartet_data(q);
            ASSERT_EQ(d_ser.size(), d_par.size());
            for (Size k = 0; k < d_ser.size(); ++k) {
                EXPECT_NEAR(d_ser[k], d_par[k], PARALLEL_TOL)
                    << "Parallel mismatch at quartet " << i
                    << " sub-quartet " << q << " element " << k;
            }
        }
    }
}

// =============================================================================
// Backend hint
// =============================================================================

TEST(EriTensorApi, ComputeWithBackendHintForceCPU) {
    auto basis = make_h2o_sto3g();
    Engine engine(basis);
    Operator coulomb = Operator::coulomb();

    const auto& quartets = basis.shell_set_quartets();
    ASSERT_FALSE(quartets.empty());

    auto buf = engine.compute(coulomb, quartets[0], BackendHint::ForceCPU);
    EXPECT_GT(buf.n_shell_quartets(), 0u);
}

// =============================================================================
// IntegralBuffer metadata correctness
// =============================================================================

TEST(EriTensorApi, IntegralBufferMetadataCorrect) {
    auto basis = make_h2_sto3g();
    Engine engine(basis);
    Operator coulomb = Operator::coulomb();

    const auto& quartets = basis.shell_set_quartets();
    ASSERT_FALSE(quartets.empty());

    for (const auto& ssq : quartets) {
        auto buf = engine.compute(coulomb, ssq);

        const auto& set_a = ssq.bra_pair().shell_set_a();
        const auto& set_b = ssq.bra_pair().shell_set_b();
        const auto& set_c = ssq.ket_pair().shell_set_a();
        const auto& set_d = ssq.ket_pair().shell_set_b();

        Size expected_quartets = set_a.n_shells() * set_b.n_shells() *
                                 set_c.n_shells() * set_d.n_shells();
        ASSERT_EQ(buf.n_shell_quartets(), expected_quartets);

        // Verify each sub-quartet has valid metadata
        Size buf_idx = 0;
        for (Size ia = 0; ia < set_a.n_shells(); ++ia) {
            for (Size ib = 0; ib < set_b.n_shells(); ++ib) {
                for (Size ic = 0; ic < set_c.n_shells(); ++ic) {
                    for (Size id = 0; id < set_d.n_shells(); ++id) {
                        const auto& meta = buf.quartet_meta(buf_idx);

                        EXPECT_EQ(meta.fi, set_a.shell(ia).function_index());
                        EXPECT_EQ(meta.fj, set_b.shell(ib).function_index());
                        EXPECT_EQ(meta.fk, set_c.shell(ic).function_index());
                        EXPECT_EQ(meta.fl, set_d.shell(id).function_index());

                        EXPECT_EQ(meta.na, set_a.shell(ia).n_functions());
                        EXPECT_EQ(meta.nb, set_b.shell(ib).n_functions());
                        EXPECT_EQ(meta.nc, set_c.shell(ic).n_functions());
                        EXPECT_EQ(meta.nd, set_d.shell(id).n_functions());

                        auto data = buf.quartet_data(buf_idx);
                        EXPECT_EQ(static_cast<int>(data.size()),
                                  meta.na * meta.nb * meta.nc * meta.nd);

                        ++buf_idx;
                    }
                }
            }
        }
    }
}

// =============================================================================
// H2O/STO-3G ERI tensor with more quartets
// =============================================================================

TEST(EriTensorApi, ComputeEriTensorH2O) {
    auto basis = make_h2o_sto3g();
    Engine engine(basis);

    auto tensor = engine.compute_eri_tensor();

    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(nbf, 7u);
    ASSERT_EQ(tensor.size(), 7u * 7u * 7u * 7u);

    // Verify symmetry (ij|kl) = (kl|ij) for a sampling of indices
    auto idx = [nbf](Size i, Size j, Size k, Size l) -> Size {
        return i * nbf * nbf * nbf + j * nbf * nbf + k * nbf + l;
    };

    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j <= i; ++j) {
            for (Size k = 0; k < nbf; ++k) {
                for (Size l = 0; l <= k; ++l) {
                    EXPECT_NEAR(tensor[idx(i, j, k, l)],
                                tensor[idx(k, l, i, j)], ERI_TOL)
                        << "Symmetry (ij|kl)=(kl|ij) failed at ("
                        << i << "," << j << "," << k << "," << l << ")";
                }
            }
        }
    }
}
