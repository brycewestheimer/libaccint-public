// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file cpu_engine.cpp
/// @brief CPU backend implementation for integral computation

#include <libaccint/engine/cpu_engine.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/core/derivative_utils.hpp>
#include <libaccint/kernels/overlap_kernel.hpp>
#include <libaccint/kernels/kinetic_kernel.hpp>
#include <libaccint/kernels/nuclear_kernel.hpp>
#include <libaccint/kernels/eri_kernel.hpp>
#include <libaccint/kernels/generated_kernel_registry.hpp>
#include <libaccint/kernels/contraction_range.hpp>
#include <libaccint/engine/thread_config.hpp>
#include <libaccint/utils/error_handling.hpp>
#include <libaccint/utils/logging.hpp>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace {

/// @brief Classify contraction range for 2-center integrals (1e)
static libaccint::kernels::ContractionRange classify_cpu_contraction_range(
    int n_prim_a, int n_prim_b) {
    const int max_k = std::max(n_prim_a, n_prim_b);
    if (max_k <= 3) return libaccint::kernels::ContractionRange::SmallK;
    if (max_k <= 6) return libaccint::kernels::ContractionRange::MediumK;
    return libaccint::kernels::ContractionRange::LargeK;
}

/// @brief Classify contraction range for 4-center integrals (2e)
static libaccint::kernels::ContractionRange classify_cpu_contraction_range_4(
    int n_prim_a, int n_prim_b, int n_prim_c, int n_prim_d) {
    const int max_k = std::max({n_prim_a, n_prim_b, n_prim_c, n_prim_d});
    if (max_k <= 3) return libaccint::kernels::ContractionRange::SmallK;
    if (max_k <= 6) return libaccint::kernels::ContractionRange::MediumK;
    return libaccint::kernels::ContractionRange::LargeK;
}

#ifdef LIBACCINT_TRACE_DISPATCH
[[nodiscard]] static std::uint64_t trace_key_2c(int la, int lb) noexcept {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(la)) << 32u) |
           static_cast<std::uint64_t>(static_cast<std::uint32_t>(lb));
}

[[nodiscard]] static std::uint64_t trace_key_4c(
    int la, int lb, int lc, int ld) noexcept {
    return (static_cast<std::uint64_t>(static_cast<std::uint16_t>(la)) << 48u) |
           (static_cast<std::uint64_t>(static_cast<std::uint16_t>(lb)) << 32u) |
           (static_cast<std::uint64_t>(static_cast<std::uint16_t>(lc)) << 16u) |
            static_cast<std::uint64_t>(static_cast<std::uint16_t>(ld));
}

static bool trace_once(std::unordered_set<std::uint64_t>& seen,
                       std::mutex& seen_mutex,
                       std::uint64_t key) {
    std::lock_guard<std::mutex> lock(seen_mutex);
    return seen.insert(key).second;
}
#endif

}  // anonymous namespace

namespace libaccint::engine {

// =============================================================================
// compute_1e_shell_pair: Single shell pair dispatch
// =============================================================================

void CpuEngine::compute_1e_shell_pair(const Operator& op,
                                       const Shell& shell_a,
                                       const Shell& shell_b,
                                       OneElectronBuffer<0>& buffer) {
    const int la = shell_a.angular_momentum();
    const int lb = shell_b.angular_momentum();
    const auto k_range = classify_cpu_contraction_range(
        static_cast<int>(shell_a.n_primitives()),
        static_cast<int>(shell_b.n_primitives()));

    switch (op.kind()) {
        case OperatorKind::Overlap: {
            auto gen_fn = kernels::cpu::generated::get_generated_overlap(la, lb, k_range);
            if (gen_fn) {
                const int na = shell_a.n_functions();
                const int nb = shell_b.n_functions();
                LIBACCINT_ASSERT(na > 0 && nb > 0, "Shell function count must be positive");
#ifdef LIBACCINT_TRACE_DISPATCH
                LIBACCINT_LOG_DEBUG("CpuEngine",
                    "dispatch overlap (" + std::to_string(la) + "," + std::to_string(lb) +
                    ") K=" + std::string(kernels::to_string(k_range)) + " → generated");
#endif
                buffer.resize(na, nb);
                buffer.clear();
                LIBACCINT_ASSERT(!buffer.data().empty(),
                                 "Buffer data must not be empty after resize");
                LIBACCINT_ASSERT(buffer.size() >= static_cast<Size>(na * nb),
                                 "Buffer too small for overlap kernel output");
                gen_fn(shell_a.exponents(), shell_a.coefficients(),
                       shell_a.center().x, shell_a.center().y, shell_a.center().z,
                       static_cast<int>(shell_a.n_primitives()),
                       shell_b.exponents(), shell_b.coefficients(),
                       shell_b.center().x, shell_b.center().y, shell_b.center().z,
                       static_cast<int>(shell_b.n_primitives()),
                       buffer.data());
            } else {
#ifdef LIBACCINT_TRACE_DISPATCH
                static std::unordered_set<std::uint64_t> traced_overlap;
                static std::mutex traced_overlap_mutex;
                if (trace_once(traced_overlap, traced_overlap_mutex,
                               trace_key_2c(la, lb))) {
                    LIBACCINT_LOG_DEBUG("CpuEngine",
                        "No generated overlap kernel for (" + std::to_string(la) + "," +
                        std::to_string(lb) + ") K=" + std::string(kernels::to_string(k_range)) +
                        " — using handwritten fallback");
                }
#endif
                kernels::compute_overlap(shell_a, shell_b, buffer);
            }
            break;
        }

        case OperatorKind::Kinetic: {
            auto gen_fn = kernels::cpu::generated::get_generated_kinetic(la, lb, k_range);
            if (gen_fn) {
                const int na = shell_a.n_functions();
                const int nb = shell_b.n_functions();
                LIBACCINT_ASSERT(na > 0 && nb > 0, "Shell function count must be positive");
#ifdef LIBACCINT_TRACE_DISPATCH
                LIBACCINT_LOG_DEBUG("CpuEngine",
                    "dispatch kinetic (" + std::to_string(la) + "," + std::to_string(lb) +
                    ") K=" + std::string(kernels::to_string(k_range)) + " → generated");
#endif
                buffer.resize(na, nb);
                buffer.clear();
                LIBACCINT_ASSERT(!buffer.data().empty(),
                                 "Buffer data must not be empty after resize");
                LIBACCINT_ASSERT(buffer.size() >= static_cast<Size>(na * nb),
                                 "Buffer too small for kinetic kernel output");
                gen_fn(shell_a.exponents(), shell_a.coefficients(),
                       shell_a.center().x, shell_a.center().y, shell_a.center().z,
                       static_cast<int>(shell_a.n_primitives()),
                       shell_b.exponents(), shell_b.coefficients(),
                       shell_b.center().x, shell_b.center().y, shell_b.center().z,
                       static_cast<int>(shell_b.n_primitives()),
                       buffer.data());
            } else {
#ifdef LIBACCINT_TRACE_DISPATCH
                static std::unordered_set<std::uint64_t> traced_kinetic;
                static std::mutex traced_kinetic_mutex;
                if (trace_once(traced_kinetic, traced_kinetic_mutex,
                               trace_key_2c(la, lb))) {
                    LIBACCINT_LOG_DEBUG("CpuEngine",
                        "No generated kinetic kernel for (" + std::to_string(la) + "," +
                        std::to_string(lb) + ") K=" + std::string(kernels::to_string(k_range)) +
                        " — using handwritten fallback");
                }
#endif
                kernels::compute_kinetic(shell_a, shell_b, buffer);
            }
            break;
        }

        case OperatorKind::Nuclear:
        case OperatorKind::PointCharge: {
            const auto& charges = op.params_as<PointChargeParams>();
            auto gen_fn = kernels::cpu::generated::get_generated_nuclear(la, lb, k_range);
            if (gen_fn) {
                const int na = shell_a.n_functions();
                const int nb = shell_b.n_functions();
                LIBACCINT_ASSERT(na > 0 && nb > 0, "Shell function count must be positive");
#ifdef LIBACCINT_TRACE_DISPATCH
                LIBACCINT_LOG_DEBUG("CpuEngine",
                    "dispatch nuclear (" + std::to_string(la) + "," + std::to_string(lb) +
                    ") K=" + std::string(kernels::to_string(k_range)) + " → generated");
#endif
                buffer.resize(na, nb);
                buffer.clear();
                LIBACCINT_ASSERT(!buffer.data().empty(),
                                 "Buffer data must not be empty after resize");
                LIBACCINT_ASSERT(buffer.size() >= static_cast<Size>(na * nb),
                                 "Buffer too small for nuclear kernel output");

                // Build interleaved charge positions [x0,y0,z0, x1,y1,z1, ...]
                const auto n_charges = static_cast<int>(charges.n_centers());
                std::vector<double> charge_positions(3 * n_charges);
                for (int c = 0; c < n_charges; ++c) {
                    charge_positions[3 * c + 0] = charges.x[static_cast<Size>(c)];
                    charge_positions[3 * c + 1] = charges.y[static_cast<Size>(c)];
                    charge_positions[3 * c + 2] = charges.z[static_cast<Size>(c)];
                }
                std::span<const double> pos_span(charge_positions);
                std::span<const double> val_span(charges.charge);

                gen_fn(shell_a.exponents(), shell_a.coefficients(),
                       shell_a.center().x, shell_a.center().y, shell_a.center().z,
                       static_cast<int>(shell_a.n_primitives()),
                       shell_b.exponents(), shell_b.coefficients(),
                       shell_b.center().x, shell_b.center().y, shell_b.center().z,
                       static_cast<int>(shell_b.n_primitives()),
                       pos_span, val_span, n_charges,
                       buffer.data());
            } else {
#ifdef LIBACCINT_TRACE_DISPATCH
                static std::unordered_set<std::uint64_t> traced_nuclear;
                static std::mutex traced_nuclear_mutex;
                if (trace_once(traced_nuclear, traced_nuclear_mutex,
                               trace_key_2c(la, lb))) {
                    LIBACCINT_LOG_DEBUG("CpuEngine",
                        "No generated nuclear kernel for (" + std::to_string(la) + "," +
                        std::to_string(lb) + ") K=" + std::string(kernels::to_string(k_range)) +
                        " — using handwritten fallback");
                }
#endif
                kernels::compute_nuclear(shell_a, shell_b, charges, buffer);
            }
            break;
        }

        default:
            throw InvalidArgumentException(
                "Unsupported operator kind for one-electron compute: " +
                std::string(to_string(op.kind())));
    }
}

// =============================================================================
// compute_1e_impl: Full basis set one-electron computation
// =============================================================================

void CpuEngine::compute_1e_impl(const OneElectronOperator& op,
                                 std::vector<Real>& result) {
    const Size nbf = basis_->n_basis_functions();

    // Resize and zero the result matrix
    result.assign(nbf * nbf, Real{0.0});

    // Handle empty basis set
    if (basis_->n_shells() == 0) {
        return;
    }

    // Use ShellSetPair worklist as the primary iteration driver
    const auto& pairs = basis_->shell_set_pairs();

    // Iterate over each contribution in the composed operator
    for (const auto& contribution : op.contributions()) {
        for (const auto& ssp : pairs) {
            const auto& set_a = ssp.shell_set_a();
            const auto& set_b = ssp.shell_set_b();
            const bool is_self_pair = (&set_a == &set_b);

            const Size na_shells = set_a.n_shells();
            const Size nb_shells = set_b.n_shells();

            for (Size i = 0; i < na_shells; ++i) {
                const auto& shell_a = set_a.shell(i);
                const Index fi = shell_a.function_index();
                const int na = shell_a.n_functions();

                // For self-pairs, only iterate upper triangle (j >= i)
                const Size j_start = is_self_pair ? i : 0;

                for (Size j = j_start; j < nb_shells; ++j) {
                    const auto& shell_b = set_b.shell(j);
                    const Index fj = shell_b.function_index();
                    const int nb = shell_b.n_functions();

                    // Compute integrals for this shell pair into the internal buffer
                    compute_1e_shell_pair(contribution.op, shell_a, shell_b, buffer_1e_);

                    // Accumulate scaled buffer values into the result matrix
                    const Real scale = contribution.scale;
                    const bool fill_sym = !is_self_pair || (i != j);

                    for (int a = 0; a < na; ++a) {
                        for (int b = 0; b < nb; ++b) {
                            const Real val = scale * buffer_1e_(a, b);
                            result[static_cast<Size>(fi + a) * nbf +
                                   static_cast<Size>(fj + b)] += val;
                            // Fill symmetric counterpart for off-diagonal or cross-set pairs
                            if (fill_sym) {
                                result[static_cast<Size>(fj + b) * nbf +
                                       static_cast<Size>(fi + a)] += val;
                            }
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// compute_2e_shell_quartet: Single shell quartet dispatch
// =============================================================================

void CpuEngine::compute_2e_shell_quartet(const Operator& op,
                                          const Shell& shell_a,
                                          const Shell& shell_b,
                                          const Shell& shell_c,
                                          const Shell& shell_d,
                                          TwoElectronBuffer<0>& buffer) {
    switch (op.kind()) {
        case OperatorKind::Coulomb: {
            const int la = shell_a.angular_momentum();
            const int lb = shell_b.angular_momentum();
            const int lc = shell_c.angular_momentum();
            const int ld = shell_d.angular_momentum();
            const auto k_range = classify_cpu_contraction_range_4(
                static_cast<int>(shell_a.n_primitives()),
                static_cast<int>(shell_b.n_primitives()),
                static_cast<int>(shell_c.n_primitives()),
                static_cast<int>(shell_d.n_primitives()));

            auto gen_fn = kernels::cpu::generated::get_generated_eri(
                la, lb, lc, ld, k_range);
            if (gen_fn) {
                const int na = shell_a.n_functions();
                const int nb = shell_b.n_functions();
                const int nc = shell_c.n_functions();
                const int nd = shell_d.n_functions();
                LIBACCINT_ASSERT(na > 0 && nb > 0 && nc > 0 && nd > 0,
                                 "Shell function count must be positive");
#ifdef LIBACCINT_TRACE_DISPATCH
                LIBACCINT_LOG_DEBUG("CpuEngine",
                    "dispatch ERI (" + std::to_string(la) + "," + std::to_string(lb) +
                    "|" + std::to_string(lc) + "," + std::to_string(ld) +
                    ") K=" + std::string(kernels::to_string(k_range)) + " → generated");
#endif
                buffer.resize(na, nb, nc, nd);
                buffer.clear();
                LIBACCINT_ASSERT(!buffer.data().empty(),
                                 "Buffer data must not be empty after resize");
                LIBACCINT_ASSERT(buffer.size() >= static_cast<Size>(na * nb * nc * nd),
                                 "Buffer too small for ERI kernel output");
                gen_fn(shell_a.exponents(), shell_a.coefficients(),
                       shell_a.center().x, shell_a.center().y, shell_a.center().z,
                       static_cast<int>(shell_a.n_primitives()),
                       shell_b.exponents(), shell_b.coefficients(),
                       shell_b.center().x, shell_b.center().y, shell_b.center().z,
                       static_cast<int>(shell_b.n_primitives()),
                       shell_c.exponents(), shell_c.coefficients(),
                       shell_c.center().x, shell_c.center().y, shell_c.center().z,
                       static_cast<int>(shell_c.n_primitives()),
                       shell_d.exponents(), shell_d.coefficients(),
                       shell_d.center().x, shell_d.center().y, shell_d.center().z,
                       static_cast<int>(shell_d.n_primitives()),
                       buffer.data());
            } else {
#ifdef LIBACCINT_TRACE_DISPATCH
                static std::unordered_set<std::uint64_t> traced_eri;
                static std::mutex traced_eri_mutex;
                if (trace_once(traced_eri, traced_eri_mutex,
                               trace_key_4c(la, lb, lc, ld))) {
                    LIBACCINT_LOG_DEBUG("CpuEngine",
                        "No generated ERI kernel for (" + std::to_string(la) + "," +
                        std::to_string(lb) + "|" + std::to_string(lc) + "," +
                        std::to_string(ld) + ") K=" + std::string(kernels::to_string(k_range)) +
                        " — using handwritten fallback");
                }
#endif
                kernels::compute_eri(shell_a, shell_b, shell_c, shell_d, buffer);
            }
            break;
        }

        default:
            throw InvalidArgumentException(
                "Unsupported operator kind for two-electron compute: " +
                std::string(to_string(op.kind())));
    }
}

// =============================================================================
// ShellSetPair-Based Computation
// =============================================================================

void CpuEngine::compute_shell_set_pair(const Operator& op,
                                        const ShellSetPair& pair,
                                        std::vector<Real>& result) {
    const Size nbf = basis_->n_basis_functions();

    // Get ShellSets from the pair
    const auto& set_a = pair.shell_set_a();
    const auto& set_b = pair.shell_set_b();
    const bool fill_symmetric_partner = (&set_a != &set_b);

    const Size n_shells_a = set_a.n_shells();
    const Size n_shells_b = set_b.n_shells();
    const Size total_pairs = n_shells_a * n_shells_b;

    // Handle empty case
    if (total_pairs == 0) {
        return;
    }

#if defined(_OPENMP)
    // OpenMP parallel execution: flatten 2D loop into 1D for better scheduling
    // Each (i, j) pair maps to a unique output region, so no races
    #pragma omp parallel
    {
        // Per-thread buffer to avoid sharing
        OneElectronBuffer<0> local_buffer;

        #pragma omp for schedule(static)
        for (Size idx = 0; idx < total_pairs; ++idx) {
            // Decode (i, j) from linear index
            const Size i = idx / n_shells_b;
            const Size j = idx % n_shells_b;

            const auto& shell_a = set_a.shell(i);
            const auto& shell_b = set_b.shell(j);

            const Index fi = shell_a.function_index();
            const Index fj = shell_b.function_index();
            const int na = shell_a.n_functions();
            const int nb = shell_b.n_functions();

            // Compute integrals for this shell pair into thread-local buffer
            compute_1e_shell_pair(op, shell_a, shell_b, local_buffer);

            // Accumulate into result matrix
            // No race: each (i, j) pair writes to unique (fi, fj) output block.
            // For cross-set pairs, also mirror to the transpose block so the
            // AO matrix semantics match the GPU path and full-basis API.
            for (int a = 0; a < na; ++a) {
                for (int b = 0; b < nb; ++b) {
                    const Real val = local_buffer(a, b);
                    result[static_cast<Size>(fi + a) * nbf +
                           static_cast<Size>(fj + b)] += val;
                    if (fill_symmetric_partner) {
                        result[static_cast<Size>(fj + b) * nbf +
                               static_cast<Size>(fi + a)] += val;
                    }
                }
            }
        }
    }
#else
    // Sequential fallback when OpenMP not available
    for (Size i = 0; i < n_shells_a; ++i) {
        const auto& shell_a = set_a.shell(i);
        const Index fi = shell_a.function_index();
        const int na = shell_a.n_functions();

        for (Size j = 0; j < n_shells_b; ++j) {
            const auto& shell_b = set_b.shell(j);
            const Index fj = shell_b.function_index();
            const int nb = shell_b.n_functions();

            // Compute integrals for this shell pair
            compute_1e_shell_pair(op, shell_a, shell_b, buffer_1e_);

            // Accumulate into result matrix
            for (int a = 0; a < na; ++a) {
                for (int b = 0; b < nb; ++b) {
                    const Real val = buffer_1e_(a, b);
                    result[static_cast<Size>(fi + a) * nbf +
                           static_cast<Size>(fj + b)] += val;
                    if (fill_symmetric_partner) {
                        result[static_cast<Size>(fj + b) * nbf +
                               static_cast<Size>(fi + a)] += val;
                    }
                }
            }
        }
    }
#endif
}

// =============================================================================
// compute_1e_parallel_impl: OpenMP-parallel one-electron computation
// =============================================================================

void CpuEngine::compute_1e_parallel_impl(const OneElectronOperator& op,
                                          std::vector<Real>& result,
                                          int n_threads) {
    const Size nbf = basis_->n_basis_functions();

    // Resize and zero the result matrix
    result.assign(nbf * nbf, Real{0.0});

    if (basis_->n_shells() == 0) {
        return;
    }

    const auto& pairs = basis_->shell_set_pairs();

    // Build a flat worklist of all shell pairs across all contributions and SSPs
    struct ShellPairTask {
        Size contribution_idx;
        Size ssp_idx;
        Size i;
        Size j;
        bool fill_sym;
    };

    std::vector<ShellPairTask> tasks;

    for (Size ci = 0; ci < op.contributions().size(); ++ci) {
        for (Size pi = 0; pi < pairs.size(); ++pi) {
            const auto& ssp = pairs[pi];
            const auto& set_a = ssp.shell_set_a();
            const auto& set_b = ssp.shell_set_b();
            const bool is_self_pair = (&set_a == &set_b);
            const Size na_shells = set_a.n_shells();
            const Size nb_shells = set_b.n_shells();

            for (Size i = 0; i < na_shells; ++i) {
                const Size j_start = is_self_pair ? i : 0;
                for (Size j = j_start; j < nb_shells; ++j) {
                    bool fill_sym = !is_self_pair || (i != j);
                    tasks.push_back({ci, pi, i, j, fill_sym});
                }
            }
        }
    }

    const Size n_tasks = tasks.size();
    if (n_tasks == 0) {
        return;
    }

#if defined(_OPENMP)
    const int actual_threads = ThreadConfig::resolve(n_threads);

    // Pre-allocate thread-local result matrices
    std::vector<std::vector<Real>> local_results(
        static_cast<std::size_t>(actual_threads),
        std::vector<Real>(nbf * nbf, Real{0.0}));

    #pragma omp parallel num_threads(actual_threads)
    {
        const int tid = omp_get_thread_num();
        auto& local_result = local_results[static_cast<std::size_t>(tid)];
        OneElectronBuffer<0> local_buffer;

        #pragma omp for schedule(dynamic, 4)
        for (Size t = 0; t < n_tasks; ++t) {
            const auto& task = tasks[t];
            const auto& contribution = op.contributions()[task.contribution_idx];
            const auto& ssp = pairs[task.ssp_idx];
            const auto& set_a = ssp.shell_set_a();
            const auto& set_b = ssp.shell_set_b();

            const auto& shell_a = set_a.shell(task.i);
            const auto& shell_b = set_b.shell(task.j);

            const Index fi = shell_a.function_index();
            const Index fj = shell_b.function_index();
            const int na = shell_a.n_functions();
            const int nb = shell_b.n_functions();

            // Compute integrals into thread-local buffer
            compute_1e_shell_pair(
                contribution.op, shell_a, shell_b, local_buffer);

            const Real scale = contribution.scale;

            for (int a = 0; a < na; ++a) {
                for (int b = 0; b < nb; ++b) {
                    const Real val = scale * local_buffer(a, b);
                    local_result[static_cast<Size>(fi + a) * nbf +
                                 static_cast<Size>(fj + b)] += val;
                    if (task.fill_sym) {
                        local_result[static_cast<Size>(fj + b) * nbf +
                                     static_cast<Size>(fi + a)] += val;
                    }
                }
            }
        }
    }

    // Reduce thread-local results into the final output
    for (int t = 0; t < actual_threads; ++t) {
        const auto& local = local_results[static_cast<std::size_t>(t)];
        for (Size i = 0; i < nbf * nbf; ++i) {
            result[i] += local[i];
        }
    }
#else
    // Fallback to sequential when OpenMP is not available
    (void)n_threads;
    compute_1e_impl(op, result);
#endif
}

}  // namespace libaccint::engine
