// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bench_stream_overlap.cu
/// @brief Self-profiling benchmark that measures stream overlap using CUDA events
///
/// Since nsys CUPTI tracing is unavailable on WSL2, this benchmark instruments
/// the pipeline directly with CUDA events to measure:
///   1. Per-stream kernel execution timelines
///   2. Overlap between concurrent kernel executions
///   3. Gap analysis between sequential vs pipelined dispatch

#include <libaccint/config.hpp>

#if LIBACCINT_USE_CUDA

#include <libaccint/engine/cuda_engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/basis/device_data.hpp>
#include <libaccint/kernels/eri_kernel_cuda.hpp>
#include <libaccint/memory/device_memory.hpp>
#include <libaccint/memory/stream_management.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/data/basis_parser.hpp>

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace libaccint;
using namespace libaccint::kernels::cuda;

namespace libaccint::device::math {
    double* boys_device_init(cudaStream_t stream);
    void boys_device_cleanup();
    double* boys_device_get_coeffs();
    bool boys_device_is_initialized();
}

// ============================================================================
// Kernel event recording
// ============================================================================

struct KernelRecord {
    int stream_id;
    int quartet_idx;
    float start_ms;     // Relative to baseline event
    float end_ms;
    float duration_ms;
    std::string label;
};

static std::vector<data::Atom> make_h2o() {
    return {
        {8, {0.0, 0.0, 0.0}},
        {1, {1.430429, 0.0, 1.107157}},
        {1, {-1.430429, 0.0, 1.107157}},
    };
}

// ============================================================================
// Sequential profiling: one stream, one SSQ at a time
// ============================================================================

static std::vector<KernelRecord> profile_sequential(
    const BasisSet& basis,
    double* d_boys_coeffs,
    cudaStream_t stream)
{
    const auto& quartets = basis.shell_set_quartets();
    std::vector<KernelRecord> records;

    // Baseline event
    cudaEvent_t baseline;
    cudaEventCreate(&baseline);
    cudaEventRecord(baseline, stream);
    cudaStreamSynchronize(stream);

    int q_idx = 0;
    for (const auto& q : quartets) {
        const auto& set_a = q.bra_pair().shell_set_a();
        const auto& set_b = q.bra_pair().shell_set_b();
        const auto& set_c = q.ket_pair().shell_set_a();
        const auto& set_d = q.ket_pair().shell_set_b();

        if (set_a.n_shells() == 0 || set_b.n_shells() == 0 ||
            set_c.n_shells() == 0 || set_d.n_shells() == 0) continue;

        auto data_a = basis::upload_shell_set(set_a, stream);
        auto data_b = basis::upload_shell_set(set_b, stream);
        auto data_c = basis::upload_shell_set(set_c, stream);
        auto data_d = basis::upload_shell_set(set_d, stream);

        basis::ShellSetQuartetDeviceData qdev;
        qdev.a = data_a; qdev.b = data_b; qdev.c = data_c; qdev.d = data_d;

        size_t out_size = eri_output_size(qdev);
        memory::DeviceBuffer<double> d_out(out_size);

        cudaEvent_t start_ev, end_ev;
        cudaEventCreate(&start_ev);
        cudaEventCreate(&end_ev);

        cudaEventRecord(start_ev, stream);
        dispatch_eri_kernel(qdev, d_boys_coeffs, d_out.data(), stream);
        cudaEventRecord(end_ev, stream);
        cudaStreamSynchronize(stream);

        float start_ms = 0, end_ms = 0;
        cudaEventElapsedTime(&start_ms, baseline, start_ev);
        cudaEventElapsedTime(&end_ms, baseline, end_ev);

        char lbl[64];
        std::snprintf(lbl, sizeof(lbl), "(%d%d|%d%d)",
                      set_a.angular_momentum(), set_b.angular_momentum(),
                      set_c.angular_momentum(), set_d.angular_momentum());

        records.push_back({0, q_idx, start_ms, end_ms, end_ms - start_ms, lbl});

        cudaEventDestroy(start_ev);
        cudaEventDestroy(end_ev);

        basis::free_shell_set_device_data(data_a);
        basis::free_shell_set_device_data(data_b);
        basis::free_shell_set_device_data(data_c);
        basis::free_shell_set_device_data(data_d);

        q_idx++;
    }

    cudaEventDestroy(baseline);
    return records;
}

// ============================================================================
// Multi-stream profiling: N streams, round-robin dispatch
// ============================================================================

static std::vector<KernelRecord> profile_multistream(
    const BasisSet& basis,
    double* d_boys_coeffs,
    int n_streams)
{
    const auto& quartets = basis.shell_set_quartets();
    std::vector<KernelRecord> records;

    // Create streams
    std::vector<cudaStream_t> streams(n_streams);
    for (auto& s : streams) cudaStreamCreate(&s);

    // Baseline event on stream 0
    cudaEvent_t baseline;
    cudaEventCreate(&baseline);
    cudaEventRecord(baseline, streams[0]);
    cudaDeviceSynchronize();

    // Per-quartet events
    struct QuartetEvents {
        cudaEvent_t start, end;
        int stream_id;
        int quartet_idx;
        std::string label;
    };
    std::vector<QuartetEvents> all_events;

    int q_idx = 0;
    int next_stream = 0;

    for (const auto& q : quartets) {
        const auto& set_a = q.bra_pair().shell_set_a();
        const auto& set_b = q.bra_pair().shell_set_b();
        const auto& set_c = q.ket_pair().shell_set_a();
        const auto& set_d = q.ket_pair().shell_set_b();

        if (set_a.n_shells() == 0 || set_b.n_shells() == 0 ||
            set_c.n_shells() == 0 || set_d.n_shells() == 0) continue;

        cudaStream_t s = streams[next_stream];

        auto data_a = basis::upload_shell_set(set_a, s);
        auto data_b = basis::upload_shell_set(set_b, s);
        auto data_c = basis::upload_shell_set(set_c, s);
        auto data_d = basis::upload_shell_set(set_d, s);

        basis::ShellSetQuartetDeviceData qdev;
        qdev.a = data_a; qdev.b = data_b; qdev.c = data_c; qdev.d = data_d;

        size_t out_size = eri_output_size(qdev);
        memory::DeviceBuffer<double> d_out(out_size);

        QuartetEvents qe;
        cudaEventCreate(&qe.start);
        cudaEventCreate(&qe.end);
        qe.stream_id = next_stream;
        qe.quartet_idx = q_idx;

        char lbl[64];
        std::snprintf(lbl, sizeof(lbl), "(%d%d|%d%d)",
                      set_a.angular_momentum(), set_b.angular_momentum(),
                      set_c.angular_momentum(), set_d.angular_momentum());
        qe.label = lbl;

        cudaEventRecord(qe.start, s);
        dispatch_eri_kernel(qdev, d_boys_coeffs, d_out.data(), s);
        cudaEventRecord(qe.end, s);

        all_events.push_back(qe);

        basis::free_shell_set_device_data(data_a);
        basis::free_shell_set_device_data(data_b);
        basis::free_shell_set_device_data(data_c);
        basis::free_shell_set_device_data(data_d);

        next_stream = (next_stream + 1) % n_streams;
        q_idx++;
    }

    cudaDeviceSynchronize();

    // Collect timing
    for (auto& qe : all_events) {
        float start_ms = 0, end_ms = 0;
        cudaEventElapsedTime(&start_ms, baseline, qe.start);
        cudaEventElapsedTime(&end_ms, baseline, qe.end);
        records.push_back({qe.stream_id, qe.quartet_idx, start_ms, end_ms,
                           end_ms - start_ms, qe.label});
        cudaEventDestroy(qe.start);
        cudaEventDestroy(qe.end);
    }

    cudaEventDestroy(baseline);
    for (auto& s : streams) cudaStreamDestroy(s);

    return records;
}

// ============================================================================
// Analysis
// ============================================================================

static void analyze_records(const std::vector<KernelRecord>& records,
                            const std::string& title,
                            int n_streams) {
    if (records.empty()) return;

    std::printf("\n=== %s ===\n", title.c_str());

    // Total wall time
    float first_start = records.front().start_ms;
    float last_end = records.back().end_ms;
    for (const auto& r : records) {
        if (r.start_ms < first_start) first_start = r.start_ms;
        if (r.end_ms > last_end) last_end = r.end_ms;
    }
    float wall_time = last_end - first_start;

    // Sum of all kernel durations
    double total_kernel_time = 0;
    for (const auto& r : records) total_kernel_time += r.duration_ms;

    // Per-stream statistics
    for (int s = 0; s < n_streams; ++s) {
        int count = 0;
        double stream_time = 0;
        float stream_first = 1e9, stream_last = 0;
        for (const auto& r : records) {
            if (r.stream_id == s) {
                count++;
                stream_time += r.duration_ms;
                if (r.start_ms < stream_first) stream_first = r.start_ms;
                if (r.end_ms > stream_last) stream_last = r.end_ms;
            }
        }
        if (count > 0) {
            float stream_span = stream_last - stream_first;
            float utilization = (stream_span > 0) ? (stream_time / stream_span * 100.0) : 0;
            std::printf("  Stream %d: %d kernels, %.3f ms compute, %.3f ms span, %.1f%% utilization\n",
                        s, count, stream_time, stream_span, utilization);
        }
    }

    // Overlap analysis: count how much time has 2+ kernels running
    // Discretize timeline into 0.001ms bins
    if (n_streams > 1) {
        const float bin_size = 0.001f;  // 1 microsecond bins
        int n_bins = static_cast<int>((wall_time / bin_size) + 1);
        if (n_bins > 10000000) n_bins = 10000000;  // Cap at 10M bins
        std::vector<int> occupancy(n_bins, 0);

        for (const auto& r : records) {
            int start_bin = static_cast<int>((r.start_ms - first_start) / bin_size);
            int end_bin = static_cast<int>((r.end_ms - first_start) / bin_size);
            start_bin = std::max(0, std::min(start_bin, n_bins - 1));
            end_bin = std::max(0, std::min(end_bin, n_bins - 1));
            for (int b = start_bin; b <= end_bin; ++b) occupancy[b]++;
        }

        int bins_0 = 0, bins_1 = 0, bins_2plus = 0;
        for (int b = 0; b < n_bins; ++b) {
            if (occupancy[b] == 0) bins_0++;
            else if (occupancy[b] == 1) bins_1++;
            else bins_2plus++;
        }

        int total_active = bins_1 + bins_2plus;
        float overlap_pct = total_active > 0 ? (100.0f * bins_2plus / total_active) : 0;
        float idle_pct = 100.0f * bins_0 / n_bins;

        std::printf("\n  Timeline analysis (%.3f ms wall time):\n", wall_time);
        std::printf("    Idle (0 kernels):    %.1f%%\n", idle_pct);
        std::printf("    Single kernel:       %.1f%%\n", 100.0f * bins_1 / n_bins);
        std::printf("    Overlapping (2+):    %.1f%%\n", 100.0f * bins_2plus / n_bins);
        std::printf("    Overlap ratio:       %.1f%% of active time\n", overlap_pct);
    }

    // Kernel duration histogram by AM type
    std::printf("\n  Kernel durations by AM type:\n");
    std::printf("  %-12s  %-6s  %-12s  %-12s  %-12s\n",
                "Quartet", "Count", "Avg (ms)", "Min (ms)", "Max (ms)");

    // Group by label
    struct Stats { int count; double sum, min_v, max_v; };
    std::vector<std::pair<std::string, Stats>> groups;
    for (const auto& r : records) {
        bool found = false;
        for (auto& g : groups) {
            if (g.first == r.label) {
                g.second.count++;
                g.second.sum += r.duration_ms;
                if (r.duration_ms < g.second.min_v) g.second.min_v = r.duration_ms;
                if (r.duration_ms > g.second.max_v) g.second.max_v = r.duration_ms;
                found = true;
                break;
            }
        }
        if (!found) {
            groups.push_back({r.label, {1, r.duration_ms, r.duration_ms, r.duration_ms}});
        }
    }

    for (const auto& [label, st] : groups) {
        std::printf("  %-12s  %-6d  %-12.4f  %-12.4f  %-12.4f\n",
                    label.c_str(), st.count, st.sum / st.count, st.min_v, st.max_v);
    }

    std::printf("\n  Summary: %.3f ms wall, %.3f ms total compute, %.2fx concurrency\n",
                wall_time, total_kernel_time, total_kernel_time / wall_time);

    // Gap analysis
    if (n_streams == 1) {
        double total_gap = 0;
        int n_gaps = 0;
        for (size_t i = 1; i < records.size(); ++i) {
            float gap = records[i].start_ms - records[i-1].end_ms;
            if (gap > 0) { total_gap += gap; n_gaps++; }
        }
        std::printf("  Inter-kernel gaps: %d gaps, %.3f ms total (%.1f%% of wall time)\n",
                    n_gaps, total_gap, 100.0 * total_gap / wall_time);
    }

    std::fflush(stdout);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::printf("No CUDA devices\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("GPU: %s (SM %d.%d, %d SMs)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    auto atoms = make_h2o();
    BasisSet basis = data::load_basis_set("aug-cc-pvdz", atoms);
    std::printf("H2O/aug-cc-pVDZ: %zu BF, %zu shells, %zu SSQuartets\n\n",
                basis.n_basis_functions(), basis.n_shells(),
                basis.shell_set_quartets().size());
    std::fflush(stdout);

    // Initialize Boys function
    double* d_boys_coeffs = nullptr;
    if (!device::math::boys_device_is_initialized()) {
        d_boys_coeffs = device::math::boys_device_init(nullptr);
    } else {
        d_boys_coeffs = device::math::boys_device_get_coeffs();
    }
    cudaDeviceSynchronize();

    cudaStream_t seq_stream;
    cudaStreamCreate(&seq_stream);

    // Warmup
    {
        auto records = profile_sequential(basis, d_boys_coeffs, seq_stream);
    }

    // Sequential
    auto seq = profile_sequential(basis, d_boys_coeffs, seq_stream);
    analyze_records(seq, "Sequential (1 stream)", 1);

    // Multi-stream with 2, 4, 8 streams
    for (int ns : {2, 4, 8}) {
        // Warmup
        profile_multistream(basis, d_boys_coeffs, ns);

        auto ms = profile_multistream(basis, d_boys_coeffs, ns);
        char title[64];
        std::snprintf(title, sizeof(title), "Multi-stream (%d streams)", ns);
        analyze_records(ms, title, ns);
    }

    cudaStreamDestroy(seq_stream);
    std::printf("\nDone.\n");
    return 0;
}

#else
#include <cstdio>
int main() { std::printf("CUDA not enabled.\n"); return 0; }
#endif
