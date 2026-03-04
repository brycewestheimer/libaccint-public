// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file batch_trace.hpp
/// @brief Optional batch-size diagnostic logging controlled by LIBACCINT_TRACE_BATCH
///
/// When the environment variable LIBACCINT_TRACE_BATCH is set (to any value),
/// trace lines are emitted to stderr for each 1e/2e dispatch and a summary is
/// printed at the end.  When the variable is unset, all methods return
/// immediately with zero overhead (no mutex, no I/O).

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>

namespace libaccint::diagnostics {

class BatchTracer {
public:
    static BatchTracer& instance() {
        static BatchTracer tracer;
        return tracer;
    }

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    /// @brief Trace a one-electron shell-set-pair dispatch
    /// @param op_name   Operator kind name (e.g. "Overlap")
    /// @param La        Angular momentum of shell set A
    /// @param Lb        Angular momentum of shell set B
    /// @param n_primitives_a  Primitives per shell in set A
    /// @param n_primitives_b  Primitives per shell in set B
    /// @param n_pairs   Number of shell pairs in the batch
    /// @param backend   "CPU" or "GPU"
    void trace_1e_dispatch(const char* op_name, int La, int Lb,
                           int n_primitives_a, int n_primitives_b,
                           size_t n_pairs, const char* backend) {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        std::cerr << "[BATCH_TRACE] 1e " << op_name
                  << " (" << La << Lb << ") K=" << n_primitives_a << "x" << n_primitives_b
                  << " n_pairs=" << n_pairs
                  << " -> " << backend << "\n";
        total_1e_dispatches_++;
        total_1e_pairs_ += n_pairs;
        if (std::string(backend) == "GPU") gpu_1e_dispatches_++;
    }

    /// @brief Trace a two-electron shell-set-quartet dispatch
    /// @param op_name     Operator kind name (e.g. "Coulomb")
    /// @param La          Bra-A angular momentum
    /// @param Lb          Bra-B angular momentum
    /// @param Lc          Ket-A angular momentum
    /// @param Ld          Ket-B angular momentum
    /// @param n_quartets  Number of shell quartets in the batch
    /// @param strategy    Dispatch strategy description
    /// @param backend     "CPU" or "GPU"
    void trace_2e_dispatch(const char* op_name, int La, int Lb, int Lc, int Ld,
                           size_t n_quartets, const char* strategy, const char* backend) {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        std::cerr << "[BATCH_TRACE] 2e " << op_name
                  << " (" << La << Lb << "|" << Lc << Ld << ")"
                  << " n_quartets=" << n_quartets
                  << " strategy=" << strategy
                  << " -> " << backend << "\n";
        total_2e_dispatches_++;
        total_2e_quartets_ += n_quartets;
        if (std::string(backend) == "GPU") gpu_2e_dispatches_++;
    }

    /// @brief Print accumulated summary to stderr
    void print_summary() {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        std::cerr << "\n[BATCH_TRACE] ========== Summary ==========\n"
                  << "[BATCH_TRACE] 1e dispatches: " << total_1e_dispatches_
                  << " (GPU: " << gpu_1e_dispatches_ << ")"
                  << " total_pairs: " << total_1e_pairs_ << "\n"
                  << "[BATCH_TRACE] 2e dispatches: " << total_2e_dispatches_
                  << " (GPU: " << gpu_2e_dispatches_ << ")"
                  << " total_quartets: " << total_2e_quartets_ << "\n";
        if (total_2e_dispatches_ > 0) {
            std::cerr << "[BATCH_TRACE] avg 2e batch size: "
                      << (total_2e_quartets_ / total_2e_dispatches_) << "\n";
        }
        std::cerr << "[BATCH_TRACE] ================================\n";
    }

    /// @brief Reset all counters (useful for testing)
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        total_1e_dispatches_ = 0; gpu_1e_dispatches_ = 0; total_1e_pairs_ = 0;
        total_2e_dispatches_ = 0; gpu_2e_dispatches_ = 0; total_2e_quartets_ = 0;
    }

private:
    BatchTracer() : enabled_(std::getenv("LIBACCINT_TRACE_BATCH") != nullptr) {}

    bool enabled_;
    std::mutex mutex_;
    size_t total_1e_dispatches_ = 0, gpu_1e_dispatches_ = 0, total_1e_pairs_ = 0;
    size_t total_2e_dispatches_ = 0, gpu_2e_dispatches_ = 0, total_2e_quartets_ = 0;
};

}  // namespace libaccint::diagnostics
