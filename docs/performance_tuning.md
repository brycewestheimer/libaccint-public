# Performance Tuning Guide

This document covers the tuning knobs available in LibAccInt for optimizing
molecular integral computation performance. It covers backend selection,
integral screening, thread tuning, GPU kernel dispatch strategy, multi-GPU
configuration, and general recommendations.

## Backend Selection

LibAccInt provides multiple execution backends that can be selected
automatically or manually.

### Execution Strategies

The library supports seven execution strategies, defined in
`include/libaccint/kernels/execution_strategy.hpp`:

| Strategy                | Description                                       |
|-------------------------|---------------------------------------------------|
| `SerialCPU`             | Single-threaded, scalar computation               |
| `SimdCPU`               | Single-threaded, SIMD vectorized (AVX/AVX-512)    |
| `ThreadedCPU`           | OpenMP multi-threaded, scalar                     |
| `ThreadedSimdCPU`       | OpenMP multi-threaded with SIMD vectorization     |
| `ThreadPerIntegralGPU`  | One GPU thread per integral (high parallelism)    |
| `WarpPerQuartetGPU`     | One warp (32 threads) per quartet (collaborative) |
| `BlockPerBatchGPU`      | One thread block per ShellSet batch (coalesced)   |

### Backend Hints

Users can influence backend selection using `BackendHint`, defined in
`include/libaccint/engine/dispatch_policy.hpp`:

| Hint          | Behavior                                                |
|---------------|---------------------------------------------------------|
| `Auto`        | Let the dispatch policy decide (default)                |
| `ForceCPU`    | Always use CPU backend                                  |
| `ForceGPU`    | Always use GPU backend (error if unavailable)           |
| `PreferCPU`   | Prefer CPU, but allow GPU if beneficial                 |
| `PreferGPU`   | Prefer GPU, fall back to CPU if unavailable             |

**C++ example:**

```cpp
Engine engine(basis);

// Automatic dispatch (default)
engine.compute_overlap_matrix(S);

// Force GPU for large computations
engine.compute_overlap_matrix(S, BackendHint::ForceGPU);

// Force CPU for debugging or benchmarking
engine.compute_overlap_matrix(S, BackendHint::ForceCPU);
```

**Python example:**

```python
import libaccint

engine = libaccint.Engine(basis)
S = engine.compute_overlap_matrix()                                    # Auto
S = engine.compute_overlap_matrix(hint=libaccint.BackendHint.ForceGPU) # GPU
S = engine.compute_overlap_matrix(hint=libaccint.BackendHint.ForceCPU) # CPU
```

### Dispatch Configuration

The `DispatchConfig` struct controls the heuristics for automatic backend
selection. It is defined in `include/libaccint/engine/dispatch_policy.hpp`.

```cpp
DispatchConfig config;
config.min_gpu_batch_size  = 16;    // Minimum quartets for GPU dispatch
config.min_gpu_primitives  = 1000;  // Minimum primitives for GPU dispatch
config.high_am_threshold   = 4;     // Angular momentum threshold for GPU preference
config.min_gpu_shells      = 10;    // Minimum shells for full-basis GPU dispatch
config.enable_auto_tuning  = false; // Enable runtime auto-tuning
config.n_gpu_slots         = 4;    // Concurrent GPU execution slots (streams + buffers)

Engine engine(basis, config);
```

**Key parameters:**

- **`min_gpu_batch_size`** (default: 16): Below this number of shell quartets,
  kernel launch overhead dominates and the CPU is used. Increase this value on
  systems with slow GPU kernel launch latency. Decrease it on systems with fast
  PCIe interconnects and low-latency GPU launches.

- **`min_gpu_primitives`** (default: 1000): The total number of Gaussian
  primitives involved in a batch. GPU streaming multiprocessors need enough
  work to stay occupied, so small primitive counts benefit from CPU execution.

- **`high_am_threshold`** (default: 4): When the total angular momentum of a
  quartet exceeds this value, the GPU is preferred even for smaller batches,
  because high-AM quartets have more FLOPs per quartet.

- **`enable_auto_tuning`** (default: false): When enabled, the dispatch policy
  uses the `DispatchRegistry` to cache timing-based decisions. The auto-tuner
  profiles the first execution of each angular momentum class and caches the
  optimal strategy for subsequent calls.

- **`n_gpu_slots`** (default: 4): Number of GPU execution slots in the pool.
  Each slot owns an independent CUDA stream and set of device output buffers,
  enabling concurrent GPU kernel execution from multiple host threads. Reduce
  on GPUs with limited memory (each slot uses ~85 MB for 2e buffers with a
  60-function basis). See the "GPU Concurrency Tuning" section for details.

### Auto-Tuning Modes

When `enable_auto_tuning` is true, the auto-tuner operates in one of three modes:

| Mode             | Description                                                |
|------------------|------------------------------------------------------------|
| `Analytical`     | Pure cost model estimation (no profiling)                  |
| `ProfileOnce`    | Profile the first call, cache the result (default)         |
| `AdaptiveTune`   | Periodically re-profile to adapt to changing conditions    |

```cpp
config.enable_auto_tuning = true;
config.auto_tune_mode = kernels::KernelCalculator::Mode::ProfileOnce;
config.auto_tune_min_batch = 100;  // Skip auto-tuning for tiny batches
```

## Integral Screening Configuration

Schwarz integral screening can dramatically reduce the number of two-electron
integrals computed. Configuration is via the `ScreeningOptions` struct, defined
in `include/libaccint/screening/screening_options.hpp`.

### Screening Presets

| Preset   | Threshold | Description                          |
|----------|-----------|--------------------------------------|
| `None`   | (off)     | No screening -- all quartets computed|
| `Loose`  | 1e-10     | Faster, lower accuracy               |
| `Normal` | 1e-12     | Good balance (default)                |
| `Tight`  | 1e-14     | Maximum accuracy, less speedup        |
| `Custom` | user      | User-specified threshold              |

**C++ example:**

```cpp
// Use a preset
auto opts = ScreeningOptions::tight();

// Enable density-weighted screening for SCF iterations
opts.density_weighted = true;

// Enable 8-fold permutation symmetry exploitation
opts.use_permutation_symmetry = true;

// Enable statistics collection
opts.enable_statistics = true;
```

### Density-Weighted Screening

When `density_weighted` is true, the screening bound becomes:

    D_max * Q_ab * Q_cd < threshold

where `D_max` is the maximum density matrix element over all AO pairs in the
quartet. This provides much tighter bounds during SCF iteration because the
density matrix becomes sparser as convergence is approached.

### Permutation Symmetry

When `use_permutation_symmetry` is true, only canonical shell quartets
(i<=j, k<=l, ij<=kl) are iterated, reducing the number of computed quartets
by up to 8x. The consumer (e.g., `FockBuilder`) must support symmetric
accumulation.

### Screening Statistics

Enable `enable_statistics` to track how effective screening is:

```cpp
opts.enable_statistics = true;
opts.verbosity = 1;  // 0 = silent, 1 = summary, 2 = detailed

// After computation:
auto stats = engine.screening_statistics();
std::cout << "Skipped " << stats.skip_percentage() << "% of quartets\n";
std::cout << "Computed: " << stats.computed_quartets
          << " / " << stats.total_quartets << "\n";
```

## OpenMP Thread Tuning

Thread management is handled by the `ThreadConfig` class, defined in
`include/libaccint/engine/thread_config.hpp`.

### Setting Thread Count

```cpp
// Auto-detect (uses OMP_NUM_THREADS or hardware threads)
ThreadConfig::recommended_threads();

// Set a specific count
ThreadConfig::set_num_threads(8);

// Query the current setting
int n = ThreadConfig::num_threads();      // 0 means auto-detect mode
int hw = ThreadConfig::hardware_threads(); // Number of logical cores

// Reset to auto-detect
ThreadConfig::reset();
```

### Scoped Thread Count

Use `ScopedThreadCount` for temporary thread count changes that revert
automatically:

```cpp
{
    ScopedThreadCount guard(4);
    // Uses 4 threads in this scope
    engine.compute_1e_parallel<0>(op, S, 0);
}
// Reverts to previous thread count here
```

### Environment Variables

- **`OMP_NUM_THREADS`**: Sets the default thread count when no explicit
  configuration is provided.

### Parallel Computation Methods

One-electron integrals support parallel computation directly:

```cpp
// Serial (single-threaded)
engine.compute_1e<0>(op, S);

// Parallel with auto-detected thread count
engine.compute_1e_parallel<0>(op, S);

// Parallel with explicit thread count
engine.compute_1e_parallel<0>(op, S, 8);
```

Two-electron integrals use parallelism through the consumer pattern:

```cpp
// Parallel two-electron compute-and-consume
engine.compute_and_consume_parallel(Operator::coulomb(), fock_builder, 8);
```

### False Sharing Prevention

When using per-thread accumulators, use `CacheLineAligned<T>` to prevent
false sharing between cores:

```cpp
#include <libaccint/engine/thread_config.hpp>

// Each element is padded to a 64-byte cache line
std::vector<CacheLineAligned<double>> per_thread_energy(n_threads);
```

### Thread Tuning Recommendations

- **Small molecules (< 50 basis functions):** Use 1-4 threads. Overhead of
  thread creation and synchronization may exceed the benefit.
- **Medium molecules (50-500 basis functions):** Use all available physical
  cores. Hyperthreading provides marginal benefit for compute-bound work.
- **Large molecules (> 500 basis functions):** Use all cores. Consider GPU
  dispatch for two-electron integrals.

## GPU Concurrency Tuning

When multiple host threads invoke the supported batched GPU APIs on the same
`Engine`, the underlying `CudaEngine` manages concurrent access via a pool
of **GPU execution slots**. Each slot owns an independent CUDA stream and
set of device output buffers, enabling concurrent kernel execution on the
GPU while keeping each call's temporary buffers isolated.

### How It Works

```
Host thread 1 ──▶ acquire slot 0 ──▶ launch kernel on stream 0 ──▶ release
Host thread 2 ──▶ acquire slot 1 ──▶ launch kernel on stream 1 ──▶ release
Host thread 3 ──▶ acquire slot 2 ──▶ launch kernel on stream 2 ──▶ release
Host thread 4 ──▶ (all slots busy) ──▶ blocks until a slot is freed ──▶ ...
```

All slot management is internal. The supported shared-instance alpha paths
are the slot-backed batched APIs such as `compute_batch(...)`,
`compute_batch_parallel(...)`, `compute_all_2e_parallel(...)`, and
`compute_shell_set_pair()`:

```cpp
// Multi-threaded GPU compute — each thread gets its own slot automatically
Engine engine(basis);
auto& quartets = basis.shell_set_quartets();
std::vector<IntegralBuffer> results(quartets.size());

#pragma omp parallel for schedule(dynamic)
for (Size i = 0; i < quartets.size(); ++i) {
    results[i] = engine.compute_batch(
        Operator::coulomb(), quartets[i], BackendHint::ForceGPU);
}
```

### Configuring Slot Count

The number of GPU execution slots is controlled by
`DispatchConfig::n_gpu_slots` (default: 4):

```cpp
DispatchConfig config;
config.n_gpu_slots = 2;   // 2 concurrent GPU streams
Engine engine(basis, config);
```

**Tuning guidance:**

| GPU Memory | Recommended `n_gpu_slots` | Notes |
|------------|--------------------------|-------|
| 4-6 GB     | 1-2                      | Each slot allocates ~85 MB for 2e buffers |
| 8-12 GB    | 2-4                      | Good balance for most workloads |
| 16+ GB     | 4-8                      | Higher concurrency for large systems |

When all slots are occupied, additional host threads block until a slot is
released. This backpressure mechanism prevents GPU out-of-memory errors.

Do not change dispatch configuration, device selection, density screening
inputs, or process-global thread settings while concurrent GPU work is in
flight. See `docs/api/THREAD_SAFETY.md` for the authoritative support
matrix.

### Serial GPU Usage

For single-threaded GPU workflows, no configuration change is needed — a
single slot is acquired and released automatically on each call:

```cpp
Engine engine(basis);

// Single-threaded GPU — works exactly as before
engine.compute_overlap_matrix(S, BackendHint::ForceGPU);
engine.compute_and_consume(Operator::coulomb(), fock, BackendHint::ForceGPU);
```

### Consumer Patterns with Parallel GPU

`compute_and_consume_parallel()` is a CPU/OpenMP entry point. For GPU
compute-and-consume, use `compute_and_consume(...)` or
`compute_shell_set_quartet(...)` and follow the consumer-specific guarantees
documented in `docs/api/THREAD_SAFETY.md`. For alpha, shared-instance GPU
concurrency is intentionally narrower than the CPU `_parallel` contract.

## K-Range Dispatch (GPU Kernel Strategy)

The GPU backend dispatches to different CUDA kernel configurations based on
the angular momentum class of each shell quartet. This is managed by the
`OptimalDispatchTable` (in `include/libaccint/kernels/optimal_dispatch_table.hpp`)
and `DispatchRegistry` (in `include/libaccint/kernels/dispatch_registry.hpp`).

### GPU Kernel Strategies

**ThreadPerIntegralGPU:** Assigns one GPU thread per individual integral
component. Best for low angular momentum (s, p) where each quartet produces
few integrals. Maximizes parallelism for small quartets.

**WarpPerQuartetGPU:** Assigns a full warp (32 threads) to each shell quartet.
The threads collaborate on the Rys quadrature roots and weights, then scatter
the results. Best for high angular momentum (d, f, g) where each quartet
produces many integral components. The collaborative approach exploits data
reuse within the warp.

**BlockPerBatchGPU:** Assigns an entire thread block to a ShellSet batch.
Provides the best memory coalescing for batched execution. Used for
medium-sized batches where the ShellSet contains many shell pairs.

### Small-Batch CPU Fallback

The `CudaEngine` implements a small-batch guard (controlled by
`DispatchConfig::min_gpu_batch_size`). When a `ShellSetQuartet` contains fewer
shell quartets than this threshold, the computation is automatically routed to
a CPU fallback engine to avoid GPU kernel launch overhead:

```cpp
CudaEngine cuda_engine(basis);
CpuEngine cpu_engine(basis);

// Wire up CPU fallback
cuda_engine.set_cpu_fallback(&cpu_engine);

// Quartets smaller than 16 will run on CPU automatically
DispatchConfig config;
config.min_gpu_batch_size = 16;
cuda_engine.set_dispatch_config(config);
```

### Dispatch Registry Warmup

For production runs, warm up the dispatch registry to avoid first-call
profiling overhead:

```cpp
auto& registry = kernels::get_dispatch_registry();
registry.warmup(basis.max_angular_momentum(), BackendType::CUDA);
```

### Pipelined ERI Computation

For large two-electron integral computations, use the multi-stream pipeline
to overlap computation with data transfer:

```cpp
EriPipelineConfig pipe_config;
pipe_config.n_slots = 4;        // 4 concurrent pipeline slots (ring buffer)
pipe_config.use_callback = false;

std::vector<double> eri_tensor(nbf * nbf * nbf * nbf);
cuda_engine.compute_eri_pipelined(eri_tensor, pipe_config);
```

**Pipeline slot count tuning:**
- **2 slots:** Minimum for overlapping compute and transfer.
- **4 slots (default):** Good balance for most GPUs.
- **8 slots:** May help on systems with high kernel launch latency.

For even better performance, use device-side scatter to avoid the per-batch
pinned-memory scatter bottleneck:

```cpp
cuda_engine.compute_eri_device_scatter(eri_tensor, pipe_config);
```

## Multi-GPU Configuration

> **Experimental:** `MultiGPUEngine` and related multi-device orchestration are
> outside the alpha performance contract.

LibAccInt supports distributing work across multiple GPU devices via the
`MultiGPUEngine`, defined in `include/libaccint/engine/multi_gpu_engine.hpp`.

### Basic Setup

```cpp
MultiGPUConfig config;
config.device_ids = {0, 1, 2, 3};    // Use GPUs 0-3 (empty = use all)
config.enable_peer_access = true;      // Enable direct GPU-GPU transfers
config.streams_per_device = 2;         // Concurrent streams per device
config.async_execution = true;         // Overlap compute and communication
config.collect_stats = true;           // Collect timing statistics

MultiGPUEngine engine(basis, config);
```

### Work Distribution

Work is partitioned across devices using the `WorkDistributor`, which supports
configurable load-balancing strategies via `DistributionConfig`. The
multi-GPU engine includes work-stealing queues for dynamic load balancing:
each device drains its own queue (LIFO for cache locality) and steals from
other devices' queues (FIFO to reduce contention) when idle.

### Resource-Aware Weights

The multi-GPU engine can dynamically adjust device weights based on live
resource availability (SM occupancy, memory pressure):

```cpp
engine.update_resource_aware_weights();
```

This queries each device's `DeviceResourceTracker` and adjusts the work
partition so that busier devices receive less work.

### Performance Statistics

```cpp
engine.compute_all_eri(fock_builder);

auto& stats = engine.stats();
std::cout << "Total time: " << stats.total_time_ms << " ms\n";
std::cout << "Load balance efficiency: "
          << stats.load_balance_efficiency() << "\n";

for (int i = 0; i < engine.device_count(); ++i) {
    std::cout << "GPU " << engine.device_ids()[i]
              << ": " << stats.per_device_quartets[i] << " quartets, "
              << stats.per_device_time_ms[i] << " ms\n";
}
```

### Multi-GPU Tuning Recommendations

- **Uniform GPUs:** Use default distribution. All devices get equal work.
- **Heterogeneous GPUs:** Enable resource-aware weights to automatically
  account for differences in SM count and clock speed.
- **High-latency interconnects (e.g., PCIe):** Increase `streams_per_device`
  to overlap more transfers with computation.
- **NVLink-connected GPUs:** Enable `enable_peer_access` for direct GPU-GPU
  data movement without host staging.

## General Performance Recommendations

### Memory Layout

- All matrices are stored in **row-major** order (C-style).
- The `IntegralBuffer` class supports both individual quartet retrieval and
  flat array access for efficient post-processing.
- Use `std::vector<Real>` for output buffers; the engine will resize them
  as needed.

### Batched vs. Shell-Level API

Always prefer the **batched** API (`compute_shell_set_pair`,
`compute_shell_set_quartet`, `compute_batch`) over the deprecated
shell-by-shell methods. The batched API:

1. Enables GPU kernel amortization across many shell pairs/quartets.
2. Reduces kernel launch overhead by processing entire `ShellSet` batches.
3. Allows the engine to make better dispatch decisions based on batch size.

### Fused One-Electron Computation

When computing overlap (S), kinetic (T), and nuclear attraction (V) matrices
together, use the fused kernel on GPU:

```cpp
cuda_engine.compute_all_1e_fused(charges, S, T, V);
```

This computes all three matrices in a single pass, sharing Gaussian product
computation and eliminating 66% of kernel launches compared to computing
S, T, and V separately.

### Recommended Workflow for Production

1. Set up `DispatchConfig` with auto-tuning enabled.
2. Warm up the dispatch registry.
3. Use `ScreeningOptions::normal()` with `density_weighted = true` for SCF.
4. Use the fused 1e API for one-electron integrals.
5. Use `compute_and_consume_parallel` for two-electron integrals on CPU,
   or `compute_shell_set_quartet` with `GpuFockBuilder` on GPU.
6. Monitor screening statistics and adjust thresholds if needed.

```cpp
// Production configuration example
DispatchConfig config;
config.enable_auto_tuning = true;
config.auto_tune_mode = kernels::KernelCalculator::Mode::ProfileOnce;

Engine engine(basis, config);

// Warm up
kernels::get_dispatch_registry().warmup(
    basis.max_angular_momentum(), BackendType::CUDA);

// Screening
auto screen = ScreeningOptions::normal();
screen.density_weighted = true;
screen.use_permutation_symmetry = true;

// Compute
engine.compute_all_1e_fused(charges, S, T, V);  // GPU-accelerated
engine.compute_and_consume_parallel(Operator::coulomb(), fock_builder);
```
