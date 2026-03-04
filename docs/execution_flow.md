# LibAccInt Execution Flow

This document describes the execution flow for molecular integral computation
in LibAccInt, centered on the primary `Engine::compute(...)` /
`compute_and_consume(...)` API. It distinguishes host-side iteration over
ShellSet work units from the batched execution that happens inside each
ShellSetPair or ShellSetQuartet on CPU and GPU backends.

## 1. CPU-Only Execution Path

The CPU-only path is always available. The primary full-basis API uses
OpenMP-backed host parallelism when OpenMP is enabled, more than one thread is
configured, and the workload is large enough; explicit `_parallel` entry
points remain available for direct control.

```mermaid
flowchart TD
    A[Define Molecule<br/>atoms + positions] --> B[Create BasisSet<br/>from shells]
    B --> C[BasisSet groups shells<br/>into ShellSets by AM, K]
    C --> D[Create Engine basis]
    D --> E[Engine creates CpuEngine]

    E --> F{Integral Type?}
    F -->|1e: S, T, V| G[1e Path]
    F -->|2e: ERIs| H[2e Path]

    subgraph one_electron [One-Electron Path]
        G --> G1[Build OneElectronOperator<br/>e.g. T + V for H_core]
        G1 --> G2[Iterate ShellSetPairs]
        G2 --> G3[OpenMP parallel:<br/>flatten 2D loop to 1D]
        G3 --> G4[Thread-local buffers<br/>OneElectronBuffer per thread]
        G4 --> G5[compute_1e_shell_pair<br/>per thread]
        G5 --> G6[Scatter to result matrix<br/>each pair has unique output region]
    end

    subgraph two_electron [Two-Electron Path]
        H --> H1[Iterate ShellSetQuartets<br/>from BasisSet worklist]
        H1 --> H2[OpenMP parallel:<br/>flatten 4D loop to 1D]
        H2 --> H3[Thread-local<br/>TwoElectronBuffer per thread]
        H3 --> H4[compute_2e_shell_quartet<br/>per thread]
        H4 --> H5[FockBuilder.accumulate<br/>J and K matrices]
    end

    G6 --> I[Assemble Matrices]
    H5 --> I
    I --> J[H_core = T + V]
    J --> K[F = H_core + J - 0.5*K]
    K --> L["E_elec = 0.5 * Tr[D*(H+F)]"]
    L --> M[E_total = E_elec + V_nn]
```

### Key Implementation Details (CPU)

- **Primary 1e path**: `Engine::compute(OneElectronOperator, result)` now
  preserves ShellSetPair batching. On CPU it uses `compute_1e_parallel()` by
  default when host parallelism is available; otherwise it falls back to the
  sequential full-basis path.

- **1e parallel with reduction**: `CpuEngine::compute_1e_parallel_impl` uses
  thread-local result matrices (`std::vector<Real>` per thread) with manual
  reduction after the parallel region. This avoids atomic operations.

- **2e integrals**: `CpuEngine::compute_and_consume_parallel` flattens the
  4D shell quartet loop into 1D, uses `schedule(dynamic, 8)` for load
  balancing (quartet compute times vary with angular momentum), and
  thread-local `TwoElectronBuffer` objects.

---

## 2. GPU-Only Execution Path

The GPU path uses CUDA kernels for compute-intensive operations.

```mermaid
flowchart TD
    A[Define Molecule] --> B[Create BasisSet]
    B --> C[Create Engine basis]
    C --> D{CUDA Available?}
    D -->|Yes| E[Engine creates CudaEngine]
    D -->|No| F[Fallback to CPU path]

    E --> G[Initialize CUDA stream<br/>+ GPU slot pool]
    G --> H[Upload Boys function<br/>Chebyshev coefficients to device]

    H --> I{Integral Type?}
    I -->|1e: Fused S+T+V| J[Fused 1e Path]
    I -->|2e: ERIs| K[2e Path]

    subgraph fused_1e [Fused One-Electron Path]
        J --> J0[Acquire GPU execution slot<br/>stream + device buffers]
        J0 --> J1[Iterate ShellSetPairs]
        J1 --> J2[Upload ShellSet SoA data<br/>centers, exponents, coefficients]
        J2 --> J3[Launch fused S+T+V kernel<br/>on slot's stream]
        J3 --> J4[D2H transfer:<br/>flat output buffer]
        J4 --> J5[Scatter to S, T, V matrices<br/>using shell function indices]
        J5 --> J6[Release slot back to pool]
    end

    subgraph gpu_2e [Two-Electron Path]
        K --> K0[Acquire GPU execution slot<br/>stream + device buffers]
        K0 --> K1[Iterate ShellSetQuartets<br/>from BasisSet worklist]
        K1 --> K2[Upload shell data<br/>to device cache]
        K2 --> K3[Launch ERI kernel<br/>on slot's stream]
        K3 --> K4[GpuFockBuilder<br/>device-side accumulate<br/>atomicAdd to J, K]
        K4 --> K5[Release slot back to pool]
    end

    J6 --> L[Assemble Matrices]
    K5 --> M[D2H transfer:<br/>J and K matrices]
    M --> L
    L --> N[H_core = T + V]
    N --> O[F = H_core + J - 0.5*K]
    O --> P[E_total]
```

### Key Implementation Details (GPU)

- **GPU execution slots**: Each GPU compute call acquires a `GpuExecutionSlot`
  (via RAII `ScopedGpuSlot`) from the `CudaEngine`'s internal `GpuSlotPool`.
  Each slot bundles an independent CUDA stream, device output buffers, and host
  staging vectors. This enables multiple host threads to drive GPU work
  concurrently without data races or serialization. The pool size is
  configurable via `DispatchConfig::n_gpu_slots` (default: 4). When all slots
  are occupied, additional threads block until a slot is released.

- **Fused 1e kernel**: `CudaEngine::compute_all_1e_fused` acquires a single
  slot for the entire iteration, uploads shell set data once, and runs a
  single kernel that computes overlap, kinetic, and nuclear attraction
  integrals simultaneously on the slot's stream. The output buffer is laid
  out as `[S_flat | T_flat | V_flat]` and scattered to full matrices on the
  host.

- **ERI kernel dispatch**: The ERI kernel is dispatched per `ShellSetQuartet`.
  The host iterates ShellSetQuartets, but each launch computes the whole
  batched quartet rather than reducing it to one shell quartet at a time.
  Handwritten generic kernels handle all AM combinations in this alpha.

- **Device-side Fock accumulation**: `GpuFockBuilder` keeps J, K, and D
  matrices on the GPU. ERI batches are accumulated directly into J and K
  using `atomicAdd`, avoiding expensive D2H transfers per batch.

- **Memory management**: `ShellSetDeviceCache` manages persistent device
  allocations for shell data (thread-safe via internal mutex), reusing buffers
  across kernel launches and slots. Boys function coefficients (`d_boys_coeffs_`)
  are shared read-only across all slots.

---

## 3. CPU+GPU Hybrid Execution Path

The Engine's `DispatchPolicy` selects the backend per operation.

```mermaid
flowchart TD
    A[Engine receives<br/>compute request] --> B{Check DispatchPolicy}

    B --> C{BackendHint?}
    C -->|ForceCPU| CPU[Route to CpuEngine]
    C -->|ForceGPU| GPU[Route to CudaEngine]
    C -->|Auto / PreferCPU / PreferGPU| D{Evaluate Heuristics}

    D --> E{Work Unit Type?}

    E -->|SingleShellPair<br/>SingleShellQuartet| CPU
    E -->|ShellSetPair<br/>ShellSetQuartet| F{batch_size >= min_gpu_batch_size?}
    E -->|FullBasis| G{n_shells >= min_gpu_shells?}

    F -->|No| CPU
    F -->|Yes| H{total_am >= high_am_threshold?}
    H -->|Yes| GPU
    H -->|No| I{n_primitives >= min_gpu_primitives?}
    I -->|Yes| GPU
    I -->|No| CPU

    G -->|No| CPU
    G -->|Yes| GPU

    CPU --> J[Execute on CPU]
    GPU --> K{GPU Available?}
    K -->|Yes| L[Execute on GPU]
    K -->|No| J
```

### Typical Hybrid Strategy

For small to medium molecules:
- **1e integrals (S, T, V)**: CPU or GPU depending on ShellSet batch size and primitive count
- **2e integrals (ERIs)**: GPU when a suitable single GPU is available; CPU otherwise

For large molecules:
- **All integrals**: GPU — both 1e and 2e benefit from GPU parallelism

### DispatchConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_gpu_batch_size` | 16 | Minimum ShellSet batch for GPU dispatch |
| `min_gpu_primitives` | 1000 | Minimum total primitives for GPU |
| `high_am_threshold` | 4 | AM threshold favoring GPU (G functions and above) |
| `min_gpu_shells` | 10 | Minimum shells for full-basis GPU dispatch |
| `enable_auto_tuning` | false | Enable runtime auto-tuning |
| `auto_tune_min_batch` | 100 | Minimum batch for auto-tuning |
| `n_gpu_slots` | 4 | Number of concurrent GPU execution slots (streams + buffers) |

---

## 4. Concurrent GPU Execution

When multiple host threads call GPU-routed computations on the same Engine,
CudaEngine manages concurrency via a pool of **GPU execution slots**. Each
slot contains:

- A dedicated CUDA stream
- Private device output buffers (1e, 2e, fused)
- Host staging vectors for D2H transfers

```mermaid
flowchart LR
    subgraph Host Threads
        T1[Thread 1]
        T2[Thread 2]
        T3[Thread 3]
        T4[Thread 4]
    end

    subgraph GpuSlotPool
        S0[Slot 0<br/>stream + buffers]
        S1[Slot 1<br/>stream + buffers]
        S2[Slot 2<br/>stream + buffers]
        S3[Slot 3<br/>stream + buffers]
    end

    subgraph GPU
        K0[Kernel on stream 0]
        K1[Kernel on stream 1]
        K2[Kernel on stream 2]
        K3[Kernel on stream 3]
    end

    T1 -->|acquire| S0
    T2 -->|acquire| S1
    T3 -->|acquire| S2
    T4 -->|acquire| S3

    S0 --> K0
    S1 --> K1
    S2 --> K2
    S3 --> K3
```

**Slot lifecycle:**

1. Host thread enters `compute_batch()` or `compute_shell_set_*()` via the
   GPU path
2. A `ScopedGpuSlot` acquires an available slot from the pool (blocks if none
   free)
3. The kernel launches on the slot's CUDA stream using the slot's device
   buffers
4. Results are transferred back to the host
5. `ScopedGpuSlot` destructor releases the slot back to the pool

**Shared resources** (safe across slots):

- `ShellSetDeviceCache`: thread-safe via internal mutex; shell data uploaded
  once and reused by all slots
- `d_boys_coeffs_`: read-only coefficient table shared by all kernels
- `GpuFockBuilder`: has its own device buffers, independent of the slot pool

**Example: parallel batch computation**

```cpp
Engine engine(basis);
auto& quartets = basis.shell_set_quartets();
std::vector<IntegralBuffer> results(quartets.size());

// Each iteration acquires a GPU slot, launches a kernel, releases the slot
#pragma omp parallel for schedule(dynamic)
for (Size i = 0; i < quartets.size(); ++i) {
    results[i] = engine.compute_batch(
        Operator::coulomb(), quartets[i], BackendHint::ForceGPU);
}
```

---

## 5. CPU Parallelization Details

LibAccInt uses **OpenMP exclusively** for CPU parallelism. There is no MPI,
no `std::thread`, and no `std::async` in the codebase.

### OpenMP Parallel Regions

There are three main OpenMP parallel regions in the CPU engine:

#### Region 1: Sequential 1e (Flat Parallel)

**File**: `src/host/engine/cpu_engine.cpp` — `compute_1e_impl`

```
#pragma omp parallel
{
    OneElectronBuffer<0> local_buffer;       // thread-local buffer

    #pragma omp for schedule(static)
    for (Size idx = 0; idx < total_pairs; ++idx) {
        Size i = idx / n_shells_b;
        Size j = idx % n_shells_b;
        // compute and scatter — no races, unique output regions
    }
}
```

The 2D loop over shell pairs `(i, j)` is flattened to a 1D loop of length
`n_shells_a * n_shells_b`. Since each `(i, j)` pair writes to a unique
region of the output matrix (determined by `function_index`), no
synchronization or reduction is needed.

#### Region 2: Parallel 1e with Reduction

**File**: `src/host/engine/cpu_engine.cpp` — `compute_1e_parallel_impl`

```
vector<CacheLineAligned<vector<Real>>> local_results(n_threads);

#pragma omp parallel num_threads(actual_threads)
{
    auto& local_result = local_results[tid];
    OneElectronBuffer<0> local_buffer;

    #pragma omp for schedule(dynamic, 4)
    for (Size t = 0; t < n_tasks; ++t) {
        // compute into local_result
    }
}

// Manual reduction
for (auto& local : local_results) {
    for (Size i = 0; i < nbf * nbf; ++i) {
        result[i] += local.get()[i];
    }
}
```

Each thread maintains a private copy of the full result matrix. After the
parallel region, results are reduced (summed) sequentially. The
`CacheLineAligned` wrapper prevents false sharing between adjacent
thread-local vectors.

#### Region 3: Parallel 2e Compute-and-Consume

**File**: `src/host/engine/cpu_engine.cpp` — `compute_and_consume_parallel`

```
#pragma omp parallel num_threads(actual_threads)
{
    TwoElectronBuffer<0> local_buffer;

    #pragma omp for schedule(dynamic, 8)
    for (Size idx = 0; idx < total_quartets; ++idx) {
        // decode (i, j, k, l) from linear index
        // compute ERIs into local_buffer
        // consumer.accumulate(...)  — consumer must be thread-safe
    }
}
```

The 4D shell quartet loop is flattened to 1D. `schedule(dynamic, 8)` is
used because higher-AM quartets take significantly longer than lower-AM
ones, so dynamic scheduling provides better load balancing.

### ThreadConfig Class

The `ThreadConfig` class (`include/libaccint/engine/thread_config.hpp`)
provides a unified interface for thread count management:

```cpp
// Query
int hw = ThreadConfig::hardware_threads();     // hardware concurrency
int rec = ThreadConfig::recommended_threads(); // reads OMP_NUM_THREADS
bool omp = ThreadConfig::openmp_available();   // compiled with OpenMP?

// Configure
ThreadConfig::set_num_threads(4);  // set explicit count
ThreadConfig::reset();             // revert to auto-detection

// Resolve (0 = auto)
int actual = ThreadConfig::resolve(n_threads);
```

Python access:
```python
import libaccint as lai
print(lai.ThreadConfig.hardware_threads())
lai.ThreadConfig.set_num_threads(4)
```

### CacheLineAligned Wrapper

Thread-local data is wrapped in `CacheLineAligned<T>` (aligned to 64 bytes)
to prevent false sharing when thread-local vectors are stored contiguously:

```cpp
template<typename T>
struct alignas(CACHE_LINE_SIZE) CacheLineAligned {
    T value;
    // implicit conversions + get() accessor
};
```

### ScopedThreadCount RAII Guard

For temporary thread count changes:

```cpp
{
    ScopedThreadCount guard(2);  // set to 2 threads
    engine.compute_and_consume_parallel(op, fock);
}  // automatically restores previous thread count
```
