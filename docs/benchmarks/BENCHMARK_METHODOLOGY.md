# Benchmark Methodology

## Overview

This document describes the methodology used for performance benchmarking in LibAccInt.
All benchmarks follow reproducible procedures designed to give statistically meaningful
results on modern hardware.

## Benchmark Infrastructure

### Google Benchmark Integration

LibAccInt uses [Google Benchmark](https://github.com/google/benchmark) (v1.8.3) for
microbenchmarks. The harness provides:

- Automatic iteration count calibration
- CPU cycle-accurate timing
- Statistical aggregation (mean, median, stddev)
- JSON/CSV output for downstream analysis

### Custom Benchmark Harness

The `BenchmarkHarness` class (`include/libaccint/benchmark/benchmark_harness.hpp`)
provides additional capabilities:

- Parametric sweeps over molecules, basis sets, integral types, and backends
- Warmup iterations to avoid cold-start artifacts
- Configurable repetition and timing thresholds
- Structured result collection with JSON/CSV export

### Comparison Framework

The `ComparisonFramework` class enables head-to-head comparison with reference
integral libraries (libint2, libcint). When reference libraries are not compiled
in, estimated timings based on known algorithmic complexity are used.

## Benchmark Categories

### 1. Per-Kernel Microbenchmarks (Task 27.1.3)

Measures individual kernel performance across all angular momentum (AM) combinations:

| Integral Type | AM Range | Shells |
|---------------|----------|--------|
| Overlap (S)   | (0,0)–(3,3) | 16 combinations |
| Kinetic (T)   | (0,0)–(3,3) | 16 combinations |
| Nuclear (V)   | (0,0)–(3,3) | 16 combinations |
| ERI           | AM 0–3 | Via Fock build |

**Methodology:**
- Two-shell basis set per AM pair
- STO-3G-like exponents (3 primitives)
- Google Benchmark auto-sizing for iteration count
- Throughput reported as integrals/second

### 2. End-to-End HF Benchmark (Task 27.1.4)

Complete Hartree-Fock Fock matrix construction:

| Molecule | Basis | Atoms | NBF | Description |
|----------|-------|-------|-----|-------------|
| H₂O     | STO-3G | 3   | 7   | Minimal reference |
| CH₄     | STO-3G | 5   | 9   | Small organic |
| (H₂O)ₙ  | STO-3G | 3n  | 7n  | System-size scaling |

**Methodology:**
- Full Fock build: S, T, V (1e) + J, K (2e)
- Random symmetric density matrix
- Threading variants: 1, 2, 4, 8 threads

### 3. CPU Optimization (Tasks 27.2.1–27.2.4)

#### SIMD Vectorization Audit
- Reports detected ISA (AVX2, SSE, scalar)
- Vector add and FMA throughput benchmarks
- Comparison of vectorized vs scalar paths

#### Cache Utilization
- Sequential vs strided access patterns
- Buffer sizes spanning L1 → L3 → main memory
- Bytes/second normalized throughput

#### Contraction Loop
- Overlap contraction timing
- Fock build contraction + accumulation

#### Memory Allocation
- `std::vector` allocation overhead
- `MemoryPool` pool allocation performance
- Size class coverage (64B → 64KB)

### 4. GPU Optimization (Tasks 27.3.1–27.3.3)

All GPU benchmarks skip gracefully when CUDA is not available.

#### Kernel Occupancy
- Fock builds at varying system sizes (1–8 H₂O molecules)
- GPU vs CPU dispatch comparison

#### Memory Transfer
- 1e integral transfer-dominated workloads
- Repeated computation data reuse verification

#### Warp Utilization
- CPU vs GPU head-to-head on same workload
- AM-stratified GPU correctness

### 5. Scaling Studies (Tasks 27.4.1–27.4.2)

#### Strong Scaling
- Fixed problem size, threads: {1, 2, 4, 8}
- Speedup = T₁/Tₙ
- Efficiency = Speedup / N
- Amdahl's law serial fraction estimate

#### Weak Scaling
- Problem size scales with thread count
- Ideal: time remains constant
- Efficiency = T₁/Tₙ

## Measurement Methodology

### Timing
- `std::chrono::steady_clock` for wall-clock time
- Google Benchmark uses CPU time by default
- Minimum 3 iterations, configurable up to 100
- Minimum 0.5s runtime per benchmark

### Warmup
- 2 warmup iterations by default (configurable)
- Prevents cold-cache and JIT compilation artifacts

### Statistical Treatment
- Mean, standard deviation, min, max reported
- Google Benchmark applies automatic outlier detection
- Custom harness uses simple mean ± stddev

### Reproducibility
- Fixed random seeds for density matrices
- Deterministic molecule geometries (equilibrium in Bohr)
- Controlled thread counts via OMP_NUM_THREADS

## Running Benchmarks

```bash
# Build with benchmark support
cmake --preset cpu-release -DLIBACCINT_BUILD_BENCHMARKS=ON
cmake --build --preset cpu-release

# Run specific benchmark
./build/cpu-release/tests/bench_kernel_microbenchmarks --benchmark_format=json

# Run validation tests
ctest --test-dir build/cpu-release -R "test_benchmark|test_scaling|test_cpu_opt|test_gpu_opt"

# Run all benchmarks
./build/cpu-release/tests/bench_hf_endtoend
./build/cpu-release/tests/bench_cpu_optimization
./build/cpu-release/tests/bench_gpu_optimization
```

## Reporting

Results are exported in JSON and CSV formats compatible with:
- Python/matplotlib for visualization
- GitHub Actions artifacts for CI tracking
- Comparison reports for reference library timing ratios
