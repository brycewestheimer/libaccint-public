# Phase 4.5 Batched Execution Baseline Measurements

## Purpose

This document captures baseline performance measurements before implementing true batched execution. These will be compared against post-implementation benchmarks to validate performance improvements.

## Test Configuration

**Hardware**: (to be filled in during benchmarking)
- CPU:
- GPU:
- Memory:

**Software**:
- Compiler:
- CUDA Version:
- Build: cpu-release / cuda-release

**Test Case**: Water molecule (H2O) with STO-3G basis
- 7 basis functions
- 5 shells (3 s-type on O, 1 s-type on each H)
- Representative small-molecule workload

## Baseline Measurements (Pre-Phase 4.5)

### CPU Sequential Performance

| Operation | Time (ms) | Notes |
|-----------|-----------|-------|
| compute_shell_set_pair (S) | - | Sequential iteration |
| compute_shell_set_pair (T) | - | Sequential iteration |
| compute_shell_set_pair (V) | - | Sequential iteration |
| compute_shell_set_quartet (ERI→Fock) | - | Sequential iteration |

### GPU "Fake Batching" Performance

| Operation | Time (ms) | Kernel Launches | Notes |
|-----------|-----------|-----------------|-------|
| compute_shell_set_pair (S) | - | - | Per-pair kernel launch |
| compute_shell_set_pair (T) | - | - | Per-pair kernel launch |
| compute_shell_set_pair (V) | - | - | Per-pair kernel launch |
| compute_eri_shell_quartet | - | - | Per-quartet kernel launch |

### Breakdown Analysis

**Kernel Launch Overhead**:
- Average kernel launch time: ~5μs (typical CUDA overhead)
- For ShellSetPair with N pairs: N × 5μs overhead
- For ShellSetQuartet with M quartets: M × 5μs overhead

**Memory Transfer Overhead**:
- Per-pair upload: ~2-3μs for small shells
- Per-pair download: ~1-2μs
- Total per-pair: ~8-10μs overhead

## Target Performance (Post-Phase 4.5)

### CPU Parallel (8 threads)

| Operation | Target Time (ms) | Target Speedup |
|-----------|-----------------|----------------|
| compute_shell_set_pair | - | ≥6× |
| compute_shell_set_quartet | - | ≥6× |

### GPU True Batching

| Operation | Target Time (ms) | Target Kernel Launches |
|-----------|-----------------|----------------------|
| compute_shell_set_pair | - | 1 per ShellSetPair |
| compute_shell_set_quartet | - | 1 ERI + 1 accumulate |

## Measurement Methodology

```cpp
// Example timing code
auto start = std::chrono::high_resolution_clock::now();

// Operation under test
engine.compute_shell_set_pair(op, pair, result);

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
```

For GPU operations, use `cudaEventRecord` with `cudaEventSynchronize` for accurate timing.

## Notes

- Measurements should be averaged over 10+ iterations after warmup
- Exclude first iteration (JIT compilation, cache warming)
- Report min/max/median alongside mean
- Document any system load or thermal throttling

---

*Baseline measurements to be recorded during implementation of Task 4.5.3.0*
