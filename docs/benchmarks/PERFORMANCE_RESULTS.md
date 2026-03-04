# Performance Results

## Overview

This document captures performance benchmark results for LibAccInt.
Results are collected on the reference hardware and updated with each release.

## Reference Hardware

| Component | Specification |
|-----------|---------------|
| CPU       | _To be filled_ |
| Cores     | _To be filled_ |
| Memory    | _To be filled_ |
| GPU       | _To be filled_ (if applicable) |
| OS        | _To be filled_ |
| Compiler  | _To be filled_ |
| Flags     | `-O3 -march=native` |

## Per-Kernel Microbenchmarks

### Overlap Integrals (S)

| AM (a,b) | Mean Time (ns) | Throughput (integrals/s) |
|----------|----------------|-------------------------|
| (0,0)    | —              | —                       |
| (0,1)    | —              | —                       |
| (1,1)    | —              | —                       |
| (0,2)    | —              | —                       |
| (1,2)    | —              | —                       |
| (2,2)    | —              | —                       |
| (0,3)    | —              | —                       |
| (1,3)    | —              | —                       |
| (2,3)    | —              | —                       |
| (3,3)    | —              | —                       |

### Kinetic Integrals (T)

| AM (a,b) | Mean Time (ns) | Throughput (integrals/s) |
|----------|----------------|-------------------------|
| (0,0)    | —              | —                       |
| (1,1)    | —              | —                       |
| (2,2)    | —              | —                       |
| (3,3)    | —              | —                       |

### Nuclear Attraction Integrals (V)

| AM (a,b) | Mean Time (ns) | Throughput (integrals/s) |
|----------|----------------|-------------------------|
| (0,0)    | —              | —                       |
| (1,1)    | —              | —                       |
| (2,2)    | —              | —                       |
| (3,3)    | —              | —                       |

### ERI / Fock Build Performance

| AM  | Mean Time (ns) | Shell Quartets/s |
|-----|----------------|------------------|
| 0   | —              | —                |
| 1   | —              | —                |
| 2   | —              | —                |
| 3   | —              | —                |

## End-to-End HF Benchmarks

### Single-Thread Performance

| Molecule | Basis | NBF | 1e Time (µs) | 2e Time (µs) | Total (µs) |
|----------|-------|-----|---------------|---------------|-------------|
| H₂O     | STO-3G | 7  | —             | —             | —           |
| CH₄     | STO-3G | 9  | —             | —             | —           |
| (H₂O)₂  | STO-3G | 14 | —             | —             | —           |
| (H₂O)₄  | STO-3G | 28 | —             | —             | —           |

### Parallel Scaling

| Threads | Time (µs) | Speedup | Efficiency |
|---------|-----------|---------|------------|
| 1       | —         | 1.00x   | 100%       |
| 2       | —         | —       | —          |
| 4       | —         | —       | —          |

## CPU Optimization Results

### SIMD Configuration

| Property | Value |
|----------|-------|
| ISA      | _Detected at build time_ |
| Width    | _Detected at build time_ |
| Alignment | _Detected at build time_ |

### Memory Allocation

| Method | Size | Mean Time (ns) | Speedup vs std::vector |
|--------|------|----------------|------------------------|
| std::vector | 256 | — | 1.0x |
| MemoryPool  | 256 | — | —    |
| std::vector | 4096 | — | 1.0x |
| MemoryPool  | 4096 | — | —    |

## GPU Performance (when available)

### GPU vs CPU Comparison

| System | Basis | CPU Time (µs) | GPU Time (µs) | Speedup |
|--------|-------|---------------|---------------|---------|
| H₂O   | STO-3G | — | — | — |
| (H₂O)₄ | STO-3G | — | — | — |
| (H₂O)₈ | STO-3G | — | — | — |

## Reference Library Comparison

### vs libint2 (estimated)

| Integral | Molecule | Basis | LibAccInt (ns) | libint2 (ns) | Ratio |
|----------|----------|-------|----------------|--------------|-------|
| Overlap  | H₂O     | STO-3G | — | — | — |
| Kinetic  | H₂O     | STO-3G | — | — | — |
| ERI      | H₂O     | STO-3G | — | — | — |

## Scaling Studies

### Strong Scaling

| Threads | Time (s) | Speedup | Efficiency | Serial Fraction |
|---------|----------|---------|------------|-----------------|
| 1       | —        | 1.00    | 100%       | —               |
| 2       | —        | —       | —          | —               |
| 4       | —        | —       | —          | —               |
| 8       | —        | —       | —          | —               |

### Weak Scaling

| Threads | Problem Size | Time (s) | Efficiency |
|---------|-------------|----------|------------|
| 1       | N           | —        | 100%       |
| 2       | 2N          | —        | —          |
| 4       | 4N          | —        | —          |
| 8       | 8N          | —        | —          |

---

_Results to be populated by running benchmarks on the reference hardware._
_See [BENCHMARK_METHODOLOGY.md](BENCHMARK_METHODOLOGY.md) for methodology details._
