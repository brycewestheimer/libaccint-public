# Performance Guidelines

## Version: 0.1.0-alpha.2

This guide covers best practices for achieving optimal performance with LibAccInt across different hardware configurations and problem sizes.

---

## 1. General Principles

### Choose the Right Abstraction Level

LibAccInt provides three levels of integral computation, from highest to lowest level:

| Level | API | Best For |
|-------|-----|----------|
| Primary | `engine.compute(op, consumer)` / `engine.compute(op, result)` | Standard SCF and matrix builds |
| Fine-grained batched | `engine.compute(op, pair, result)` / `engine.compute(op, quartet, consumer)` | Custom iteration order, screening |
| Advanced low-level | `engine.compute(op, shell_a, shell_b, buffer)` | Maximum control, custom algorithms |

**Recommendation**: Use the primary `Engine::compute(...)` and `compute_and_consume(...)` entry points unless you need custom worklists. Those are the alpha-promised APIs and preserve ShellSet batching internally.

### Compute-and-Consume Pattern

Always use `FockBuilder` or custom consumers instead of materializing the full ERI tensor:

```cpp
// GOOD: O(N²) memory via compute-and-consume
consumers::FockBuilder fock(nbf);
fock.set_density(D.data(), nbf);
engine.compute(Operator::coulomb(), fock);

// BAD: O(N⁴) memory for full tensor (do NOT do this for large systems)
// std::vector<Real> eri(nbf * nbf * nbf * nbf);
```

Memory savings: For 100 basis functions, full ERI = 800 MB vs FockBuilder = 160 KB.

---

## 2. CPU Performance

### OpenMP Parallelism

The supported CPU thread-parallel interface is the explicit `_parallel`
family, such as `compute_1e_parallel(...)` and
`compute_and_consume_parallel(...)`. These entry points use OpenMP-managed
parallelism; they are not a promise that arbitrary shared-instance `Engine`
calls are re-entrant.

```cpp
// Parallel Fock build (recommended for > 20 basis functions)
consumers::FockBuilder fock(nbf);
fock.set_threading_strategy(consumers::FockThreadingStrategy::ThreadLocal);
fock.set_density(D.data(), nbf);
fock.prepare_parallel();
engine.compute_and_consume_parallel(Operator::coulomb(), fock);
fock.finalize_parallel();
```

`compute_and_consume_parallel(...)` is CPU/OpenMP-only. For the broader alpha
threading contract, see `docs/api/THREAD_SAFETY.md`.

**Threading strategy comparison**:
| Strategy | Overhead | Scalability | Best For |
|----------|----------|-------------|----------|
| `Sequential` | None | N/A | Small systems, single-thread |
| `Atomic` | Low | Moderate | Few threads, large quartets |
| `ThreadLocal` | Memory | Excellent | Many threads, large systems |

### Vectorization

LibAccInt auto-detects and uses SIMD instructions. Check at runtime:

```cpp
std::cout << "Vector ISA: " << vector_isa() << "\n";
std::cout << "Vector width: " << vector_width() << "\n";
```

**Optimal hardware**: AVX2 (8-wide double) or AVX-512 (8-wide double) provides 2-4x speedup over scalar code for primitive contraction loops.

### Screening

Schwarz screening dramatically reduces computation for large systems:

```cpp
screening::ScreeningOptions opts = screening::ScreeningOptions::normal();
opts.density_weighted = true;  // Enable density screening

engine.compute_and_consume_screened_parallel(
    Operator::coulomb(), fock, opts);
```

**Expected speedup from screening by system size**:
| Basis Functions | Screening Speedup |
|----------------|-------------------|
| 50 | 1.5-2x |
| 200 | 5-10x |
| 1000 | 50-100x |
| 5000 | 500-1000x |

---

## 3. GPU Performance

### When to Use GPU

GPU acceleration is most beneficial for:
- Large basis sets (> 100 basis functions)
- High angular momentum (d, f functions)
- Many primitives per shell

For small systems (< 50 basis functions), CPU may be faster due to GPU launch overhead.

### Dispatch Hints

```cpp
Engine engine(basis);

// Let LibAccInt decide (recommended)
engine.compute(op, consumer, BackendHint::Auto);

// Force GPU for known-large work
engine.compute(op, consumer, BackendHint::PreferGPU);

// Force CPU (e.g., for small systems or debugging)
engine.compute(op, consumer, BackendHint::ForceCPU);
```

### GPU Batching

On GPU, the primary API batches work inside each `ShellSetPair` or `ShellSetQuartet` launch. The host still iterates work units, but the work inside each unit is not reduced to one shell pair or quartet at a time.

### GPU Memory Management

For large systems, monitor GPU memory:
- Each ShellSetQuartet batch requires O(batch_size × n_functions⁴) temporary GPU memory
- The Engine manages memory batching automatically
- Set `DispatchConfig::gpu_memory_limit` to control maximum GPU allocation

---

## 4. Density Fitting Performance

Density fitting (DF) trades a small accuracy loss for significant speedup:

| Operation | Conventional | Density Fitting |
|-----------|-------------|-----------------|
| Scaling | O(N⁴) | O(N² × N_aux) |
| Memory | O(N²) | O(N² × N_aux) |
| Typical speedup | — | 3-10x |

```cpp
// DF-HF configuration for optimal performance
consumers::DFFockBuilderConfig config;
config.compute_coulomb = true;
config.compute_exchange = true;
config.use_symmetry = true;     // Exploit permutational symmetry
config.memory_limit_mb = 4096;  // Limit B tensor memory
config.n_threads = 0;           // Auto-detect thread count

consumers::DFFockBuilder df_fock(orbital_basis, aux_basis, config);
```

### Choosing the Auxiliary Basis Set

| Orbital Basis | Recommended Auxiliary | N_aux / N_orb ratio |
|--------------|----------------------|---------------------|
| STO-3G | — (use conventional) | — |
| cc-pVDZ | cc-pVDZ-JKFIT | ~3x |
| cc-pVTZ | cc-pVTZ-JKFIT | ~3x |
| def2-SVP | def2-SVP-JKFIT | ~3x |

---

## 5. Memory Optimization

### Buffer Reuse

Reuse result vectors across iterations:

```cpp
std::vector<Real> S(nbf * nbf);
std::vector<Real> T(nbf * nbf);

// These reuse the same memory allocation
engine.compute_overlap_matrix(S);
engine.compute_kinetic_matrix(T);
```

### Memory Estimation

| Component | Memory | Formula |
|-----------|--------|---------|
| Overlap matrix | O(N²) | 8·N² bytes |
| Fock builder | O(N²) | 24·N² bytes (J, K, D) |
| Schwarz bounds | O(n_shells²) | 8·n_shells² bytes |
| DF B tensor | O(N²·N_aux) | 8·N²·N_aux bytes |

---

## 6. Benchmarking Tips

### Use LibAccInt's Benchmark Harness

```cpp
#include <libaccint/benchmark/benchmark_harness.hpp>

benchmark::BenchmarkHarness harness;
harness.add_system("H2O/STO-3G", water_basis);
harness.add_system("C6H6/cc-pVDZ", benzene_basis);
harness.run();
harness.report();
```

### Profiling Checklist

1. **Verify vectorization**: Check `vector_isa()` matches your hardware
2. **Check screening**: Monitor skipped quartets percentage
3. **Profile GPU dispatch**: Use `BackendHint::ForceCPU` vs `PreferGPU` to compare
4. **Thread scaling**: Measure with 1, 2, 4, 8, ... threads
5. **Memory pressure**: Monitor peak memory usage

### Common Performance Pitfalls

| Issue | Symptom | Fix |
|-------|---------|-----|
| No screening | Slow for large systems | Enable `ScreeningOptions::normal()` |
| Sequential FockBuilder | Poor parallel scaling | Use `ThreadLocal` strategy |
| GPU for small systems | Slower than CPU | Use `BackendHint::Auto` |
| Full ERI storage | Out of memory | Use compute-and-consume pattern |

---

## Experimental APIs

The following remain public for specialist users but are outside the alpha performance contract:

- `MultiGPUEngine`
- `MultiGPUFockBuilder`
- `CudaEngine::compute_eri_pipelined()`
- `CudaEngine::compute_eri_device_scatter()`
| Wrong auxiliary basis | DF errors too large | Use matched JKFIT basis |
| Excessive precision | Slow screening | Use `ScreeningPreset::Normal` |
