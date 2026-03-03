# Migration Guide: Shell-Level Loops → ShellSet Work Units

LibAccInt v0.1.0-alpha.2 introduces **ShellSet work units** — a batched abstraction that replaces
hand-written shell loops with a higher-level API. This guide shows how to migrate
existing code from explicit O(N²) / O(N⁴) shell loops to the new ShellSet-based
compute interface.

## Why Migrate?

| Shell-level loops | ShellSet work units |
|---|---|
| Manual N² or N⁴ index iteration | Iterator / full-basis one-liner |
| Hand-rolled buffer → matrix copy | Engine handles accumulation |
| No batching; one integral at a time | Shells grouped by (AM, K) for SIMD/GPU batching |
| Screening must be coded manually | Worklist filtering with `std::span` |

ShellSet grouping enables **vectorised recurrence** on CPU and **warp-level batching**
on GPU — gains that are impossible when the caller drives one shell pair at a time.

---

## One-Electron Integrals

### Before — Shell-Level N² Loop

```cpp
#include <libaccint/libaccint.hpp>
using namespace libaccint;

Size n_shells = basis.n_shells();
Size nbf = basis.n_basis_functions();
std::vector<Real> S(nbf * nbf, 0.0);
OneElectronBuffer<0> buffer;

for (Size i = 0; i < n_shells; ++i) {
    for (Size j = i; j < n_shells; ++j) {
        engine.compute(Operator::overlap(), basis.shell(i), basis.shell(j), buffer);
        // ... manually copy buffer into S with index math ...
    }
}
```

Problems:

* You must track shell offsets and copy from `buffer` into `S` yourself.
* No batching — each `compute` call processes a single (i, j) pair.

### After — ShellSet Work Units

**Option A — Explicit pair iteration:**

```cpp
#include <libaccint/libaccint.hpp>
using namespace libaccint;

Size nbf = basis.n_basis_functions();
std::vector<Real> S(nbf * nbf, 0.0);

for (const auto& pair : basis.shell_set_pairs()) {
    engine.compute(Operator::overlap(), pair, S);
}
```

**Option B — Full-basis one-liner:**

```cpp
std::vector<Real> S(nbf * nbf, 0.0);
engine.compute(OneElectronOperator(Operator::overlap()), S);
```

The primary `engine.compute(...)` API iterates all shell-set pairs internally,
preserves batched execution semantics, accumulates into `S`, and handles symmetry.

---

## Two-Electron Integrals (Fock Build)

### Before — Shell-Level N⁴ Loop

```cpp
#include <libaccint/libaccint.hpp>
using namespace libaccint;

TwoElectronBuffer<0> buffer;
for (Size i = 0; i < n_shells; ++i) {
    for (Size j = 0; j < n_shells; ++j) {
        for (Size k = 0; k < n_shells; ++k) {
            for (Size l = 0; l < n_shells; ++l) {
                engine.compute(Operator::coulomb(),
                    basis.shell(i), basis.shell(j),
                    basis.shell(k), basis.shell(l), buffer);
                fock.accumulate(buffer, ...);
            }
        }
    }
}
```

Problems:

* Four nested loops with O(N⁴) iterations.
* No screening — every quartet is computed.
* Manual Fock accumulation logic.

### After — ShellSet Work Units

**Option 1 — Full-basis convenience (simplest):**

```cpp
#include <libaccint/libaccint.hpp>
using namespace libaccint;

consumers::FockBuilder fock(nbf);
fock.set_density(D.data(), nbf);

engine.compute(Operator::coulomb(), fock);
```

The engine generates the worklist, iterates all quartets internally, and
accumulates Coulomb + exchange contributions into the Fock matrix via the
`FockBuilder` consumer.

**Option 2 — Explicit quartet iteration:**

```cpp
consumers::FockBuilder fock(nbf);
fock.set_density(D.data(), nbf);

for (const auto& quartet : basis.shell_set_quartets()) {
    engine.compute(Operator::coulomb(), quartet, fock);
}
```

Useful when you want to add per-quartet logic (logging, timing, custom
screening) while still using batched compute.

**Option 3 — Custom worklist (screening / partitioning):**

```cpp
consumers::FockBuilder fock(nbf);
fock.set_density(D.data(), nbf);

auto quartets = basis.shell_set_quartets();
// ... filter quartets (Schwarz screening, density screening, etc.) ...
engine.compute(Operator::coulomb(), std::span(quartets), fock);
```

Pass any contiguous range of `ShellSetQuartet` objects as a `std::span`.
This is the preferred pattern for integral-direct SCF with screening.

---

## API Quick Reference

### Engine `compute` Overloads

| Signature | Use case |
|---|---|
| `compute(Operator, ShellSetPair, vector<Real>&)` | Batched 1e: one pair at a time |
| `compute(OneElectronOperator, vector<Real>&)` | Full-basis 1e (one-liner) |
| `compute(Operator, ShellSetQuartet, Consumer&)` | Batched 2e: one quartet at a time |
| `compute(Operator, Consumer&)` | Full-basis 2e (one-liner) |
| `compute(Operator, span<const ShellSetQuartet>, Consumer&)` | 2e with custom worklist |

### Key Types

| Type | Definition |
|---|---|
| `Real` | `double` |
| `Size` | `std::size_t` |
| `Index` | `std::int64_t` |
| `consumers::FockBuilder` | Fock-matrix consumer (namespace `libaccint::consumers`) |

---

## Performance Guidance

1. **Prefer the full-basis one-liner** (`engine.compute(op, consumer)`) unless
   you need custom quartet ordering or screening. It gives the engine maximum
   freedom to batch and schedule work.

2. **ShellSet grouping** clusters shells by angular momentum (AM) and
   contraction degree (K). This enables:
   - SIMD-width recurrence on CPU (AVX-512, NEON, …)
   - Warp-level batching on GPU (CUDA)

3. **Worklists are cached.** `basis.shell_set_pairs()` and
   `basis.shell_set_quartets()` only allocate on the first call; subsequent
   calls return a reference to the cached list.

4. **Screening with worklists.** Copy the cached list, remove insignificant
   quartets (Schwarz / density screening), and pass the filtered
   `std::span` to `compute`. This avoids rebuilding the grouping structure.

---

## Parallel Fock Builds

For multi-threaded integral-direct SCF, `FockBuilder` supports two threading
strategies:

```cpp
// Atomic accumulation (lock-free, good for small basis sets)
consumers::FockBuilder fock(nbf, FockThreadingStrategy::Atomic);

// Thread-local accumulation (better scaling for large basis sets)
consumers::FockBuilder fock(nbf, FockThreadingStrategy::ThreadLocal);
```

Parallel usage pattern:

```cpp
fock.prepare_parallel();   // allocate thread-local buffers (if ThreadLocal)

#pragma omp parallel for schedule(dynamic)
for (Size q = 0; q < quartets.size(); ++q) {
    engine.compute(Operator::coulomb(), quartets[q], fock);
}

fock.finalize_parallel();  // reduce thread-local buffers into final F
```

---

## Troubleshooting

### Deprecated generators

`generate_shell_set_pairs()` and `generate_shell_set_quartets()` are
**deprecated**. Use the cached accessors instead:

```cpp
// ✗ deprecated — allocates every time
auto pairs = basis.generate_shell_set_pairs();

// ✓ preferred — cached, returns a const reference
const auto& pairs = basis.shell_set_pairs();
```

### Consumer interface requirements

A two-electron consumer must provide:

```cpp
void accumulate(const TwoElectronBuffer<0>& buffer,
                Index i, Index j, Index k, Index l,
                int ni, int nj, int nk, int nl);
```

The engine extracts shell indices and basis-function counts from the
`ShellSetQuartet` and passes them to `accumulate` automatically — you never
need to compute offsets yourself.

### Matrix ordering

One-electron matrices (overlap, kinetic, nuclear attraction) are accumulated in
**row-major** order into a flat `std::vector<Real>` of size `nbf × nbf`.
Access element (μ, ν) at index `μ * nbf + ν`.
