# Frequently Asked Questions (FAQ)

## Version: 0.1.0-alpha.2

---

## General

### What is LibAccInt?

LibAccInt (**Lib**rary for **Acc**elerated **Int**egrals) is a high-performance molecular integral library for computational quantum chemistry. It provides:

- Overlap, kinetic, nuclear attraction, and electron repulsion integrals
- GPU acceleration via CUDA
- Built-in integral screening (Schwarz)
- Density fitting approximation
- Fused compute-and-consume pattern for memory efficiency
- Primary `Engine::compute(...)` API with ShellSet-based batching

### What basis sets are supported?

- **Built-in**: STO-3G (hardcoded for H, C, N, O, F)
- **BSE JSON**: Any basis set from the [Basis Set Exchange](https://www.basissetexchange.org/) can be loaded via JSON files
- **Bundled**: STO-3G, cc-pVDZ, cc-pVTZ in `share/basis_sets/`
- **Custom**: Construct `Shell` objects manually for any basis

### What angular momentum is supported?

Stable support in this cycle is angular momentum up to l=4 (G functions), covering:
- s (l=0), p (l=1), d (l=2), f (l=3), g (l=4)

Attempts to configure higher AM values fail fast at build boundaries.

### What operators are available?

| Operator | API | Type |
|----------|-----|------|
| Overlap | `Operator::overlap()` | 1-electron |
| Kinetic energy | `Operator::kinetic()` | 1-electron |
| Nuclear attraction | `Operator::nuclear(charges)` | 1-electron |
| Coulomb (ERI) | `Operator::coulomb()` | 2-electron |
| Range-separated (erf) | `Operator::erf_coulomb(ω)` | 2-electron |
| Range-separated (erfc) | `Operator::erfc_coulomb(ω)` | 2-electron |
| Point charges | `Operator::point_charges(charges)` | 1-electron |

---

## Building

### What compilers are supported?

- GCC 12+ (recommended)
- Clang 16+
- Apple Clang 15+ (macOS, CPU only)
- MSVC 2022+ (experimental)
- nvcc 12+ (for CUDA)

### How do I build without GPU support?

```bash
cmake --preset cpu-release
cmake --build --preset cpu-release
```

### How do I build with CUDA support?

```bash
cmake --preset cuda-release
cmake --build --preset cuda-release
```

Requires CUDA Toolkit 12.0+ installed and `nvcc` in PATH.

### Build fails with "C++20 features not supported"

Ensure your compiler supports C++20. With GCC, use version 12 or later:
```bash
sudo apt install g++-12
cmake -B build -DCMAKE_CXX_COMPILER=g++-12
```

### CMake can't find LibAccInt after installation

Ensure the install prefix is in your CMAKE_PREFIX_PATH:
```bash
cmake -B build -DCMAKE_PREFIX_PATH=/usr/local
```

Or use `find_package` with the correct config path:
```cmake
find_package(LibAccInt REQUIRED CONFIG)
```

---

## Usage

### How do I do a basic HF calculation?

See `examples/basic_hf_energy.cpp` for a complete example. The basic steps:

1. Define atoms and create a basis set
2. Create an `Engine`
3. Compute S, T, V matrices
4. Use `FockBuilder` for two-electron integrals
5. Build the Fock matrix: F = H + J - 0.5K

### Why not store the full ERI tensor?

The four-index ERI tensor scales as O(N⁴) in memory. For 100 basis functions, that's 800 MB. For 1000 basis functions, it's 8 TB. The compute-and-consume pattern keeps memory at O(N²) by processing integrals on-the-fly.

### How do I use OpenMP parallelism?

`compute_and_consume_parallel(...)` is the CPU/OpenMP path:

```cpp
consumers::FockBuilder fock(nbf);
fock.set_threading_strategy(consumers::FockThreadingStrategy::ThreadLocal);
fock.set_density(D.data(), nbf);
fock.prepare_parallel();
engine.compute_and_consume_parallel(Operator::coulomb(), fock);
fock.finalize_parallel();
```

Set the thread count via `OMP_NUM_THREADS` environment variable or pass explicitly:
```cpp
engine.compute_and_consume_parallel(Operator::coulomb(), fock, 8);
```

For shared-instance GPU concurrency and MPI/threading limits, see
`docs/api/THREAD_SAFETY.md`.

### How do I implement a custom consumer?

Create a class with an `accumulate` method matching this signature:

```cpp
class MyConsumer {
public:
    void accumulate(const TwoElectronBuffer<0>& buffer,
                    Index fa, Index fb, Index fc, Index fd,
                    int na, int nb, int nc, int nd) {
        // Process integrals here
        for (int a = 0; a < na; ++a)
            for (int b = 0; b < nb; ++b)
                for (int c = 0; c < nc; ++c)
                    for (int d = 0; d < nd; ++d) {
                        Real val = buffer(a, b, c, d);
                        // Your accumulation logic
                    }
    }
};
```

See `examples/custom_consumer.cpp` for complete examples.

### How do I compute gradients?

Gradient and Hessian APIs are not part of `0.1.0-alpha.2`. The alpha release
focuses on energy-level one- and two-electron integrals through the primary
`Engine::compute(...)` and `compute_and_consume(...)` interfaces.

### How do I use density fitting?

```cpp
consumers::DFFockBuilder df_fock(orbital_basis, auxiliary_basis);
df_fock.set_density(D.data(), nbf);
df_fock.compute();
auto J = df_fock.get_coulomb_matrix();
```

See `examples/density_fitting.cpp` for details.

---

## Performance

### My calculation is slow. What should I check?

1. **Enable screening**: `ScreeningOptions::normal()` can skip >90% of quartets
2. **Use parallelism**: `compute_and_consume_parallel()` with ThreadLocal strategy
3. **Try GPU**: `BackendHint::PreferGPU` for large systems
4. **Use density fitting**: `DFFockBuilder` reduces scaling from O(N⁴) to O(N²·N_aux)
5. **Check vectorization**: Ensure AVX2/AVX-512 is being used (`vector_isa()`)

### GPU is slower than CPU for my system

This is expected for small systems (< ~50 basis functions). GPU overhead (data transfer, kernel launch) dominates for small problems. Use `BackendHint::Auto` to let LibAccInt decide.

### How much memory does LibAccInt use?

| Component | Memory |
|-----------|--------|
| BasisSet | ~100 bytes per shell |
| S/T/V/J/K matrices | 8·N² bytes each |
| FockBuilder | ~24·N² bytes |
| DF B tensor | 8·N²·N_aux bytes |
| Schwarz bounds | 8·n_shells² bytes |

---

## Accuracy

### What numerical precision do the integrals have?

Integrals are computed in double precision (64-bit). Expected accuracy:
- One-electron integrals: ~1e-15 relative error
- Two-electron integrals: ~1e-14 relative error
- GPU results match CPU to ~1e-12

### Are GPU results identical to CPU results?

Not bit-for-bit identical due to floating-point non-associativity, but they agree to ~1e-12 or better, which is within the precision requirement for quantum chemistry.

### Does screening affect accuracy?

With `ScreeningPreset::Normal` (threshold 1e-12), the total energy error from screening is typically < 1e-10 Hartree — well below chemical accuracy (1e-6 Hartree).

---

## Troubleshooting

### "Invalid argument: unsupported element Z=..."

The built-in STO-3G basis only supports H, C, N, O, F. For other elements, load a BSE JSON file:
```cpp
auto basis = data::load_basis_set("cc-pvdz", atoms);
```

### Segfault when accessing BasisSet

Ensure the `BasisSet` outlives the `Engine`. The Engine stores a reference to the BasisSet.

### "Not implemented: ..." exception

Some features are documented but not yet implemented. Check the error message for which phase will add the feature.

### Link errors with "undefined reference to libaccint..."

Ensure you're linking against the library:
```cmake
target_link_libraries(my_app PRIVATE LibAccInt::libaccint)
```

Or via pkg-config:
```bash
g++ my_app.cpp $(pkg-config --cflags --libs libaccint) -o my_app
```
