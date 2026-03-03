# C++ Hartree-Fock Mini-Application

This mini-app demonstrates closed-shell RHF using LibAccInt's **ShellSet work-unit** APIs.
It accepts molecule and basis arguments, computes one-electron integrals via
`ShellSetPair` loops, and supports two-electron builds via either:

- `buffer` mode: compute `ShellSetQuartet -> IntegralBuffer`, then build J/K on host
- `consumer` mode: compute `ShellSetQuartet -> FockBuilder` directly
- `compare` mode: run both and report max |delta|

It never uses `compute_eri_tensor()`.

## Build

```bash
cd examples/mini-apps/cpp-hf
cmake -B build
cmake --build build
```

For CUDA-enabled builds, configure LibAccInt with `-DLIBACCINT_USE_CUDA=ON`.

### Full Build Details

Prerequisites:

- CMake >= 3.25
- C++20 compiler (GCC 11+ or Clang 14+)
- Git and network access (CMake fetches third-party deps on first configure)

CPU-only build from repository root:

```bash
cd /path/to/libaccint
cmake -S examples/mini-apps/cpp-hf -B build/miniapps/cpp-hf-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DLIBACCINT_BUILD_TESTS=OFF \
  -DLIBACCINT_BUILD_EXAMPLES=OFF \
  -DLIBACCINT_BUILD_BENCHMARKS=OFF \
  -DLIBACCINT_USE_CUDA=OFF
cmake --build build/miniapps/cpp-hf-cpu --target hf_miniapp -j
```

CUDA build from repository root:

```bash
cd /path/to/libaccint
cmake -S examples/mini-apps/cpp-hf -B build/miniapps/cpp-hf-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DLIBACCINT_BUILD_TESTS=OFF \
  -DLIBACCINT_BUILD_EXAMPLES=OFF \
  -DLIBACCINT_BUILD_BENCHMARKS=OFF \
  -DLIBACCINT_USE_CUDA=ON
cmake --build build/miniapps/cpp-hf-cuda --target hf_miniapp -j
```

## Run

```bash
./build/hf_miniapp --molecule h2o --basis sto-3g --two-e-mode buffer
```

### Common Options

- `--molecule`, `-m`: preset (`h2`, `h2o`) or inline geometry
- `--basis`, `-b`: basis name or `.json` basis file path
- `--charge`: molecular charge (default `0`)
- `--backend`: `auto|force-cpu|prefer-cpu|prefer-gpu|force-gpu`
- `--two-e-mode`: `buffer|consumer|compare`
- `--units`: `bohr|angstrom` for custom inline geometries

### Inline Geometry Example

```bash
./build/hf_miniapp \
  --molecule "H 0 0 0; H 0 0 1.4" \
  --basis sto-3g \
  --two-e-mode compare
```

### Suggested Test Runs

CPU validation:

```bash
cd /path/to/libaccint
build/miniapps/cpp-hf-cpu/hf_miniapp --molecule h2 --basis sto-3g --backend force-cpu --two-e-mode buffer
build/miniapps/cpp-hf-cpu/hf_miniapp --molecule h2o --basis sto-3g --backend force-cpu --two-e-mode consumer
build/miniapps/cpp-hf-cpu/hf_miniapp --molecule h2o --basis sto-3g --backend force-cpu --two-e-mode compare
```

CUDA routing check (if built with CUDA and GPU is available):

```bash
cd /path/to/libaccint
build/miniapps/cpp-hf-cuda/hf_miniapp --molecule h2o --basis sto-3g --backend prefer-gpu --two-e-mode buffer
build/miniapps/cpp-hf-cuda/hf_miniapp --molecule h2o --basis sto-3g --backend prefer-gpu --two-e-mode consumer
```

## Notes

- Coordinates are interpreted as Bohr unless `--units angstrom` is used.
- The mini-app is RHF-only (closed-shell, even electron count).
