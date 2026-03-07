# LibAccInt

**Lib**rary for **Acc**elerated **Int**egrals — A high-performance molecular integral library for computational quantum chemistry.

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Version](https://img.shields.io/badge/version-0.1.0--alpha.2-orange.svg)](https://github.com/brycewestheimer/libaccint-public/releases)
[![CPU CI](https://github.com/brycewestheimer/libaccint-public/actions/workflows/cpu-ci.yml/badge.svg)](https://github.com/brycewestheimer/libaccint-public/actions/workflows/cpu-ci.yml)
[![CUDA CI](https://github.com/brycewestheimer/libaccint-public/actions/workflows/cuda-ci.yml/badge.svg)](https://github.com/brycewestheimer/libaccint-public/actions/workflows/cuda-ci.yml)
[![Quality Gate](https://github.com/brycewestheimer/libaccint-public/actions/workflows/quality-gate.yml/badge.svg)](https://github.com/brycewestheimer/libaccint-public/actions/workflows/quality-gate.yml)
[![Coverage](https://github.com/brycewestheimer/libaccint-public/actions/workflows/coverage.yml/badge.svg)](https://github.com/brycewestheimer/libaccint-public/actions/workflows/coverage.yml)

> **Alpha Release (v0.1.0-alpha.2):** This alpha release is primarily intended
> to expose the LibAccInt API and demonstrate basic usage patterns, including the
> shift from individual shell pairs/quartets to ShellSet-based batches as work
> units. This alpha is released under the BSD 3-Clause License; the final
> release will be relicensed under the MIT License. High-performance
> AM-specialized generated kernels will be available in a future public release.

## Overview

LibAccInt computes one- and two-electron molecular integrals over Cartesian Gaussian basis functions using Rys quadrature, targeting Hartree-Fock and post-HF methods in quantum chemistry. The primary API is `Engine::compute(...)`, which routes full-basis and ShellSet work units through CPU or GPU backends while preserving batched execution inside each `ShellSetPair` or `ShellSetQuartet`.

## Features

- **One-electron integrals** — Overlap (S), kinetic energy (T), nuclear attraction (V), core Hamiltonian (H = T + V)
- **Two-electron integrals** — Electron repulsion integrals (ERI) via Rys quadrature
- **Angular momentum** — stable contract support through G (l<=4) across CPU/CUDA and spherical transforms
- **CPU backend** — OpenMP parallelism with SIMD vectorization
- **GPU backend** — CUDA batched kernels with device-side Fock accumulation on a single GPU; concurrent multi-stream execution via slot pool
- **Primary APIs** — `Engine::compute(...)`, `Engine::compute_and_consume(...)`
- **Consumer APIs** — `FockBuilder` (CPU), `GpuFockBuilder` (CUDA)
- **Secondary convenience APIs** — `compute_overlap_matrix()`, `compute_kinetic_matrix()`, `compute_nuclear_matrix()`, `compute_eri_tensor()`, `compute_eri_block()`
- **Advanced APIs** — ShellSet work-unit overloads, `compute_batch_parallel()`, `compute_quartet()`, `compute_pair()`
- **Schwarz screening** — `compute_batch_screened()`, `compute_screening_statistics()`
- **Auto-tuning dispatch** — Intelligent CPU/GPU routing based on analytical cost models
- **Python bindings** — pybind11 with NumPy integration; one-liner access to matrices and tensors
- **Basis data** — built-in `sto-3g` plus 28 bundled Basis Set Exchange JSON files loadable via `load_basis_set()`

## Quick Start (C++)

```cpp
#include <libaccint/libaccint.hpp>
using namespace libaccint;

int main() {
    // Define H₂O molecule (Bohr coordinates)
    std::vector<data::Atom> atoms = {
        {8, {0.0, 0.0, 0.0}},               // O
        {1, {0.0, 1.43233673, -1.10866041}}, // H
        {1, {0.0, -1.43233673, -1.10866041}} // H
    };

    auto basis = data::create_builtin_basis("sto-3g", atoms);
    Engine engine(basis);

    // One-electron matrices through the primary API
    std::vector<Real> S;
    engine.compute(OneElectronOperator(Operator::overlap()), S);

    // Fock matrix build with the primary consumer API
    Size nbf = basis.n_basis_functions();
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0;
    }
    consumers::FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);
    engine.compute(Operator::coulomb(), fock);

    auto J = fock.get_coulomb_matrix();
    auto K = fock.get_exchange_matrix();
    return 0;
}
```

## Quick Start (Python)

```python
import libaccint
import numpy as np

atoms = [
    libaccint.Atom(8, [0.0, 0.0, 0.0]),
    libaccint.Atom(1, [0.0, 1.43233673, -1.10866041]),
    libaccint.Atom(1, [0.0, -1.43233673, -1.10866041]),
]

basis = libaccint.basis_set("sto-3g", atoms)
engine = libaccint.Engine(basis)

S = engine.compute_overlap_matrix()        # primary matrix helper
T = engine.compute_kinetic_matrix()
H = engine.compute_core_hamiltonian(atoms)
```

See `examples/python/` and `examples/mini-apps/python-hf/` for complete Hartree-Fock programs.

## Installation

### Building from Source

**Prerequisites:** CMake 3.25+, C++20 compiler (GCC 11+ / Clang 14+). Developer builds can fetch some third-party dependencies automatically; release/package builds require preinstalled `nlohmann_json` and `pybind11`.

```bash
# CPU Release build (recommended)
cmake --preset cpu-release
cmake --build --preset cpu-release

# Run tests
ctest --test-dir build/cpu-release

# Install
cmake --install build/cpu-release --prefix /usr/local
```

### Python Bindings

```bash
# Via CMake
cmake --preset cpu-release -DLIBACCINT_BUILD_PYTHON=ON
cmake --build --preset cpu-release

# Or via pip (editable install)
cd python && pip install -e .
```

### CMake Presets

| Preset | Description |
|--------|-------------|
| `cpu-debug` | CPU build with debug symbols |
| `cpu-release` | CPU build with optimizations |
| `cuda-release` | CUDA + CPU build (requires CUDA Toolkit) |
| `cuda-release-safe` | CUDA + Python safe build for low-process/low-memory environments (tests/examples off, generated registries off, jobs=1) |
| `cuda-release-safe-mpi-tests` | WSL2-safe CUDA + MPI test build using system OpenMPI and capped build/test parallelism |

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `LIBACCINT_USE_CUDA` | AUTO | Enable CUDA backend (AUTO/ON/OFF) |
| `LIBACCINT_USE_MPI` | AUTO | Enable MPI backend (AUTO/ON/OFF) |
| `LIBACCINT_BUILD_TESTS` | ON | Build test suite |
| `LIBACCINT_BUILD_EXAMPLES` | ON | Build examples |
| `LIBACCINT_USE_OPENMP` | ON | Enable OpenMP for CPU parallelization |
| `LIBACCINT_ENABLE_CPU_GENERATED_REGISTRY` | OFF | Reserved for future release (generated kernels not included in alpha) |
| `LIBACCINT_ENABLE_CUDA_GENERATED_REGISTRY` | OFF | Reserved for future release (generated kernels not included in alpha) |
| `LIBACCINT_CUDA_HEAVY_TU_COMPILE_JOBS` | 1 | Ninja job-pool cap for heavy CUDA TUs (`eri_kernel*`, CUDA registry); set `0` to disable throttling |
| `LIBACCINT_CUDA_HEAVY_TU_OPT_LEVEL` | 3 | Optimization override for heavy CUDA TUs only (`0..3`); lower values reduce compile-time pressure |
| `LIBACCINT_BUILD_PYTHON` | OFF | Build Python bindings |
### Low-Memory CUDA/Python Build (WSL2-safe)

```bash
cmake --preset cuda-release-safe
cmake --build --preset cuda-release-safe
pip install -e ./python
```

This preset disables tests/examples and generated registries, and forces low
concurrency for heavy CUDA translation units to reduce process/memory spikes.

For WSL2 CUDA and MPI testing on this host class, use an explicit CUDA
compiler and the system OpenMPI toolchain. The first CUDA test on an RTX 5070
may spend tens of seconds in PTX JIT warmup before subsequent tests run
normally.

For repeatable local validation, use:

```bash
cmake --preset cuda-release-safe-mpi-tests
cmake --build --preset cuda-release-safe-mpi-tests
ctest --preset cuda-release-safe-mpi-tests
```

To build the deterministic source archive used by release packaging:

```bash
./packaging/release/build_source_release.sh
```

### Spack Build

LibAccInt ships an in-tree Spack repository in `packaging/spack`.

```bash
spack repo add ./packaging/spack
spack install libaccint@main +openmp
spack install libaccint@main +mpi
spack install libaccint@main +cuda cuda_arch=90
```

For local development against the current checkout:

```bash
spack repo add ./packaging/spack
spack dev-build -d . libaccint@main +openmp
```

The Spack recipe disables CMake `FetchContent`, enables relocatable installs,
and expects all dependencies to come from Spack. Release-package hashes are
based on the deterministic source archive uploaded by the release workflow:
`libaccint-${version}.tar.gz`.

### Release Scope Notes

- Stable angular-momentum contract for this cycle is G-only (`l<=4`) across CPU/CUDA and spherical transforms.
- Gradient and Hessian integrals are planned for a future release.
- Single-GPU execution is the supported GPU scope for the alpha release.
- `MultiGPUEngine`, `MultiGPUFockBuilder`, `CudaEngine::compute_eri_pipelined()`, and `CudaEngine::compute_eri_device_scatter()` are experimental.

## Basis Sets

LibAccInt currently provides one built-in basis (`sto-3g`) via `create_builtin_basis()`.
For broader coverage, 28 Basis Set Exchange JSON basis files are bundled and loadable by name via `load_basis_set()`.

| Category | Basis Sets |
|----------|------------|
| **Pople** | STO-3G, 3-21G, 6-31G, 6-31G\*, 6-31G\*\*, 6-31+G\*, 6-31++G\*\*, 6-311G, 6-311G\*, 6-311G\*\*, 6-311+G\*, 6-311+G\*\*, 6-311++G\*\* |
| **Dunning** | cc-pVDZ, cc-pVTZ, cc-pVQZ, cc-pV5Z, aug-cc-pVDZ, aug-cc-pVTZ, aug-cc-pVQZ, aug-cc-pV5Z |
| **Auxiliary (JKFIT)** | cc-pVTZ-JKFIT, cc-pVQZ-JKFIT, cc-pV5Z-JKFIT |
| **Auxiliary (RIFIT)** | cc-pVDZ-RIFIT, cc-pVTZ-RIFIT, cc-pVQZ-RIFIT, cc-pV5Z-RIFIT |

```cpp
// Built-in basis set (currently STO-3G)
auto basis = data::create_builtin_basis("sto-3g", atoms);

// Load bundled Basis Set Exchange JSON by name
auto basis2 = data::load_basis_set("cc-pvdz", atoms);
```

Basis set data sourced from the [Basis Set Exchange](https://www.basissetexchange.org/). See [share/basis_sets/ATTRIBUTION.md](share/basis_sets/ATTRIBUTION.md) for full attribution.

## API Overview

| Component | Key Methods / Purpose |
|-----------|----------------------|
| **Engine** | Central orchestrator — routes to CPU or GPU backend |
| **Operators** | `Operator::overlap()`, `kinetic()`, `nuclear()`, `coulomb()` |
| **FockBuilder** | CPU consumer — accumulates J and K matrices during integral computation |
| **GpuFockBuilder** | CUDA consumer — device-side Fock accumulation without host transfers |
| `compute_eri_tensor()` | Full (μν\|λσ) tensor — for small basis sets |
| `compute_eri_block()` | Block of ERIs for a given shell quartet range |
| `compute_batch_parallel()` | OpenMP-parallel batched integral evaluation |
| `compute_all_2e_parallel()` | Parallel computation over all unique shell quartets |
| `compute_batch_screened()` | Schwarz-screened batched computation |
| **DispatchPolicy** | Auto-tuning CPU/GPU routing with cost model |
| `DispatchConfig::n_gpu_slots` | Concurrent GPU execution slots (default: 4) |

### Concurrent GPU Execution

For alpha, shared-instance GPU concurrency is supported for the batched,
slot-backed APIs such as `compute_batch_parallel()`,
`compute_all_2e_parallel()`, `compute_shell_set_pair()`, and batched
`compute_batch(...)` calls routed to GPU. The underlying `CudaEngine`
manages a pool of GPU execution slots, each with its own CUDA stream and
device buffers:

```cpp
DispatchConfig config;
config.n_gpu_slots = 4;  // 4 concurrent GPU streams (reduce for small GPUs)
Engine engine(basis, config);

// Multi-threaded GPU computation over the supported batched GPU path
auto results = engine.compute_batch_parallel(
    Operator::coulomb(), basis.shell_set_quartets(),
    /*n_threads=*/4, BackendHint::ForceGPU);
```

Do not mutate dispatch/device configuration, screening inputs, or thread
configuration while concurrent GPU work is active. The authoritative alpha
support matrix is in `docs/api/THREAD_SAFETY.md`.

Full API documentation is available in `docs/`.

## Examples

| Example | Description |
|---------|-------------|
| `basic_hf_energy.cpp` | Minimal Hartree-Fock energy calculation |
| `hf_calculation.cpp` | Full HF with SCF convergence |
| `shellset_batched_1e.cpp` | Batched one-electron integral evaluation |
| `shellset_batched_fock.cpp` | Fock matrix build via compute-and-consume |
| `gpu_fock_build.cpp` | Device-side Fock accumulation (CUDA) |
| `gpu_hf_workflow.cpp` | Complete GPU-accelerated HF workflow |
| `density_fitting.cpp` | Density-fitted integrals |
| `custom_consumer.cpp` | Writing a custom integral consumer |
| `qmm_embedding.cpp` | QM/MM-style embedding |
| `hf_calculation.py` | Python HF calculation |

Complete mini-app HF programs (both C++ and Python) are in `examples/mini-apps/`.

## Documentation

The `docs/` directory contains full Sphinx + Doxygen documentation covering API reference, theory, user guide, and developer guide. Build with:

```bash
cd docs && make html
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code style, testing, and the pull request process.

## License

This alpha release is licensed under the **BSD 3-Clause License** — see [LICENSE](LICENSE) for details.
The final release of LibAccInt will be relicensed under the **MIT License**.

Copyright (c) 2026 Bryce M. Westheimer

## Citation

If you use LibAccInt in your research, please cite:

```bibtex
@software{libaccint2026,
  title  = {LibAccInt: Library for Accelerated Integrals},
  author = {Bryce M. Westheimer},
  year   = {2026},
  url    = {https://github.com/brycewestheimer/libaccint-public},
  note   = {v0.1.0-alpha.2}
}
```

**Acknowledgments:** Rys quadrature implementation based on methods from [libint](https://github.com/evaleev/libint). Inspired by [libcint](https://github.com/sunqm/libcint) and [GPU4PySCF](https://github.com/pyscf/gpu4pyscf). Basis set data from the [Basis Set Exchange](https://www.basissetexchange.org/).

## Planned Extensions

- Gradient and Hessian (first and second derivative) integrals
- AM-specialized generated kernels via code generation framework
- Higher angular momentum beyond G (h, i and above)
- Spherical harmonics (solid harmonics transformation)
- Density fitting / resolution of the identity (RI)
- Multi-GPU support
