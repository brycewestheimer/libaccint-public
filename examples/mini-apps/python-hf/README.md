# Python Hartree-Fock Mini-Application

This mini-app demonstrates closed-shell RHF with LibAccInt Python bindings using
**ShellSetPair/ShellSetQuartet work units**.

It supports two explicit two-electron workflows:

- `buffer` mode: `ShellSetQuartet -> IntegralBuffer`, then host-side J/K build
- `consumer` mode: `ShellSetQuartet -> FockBuilder` accumulation
- `compare` mode: run both and print max |delta|

It does not use `compute_eri_tensor()`.

## Requirements

- Python >= 3.9
- NumPy
- LibAccInt Python bindings (`-DLIBACCINT_BUILD_PYTHON=ON`)

## Build And Install

### Option A: `venv` (recommended)

```bash
cd /path/to/libaccint
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy
```

Build libaccint + Python bindings:

```bash
cd /path/to/libaccint
cmake --preset cpu-release -DLIBACCINT_BUILD_PYTHON=ON -DLIBACCINT_BUILD_TESTS=OFF
cmake --build --preset cpu-release
python -m pip install -e ./python
python -c "import libaccint; print(libaccint.__version__)"
```

### Option B: conda

```bash
cd /path/to/libaccint
conda create -n libaccint-dev python=3.11 -y
conda activate libaccint-dev
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy
cmake --preset cpu-release -DLIBACCINT_BUILD_PYTHON=ON -DLIBACCINT_BUILD_TESTS=OFF
cmake --build --preset cpu-release
python -m pip install -e ./python
python -c "import libaccint; print(libaccint.__version__)"
```

### Optional CUDA Build

If you want GPU routing in the Python mini-app:

```bash
cd /path/to/libaccint
cmake --preset cuda-release-safe
cmake --build --preset cuda-release-safe
python -m pip install -e ./python
```

`cuda-release-safe` is tuned for constrained environments (WSL2/process-limited)
by disabling generated registries and throttling heavy CUDA translation units.

## Usage

```bash
cd examples/mini-apps/python-hf
python hf.py --molecule h2o --basis sto-3g --two-e-mode buffer
```

### Common Options

- `--molecule`, `-m`: preset (`h2`, `h2o`) or inline geometry
- `--basis`, `-b`: basis name (example uses built-in `sto-3g`)
- `--charge`: molecular charge (default `0`)
- `--backend`: `auto|force-cpu|prefer-cpu|prefer-gpu|force-gpu`
- `--two-e-mode`: `buffer|consumer|compare`
- `--units`: `bohr|angstrom` for custom inline geometries

### Inline Geometry Example

```bash
python hf.py \
  --molecule "H 0 0 0; H 0 0 1.4" \
  --basis sto-3g \
  --two-e-mode compare
```

### Suggested Test Runs

CPU validation:

```bash
cd /path/to/libaccint/examples/mini-apps/python-hf
python hf.py --molecule h2 --basis sto-3g --backend force-cpu --two-e-mode buffer
python hf.py --molecule h2o --basis sto-3g --backend force-cpu --two-e-mode consumer
python hf.py --molecule h2o --basis sto-3g --backend force-cpu --two-e-mode compare
```

CUDA routing check (if built with CUDA and GPU is available):

```bash
cd /path/to/libaccint/examples/mini-apps/python-hf
python hf.py --molecule h2o --basis sto-3g --backend prefer-gpu --two-e-mode buffer
python hf.py --molecule h2o --basis sto-3g --backend prefer-gpu --two-e-mode consumer
```

## Notes

- Coordinates are interpreted as Bohr unless `--units angstrom` is used.
- The mini-app is RHF-only (closed-shell, even electron count).
