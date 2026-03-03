# LibAccInt Python Bindings

Python bindings for LibAccInt, providing a Pythonic interface for molecular integral computation with NumPy integration.

## Installation

### From source (development)

```bash
# Configure with Python bindings enabled
cmake --preset cpu-release -DLIBACCINT_BUILD_PYTHON=ON

# Build
cmake --build --preset cpu-release

# Install in editable mode
cd python
pip install -e .
```

### Verify installation

```python
import libaccint
print(libaccint.__version__)
```

## Quick Start

```python
import numpy as np
import libaccint

# Create atoms (H2O)
atoms = [
    libaccint.Atom(8, [0.0, 0.0, 0.0]),           # O at origin
    libaccint.Atom(1, [0.0, 1.43233673, -1.10866041]),   # H1
    libaccint.Atom(1, [0.0, -1.43233673, -1.10866041]),  # H2
]

# Create basis set
basis = libaccint.basis_set("sto-3g", atoms)

# Create integral engine
engine = libaccint.Engine(basis)

# Compute overlap matrix
S = engine.compute_overlap_matrix()
print(f"Overlap matrix shape: {S.shape}")

# Compute kinetic energy matrix
T = engine.compute_kinetic_matrix()

# Compute nuclear attraction matrix
V = engine.compute_nuclear_matrix(atoms)

# Compute core Hamiltonian
H = T + V
```

## API Overview

### Core Types

- `Atom`: Atomic center with atomic number and position
- `Shell`: Contracted Gaussian shell
- `BasisSet`: Collection of shells organized for computation
- `Engine`: Main computation orchestrator

### Operators

- `Operator.overlap()`: Overlap operator (S)
- `Operator.kinetic()`: Kinetic energy operator (T)
- `Operator.nuclear(charges)`: Nuclear attraction operator (V)
- `Operator.coulomb()`: Two-electron Coulomb operator

### Convenience Functions

- `libaccint.basis_set(name, atoms)`: Create a basis set
- `libaccint.compute_overlap(basis)`: Compute overlap matrix
- `libaccint.compute_kinetic(basis)`: Compute kinetic matrix
- `libaccint.compute_nuclear(basis, atoms)`: Compute nuclear matrix
- `libaccint.build_fock(engine, density)`: Build Fock matrix

## License

MIT License - see LICENSE file in the repository root.
