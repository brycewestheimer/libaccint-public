# Reference Data Generation Scripts

This directory contains scripts for generating reference data for LibAccInt:
1. **Mathematical reference data** (Boys function, Rys quadrature) - `generate_math_reference.py`
2. **Integral reference data** (using PySCF) - `generate_reference.py`

## Mathematical Reference Data (`generate_math_reference.py`)

### Requirements
- Python 3.7+
- mpmath (install with `pip install mpmath`)

### Usage

Generate high-precision reference data for Boys function and Rys quadrature:
```bash
python3 generate_math_reference.py
```

This creates:
- `tests/data/boys_reference.json` - Boys function F_n(T) reference values
- `tests/data/rys_reference.json` - Rys quadrature roots and weights

### Command-line Options
- `--output-dir PATH` - Output directory (default: `tests/data/`)
- `--precision N` - Decimal digits for mpmath (default: 50)
- `--n-boys-points N` - Number of T points for Boys function (default: 200)
- `--n-rys-points N` - Number of T points for Rys quadrature (default: 50)
- `--boys-only` - Generate only Boys function reference
- `--rys-only` - Generate only Rys quadrature reference

### Boys Function Reference
- **n range**: 0 to 30
- **T points**: 200 points in [0, 100]
- **Precision**: 50 decimal digits
- **Total**: 6,200 reference values

### Rys Quadrature Reference
- **n_roots range**: 1 to 10
- **T points**: 50 points in [0, 50]
- **Precision**: 50 decimal digits
- **Total**: 500 quadrature rules
- **Algorithm**: Golub-Welsch using moment-based approach

---

## PySCF Reference Data Generator (`generate_reference.py`)

This script generates reference integral data using PySCF.

## Requirements

- Python 3.7+
- PySCF (install with `pip install pyscf` or `conda install -c pyscf pyscf`)
- NumPy (automatically installed with PySCF)

## Usage

Basic usage (generates all molecules with default basis sets):
```bash
python3 generate_reference.py
```

### Command Line Options

- `--molecules`: Select specific molecules (default: H2 H2O NH3 CH4)
- `--basis-sets`: Select specific basis sets (default: STO-3G cc-pVDZ)
- `--output-dir`: Output directory for JSON files (default: tests/data/reference/)
- `--verbose`: Print detailed progress information

### Examples

Generate reference data for only H2 and H2O:
```bash
python3 generate_reference.py --molecules H2 H2O
```

Generate reference data with STO-3G basis only:
```bash
python3 generate_reference.py --basis-sets STO-3G
```

Generate with verbose output:
```bash
python3 generate_reference.py --verbose
```

Custom output directory:
```bash
python3 generate_reference.py --output-dir /path/to/output
```

## Output Format

The script generates JSON files with the following structure:

```json
{
  "format_version": "1.0",
  "generator": "PySCF",
  "pyscf_version": "2.x.x",
  "generated_date": "2026-02-01T...",
  "molecule": {
    "name": "H2O",
    "description": "...",
    "atoms": ["O", "H", "H"],
    "geometry_angstrom": [[0.0, 0.0, 0.0], ...],
    "geometry_bohr": [[0.0, 0.0, 0.0], ...],
    "charge": 0,
    "spin": 0
  },
  "basis_set": "STO-3G",
  "n_atoms": 3,
  "n_basis": 7,
  "n_shells": 5,
  "shells": [
    {
      "index": 0,
      "atom": 0,
      "angular_momentum": 0,
      "n_primitives": 3,
      "n_contracted": 1,
      "exponents": [...],
      "coefficients": [...]
    },
    ...
  ],
  "integrals": {
    "overlap": {
      "matrix": [...],
      "shape": [7, 7]
    },
    "kinetic": {
      "matrix": [...],
      "shape": [7, 7]
    },
    "nuclear_attraction": {
      "matrix": [...],
      "shape": [7, 7]
    },
    "electron_repulsion": {
      "tensor": [...],
      "shape": [7, 7, 7, 7]
    }
  },
  "shell_pair_blocks": {
    "overlap": [
      {
        "shell_i": 0,
        "shell_j": 0,
        "shell_i_angular_momentum": 0,
        "shell_j_angular_momentum": 0,
        "shell_i_nprim": 3,
        "shell_j_nprim": 3,
        "basis_range_i": [0, 1],
        "basis_range_j": [0, 1],
        "block_shape": [1, 1],
        "overlap_block": [1.0]
      },
      ...
    ],
    "kinetic": [...],
    "nuclear_attraction": [...]
  },
  "shell_quartet_blocks": {
    "electron_repulsion": [
      {
        "shell_i": 0,
        "shell_j": 0,
        "shell_k": 0,
        "shell_l": 0,
        "shell_i_angular_momentum": 0,
        "shell_j_angular_momentum": 0,
        "shell_k_angular_momentum": 0,
        "shell_l_angular_momentum": 0,
        "shell_i_nprim": 3,
        "shell_j_nprim": 3,
        "shell_k_nprim": 3,
        "shell_l_nprim": 3,
        "basis_range_i": [0, 1],
        "basis_range_j": [0, 1],
        "basis_range_k": [0, 1],
        "basis_range_l": [0, 1],
        "block_shape": [1, 1, 1, 1],
        "eri_block": [...]
      },
      ...
    ]
  }
}
```

## Standard Molecular Geometries

### H2 (Hydrogen molecule)
- Bond length: 0.74 Angstrom
- Atoms: H at (0.0, 0.0, 0.0), H at (0.74, 0.0, 0.0)

### H2O (Water)
- O-H bond length: 0.96 Angstrom
- H-O-H angle: 104.5 degrees
- Atoms: O at origin, H atoms positioned with C2v symmetry

### NH3 (Ammonia)
- N-H bond length: 1.012 Angstrom
- H-N-H angle: 106.7 degrees
- Atoms: N at origin, H atoms in pyramidal geometry

### CH4 (Methane)
- C-H bond length: 1.089 Angstrom
- Atoms: C at origin, H atoms in tetrahedral geometry

## Notes

- All matrices are stored in row-major order as flat arrays
- Shell quartet blocks use 8-fold permutational symmetry (only unique quartets stored)
- Coordinates are provided in both Angstrom and Bohr
- Shell information includes exponents and contraction coefficients
