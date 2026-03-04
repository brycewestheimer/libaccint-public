# Basis Set Support

## Version: 0.1.0-alpha.2

## Overview

LibAccInt supports multiple basis set sources:

1. **Built-in basis sets** — hardcoded for immediate use
2. **Basis Set Exchange (BSE) JSON** — standard community format
3. **Bundled BSE JSON files** — included with the distribution
4. **Manual construction** — build custom basis sets from Shell objects

---

## Built-in Basis Sets

The `data::create_builtin_basis()` function provides hardcoded STO-3G basis functions for:

| Element | Symbol | Shells |
|---------|--------|--------|
| Hydrogen | H | 1s |
| Carbon | C | 1s, 2s, 2p |
| Nitrogen | N | 1s, 2s, 2p |
| Oxygen | O | 1s, 2s, 2p |
| Fluorine | F | 1s, 2s, 2p |

### Usage

```cpp
#include <libaccint/data/basis_parser.hpp>

// Create a BasisSet for water (H₂O) using STO-3G
std::vector<data::Atom> atoms = {
    {8, {0.0, 0.0, 0.0}},         // Oxygen
    {1, {0.0, 1.43, 1.11}},       // Hydrogen
    {1, {0.0, -1.43, 1.11}}       // Hydrogen
};

BasisSet basis = data::create_builtin_basis(atoms);
```

---

## BSE JSON Format

The [Basis Set Exchange](https://www.basissetexchange.org/) (BSE) provides thousands of basis sets in a standardized JSON format. LibAccInt includes a parser for this format.

### Downloading from BSE

Via web interface:
1. Visit https://www.basissetexchange.org/
2. Select a basis set and elements
3. Download in JSON format (BSE Schema v0.1)

Via API:
```bash
curl -X GET "https://www.basissetexchange.org/api/basis/cc-pvdz/format/json" \
     -H "Accept: application/json" > cc-pvdz.json
```

Via Python:
```python
import basis_set_exchange as bse
bse.get_basis("cc-pvdz", fmt="json", header=False)
```

### JSON Schema

BSE JSON files follow this structure:
```json
{
  "molssi_bse_schema": {
    "schema_type": "complete",
    "schema_version": "0.1"
  },
  "name": "cc-pVDZ",
  "description": "Correlation-consistent polarized Valence Double-Zeta",
  "elements": {
    "1": {
      "electron_shells": [
        {
          "function_type": "gto",
          "region": "",
          "angular_momentum": [0],
          "exponents": ["13.0100000", "1.9620000", "0.4446000"],
          "coefficients": [
            ["0.0196850", "0.1379770", "0.4781480"]
          ]
        }
      ]
    }
  }
}
```

### Using the BSE Parser

```cpp
#include <libaccint/data/bse_json_parser.hpp>

// Parse from file
auto shells = data::BseJsonParser::parse_file("cc-pvdz.json", atoms);
BasisSet basis(shells);

// Parse from string
std::string json_str = /* ... */;
auto shells2 = data::BseJsonParser::parse(json_str, atoms);

// Validate a BSE JSON file
bool valid = data::BseJsonParser::validate(json_str);

// Get metadata
std::string name = data::BseJsonParser::get_name(json_str);
std::string desc = data::BseJsonParser::get_description(json_str);
auto elements = data::BseJsonParser::get_supported_elements(json_str);
```

### SP (Shared-exponent) Shells

Some basis sets (e.g., 6-31G) use SP shells where s and p functions share exponents but have different contraction coefficients. The BSE parser handles this by creating separate s and p shells:

```json
{
  "angular_momentum": [0, 1],
  "exponents": ["0.1687144", "0.6239137"],
  "coefficients": [
    ["-0.2600918", "1.1434564"],
    ["0.1615948", "0.9156630"]
  ]
}
```

This produces two shells: an s shell with the first coefficient set and a p shell with the second.

---

## Bundled Basis Sets

LibAccInt includes pre-downloaded BSE JSON files in `share/basis_sets/`:

| File | Basis Set | Elements |
|------|-----------|----------|
| `sto-3g.json` | STO-3G | H–Xe |
| `cc-pvdz.json` | cc-pVDZ | H–Ar, Ca–Kr |
| `cc-pvtz.json` | cc-pVTZ | H–Ar, Ca–Kr |

### Using Bundled Basis Sets

```cpp
#include <libaccint/data/bse_json_parser.hpp>

// At runtime, find the share directory relative to the library
auto shells = data::BseJsonParser::parse_file(
    LIBACCINT_SHARE_DIR "/basis_sets/cc-pvdz.json", atoms);
BasisSet basis(shells);
```

---

## Manual Basis Set Construction

For maximum flexibility, construct Shell objects directly:

```cpp
#include <libaccint/shell.hpp>

// Create an s-type shell with 3 primitives (STO-3G hydrogen)
Shell s_shell(
    0,                              // angular momentum (s)
    {3.42525091, 0.62353140, 0.16885540},  // exponents
    {0.15432897, 0.53532814, 0.44463454},  // coefficients
    {0.0, 0.0, 0.0}                // center
);

// Create a p-type shell
Shell p_shell(
    1,                             // angular momentum (p)
    {2.9412494, 0.6834831, 0.2222899},
    {0.1559163, 0.6076837, 0.3919574},
    {0.0, 0.0, 0.0}
);

// Collect into a BasisSet
std::vector<Shell> shells = {s_shell, p_shell};
BasisSet basis(shells);
```

### General Contraction

For generally-contracted basis sets (e.g., ANO, cc-pVnZ-F12), use multiple coefficient vectors:

```cpp
// Generally contracted shell with 2 contractions
Shell gc_shell(
    0,  // angular momentum
    {100.0, 20.0, 5.0, 1.0},     // exponents (shared)
    {                               // contraction coefficients
        {0.01, 0.10, 0.30, 0.60},  // contraction 1
        {0.05, 0.20, 0.50, 0.25}   // contraction 2
    },
    {0.0, 0.0, 0.0}
);
```

---

## Auxiliary Basis Sets

Density fitting requires auxiliary basis sets. These follow the same format but construct an `AuxiliaryBasisSet`:

```cpp
#include <libaccint/data/bse_json_parser.hpp>

// Load auxiliary basis
auto aux_shells = data::BseJsonParser::parse_file("cc-pvdz-ri.json", atoms);
AuxiliaryBasisSet aux_basis(aux_shells);
```

Common auxiliary basis set pairings:

| Orbital Basis | Coulomb Fitting | Exchange Fitting |
|---------------|-----------------|------------------|
| cc-pVDZ | cc-pVDZ-JKFIT | cc-pVDZ-RI |
| cc-pVTZ | cc-pVTZ-JKFIT | cc-pVTZ-RI |
| def2-SVP | def2-SVP-JKFIT | def2-SVP-RI |
| def2-TZVP | def2-TZVP-JKFIT | def2-TZVP-RI |

---

## Basis Set Quality Guide

### Minimal Basis Sets
- **STO-3G**: Minimal basis, fast but low accuracy. Good for testing and prototyping.

### Double-Zeta
- **cc-pVDZ**: Correlation-consistent, polarized. First real production basis.
- **def2-SVP**: Ahlrichs split-valence polarized. Good all-around choice.

### Triple-Zeta
- **cc-pVTZ**: ~5x more functions than DZ. Standard production basis.
- **def2-TZVP**: Good balance of accuracy and cost.

### Quadruple-Zeta and Beyond
- **cc-pVQZ**, **cc-pV5Z**: Near CBS limit. Very expensive.

### Recommendations

| Application | Recommended Basis |
|-------------|-------------------|
| Testing/prototyping | STO-3G |
| DFT geometry optimization | def2-SVP or cc-pVDZ |
| DFT single-point | def2-TZVP or cc-pVTZ |
| Correlated (MP2, CCSD) | cc-pVTZ or larger |
| Benchmark | cc-pVQZ / CBS extrapolation |
