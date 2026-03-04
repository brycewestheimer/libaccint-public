# API Reference Audit

## Version: 0.1.0-alpha.2
## Last Updated: 2026-02-06

This document provides a complete audit of all public headers in LibAccInt, documenting every public type, function, and constant.

For alpha concurrency guarantees, use `docs/api/THREAD_SAFETY.md` as the source of truth.

---

## Core Types (`core/types.hpp`)

### Type Aliases
| Name | Definition | Description |
|------|-----------|-------------|
| `Real` | `double` | Default floating-point type |
| `Float` | `float` | Single-precision type (GPU operations) |
| `Index` | `std::int64_t` | Shell/function index type |
| `Size` | `std::size_t` | Size/count type |

### Enumerations
| Name | Values | Description |
|------|--------|-------------|
| `AngularMomentum` | `S=0, P=1, D=2, F=3, G=4, H=5, I=6` | Angular momentum quantum number |
| `DerivativeOrder` | `Energy=0, Gradient=1, Hessian=2` | Derivative order |

### Constants
| Name | Value | Description |
|------|-------|-------------|
| `MAX_ANGULAR_MOMENTUM` | `6` | Maximum supported l (I functions) |
| `MAX_RYS_ROOTS` | `15` | Maximum Rys quadrature roots |

### Free Functions
| Signature | Description |
|-----------|-------------|
| `int to_int(AngularMomentum)` | Convert enum to integer |
| `int n_cartesian(int l)` | Number of Cartesian functions: (l+1)(l+2)/2 |
| `int n_spherical(int l)` | Number of spherical functions: 2l+1 |
| `int n_functions(int l, bool spherical)` | Function count for given convention |

### Structs
| Name | Members | Description |
|------|---------|-------------|
| `Point3D` | `double x, y, z` | 3D point in space |

---

## Basis Set (`basis/shell.hpp`, `basis/basis_set.hpp`)

### `Shell`
| Method | Signature | Description |
|--------|-----------|-------------|
| Constructor | `Shell(int am, Point3D center, vector<Real> exps, vector<Real> coeffs)` | Auto-normalizing constructor |
| Constructor | `Shell(AngularMomentum am, Point3D center, vector<Real> exps, vector<Real> coeffs)` | Enum AM constructor |
| Constructor | `Shell(PreNormalizedTag, int am, Point3D, vector<Real>, vector<Real>)` | Pre-normalized constructor |
| `angular_momentum()` | `int` | Angular momentum quantum number |
| `center()` | `const Point3D&` | Shell center position |
| `n_primitives()` | `Size` | Number of primitive Gaussians |
| `n_functions()` | `int` | Number of Cartesian basis functions |
| `exponents()` | `span<const Real>` | Primitive exponents |
| `coefficients()` | `span<const Real>` | Contraction coefficients |
| `valid()` | `bool` | Whether shell has primitives |
| `atom_index()` | `Index` | Atom index (-1 if unset) |
| `shell_index()` | `Index` | Shell index (-1 if unset) |
| `function_index()` | `Index` | Basis function offset (-1 if unset) |

### `BasisSet`
| Method | Signature | Description |
|--------|-----------|-------------|
| Constructor | `BasisSet(vector<Shell> shells)` | Construct from shells |
| `n_shells()` | `Size` | Total number of shells |
| `n_basis_functions()` | `Size` | Total number of basis functions |
| `max_angular_momentum()` | `int` | Maximum AM across all shells |
| `shell(Size i)` | `const Shell&` | Access shell by index |
| `shells()` | `span<const Shell>` | All shells as span |
| `shell_set_pairs()` | `const vector<ShellSetPair>&` | ShellSetPairs for iteration |
| `shell_set_quartets()` | `const vector<ShellSetQuartet>&` | ShellSetQuartets for iteration |

### `AuxiliaryBasisSet`
| Method | Signature | Description |
|--------|-----------|-------------|
| Constructor | `AuxiliaryBasisSet(vector<Shell> shells)` | Construct from shells |
| `n_shells()` | `Size` | Total number of shells |
| `n_basis_functions()` | `Size` | Total number of basis functions |

---

## Operators (`operators/operator.hpp`, `operators/operator_types.hpp`)

### `OperatorKind` Enumeration
| Value | Type | Parameters |
|-------|------|-----------|
| `Overlap` | 1e | None |
| `Kinetic` | 1e | None |
| `Nuclear` | 1e | `PointChargeParams` |
| `PointCharge` | 1e | `PointChargeParams` |
| `DistributedMultipole` | 1e | `DistributedMultipoleParams` |
| `Coulomb` | 2e | None |
| `ErfCoulomb` | 2e | `RangeSeparatedParams` |
| `ErfcCoulomb` | 2e | `RangeSeparatedParams` |
| `ElectricDipole` | 1e | `MultipoleOriginParams` |
| `LinearMomentum` | 1e | None |

### `Operator` (Factory Pattern)
| Factory Method | Description |
|---------------|-------------|
| `Operator::overlap()` | Overlap operator |
| `Operator::kinetic()` | Kinetic energy operator |
| `Operator::coulomb()` | Coulomb 1/r₁₂ operator |
| `Operator::nuclear(PointChargeParams)` | Nuclear attraction |
| `Operator::point_charges(PointChargeParams)` | Point charge interaction |
| `Operator::erf_coulomb(Real omega)` | erf(ωr₁₂)/r₁₂ |
| `Operator::erfc_coulomb(Real omega)` | erfc(ωr₁₂)/r₁₂ |

### `PointChargeParams`
| Member | Type | Description |
|--------|------|-------------|
| `x` | `vector<Real>` | X-coordinates |
| `y` | `vector<Real>` | Y-coordinates |
| `z` | `vector<Real>` | Z-coordinates |
| `charge` | `vector<Real>` | Charges |
| `n_centers()` | `Size` | Number of charge centers |

### `OneElectronOperator`
| Method | Description |
|--------|-------------|
| `OneElectronOperator(Operator op)` | Construct from single operator |
| `add(Operator op, Real scale)` | Add scaled contribution |
| `operator+` | Combine operators |
| `operator*` | Scale operator |

---

## Engine (`engine/engine.hpp`)

### `Engine`
| Method | Signature | Description |
|--------|-----------|-------------|
| Constructor | `Engine(const BasisSet&, DispatchConfig)` | Create engine |
| `basis()` | `const BasisSet&` | Get basis set |
| `gpu_available()` | `bool` | GPU backend available? |
| `compute_overlap_matrix(vector<Real>&)` | Compute S matrix |
| `compute_kinetic_matrix(vector<Real>&)` | Compute T matrix |
| `compute_nuclear_matrix(PointChargeParams, vector<Real>&)` | Compute V matrix |
| `compute_core_hamiltonian(PointChargeParams, vector<Real>&)` | Compute H = T + V |
| `compute(Operator, Consumer&)` | Compute-and-consume (full basis) |
| `compute(Operator, ShellSetPair, vector<Real>&)` | Compute for shell set pair |
| `compute(Operator, Shell, Shell, Buffer&)` | Single shell pair |
| `compute_and_consume(Operator, Consumer&)` | Two-electron full basis |
| `compute_and_consume_parallel(Operator, Consumer&, int)` | Parallel two-electron (CPU/OpenMP only) |

### Non-Consumer Compute Methods

These methods return integral values directly without requiring a consumer callback.

#### `compute_quartet`

```cpp
IntegralBuffer compute(const Operator& op, const ShellSetQuartet& quartet,
                       BackendHint hint = BackendHint::Auto,
                       const ScreeningOptions& screening = ScreeningOptions::none())
```

Compute two-electron integrals for a single ShellSetQuartet without a consumer callback. Returns an `IntegralBuffer` containing all computed integral values and metadata.

**Parameters:**
- `op` — Two-electron operator (e.g., `Operator::coulomb()`)
- `quartet` — ShellSetQuartet to compute
- `hint` — Backend selection hint (default: `Auto`)
- `screening` — Optional screening parameters

#### `compute_pair`

```cpp
IntegralBuffer compute(const Operator& op, const ShellSetPair& pair,
                       BackendHint hint = BackendHint::Auto)
```

Compute one-electron integrals for a single ShellSetPair without a consumer. Returns an `IntegralBuffer`.

**Parameters:**
- `op` — One-electron operator
- `pair` — ShellSetPair to compute
- `hint` — Backend selection hint (default: `Auto`)

#### `compute_eri_tensor`

```cpp
std::vector<Real> compute_eri_tensor(const Operator& op = Operator::coulomb(),
                                     BackendHint hint = BackendHint::Auto)
```

Compute the full 4-index ERI tensor as a flat vector of size nbf⁴. Row-major indexing: `index = i*nbf³ + j*nbf² + k*nbf + l`. Safety limit: nbf ≤ 200.

**Parameters:**
- `op` — Two-electron operator (default: `Operator::coulomb()`)
- `hint` — Backend selection hint (default: `Auto`)

**Returns:** `std::vector<Real>` of size nbf⁴

**Python:** Returns `numpy.ndarray` of shape `(nbf, nbf, nbf, nbf)`.

#### `compute_eri_block`

```cpp
std::vector<Real> compute_eri_block(const Operator& op,
                                    const ShellSetQuartet& quartet,
                                    BackendHint hint = BackendHint::Auto)
```

Compute ERIs for a single ShellSetQuartet, scattered into a block with local basis-function indexing.

**Parameters:**
- `op` — Two-electron operator
- `quartet` — ShellSetQuartet to compute
- `hint` — Backend selection hint (default: `Auto`)

**Returns:** `std::vector<Real>` with local basis-function indexed integrals

#### `compute_batch_parallel`

```cpp
std::vector<IntegralBuffer> compute_batch_parallel(
    const Operator& op, std::span<const ShellSetQuartet> quartets,
    int n_threads = 0, BackendHint hint = BackendHint::Auto,
    const ScreeningOptions& screening = ScreeningOptions::none())
```

Parallel version of compute_batch using OpenMP. Each ShellSetQuartet is computed independently. On GPU-backed executions this is the recommended shared-engine batch API because each call acquires its own execution slot.

**Parameters:**
- `op` — Two-electron operator
- `quartets` — Span of ShellSetQuartets to compute
- `n_threads` — Number of threads (0 = auto-detect)
- `hint` — Backend selection hint (default: `Auto`)
- `screening` — Optional screening parameters

**Returns:** `std::vector<IntegralBuffer>` — one per input quartet

#### `compute_all_2e_parallel`

```cpp
std::vector<IntegralBuffer> compute_all_2e_parallel(
    const Operator& op, int n_threads = 0,
    BackendHint hint = BackendHint::Auto)
```

Compute all two-electron integrals for the full basis in parallel. On GPU-backed executions this remains a batched shared-engine API.

**Parameters:**
- `op` — Two-electron operator
- `n_threads` — Number of threads (0 = auto-detect)
- `hint` — Backend selection hint (default: `Auto`)

**Returns:** `std::vector<IntegralBuffer>` — one per ShellSetQuartet in the basis

#### `compute_batch_screened`

```cpp
std::vector<IntegralBuffer> compute_batch_screened(
    const Operator& op, const ScreeningOptions& screening,
    BackendHint hint = BackendHint::Auto)
```

Compute all two-electron integrals with Schwarz screening. Precomputes Schwarz bounds automatically. Empty `IntegralBuffer` entries in the result indicate screened-out quartets.

**Parameters:**
- `op` — Two-electron operator
- `screening` — Screening options including threshold
- `hint` — Backend selection hint (default: `Auto`)

**Returns:** `std::vector<IntegralBuffer>` — empty buffers for screened-out quartets

#### `compute_screening_statistics` (static)

```cpp
static ScreeningStatistics compute_screening_statistics(
    const std::vector<IntegralBuffer>& results)
```

Analyze batch results to count computed vs. screened-out quartets.

**Parameters:**
- `results` — Vector of IntegralBuffers from a batch or screened computation

**Returns:** `ScreeningStatistics` with `n_computed`, `n_screened`, and `screening_ratio` fields

### `BackendHint`
| Value | Description |
|-------|-------------|
| `Auto` | Automatic dispatch (default) |
| `ForceCPU` | Always use CPU |
| `PreferGPU` | Prefer GPU when available |
| `ForceGPU` | Require GPU (throws if unavailable) |

---

## Consumers (`consumers/`)

### `FockBuilder`
| Method | Description |
|--------|-------------|
| `FockBuilder(Size nbf)` | Construct for nbf basis functions |
| `set_density(const Real*, Size)` | Set density matrix |
| `accumulate(buffer, fa, fb, fc, fd, na, nb, nc, nd)` | Process quartet |
| `get_coulomb_matrix()` | Get J matrix |
| `get_exchange_matrix()` | Get K matrix |
| `get_fock_matrix(span<const Real> H_core, Real exchange_fraction)` | Get F = H + J - xK |
| `reset()` | Zero out J and K |
| `set_threading_strategy(FockThreadingStrategy)` | Set thread safety |

### `DFFockBuilder`
| Method | Description |
|--------|-------------|
| `DFFockBuilder(BasisSet, AuxBasisSet, Config)` | Construct DF builder |
| `set_density(const Real*, Size)` | Set density matrix |
| `compute()` | Compute DF-J and DF-K |
| `get_coulomb_matrix()` | Get DF-J matrix |
| `get_exchange_matrix()` | Get DF-K matrix |

---

## Data (`data/`)

### `data::Atom`
| Member | Type | Description |
|--------|------|-------------|
| `atomic_number` | `int` | Atomic number (Z) |
| `position` | `Point3D` | Position in Bohr |

### Functions
| Signature | Description |
|-----------|-------------|
| `create_builtin_basis(string, vector<Atom>)` | Create built-in basis |
| `create_sto3g(vector<Atom>)` | Create STO-3G basis |
| `load_basis_set(string, vector<Atom>)` | Load from BSE JSON file |
| `load_basis_set_from_file(string, vector<Atom>)` | Load from file path |

### `BseJsonParser`
| Method | Description |
|--------|-------------|
| `parse(string, vector<Atom>)` | Parse BSE JSON string |
| `parse_file(string, vector<Atom>)` | Parse BSE JSON file |
| `validate(string)` | Validate JSON format |
| `get_name(string)` | Get basis set name |
| `get_supported_elements(string)` | List supported elements |

---

## Configuration (`config.hpp`)

### Macros
| Name | Description |
|------|-------------|
| `LIBACCINT_VERSION` | Version string (e.g., "0.1.0-alpha.2") |
| `LIBACCINT_VERSION_MAJOR` | Major version number |
| `LIBACCINT_VERSION_MINOR` | Minor version number |
| `LIBACCINT_VERSION_PATCH` | Patch version number |
| `LIBACCINT_USE_CUDA` | 1 if CUDA enabled |

### Functions
| Signature | Description |
|-----------|-------------|
| `version()` | Version string |
| `has_cuda_backend()` | CUDA available? |
| `has_openmp()` | OpenMP available? |

---

## Error Handling (`utils/error_handling.hpp`)

### Exception Hierarchy
```
std::runtime_error
└── libaccint::Exception
    ├── InvalidArgumentException    — Invalid input parameters
    ├── InvalidStateException       — Invalid configuration/state
    ├── NotImplementedException     — Feature not yet implemented
    ├── MemoryException             — Resource allocation failure
    ├── BackendException            — GPU backend errors
    └── NumericalException          — Numerical issues
```

### Macros
| Name | Description |
|------|-------------|
| `LIBACCINT_ASSERT(cond, msg)` | Assert or throw InvalidStateException |
