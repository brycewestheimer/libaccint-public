# Migration Guide: From libint2/libcint to LibAccInt

## Version: 0.1.0-alpha.2

This guide helps users migrating from libint2 or libcint to LibAccInt, mapping familiar concepts and API patterns to their LibAccInt equivalents.

---

## Quick Comparison

| Feature | libint2 | libcint | LibAccInt |
|---------|---------|---------|-----------|
| Language | C++11 | C | C++20 |
| GPU support | No | No | CUDA (single GPU in alpha) |
| Compute model | Buffer-based | Buffer-based | `Engine::compute(...)` / compute-and-consume |
| Basis sets | BSE JSON | Internal | BSE JSON + built-in |
| Build system | CMake | CMake | CMake (presets) |
| Threading | Manual | Manual | Built-in OpenMP |
| Screening | Manual | Manual | Built-in Schwarz |

---

## Migration from libint2

### Initialization

```cpp
// libint2
libint2::initialize();
// ... computation ...
libint2::finalize();

// LibAccInt — no global init/finalize required
// Just construct your objects and go
```

### Shell Construction

```cpp
// libint2
libint2::Shell s(
    {3.42525091, 0.62391373, 0.16885540},    // exponents
    {{0, false, {0.15432897, 0.53532814, 0.44463454}}},  // {l, pure, coeffs}
    {{0.0, 0.0, 0.0}}                         // center
);

// LibAccInt
libaccint::Shell s(
    0,                                         // angular momentum
    libaccint::Point3D{0.0, 0.0, 0.0},       // center
    {3.42525091, 0.62391373, 0.16885540},     // exponents
    {0.15432897, 0.53532814, 0.44463454}      // coefficients
);
```

### Basis Set Creation

```cpp
// libint2
auto shells = libint2::BasisSet("sto-3g", atoms);

// LibAccInt
auto basis = libaccint::data::create_builtin_basis("STO-3G", atoms);
// Or from BSE JSON file:
auto basis = libaccint::data::load_basis_set("cc-pvdz", atoms);
```

### One-Electron Integrals

```cpp
// libint2
libint2::Engine engine(libint2::Operator::overlap, max_nprim, max_l);
engine.compute(s1, s2);
const auto& buf = engine.results();
// buf[0] contains the integrals (or nullptr if screened)

// LibAccInt — shell-pair level
libaccint::Engine engine(basis);
libaccint::OneElectronBuffer<0> buffer;
buffer.resize(s1.n_functions(), s2.n_functions());
engine.compute(libaccint::Operator::overlap(), s1, s2, buffer);
// Access: buffer(i, j)

// LibAccInt — full matrix (preferred)
std::vector<libaccint::Real> S(nbf * nbf, 0.0);
engine.compute_overlap_matrix(S);
```

### Two-Electron Integrals (ERIs)

```cpp
// libint2
libint2::Engine engine(libint2::Operator::coulomb, max_nprim, max_l);
engine.compute(s1, s2, s3, s4);
// Access: engine.results()[0]

// LibAccInt — shell quartet level
libaccint::Engine engine(basis);
libaccint::TwoElectronBuffer<0> buffer;
engine.compute(libaccint::Operator::coulomb(), s1, s2, s3, s4, buffer);
// Access: buffer(a, b, c, d)

// LibAccInt — fused Fock build (preferred)
libaccint::consumers::FockBuilder fock(nbf);
fock.set_density(D.data(), nbf);
engine.compute(libaccint::Operator::coulomb(), fock);
auto J = fock.get_coulomb_matrix();
```

### Atom Definition

```cpp
// libint2
libint2::Atom atom;
atom.atomic_number = 1;
atom.x = 0.0; atom.y = 0.0; atom.z = 0.0;

// LibAccInt
libaccint::data::Atom atom{1, {0.0, 0.0, 0.0}};
```

### Key Differences from libint2

1. **No global state**: LibAccInt has no `initialize()`/`finalize()` calls
2. **No code generation at build time**: LibAccInt uses pre-compiled kernels
3. **Fused computation**: Prefer `FockBuilder` over raw ERI buffers
4. **GPU acceleration**: Transparent GPU dispatch via `BackendHint`
5. **Built-in screening**: Use `ScreeningOptions` instead of manual Schwarz
6. **Modern C++20**: Concepts, spans, structured bindings

---

## Migration from libcint

### Environment Setup

```c
// libcint
int *atm, *bas;
double *env;
// ... fill atm, bas, env arrays ...
int natm = ..., nbas = ...;

// LibAccInt — no low-level arrays needed
std::vector<libaccint::data::Atom> atoms = {
    {8, {0.0, 0.0, 0.0}},
    {1, {0.0, 0.0, 1.8}},
};
auto basis = libaccint::data::create_builtin_basis("STO-3G", atoms);
```

### One-Electron Integrals

```c
// libcint
double buf[nf_i * nf_j];
int shls[2] = {i, j};
cint1e_ovlp_cart(buf, shls, atm, natm, bas, nbas, env);

// LibAccInt
libaccint::Engine engine(basis);
std::vector<libaccint::Real> S(nbf * nbf, 0.0);
engine.compute_overlap_matrix(S);
```

### Two-Electron Integrals

```c
// libcint
double buf[nf_i * nf_j * nf_k * nf_l];
int shls[4] = {i, j, k, l};
cint2e_cart(buf, shls, atm, natm, bas, nbas, env, NULL);

// LibAccInt — direct Fock build
libaccint::consumers::FockBuilder fock(nbf);
fock.set_density(D, nbf);
engine.compute(libaccint::Operator::coulomb(), fock);
```

### Key Differences from libcint

1. **C++ vs C**: LibAccInt uses C++20 with RAII, no manual memory management
2. **No atm/bas/env arrays**: Shell objects encapsulate all data
3. **Automatic normalization**: Shell constructors normalize by default
4. **Object-oriented**: Engine, BasisSet, Shell are classes with methods
5. **GPU support**: Add `BackendHint::PreferGPU` for acceleration

---

## Feature Mapping Table

| Operation | libint2 | libcint | LibAccInt |
|-----------|---------|---------|-----------|
| Overlap | `Operator::overlap` | `cint1e_ovlp` | `Operator::overlap()` |
| Kinetic | `Operator::kinetic` | `cint1e_kin` | `Operator::kinetic()` |
| Nuclear | `Operator::nuclear` | `cint1e_nuc` | `Operator::nuclear(charges)` |
| ERI | `Operator::coulomb` | `cint2e` | `Operator::coulomb()` |
| Erf-ERI | `Operator::stg` | — | `Operator::erf_coulomb(ω)` |
| Density fitting | — | — | `DFFockBuilder` |
| Screening | Manual | — | `ScreeningOptions` |
| GPU | — | — | `BackendHint::PreferGPU` |
| Parallel | Manual OpenMP | — | `compute_and_consume_parallel()` |

---

## Common Migration Patterns

### Pattern: Full SCF Integral Computation

```cpp
// Before (libint2-style):
//   1. Create Engine for each operator type
//   2. Loop over shell pairs/quartets
//   3. Manually accumulate into matrices

// After (LibAccInt):
Engine engine(basis);

// One-electron integrals in a single call
std::vector<Real> S(nbf * nbf), H(nbf * nbf);
engine.compute_overlap_matrix(S);
engine.compute_core_hamiltonian(charges, H);

// Two-electron integrals with fused Fock build
consumers::FockBuilder fock(nbf);
fock.set_density(D.data(), nbf);
engine.compute(Operator::coulomb(), fock);
auto F = fock.get_fock_matrix(std::span(H), 0.5);
```

### Pattern: Screening

```cpp
// Before: Manual Schwarz bound computation and comparison
// After:
screening::ScreeningOptions opts = screening::ScreeningOptions::normal();
engine.compute_and_consume_screened_parallel(Operator::coulomb(), fock, opts);
```

---

## Getting Help

- API Reference: `docs/api/API_REFERENCE.md`
- Examples: `examples/` directory
- FAQ: `docs/user_guide/FAQ.md`
