# Getting Started with LibAccInt

## Version: 0.1.0-alpha.2

This tutorial walks you through your first LibAccInt calculation — computing the molecular integrals for H₂ using the STO-3G basis set.

---

## Prerequisites

- C++20 compiler (GCC 12+, Clang 16+, or MSVC 2022+)
- CMake 3.25+
- (Optional) CUDA Toolkit 12+ for GPU acceleration

---

## Installation

### From Source

```bash
git clone https://github.com/libaccint/libaccint.git
cd libaccint

# Configure and build
cmake --preset cpu-release
cmake --build --preset cpu-release

# Run tests
ctest --test-dir build/cpu-release

# Install
sudo cmake --install build/cpu-release
```

### Using as a CMake Dependency

```cmake
# In your CMakeLists.txt
find_package(LibAccInt REQUIRED)
target_link_libraries(my_app PRIVATE LibAccInt::libaccint)
```

Or using FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(
    libaccint
    GIT_REPOSITORY https://github.com/libaccint/libaccint.git
    GIT_TAG v0.1.0-alpha.2
)
FetchContent_MakeAvailable(libaccint)
target_link_libraries(my_app PRIVATE LibAccInt::libaccint)
```

---

## Your First Program

### Step 1: Include Headers

```cpp
#include <libaccint/libaccint.hpp>  // All-in-one header
#include <iostream>
#include <vector>

using namespace libaccint;
```

### Step 2: Define the Molecule

Atoms are specified by atomic number and position in Bohr (atomic units):

```cpp
// H₂ molecule with bond length 1.4 bohr
std::vector<data::Atom> atoms = {
    {1, {0.0, 0.0, 0.0}},     // H at origin
    {1, {0.0, 0.0, 1.4}},     // H at z = 1.4 bohr
};
```

### Step 3: Create a Basis Set

```cpp
// Use a built-in basis set
BasisSet basis = data::create_builtin_basis("STO-3G", atoms);
Size nbf = basis.n_basis_functions();
std::cout << "Basis functions: " << nbf << "\n";  // Output: 2
```

Or create shells manually:

```cpp
Shell s1(0,                                      // Angular momentum (s)
         Point3D{0.0, 0.0, 0.0},                // Center
         {3.42525091, 0.62391373, 0.16885540},   // Exponents
         {0.15432897, 0.53532814, 0.44463454});  // Coefficients

Shell s2(0,
         Point3D{0.0, 0.0, 1.4},
         {3.42525091, 0.62391373, 0.16885540},
         {0.15432897, 0.53532814, 0.44463454});

BasisSet basis({s1, s2});
```

### Step 4: Create the Engine

```cpp
Engine engine(basis);
```

### Step 5: Compute One-Electron Integrals

```cpp
// Overlap matrix S
std::vector<Real> S(nbf * nbf, 0.0);
engine.compute_overlap_matrix(S);

// Kinetic energy matrix T
std::vector<Real> T(nbf * nbf, 0.0);
engine.compute_kinetic_matrix(T);

// Nuclear attraction matrix V
PointChargeParams charges;
for (const auto& atom : atoms) {
    charges.x.push_back(atom.position.x);
    charges.y.push_back(atom.position.y);
    charges.z.push_back(atom.position.z);
    charges.charge.push_back(static_cast<Real>(atom.atomic_number));
}

std::vector<Real> V(nbf * nbf, 0.0);
engine.compute_nuclear_matrix(charges, V);

// Or compute T+V directly:
std::vector<Real> H_core(nbf * nbf, 0.0);
engine.compute_core_hamiltonian(charges, H_core);
```

### Step 6: Compute Two-Electron Integrals

LibAccInt uses the **compute-and-consume** pattern — integrals are accumulated into consumers (like `FockBuilder`) without storing the full ERI tensor:

```cpp
// Create a FockBuilder consumer
consumers::FockBuilder fock(nbf);

// Set the density matrix
std::vector<Real> D(nbf * nbf, 0.0);
D[0] = 0.5;  // Simple diagonal density
D[nbf + 1] = 0.5;
fock.set_density(D.data(), nbf);

// Compute two-electron integrals and accumulate J/K
engine.compute(Operator::coulomb(), fock);

// Retrieve results
auto J = fock.get_coulomb_matrix();  // Coulomb matrix
auto K = fock.get_exchange_matrix(); // Exchange matrix

// Build Fock matrix: F = H_core + J - 0.5*K
auto F = fock.get_fock_matrix(
    std::span<const Real>(H_core),
    0.5  // Exchange fraction for RHF
);
```

---

## Loading Basis Sets from Files

LibAccInt can parse basis sets in the BSE (Basis Set Exchange) JSON format:

```cpp
#include <libaccint/data/basis_parser.hpp>

// Load from the bundled basis set data
BasisSet basis = data::load_basis_set("cc-pvdz", atoms);

// Or from a specific file path
BasisSet basis = data::load_basis_set_from_file("/path/to/basis.json", atoms);
```

```cpp
#include <libaccint/data/bse_json_parser.hpp>

// Parse BSE JSON directly
std::string json = read_json_file("my_basis.json");
BasisSet basis = data::BseJsonParser::parse(json, atoms);
```

---

## What's Next?

- **[GPU Setup Guide](GPU_SETUP.md)** — Enable GPU acceleration
- **[Migration Guide](MIGRATION_GUIDE.md)** — Coming from libint2 or libcint
- **[Performance Guidelines](../api/PERFORMANCE_GUIDELINES.md)** — Optimize your code
- **[API Reference](../api/API_REFERENCE.md)** — Complete API documentation
- **[Examples](../../examples/)** — More example programs

---

## Quick Reference Card

| Task | Code |
|------|------|
| Create basis | `data::create_builtin_basis("STO-3G", atoms)` |
| Create engine | `Engine engine(basis)` |
| Overlap matrix | `engine.compute_overlap_matrix(S)` |
| Kinetic matrix | `engine.compute_kinetic_matrix(T)` |
| Nuclear matrix | `engine.compute_nuclear_matrix(charges, V)` |
| Core Hamiltonian | `engine.compute_core_hamiltonian(charges, H)` |
| Fock build | `engine.compute(Operator::coulomb(), fock)` |
| Get J matrix | `fock.get_coulomb_matrix()` |
| Get K matrix | `fock.get_exchange_matrix()` |
| Version | `libaccint::version()` |
