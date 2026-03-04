.. _getting-started:

Getting Started
===============

This guide will help you install LibAccInt and run your first molecular integral
computation in just a few minutes.

What is LibAccInt?
------------------

LibAccInt is a high-performance library for computing molecular integrals used in
computational quantum chemistry. It supports:

- **One-electron integrals**: Overlap (S), kinetic energy (T), nuclear attraction (V)
- **Two-electron integrals**: Electron repulsion integrals (ERI)
- **Angular momentum**: s, p, d functions (Cartesian Gaussians)
- **GPU acceleration**: CUDA backend

Prerequisites
-------------

Before installing LibAccInt, ensure you have:

- **Operating System**: Linux, macOS, or Windows (WSL2 recommended)
- **Compiler**: C++20 capable (GCC 11+, Clang 14+, MSVC 19.30+)
- **CMake**: Version 3.25 or later
- **Internet connection**: For fetching dependencies (GoogleTest, Eigen, nlohmann/json)

Optional:

- **CUDA Toolkit 11.0+**: For NVIDIA GPU support

Quick Installation
------------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/yourusername/libaccint.git
      cd libaccint

2. Configure and build:

   .. code-block:: bash

      cmake --preset cpu-release
      cmake --build --preset cpu-release

3. Verify the installation:

   .. code-block:: bash

      ctest --test-dir build/cpu-release

Your First Program
------------------

Let's compute the overlap matrix for a water molecule using the STO-3G basis set.

Create a file ``hello_libaccint.cpp``:

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>
   #include <iostream>

   using namespace libaccint;

   int main() {
       // Define H2O molecule (coordinates in Bohr)
       std::vector<data::Atom> atoms = {
           {8, {0.0, 0.0, 0.0}},               // O at origin
           {1, {0.0, 1.43233673, -1.10866041}}, // H
           {1, {0.0, -1.43233673, -1.10866041}} // H
       };

       // Create STO-3G basis set
       auto basis = data::create_sto3g(atoms);

       // Create integral engine
       Engine engine(basis);

       // Compute overlap matrix
       std::vector<Real> S;
       engine.compute(OneElectronOperator(Operator::overlap()), S);

       // Print results
       Size nbf = basis.n_basis_functions();
       std::cout << "Basis functions: " << nbf << "\n";
       std::cout << "Overlap matrix (" << nbf << "x" << nbf << "):\n";

       for (Size i = 0; i < nbf; ++i) {
           for (Size j = 0; j < nbf; ++j) {
               std::cout << std::fixed << std::setprecision(4)
                         << S[i * nbf + j] << " ";
           }
           std::cout << "\n";
       }

       return 0;
   }

Compile and run:

.. code-block:: bash

   g++ -std=c++20 -I/path/to/libaccint/include hello_libaccint.cpp \
       -L/path/to/libaccint/build/cpu-release -laccint -o hello
   ./hello

Python Quick Start
------------------

If you have Python bindings installed:

.. code-block:: python

   import libaccint

   # Define water molecule
   atoms = [
       libaccint.Atom(8, [0.0, 0.0, 0.0]),
       libaccint.Atom(1, [0.0, 1.43233673, -1.10866041]),
       libaccint.Atom(1, [0.0, -1.43233673, -1.10866041]),
   ]

   # Create basis set and engine
   basis = libaccint.basis_set("sto-3g", atoms)
   engine = libaccint.Engine(basis)

   # Compute overlap matrix
   S = engine.compute_overlap_matrix()
   print(f"Overlap matrix shape: {S.shape}")
   print(S)

Next Steps
----------

- :doc:`installation` - Detailed platform-specific installation
- :doc:`tutorials/index` - Step-by-step tutorials
- :doc:`/api/cpp/index` - Complete API documentation
- :doc:`cookbook` - Common recipes and patterns
