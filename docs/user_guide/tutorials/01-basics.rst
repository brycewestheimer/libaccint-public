.. _tutorial-basics:

Tutorial 1: Basics
==================

This tutorial introduces the core concepts of LibAccInt: shells, basis sets,
and the integral engine.

Core Concepts
-------------

Shell
~~~~~

A **Shell** represents a contracted Gaussian basis function centered at a point
in 3D space. It consists of:

- **Center**: The (x, y, z) position (usually an atom's coordinates)
- **Angular momentum**: s (L=0), p (L=1), d (L=2), etc.
- **Exponents**: The α values in exp(-α|r-R|²)
- **Coefficients**: Contraction coefficients for the primitives

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>
   using namespace libaccint;

   // Create an s-type shell at the origin with 3 primitives (like STO-3G)
   Shell s_shell(
       AngularMomentum::S,           // s-type (L=0)
       Point3D{0.0, 0.0, 0.0},       // center at origin
       {3.42525091, 0.62391373, 0.16885540},  // exponents
       {0.15432897, 0.53532814, 0.44463454}   // coefficients
   );

ShellSet
~~~~~~~~

A **ShellSet** groups shells with identical angular momentum and primitive count
for efficient batch processing. This is key to GPU efficiency.

.. code-block:: cpp

   // Create ShellSet from shells with same AM and primitive count
   std::vector<Shell> shells = /* ... */;
   ShellSet shell_set(shells);  // Validates uniformity automatically

BasisSet
~~~~~~~~

A **BasisSet** contains all shells for a molecule, organized for efficient computation.

.. code-block:: cpp

   // Create basis set from shells
   BasisSet basis(shells);

   std::cout << "Number of shells: " << basis.n_shells() << "\n";
   std::cout << "Number of basis functions: " << basis.n_basis_functions() << "\n";

Using Built-in Basis Sets
-------------------------

LibAccInt includes built-in support for common basis sets:

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>

   using namespace libaccint;
   using namespace libaccint::data;

   // Define atoms (atomic number, position in Bohr)
   std::vector<Atom> atoms = {
       {8, {0.0, 0.0, 0.0}},                 // Oxygen
       {1, {0.0, 1.43233673, -1.10866041}},  // Hydrogen
       {1, {0.0, -1.43233673, -1.10866041}}  // Hydrogen
   };

   // Create STO-3G basis (built-in)
   BasisSet sto3g = create_sto3g(atoms);

   // Or load from JSON file
   BasisSet cc_pvdz = load_basis_from_json("cc-pvdz.json", atoms);

Creating the Engine
-------------------

The **Engine** is the central orchestrator for integral computation:

.. code-block:: cpp

   // Create engine with default settings
   Engine engine(basis);

   // Check available backends
   std::cout << "GPU available: " << (engine.gpu_available() ? "yes" : "no") << "\n";

   // Configure dispatch behavior
   engine.set_dispatch_config(DispatchConfig{
       .prefer_gpu = true,
       .min_work_for_gpu = 1000
   });

Understanding Results
---------------------

One-electron integrals are returned as flattened row-major matrices:

.. code-block:: cpp

   std::vector<Real> S;
   engine.compute(OneElectronOperator(Operator::overlap()), S);

   // Access element (i, j)
   Size nbf = basis.n_basis_functions();
   Real S_ij = S[i * nbf + j];

Two-electron integrals use a more complex layout due to their 4-dimensional nature.
See :doc:`03-two-electron` for details.

Complete Example
----------------

Here's a complete program demonstrating basic concepts:

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>
   #include <iostream>
   #include <iomanip>

   using namespace libaccint;

   int main() {
       // Create H2O molecule
       std::vector<data::Atom> atoms = {
           {8, {0.0, 0.0, 0.0}},
           {1, {0.0, 1.43233673, -1.10866041}},
           {1, {0.0, -1.43233673, -1.10866041}}
       };

       // Create basis set
       BasisSet basis = data::create_sto3g(atoms);

       std::cout << "Molecule: H2O\n";
       std::cout << "Basis: STO-3G\n";
       std::cout << "Shells: " << basis.n_shells() << "\n";
       std::cout << "Basis functions: " << basis.n_basis_functions() << "\n";
       std::cout << "Max angular momentum: " << basis.max_angular_momentum() << "\n";

       // Create engine
       Engine engine(basis);
       std::cout << "GPU available: " << (engine.gpu_available() ? "yes" : "no") << "\n";

       // Compute overlap
       std::vector<Real> S;
       engine.compute(OneElectronOperator(Operator::overlap()), S);

       // Print diagonal elements
       Size nbf = basis.n_basis_functions();
       std::cout << "\nOverlap matrix diagonal:\n";
       for (Size i = 0; i < nbf; ++i) {
           std::cout << "  S[" << i << "," << i << "] = "
                     << std::fixed << std::setprecision(6) << S[i * nbf + i] << "\n";
       }

       return 0;
   }

Next Steps
----------

- :doc:`02-one-electron` - Learn about all one-electron integrals
- :doc:`03-two-electron` - Compute two-electron integrals
