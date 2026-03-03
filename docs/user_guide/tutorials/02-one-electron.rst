.. _tutorial-one-electron:

Tutorial 2: One-Electron Integrals
==================================

This tutorial covers computing one-electron integrals: overlap, kinetic energy,
and nuclear attraction.

One-Electron Integral Types
---------------------------

LibAccInt supports three types of one-electron integrals:

Overlap Integrals (S)
~~~~~~~~~~~~~~~~~~~~~

.. math::

   S_{\mu\nu} = \int \phi_\mu(\mathbf{r}) \phi_\nu(\mathbf{r}) \, d\mathbf{r}

The overlap integral measures how much two basis functions overlap in space.

Kinetic Energy Integrals (T)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   T_{\mu\nu} = -\frac{1}{2} \int \phi_\mu(\mathbf{r}) \nabla^2 \phi_\nu(\mathbf{r}) \, d\mathbf{r}

The kinetic energy integral represents the kinetic energy of an electron.

Nuclear Attraction Integrals (V)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   V_{\mu\nu} = \sum_A \int \phi_\mu(\mathbf{r}) \frac{-Z_A}{|\mathbf{r} - \mathbf{R}_A|} \phi_\nu(\mathbf{r}) \, d\mathbf{r}

The nuclear attraction integral represents the potential energy from electron-nucleus attraction.

Computing One-Electron Integrals
--------------------------------

Full Matrix Computation
~~~~~~~~~~~~~~~~~~~~~~~

The most common use case is computing the full integral matrix:

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>

   using namespace libaccint;

   // Setup (from Tutorial 1)
   std::vector<data::Atom> atoms = {
       {8, {0.0, 0.0, 0.0}},
       {1, {0.0, 1.43233673, -1.10866041}},
       {1, {0.0, -1.43233673, -1.10866041}}
   };
   BasisSet basis = data::create_sto3g(atoms);
   Engine engine(basis);

   // Compute overlap matrix
   std::vector<Real> S;
   engine.compute(OneElectronOperator(Operator::overlap()), S);

   // Compute kinetic energy matrix
   std::vector<Real> T;
   engine.compute(OneElectronOperator(Operator::kinetic()), T);

   // Compute nuclear attraction matrix
   PointChargeParams charges;
   for (const auto& atom : atoms) {
       charges.x.push_back(atom.position.x);
       charges.y.push_back(atom.position.y);
       charges.z.push_back(atom.position.z);
       charges.charge.push_back(static_cast<Real>(atom.atomic_number));
   }

   std::vector<Real> V;
   engine.compute(OneElectronOperator(Operator::nuclear(charges)), V);

Core Hamiltonian
~~~~~~~~~~~~~~~~

The core Hamiltonian is the sum of kinetic and nuclear attraction:

.. math::

   H^{\text{core}}_{\mu\nu} = T_{\mu\nu} + V_{\mu\nu}

.. code-block:: cpp

   // Compute core Hamiltonian
   Size nbf = basis.n_basis_functions();
   std::vector<Real> H_core(nbf * nbf);

   for (Size i = 0; i < nbf * nbf; ++i) {
       H_core[i] = T[i] + V[i];
   }

Shell-Pair Level Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For fine-grained control, compute integrals for specific shell pairs:

.. code-block:: cpp

   // Create a buffer for integrals
   OneElectronBuffer<0> buffer;  // <0> means no derivatives

   // Compute overlap for shell pair (0, 1)
   engine.compute(Operator::overlap(),
                  basis.shell(0), basis.shell(1),
                  buffer);

   // Access the computed integrals
   auto integrals = buffer.data();
   Size nbf_a = basis.shell(0).n_functions();  // 1 for s, 3 for p, 6 for d
   Size nbf_b = basis.shell(1).n_functions();

   for (Size a = 0; a < nbf_a; ++a) {
       for (Size b = 0; b < nbf_b; ++b) {
           Real S_ab = integrals[a * nbf_b + b];
           std::cout << "S[" << a << "," << b << "] = " << S_ab << "\n";
       }
   }

Backend Hints
-------------

Control which backend processes the computation:

.. code-block:: cpp

   // Force CPU computation
   engine.compute(OneElectronOperator(Operator::overlap()), S, BackendHint::ForceCPU);

   // Prefer GPU for large computations
   engine.compute(OneElectronOperator(Operator::overlap()), S, BackendHint::PreferGPU);

   // Let engine decide (default)
   engine.compute(OneElectronOperator(Operator::overlap()), S, BackendHint::Auto);

Complete Example
----------------

Here's a complete program computing all one-electron integrals:

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>
   #include <iostream>
   #include <iomanip>
   #include <cmath>

   using namespace libaccint;

   void print_matrix(const std::string& name,
                     const std::vector<Real>& M,
                     Size n) {
       std::cout << "\n" << name << " matrix:\n";
       for (Size i = 0; i < n; ++i) {
           for (Size j = 0; j < n; ++j) {
               std::cout << std::setw(10) << std::fixed << std::setprecision(4)
                         << M[i * n + j];
           }
           std::cout << "\n";
       }
   }

   int main() {
       // H2O molecule
       std::vector<data::Atom> atoms = {
           {8, {0.0, 0.0, 0.0}},
           {1, {0.0, 1.43233673, -1.10866041}},
           {1, {0.0, -1.43233673, -1.10866041}}
       };

       BasisSet basis = data::create_sto3g(atoms);
       Engine engine(basis);
       Size nbf = basis.n_basis_functions();

       // Compute all one-electron integrals
       std::vector<Real> S, T, V;

       engine.compute(OneElectronOperator(Operator::overlap()), S);
       engine.compute(OneElectronOperator(Operator::kinetic()), T);

       PointChargeParams charges;
       for (const auto& atom : atoms) {
           charges.x.push_back(atom.position.x);
           charges.y.push_back(atom.position.y);
           charges.z.push_back(atom.position.z);
           charges.charge.push_back(static_cast<Real>(atom.atomic_number));
       }
       engine.compute(OneElectronOperator(Operator::nuclear(charges)), V);

       // Print matrices
       print_matrix("Overlap (S)", S, nbf);
       print_matrix("Kinetic (T)", T, nbf);
       print_matrix("Nuclear (V)", V, nbf);

       // Compute and print core Hamiltonian
       std::vector<Real> H(nbf * nbf);
       for (Size i = 0; i < nbf * nbf; ++i) {
           H[i] = T[i] + V[i];
       }
       print_matrix("Core Hamiltonian (H)", H, nbf);

       // Calculate nuclear repulsion energy
       Real E_nn = 0.0;
       for (size_t i = 0; i < atoms.size(); ++i) {
           for (size_t j = i + 1; j < atoms.size(); ++j) {
               Real dx = atoms[i].position.x - atoms[j].position.x;
               Real dy = atoms[i].position.y - atoms[j].position.y;
               Real dz = atoms[i].position.z - atoms[j].position.z;
               Real r = std::sqrt(dx*dx + dy*dy + dz*dz);
               E_nn += Real(atoms[i].atomic_number * atoms[j].atomic_number) / r;
           }
       }
       std::cout << "\nNuclear repulsion energy: " << E_nn << " Hartree\n";

       return 0;
   }

Next Steps
----------

- :doc:`03-two-electron` - Compute two-electron integrals
- :doc:`04-fock-matrix` - Build Fock matrices
