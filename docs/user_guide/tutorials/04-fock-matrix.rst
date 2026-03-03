.. _tutorial-fock-matrix:

Tutorial 4: Fock Matrix Construction
====================================

This tutorial covers building Fock matrices using LibAccInt's compute-and-consume
pattern for efficient memory usage.

The Fock Matrix
---------------

In Hartree-Fock theory, the Fock matrix is:

.. math::

   F_{\mu\nu} = H^{\text{core}}_{\mu\nu} + G_{\mu\nu}

where the two-electron contribution is:

.. math::

   G_{\mu\nu} = \sum_{\lambda\sigma} D_{\lambda\sigma} \left[
   2(\mu\nu|\lambda\sigma) - (\mu\lambda|\nu\sigma)
   \right]

This can be split into Coulomb (J) and exchange (K) terms:

.. math::

   J_{\mu\nu} &= \sum_{\lambda\sigma} D_{\lambda\sigma} (\mu\nu|\lambda\sigma) \\
   K_{\mu\nu} &= \sum_{\lambda\sigma} D_{\lambda\sigma} (\mu\lambda|\nu\sigma) \\
   G_{\mu\nu} &= 2J_{\mu\nu} - K_{\mu\nu}

The Compute-and-Consume Pattern
-------------------------------

Storing the full ERI tensor requires O(N⁴) memory, which is prohibitive for large
molecules. LibAccInt uses a **compute-and-consume** pattern where integrals are:

1. Computed in batches
2. Immediately contracted into J/K matrices
3. Discarded without storage

This reduces memory from O(N⁴) to O(N²).

Using FockBuilder
-----------------

Basic Usage
~~~~~~~~~~~

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>
   #include <vector>

   using namespace libaccint;

   // Setup molecule and basis
   std::vector<data::Atom> atoms = {
       {8, {0.0, 0.0, 0.0}},
       {1, {0.0, 1.43233673, -1.10866041}},
       {1, {0.0, -1.43233673, -1.10866041}}
   };
   BasisSet basis = data::create_sto3g(atoms);
   Engine engine(basis);

   Size nbf = basis.n_basis_functions();

   // Create initial density matrix (e.g., from diagonalizing H_core)
   std::vector<Real> D(nbf * nbf, 0.0);
   // ... populate D ...

   // Create FockBuilder
   consumers::FockBuilder fock(basis);

   // Set the density matrix
   fock.set_density(D);

   // Compute Fock matrix (computes and consumes ERIs internally)
   engine.compute(Operator::coulomb(), fock);

   // Get results
   std::vector<Real> J = fock.get_coulomb();
   std::vector<Real> K = fock.get_exchange();

   // Build G = 2J - K
   std::vector<Real> G(nbf * nbf);
   for (Size i = 0; i < nbf * nbf; ++i) {
       G[i] = 2.0 * J[i] - K[i];
   }

Multiple SCF Iterations
~~~~~~~~~~~~~~~~~~~~~~~

FockBuilder is designed for iterative use:

.. code-block:: cpp

   consumers::FockBuilder fock(basis);

   for (int iter = 0; iter < max_iter; ++iter) {
       // Update density
       fock.set_density(D);

       // Reset accumulators
       fock.reset();

       // Compute new Fock matrix
       engine.compute(Operator::coulomb(), fock);

       // Form F = H_core + 2J - K
       for (Size i = 0; i < nbf * nbf; ++i) {
           F[i] = H_core[i] + 2.0 * fock.get_coulomb()[i] - fock.get_exchange()[i];
       }

       // Diagonalize F, update D, check convergence...
       if (converged) break;
   }

GPU-Accelerated Fock Build
--------------------------

When a GPU is available, FockBuilder automatically uses device-side accumulation
for maximum performance:

.. code-block:: cpp

   // Create engine with GPU preference
   Engine engine(basis);
   engine.set_dispatch_config(DispatchConfig{
       .prefer_gpu = true,
       .min_work_for_gpu = 100
   });

   consumers::FockBuilder fock(basis);
   fock.set_density(D);

   // This will use GPU if available
   engine.compute(Operator::coulomb(), fock, BackendHint::PreferGPU);

The GPU implementation:

- Keeps density matrix on device
- Accumulates J/K on device
- Only transfers final matrices back to host

Custom Consumers
----------------

You can create custom consumers for other contractions:

.. code-block:: cpp

   // Consumer interface (simplified)
   class MyConsumer {
   public:
       void accumulate(
           const ShellSetQuartet& quartet,
           std::span<const Real> integrals);

       void finalize();
   };

Complete SCF Example
--------------------

Here's a minimal restricted Hartree-Fock implementation:

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>
   #include <Eigen/Dense>
   #include <iostream>

   using namespace libaccint;

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
       int n_electrons = 10;  // H2O
       int n_occupied = n_electrons / 2;

       // Compute one-electron integrals
       std::vector<Real> S_vec, T_vec, V_vec;
       engine.compute(OneElectronOperator(Operator::overlap()), S_vec);
       engine.compute(OneElectronOperator(Operator::kinetic()), T_vec);

       PointChargeParams charges;
       for (const auto& atom : atoms) {
           charges.x.push_back(atom.position.x);
           charges.y.push_back(atom.position.y);
           charges.z.push_back(atom.position.z);
           charges.charge.push_back(static_cast<Real>(atom.atomic_number));
       }
       engine.compute(OneElectronOperator(Operator::nuclear(charges)), V_vec);

       // Convert to Eigen matrices
       Eigen::Map<Eigen::MatrixXd> S(S_vec.data(), nbf, nbf);
       Eigen::Map<Eigen::MatrixXd> H(T_vec.data(), nbf, nbf);
       Eigen::Map<Eigen::MatrixXd> V(V_vec.data(), nbf, nbf);
       H += V;  // H_core = T + V

       // Symmetric orthogonalization
       Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(S);
       Eigen::MatrixXd X = solver.eigenvectors() *
           solver.eigenvalues().array().rsqrt().matrix().asDiagonal() *
           solver.eigenvectors().transpose();

       // Initial density from core Hamiltonian
       Eigen::MatrixXd F = H;
       Eigen::MatrixXd Fp = X.transpose() * F * X;
       Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> fock_solver(Fp);
       Eigen::MatrixXd C = X * fock_solver.eigenvectors();
       Eigen::MatrixXd D = 2.0 * C.leftCols(n_occupied) *
                           C.leftCols(n_occupied).transpose();

       // SCF iteration
       consumers::FockBuilder fock(basis);
       double E_old = 0.0;

       for (int iter = 0; iter < 50; ++iter) {
           // Build Fock matrix
           std::vector<Real> D_vec(D.data(), D.data() + nbf*nbf);
           fock.set_density(D_vec);
           fock.reset();
           engine.compute(Operator::coulomb(), fock);

           auto J = fock.get_coulomb();
           auto K = fock.get_exchange();

           // F = H + 2J - K
           for (Size i = 0; i < nbf*nbf; ++i) {
               F.data()[i] = H.data()[i] + 2.0 * J[i] - K[i];
           }

           // Compute energy
           double E = 0.0;
           for (Size i = 0; i < nbf*nbf; ++i) {
               E += 0.5 * D.data()[i] * (H.data()[i] + F.data()[i]);
           }

           std::cout << "Iter " << iter << ": E = " << E << "\n";

           if (std::abs(E - E_old) < 1e-8) {
               std::cout << "Converged!\n";
               break;
           }
           E_old = E;

           // Diagonalize
           Fp = X.transpose() * F * X;
           fock_solver.compute(Fp);
           C = X * fock_solver.eigenvectors();
           D = 2.0 * C.leftCols(n_occupied) *
               C.leftCols(n_occupied).transpose();
       }

       return 0;
   }

Next Steps
----------

- :doc:`05-gpu-acceleration` - Learn about GPU backends
- :doc:`/api/cpp/consumers` - Complete consumer and FockBuilder API reference
