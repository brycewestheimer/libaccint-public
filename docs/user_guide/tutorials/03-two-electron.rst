.. _tutorial-two-electron:

Tutorial 3: Two-Electron Integrals
==================================

This tutorial covers computing two-electron repulsion integrals (ERIs) and
understanding their structure.

Two-Electron Integrals
----------------------

The two-electron repulsion integral (ERI) is:

.. math::

   (\mu\nu|\lambda\sigma) = \int\int \phi_\mu(\mathbf{r}_1) \phi_\nu(\mathbf{r}_1)
   \frac{1}{|\mathbf{r}_1 - \mathbf{r}_2|}
   \phi_\lambda(\mathbf{r}_2) \phi_\sigma(\mathbf{r}_2) \, d\mathbf{r}_1 d\mathbf{r}_2

These integrals represent the Coulomb repulsion between electrons and are the most
computationally expensive part of quantum chemistry calculations.

Rys Quadrature
~~~~~~~~~~~~~~

LibAccInt uses Rys quadrature for computing ERIs, which is:

- Efficient for contracted Gaussians
- Well-suited for GPU parallelization
- Numerically stable across all angular momenta

Computing Two-Electron Integrals
--------------------------------

Shell Quartet Computation
~~~~~~~~~~~~~~~~~~~~~~~~~

Two-electron integrals are computed over quartets of shells:

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>

   using namespace libaccint;

   // Setup
   std::vector<data::Atom> atoms = {
       {8, {0.0, 0.0, 0.0}},
       {1, {0.0, 1.43233673, -1.10866041}},
       {1, {0.0, -1.43233673, -1.10866041}}
   };
   BasisSet basis = data::create_sto3g(atoms);
   Engine engine(basis);

   // Create buffer for integrals
   TwoElectronBuffer<0> buffer;  // <0> means no derivatives

   // Compute ERI for shell quartet (0, 0, 0, 0)
   engine.compute(Operator::coulomb(),
                  basis.shell(0), basis.shell(0),
                  basis.shell(0), basis.shell(0),
                  buffer);

   // Access the (0,0|0,0) integral
   std::cout << "(00|00) = " << buffer.data()[0] << "\n";

Integral Indexing
~~~~~~~~~~~~~~~~~

For a shell quartet (a, b, c, d), the buffer stores integrals in row-major order:

.. code-block:: cpp

   // Shell sizes
   Size na = basis.shell(a).n_functions();
   Size nb = basis.shell(b).n_functions();
   Size nc = basis.shell(c).n_functions();
   Size nd = basis.shell(d).n_functions();

   // Access integral (i, j | k, l)
   // where i ∈ [0, na), j ∈ [0, nb), k ∈ [0, nc), l ∈ [0, nd)
   Size idx = ((i * nb + j) * nc + k) * nd + l;
   Real eri = buffer.data()[idx];

Full ERI Tensor
~~~~~~~~~~~~~~~

For small systems, you can compute the full ERI tensor:

.. code-block:: cpp

   Size nbf = basis.n_basis_functions();
   Size n_shells = basis.n_shells();

   // Allocate full tensor (can be very large!)
   std::vector<Real> eri_tensor(nbf * nbf * nbf * nbf, 0.0);

   TwoElectronBuffer<0> buffer;

   for (Size a = 0; a < n_shells; ++a) {
       Size start_a = basis.shell(a).function_index();
       Size na = basis.shell(a).n_functions();

       for (Size b = 0; b < n_shells; ++b) {
           Size start_b = basis.shell(b).function_index();
           Size nb = basis.shell(b).n_functions();

           for (Size c = 0; c < n_shells; ++c) {
               Size start_c = basis.shell(c).function_index();
               Size nc = basis.shell(c).n_functions();

               for (Size d = 0; d < n_shells; ++d) {
                   Size start_d = basis.shell(d).function_index();
                   Size nd = basis.shell(d).n_functions();

                   engine.compute(Operator::coulomb(),
                                  basis.shell(a), basis.shell(b),
                                  basis.shell(c), basis.shell(d),
                                  buffer);

                   // Copy to full tensor
                   Size idx = 0;
                   for (Size i = 0; i < na; ++i) {
                       for (Size j = 0; j < nb; ++j) {
                           for (Size k = 0; k < nc; ++k) {
                               for (Size l = 0; l < nd; ++l, ++idx) {
                                   Size full_idx =
                                       (((start_a + i) * nbf + (start_b + j)) * nbf
                                        + (start_c + k)) * nbf + (start_d + l);
                                   eri_tensor[full_idx] = buffer.data()[idx];
                               }
                           }
                       }
                   }
               }
           }
       }
   }

ERI Symmetry
------------

ERIs have 8-fold symmetry:

.. math::

   (\mu\nu|\lambda\sigma) = (\nu\mu|\lambda\sigma) = (\mu\nu|\sigma\lambda)
   = (\nu\mu|\sigma\lambda) = (\lambda\sigma|\mu\nu) = ...

You can exploit this to reduce computation by a factor of ~8:

.. code-block:: cpp

   // Only compute unique quartets
   for (Size a = 0; a < n_shells; ++a) {
       for (Size b = 0; b <= a; ++b) {            // b <= a
           for (Size c = 0; c <= a; ++c) {        // c <= a
               Size d_max = (c == a) ? b : c;
               for (Size d = 0; d <= d_max; ++d) {  // symmetry restricted
                   // Compute (a,b|c,d)
                   engine.compute(Operator::coulomb(),
                                  basis.shell(a), basis.shell(b),
                                  basis.shell(c), basis.shell(d),
                                  buffer);
                   // Apply permutations to fill all 8 equivalent integrals
               }
           }
       }
   }

Direct SCF (Recommended)
------------------------

For large molecules, **never** store the full ERI tensor. Instead, use the
compute-and-consume pattern via FockBuilder:

.. code-block:: cpp

   consumers::FockBuilder fock(basis);
   fock.set_density(D);  // Set density matrix

   // Compute ERIs and contract directly into Fock matrix
   engine.compute(Operator::coulomb(), fock);

   auto J = fock.get_coulomb();    // Coulomb matrix
   auto K = fock.get_exchange();   // Exchange matrix

See :doc:`04-fock-matrix` for details.

Complete Example
----------------

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>
   #include <iostream>
   #include <iomanip>

   using namespace libaccint;

   int main() {
       // Minimal example: H2 molecule
       std::vector<data::Atom> atoms = {
           {1, {0.0, 0.0, 0.0}},
           {1, {0.0, 0.0, 1.4}}  // 1.4 Bohr separation
       };

       BasisSet basis = data::create_sto3g(atoms);
       Engine engine(basis);

       Size nbf = basis.n_basis_functions();  // 2 for H2/STO-3G
       std::cout << "H2/STO-3G: " << nbf << " basis functions\n\n";

       TwoElectronBuffer<0> buffer;

       // Print all unique ERIs
       std::cout << "Two-electron integrals (μν|λσ):\n";
       for (Size a = 0; a < basis.n_shells(); ++a) {
           for (Size b = 0; b <= a; ++b) {
               for (Size c = 0; c <= a; ++c) {
                   Size d_max = (c == a) ? b : c;
                   for (Size d = 0; d <= d_max; ++d) {
                       engine.compute(Operator::coulomb(),
                                      basis.shell(a), basis.shell(b),
                                      basis.shell(c), basis.shell(d),
                                      buffer);

                       std::cout << "(" << a << b << "|" << c << d << ") = "
                                 << std::fixed << std::setprecision(6)
                                 << buffer.data()[0] << "\n";
                   }
               }
           }
       }

       return 0;
   }

Next Steps
----------

- :doc:`04-fock-matrix` - Build Fock matrices efficiently
- :doc:`05-gpu-acceleration` - Accelerate ERI computation on GPU
