.. _batch-computation:

Batch Computation Guide
=======================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

GPU acceleration delivers the largest speedups when the hardware can process
many integrals in a single kernel launch.  LibAccInt achieves this by grouping
shells that share the same angular momentum (AM) and contraction degree (*K*,
the number of primitive Gaussians per shell) into **ShellSets**.  Every shell
inside a ShellSet is stored in Structure-of-Arrays (SoA) layout so that
GPU threads can coalesce memory accesses across the batch.

Rather than computing one ``(shell_a, shell_b)`` pair at a time, the engine
iterates over **ShellSetPairs** and **ShellSetQuartets** — work units that
contain *all* shell combinations with the same AM/contraction signature.  This
turns many tiny kernel launches into a small number of large, GPU-friendly
dispatches.

The ShellSet Hierarchy
----------------------

ShellSet
~~~~~~~~

A ``ShellSet`` groups every shell in the basis that has the same angular
momentum and the same number of primitive Gaussians.  Internally the shell
data (centres, exponents, contraction coefficients, basis-function offsets)
is stored in SoA layout for GPU-coalesced access.

Key accessors:

.. code-block:: cpp

   const ShellSet& ss = /* ... */;
   int am  = ss.angular_momentum();        // e.g. 0 (s), 1 (p), 2 (d)
   int K   = ss.n_primitives_per_shell();   // contraction degree
   Size ns = ss.n_shells();                 // number of shells in this set

ShellSetPair
~~~~~~~~~~~~

A ``ShellSetPair`` pairs a *bra* and a *ket* ``ShellSet``.  The total number
of shell pairs in the batch is the product of the two shell counts:
``n_pairs() = n_shells_a × n_shells_b``.

ShellSetPairs are the work units for **one-electron** integrals (overlap,
kinetic energy, nuclear attraction).

.. code-block:: cpp

   const ShellSetPair& ssp = /* ... */;
   const ShellSet& bra = ssp.shell_set_a();
   const ShellSet& ket = ssp.shell_set_b();
   Size n = ssp.n_pairs();                  // total shell pairs in batch

ShellSetQuartet
~~~~~~~~~~~~~~~

A ``ShellSetQuartet`` pairs two ``ShellSetPairs`` (bra and ket).  The total
number of shell quartets is the product of the four shell counts:
``n_quartets() = n_shells_a × n_shells_b × n_shells_c × n_shells_d``.

ShellSetQuartets are the work units for **two-electron** integrals (ERIs).
They are the primary GPU work unit because the quartic scaling produces large
batches even for modest basis sets.

.. code-block:: cpp

   const ShellSetQuartet& ssq = /* ... */;
   const ShellSetPair& bra = ssq.bra_pair();
   const ShellSetPair& ket = ssq.ket_pair();
   Size n = ssq.n_quartets();               // total shell quartets

Getting Work Lists
------------------

``BasisSet`` lazily generates and caches the complete sets of ShellSetPairs
and ShellSetQuartets.  The first call computes the work list; every subsequent
call returns the cached vector in *O(1)*.

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>

   using namespace libaccint;

   BasisSet basis = data::load_basis_set("aug-cc-pvdz", atoms);

   // 1-electron work list (upper-triangle pairs)
   const auto& pairs = basis.shell_set_pairs();

   // 2-electron work list (upper-triangle quartets)
   const auto& quartets = basis.shell_set_quartets();

.. note::

   ``shell_set_pairs()`` and ``shell_set_quartets()`` use a mutex-protected
   cache and are safe to call concurrently on a shared ``BasisSet``.  However,
   ``clear_work_unit_cache()`` invalidates previously returned references and
   must only be called when no computation is using those work lists.

Computing 1-Electron Integrals
------------------------------

Convenience Methods
~~~~~~~~~~~~~~~~~~~

For the most common one-electron integrals the ``Engine`` provides dedicated
methods that handle the full basis automatically:

.. code-block:: cpp

   Engine engine(basis);
   std::vector<Real> S, T, V;

   engine.compute_overlap_matrix(S);
   engine.compute_kinetic_matrix(T);

   // Nuclear attraction requires point-charge parameters
   PointChargeParams charges;
   for (const auto& atom : atoms) {
       charges.x.push_back(atom.position.x);
       charges.y.push_back(atom.position.y);
       charges.z.push_back(atom.position.z);
       charges.charge.push_back(static_cast<Real>(atom.atomic_number));
   }
   engine.compute_nuclear_matrix(charges, V);

   // Core Hamiltonian in a single call (T + V)
   std::vector<Real> H_core;
   engine.compute_core_hamiltonian(charges, H_core);

Every convenience method accepts an optional ``BackendHint`` argument:

.. code-block:: cpp

   engine.compute_overlap_matrix(S, BackendHint::ForceCPU);
   engine.compute_overlap_matrix(S, BackendHint::PreferGPU);

Manual ShellSetPair Loop
~~~~~~~~~~~~~~~~~~~~~~~~

For advanced use cases — partial basis evaluation, custom screening, or mixed
backend control — iterate the ShellSetPairs explicitly:

.. code-block:: cpp

   std::vector<Real> result(nbf * nbf, 0.0);

   engine.compute_1e(
       OneElectronOperator(Operator::overlap()),
       basis.shell_set_pairs(),   // std::span<const ShellSetPair>
       result);

Or drive individual pairs with per-pair dispatch hints:

.. code-block:: cpp

   for (const auto& pair : basis.shell_set_pairs()) {
       engine.compute_shell_set_pair(Operator::overlap(), pair, result,
                                     BackendHint::Auto);
   }

Computing 2-Electron Integrals
------------------------------

The Compute-and-Consume Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two-electron integrals produce *O(N⁴)* data, far too much to store for
realistic basis sets.  LibAccInt therefore uses a **compute-and-consume**
pattern: the engine computes a batch of ERIs and immediately passes them to
a *consumer* (e.g. ``FockBuilder``) that accumulates contributions into
compact matrices.

.. code-block:: cpp

   #include <libaccint/consumers/fock_builder.hpp>

   Size nbf = basis.n_basis_functions();
   consumers::FockBuilder fock(nbf);
   fock.set_density(D.data(), nbf);

   // Full basis — engine iterates all ShellSetQuartets internally
   engine.compute_and_consume(Operator::coulomb(), fock);

   // Retrieve results
   auto J = fock.get_coulomb_matrix();   // std::span<const Real>
   auto K = fock.get_exchange_matrix();  // std::span<const Real>

Custom Work Lists
~~~~~~~~~~~~~~~~~

You can supply your own subset of quartets — useful for integral screening
or distributed computation across nodes:

.. code-block:: cpp

   const auto& all_quartets = basis.shell_set_quartets();

   // Example: take only the first half
   std::span<const ShellSetQuartet> subset(
       all_quartets.data(), all_quartets.size() / 2);

   engine.compute_and_consume(Operator::coulomb(), subset, fock);

Parallel Computation
~~~~~~~~~~~~~~~~~~~~

For CPU-parallel Fock builds, use the parallel API.  The consumer must be
prepared for multi-threaded accumulation:

.. code-block:: cpp

   consumers::FockBuilder fock(nbf);
   fock.set_density(D.data(), nbf);

   // Prepare thread-local buffers (0 = auto-detect thread count)
   fock.prepare_parallel(0);

   engine.compute_and_consume_parallel(Operator::coulomb(), fock,
                                       /* n_threads = */ 4);

   // Thread-local buffers are reduced automatically
   auto J = fock.get_coulomb_matrix();
   auto K = fock.get_exchange_matrix();

Manual ShellSetQuartet Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the finest-grained control, iterate quartets yourself:

.. code-block:: cpp

   for (const auto& quartet : basis.shell_set_quartets()) {
       engine.compute_shell_set_quartet(Operator::coulomb(), quartet, fock);
   }

This lets you interleave screening, logging, or custom scheduling between
dispatches.

GPU Dispatch Heuristics
-----------------------

The ``Engine`` uses a ``DispatchPolicy`` to decide whether each work unit
runs on the CPU or GPU.  The policy considers batch size, total angular
momentum, primitive count, and any user-provided hint.

Tuning DispatchConfig
~~~~~~~~~~~~~~~~~~~~~

Pass a ``DispatchConfig`` to the ``Engine`` constructor to adjust thresholds:

.. code-block:: cpp

   DispatchConfig config;
   config.min_gpu_batch_size = 16;     // Quartets below this → CPU
   config.min_gpu_primitives = 1000;   // Primitive count threshold
   config.high_am_threshold = 4;       // AM ≥ 4 always prefers GPU
   config.min_gpu_shells = 10;         // Min shells for full-basis GPU

   Engine engine(basis, config);

   // Config can also be changed at runtime
   config.min_gpu_batch_size = 32;
   engine.set_dispatch_config(config);

BackendHint
~~~~~~~~~~~

Every compute method accepts a ``BackendHint`` to override the dispatch
policy for that particular call:

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Hint
     - Behaviour
   * - ``Auto``
     - Let the dispatch policy decide (default)
   * - ``ForceCPU``
     - Always use the CPU backend
   * - ``ForceGPU``
     - Always use the GPU backend (throws if unavailable)
   * - ``PreferCPU``
     - Prefer CPU, but use GPU if the policy considers it beneficial
   * - ``PreferGPU``
     - Prefer GPU, fall back to CPU if unavailable

Diagnostic Tracing
~~~~~~~~~~~~~~~~~~

Set the ``LIBACCINT_TRACE_BATCH`` environment variable to see per-dispatch
information and a summary on ``stderr``.  When the variable is unset there is
**zero** runtime overhead (no mutex, no I/O).

.. code-block:: bash

   LIBACCINT_TRACE_BATCH=1 ./my_program

Example output:

.. code-block:: text

   [BATCH_TRACE] 2e Coulomb (00|00) n_quartets=16 strategy=batched -> CPU
   [BATCH_TRACE] 2e Coulomb (01|01) n_quartets=81 strategy=batched -> GPU
   ...
   [BATCH_TRACE] ========== Summary ==========
   [BATCH_TRACE] 1e dispatches: 6 (GPU: 0) total_pairs: 42
   [BATCH_TRACE] 2e dispatches: 10 (GPU: 7) total_quartets: 1296
   [BATCH_TRACE] avg 2e batch size: 129
   [BATCH_TRACE] ================================

Worked Example: J/K Matrices for Water
---------------------------------------

The following complete program computes the Coulomb and exchange matrices for
a water molecule using the cc-pVDZ basis set.

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>
   #include <libaccint/data/basis_parser.hpp>
   #include <libaccint/consumers/fock_builder.hpp>

   #include <iostream>
   #include <vector>

   using namespace libaccint;

   int main() {
       // Define water geometry (coordinates in Bohr)
       std::vector<data::Atom> atoms = {
           {8, {0.0, 0.0, 0.0}},           // O
           {1, {0.0, 1.43, -1.11}},         // H
           {1, {0.0, -1.43, -1.11}}         // H
       };

       // Load basis set
       BasisSet basis = data::load_basis_set("cc-pvdz", atoms);
       Size nbf = basis.n_basis_functions();

       // Create engine (default dispatch heuristics)
       Engine engine(basis);

       // --- 1-electron integrals ---
       std::vector<Real> S, T, V;
       engine.compute_overlap_matrix(S);
       engine.compute_kinetic_matrix(T);

       PointChargeParams charges;
       for (const auto& atom : atoms) {
           charges.x.push_back(atom.position.x);
           charges.y.push_back(atom.position.y);
           charges.z.push_back(atom.position.z);
           charges.charge.push_back(static_cast<Real>(atom.atomic_number));
       }
       engine.compute_nuclear_matrix(charges, V);

       // --- 2-electron integrals with FockBuilder ---
       consumers::FockBuilder fock(nbf);

       // Use identity density for the first SCF iteration
       std::vector<Real> D(nbf * nbf, 0.0);
       for (Size i = 0; i < nbf; ++i) D[i * nbf + i] = 1.0;
       fock.set_density(D.data(), nbf);

       // Compute J and K matrices (batch dispatch, GPU if available)
       engine.compute_and_consume(Operator::coulomb(), fock);

       auto J = fock.get_coulomb_matrix();
       auto K = fock.get_exchange_matrix();

       // --- Build Fock matrix: F = H_core + J - 0.5*K (RHF) ---
       std::vector<Real> H_core(nbf * nbf);
       for (Size i = 0; i < nbf * nbf; ++i) H_core[i] = T[i] + V[i];

       auto F = fock.get_fock_matrix(H_core, 0.5);

       std::cout << "Basis functions: " << nbf << "\n"
                 << "ShellSetQuartets: " << basis.shell_set_quartets().size() << "\n"
                 << "Fock matrix F[0][0] = " << F[0] << "\n";

       return 0;
   }

.. seealso::

   - :doc:`getting-started` — Installation and first program
   - :doc:`cookbook` — Additional recipes for common tasks
   - :doc:`/theory/gpu_optimization` — Theory behind GPU dispatch decisions

Non-Consumer Compute Methods
-----------------------------

For cases where you need raw integral values rather than accumulated matrices,
LibAccInt provides **non-consumer compute methods** that return
``IntegralBuffer`` objects directly. These bypass the consumer callback pattern
and are useful for debugging, validation, or algorithms that need explicit
access to individual integral values.

Single Quartet/Pair
~~~~~~~~~~~~~~~~~~~

Compute integrals for a single work unit and get results back directly:

.. code-block:: cpp

   Engine engine(basis);

   // Two-electron: single ShellSetQuartet → IntegralBuffer
   const auto& quartets = basis.shell_set_quartets();
   IntegralBuffer buf = engine.compute(Operator::coulomb(), quartets[0]);

   // One-electron: single ShellSetPair → IntegralBuffer
   const auto& pairs = basis.shell_set_pairs();
   IntegralBuffer buf_1e = engine.compute(Operator::overlap(), pairs[0]);

   // Access the raw integral values
   for (Size i = 0; i < buf.size(); ++i) {
       Real value = buf[i];
   }

Full ERI Tensor
~~~~~~~~~~~~~~~

For small basis sets, compute the complete 4-index ERI tensor as a flat
vector. This is convenient for validation and educational use but scales as
*O(nbf⁴)* in both time and memory. A safety limit of ``nbf ≤ 200`` is
enforced.

.. code-block:: cpp

   // Full tensor (row-major: index = i*nbf³ + j*nbf² + k*nbf + l)
   auto eri = engine.compute_eri_tensor();

   // Single quartet block with local indexing
   auto block = engine.compute_eri_block(Operator::coulomb(), quartets[0]);

Parallel and Screened Batch
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Non-consumer methods also support parallelism and screening:

.. code-block:: cpp

   // Parallel: compute all quartets across threads
   auto results = engine.compute_all_2e_parallel(Operator::coulomb());

   // Parallel: compute a subset of quartets
   auto partial = engine.compute_batch_parallel(
       Operator::coulomb(), quartets, /* n_threads = */ 4);

   // Screened: skip negligible quartets via Schwarz inequality
   ScreeningOptions opts;
   opts.threshold = 1e-10;
   auto screened = engine.compute_batch_screened(Operator::coulomb(), opts);

   // Analyze screening efficiency
   auto stats = Engine::compute_screening_statistics(screened);

When to Use Non-Consumer Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Use Case
     - Recommended Method
   * - Build Fock / gradient matrices
     - ``compute_and_consume()`` with a consumer
   * - Validate integrals against reference
     - ``compute_eri_tensor()``
   * - Access individual quartet integrals
     - ``compute()`` (single quartet)
   * - Parallel batch without accumulation
     - ``compute_batch_parallel()``
   * - Skip negligible integrals
     - ``compute_batch_screened()``
