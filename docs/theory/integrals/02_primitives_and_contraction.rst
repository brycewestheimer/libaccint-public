.. _theory-integrals-primitives-contraction:

Primitives and Contraction
==========================

Overview
--------

Integral algorithms are derived at the primitive level and assembled into
contracted AO integrals by linear combination. Correct and efficient
contraction is central to practical performance.

Primitive-to-Contracted Linear Structure
----------------------------------------

For contracted AOs :math:`\mu,\nu,\lambda,\sigma`:

.. math::

   (\mu\nu|\lambda\sigma)
   = \sum_{p\in\mu}\sum_{q\in\nu}\sum_{r\in\lambda}\sum_{s\in\sigma}
   d_{\mu p}d_{\nu q}d_{\lambda r}d_{\sigma s}
   (pq|rs)

where :math:`d_{\mu p}=c_{\mu p}N_{\mu p}` includes contraction and
normalization.

Equivalent equations hold for one-electron classes with two primitive indices.

Contraction Types
-----------------

Segmented contraction
~~~~~~~~~~~~~~~~~~~~~

Each contracted function uses a dedicated subset of primitives.
Advantages: simple loops, less coefficient indirection.

General contraction
~~~~~~~~~~~~~~~~~~~

Multiple contracted functions share broad primitive sets.
Advantages: compact basis representations and potential reuse of primitive
intermediates.

Implementation implications:

- segmented favors direct accumulate-per-output,
- general often benefits from matrix-like batched contraction kernels.

Loop Organization Strategies
----------------------------

Two common accumulation patterns:

Primitive-first accumulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Compute primitive integral block.
2. Immediately multiply coefficients and add to contracted outputs.

Good when primitive blocks are small and cache-local.

Contract-first (intermediate) accumulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Compute many primitive blocks.
2. Perform contraction with batched linear algebra (GEMM-like).

Good when contraction lengths are large and vector hardware is strong.

Pseudocode (ERI primitive-first)
--------------------------------

.. code-block:: cpp

   for (pair_ab in shell_pairs) {
     for (pair_cd in shell_pairs) {
       if (!passes_screen(pair_ab, pair_cd)) continue;
       zero(contracted_block);
       for (p in prims_a)
         for (q in prims_b)
           for (r in prims_c)
             for (s in prims_d) {
               prim = eri_primitive(p,q,r,s);
               scale = d_ap * d_bq * d_cr * d_ds;
               contracted_block += scale * prim;
             }
       store(contracted_block);
     }
   }

Screening Placement
-------------------

Screening can occur at multiple levels:

- shell-pair bounds before entering primitive loops,
- primitive-pair prefactor magnitude checks,
- post-contraction thresholding before storage.

Early screening reduces arithmetic. Late screening improves robustness when
cancellation is strong. Most engines use both.

Memory Layout and Vectorization
-------------------------------

Recommended data layout for primitives:

- exponents, coefficients, normalization in separate arrays,
- contiguous primitive records per shell,
- precomputed pair invariants in compact SoA arrays.

This supports:

- SIMD over primitive pairs,
- GPU thread mapping over primitive combinations,
- low-overhead gather for coefficient products.

Numerical Concerns
------------------

Contraction accumulations are vulnerable to cancellation.

Mitigations:

- accumulate in higher precision than storage precision,
- use deterministic summation order,
- optionally use compensation for extreme basis sets.

See :doc:`21_numerical_stability` for detailed error control.

Cross References
----------------

- Basis definitions: :doc:`01_gaussian_basis`
- Recurrence backends: :doc:`07_recurrence_relations`
- Parallel contraction strategy: :doc:`22_parallelization_and_acceleration`
