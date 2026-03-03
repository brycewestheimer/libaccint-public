.. _theory-integrals-cholesky:

Cholesky Decomposition of ERIs
==============================

Overview
--------

Pivoted Cholesky decomposition (CD) approximates the ERI matrix with
threshold-controlled low rank and avoids pre-defined auxiliary bases.

ERI Matrix Formulation
----------------------

Flatten pair index :math:`(\mu\nu)\to I`, define symmetric PSD matrix
:math:`V_{IJ}` from ERIs.

Goal:

.. math::

   V \approx LL^T

with columns of :math:`L` (Cholesky vectors) generated iteratively until
residual criterion is met.

Pivoted CD Derivation
---------------------

At iteration `k`, choose pivot :math:`p` maximizing residual diagonal
:math:`d_I^{(k)}`.

New vector element for index `I`:

.. math::

   L_{Ik} =
   \frac{1}{\sqrt{d_p^{(k)}}}
   \left(V_{Ip} - \sum_{j<k} L_{Ij}L_{pj}\right)

Residual update:

.. math::

   d_I^{(k+1)} = d_I^{(k)} - L_{Ik}^2

Stop when :math:`\max_I d_I^{(k)} < \tau`.

Threshold :math:`\tau` sets decomposition accuracy.

Algorithmic Pipeline
--------------------

1. initialize residual diagonal,
2. iterative pivot select + vector build,
3. residual update,
4. optional block/parallel update,
5. finalize Cholesky vectors for downstream use.

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   void pivoted_cd(const ERIAccess& eri, double tau, Matrix& L) {
     Vector d = eri.diagonal();
     std::vector<int> pivots;

     while (true) {
       int p = argmax(d);
       if (d[p] < tau) break;

       int k = (int)pivots.size();
       pivots.push_back(p);
       ensure_col(L, k);

       double inv_sqrt = 1.0 / std::sqrt(d[p]);
       for (int I = 0; I < eri.size(); ++I) {
         double vip = eri(I, p);
         double proj = 0.0;
         for (int j = 0; j < k; ++j) proj += L(I, j) * L(p, j);
         L(I, k) = (vip - proj) * inv_sqrt;
       }

       for (int I = 0; I < eri.size(); ++I) d[I] -= L(I, k) * L(I, k);
     }
   }

Blocked and Distributed CD
--------------------------

For large systems, matrix-free and blocked formulations are required:

- block pivot selection,
- batched column updates,
- distributed residual tracking.

Communication patterns and reproducibility policy should be explicitly designed.

Relation to DF
--------------

DF and CD both produce low-rank Coulomb factors. CD is adaptive and data-driven;
DF is auxiliary-basis driven. Choice depends on ecosystem, accuracy needs, and
performance characteristics.

Validation Protocol
-------------------

1. residual-diagonal monotonic decrease checks,
2. reconstructed tensor error norms,
3. energy/gradient deviations vs full ERI and DF baselines,
4. rank-scaling curves vs threshold.

Cross References
----------------

- DF methodology: :doc:`19_density_fitting`
- Method selection criteria: :doc:`24_method_selection_guide`
- Benchmarking: :doc:`23_validation_and_benchmarking`
