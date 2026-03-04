.. _theory-integrals-density-fitting:

Density Fitting (RI)
====================

Overview
--------

Density fitting (resolution of the identity, RI) approximates four-center ERIs
with an auxiliary basis expansion, reducing cost and memory in Coulomb/exchange
workflows.

Variational Derivation
----------------------

Approximate AO-pair density by auxiliary basis:

.. math::

   \rho_{\mu\nu}(\mathbf{r}) = \chi_\mu(\mathbf{r})\chi_\nu(\mathbf{r})
   \approx \sum_P C_{\mu\nu}^P\,\tilde{\chi}_P(\mathbf{r})

Minimize Coulomb-metric residual:

.. math::

   \min_C
   \left\|
   \rho_{\mu\nu} - \sum_P C_{\mu\nu}^P\tilde{\chi}_P
   \right\|_J^2

Normal equations:

.. math::
   :label: eq-df-normal

   \sum_Q (P|Q) C_{\mu\nu}^Q = (P|\mu\nu)

with

.. math::

   (P|Q)=\iint \tilde{\chi}_P(1)\frac{1}{r_{12}}\tilde{\chi}_Q(2) d1 d2

.. math::

   (P|\mu\nu)=\iint \tilde{\chi}_P(1)\frac{1}{r_{12}}\chi_\mu(2)\chi_\nu(2) d1 d2

ERI Reconstruction
------------------

Using :eq:`eq-df-normal`:

.. math::

   (\mu\nu|\lambda\sigma)
   \approx
   \sum_{PQ}(\mu\nu|P)(P|Q)^{-1}(Q|\lambda\sigma)

Factorized form using metric Cholesky :math:`(P|Q)=LL^T`:

.. math::

   B_{\mu\nu}^X = \sum_P (\mu\nu|P)(L^{-1})_{PX},
   \qquad
   (\mu\nu|\lambda\sigma) \approx \sum_X B_{\mu\nu}^X B_{\lambda\sigma}^X

Algorithmic Pipeline
--------------------

1. Build two-center metric `(P|Q)`.
2. Factorize metric with conditioning safeguards.
3. Build three-center tensors `(mu nu|P)` in blocks.
4. Whiten/project into factor form `B_{mu nu}^X`.
5. Consume factors in J/K/Fock builders.

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   void build_df_factors(const Basis& ao, const Basis& aux, DFFactors& B) {
     Matrix M = build_metric_2c2e(aux);            // (P|Q)
     Factor L = stable_factorize_metric(M);        // Cholesky/eigen with cutoff

     for (AOPairBlock blk : ao_pair_blocks(ao)) {
       Matrix T = build_3c2e_block(blk, aux);      // (mu nu|P)
       Matrix W = solve_right_triangular(T, L);    // T * L^{-T}
       store_df_block(blk, W, B);
     }
   }

Conditioning and Regularization
-------------------------------

Auxiliary metric may be ill-conditioned. Common safeguards:

- pivoted factorization,
- eigenvalue thresholding,
- reproducible regularization policy documented to users.

Accuracy and Error Budget
-------------------------

DF error is controlled by auxiliary basis quality and metric thresholding.
Validation should report error in:

- total energies,
- gradients,
- selected tensor norms relative to full ERI baseline.

Cross References
----------------

- Cholesky alternative: :doc:`20_cholesky_decomposition`
- Integral classes: :doc:`04_integral_classes`
- Validation harness: :doc:`23_validation_and_benchmarking`
