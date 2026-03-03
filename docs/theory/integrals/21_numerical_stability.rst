.. _theory-integrals-numerical-stability:

Numerical Stability and Error Control
=====================================

Overview
--------

Integral algorithms span extreme parameter regimes. Numerical stability must be
engineered explicitly to avoid downstream SCF failures, noisy derivatives, and
non-reproducible behavior.

Floating-Point Error Model
--------------------------

For floating-point operations with machine epsilon :math:`\varepsilon`, each
operation incurs relative perturbation :math:`\delta` with
:math:`|\delta|\lesssim\varepsilon`.

For recurrence chains and large contractions, accumulated error scales with both
operation count and conditioning of coefficients. Therefore, raw flop count is
not sufficient as a quality indicator.

Dominant Instability Sources
----------------------------

1. deep RR/HRR chains with cancellation,
2. Boys/root approximations near regime boundaries,
3. underflow/overflow in Gaussian prefactors,
4. non-deterministic reduction order in parallel execution,
5. ill-conditioned metric solves (DF/CD).

Regime Classification
---------------------

Practical engines should classify work items by risk features:

- exponent spread (`max(alpha)/min(alpha)`),
- center-distance extremes,
- angular momentum depth,
- estimated cancellation score,
- approximation branch proximity.

High-risk items can trigger stable fallback kernels.

Precision Policy
----------------

Recommended default policy:

- critical scalar intermediates in double precision,
- contraction/reduction in double precision,
- optional float output cast at end,
- deterministic mode for validation and regression tracking.

Stable Summation Strategies
---------------------------

For long sums use one of:

- pairwise reduction,
- Kahan/Neumaier compensation,
- fixed-tree deterministic reduction.

Choice depends on required reproducibility/performance tradeoff.

Boundary Continuity Controls
----------------------------

Piecewise approximations (Boys, roots/weights) require explicit boundary tests:

.. math::

   |f_{left}(T_b) - f_{right}(T_b)| < \tau_{cont}

for each branch boundary :math:`T_b` and tolerance :math:`\tau_{cont}`.

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   EvalPath choose_stable_path(const WorkItem& w) {
     Risk r = estimate_risk(w);

     if (r.high_exponent_spread || r.high_L || r.near_branch_boundary) {
       return EvalPath::Stable;
     }
     return EvalPath::Fast;
   }

   void accumulate_deterministic(const double* x, int n, double& out) {
     // fixed pairwise tree for reproducibility
     std::vector<double> tmp(x, x + n);
     while (tmp.size() > 1) {
       std::vector<double> next;
       next.reserve((tmp.size() + 1) / 2);
       for (size_t i = 0; i < tmp.size(); i += 2) {
         if (i + 1 < tmp.size()) next.push_back(tmp[i] + tmp[i + 1]);
         else next.push_back(tmp[i]);
       }
       tmp.swap(next);
     }
     out = tmp.empty() ? 0.0 : tmp[0];
   }

Validation Metrics
------------------

Track both local and global metrics:

- ULP/relative error against high-precision references,
- invariance residuals,
- energy/gradient drift in full SCF flows,
- backend-to-backend differential error.

Cross References
----------------

- Boys stability: :doc:`05_boys_functions`
- RR behavior: :doc:`07_recurrence_relations`
- Parallel reproducibility: :doc:`22_parallelization_and_acceleration`
