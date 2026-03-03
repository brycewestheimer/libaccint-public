.. _theory-integrals-boys:

Boys Functions
==============

Definition
----------

The Boys function of nonnegative integer order :math:`m` is

.. math::
   :label: eq-boys-main

   F_m(T) = \int_0^1 u^{2m} e^{-T u^2} \, du, \qquad T \ge 0

It appears in Coulombic Gaussian integrals after Gaussian product reduction and
Laplace-type transforms of :math:`1/r`.

Equivalent Closed Forms
-----------------------

Substitute :math:`t = Tu^2` in :eq:`eq-boys-main`:

.. math::

   F_m(T)
   = \frac{1}{2} T^{-(m+1/2)} \gamma\!\left(m+\frac{1}{2}, T\right)

where :math:`\gamma(a,x)` is the lower incomplete gamma function.

Two useful limits:

.. math::

   F_m(0) = \frac{1}{2m+1}

.. math::

   F_m(T) \sim \frac{(2m-1)!!}{2^{m+1}}\sqrt{\pi}\,T^{-(m+1/2)}
   \quad (T\to\infty)

Series Expansion at Small :math:`T`
-----------------------------------

Expanding :math:`e^{-Tu^2}` and integrating termwise gives

.. math::
   :label: eq-boys-series

   F_m(T) = \sum_{k=0}^{\infty}
   \frac{(-T)^k}{k!\,(2m+2k+1)}

This is stable for sufficiently small :math:`T`; stop when next term is below a
target absolute tolerance.

Recurrence Relations
--------------------

From integration by parts:

.. math::
   :label: eq-boys-upward

   F_{m+1}(T) = \frac{(2m+1)F_m(T)-e^{-T}}{2T}

.. math::
   :label: eq-boys-downward

   F_{m-1}(T) = \frac{2TF_m(T)+e^{-T}}{2m-1}

Practical stability:

- upward recurrence is typically stable for moderate/large :math:`T`,
- downward recurrence is stable when seeded with an accurate high-order value,
- near :math:`T=0`, use :eq:`eq-boys-series`.

Error Propagation and Conditioning
----------------------------------

Define absolute evaluation error :math:`\delta F_m`. In RR-based integral
pipelines, propagated error is approximately linear in :math:`\delta F_m` with
coefficients determined by recurrence geometry and contraction weights.

In practice:

- low-order :math:`F_0,F_1` dominate many paths and need strict relative error,
- high-order :math:`F_m` at large :math:`T` may be tiny; absolute error control
  can matter more than relative error.

A robust implementation enforces both absolute and relative criteria.

Piecewise Evaluation Strategy
-----------------------------

A standard production strategy uses three regimes:

1. **Small** :math:`T \le T_s`: series :eq:`eq-boys-series`.
2. **Medium** :math:`T_s < T < T_l`: interpolation or minimax approximation for
   :math:`F_0`, then recurrence for the rest.
3. **Large** :math:`T \ge T_l`: asymptotic seed at :math:`m_{\max}` followed by
   downward recurrence :eq:`eq-boys-downward`.

`T_s` and `T_l` should be chosen from measured error, not hard-coded folklore.

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   struct BoysConfig {
     double t_small;
     double t_large;
     double abs_tol;
     double rel_tol;
   };

   // F has size m_max + 1
   void boys_eval(int m_max, double T, const BoysConfig& cfg, double* F) {
     if (T <= cfg.t_small) {
       // Direct small-T series for each m
       for (int m = 0; m <= m_max; ++m) {
         double term = 1.0 / (2.0 * m + 1.0);
         double sum = term;
         for (int k = 1; k < 256; ++k) {
           term *= -T / k;
           double add = term / (2.0 * m + 2.0 * k + 1.0);
           sum += add;
           if (std::abs(add) <= cfg.abs_tol + cfg.rel_tol * std::abs(sum)) break;
         }
         F[m] = sum;
       }
       return;
     }

     if (T < cfg.t_large) {
       // Medium-T path: evaluate F0 by minimax/interp, recurse upward
       F[0] = boys_f0_medium(T); // implementation-specific approx
       double expT = std::exp(-T);
       for (int m = 0; m < m_max; ++m) {
         F[m + 1] = ((2.0 * m + 1.0) * F[m] - expT) / (2.0 * T);
       }
       return;
     }

     // Large-T path: asymptotic seed at highest m, recurse downward
     int M = m_max;
     double sqrtT = std::sqrt(T);
     double df = double_factorial(2 * M - 1);
     F[M] = 0.5 * std::sqrt(M_PI) * df / std::pow(2.0, M) * std::pow(T, -(M + 0.5));

     double expT = std::exp(-T);
     for (int m = M; m >= 1; --m) {
       F[m - 1] = (2.0 * T * F[m] + expT) / (2.0 * m - 1.0);
     }
   }

SIMD/GPU Batch Evaluation
-------------------------

For integral kernels, single-value calls are too expensive. Use batch APIs:

.. code-block:: cpp

   void boys_eval_batch(int n, int m_max, const double* T, double* F);

Recommended layout:

- `T[i]` contiguous,
- `F[i*(m_max+1) + m]` contiguous in `m` for recurrence locality,
- branch binning by `T` regime to reduce divergence on GPU.

Validation Protocol
-------------------

1. Compare against high-precision gamma-function reference on dense `(m,T)` grids.
2. Verify recurrence self-consistency with round-trip checks.
3. Sweep boundary neighborhoods around `T_s`, `T_l` for continuity.
4. Benchmark vectorized throughput and branch-mix sensitivity.

Cross References
----------------

- One-electron Coulomb integrals: :doc:`06_one_electron_integrals`
- OS/RR dependence on :math:`F_m`: :doc:`08_obara_saika_method`,
  :doc:`07_recurrence_relations`
- Stability policy: :doc:`21_numerical_stability`
