.. _theory-integrals-screening:

Screening and Integral Bounds
=============================

Overview
--------

Screening is the principal mechanism that makes large-scale Gaussian integral
computation tractable. The central goal is to avoid evaluating shell pairs or
quartets whose contribution to target quantities is provably negligible.

Bound Hierarchy
---------------

A practical screening stack uses progressively tighter bounds:

1. shell-pair prefilter (very cheap),
2. shell-quartet bound (moderate cost),
3. primitive-level prefactor bound (optional),
4. post-contraction threshold (final truncation).

Each stage should be monotone-safe: if later stages are skipped, the early stage
must still prevent false negatives relative to configured tolerance policy.

Schwarz Inequality Derivation
-----------------------------

For ERIs:

.. math::
   :label: eq-schwarz

   |(\mu\nu|\lambda\sigma)|^2
   \le (\mu\nu|\mu\nu)(\lambda\sigma|\lambda\sigma)

Thus

.. math::

   |(\mu\nu|\lambda\sigma)|
   \le
   \sqrt{(\mu\nu|\mu\nu)}\sqrt{(\lambda\sigma|\lambda\sigma)}

At shell level, define

.. math::

   B_{ab} = \max_{\mu\in a,\nu\in b} \sqrt{(\mu\nu|\mu\nu)}

then

.. math::
   :label: eq-shell-schwarz

   |(ab|cd)| \le B_{ab}B_{cd}

This is the canonical first-stage shell-quartet screen.

Distance-Dependent Refinements
------------------------------

Schwarz can be loose for distant shell pairs. Refinements include:

- multipole-distance bounds,
- attenuated Coulomb envelopes,
- center-separation dependent prefactor bounds.

A generic refined bound can be written as

.. math::

   |(ab|cd)| \le B_{ab}B_{cd}\,f(R_{PQ})

with :math:`0 < f(R) \le 1` decaying with inter-pair distance.

Threshold Semantics and Error Budget
------------------------------------

Screening thresholds must map to user-visible error targets. Recommended policy:

- integral-level absolute threshold :math:`\tau_I`,
- property-level tolerance (energy/gradient) :math:`\tau_E,\tau_G`,
- deterministic mapping from :math:`\tau_E` to internal :math:`\tau_I`.

Avoid undocumented, method-specific hidden thresholds.

Implementation Architecture
---------------------------

Data structures:

- shell-pair table (`ab` metadata, bound, geometry features),
- sorted/bucketed pair lists by bound magnitude,
- quartet iterator that combines pair lists with bound short-circuit.

Precompute once per SCF iteration when geometry/basis fixed.

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   struct ScreenConfig {
     double tau_shell;
     double tau_quartet;
     double tau_primitive;
   };

   bool pass_shell_quartet(const ShellPair& ab,
                           const ShellPair& cd,
                           const ScreenConfig& cfg) {
     // Stage 1: Schwarz shell bound
     double ub = ab.schwarz_bound * cd.schwarz_bound;
     if (ub < cfg.tau_shell) return false;

     // Stage 2: distance refinement
     ub *= distance_decay_factor(ab.center, cd.center);
     if (ub < cfg.tau_quartet) return false;

     return true;
   }

   void screened_eri_driver(const PairList& pairs, const ScreenConfig& cfg) {
     for (auto ab : pairs) {
       for (auto cd : candidate_pairs_for(ab)) {
         if (!pass_shell_quartet(ab, cd, cfg)) continue;
         eval_shell_quartet_with_optional_primitive_screen(ab, cd, cfg.tau_primitive);
       }
     }
   }

Parallel and Load-Balance Concerns
----------------------------------

Screening increases irregularity:

- uneven survivors per shell pair,
- dynamic task sizes,
- variable arithmetic intensity.

Mitigations:

- work-stealing task schedulers,
- survivor-count aware batching,
- separate queues by estimated quartet cost.

Validation Protocol
-------------------

1. compare screened vs unscreened integrals on small reference sets,
2. sweep thresholds and measure property errors,
3. verify no false negatives relative to bound mathematics,
4. monitor survivor statistics regressions in CI.

Cross References
----------------

- ERI methods: :doc:`15_two_electron_eri_algorithms`
- Numerical policy: :doc:`21_numerical_stability`
- Benchmark procedure: :doc:`23_validation_and_benchmarking`
