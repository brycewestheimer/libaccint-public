.. _theory-integrals-eri-algorithms:

Two-Electron ERI Algorithm Families
===================================

Purpose
-------

This chapter provides a method-selection and implementation comparison framework
across major ERI algorithm families: OS/HGP, Rys, MD, rotated variants, and
TRn/fused-transfer extensions.

Unified Cost Model
------------------

A practical per-shell-quartet runtime model:

.. math::

   T \approx \frac{F}{P_{\text{eff}}} + \frac{B}{BW_{\text{eff}}} + O

where

- :math:`F` = floating-point work,
- :math:`B` = bytes moved,
- :math:`O` = control/scheduling overhead,
- :math:`P_{\text{eff}}, BW_{\text{eff}}` are effective compute/bandwidth.

Different methods trade these terms differently.

Method Profiles
---------------

OS/HGP
~~~~~~

- Strengths: mature RR machinery, efficient low-to-moderate `L`, broad usage.
- Weaknesses: intermediate growth and memory movement at high `L`.

Rys
~~~

- Strengths: regular root loop, robust numerics, strong SIMD/GPU mapping.
- Weaknesses: root/weight machinery and root-count dependence.

MD
~~

- Strengths: clean Hermite formalism, reusable coefficient structures.
- Weaknesses: table/tensor overhead if layout is not optimized.

Rotated variants
~~~~~~~~~~~~~~~~

- Strengths: can improve conditioning for geometry-specific cases.
- Weaknesses: transform overhead and added complexity.

TRn/fused transfer
~~~~~~~~~~~~~~~~~~

- Strengths: lower memory traffic in transfer-heavy kernels.
- Weaknesses: high tuning complexity and potential register pressure issues.

Comparison Matrix
-----------------

.. list-table::
   :header-rows: 1

   * - Criterion
     - OS/HGP
     - Rys
     - MD
     - TRn/fused
   * - Core primitive
     - VRR/HRR
     - root-weight sum
     - Hermite contraction
     - fused transfer DAG
   * - Memory pressure
     - medium/high
     - medium
     - medium/high
     - low/medium (if tuned)
   * - Control regularity
     - schedule-dependent
     - high
     - medium
     - schedule-dependent
   * - High-L behavior
     - can degrade
     - often robust
     - mixed, layout-sensitive
     - helps transfer-heavy cases
   * - GPU friendliness
     - good with tuning
     - strong
     - good with tensor layout
     - strong if occupancy maintained

Dispatch Strategy
-----------------

Use hybrid runtime dispatch based on features:

- angular tuple,
- contraction lengths,
- screening density,
- backend/hardware,
- requested reproducibility mode.

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   Method choose_eri_method(const WorkItem& w, const DeviceInfo& dev) {
     CandidateSet c = capability_filter(w, dev);

     for (auto& m : c) {
       m.cost = estimate_cost(m, w, dev);        // F/B/O model
       m.risk = estimate_numeric_risk(m, w);     // stability score
       m.total = m.cost + risk_penalty(m.risk);
     }

     Method picked = argmin_total(c);
     if (picked.risk > RISK_LIMIT) picked = stable_fallback(c);
     return picked;
   }

Calibration and Benchmarking
----------------------------

Dispatch models must be calibrated with measured data from
:doc:`23_validation_and_benchmarking`. Static heuristics without calibration are
rarely reliable across hardware generations.

Validation Across Methods
-------------------------

Maintain differential test suites that compare all enabled methods on shared
domains, including stress regions near dispatch crossover boundaries.

Cross References
----------------

- Method chapters: :doc:`08_obara_saika_method`,
  :doc:`09_head_gordon_pople_method`,
  :doc:`10_rys_quadrature_method`,
  :doc:`11_mcmurchie_davidson_method`,
  :doc:`14_trn_fused_hrn_method`
- Dispatch policy chapter: :doc:`24_method_selection_guide`
