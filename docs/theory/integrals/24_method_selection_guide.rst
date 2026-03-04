.. _theory-integrals-method-selection:

Method Selection Guide
======================

Overview
--------

This chapter defines a concrete dispatch framework for selecting integral
methods based on workload, accuracy requirements, and hardware characteristics.

Selection Features
------------------

Per work item, collect:

- operator class (`1e`, `4c2e`, `3c2e`, derivatives),
- shell tuple/angular momentum,
- contraction lengths,
- geometry features (distance, anisotropy),
- screening survivor density,
- backend capabilities,
- reproducibility mode.

Capability Filter
-----------------

Before scoring, remove methods that cannot support requested features.

Example:

- rotated-MD may be disabled for unsupported derivative class,
- GPU path may be unavailable for specific high-`L` tuples,
- strict deterministic mode may exclude some fused kernels.

Cost and Risk Scoring
---------------------

Compute total score:

.. math::

   S(m) = C_{time}(m) + \lambda_r C_{risk}(m) + \lambda_m C_{memory}(m)

where

- :math:`C_{time}` is calibrated runtime model,
- :math:`C_{risk}` is numerical risk estimate,
- :math:`C_{memory}` penalizes memory pressure.

Choose method with minimum score, with stable fallback if risk exceeds limit.

Recommended Baselines
---------------------

- low/moderate `L` ERI: OS/HGP or Rys by backend calibration,
- broad/high `L`: Rys often stable and regular,
- table-friendly structured workloads: MD may be advantageous,
- transfer-heavy memory-bound paths: TRn/fused transfer,
- reduced-rank workflows: DF/CD according to error-cost target.

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   Method dispatch_method(const WorkItem& w, const DispatchConfig& cfg) {
     auto candidates = capability_filter(w, cfg);

     Method best = Method::None;
     double best_score = std::numeric_limits<double>::infinity();

     for (Method m : candidates) {
       double c_time = estimate_time(m, w, cfg.models);
       double c_risk = estimate_numeric_risk(m, w);
       double c_mem  = estimate_memory_pressure(m, w);

       double score = c_time + cfg.lambda_r * c_risk + cfg.lambda_m * c_mem;
       if (score < best_score) { best_score = score; best = m; }
     }

     if (estimate_numeric_risk(best, w) > cfg.risk_limit)
       best = stable_fallback(candidates);

     return best;
   }

Calibration Process
-------------------

Dispatch models require periodic calibration on target hardware:

1. run benchmark matrix,
2. fit/update per-method cost coefficients,
3. validate predictive error,
4. update thresholds and fallback limits.

Validation of Dispatch Policy
-----------------------------

1. compare selected-path outputs to forced-method baselines,
2. verify fallback trigger correctness,
3. audit dispatch distribution over real workloads,
4. track changes over commits to detect policy drift.

Cross References
----------------

- ERI method comparison: :doc:`15_two_electron_eri_algorithms`
- Stability controls: :doc:`21_numerical_stability`
- Benchmark calibration: :doc:`23_validation_and_benchmarking`
