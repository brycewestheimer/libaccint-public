.. _theory-integrals-validation:

Validation, Testing, and Benchmarking
=====================================

Overview
--------

A production integral library requires continuous correctness and performance
validation. This chapter defines a reproducible testing stack and benchmarking
protocol.

Correctness Layers
------------------

1. **Unit layer**: special functions, recurrence identities, transform matrices.
2. **Kernel layer**: shell-block outputs for one-/two-electron classes.
3. **Workflow layer**: SCF-level energies/gradients/Hessians.
4. **Cross-method layer**: OS vs Rys vs MD vs fused/rotated variants.

Reference Data Construction
---------------------------

Reference sources:

- high-precision numerical evaluation,
- trusted external engines,
- analytic closed forms for low-order cases.

Every fixture should store:

- geometry and basis metadata,
- method/precision settings,
- expected values and tolerance bands,
- generation provenance.

Tolerance Framework
-------------------

Use class-specific tolerances:

- scalar integrals: absolute + relative,
- matrix/tensor blocks: norm-based,
- properties: energy/gradient/Hessian tolerances.

Maintain separate thresholds for deterministic and throughput modes.

Benchmark Dimensions
--------------------

Benchmarks should sweep:

- basis size and max angular momentum,
- contraction depth,
- screening thresholds,
- hardware backend,
- algorithm path.

Report:

- runtime and throughput,
- error versus reference,
- screening survivor rates,
- resource metrics (bandwidth, occupancy, vector utilization).

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   void run_validation_suite(const Config& cfg) {
     for (const TestCase& tc : cfg.test_cases) {
       Result r = evaluate_case(tc, cfg.method, cfg.mode);
       Metrics m = compare_to_reference(r, tc.reference);
       record_metrics(tc.id, m);
       assert_with_tolerance(tc.id, m, cfg.tolerances);
     }
   }

   void run_benchmark_suite(const BenchConfig& bc) {
     for (const BenchCase& b : bc.cases) {
       PerfStats p = time_case_repeated(b, bc.repeats);
       ErrStats  e = error_case(b, bc.reference_mode);
       log_benchmark_row(b, p, e);
     }
   }

Regression Detection
--------------------

Performance regressions should use statistical guards:

- warmup exclusion,
- repeated runs and confidence intervals,
- baseline windows per hardware profile.

Correctness regressions should fail immediately if tolerances exceeded.

Reproducibility Metadata
------------------------

Each result record should include:

- commit SHA,
- compiler/toolchain,
- CPU/GPU model and driver,
- runtime flags and thresholds,
- dataset version.

Cross References
----------------

- Numerical controls: :doc:`21_numerical_stability`
- Method dispatch and policy: :doc:`24_method_selection_guide`
- Parallel execution factors: :doc:`22_parallelization_and_acceleration`
