.. _theory-integrals-parallel:

Parallelization and Hardware Acceleration
=========================================

Overview
--------

Integral workloads are highly parallel but irregular. Efficient execution
requires matching algorithm structure to CPU/GPU hardware while preserving
numerical constraints.

Work Decomposition Model
------------------------

Potential parallel axes:

- shell-pair / shell-quartet tasks,
- primitive combinations,
- recurrence tiles,
- quadrature roots.

Good decomposition balances load after screening and minimizes synchronization.

CPU Strategy
------------

- task parallelism over shell entities,
- SIMD over roots/primitives/components,
- cache-aware blocking for intermediates and outputs,
- NUMA-aware placement for large systems.

Avoid false sharing in output matrices by block partitioning or thread-local
buffers with deterministic reduction.

GPU Strategy
------------

- map shell tiles to thread blocks,
- use shared memory for reused invariants/intermediates,
- keep memory accesses coalesced,
- tune register pressure vs occupancy.

Group work by shell tuple shape to reduce warp divergence.

Memory-Traffic Model
--------------------

For kernel with bytes :math:`B` and flops :math:`F`, runtime is bounded by

.. math::

   T \gtrsim \max\left(\frac{F}{P_{peak}},\frac{B}{BW_{peak}}\right)

Optimization target depends on whether kernel is compute- or bandwidth-bound.

Pipeline Overlap
----------------

In heterogeneous execution, overlap:

- host preprocessing,
- device transfers,
- kernel execution,
- consumer accumulation.

Use streams/events with explicit dependency edges.

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   void launch_eri_pipeline(const WorkQueue& Q) {
     while (auto batch = Q.next_batch()) {
       preprocess_batch_on_cpu(batch);

       if (batch.on_gpu()) {
         cudaMemcpyAsync(batch.dev_in, batch.host_in, batch.bytes_in, H2D, batch.stream);
         eri_kernel<<<batch.grid, batch.block, batch.smem, batch.stream>>>(batch.dev_in, batch.dev_out);
         cudaMemcpyAsync(batch.host_out, batch.dev_out, batch.bytes_out, D2H, batch.stream);
       } else {
         eri_cpu_kernel(batch.host_in, batch.host_out);
       }

       enqueue_consumer(batch.host_out);
     }
   }

Auto-Tuning and Dispatch
------------------------

Backend-specific tuning knobs include:

- tile sizes,
- unroll factors,
- vector width,
- fusion depth,
- task granularity.

Dispatch should use calibrated models, not static guesses.

Correctness Under Parallelism
-----------------------------

Define two execution modes:

- **deterministic**: fixed schedule/reduction tree for reproducibility,
- **throughput**: relaxed ordering for performance with tolerance-based checks.

Validation Protocol
-------------------

1. backend differential tests on shared work items,
2. scaling curves vs thread count/SM count,
3. profiling counters for occupancy/vector efficiency/bandwidth,
4. reproducibility tests in deterministic mode.

Cross References
----------------

- Fused transfer kernels: :doc:`14_trn_fused_hrn_method`
- Stability policy: :doc:`21_numerical_stability`
- Benchmark process: :doc:`23_validation_and_benchmarking`
