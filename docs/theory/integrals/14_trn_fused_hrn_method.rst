.. _theory-integrals-trn-hrn:

TRn (Fused HRn) Method
======================

Overview
--------

TRn/fused-HRn reorganizes transfer recurrences to minimize temporary tensors and
memory traffic. Instead of materializing each transfer level, multiple steps are
fused into compact kernels.

From Staged to Fused Transfer
-----------------------------

Suppose staged transfer computes

.. math::

   X^{(0)} \to X^{(1)} \to \cdots \to X^{(n)}

where each stage follows HRR-like identity
:math:`X^{(k+1)} = A^{(k)}X^{(k)} + B^{(k)}Y^{(k)}`.

Fused formulation algebraically substitutes intermediate stages to compute final
nodes directly from a smaller set of source nodes. This trades more arithmetic
for fewer loads/stores.

Why It Matters
--------------

Many ERI kernels are memory-bandwidth limited. Fusion can:

- reduce bytes moved per target integral,
- improve arithmetic intensity,
- increase locality for SIMD/GPU registers/shared memory.

Design Constraints
------------------

Fusion must balance:

- register pressure (too much fusion can reduce GPU occupancy),
- code size explosion (for many shell tuples),
- reproducibility (different summation orders).

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   void fused_transfer_kernel(const TransferPlan& plan,
                              const TempTensor& src,
                              TempTensor& dst,
                              const CoeffPack& coeff) {
     // plan groups multiple HRR/TRR steps into fused tiles
     for (const FusedTile& tile : plan.tiles()) {
       // Load minimal source set into fast storage
       FastBuf fb;
       fb.load(src, tile.source_nodes);

       // Compute requested destination nodes without writing intermediates
       for (const FusedExpr& expr : tile.expressions) {
         Value v = 0.0;
         for (const Term& t : expr.terms) {
           v += t.alpha * fb.get(t.node);
         }
         dst.store(expr.dst_node, v);
       }
     }
   }

Auto-Tuning and Code Generation
-------------------------------

TRn fusion is often best produced by code generation or offline schedule
optimization. Tunables include:

- fusion depth,
- tile size,
- unroll factors,
- vector lane mapping.

Auto-tuning should optimize measured runtime under representative shell-mix
workloads, not microkernels alone.

Numerical Effects
-----------------

Fusion changes evaluation order and can slightly alter floating-point rounding.
Policy recommendations:

- deterministic validation mode with fixed fusion schedule,
- tolerance-aware comparison in performance mode,
- regression tests guarding accuracy drift.

Validation and Profiling
------------------------

1. algebraic equality vs unfused path on deterministic settings,
2. memory-traffic reduction confirmation via profiler counters,
3. occupancy/register tradeoff studies on GPU,
4. end-to-end SCF energy/gradient consistency checks.

Cross References
----------------

- HGP transfer baseline: :doc:`09_head_gordon_pople_method`
- Parallel execution concerns: :doc:`22_parallelization_and_acceleration`
- Stability controls: :doc:`21_numerical_stability`
