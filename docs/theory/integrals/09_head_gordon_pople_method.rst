.. _theory-integrals-hgp:

Head-Gordon-Pople Method
========================

Overview
--------

Head-Gordon-Pople (HGP) is centered on efficient horizontal recurrence relation
(HRR) transfer of angular momentum between centers. In practice, HGP is usually
combined with a VRR-based seed generator (often OS).

Derivation of HRR Transfer
--------------------------

Using coordinate identity

.. math::

   (x-B_x) = (x-A_x) + (A_x-B_x)

and substituting into integral definitions yields

.. math::
   :label: eq-hgp-hrr

   I(\mathbf{a},\mathbf{b}+\mathbf{1}_i;\mathbf{c},\mathbf{d})
   = I(\mathbf{a}+\mathbf{1}_i,\mathbf{b};\mathbf{c},\mathbf{d})
   + (A_i-B_i)I(\mathbf{a},\mathbf{b};\mathbf{c},\mathbf{d})

Similarly for ket side:

.. math::

   I(\mathbf{a},\mathbf{b};\mathbf{c},\mathbf{d}+\mathbf{1}_i)
   = I(\mathbf{a},\mathbf{b};\mathbf{c}+\mathbf{1}_i,\mathbf{d})
   + (C_i-D_i)I(\mathbf{a},\mathbf{b};\mathbf{c},\mathbf{d})

Important property: HRR does not raise Boys order and therefore is often
numerically cleaner than additional VRR stages.

OS+HGP Combined Algorithm
-------------------------

Common design:

1. Use OS VRR to generate momentum on fewer centers.
2. Apply HGP HRR chains to move momentum to target tuple.
3. Contract and transform output.

This splits complexity into generation and transfer phases.

Transfer Scheduling
-------------------

Given target tuple `(a,b,c,d)`, a transfer plan chooses an order to move
components from temporary source tuple to target. Good transfer plans:

- minimize peak intermediate count,
- maximize locality of contiguous component blocks,
- support vectorized operations.

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   void hrr_transfer(const HRRPlan& plan,
                     const TempTensor& src,
                     TempTensor& dst,
                     const Geometry& g) {
     dst.clear();

     for (const HRRStep& step : plan.steps()) {
       // Example step: build node N from node L (left) and node R (reference)
       // N = L + delta * R
       double delta = (step.side == Side::Bra)
         ? g.A[step.axis] - g.B[step.axis]
         : g.C[step.axis] - g.D[step.axis];

       for (int idx = 0; idx < step.block_size; ++idx) {
         dst(step.dst, idx) = src(step.left, idx) + delta * src(step.ref, idx);
       }
     }
   }

Fusion with VRR/Contraction
---------------------------

For performance, many engines fuse short HRR chains with either:

- final VRR output production, or
- contraction accumulation.

Fusion reduces memory traffic but must be balanced against register pressure.

Numerical Notes
---------------

Potential issues:

- cancellation if `A_i-B_i` or `C_i-D_i` is tiny and source terms are opposite,
- loss of reproducibility under unordered parallel transfer/accumulation.

Countermeasures:

- fixed transfer order in deterministic mode,
- mixed precision accumulation when needed,
- path-consistency tests across alternative transfer plans.

Validation Checklist
--------------------

1. Algebraic equality across alternate HRR orderings.
2. Equality with direct non-HRR small-tuple evaluation.
3. Permutation symmetry checks post-transfer.

Cross References
----------------

- OS source generation: :doc:`08_obara_saika_method`
- Transfer fusion concepts: :doc:`14_trn_fused_hrn_method`
- Method comparison: :doc:`15_two_electron_eri_algorithms`
