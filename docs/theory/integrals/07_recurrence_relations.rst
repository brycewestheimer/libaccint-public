.. _theory-integrals-rr:

Recurrence Relations (RRs)
==========================

Overview
--------

Recurrence relations are the core mechanism for evaluating high-angular-momentum
Gaussian integrals from low-order seed integrals. They replace expensive direct
symbolic integration with structured algebraic lifting.

General Setup
-------------

Let an integral family be indexed by angular tuples and optional auxiliary
index :math:`m` (for Boys order or analogous parameter):

.. math::

   I\big(\mathbf{a},\mathbf{b},\mathbf{c},\mathbf{d};m\big)

RRs provide identities that increment/decrement one component of these tuples.

Vertical Recurrence Relations (VRR)
-----------------------------------

VRR raises momentum on a chosen center while keeping center assignment fixed.
A concrete one-electron nuclear-attraction VRR along component `i` can be
written as:

.. math::
   :label: eq-rr-vrr-one-electron

   V_{a_i+1,b}^{(m)} =
   (P_i-A_i)V_{a,b}^{(m)}
   - (P_i-C_i)V_{a,b}^{(m+1)}
   + \frac{a_i}{2\zeta}\left(V_{a_i-1,b}^{(m)}-V_{a_i-1,b}^{(m+1)}\right)
   + \frac{b_i}{2\zeta}\left(V_{a,b_i-1}^{(m)}-V_{a,b_i-1}^{(m+1)}\right)

For two-electron ERIs, a representative OS VRR is:

.. math::
   :label: eq-rr-vrr-eri

   I_{a_i+1,b,c,d}^{(m)} =
   (P_i-A_i)I_{a,b,c,d}^{(m)}
   + (W_i-P_i)I_{a,b,c,d}^{(m+1)}
   + \frac{a_i}{2\zeta}\left(I_{a_i-1,b,c,d}^{(m)} - \frac{\rho}{\zeta}I_{a_i-1,b,c,d}^{(m+1)}\right)
   + \frac{c_i}{2(\zeta+\eta)}I_{a,b,c_i-1,d}^{(m+1)}

with :math:`\zeta,\eta,\rho` defined in :doc:`25_notation_appendix`.

Horizontal Recurrence Relations (HRR)
-------------------------------------

HRR transfers angular momentum between centers using coordinate identities, for
example on bra side:

.. math::
   :label: eq-rr-hrr-bra

   I(\mathbf{a},\mathbf{b}+\mathbf{1}_i;\mathbf{c},\mathbf{d})
   =
   I(\mathbf{a}+\mathbf{1}_i,\mathbf{b};\mathbf{c},\mathbf{d})
   + (A_i-B_i)
   I(\mathbf{a},\mathbf{b};\mathbf{c},\mathbf{d})

Similarly on ket side:

.. math::

   I(\mathbf{a},\mathbf{b};\mathbf{c},\mathbf{d}+\mathbf{1}_i)
   =
   I(\mathbf{a},\mathbf{b};\mathbf{c}+\mathbf{1}_i,\mathbf{d})
   + (C_i-D_i)
   I(\mathbf{a},\mathbf{b};\mathbf{c},\mathbf{d})

Transfer Recurrences (TRR/TRn)
------------------------------

Transfer recurrence families generalize HRR-like motion and can be algebraically
fused to reduce temporaries (chapter :doc:`14_trn_fused_hrn_method`).

Base Integrals
--------------

RR chains must terminate at analytically evaluated seeds:

- overlap-like seeds for non-Coulomb classes,
- Boys-dependent seeds for Coulombic classes,
- quadrature moments for Rys-like variants.

Seed quality determines global numerical stability.

Recurrence Graph Formulation
----------------------------

Represent RR equations as a DAG:

- node: integral index tuple,
- edge: dependency with scalar coefficient.

Execution is a topological traversal over this DAG. An optimized traversal
minimizes peak live intermediates and memory movement.

Scheduling Patterns
-------------------

1. **Static schedule** for fixed shell tuple classes (fast, codegen-friendly).
2. **Dynamic schedule** from generic DAG rules (flexible, lower code volume).
3. **Fused schedule** combining adjacent passes to reduce writes.

Resource Tradeoffs
------------------

- Unrolled static kernels reduce branch overhead but increase binary size.
- Generic kernels lower code size but increase indexing overhead.
- Fusion reduces bandwidth but may increase register pressure.

Numerical Stability Map
-----------------------

Stability depends on recurrence direction and coefficients. Practical policy:

1. choose direction by regime,
2. avoid subtractive cancellation hotspots where possible,
3. accumulate in higher precision,
4. enable fallback path for pathological tuples.

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   struct RRTask {
     ShellTuple shells;
     int m_max;
   };

   void eval_rr_block(const RRTask& t, Buffer& out) {
     // 1) Build seed layer (often s-type + Boys)
     SeedLayer seed = build_seed_layer(t.shells, t.m_max);

     // 2) Plan recurrence schedule for this shell tuple
     RRSchedule sched = plan_schedule(t.shells, t.m_max);

     // 3) Execute schedule
     TempPool temp;
     temp.init(sched.required_capacity());

     for (const RRStep& step : sched.steps()) {
       // step describes destination node and source dependencies
       Value v = 0.0;
       for (const RRDep& dep : step.deps) {
         v += dep.coeff * temp.get_or_seed(dep.node, seed);
       }
       temp.set(step.dst, v);
     }

     // 4) Gather requested target block and contract/store
     gather_targets(temp, t.shells, out);
   }

Validation
----------

1. Path equivalence: evaluate same target via alternate RR routes.
2. Low-order spot checks against direct analytic formulas.
3. Stress tests across high `L`, diffuse/tight exponent extremes.
4. Deterministic replay tests under multithread/GPU backends.

Cross References
----------------

- Boys seeds: :doc:`05_boys_functions`
- OS/HGP instantiations: :doc:`08_obara_saika_method`,
  :doc:`09_head_gordon_pople_method`
- Fused transfer forms: :doc:`14_trn_fused_hrn_method`
