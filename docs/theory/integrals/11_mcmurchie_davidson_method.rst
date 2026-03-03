.. _theory-integrals-md:

McMurchie-Davidson Method
=========================

Overview
--------

The McMurchie-Davidson (MD) method evaluates Gaussian integrals via Hermite
Gaussian expansion. Polynomial factors are projected into Hermite coefficients,
then contracted with Hermite Coulomb/overlap moments.

Hermite Expansion Basis
-----------------------

For one axis:

.. math::
   :label: eq-md-hermite-expand

   (x-A_x)^a(x-B_x)^b e^{-\alpha(x-A_x)^2-\beta(x-B_x)^2}
   = \sum_{t=0}^{a+b} E_t^{ab}(x-P_x)^t e^{-\zeta(x-P_x)^2}

with :math:`\zeta=\alpha+\beta`, :math:`P_x=(\alpha A_x+\beta B_x)/\zeta`.

Coefficient Recurrences
-----------------------

A standard recursion set for :math:`E_t^{ab}` is

.. math::

   E_t^{a+1,b}
   = \frac{1}{2\zeta}E_{t-1}^{ab}
   + (P_x-A_x)E_t^{ab}
   + (t+1)E_{t+1}^{ab}

.. math::

   E_t^{a,b+1}
   = \frac{1}{2\zeta}E_{t-1}^{ab}
   + (P_x-B_x)E_t^{ab}
   + (t+1)E_{t+1}^{ab}

with base :math:`E_0^{00}=1`, others zero outside range.

3D separability gives coefficient products over x/y/z.

Hermite Coulomb Terms
---------------------

For Coulombic classes, MD uses Hermite Coulomb intermediates
:math:`R_{tuv}^{(n)}` with their own recurrences (in Cartesian components and
auxiliary order :math:`n`). These are seeded from Boys-like scalar terms.

Then ERI-like targets are assembled as coefficient contractions:

.. math::

   (ab|cd) = \sum_{tuv}\sum_{\tau\nu\phi}
   E_{tuv}^{ab}\,E_{\tau\nu\phi}^{cd}\,R_{t+\tau,\,u+\nu,\,v+\phi}^{(0)}

where multi-index notation is used for compactness.

Algorithm Structure
-------------------

1. Build pair invariants for AB and CD.
2. Generate Hermite coefficient tables for both pairs.
3. Generate required Hermite Coulomb/overlap intermediates.
4. Contract coefficient tables with intermediates.
5. Contract primitives and transform/store outputs.

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   void eri_md_shell_quartet(const ShellQuartet& q, Block& out) {
     zero(out);

     for (PrimAB ab : q.ab_primitives()) {
       for (PrimCD cd : q.cd_primitives()) {
         if (!primitive_bound_pass(ab, cd)) continue;

         MDInvariants inv = build_md_invariants(ab, cd);

         // 1) Hermite coefficient tables for AB and CD
         ETCoeff Eab = build_E_coeff(q.aL(), q.bL(), inv.ab);
         ETCoeff Ecd = build_E_coeff(q.cL(), q.dL(), inv.cd);

         // 2) Hermite Coulomb intermediates R_{tuv}^{(n)}
         RCoeff R = build_R_coeff(q.max_rank(), inv);

         // 3) Contract to Cartesian quartet block
         TempTensor prim_block;
         md_contract(Eab, Ecd, R, prim_block);

         contract_and_accumulate(prim_block, ab, cd, out);
       }
     }

     if (q.output_is_spherical()) apply_cart_to_sph_transform(q, out);
   }

Data Layout and Optimization
----------------------------

MD performance depends heavily on layout of E/R tables.

Recommended patterns:

- contiguous `t`/`u`/`v` slices for inner loops,
- preallocated scratch blocks per thread/task,
- kernel specialization for common `(L_a,L_b,L_c,L_d)` tuples,
- fusion of table generation with contraction for small tuples.

Numerical Behavior
------------------

Potential issues include recurrence drift in high orders and cancellation in
large tensor contractions. Mitigations:

- stable recurrence ordering,
- higher precision accumulation,
- normalization/scaling of intermediates where needed.

Validation Checklist
--------------------

1. coefficient table checks against low-order analytic values,
2. ERI block differential tests vs OS/Rys,
3. high-`L` and extreme-exponent stress suites.

Cross References
----------------

- Gaussian product foundation: :doc:`03_gaussian_product_theorem`
- Rys and OS alternatives: :doc:`10_rys_quadrature_method`,
  :doc:`08_obara_saika_method`
- Rotated hybrid MD: :doc:`13_rotated_axis_mcmurchie_davidson`
