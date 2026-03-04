.. _theory-integrals-rys:

Rys Quadrature Method
=====================

Overview
--------

Rys quadrature evaluates Coulombic Gaussian integrals by reducing the problem to
finite weighted sums over quadrature roots. It is a robust alternative to
Boys-driven RR pipelines and is widely used in high-performance ERI engines.

Transform Idea
--------------

After Gaussian product reduction and integral transforms, ERI expressions can be
rewritten in a one-dimensional integral form that is exactly represented by an
:math:`n_r`-point quadrature for required polynomial degree.

Operationally:

.. math::
   :label: eq-rys-sum

   (\mu\nu|\lambda\sigma)
   = \sum_{k=1}^{n_r} w_k\,X_k\,Y_k\,Z_k

where :math:`X_k,Y_k,Z_k` are axis-separated recurrence values evaluated at root
:math:`u_k`.

Quadrature Order
----------------

For shell tuple `(a,b,c,d)`, root count is chosen from total angular momentum
requirements of the transformed polynomial degree:

.. math::

   n_r = \left\lceil\frac{L_a+L_b+L_c+L_d+1}{2}\right\rceil

where :math:`L_x` is shell angular momentum on center `x`. In practice this is
a small integer for most chemistry workloads.

Root/Weight Generation
----------------------

Common implementation options:

1. pretabulated approximants over parameter ranges,
2. polynomial/rational evaluators,
3. Newton refinement from robust initial guesses.

Constraints to enforce:

- roots are in valid interval and strictly ordered,
- weights are positive,
- generated moments match expected low-order moments.

Axis Recurrence Evaluation
--------------------------

For each root `k`, evaluate 1D recurrence chains along x/y/z independently.
This creates natural vectorization and parallelization opportunities:

- SIMD across roots,
- thread blocks across primitive quartets,
- fused accumulation of root contributions.

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   void eri_rys_shell_quartet(const ShellQuartet& q, Block& out) {
     zero(out);

     for (PrimAB ab : q.ab_primitives()) {
       for (PrimCD cd : q.cd_primitives()) {
         if (!primitive_bound_pass(ab, cd)) continue;

         RysParams p = build_rys_params(ab, cd, q);
         int nr = required_root_count(q.angular_tuple());

         SmallVec<double> roots(nr), weights(nr);
         rys_roots_weights(nr, p.T, roots.data(), weights.data());

         TempAxis ax, ay, az;
         ax.init(q.x_extent()); ay.init(q.y_extent()); az.init(q.z_extent());

         for (int k = 0; k < nr; ++k) {
           eval_axis_recurrence_x(q, p, roots[k], ax);
           eval_axis_recurrence_y(q, p, roots[k], ay);
           eval_axis_recurrence_z(q, p, roots[k], az);

           accumulate_root_contribution(q, weights[k], ax, ay, az, out);
         }
       }
     }

     if (q.output_is_spherical()) apply_cart_to_sph_transform(q, out);
   }

Complexity and Performance
--------------------------

Cost per primitive quartet is roughly proportional to:

.. math::

   n_r \times (C_x + C_y + C_z + C_{acc})

where axis costs depend on angular momentum and recurrence depth.

Rys typically offers:

- regular control flow,
- strong numerical behavior,
- efficient GPU mapping when shell tuples are batched by shape.

Numerical Stability
-------------------

Primary risks:

- inaccurate roots/weights,
- cancellation in weighted root sum,
- scaling issues in axis recurrences.

Recommended safeguards:

- moment-validation tests for root/weight generator,
- pairwise or compensated accumulation for large blocks,
- regime-specific scaling and normalization in recurrence kernels.

Validation Checklist
--------------------

1. root/weight moment consistency tests,
2. differential comparison with OS/MD on random quartets,
3. stress cases at high angular momentum and large separations,
4. backend consistency tests under deterministic mode.

Cross References
----------------

- OS/HGP alternatives: :doc:`08_obara_saika_method`,
  :doc:`09_head_gordon_pople_method`
- Family-level selection: :doc:`15_two_electron_eri_algorithms`
