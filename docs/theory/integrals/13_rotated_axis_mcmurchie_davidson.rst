.. _theory-integrals-rotated-md:

Rotated Axis + McMurchie-Davidson Method
========================================

Overview
--------

This hybrid combines coordinate rotation with MD Hermite expansion. The intent
is to preserve MD structure while exploiting geometry-aligned frames for better
conditioning or reduced effective complexity in selected cases.

Pipeline Derivation
-------------------

Let :math:`R` be rotation matrix. Then

.. math::

   \phi(\mathbf{r};\mathbf{A}) \to \phi'(\mathbf{r}';\mathbf{A}')
   \quad\text{with}\quad
   \mathbf{r}'=R\mathbf{r},\; \mathbf{A}'=R\mathbf{A}

Build MD coefficients :math:`E'` and intermediates :math:`R'_{tuv}` in rotated
frame, compute integral block :math:`I'`, then transform output components back
when required by API convention.

Algorithm
---------

1. Build rotation from shell-pair/quartet geometry.
2. Rotate centers and relevant displacement vectors.
3. Generate MD tables in rotated frame.
4. Compute primitive/contracted block in rotated frame.
5. Apply inverse component transform to obtain canonical output.

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   void eri_rotated_md(const ShellQuartet& q, Block& out) {
     Rotation R;
     if (!build_stable_frame(q.geometry(), R)) {
       eri_md_shell_quartet(q, out); // fallback
       return;
     }

     RotatedShellQuartet qr = rotate_shell_quartet(q, R);
     Block rot_block;
     eri_md_shell_quartet(qr, rot_block);

     inverse_component_transform(q, R, rot_block, out);
   }

Heuristics for Enabling Hybrid Path
-----------------------------------

Useful enable criteria:

- large anisotropy in center separations,
- shell tuples with high momentum where base MD shows instability,
- cached frame reuse opportunities across many primitive pairs.

Disable when transform overhead dominates.

Numerical and Performance Notes
-------------------------------

- floating-point non-associativity means rotated and non-rotated outputs may
  differ at last-bit level,
- deterministic mode should fix frame construction and fallback logic,
- benchmark both accuracy and throughput before enabling by default.

Validation Checklist
--------------------

1. covariance/invariance checks under global rotations,
2. differential comparison to plain MD and OS/Rys references,
3. fallback-trigger tests for degenerate geometry cases.

Cross References
----------------

- MD method details: :doc:`11_mcmurchie_davidson_method`
- Rotated-axis fundamentals: :doc:`12_rotated_axis_method`
- Method selection policy: :doc:`24_method_selection_guide`
