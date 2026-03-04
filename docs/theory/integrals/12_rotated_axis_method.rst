.. _theory-integrals-rotated-axis:

Rotated Axis Method
===================

Overview
--------

Rotated-axis methods evaluate integrals in a locally aligned coordinate system
to simplify recurrence structure or improve numerical conditioning for specific
geometries.

Coordinate Transformation
-------------------------

Choose orthonormal basis vectors :math:`\hat{e}_1,\hat{e}_2,\hat{e}_3` and
rotation matrix :math:`R` with rows (or columns, by convention) equal to these
vectors. Transform centers and vectors by

.. math::

   \mathbf{x}' = R\mathbf{x}

with consistent convention throughout the pipeline.

Frame Construction
------------------

A common choice:

1. :math:`\hat{e}_1` along dominant center-separation direction,
2. :math:`\hat{e}_2` from a stable orthogonal reference,
3. :math:`\hat{e}_3 = \hat{e}_1 \times \hat{e}_2`.

Degenerate cases (near-zero norms or near-collinearity) require deterministic
fallback rules.

How Rotation Helps
------------------

In aligned coordinates, certain recurrence coefficients can become simpler or
better conditioned. This can reduce cancellation and improve SIMD/GPU behavior
for targeted shell classes.

Cost Model
----------

Net benefit requires

.. math::

   C_{\text{rotate}} + C_{\text{kernel,rot}} + C_{\text{back}}
   < C_{\text{kernel,base}}

Hence rotation is usually applied adaptively, not globally.

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   bool try_rotated_path(const WorkItem& w, Result& out) {
     Rotation R;
     if (!build_stable_frame(w.geometry, R)) return false;

     RotatedGeometry g = apply_rotation(w.geometry, R);
     TempResult rot_out;

     // Evaluate chosen kernel in rotated frame
     eval_kernel_rotated(w, g, rot_out);

     // Back-transform tensor components if required
     back_transform_result(rot_out, R, out);
     return true;
   }

Data and Convention Constraints
-------------------------------

- rotation convention must match basis transform convention,
- component ordering must remain canonical after back-transform,
- invariance tests must be part of CI.

Numerical Validation
--------------------

1. random rigid-rotation invariance tests,
2. rotated vs unrotated differential checks,
3. degeneracy stress tests for frame construction.

Cross References
----------------

- Basis transform conventions: :doc:`17_coordinate_and_basis_transforms`
- Hybrid rotated-MD pipeline: :doc:`13_rotated_axis_mcmurchie_davidson`
