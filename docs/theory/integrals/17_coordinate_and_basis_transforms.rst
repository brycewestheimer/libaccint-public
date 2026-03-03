.. _theory-integrals-transforms:

Coordinate and Basis Transforms
===============================

Overview
--------

Integral engines frequently compute in one representation and expose results in
another. This chapter formalizes coordinate and basis transforms required for
correctness and interoperability.

Transform Categories
--------------------

1. **Coordinate transforms**: rotate/translate geometric data.
2. **Basis transforms**: map Cartesian shell components to spherical harmonics.
3. **Tensor assembly transforms**: apply Kronecker-structured transforms to
   multi-center blocks.

Coordinate Transform Theory
---------------------------

For orthogonal rotation matrix :math:`R`:

.. math::

   \mathbf{x}' = R\mathbf{x}, \qquad R^TR = I

Scalar integrals are rotationally invariant. Component-valued objects transform
covariantly by order (vector, rank-2 tensor, etc.).

Translation by :math:`\mathbf{t}` shifts centers but leaves relative vectors and
therefore integral values unchanged when all centers shift consistently.

Cartesian-to-Spherical Basis Transform
--------------------------------------

For shell order :math:`L`, define transformation matrix
:math:`T^{(L)}\in\mathbb{R}^{(2L+1)\times n_{cart}(L)}`:

.. math::
   :label: eq-cart-sph

   \mathbf{c}^{(L)}_{sph} = T^{(L)} \mathbf{c}^{(L)}_{cart}

For multi-index blocks, apply transforms via tensor products.
For an ERI block `(ab|cd)`:

.. math::

   V_{sph} =
   (T^{(L_a)}\otimes T^{(L_b)})
   V_{cart}
   (T^{(L_c)}\otimes T^{(L_d)})^T

Convention Management
---------------------

Critical conventions:

- spherical harmonic phase/sign,
- component ordering within each `L`,
- normalization convention (unitary vs non-unitary maps).

A library must publish these conventions and test cross-tool compatibility.

Transform Placement Strategies
------------------------------

- **Late transform**: compute kernels in Cartesian, transform at output.
- **Early transform**: transform intermediates early to reduce dimensionality.
- **Hybrid**: late for some classes, early for others.

Late transform is simplest; early/hybrid may win when transform cost is
amortized by downstream reductions.

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   void apply_eri_cart_to_sph(const ShellTuple& q,
                              const Matrix& Vcart,
                              Matrix& Vsph) {
     const Matrix& Ta = transform_L(q.La);
     const Matrix& Tb = transform_L(q.Lb);
     const Matrix& Tc = transform_L(q.Lc);
     const Matrix& Td = transform_L(q.Ld);

     Matrix L = kron(Ta, Tb);
     Matrix R = kron(Tc, Td);

     // Vsph = L * Vcart * R^T
     gemm(L, Vcart, tmp);
     gemm(tmp, transpose(R), Vsph);
   }

Optimization Notes
------------------

- Avoid explicit Kronecker materialization for small shells; use blocked kernels.
- Cache transform matrices by `L`.
- Fuse transform with contraction/store when possible.

Validation
----------

1. round-trip tests (`cart -> sph -> cart`),
2. convention checks against known reference coefficients,
3. rotational invariance checks for scalar observables,
4. component-covariance checks for vector/tensor operators.

Cross References
----------------

- Basis definitions: :doc:`01_gaussian_basis`
- Rotated-axis methods: :doc:`12_rotated_axis_method`
- Stability considerations: :doc:`21_numerical_stability`
