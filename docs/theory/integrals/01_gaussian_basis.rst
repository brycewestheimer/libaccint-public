.. _theory-integrals-gaussian-basis:

Gaussian Basis Functions
========================

Introduction
------------

Most quantum chemistry integral engines use Gaussian-type orbitals (GTOs)
because products of Gaussians remain Gaussian, enabling analytic and recursive
integral algorithms.

Cartesian Primitive GTOs
------------------------

A Cartesian primitive at center :math:`\mathbf{A}` with exponent :math:`\alpha`:

.. math::
   :label: eq-gto-cart

   \phi_{\mathbf{a}}(\mathbf{r}) = (x-A_x)^{a_x}(y-A_y)^{a_y}(z-A_z)^{a_z}
   e^{-\alpha\lVert\mathbf{r}-\mathbf{A}\rVert^2}

with :math:`\mathbf{a}=(a_x,a_y,a_z)` and total angular momentum
:math:`L=a_x+a_y+a_z`.

Number of Cartesian functions for shell :math:`L`:

.. math::

   n_{\mathrm{cart}}(L) = \frac{(L+1)(L+2)}{2}

Examples: :math:`s=1`, :math:`p=3`, :math:`d=6`, :math:`f=10`, :math:`g=15`.

Spherical Harmonic Basis
------------------------

Spherical (real solid harmonic) shells use:

.. math::

   n_{\mathrm{sph}}(L) = 2L+1

Examples: :math:`d=5`, :math:`f=7`, :math:`g=9`.

Practical implication:

- Cartesian kernels are algebraically convenient for recurrences.
- Spherical functions reduce AO dimension and often reduce downstream cost.

Most engines compute in Cartesian form and transform to spherical afterward
(chapter :doc:`17_coordinate_and_basis_transforms`).

Normalization
-------------

Normalized primitive:

.. math::

   \tilde{\phi}_{\mathbf{a}}(\mathbf{r}) = N(\alpha,\mathbf{a})\,\phi_{\mathbf{a}}(\mathbf{r})

Cartesian normalization factor:

.. math::

   N(\alpha,\mathbf{a}) =
   \left(\frac{2\alpha}{\pi}\right)^{3/4}
   \sqrt{\frac{(4\alpha)^{L}}{(2a_x-1)!!(2a_y-1)!!(2a_z-1)!!}}

with :math:`(-1)!!=1` by convention.

Normalization must be handled consistently at one location in the code path
(primitive preload, contraction preload, or post-kernel scaling) to avoid
double-normalization bugs.

Function Ordering Conventions
-----------------------------

Shell component order affects ABI, tensor indexing, and reproducibility.
Typical Cartesian order for :math:`d` shell:

.. code-block:: text

   xx, xy, xz, yy, yz, zz

The implementation must define one canonical order and use it in:

- basis parsing,
- kernel output layout,
- transforms,
- public API buffer indexing.

Recommended invariant tests:

- round-trip Cartesian->spherical->Cartesian,
- permutation checks against reference ordering tables,
- AO block stride and leading-dimension assertions.

Exponent Regimes and Conditioning
---------------------------------

Large :math:`\alpha` (tight primitives) and very small :math:`\alpha`
(diffuse primitives) both stress floating-point arithmetic.

- Tight primitives can create rapid decay and severe exponent-scale mismatch.
- Diffuse primitives produce near-linear dependencies in overlap matrices.

In practice, robust engines monitor overlap matrix condition numbers and may
trigger basis pruning or regularization strategies in downstream solvers.

Implementation Checklist
------------------------

1. Parse basis metadata into immutable shell records.
2. Precompute and cache normalization factors.
3. Store exponents/coefficients in structure-of-arrays form for vectorization.
4. Encode shell offsets and component counts for dense and packed output paths.
5. Enforce ordering conventions at API boundaries.

Cross References
----------------

- Primitive/contracted assembly: :doc:`02_primitives_and_contraction`
- Transform conventions: :doc:`17_coordinate_and_basis_transforms`
- Numerical issues: :doc:`21_numerical_stability`
