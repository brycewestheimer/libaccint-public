.. _theory-integrals-overview:

Overview, Scope, and Notation
=============================

Purpose of This Section
-----------------------

This section is a standalone technical text on molecular integral evaluation in
Gaussian basis sets. It is written for readers who need both:

- mathematical understanding (derivations, recurrences, approximation error), and
- implementation understanding (data layout, scheduling, CPU/GPU execution).

The goal is practical completeness: enough theory and algorithm detail to design,
implement, validate, and tune integral kernels used in Hartree-Fock, DFT, and
post-HF workflows.

What Is Covered
---------------

The chapter sequence follows the same dependency structure as a production
integral engine:

1. basis functions, primitives, contraction, and Gaussian identities,
2. core special functions (Boys),
3. one-electron and two-electron integral classes,
4. recurrence and quadrature algorithm families,
5. screening, transforms, derivatives, and approximations (DF/CD),
6. numerical stability, parallelization, and validation.

Supported operator classes include one-electron operators, four-center ERIs,
three-center and two-center Coulomb metric integrals, and derivative classes.

Prerequisites
-------------

This text assumes familiarity with:

- linear algebra (matrix/tensor indexing, symmetric factorizations),
- multivariable calculus and Gaussian integrals,
- basic electronic structure equations (AO basis, Fock build),
- floating-point arithmetic and numerical error behavior.

Notation and Conventions
------------------------

Centers and vectors
~~~~~~~~~~~~~~~~~~~

- Nuclear or basis centers are uppercase bold vectors: :math:`\mathbf{A}, \mathbf{B}, \mathbf{C}, \mathbf{D}`.
- Product centers are :math:`\mathbf{P}` for pair :math:`(A,B)` and :math:`\mathbf{Q}` for pair :math:`(C,D)`.
- Cartesian components use subscripts :math:`x,y,z`.

Primitive Gaussian
~~~~~~~~~~~~~~~~~~

A Cartesian primitive centered at :math:`\mathbf{A}`:

.. math::
   :label: eq-overview-primitive

   \phi_{\mathbf{a}}(\mathbf{r};\alpha,\mathbf{A}) = (x-A_x)^{a_x}(y-A_y)^{a_y}(z-A_z)^{a_z}
   \exp\left(-\alpha\lVert \mathbf{r}-\mathbf{A}\rVert^2\right)

with angular momentum tuple :math:`\mathbf{a}=(a_x,a_y,a_z)` and shell order
:math:`L_a=a_x+a_y+a_z`.

Contraction
~~~~~~~~~~~

A contracted AO:

.. math::
   :label: eq-overview-contracted

   \chi_\mu(\mathbf{r}) = \sum_{p=1}^{K_\mu} c_{\mu p}\,N_{\mu p}\,\phi_{\mathbf{a}_\mu}(\mathbf{r};\alpha_{\mu p},\mathbf{A}_\mu)

where :math:`N_{\mu p}` is primitive normalization and :math:`c_{\mu p}` is the
contraction coefficient.

Common scalars
~~~~~~~~~~~~~~

For a primitive pair :math:`(\alpha,\beta)` at centers :math:`A,B`:

.. math::

   \zeta = \alpha + \beta,\qquad
   \mu_{AB} = \frac{\alpha\beta}{\zeta},\qquad
   \mathbf{P} = \frac{\alpha\mathbf{A}+\beta\mathbf{B}}{\zeta}

For two primitive pairs `(AB)` and `(CD)`:

.. math::

   \eta = \gamma + \delta,\qquad
   \rho = \frac{\zeta\eta}{\zeta+\eta}

Bra-ket and ERI notation
~~~~~~~~~~~~~~~~~~~~~~~~

- One-electron integral: :math:`\braket{\mu}{\hat{O}\nu}`.
- Electron repulsion integral (ERI):

.. math::

   (\mu\nu|\lambda\sigma) = \iint \chi_\mu(\mathbf{r}_1)\chi_\nu(\mathbf{r}_1)
   \frac{1}{r_{12}}
   \chi_\lambda(\mathbf{r}_2)\chi_\sigma(\mathbf{r}_2)
   d\mathbf{r}_1 d\mathbf{r}_2

Execution Pipeline (Implementation View)
----------------------------------------

A high-level integral engine usually executes the following pipeline:

1. Enumerate shell pairs/quartets and apply cheap bounds.
2. Build primitive-pair invariants (:math:`\zeta`, :math:`\eta`, :math:`\mu_{AB}`, :math:`\mu_{CD}`, :math:`\rho`, centers, prefactors).
3. Evaluate base integrals (Boys or quadrature roots/weights).
4. Build higher angular momentum via recurrence or Hermite expansion.
5. Contract primitives into AO-shell blocks.
6. Apply Cartesian-to-spherical transforms if required.
7. Accumulate into consumer buffers (matrix/tensor builders).

Every later chapter maps to one or more of these stages.

Complexity Baseline
-------------------

Ignoring screening and sparsity:

- one-electron matrices scale as :math:`O(N^2)`,
- four-center ERIs scale as :math:`O(N^4)` in AO count,
- DF and CD reduce effective scaling/storage by low-rank factorization.

Real systems rely on screening and locality to avoid asymptotic worst-case cost.

How to Read the Remaining Chapters
----------------------------------

- Read chapters :doc:`01_gaussian_basis`, :doc:`02_primitives_and_contraction`,
  and :doc:`03_gaussian_product_theorem` first.
- Then read :doc:`05_boys_functions` and :doc:`07_recurrence_relations`.
- Algorithm chapters (:doc:`08_obara_saika_method`, :doc:`10_rys_quadrature_method`,
  :doc:`11_mcmurchie_davidson_method`) are easiest once those foundations are clear.

Validation Philosophy
---------------------

All methods should be validated against three axes:

1. mathematical correctness (agreement with reference engines),
2. numerical robustness (pathological exponent/distance regimes),
3. performance behavior (cost and scaling consistent with model).

The dedicated validation framework is documented in
:doc:`23_validation_and_benchmarking`.
