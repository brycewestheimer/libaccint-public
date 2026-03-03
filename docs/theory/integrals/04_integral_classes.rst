.. _theory-integrals-classes:

Integral Classes and Operator Families
======================================

Overview
--------

Quantum chemistry integral engines implement multiple classes with different
operators, dimensions, and algorithmic kernels. A clear taxonomy helps align
math with software architecture.

One-Electron Classes
--------------------

Canonical matrix elements:

.. math::

   S_{\mu\nu}=\braket{\mu}{\nu},\qquad
   T_{\mu\nu}=\braket{\mu}{-\frac{1}{2}\nabla^2\nu},\qquad
   V_{\mu\nu}=\sum_A\braket{\mu}{\frac{-Z_A}{r_A}\nu}

Additional one-electron classes often include dipole, quadrupole, momentum,
and angular-momentum operators.

Two-Electron and Reduced-Center Classes
---------------------------------------

Four-center ERI:

.. math::

   (\mu\nu|\lambda\sigma)

Three-center (DF/RI):

.. math::

   (\mu\nu|P)

Two-center metric:

.. math::

   (P|Q)

These classes are coupled in density-fitting workflows
(:doc:`19_density_fitting`).

Derivative Classes
------------------

Common derivative families:

- first derivatives (nuclear gradients): :math:`\partial (\cdot)/\partial R_{A\xi}`,
- second derivatives (Hessians): :math:`\partial^2 (\cdot)/\partial R_{A\xi}\partial R_{B\eta}`.

Derivative classes increase tensor dimensionality and recurrence depth
(:doc:`18_derivative_integrals`).

Symmetry Classes
----------------

Useful symmetries:

- one-electron: :math:`S_{\mu\nu}=S_{\nu\mu}` (similarly for Hermitian operators),
- ERI permutation symmetry up to 8-fold in real AO basis.

Storage and compute implementations may exploit reduced domains, but must keep
API semantics explicit (packed vs full).

Method Compatibility Matrix
---------------------------

.. list-table::
   :header-rows: 1

   * - Class
     - Typical Methods
     - Key Special Function
   * - Overlap / Kinetic
     - OS recurrences, MD
     - none (Gaussian moments)
   * - Nuclear attraction
     - OS, MD, Rys variants
     - Boys
   * - 4c2e ERI
     - OS+HGP, Rys, MD, TRn
     - Boys or quadrature roots/weights
   * - 3c2e / 2c2e
     - Rys, MD, specialized OS forms
     - Boys or quadrature
   * - Derivatives
     - Differentiated RR/MD/Rys
     - same as parent class

API and Buffer Design Implications
----------------------------------

Integral class should be explicit in API dispatch to determine:

- shell tuple arity (pair/quartet/triple),
- output shape and symmetry mode,
- required preprocessing (atom loops, metric factors),
- kernel family and precision policy.

Cross References
----------------

- One-electron details: :doc:`06_one_electron_integrals`
- ERI algorithm families: :doc:`15_two_electron_eri_algorithms`
- DF/CD approximations: :doc:`19_density_fitting`, :doc:`20_cholesky_decomposition`
