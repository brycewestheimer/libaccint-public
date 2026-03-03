.. _theory-integrals-notation-appendix:

Notation Appendix
=================

Purpose
-------

This appendix defines the canonical symbols and index conventions used across
all integral theory chapters.

Core Geometric Symbols
----------------------

.. list-table::
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\mathbf{A},\mathbf{B},\mathbf{C},\mathbf{D}`
     - Basis centers
   * - :math:`\mathbf{P}`
     - Product center for pair `(A,B)`
   * - :math:`\mathbf{Q}`
     - Product center for pair `(C,D)`
   * - :math:`\mathbf{W}`
     - Composite center used in some RR derivations
   * - :math:`R_{AB}`
     - Distance :math:`\lVert \mathbf{A}-\mathbf{B}\rVert`

Exponent and Pair Scalars
-------------------------

.. list-table::
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\alpha,\beta,\gamma,\delta`
     - Primitive exponents
   * - :math:`\zeta=\alpha+\beta`
     - Bra-side exponent sum
   * - :math:`\eta=\gamma+\delta`
     - Ket-side exponent sum
   * - :math:`\mu_{AB}=\frac{\alpha\beta}{\zeta}`
     - Pair-width factor for primitive pair `(A,B)`
   * - :math:`\rho=\frac{\zeta\eta}{\zeta+\eta}`
     - Composite pair factor for ERI transforms between `(AB)` and `(CD)`
   * - :math:`K_{AB}`
     - Gaussian product prefactor for pair `(A,B)`

Angular Momentum and Shell Indices
----------------------------------

.. list-table::
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\mathbf{a}=(a_x,a_y,a_z)`
     - Cartesian angular momentum tuple
   * - :math:`L_a=a_x+a_y+a_z`
     - Shell order for tuple :math:`\mathbf{a}`
   * - :math:`\mathbf{1}_i`
     - Unit increment in component `i` (`x`,`y`,`z`)
   * - :math:`\mu,\nu,\lambda,\sigma`
     - AO function indices

Integral and Auxiliary Symbols
------------------------------

.. list-table::
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`(\mu\nu|\lambda\sigma)`
     - Four-center ERI
   * - :math:`F_m(T)`
     - Boys function order `m`
   * - :math:`u_k,w_k`
     - Rys root and weight
   * - :math:`E_t^{ab}`
     - 1D Hermite coefficient (MD)
   * - :math:`R_{tuv}^{(n)}`
     - Hermite Coulomb intermediate (MD family)

Transform Symbols
-----------------

.. list-table::
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`R`
     - Rotation matrix
   * - :math:`T^{(L)}`
     - Cartesian-to-spherical transform for shell order `L`

Conventions
-----------

- Vectors are bold symbols.
- Distances without square are Euclidean norms.
- Component indices use `i` for axis-specific formulas.
- All equations assume atomic units unless explicitly stated otherwise.

Cross References
----------------

- High-level overview: :doc:`00_overview`
- Symbol lookup table: :doc:`26_symbol_index`
