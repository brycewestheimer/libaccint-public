.. _theory-integrals-gpt:

Gaussian Product Theorem
========================

Statement
---------

For two s-type Gaussian factors:

.. math::

   e^{-\alpha\lVert\mathbf{r}-\mathbf{A}\rVert^2}
   e^{-\beta\lVert\mathbf{r}-\mathbf{B}\rVert^2}
   = K_{AB}\,e^{-\zeta\lVert\mathbf{r}-\mathbf{P}\rVert^2}

with

.. math::

   \zeta=\alpha+\beta,\qquad
   \mathbf{P}=\frac{\alpha\mathbf{A}+\beta\mathbf{B}}{\zeta},\qquad
   K_{AB}=\exp\left(-\frac{\alpha\beta}{\zeta}\lVert\mathbf{A}-\mathbf{B}\rVert^2\right)

This identity is the algebraic core of nearly all Gaussian integral algorithms.

Derivation Sketch
-----------------

Expand both squared distances and collect :math:`\mathbf{r}` terms:

.. math::

   -\alpha(\mathbf{r}-\mathbf{A})^2 - \beta(\mathbf{r}-\mathbf{B})^2
   = -\zeta(\mathbf{r}-\mathbf{P})^2
   - \frac{\alpha\beta}{\zeta}(\mathbf{A}-\mathbf{B})^2

The second term is independent of :math:`\mathbf{r}` and becomes :math:`K_{AB}`.

Why It Matters
--------------

GPT enables:

- reduction of two-center dependence to one product center,
- axis separability in Cartesian components,
- recurrence construction around shared centers,
- efficient precomputation of pair invariants.

Hermite Gaussian Expansion Link
-------------------------------

For higher angular momentum, polynomial prefactors are expanded in Hermite form:

.. math::

   (x-A_x)^{a_x}(x-B_x)^{b_x}e^{-\alpha(x-A_x)^2-\beta(x-B_x)^2}
   = \sum_{t=0}^{a_x+b_x} E_t^{ab}(x-P_x)^t e^{-\zeta(x-P_x)^2}

Analogous expansions hold for :math:`y` and :math:`z`.
This is the entry point for McMurchie-Davidson coefficients
(:doc:`11_mcmurchie_davidson_method`).

Implementation Invariants
-------------------------

For each primitive pair `(p,q)` cache once:

- :math:`\zeta`, :math:`\mu_{AB}=\alpha\beta/\zeta`,
- :math:`\mathbf{P}`,
- :math:`K_{AB}`,
- pair distance :math:`R_{AB}^2`.

Avoid recomputing these inside inner recurrences.

Numerical Behavior
------------------

The prefactor :math:`K_{AB}` may underflow when centers are far apart or
exponents are large. This behavior is physically expected and useful for
screening; do not attempt to "rescue" tiny values without a clear error policy.

Common Mistakes
---------------

- Using squared distance and distance interchangeably in prefactors.
- Mixing center coordinates from different unit systems.
- Recomputing :math:`\mathbf{P}` with non-associative arithmetic in different
  locations, causing reproducibility drift.

Cross References
----------------

- Primitive and contraction loops: :doc:`02_primitives_and_contraction`
- Boys function dependence: :doc:`05_boys_functions`
- Hermite methods: :doc:`11_mcmurchie_davidson_method`
