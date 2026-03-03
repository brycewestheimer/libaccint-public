.. _theory-integrals-derivatives:

Derivative Integrals
====================

Overview
--------

Analytic derivatives of molecular integrals provide gradients, Hessians, and
response properties without finite-difference noise.

Derivative Classes
------------------

- first derivatives with respect to nuclear coordinates,
- second derivatives (Hessian-level),
- higher-order derivatives in specialized response theories.

Primitive Derivative Identity
-----------------------------

For Cartesian primitive centered at :math:`A_x`, differentiating with respect to
center coordinate gives

.. math::

   \frac{\partial}{\partial A_x}
   \Big[(x-A_x)^a e^{-\alpha(x-A_x)^2}\Big]
   = -a(x-A_x)^{a-1}e^{-\alpha(x-A_x)^2}
   + 2\alpha(x-A_x)^{a+1}e^{-\alpha(x-A_x)^2}

So derivatives map to linear combinations of shifted-angular-momentum integrals.
This is the key reason derivative kernels can reuse scalar integral machinery.

First-Derivative Integral Form
------------------------------

For overlap-like scalar classes, the center derivative identity becomes:

.. math::
   :label: eq-deriv-first

   \frac{\partial}{\partial A_x}I(\mathbf{a},\mathbf{b})
   =
   2\alpha\,I(\mathbf{a}+\mathbf{1}_x,\mathbf{b})
   - a_x\,I(\mathbf{a}-\mathbf{1}_x,\mathbf{b})

and similarly for derivatives on center `B`:

.. math::

   \frac{\partial}{\partial B_x}I(\mathbf{a},\mathbf{b})
   =
   2\beta\,I(\mathbf{a},\mathbf{b}+\mathbf{1}_x)
   - b_x\,I(\mathbf{a},\mathbf{b}-\mathbf{1}_x)

Second derivatives follow by applying :eq:`eq-deriv-first` again. For a pure
second derivative on center `A`:

.. math::
   :label: eq-deriv-second

   \frac{\partial^2}{\partial A_x^2}I(\mathbf{a},\mathbf{b})
   =
   4\alpha^2 I(\mathbf{a}+2\mathbf{1}_x,\mathbf{b})
   -2\alpha(2a_x+1)I(\mathbf{a},\mathbf{b})
   +a_x(a_x-1)I(\mathbf{a}-2\mathbf{1}_x,\mathbf{b})

Mixed derivatives are expanded analogously, e.g.

.. math::

   \frac{\partial^2}{\partial A_x\partial B_x}I(\mathbf{a},\mathbf{b})
   =
   4\alpha\beta I(\mathbf{a}+\mathbf{1}_x,\mathbf{b}+\mathbf{1}_x)
   -2\alpha b_x I(\mathbf{a}+\mathbf{1}_x,\mathbf{b}-\mathbf{1}_x)
   -2\beta a_x I(\mathbf{a}-\mathbf{1}_x,\mathbf{b}+\mathbf{1}_x)
   +a_x b_x I(\mathbf{a}-\mathbf{1}_x,\mathbf{b}-\mathbf{1}_x)

Translational and Rotational Invariance Constraints
---------------------------------------------------

Useful checks:

.. math::

   \sum_A \frac{\partial I}{\partial \mathbf{R}_A} = 0

and rotational consistency constraints for complete operator combinations.

Algorithmic Strategies
----------------------

1. differentiate RR relations directly,
2. evaluate shifted scalar classes and linearly combine,
3. mixed strategy: differentiated seeds + scalar RR reuse.

Choice depends on code complexity and performance goals.

For ERI derivatives, apply the same stencil logic on each center index of
:math:`(\mathbf{a}\mathbf{b}|\mathbf{c}\mathbf{d})`; practical implementations
reuse scalar ERI kernels with shifted tuples and shared invariants.

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   void one_electron_gradient_block(const ShellPair& sp, GradBlock& G) {
     // Example: derivative w.r.t. center A components
     ScalarBlock I_minus_x, I_plus_x, I_minus_y, I_plus_y, I_minus_z, I_plus_z;

     eval_shifted_scalar(sp.shift_ax_minus(), I_minus_x);
     eval_shifted_scalar(sp.shift_ax_plus(),  I_plus_x);
     eval_shifted_scalar(sp.shift_ay_minus(), I_minus_y);
     eval_shifted_scalar(sp.shift_ay_plus(),  I_plus_y);
     eval_shifted_scalar(sp.shift_az_minus(), I_minus_z);
     eval_shifted_scalar(sp.shift_az_plus(),  I_plus_z);

     combine_derivative_stencil(sp, I_minus_x, I_plus_x, G.dx);
     combine_derivative_stencil(sp, I_minus_y, I_plus_y, G.dy);
     combine_derivative_stencil(sp, I_minus_z, I_plus_z, G.dz);
   }

Data Layout
-----------

Derivative outputs are tensor-valued. Recommended layout includes explicit
component stride and atom-major grouping to simplify accumulation in downstream
geometry derivatives.

Numerical Considerations
------------------------

- derivative stencils can amplify cancellation,
- finite-difference checks are sensitive to step choice,
- mixed precision can destabilize Hessian-level terms.

Use stricter tolerances for translational invariance than for individual element
comparisons.

Validation Protocol
-------------------

1. finite-difference checks for gradients and Hessians,
2. invariance constraints,
3. cross-method checks for derivative classes,
4. SCF-level gradient/Hessian agreement on benchmark systems.

Cross References
----------------

- RR backbone: :doc:`07_recurrence_relations`
- One-electron scalar classes: :doc:`06_one_electron_integrals`
- Validation framework: :doc:`23_validation_and_benchmarking`
