.. _api-math:

Math API
========

Mathematical utilities for integral computation.

Boys Function
-------------

.. doxygenclass:: libaccint::math::BoysFunction
   :members:
   :undoc-members:

The Boys function :math:`F_m(T)` is defined as:

.. math::

   F_m(T) = \int_0^1 t^{2m} e^{-Tt^2} dt

LibAccInt uses Chebyshev interpolation for :math:`T \leq 40` and asymptotic
expansion for :math:`T > 40`.

Rys Quadrature
--------------

.. doxygenclass:: libaccint::math::RysQuadrature
   :members:
   :undoc-members:

Rys quadrature evaluates molecular integrals using roots and weights of Rys
polynomials:

.. math::

   \int_0^1 t^{2m} e^{-Tt^2} P(t) dt = \sum_{i=1}^{n} w_i P(t_i)

Gaussian Product
----------------

.. doxygenclass:: libaccint::math::GaussianProduct
   :members:
   :undoc-members:

The Gaussian product theorem states:

.. math::

   \exp(-\alpha |\mathbf{r} - \mathbf{A}|^2) \exp(-\beta |\mathbf{r} - \mathbf{B}|^2)
   = K_{AB} \exp(-\zeta |\mathbf{r} - \mathbf{P}|^2)

where :math:`\zeta = \alpha + \beta`, :math:`\mathbf{P} = (\alpha\mathbf{A} + \beta\mathbf{B})/\zeta`,
and :math:`K_{AB} = \exp(-\alpha\beta|\mathbf{A}-\mathbf{B}|^2/\zeta)`.

Normalization
-------------

.. doxygennamespace:: libaccint::math::normalization
   :members:

Cartesian Indices
-----------------

.. doxygenclass:: libaccint::math::CartesianIndices
   :members:
   :undoc-members:

Utility functions for mapping between linear and (i, j, k) indices for
Cartesian Gaussian functions.
