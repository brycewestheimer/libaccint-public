.. _api-buffers:

Buffers API
===========

Buffer classes for storing computed integrals.

OneElectronBuffer
-----------------

.. doxygenclass:: libaccint::OneElectronBuffer
   :members:
   :undoc-members:

TwoElectronBuffer
-----------------

.. doxygenclass:: libaccint::TwoElectronBuffer
   :members:
   :undoc-members:

Buffer Layout
-------------

Integrals are stored in row-major order within buffers:

**One-electron (shell pair):**

For shells with ``na`` and ``nb`` basis functions:

.. code-block:: cpp

   // Access integral (i, j) where i ∈ [0, na), j ∈ [0, nb)
   Real integral = buffer.data()[i * nb + j];

**Two-electron (shell quartet):**

For shells with ``na``, ``nb``, ``nc``, ``nd`` basis functions:

.. code-block:: cpp

   // Access integral (i, j | k, l)
   Size idx = ((i * nb + j) * nc + k) * nd + l;
   Real integral = buffer.data()[idx];
