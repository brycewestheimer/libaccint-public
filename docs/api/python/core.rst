.. _api-python-core:

Core Classes
============

.. module:: libaccint
   :no-index:

Engine
------

.. autoclass:: Engine
   :members:
   :undoc-members:
   :show-inheritance:

   The Engine class is the central orchestrator for integral computation.

   **Example:**

   .. code-block:: python

      import libaccint

      atoms = [libaccint.Atom(8, [0, 0, 0])]
      basis = libaccint.basis_set("sto-3g", atoms)
      engine = libaccint.Engine(basis)

      S = engine.compute_overlap_matrix()

BasisSet
--------

.. autoclass:: BasisSet
   :members:
   :undoc-members:
   :show-inheritance:

   A collection of shells representing the molecular basis set.

Shell
-----

.. autoclass:: Shell
   :members:
   :undoc-members:
   :show-inheritance:

   A contracted Gaussian shell.

   **Example:**

   .. code-block:: python

      shell = libaccint.Shell(
          am=0,  # s-type
          center=[0.0, 0.0, 0.0],
          exponents=[3.42, 0.62, 0.17],
          coefficients=[0.15, 0.54, 0.44]
      )

ShellSet
--------

.. autoclass:: ShellSet
   :members:
   :undoc-members:
   :show-inheritance:

   A batch of shells with uniform angular momentum and primitive count.

Atom
----

.. autoclass:: Atom
   :members:
   :undoc-members:
   :show-inheritance:

   Represents an atom with atomic number and position.

   **Example:**

   .. code-block:: python

      # Oxygen at origin
      oxygen = libaccint.Atom(8, [0.0, 0.0, 0.0])

      # Hydrogen
      hydrogen = libaccint.Atom(1, [0.0, 1.43, -1.11])

FockBuilder
-----------

.. autoclass:: FockBuilder
   :members:
   :undoc-members:
   :show-inheritance:

   Consumer for building Fock matrices via compute-and-consume pattern.

   **Example:**

   .. code-block:: python

      fock = libaccint.FockBuilder(nbf)
      fock.set_density(D)  # D is a numpy array

      engine.compute_and_consume(libaccint.Operator.coulomb(), fock)

      J = fock.get_coulomb_matrix()
      K = fock.get_exchange_matrix()

Operators
---------

Operator
~~~~~~~~

.. autoclass:: Operator
   :members:
   :undoc-members:

   Factory class for creating integral operators.

   **Class Methods:**

   - ``overlap()`` - Create overlap operator
   - ``kinetic()`` - Create kinetic energy operator
   - ``nuclear(charges)`` - Create nuclear attraction operator
   - ``coulomb()`` - Create Coulomb (ERI) operator

OneElectronOperator
~~~~~~~~~~~~~~~~~~~

.. autoclass:: OneElectronOperator
   :members:
   :undoc-members:

OperatorKind
~~~~~~~~~~~~

.. autoclass:: OperatorKind
   :members:
   :undoc-members:

   Enumeration of operator types: ``OVERLAP``, ``KINETIC``, ``NUCLEAR``, ``COULOMB``.

PointChargeParams
~~~~~~~~~~~~~~~~~

.. autoclass:: PointChargeParams
   :members:
   :undoc-members:

   Parameters for nuclear attraction integrals.

RangeSeparatedParams
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RangeSeparatedParams
   :members:
   :undoc-members:

   Parameters for range-separated operators.

Buffers
-------

OneElectronBuffer
~~~~~~~~~~~~~~~~~

.. autoclass:: OneElectronBuffer
   :members:
   :undoc-members:

TwoElectronBuffer
~~~~~~~~~~~~~~~~~

.. autoclass:: TwoElectronBuffer
   :members:
   :undoc-members:

Backend Control
---------------

BackendType
~~~~~~~~~~~

.. autoclass:: BackendType
   :members:
   :undoc-members:

   Enumeration: ``CPU``, ``CUDA``.

BackendHint
~~~~~~~~~~~

.. autoclass:: BackendHint
   :members:
   :undoc-members:

   Enumeration: ``Auto``, ``ForceCPU``, ``PreferGPU``, ``ForceGPU``.

is_backend_available
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: is_backend_available

   Check if a backend is available at runtime.

   **Example:**

   .. code-block:: python

      if libaccint.is_backend_available(libaccint.BackendType.CUDA):
          print("CUDA is available!")

Types
-----

Real
~~~~

Floating-point type used for integral values (typically ``float64``).

Index
~~~~~

Signed integer type for indices.

Size
~~~~

Unsigned integer type for sizes and counts.

AngularMomentum
~~~~~~~~~~~~~~~

.. autoclass:: AngularMomentum
   :members:
   :undoc-members:

   Enumeration: ``S``, ``P``, ``D``, ``F``, ``G``, ``H``.

Point3D
~~~~~~~

.. autoclass:: Point3D
   :members:
   :undoc-members:

   3D point with x, y, z coordinates.
