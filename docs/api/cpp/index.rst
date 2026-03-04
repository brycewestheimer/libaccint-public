.. _api-cpp:

C++ API Reference
=================

This section provides complete API documentation for LibAccInt's C++ interface,
generated from source code documentation using Doxygen and Breathe.

.. toctree::
   :maxdepth: 2
   :caption: API Modules

   engine
   basis
   operators
   consumers
   buffers
   memory
   math
   utils

Overview
--------

The LibAccInt C++ API is organized into the following namespaces:

``libaccint``
   Root namespace containing core types and the main Engine class.

``libaccint::basis``
   Basis set types: Shell, ShellSet, BasisSet.

``libaccint::engine``
   Engine implementations: CpuEngine, CudaEngine.

``libaccint::consumers``
   Integral consumers: FockBuilder, GpuFockBuilder.

``libaccint::memory``
   Memory management utilities.

``libaccint::math``
   Mathematical utilities: Boys function, Rys quadrature.

``libaccint::data``
   Basis set data and parsers.

Core Classes
------------

Engine
~~~~~~

.. doxygenclass:: libaccint::Engine
   :members:
   :undoc-members:

BasisSet
~~~~~~~~

.. doxygenclass:: libaccint::BasisSet
   :members:
   :undoc-members:

Shell
~~~~~

.. doxygenclass:: libaccint::Shell
   :members:
   :undoc-members:

FockBuilder
~~~~~~~~~~~

.. doxygenclass:: libaccint::consumers::FockBuilder
   :members:
   :undoc-members:

Core Types
----------

.. doxygentypedef:: libaccint::Real
.. doxygentypedef:: libaccint::Index
.. doxygentypedef:: libaccint::Size

.. doxygenstruct:: libaccint::Point3D
   :members:

.. doxygenenum:: libaccint::AngularMomentum
.. doxygenenum:: libaccint::BackendType
.. doxygenenum:: libaccint::BackendHint
