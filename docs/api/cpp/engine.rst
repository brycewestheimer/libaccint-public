.. _api-engine:

Engine API
==========

The Engine class is the central orchestrator for molecular integral computation.

Engine Class
------------

.. doxygenclass:: libaccint::Engine
   :members:
   :protected-members:
   :undoc-members:

CpuEngine
---------

.. doxygenclass:: libaccint::engine::CpuEngine
   :members:
   :undoc-members:

CudaEngine
----------

.. doxygenclass:: libaccint::engine::CudaEngine
   :members:
   :undoc-members:

Dispatch Policy
---------------

.. doxygenclass:: libaccint::DispatchPolicy
   :members:

.. doxygenstruct:: libaccint::DispatchConfig
   :members:

Backend Types
-------------

.. doxygenenum:: libaccint::BackendType

.. doxygenenum:: libaccint::BackendHint

.. doxygenfunction:: libaccint::is_backend_available
