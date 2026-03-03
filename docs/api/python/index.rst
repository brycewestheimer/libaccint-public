.. _api-python:

Python API Reference
====================

This section documents the Python bindings for LibAccInt.

.. module:: libaccint

Installation
------------

.. code-block:: bash

   # Build with Python bindings
   cmake --preset cpu-release -DLIBACCINT_BUILD_PYTHON=ON
   cmake --build --preset cpu-release

   # Install
   cd python
   pip install -e .

Module Contents
---------------

.. toctree::
   :maxdepth: 2

   core
   convenience

Quick Reference
---------------

Core Classes
~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   Engine
   BasisSet
   Shell
   ShellSet
   Atom
   FockBuilder

Operators
~~~~~~~~~

.. autosummary::
   :nosignatures:

   Operator
   OneElectronOperator
   OperatorKind
   PointChargeParams
   RangeSeparatedParams

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   basis_set
   compute_overlap
   compute_kinetic
   compute_nuclear
   compute_core_hamiltonian
   build_fock

Types
~~~~~

.. autosummary::
   :nosignatures:

   Real
   Index
   Size
   AngularMomentum
   Point3D
   BackendType
   BackendHint
