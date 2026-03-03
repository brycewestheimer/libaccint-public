.. LibAccInt documentation master file

LibAccInt Documentation
=======================

**LibAccInt** — Library for Accelerated Integrals

A high-performance molecular integral library for computational quantum chemistry
with GPU acceleration (CUDA) and optimized CPU paths.

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

.. image:: https://img.shields.io/badge/C%2B%2B-20-blue.svg
   :target: https://en.cppreference.com/w/cpp/20

Features
--------

- **One-electron integrals**: Overlap (S), kinetic energy (T), nuclear attraction (V)
- **Two-electron integrals**: Electron repulsion integrals (ERI) via Rys quadrature
- **GPU acceleration**: CUDA backend for modern NVIDIA GPUs
- **High performance**: OpenMP parallelization and SIMD optimization
- **Python bindings**: NumPy-integrated Python interface
- **Modern C++20**: Clean API with concepts, spans, and structured bindings

Quick Start
-----------

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>

   using namespace libaccint;

   int main() {
       // Define H2O molecule
       std::vector<data::Atom> atoms = {
           {8, {0.0, 0.0, 0.0}},
           {1, {0.0, 1.43, -1.11}},
           {1, {0.0, -1.43, -1.11}}
       };

       // Create basis set and engine
       auto basis = data::create_builtin_basis("sto-3g", atoms);
       Engine engine(basis);

       // Compute overlap matrix
       std::vector<Real> S;
       engine.compute_overlap_matrix(S);

       return 0;
   }

Contents
--------

.. toctree::
   :maxdepth: 3
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/cpp/index
   api/python/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer_guide/architecture
   developer_guide/contributing
   developer_guide/code_style
   developer_guide/testing

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/integrals
   theory/gpu_optimization
   theory/performance_tuning

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`search`
