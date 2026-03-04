.. _faq:

Frequently Asked Questions
==========================

General
-------

What is LibAccInt?
~~~~~~~~~~~~~~~~~~

LibAccInt (**Lib**\rary for **Acc**\elerated **Int**\egrals) is a high-performance
library for computing molecular integrals used in quantum chemistry. It supports
both CPU and GPU (CUDA) backends.

What integrals does LibAccInt support?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently supported integrals:

- **One-electron**: Overlap (S), kinetic energy (T), nuclear attraction (V)
- **Two-electron**: Electron repulsion integrals (ERI) via Rys quadrature
- **Angular momentum**: Stable support up to g functions (Cartesian/spherical, L=0..4)

Configuring AM values above ``L=4`` is rejected at build/codegen boundaries with a
diagnostic describing the current stable contract.

What basis sets are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Built-in**: STO-3G
- **JSON loading**: Any basis set in QCSchema JSON format (cc-pVDZ, cc-pVTZ, etc.)
- **Custom**: Create your own shells programmatically

Installation
------------

Why does CMake fail with "CMake version too old"?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LibAccInt requires CMake 3.25+. Upgrade with:

.. code-block:: bash

   pip install cmake --upgrade

Or download from https://cmake.org/download/

Why does compilation fail with C++20 errors?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LibAccInt requires a C++20 compiler. Supported versions:

- GCC 11+
- Clang 14+
- MSVC 19.30+ (Visual Studio 2022)

On Ubuntu:

.. code-block:: bash

   sudo apt install g++-11
   export CXX=g++-11
   cmake --preset cpu-release

Why isn't CUDA detected?
~~~~~~~~~~~~~~~~~~~~~~~~

Ensure CUDA is in your PATH:

.. code-block:: bash

   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   nvcc --version  # Should show version

Then reconfigure:

.. code-block:: bash

   cmake --preset cuda-release

Why is OpenMP not found on macOS?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apple's Clang doesn't include OpenMP. Use Homebrew's LLVM:

.. code-block:: bash

   brew install llvm
   cmake --preset cpu-release \
       -DCMAKE_C_COMPILER=$(brew --prefix llvm)/bin/clang \
       -DCMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++

Performance
-----------

Why is GPU slower than CPU for my calculation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU acceleration shines for large workloads. The overhead of kernel launches and
data transfers means small systems (like H2O/STO-3G) are faster on CPU.

GPU speedup typically requires:

- 50+ atoms
- Large basis sets (cc-pVDZ or larger)
- 1000+ shell quartets

How can I improve performance?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use batched APIs**: Call ``engine.compute()`` instead of per-shell-pair loops
2. **Use FockBuilder**: Fuses computation and contraction, reducing memory
3. **Enable OpenMP**: Set ``-DLIBACCINT_USE_OPENMP=ON`` (default)
4. **Use GPU for large systems**: Set ``BackendHint::PreferGPU``

Why is memory usage high?
~~~~~~~~~~~~~~~~~~~~~~~~~

If you're storing the full ERI tensor, memory scales as O(N⁴). For large systems,
use the compute-and-consume pattern with FockBuilder instead.

Usage
-----

How do I create a custom basis set?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create shells manually:

.. code-block:: cpp

   std::vector<Shell> shells;
   shells.emplace_back(
       AngularMomentum::S,
       Point3D{0.0, 0.0, 0.0},
       std::vector<Real>{3.42, 0.62, 0.17},   // exponents
       std::vector<Real>{0.15, 0.54, 0.44}    // coefficients
   );

   BasisSet basis(shells);

How do I use a different basis set?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load from QCSchema JSON:

.. code-block:: cpp

   BasisSet basis = data::load_basis_from_json("cc-pvdz.json", atoms);

Or download from the Basis Set Exchange: https://www.basissetexchange.org/

Why are my nuclear attraction integrals wrong?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure you're using Bohr (atomic units) for coordinates, not Angstroms.
1 Bohr ≈ 0.529177 Å.

.. code-block:: cpp

   // Correct: Bohr
   data::Atom oxygen{8, {0.0, 0.0, 0.0}};

   // Wrong: Angstroms
   // data::Atom oxygen{8, {0.0, 0.0, 0.0}};  // Would need conversion

How do I handle derivatives?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gradient and Hessian support is planned for future releases. Check
the development roadmap for status.

Python
------

Why does ``import libaccint`` fail?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure the Python bindings are installed:

.. code-block:: bash

   cd libaccint/python
   pip install -e .

And that you built with Python support:

.. code-block:: bash

   cmake --preset cpu-release -DLIBACCINT_BUILD_PYTHON=ON

How do I convert to/from other libraries?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LibAccInt returns NumPy arrays which work with most Python libraries:

.. code-block:: python

   import numpy as np
   from scipy.linalg import eigh

   S = engine.compute_overlap_matrix()  # NumPy array
   eigenvalues, eigenvectors = eigh(S)  # Use with SciPy

Development
-----------

How do I contribute?
~~~~~~~~~~~~~~~~~~~~

See :ref:`contributing` for contribution guidelines. In brief:

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

How do I report bugs?
~~~~~~~~~~~~~~~~~~~~~

Open an issue on GitHub with:

- LibAccInt version
- Operating system and compiler
- Minimal reproducing example
- Expected vs actual behavior

Where can I get help?
~~~~~~~~~~~~~~~~~~~~~

- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: Questions and community help
- Documentation: You're reading it!
