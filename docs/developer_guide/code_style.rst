.. _code-style:

Code Style Guide
================

This document describes the coding standards for LibAccInt.

C++ Style
---------

General Guidelines
~~~~~~~~~~~~~~~~~~

- **C++ Standard**: C++20
- **Line Length**: 100 characters maximum
- **Indentation**: 4 spaces (no tabs)
- **Braces**: Allman style for functions, K&R for control structures

Naming Conventions
~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Classes: PascalCase
   class ShellSetQuartet;

   // Functions/Methods: snake_case
   void compute_overlap();
   Real get_value() const;

   // Variables: snake_case
   Size n_basis_functions;
   Real total_energy;

   // Constants: UPPER_SNAKE_CASE
   constexpr Real MAX_ANGULAR_MOMENTUM = 6;

   // Template Parameters: PascalCase
   template<typename BufferType, int DerivOrder>

   // Namespaces: lowercase
   namespace libaccint::math { }

   // Macros: UPPER_SNAKE_CASE with prefix
   #define LIBACCINT_ASSERT(cond, msg)

   // Private members: trailing underscore
   class Shell {
       Real* exponents_;
       Size n_primitives_;
   };

File Organization
~~~~~~~~~~~~~~~~~

**Headers (.hpp)**:

.. code-block:: cpp

   // SPDX-License-Identifier: BSD-3-Clause
   // Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

   #pragma once

   /// @file shell.hpp
   /// @brief Shell class for contracted Gaussian functions

   #include <libaccint/core/types.hpp>  // Project headers first
   #include <vector>                     // Standard library
   #include <span>

   namespace libaccint {

   /// @brief Brief description
   ///
   /// Detailed description.
   class Shell {
   public:
       // Constructors
       Shell();
       explicit Shell(int am);

       // Public methods
       int angular_momentum() const noexcept;

   private:
       int am_;
   };

   }  // namespace libaccint

**Source files (.cpp)**:

.. code-block:: cpp

   // SPDX-License-Identifier: BSD-3-Clause
   // Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

   #include <libaccint/basis/shell.hpp>

   #include <libaccint/utils/error_handling.hpp>

   #include <algorithm>
   #include <cmath>

   namespace libaccint {

   Shell::Shell() : am_(0) {}

   Shell::Shell(int am) : am_(am) {
       LIBACCINT_ASSERT(am >= 0 && am <= MAX_AM, "Invalid AM");
   }

   int Shell::angular_momentum() const noexcept {
       return am_;
   }

   }  // namespace libaccint

Modern C++ Features
~~~~~~~~~~~~~~~~~~~

**Use**:

.. code-block:: cpp

   // std::span for array views
   void process(std::span<const Real> data);

   // Structured bindings
   auto [J, K] = compute_jk(D);

   // constexpr where possible
   constexpr Size n_cartesian(int L) { return (L+1)*(L+2)/2; }

   // [[nodiscard]] for important return values
   [[nodiscard]] Size size() const noexcept;

   // Concepts for templates
   template<IntegralConsumer C>
   void compute(Operator op, C& consumer);

**Avoid**:

.. code-block:: cpp

   // Raw pointers for ownership (use smart pointers)
   // new/delete (use make_unique/make_shared)
   // C-style casts (use static_cast, etc.)
   // macros for constants (use constexpr)

Documentation Style
~~~~~~~~~~~~~~~~~~~

Use Doxygen-style comments:

.. code-block:: cpp

   /// @brief Compute overlap integral for a shell pair
   ///
   /// Computes the overlap integral matrix elements between two shells
   /// using the Obara-Saika recurrence relation.
   ///
   /// @param shell_a First shell
   /// @param shell_b Second shell
   /// @param[out] buffer Output buffer for integrals
   ///
   /// @throws InvalidArgumentException if shells are invalid
   ///
   /// @note Buffer must be pre-allocated with sufficient size
   ///
   /// @see Shell::n_functions() for determining buffer size
   void compute_overlap(const Shell& shell_a,
                        const Shell& shell_b,
                        OneElectronBuffer& buffer);

Error Handling
~~~~~~~~~~~~~~

.. code-block:: cpp

   // Use exceptions for errors
   if (am < 0 || am > MAX_AM) {
       throw InvalidArgumentException(
           "Angular momentum must be in [0, " +
           std::to_string(MAX_AM) + "]");
   }

   // Use assertions for programming errors (debug only)
   LIBACCINT_ASSERT(ptr != nullptr, "Null pointer");

   // Use optional for missing values
   std::optional<Real> find_value(Key key);

   // Use expected/result for recoverable errors (C++23)

Python Style
------------

Follow PEP 8 with these specifics:

.. code-block:: python

   """Module docstring."""

   import numpy as np
   from typing import List, Optional

   import libaccint


   class MyClass:
       """Class docstring.

       Attributes:
           value: Description of value.
       """

       def __init__(self, value: float) -> None:
           """Initialize MyClass.

           Args:
               value: Initial value.
           """
           self.value = value

       def compute(self, data: np.ndarray) -> np.ndarray:
           """Compute something.

           Args:
               data: Input array.

           Returns:
               Computed result.

           Raises:
               ValueError: If data is empty.
           """
           if data.size == 0:
               raise ValueError("Data cannot be empty")
           return data * self.value

Formatting Tools
----------------

C++
~~~

Use clang-format with provided configuration:

.. code-block:: bash

   # Format a file
   clang-format -i include/libaccint/engine/engine.hpp

   # Format all files
   find include src -name "*.hpp" -o -name "*.cpp" | xargs clang-format -i

   # Check formatting (CI)
   find include src -name "*.hpp" -o -name "*.cpp" | xargs clang-format --dry-run -Werror

Python
~~~~~~

Use black and isort:

.. code-block:: bash

   # Format
   black python/
   isort python/

   # Check
   black --check python/
   isort --check python/

CMake
~~~~~

.. code-block:: cmake

   # Function names: lowercase with underscores
   function(configure_backend backend_name)

   # Variables: UPPER_CASE for cache, lower_case for local
   set(LIBACCINT_USE_CUDA ON CACHE BOOL "Enable CUDA")
   set(source_files src/main.cpp)

   # Consistent indentation (2 spaces)
   if(LIBACCINT_USE_CUDA)
     find_package(CUDAToolkit REQUIRED)
     target_link_libraries(libaccint PRIVATE CUDA::cudart)
   endif()

Pre-commit Hooks
----------------

Install pre-commit hooks to check formatting before commits:

.. code-block:: bash

   pip install pre-commit
   pre-commit install

   # Run manually
   pre-commit run --all-files
