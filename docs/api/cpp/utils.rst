.. _api-utils:

Utilities API
=============

Utility classes and functions.

Error Handling
--------------

.. doxygenclass:: libaccint::Exception
   :members:

.. doxygenclass:: libaccint::InvalidArgumentException
   :members:

.. doxygenclass:: libaccint::BackendException
   :members:

.. doxygenclass:: libaccint::NotImplementedException
   :members:

Error Macros
~~~~~~~~~~~~

.. code-block:: cpp

   // Throw with message
   LIBACCINT_THROW(InvalidArgumentException, "Invalid shell");

   // Assert with message
   LIBACCINT_ASSERT(n > 0, "n must be positive");

   // Check precondition
   LIBACCINT_PRECONDITION(ptr != nullptr);

Constants
---------

.. doxygennamespace:: libaccint::constants
   :members:

Key constants:

- ``PI`` - π
- ``TWO_PI`` - 2π
- ``SQRT_PI`` - √π
- ``BOHR_TO_ANGSTROM`` - Bohr to Angstrom conversion
- ``ANGSTROM_TO_BOHR`` - Angstrom to Bohr conversion
- ``HARTREE_TO_EV`` - Hartree to eV conversion
- ``MAX_ANGULAR_MOMENTUM`` - Maximum supported L value

Matrix Assembly
---------------

.. doxygennamespace:: libaccint::utils::matrix
   :members:

Timing Utilities
----------------

.. doxygenclass:: libaccint::utils::Timer
   :members:

.. code-block:: cpp

   utils::Timer timer;
   timer.start();
   // ... computation ...
   timer.stop();
   std::cout << "Elapsed: " << timer.elapsed_ms() << " ms\n";
