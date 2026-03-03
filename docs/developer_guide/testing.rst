.. _testing:

Testing Guide
=============

This guide covers testing practices and infrastructure for LibAccInt.

Testing Philosophy
------------------

LibAccInt follows these testing principles:

1. **Correctness First**: Numerical results must match reference implementations
2. **Coverage**: All public APIs should have tests
3. **Performance Regression**: Benchmark tests catch slowdowns
4. **CI Integration**: All tests run on every PR

Test Organization
-----------------

.. code-block:: text

   tests/
   ├── unit/           # Unit tests for individual components
   │   ├── basis/      # Shell, ShellSet, BasisSet tests
   │   ├── engine/     # Engine tests
   │   ├── math/       # Mathematical utility tests
   │   └── kernels/    # Kernel correctness tests
   │
   ├── integration/    # Integration tests
   │   ├── scf/        # SCF calculation tests
   │   └── gpu/        # GPU backend tests
   │
   ├── reference/      # Reference data
   │   ├── psi4/       # Psi4 reference values
   │   └── libint2/    # Libint2 reference values
   │
   ├── benchmark/      # Performance benchmarks
   │
   └── fixtures/       # Test fixtures and utilities

Running Tests
-------------

Basic Commands
~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   ctest --test-dir build/cpu-release

   # Run with verbose output
   ctest --test-dir build/cpu-release -V

   # Run specific test by name
   ctest --test-dir build/cpu-release -R test_overlap

   # Run tests matching pattern
   ctest --test-dir build/cpu-release -R "test_.*_kernel"

   # Run excluding pattern
   ctest --test-dir build/cpu-release -E benchmark

   # Run with parallelism
   ctest --test-dir build/cpu-release -j8

WSL2 CUDA + MPI Notes
~~~~~~~~~~~~~~~~~~~~~

For WSL2 validation on RTX 50-series GPUs and system OpenMPI installs, use a
fresh build directory and explicit tool paths, or the preset below:

.. code-block:: bash

   cmake --preset cuda-release-safe-mpi-tests
   cmake --build --preset cuda-release-safe-mpi-tests
   ctest --preset cuda-release-safe-mpi-tests

The preset expands to the same system CUDA/OpenMPI configuration. The manual
equivalent is:

.. code-block:: bash

   cmake -S . -B build/cuda-release-safe-tests -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DLIBACCINT_USE_CUDA=ON \
     -DLIBACCINT_USE_MPI=ON \
     -DMPI_CXX_COMPILER=/usr/bin/mpicxx \
     -DGTest_DIR=/usr/lib/x86_64-linux-gnu/cmake/GTest \
     -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
     -DCUDAToolkit_ROOT=/usr/local/cuda-12.6

   cmake --build build/cuda-release-safe-tests \
     --target test_mpi_fock_builder test_shellsetquartet_batched_fock \
     --parallel 6

   ./build/cuda-release-safe-tests/tests/test_shellsetquartet_batched_fock
   /usr/bin/mpirun -np 2 ./build/cuda-release-safe-tests/tests/test_mpi_fock_builder

Use ``/usr/bin/mpirun`` and ``/usr/bin/mpicxx`` to avoid accidentally picking a
conda-installed MPI wrapper from ``PATH``.

On an RTX 5070, the first CUDA test may spend roughly 30-40 seconds in PTX JIT
before the rest of the suite runs at normal speed. That is expected when the
build targets ``sm_90`` and relies on JIT for Blackwell-class hardware.

If tests are being run inside a restricted sandbox, CUDA discovery and PMIx
startup may fail even when the WSL2 host is configured correctly. In that case,
rerun the runtime tests unsandboxed.

Debug Builds
~~~~~~~~~~~~

For debugging failing tests:

.. code-block:: bash

   cmake --preset cpu-debug
   cmake --build --preset cpu-debug

   # Run single test with full output
   ctest --test-dir build/cpu-debug -R test_overlap -V

   # Or run directly
   ./build/cpu-debug/tests/test_overlap

Address Sanitizer
~~~~~~~~~~~~~~~~~

Catch memory errors:

.. code-block:: bash

   cmake --preset cpu-debug-asan
   cmake --build --preset cpu-debug-asan
   ctest --test-dir build/cpu-debug-asan

Writing Tests
-------------

Unit Tests
~~~~~~~~~~

Use GoogleTest:

.. code-block:: cpp

   // tests/unit/basis/test_shell.cpp
   #include <gtest/gtest.h>
   #include <libaccint/basis/shell.hpp>

   using namespace libaccint;

   class ShellTest : public ::testing::Test {
   protected:
       void SetUp() override {
           s_shell_ = Shell(AngularMomentum::S, {0, 0, 0}, {1.0}, {1.0});
           p_shell_ = Shell(AngularMomentum::P, {0, 0, 0}, {1.0}, {1.0});
       }

       Shell s_shell_;
       Shell p_shell_;
   };

   TEST_F(ShellTest, AngularMomentumCorrect) {
       EXPECT_EQ(s_shell_.angular_momentum(), 0);
       EXPECT_EQ(p_shell_.angular_momentum(), 1);
   }

   TEST_F(ShellTest, NFunctionsCorrect) {
       EXPECT_EQ(s_shell_.n_functions(), 1);  // 1 s function
       EXPECT_EQ(p_shell_.n_functions(), 3);  // 3 p functions (px, py, pz)
   }

   TEST_F(ShellTest, NormalizationSelfOverlap) {
       // Self-overlap should be 1.0 after normalization
       // (test by computing overlap integral)
   }

   TEST(ShellDeathTest, NegativeAMThrows) {
       EXPECT_THROW(Shell(-1, {0,0,0}, {1.0}, {1.0}),
                    InvalidArgumentException);
   }

Numerical Tests
~~~~~~~~~~~~~~~

Use test utilities for floating-point comparisons:

.. code-block:: cpp

   #include "test_utils.hpp"

   TEST(OverlapTest, H2OReference) {
       // Setup H2O/STO-3G
       auto [basis, engine] = create_h2o_sto3g();

       std::vector<Real> S;
       engine.compute(OneElectronOperator(Operator::overlap()), S);

       // Reference from Psi4
       std::vector<Real> S_ref = load_reference("h2o_sto3g_overlap.txt");

       EXPECT_MATRICES_NEAR(S, S_ref, 1e-10);
   }

   TEST(ERITest, TwoElectronIntegral) {
       // Test specific integral value
       auto [basis, engine] = create_h2();
       TwoElectronBuffer<0> buffer;

       engine.compute(Operator::coulomb(),
                      basis.shell(0), basis.shell(0),
                      basis.shell(0), basis.shell(0),
                      buffer);

       // Reference: (ss|ss) for H2/STO-3G
       EXPECT_NEAR(buffer.data()[0], 0.7746059439198978, 1e-10);
   }

Parameterized Tests
~~~~~~~~~~~~~~~~~~~

Test multiple cases:

.. code-block:: cpp

   class AMTest : public ::testing::TestWithParam<int> {};

   TEST_P(AMTest, NFunctionsFormula) {
       int L = GetParam();
       EXPECT_EQ(n_cartesian(L), (L+1)*(L+2)/2);
   }

   INSTANTIATE_TEST_SUITE_P(
       AllAM, AMTest,
       ::testing::Range(0, 7)  // Test L=0 through L=6
   );

GPU Tests
~~~~~~~~~

Test GPU functionality:

.. code-block:: cpp

   #if LIBACCINT_USE_CUDA

   TEST(CudaEngineTest, OverlapMatchesCPU) {
       auto [basis, engine] = create_h2o_sto3g();

       std::vector<Real> S_cpu, S_gpu;

       engine.compute(Operator::overlap(), S_cpu, BackendHint::ForceCPU);
       engine.compute(Operator::overlap(), S_gpu, BackendHint::ForceGPU);

       EXPECT_MATRICES_NEAR(S_cpu, S_gpu, 1e-12);
   }

   #endif  // LIBACCINT_USE_CUDA

Test Utilities
--------------

The ``test_utils.hpp`` header provides:

.. code-block:: cpp

   // Matrix comparison
   EXPECT_MATRICES_NEAR(actual, expected, tolerance);

   // Relative error check
   EXPECT_RELATIVE_ERROR_BELOW(actual, expected, tolerance);

   // Load reference data
   std::vector<Real> ref = load_reference("path/to/reference.txt");

   // Create standard test systems
   auto [basis, engine] = create_h2o_sto3g();
   auto [basis, engine] = create_benzene_sto3g();

Reference Data
--------------

Reference values are generated using established codes:

- **Psi4**: Primary reference for most integrals
- **Libint2**: Cross-validation
- **ORCA**: Additional validation for special cases

Generating references:

.. code-block:: python

   # scripts/generate_reference.py
   import psi4

   mol = psi4.geometry("""
   O  0.0  0.0  0.0
   H  0.0  1.43  -1.11
   H  0.0 -1.43  -1.11
   """)

   psi4.set_options({'basis': 'sto-3g'})
   wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
   mints = psi4.core.MintsHelper(wfn)

   S = mints.ao_overlap().np
   np.savetxt('h2o_sto3g_overlap.txt', S.flatten())

Benchmarking
------------

Performance tests:

.. code-block:: cpp

   #include <benchmark/benchmark.h>

   static void BM_OverlapMatrix(benchmark::State& state) {
       auto [basis, engine] = create_h2o_sto3g();
       std::vector<Real> S;

       for (auto _ : state) {
           engine.compute(Operator::overlap(), S);
           benchmark::DoNotOptimize(S);
       }
   }
   BENCHMARK(BM_OverlapMatrix);

Run benchmarks:

.. code-block:: bash

   ./build/cpu-release/tests/benchmarks

CI/CD Integration
-----------------

Tests run automatically on:

- Every push
- Every pull request
- Nightly scheduled runs

CI checks:

- Build on Linux/macOS/Windows
- Run test suite
- Check code formatting
- Run address sanitizer
- Check Doxygen warnings

Best Practices
--------------

1. **Test edge cases**: Empty input, single element, maximum size
2. **Test error conditions**: Invalid input should throw
3. **Use fixtures**: Reuse setup code
4. **Keep tests fast**: Unit tests should run in < 1 second
5. **Isolate tests**: No test should depend on another
6. **Name tests clearly**: ``TEST(Component, WhatItDoes)``
7. **Document test purpose**: Add comments for complex tests
