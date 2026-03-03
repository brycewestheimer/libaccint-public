.. _theory-performance:

Performance Tuning Guide
========================

This guide helps you optimize LibAccInt performance for your specific use case.

Profiling Your Workload
-----------------------

Before optimizing, measure:

.. code-block:: cpp

   #include <chrono>
   #include <iostream>

   auto start = std::chrono::high_resolution_clock::now();

   engine.compute(Operator::coulomb(), fock);

   auto end = std::chrono::high_resolution_clock::now();
   auto ms = std::chrono::duration<double, std::milli>(end - start).count();
   std::cout << "Fock build: " << ms << " ms\n";

Or use the built-in timer:

.. code-block:: cpp

   utils::Timer timer;
   timer.start();
   engine.compute(op, result);
   timer.stop();
   std::cout << timer.elapsed_ms() << " ms\n";

Backend Selection
-----------------

Choosing the Right Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~

+------------------+------------------+------------------+
| Workload         | Recommended      | Reason           |
+==================+==================+==================+
| < 50 atoms       | CPU              | GPU overhead     |
+------------------+------------------+------------------+
| 50-500 atoms     | Auto or GPU      | GPU shows benefit|
+------------------+------------------+------------------+
| > 500 atoms      | GPU              | Maximum speedup  |
+------------------+------------------+------------------+
| STO-3G           | CPU              | Too small        |
+------------------+------------------+------------------+
| cc-pVDZ+         | GPU              | Enough work      |
+------------------+------------------+------------------+

Tuning Dispatch Thresholds
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   DispatchConfig config{
       .min_work_for_gpu = 500,     // Shell quartets needed for GPU
       .prefer_gpu = true,          // Default to GPU when above threshold
       .enable_auto_tuning = true,  // Learn optimal thresholds at runtime
   };

   engine.set_dispatch_config(config);

Auto-Tuning
~~~~~~~~~~~

Enable runtime auto-tuning to find optimal thresholds:

.. code-block:: cpp

   // Engine learns optimal dispatch points during first few runs
   config.enable_auto_tuning = true;

   // After warmup, thresholds are optimized for your hardware
   for (int i = 0; i < 10; ++i) {
       engine.compute(op, result);  // Learning phase
   }

CPU Optimization
----------------

OpenMP Tuning
~~~~~~~~~~~~~

.. code-block:: bash

   # Set thread count (usually = physical cores)
   export OMP_NUM_THREADS=8

   # Bind threads to cores
   export OMP_PROC_BIND=true

   # Set thread placement
   export OMP_PLACES=cores

For hyperthreaded systems, using physical cores often works best.

SIMD Optimization
~~~~~~~~~~~~~~~~~

Ensure SIMD is enabled at compile time:

.. code-block:: bash

   cmake --preset cpu-release  # Includes -march=native

Check SIMD usage:

.. code-block:: bash

   objdump -d libaccint.so | grep -c "vfmadd\|vmul\|vadd"

Memory Allocation
~~~~~~~~~~~~~~~~~

Pre-allocate buffers for repeated computations:

.. code-block:: cpp

   // Allocate once, reuse
   std::vector<Real> S(nbf * nbf);
   std::vector<Real> T(nbf * nbf);

   for (int iter = 0; iter < max_iter; ++iter) {
       engine.compute(Operator::overlap(), S);  // Reuses buffer
       engine.compute(Operator::kinetic(), T);
   }

GPU Optimization
----------------

Memory Transfer
~~~~~~~~~~~~~~~

Minimize CPU-GPU transfers:

.. code-block:: cpp

   // Bad: Transfer per operation
   for (int i = 0; i < n_ops; ++i) {
       engine.compute(op, result);  // Transfer each time
   }

   // Good: Batch operations
   engine.compute_batch(ops, results);  // Single transfer

Use FockBuilder
~~~~~~~~~~~~~~~

FockBuilder keeps data on GPU:

.. code-block:: cpp

   consumers::FockBuilder fock(basis);
   fock.set_density(D);  // Uploads once

   for (int iter = 0; iter < max_iter; ++iter) {
       engine.compute(Operator::coulomb(), fock);  // No D transfer
       auto J = fock.get_coulomb();  // Downloads J/K once
   }

GPU Memory
~~~~~~~~~~

Monitor GPU memory:

.. code-block:: cpp

   auto& mm = memory::MemoryManager::instance();
   std::cout << "GPU memory: " << mm.gpu_bytes_used() / 1e9 << " GB\n";

Reduce memory for large systems:

.. code-block:: cpp

   // Process in batches if memory-limited
   config.max_batch_size = 10000;  // Limit batch size

Basis Set Considerations
------------------------

Basis Set Impact
~~~~~~~~~~~~~~~~

+------------+------------------+------------------+
| Basis      | ERIs (relative)  | Memory (relative)|
+============+==================+==================+
| STO-3G     | 1×               | 1×               |
+------------+------------------+------------------+
| 6-31G      | 4×               | 2×               |
+------------+------------------+------------------+
| cc-pVDZ    | 16×              | 4×               |
+------------+------------------+------------------+
| cc-pVTZ    | 81×              | 9×               |
+------------+------------------+------------------+

Scaling is approximately:

- Basis functions: O(atoms × functions_per_atom)
- ERIs: O(N⁴) without screening, O(N²) with screening

Angular Momentum
~~~~~~~~~~~~~~~~

Higher angular momentum (d, f, g) is more expensive:

- More basis functions per shell
- More complex recursion relations
- More Rys roots needed

For benchmarking, start with STO-3G and scale up.

Screening
---------

Schwarz Screening
~~~~~~~~~~~~~~~~~

Enable Schwarz screening to skip negligible integrals:

.. code-block:: cpp

   ScreeningOptions opts{
       .threshold = 1e-12,
       .use_schwarz = true,
   };

   engine.set_screening_options(opts);

For large molecules, screening can reduce ERI count by 90%+.

Distance-Based Screening
~~~~~~~~~~~~~~~~~~~~~~~~

Combine with distance cutoffs:

.. code-block:: cpp

   opts.use_distance_cutoff = true;
   opts.distance_threshold = 20.0;  // Bohr

Benchmarking Tips
-----------------

1. **Warmup**: Discard first few iterations (caching, JIT compilation)

   .. code-block:: cpp

      // Warmup
      for (int i = 0; i < 5; ++i) {
          engine.compute(op, result);
      }

      // Benchmark
      timer.start();
      for (int i = 0; i < 100; ++i) {
          engine.compute(op, result);
      }
      timer.stop();
      std::cout << timer.elapsed_ms() / 100 << " ms per iteration\n";

2. **Prevent optimization**: Use results to prevent dead code elimination

   .. code-block:: cpp

      benchmark::DoNotOptimize(result);

3. **Statistical significance**: Run multiple iterations, report mean ± std

4. **Compare fair**: Same molecule, same basis, same precision

5. **Profile, don't guess**: Use profiling tools before optimizing

Common Performance Pitfalls
---------------------------

1. **Not using batched APIs**: Per-shell-pair loops are slow
2. **Storing full ERI tensor**: Use FockBuilder instead
3. **Wrong backend for workload**: Check dispatch thresholds
4. **Disabled OpenMP**: Verify ``OMP_NUM_THREADS`` is set
5. **Debug builds**: Always benchmark Release builds
6. **Memory thrashing**: Pre-allocate buffers

Performance Checklist
---------------------

- [ ] Using Release build (``cpu-release`` or ``cuda-release``)
- [ ] OpenMP enabled and thread count set
- [ ] Using batched/matrix APIs instead of per-pair
- [ ] FockBuilder used for Fock matrix construction
- [ ] Appropriate backend selected for workload size
- [ ] Screening enabled for large molecules
- [ ] Buffers pre-allocated and reused
- [ ] Profiled to identify actual bottleneck
