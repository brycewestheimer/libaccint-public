.. _tutorial-gpu:

Tutorial 5: GPU Acceleration
============================

This tutorial covers using LibAccInt's CUDA backend for GPU-accelerated
integral computation.

GPU Architecture Overview
-------------------------

LibAccInt's GPU backend uses:

- **Batched kernels**: Process entire ShellSets in a single kernel launch
- **Device-side accumulation**: Build Fock matrices without CPU-GPU transfers
- **Intelligent dispatch**: Automatically routes work to CPU or GPU based on size

When to Use GPU
---------------

GPU acceleration provides speedup for:

- **Large molecules**: 50+ atoms typically shows benefit
- **Large basis sets**: cc-pVDZ and larger
- **Many shell pairs**: More parallelism for the GPU

For small systems like H2O/STO-3G, CPU is often faster due to kernel launch overhead.

Checking GPU Availability
-------------------------

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>

   using namespace libaccint;

   // Check at compile time
   #if LIBACCINT_USE_CUDA
   std::cout << "CUDA support compiled in\n";
   #endif

   // Check at runtime
   if (is_backend_available(BackendType::CUDA)) {
       std::cout << "CUDA GPU available\n";
   }

   // Via engine
   Engine engine(basis);
   std::cout << "Engine GPU available: " << engine.gpu_available() << "\n";

Controlling Backend Selection
-----------------------------

BackendHint
~~~~~~~~~~~

Control per-operation backend selection:

.. code-block:: cpp

   // Let engine decide (default)
   engine.compute(op, result, BackendHint::Auto);

   // Force CPU
   engine.compute(op, result, BackendHint::ForceCPU);

   // Prefer GPU (falls back to CPU if unavailable)
   engine.compute(op, result, BackendHint::PreferGPU);

   // Force GPU (throws if unavailable)
   engine.compute(op, result, BackendHint::ForceGPU);

DispatchConfig
~~~~~~~~~~~~~~

Configure global dispatch behavior:

.. code-block:: cpp

   DispatchConfig config{
       .prefer_gpu = true,              // Default to GPU when available
       .min_work_for_gpu = 1000,        // Minimum work units for GPU dispatch
       .enable_auto_tuning = true,      // Learn optimal thresholds at runtime
   };

   Engine engine(basis, config);

   // Or update later
   engine.set_dispatch_config(config);

Direct Backend Access
~~~~~~~~~~~~~~~~~~~~~

For advanced control, access backends directly:

.. code-block:: cpp

   // CPU engine (always available)
   auto& cpu = engine.cpu_engine();
   cpu.compute(op, buffer);

   #if LIBACCINT_USE_CUDA
   // CUDA engine (only if GPU available)
   if (engine.gpu_available()) {
       auto& cuda = engine.cuda_engine();
       cuda.compute(op, buffer);
   }
   #endif

GPU Fock Matrix Construction
----------------------------

The GPU Fock builder uses device-side accumulation:

.. code-block:: cpp

   consumers::FockBuilder fock(basis);
   fock.set_density(D);

   // Compute on GPU - density uploaded once, J/K accumulated on device
   engine.compute(Operator::coulomb(), fock, BackendHint::PreferGPU);

   // Results automatically transferred back
   auto J = fock.get_coulomb();
   auto K = fock.get_exchange();

Memory Management
-----------------

GPU memory is managed automatically, but you can configure behavior:

.. code-block:: cpp

   // Get memory manager
   auto& mm = memory::MemoryManager::instance();

   // Configure memory pool
   mm.set_gpu_memory_limit(8 * 1024 * 1024 * 1024);  // 8 GB

   // Get current usage
   std::cout << "GPU memory used: " << mm.gpu_bytes_used() << " bytes\n";

Performance Tips
----------------

1. **Batch operations**: Process all shell pairs of same type together

   .. code-block:: cpp

      // Good: Process all at once
      engine.compute(op, S);

      // Less efficient: Process one at a time
      for (auto& pair : pairs) {
          engine.compute(op, pair, buffer);
      }

2. **Minimize transfers**: Keep data on GPU across operations

   .. code-block:: cpp

      fock.set_density(D);
      // Multiple SCF iterations without re-uploading basis data
      for (int iter = 0; iter < max_iter; ++iter) {
          engine.compute(Operator::coulomb(), fock, BackendHint::PreferGPU);
          // Update density on host, will be re-uploaded
          fock.set_density(D_new);
      }

3. **Use appropriate basis**: GPU shines with larger basis sets

4. **Profile your workload**: Use ``BackendHint::ForceCPU`` vs ``ForceGPU`` to benchmark

Benchmarking Example
--------------------

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>
   #include <chrono>
   #include <iostream>

   using namespace libaccint;

   int main() {
       // Setup large system for meaningful benchmark
       std::vector<data::Atom> atoms = /* large molecule */;
       BasisSet basis = data::create_sto3g(atoms);
       Engine engine(basis);

       Size nbf = basis.n_basis_functions();
       std::vector<Real> D(nbf * nbf, 0.1);  // Dummy density

       consumers::FockBuilder fock(basis);
       fock.set_density(D);

       // Benchmark CPU
       auto cpu_start = std::chrono::high_resolution_clock::now();
       fock.reset();
       engine.compute(Operator::coulomb(), fock, BackendHint::ForceCPU);
       auto cpu_end = std::chrono::high_resolution_clock::now();

       auto cpu_time = std::chrono::duration<double, std::milli>(
           cpu_end - cpu_start).count();
       std::cout << "CPU time: " << cpu_time << " ms\n";

       if (engine.gpu_available()) {
           // Benchmark GPU (warmup)
           fock.reset();
           engine.compute(Operator::coulomb(), fock, BackendHint::ForceGPU);

           // Benchmark GPU
           auto gpu_start = std::chrono::high_resolution_clock::now();
           fock.reset();
           engine.compute(Operator::coulomb(), fock, BackendHint::ForceGPU);
           auto gpu_end = std::chrono::high_resolution_clock::now();

           auto gpu_time = std::chrono::duration<double, std::milli>(
               gpu_end - gpu_start).count();
           std::cout << "GPU time: " << gpu_time << " ms\n";
           std::cout << "Speedup: " << cpu_time / gpu_time << "x\n";
       }

       return 0;
   }

Troubleshooting
---------------

**"No CUDA devices found"**
   - Check that NVIDIA drivers are installed
   - Verify CUDA toolkit is in PATH
   - Run ``nvidia-smi`` to check GPU status

**GPU slower than CPU**
   - Normal for small systems (<1000 shell quartets)
   - Try larger molecules or basis sets
   - Check that you're using batched APIs, not shell-by-shell

**Out of GPU memory**
   - Reduce batch size
   - Use CPU for part of the computation
   - Close other GPU applications

Next Steps
----------

- :doc:`06-python-bindings` - Use GPU from Python
- :doc:`/theory/performance_tuning` - Advanced performance optimization
