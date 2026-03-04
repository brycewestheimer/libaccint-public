.. _theory-gpu:

GPU Optimization Strategies
===========================

This document describes the GPU optimization techniques used in LibAccInt.

Overview
--------

GPU acceleration for molecular integrals presents unique challenges:

1. **Irregular parallelism**: Shells have varying sizes
2. **Memory bandwidth**: Large intermediate data
3. **Branch divergence**: Different code paths per angular momentum

LibAccInt addresses these through batched execution and specialized kernels.

ShellSet Batching
-----------------

The key innovation is **ShellSet batching**: grouping shells with identical
properties for uniform parallel execution.

Traditional Approach
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   For each shell pair (i, j):
       Launch kernel
       Synchronize
   → Many small kernel launches, high overhead

ShellSet Approach
~~~~~~~~~~~~~~~~~

.. code-block:: text

   Group shells by (AM, n_primitives)
   For each ShellSet pair:
       Launch single kernel for entire batch
   → Few large kernel launches, high throughput

Benefits:

- Amortizes kernel launch overhead
- Enables coalesced memory access
- Maximizes GPU occupancy

Structure-of-Arrays (SoA)
-------------------------

GPU memory access patterns favor contiguous data:

Array-of-Structures (Bad)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   struct Shell {
       Real center[3];
       Real exponents[N];
       Real coefficients[N];
   };
   Shell shells[M];

   // Thread 0 accesses shells[0].center[0]
   // Thread 1 accesses shells[1].center[0]
   // → Non-contiguous, poor coalescing

Structure-of-Arrays (Good)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   struct ShellSetDataSoA {
       Real* centers_x;      // All x coordinates contiguous
       Real* centers_y;      // All y coordinates contiguous
       Real* centers_z;
       Real* exponents;      // All exponents contiguous
       Real* coefficients;
   };

   // Thread 0 accesses centers_x[0]
   // Thread 1 accesses centers_x[1]
   // → Contiguous, perfect coalescing

Memory Coalescing
~~~~~~~~~~~~~~~~~

For 128-byte cache lines with 32-thread warps:

- Perfect coalescing: 1 transaction per warp
- Random access: up to 32 transactions per warp

SoA achieves near-perfect coalescing for all data access.

Kernel Specialization
---------------------

Per-AM Class Kernels
~~~~~~~~~~~~~~~~~~~~

Rather than runtime conditionals, generate specialized kernels:

.. code-block:: cpp

   // Generated kernels for each AM combination
   template<> void eri_kernel<0,0,0,0>(...);  // (ss|ss)
   template<> void eri_kernel<0,0,0,1>(...);  // (ss|sp)
   template<> void eri_kernel<0,0,1,1>(...);  // (ss|pp)
   // ... etc

Benefits:

- No branch divergence within a kernel
- Compiler can fully unroll loops
- Register usage optimized per case

Code Generation
~~~~~~~~~~~~~~~

LibAccInt generates kernels using Python:

.. code-block:: bash

   libaccint-codegen --max-am 3 --backends cuda --output generated/

This produces optimized kernels for all AM combinations up to d-functions.

GPU Memory Management
---------------------

Memory Pools
~~~~~~~~~~~~

Avoid repeated allocation/deallocation:

.. code-block:: cpp

   class MemoryPool {
       std::vector<void*> free_list_;
       size_t block_size_;

   public:
       void* allocate() {
           if (free_list_.empty()) {
               cudaMalloc(&ptr, block_size_);
               return ptr;
           }
           return free_list_.pop_back();
       }

       void deallocate(void* ptr) {
           free_list_.push_back(ptr);  // Don't actually free
       }
   };

Pinned Memory
~~~~~~~~~~~~~

For CPU-GPU transfers, use pinned (page-locked) memory:

.. code-block:: cpp

   // Pageable memory (slow transfer)
   std::vector<Real> data(n);
   cudaMemcpy(d_data, data.data(), ...);

   // Pinned memory (fast transfer)
   Real* h_data;
   cudaMallocHost(&h_data, n * sizeof(Real));
   cudaMemcpy(d_data, h_data, ...);  // ~2x faster

Streams and Overlap
~~~~~~~~~~~~~~~~~~~

Overlap computation with data transfer:

.. code-block:: cpp

   // Stream 1: compute batch A
   kernel<<<grid, block, 0, stream1>>>(batch_a);

   // Stream 2: transfer batch B while A computes
   cudaMemcpyAsync(d_batch_b, h_batch_b, ..., stream2);

   // Stream 1: compute batch B after A finishes
   cudaStreamSynchronize(stream1);
   kernel<<<grid, block, 0, stream1>>>(batch_b);

Device-Side Fock Accumulation
-----------------------------

The key optimization for Fock matrix construction:

Traditional Approach
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   For each shell quartet:
       Compute ERIs on GPU
       Transfer to CPU
       Contract with density on CPU
       Accumulate into J/K
   → O(N⁴) data transfer

Device-Side Approach
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Upload density matrix once
   For each ShellSet quartet:
       Compute ERIs on GPU
       Contract and accumulate on GPU (atomic adds)
   Download J/K once
   → O(N²) data transfer

.. code-block:: cpp

   __global__ void fused_eri_fock_kernel(...) {
       // Compute ERI in registers
       Real eri = compute_eri(...);

       // Contract with density (from global memory)
       Real d_ls = D[l * nbf + s];

       // Atomic accumulate into J and K
       atomicAdd(&J[m * nbf + n], eri * d_ls);
       atomicAdd(&K[m * nbf + l], eri * d_ns);
   }

Thread Organization
-------------------

Grid and Block Sizing
~~~~~~~~~~~~~~~~~~~~~

For ERI kernels:

.. code-block:: cpp

   // One block per shell quartet
   dim3 grid(n_quartets);

   // Threads handle primitives and/or basis functions
   dim3 block(n_primitives * n_functions);

For simple integrals (overlap, kinetic):

.. code-block:: cpp

   // One thread per basis function pair
   dim3 grid((n_pairs + 255) / 256);
   dim3 block(256);

Shared Memory Usage
~~~~~~~~~~~~~~~~~~~

Use shared memory for frequently accessed data:

.. code-block:: cpp

   __global__ void eri_kernel(...) {
       __shared__ Real s_exponents[MAX_PRIMITIVES];
       __shared__ Real s_coefficients[MAX_PRIMITIVES];

       // Load to shared memory (coalesced)
       if (threadIdx.x < n_primitives) {
           s_exponents[threadIdx.x] = exponents[shell_idx * n_primitives + threadIdx.x];
       }
       __syncthreads();

       // Use shared memory (fast, no bank conflicts)
       Real exp = s_exponents[prim_idx];
   }

Warp-Level Primitives
~~~~~~~~~~~~~~~~~~~~~

Use warp shuffle for reductions:

.. code-block:: cpp

   __device__ Real warp_reduce_sum(Real val) {
       for (int offset = 16; offset > 0; offset /= 2) {
           val += __shfl_down_sync(0xffffffff, val, offset);
       }
       return val;
   }

Performance Profiling
---------------------

Using NVIDIA Tools
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Profile with Nsight Compute
   ncu --set full ./benchmark

   # Profile with Nsight Systems
   nsys profile ./benchmark

Key Metrics
~~~~~~~~~~~

- **Occupancy**: Target 50%+ for compute-bound kernels
- **Memory throughput**: Target 80%+ of peak bandwidth
- **Warp efficiency**: Target 90%+ (minimal divergence)

When GPU is Slower
------------------

GPU overhead exceeds benefit for:

- Small molecules (< 50 atoms)
- Small basis sets (STO-3G)
- Few shell quartets (< 1000)

The dispatch policy automatically routes small work to CPU:

.. code-block:: cpp

   bool should_use_gpu(Size n_quartets) {
       return n_quartets >= MIN_WORK_FOR_GPU;  // Default: 1000
   }

References
----------

1. Ufimtsev, I. S.; Martínez, T. J. J. Chem. Theory Comput. **2008**, 4, 222.
2. Titov, A. V.; et al. J. Chem. Theory Comput. **2013**, 9, 213.
3. NVIDIA CUDA C++ Programming Guide.
