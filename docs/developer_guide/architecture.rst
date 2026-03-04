.. _architecture:

Architecture Overview
=====================

This document describes LibAccInt's high-level architecture and core design decisions.

Design Philosophy
-----------------

LibAccInt is built around several key principles:

1. **Performance First**: Every design decision prioritizes computational efficiency
2. **GPU-Friendly**: Data structures and algorithms optimized for GPU parallelization
3. **Flexibility**: Support multiple backends without code duplication
4. **Modern C++**: Leverage C++20 features for safety and expressiveness

High-Level Architecture
-----------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                         User Code                               │
   └─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                          Engine                                  │
   │  ┌─────────────────────────────────────────────────────────────┐│
   │  │                    DispatchPolicy                           ││
   │  │  (Automatic CPU/GPU routing based on work size & hints)     ││
   │  └─────────────────────────────────────────────────────────────┘│
   │                    ┌────────────┬────────────┐                  │
   │                    ▼            ▼            ▼                  │
   │               CpuEngine    CudaEngine                            │
   └─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                         Kernels                                  │
   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
   │  │ Overlap  │ │ Kinetic  │ │ Nuclear  │ │   ERI    │           │
   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
   └─────────────────────────────────────────────────────────────────┘

Core Components
---------------

Engine
~~~~~~

The ``Engine`` class is the central orchestrator:

- Owns CPU and GPU backend engines
- Implements ``DispatchPolicy`` for automatic backend selection
- Provides unified ``compute()`` API that routes to appropriate backend
- Manages per-backend resources

.. code-block:: cpp

   // Engine automatically routes to optimal backend
   Engine engine(basis);
   engine.compute(op, result);  // CPU or GPU based on heuristics

   // Or force a specific backend
   engine.compute(op, result, BackendHint::ForceGPU);

ShellSet Batching
~~~~~~~~~~~~~~~~~

Shells are grouped into ``ShellSet`` objects for batch processing:

.. code-block:: text

   Shells (variable AM, variable primitives)
       │
       ▼ Group by (AM, n_primitives)
   ShellSets (uniform AM, uniform primitives)
       │
       ▼ Form pairs
   ShellSetPairs
       │
       ▼ Form quartets
   ShellSetQuartets

Benefits:

- All shells in a ShellSet have identical computation structure
- Enables SIMD vectorization on CPU
- Enables coalesced memory access on GPU
- Single kernel launch per ShellSetQuartet (vs. per shell quartet)

Structure-of-Arrays (SoA)
~~~~~~~~~~~~~~~~~~~~~~~~~

GPU-friendly memory layout:

.. code-block:: cpp

   // Array-of-Structures (AoS) - bad for GPU
   struct Shell { Point3D center; Real exponents[N]; ... };
   Shell shells[M];  // Non-contiguous memory access

   // Structure-of-Arrays (SoA) - good for GPU
   struct ShellSetDataSoA {
       Real* centers_x;  // Contiguous memory
       Real* centers_y;
       Real* centers_z;
       Real* exponents;  // All exponents together
       Real* coefficients;
   };

Compute-and-Consume Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fuses computation with consumption to avoid O(N⁴) memory:

.. code-block:: cpp

   // Traditional: Store all ERIs (O(N⁴) memory)
   std::vector<Real> eris = compute_all_eris();
   contract_into_fock(eris, D, J, K);

   // Compute-and-consume: O(N²) memory
   FockBuilder fock(basis);
   fock.set_density(D);
   engine.compute(Operator::coulomb(), fock);  // Fused
   // Integrals are contracted immediately, never stored

Directory Structure
-------------------

.. code-block:: text

   libaccint/
   ├── include/libaccint/     # Public headers
   │   ├── basis/             # Shell, ShellSet, BasisSet
   │   ├── engine/            # Engine, CpuEngine, CudaEngine
   │   ├── consumers/         # FockBuilder, etc.
   │   ├── operators/         # Operator types
   │   ├── buffers/           # Integral buffers
   │   ├── memory/            # Memory management
   │   ├── math/              # Boys, Rys, Gaussian utilities
   │   └── utils/             # Error handling, constants
   │
   ├── src/                   # Implementation
   │   ├── host/              # CPU-specific code
   │   ├── device/            # GPU-specific code (CUDA)
   │   └── shared/            # Backend-agnostic code
   │
   ├── codegen/               # Kernel code generator (not included in alpha)
   │   └── libaccint_codegen/ # Python generator
   │
   └── tests/                 # Test suite
       ├── unit/              # Unit tests
       ├── integration/       # Integration tests
       └── benchmark/         # Performance tests

Namespace Organization
----------------------

.. code-block:: cpp

   namespace libaccint {
       // Root namespace: core types, Engine

       namespace basis {
           // Shell, ShellSet, BasisSet
       }

       namespace engine {
           // CpuEngine, CudaEngine
       }

       namespace consumers {
           // FockBuilder, GpuFockBuilder
       }

       namespace memory {
           // MemoryManager, DeviceMemory
       }

       namespace math {
           // BoysFunction, RysQuadrature
       }

       namespace data {
           // Atom, basis set loaders
       }

       namespace utils {
           // Error handling, constants
       }
   }

Data Flow
---------

One-Electron Integrals
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   BasisSet
       │
       ▼ Create pairs
   ShellSetPairs
       │
       ▼ For each pair
   ┌────────────────┐
   │  Compute kernel│  (overlap/kinetic/nuclear)
   └────────────────┘
       │
       ▼ Accumulate
   Result Matrix (N×N)

Two-Electron Integrals (Fock Build)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   BasisSet + Density Matrix
       │
       ▼ Create quartets
   ShellSetQuartets
       │
       ▼ For each quartet
   ┌────────────────────────────────────┐
   │  ERI kernel  →  Contract with D   │  (fused)
   └────────────────────────────────────┘
       │
       ▼ Accumulate
   J Matrix + K Matrix

Extension Points
----------------

Adding a New Backend
~~~~~~~~~~~~~~~~~~~~

1. Create a new directory under ``src/device/``
2. Implement an engine class following the ``CudaEngine`` pattern
3. Add to ``BackendType`` enum
4. Register in ``DispatchPolicy``

Adding a New Integral Type
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Add to ``OperatorKind`` enum
2. Create kernel in ``src/host/kernels/`` and ``src/device/cuda/kernels/``
3. Register in kernel dispatch tables
4. Add tests

Creating Custom Consumers
~~~~~~~~~~~~~~~~~~~~~~~~~

Implement the ``IntegralConsumer`` concept:

.. code-block:: cpp

   class MyConsumer {
   public:
       void accumulate(const ShellSetQuartet& quartet,
                       std::span<const Real> integrals);
       void finalize();
   };

See Also
--------

- :doc:`contributing` - How to contribute
- :doc:`code_style` - Coding standards
- :doc:`testing` - Testing guide
