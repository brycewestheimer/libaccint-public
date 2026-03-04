.. _api-memory:

Memory API
==========

Memory management utilities for CPU and GPU memory.

MemoryManager
-------------

.. doxygenclass:: libaccint::memory::MemoryManager
   :members:
   :undoc-members:

DeviceMemory
------------

.. doxygenclass:: libaccint::memory::DeviceMemory
   :members:
   :undoc-members:

Memory Pool
-----------

The memory pool provides efficient allocation for repeated operations:

.. code-block:: cpp

   auto& mm = memory::MemoryManager::instance();

   // Configure GPU memory limit
   mm.set_gpu_memory_limit(8ULL * 1024 * 1024 * 1024);  // 8 GB

   // Check current usage
   std::cout << "GPU memory used: " << mm.gpu_bytes_used() << " bytes\n";
   std::cout << "GPU memory available: " << mm.gpu_bytes_available() << " bytes\n";
