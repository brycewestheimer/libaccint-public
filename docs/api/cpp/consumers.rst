.. _api-consumers:

Consumers API
=============

Integral consumers for the compute-and-consume pattern.

FockBuilder
-----------

.. doxygenclass:: libaccint::consumers::FockBuilder
   :members:
   :undoc-members:

GpuFockBuilder
--------------

.. doxygenclass:: libaccint::consumers::GpuFockBuilder
   :members:
   :undoc-members:

IntegralConsumer Concept
------------------------

The ``IntegralConsumer`` concept defines the interface for custom consumers:

.. code-block:: cpp

   template<typename T>
   concept IntegralConsumer = requires(T consumer,
                                       const TwoElectronBuffer<0>& buffer,
                                       Index fa, Index fb, Index fc, Index fd,
                                       int na, int nb, int nc, int nd) {
       { consumer.accumulate(buffer, fa, fb, fc, fd, na, nb, nc, nd) } -> std::same_as<void>;
   };

Implementing Custom Consumers
-----------------------------

To create a custom consumer:

.. code-block:: cpp

   class MyConsumer {
   public:
      void accumulate(const TwoElectronBuffer<0>& buffer,
                  Index fa, Index fb, Index fc, Index fd,
                  int na, int nb, int nc, int nd) {
           // Process integrals as they are computed
       }
   };
