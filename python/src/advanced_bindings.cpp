// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file advanced_bindings.cpp
/// @brief pybind11 bindings for advanced types: ThreadConfig, ShellSetKey,
///        ShellSetPair, ShellSetQuartet, PrimitivePairData, IntegralBuffer,
///        CpuEngine, CudaEngine, GpuFockBuilder, DeviceInfo, StreamHandle,
///        BackendError, and utility functions.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <libaccint/config.hpp>
#include <libaccint/core/backend.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/basis/primitive_pair_data.hpp>
#include <libaccint/engine/thread_config.hpp>
#include <libaccint/engine/integral_buffer.hpp>
#include <libaccint/engine/cpu_engine.hpp>
#include <libaccint/operators/one_electron_operator.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>

#if LIBACCINT_USE_CUDA
#include <libaccint/engine/cuda_engine.hpp>
#include <libaccint/consumers/fock_builder_gpu.hpp>
#endif

namespace py = pybind11;

void bind_advanced(py::module_& m) {
    using namespace libaccint;

    // ========================================================================
    // BackendError exception
    // ========================================================================
    py::register_exception<BackendError>(m, "BackendError", PyExc_RuntimeError);

    // ========================================================================
    // Utility functions
    // ========================================================================
    m.def("version", &version, "Get LibAccInt version string");
    m.def("has_cuda_backend", &has_cuda_backend,
          "Check if CUDA backend was compiled in");
    m.def("has_openmp", &has_openmp,
          "Check if OpenMP support was compiled in");
    m.def("get_device_info", &get_device_info,
          py::arg("backend"), py::arg("device_id") = 0,
          R"pbdoc(
        Get information about a compute device.

        Parameters
        ----------
        backend : BackendType
            Backend to query (CUDA).
        device_id : int, default=0
            Device index.

        Returns
        -------
        DeviceInfo
            Device information struct.
    )pbdoc");
    m.def("get_device_count", &get_device_count,
          py::arg("backend"),
          "Get number of devices for a given backend");

    // ========================================================================
    // ThreadConfig
    // ========================================================================
    py::class_<engine::ThreadConfig>(m, "ThreadConfig", R"pbdoc(
        Thread configuration for OpenMP parallelism.

        Provides static methods to query and control the number of threads
        used for parallel integral computation. All methods are static.

        Examples
        --------
        >>> ThreadConfig.hardware_threads()
        8
        >>> ThreadConfig.set_num_threads(4)
        >>> ThreadConfig.effective_threads()
        4
    )pbdoc")
        .def_static("hardware_threads", &engine::ThreadConfig::hardware_threads,
                     "Get number of hardware threads available")
        .def_static("recommended_threads", &engine::ThreadConfig::recommended_threads,
                     "Get recommended thread count (auto-detected)")
        .def_static("set_num_threads", &engine::ThreadConfig::set_num_threads,
                     py::arg("n"),
                     "Set thread count (0 = auto-detect)")
        .def_static("num_threads", &engine::ThreadConfig::num_threads,
                     "Get configured thread count")
        .def_static("reset", &engine::ThreadConfig::reset,
                     "Reset to auto-detection")
        .def_static("openmp_available", &engine::ThreadConfig::openmp_available,
                     "Check if OpenMP is available at runtime")
        .def_static("effective_threads", &engine::ThreadConfig::effective_threads,
                     "Get effective threads for next parallel region")
        .def_static("resolve", &engine::ThreadConfig::resolve,
                     py::arg("n_threads"),
                     "Resolve thread count (0 = use configured/auto)");

    // ========================================================================
    // ShellSetKey
    // ========================================================================
    py::class_<ShellSetKey>(m, "ShellSetKey", R"pbdoc(
        Key for organizing ShellSets by angular momentum and primitive count.

        Used as a hash map key to look up ShellSets.

        Parameters
        ----------
        angular_momentum : int
            Angular momentum quantum number.
        n_primitives : int
            Number of primitives per shell.
    )pbdoc")
        .def(py::init<>(), "Create default key (am=0, n_prim=0)")
        .def(py::init<int, int>(),
             py::arg("angular_momentum"), py::arg("n_primitives"),
             "Create key with specified angular momentum and primitive count")
        .def_readwrite("angular_momentum", &ShellSetKey::angular_momentum,
                       "Angular momentum quantum number")
        .def_readwrite("n_primitives", &ShellSetKey::n_primitives,
                       "Number of primitives per shell")
        .def("__eq__", &ShellSetKey::operator==)
        .def("__hash__", [](const ShellSetKey& k) {
            return std::hash<ShellSetKey>{}(k);
        })
        .def("__repr__", [](const ShellSetKey& k) {
            return "ShellSetKey(am=" + std::to_string(k.angular_momentum) +
                   ", n_prim=" + std::to_string(k.n_primitives) + ")";
        });

    // ========================================================================
    // ShellSetPair
    // ========================================================================
    py::class_<ShellSetPair>(m, "ShellSetPair", R"pbdoc(
        Pair of ShellSets for one-electron and two-electron integrals.

        Combines two ShellSets and provides lazy-computed Schwarz bounds
        and primitive pair data for efficient integral screening and
        computation.

        Parameters
        ----------
        set_a : ShellSet
            First (bra) ShellSet.
        set_b : ShellSet
            Second (ket) ShellSet.
    )pbdoc")
        .def(py::init<const ShellSet&, const ShellSet&>(),
             py::arg("set_a"), py::arg("set_b"),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
             "Create a ShellSetPair from two ShellSets")
        .def("shell_set_a", &ShellSetPair::shell_set_a,
             py::return_value_policy::reference_internal,
             "Get the first (bra) ShellSet")
        .def("shell_set_b", &ShellSetPair::shell_set_b,
             py::return_value_policy::reference_internal,
             "Get the second (ket) ShellSet")
        .def("La", &ShellSetPair::La, "Angular momentum of ShellSet A")
        .def("Lb", &ShellSetPair::Lb, "Angular momentum of ShellSet B")
        .def("L_total", &ShellSetPair::L_total, "Total angular momentum (La + Lb)")
        .def("n_pairs", &ShellSetPair::n_pairs,
             "Number of shell pairs (n_shells_a * n_shells_b)")
        .def("schwarz_bound", &ShellSetPair::schwarz_bound,
             "Get Schwarz bound Q_ab (lazy-computed)")
        .def("precompute_schwarz_bound", &ShellSetPair::precompute_schwarz_bound,
             "Force eager computation of Schwarz bound")
        .def("schwarz_computed", &ShellSetPair::schwarz_computed,
             "Check if Schwarz bound has been computed")
        .def("pair_data", &ShellSetPair::pair_data,
             py::return_value_policy::reference_internal,
             "Get primitive pair data (lazy-computed)")
        .def("precompute_pair_data", &ShellSetPair::precompute_pair_data,
             "Force eager computation of pair data")
        .def("pair_data_ready", &ShellSetPair::pair_data_ready,
             "Check if pair data has been computed")
        .def("__repr__", [](const ShellSetPair& p) {
            std::string am_a(1, "SPDFGHI"[p.La()]);
            std::string am_b(1, "SPDFGHI"[p.Lb()]);
            return "ShellSetPair(" + am_a + ", " + am_b +
                   ", n_pairs=" + std::to_string(p.n_pairs()) + ")";
        });

    // ========================================================================
    // ShellSetQuartet
    // ========================================================================
    py::class_<ShellSetQuartet>(m, "ShellSetQuartet", R"pbdoc(
        Quartet of ShellSets for two-electron integrals.

        Combines a bra ShellSetPair and a ket ShellSetPair for
        electron repulsion integral computation.

        Parameters
        ----------
        bra : ShellSetPair
            Bra (ab) shell set pair.
        ket : ShellSetPair
            Ket (cd) shell set pair.
    )pbdoc")
        .def(py::init<const ShellSetPair&, const ShellSetPair&>(),
             py::arg("bra"), py::arg("ket"),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
             "Create a ShellSetQuartet from bra and ket pairs")
        .def("bra_pair", &ShellSetQuartet::bra_pair,
             py::return_value_policy::reference_internal,
             "Get the bra (ab) ShellSetPair")
        .def("ket_pair", &ShellSetQuartet::ket_pair,
             py::return_value_policy::reference_internal,
             "Get the ket (cd) ShellSetPair")
        .def("La", &ShellSetQuartet::La, "Angular momentum of center A")
        .def("Lb", &ShellSetQuartet::Lb, "Angular momentum of center B")
        .def("Lc", &ShellSetQuartet::Lc, "Angular momentum of center C")
        .def("Ld", &ShellSetQuartet::Ld, "Angular momentum of center D")
        .def("L_total", &ShellSetQuartet::L_total,
             "Total angular momentum (La + Lb + Lc + Ld)")
        .def("n_quartets", &ShellSetQuartet::n_quartets,
             "Number of shell quartets")
        .def("schwarz_bound", &ShellSetQuartet::schwarz_bound,
             "Get combined Schwarz bound (Q_ab * Q_cd)")
        .def("__repr__", [](const ShellSetQuartet& q) {
            return "ShellSetQuartet(L=" +
                   std::to_string(q.La()) + std::to_string(q.Lb()) +
                   std::to_string(q.Lc()) + std::to_string(q.Ld()) +
                   ", n_quartets=" + std::to_string(q.n_quartets()) + ")";
        });

    // ========================================================================
    // PrimitivePairData
    // ========================================================================
    py::class_<PrimitivePairData>(m, "PrimitivePairData", R"pbdoc(
        Pre-computed Gaussian product data in Structure-of-Arrays layout.

        Contains all the primitive pair quantities needed for integral
        kernels: product centers, combined exponents, overlap prefactors, etc.
    )pbdoc")
        // Read-only NumPy array properties for SoA vectors
        .def_property_readonly("Px", [](const PrimitivePairData& d) {
            return py::array_t<Real>(d.Px.size(), d.Px.data());
        }, "Product center x-coordinates")
        .def_property_readonly("Py", [](const PrimitivePairData& d) {
            return py::array_t<Real>(d.Py.size(), d.Py.data());
        }, "Product center y-coordinates")
        .def_property_readonly("Pz", [](const PrimitivePairData& d) {
            return py::array_t<Real>(d.Pz.size(), d.Pz.data());
        }, "Product center z-coordinates")
        .def_property_readonly("zeta", [](const PrimitivePairData& d) {
            return py::array_t<Real>(d.zeta.size(), d.zeta.data());
        }, "Combined exponents (alpha + beta)")
        .def_property_readonly("one_over_2zeta", [](const PrimitivePairData& d) {
            return py::array_t<Real>(d.one_over_2zeta.size(), d.one_over_2zeta.data());
        }, "Precomputed 1 / (2 * zeta)")
        .def_property_readonly("mu", [](const PrimitivePairData& d) {
            return py::array_t<Real>(d.mu.size(), d.mu.data());
        }, "Reduced exponents (alpha*beta / zeta)")
        .def_property_readonly("K_AB", [](const PrimitivePairData& d) {
            return py::array_t<Real>(d.K_AB.size(), d.K_AB.data());
        }, "Gaussian overlap prefactor exp(-mu*|A-B|^2)")
        .def_property_readonly("coeff_product", [](const PrimitivePairData& d) {
            return py::array_t<Real>(d.coeff_product.size(), d.coeff_product.data());
        }, "Contraction coefficient products (c_a * c_b)")
        .def_property_readonly("PA_x", [](const PrimitivePairData& d) {
            return py::array_t<Real>(d.PA_x.size(), d.PA_x.data());
        }, "P - A displacement x")
        .def_property_readonly("PA_y", [](const PrimitivePairData& d) {
            return py::array_t<Real>(d.PA_y.size(), d.PA_y.data());
        }, "P - A displacement y")
        .def_property_readonly("PA_z", [](const PrimitivePairData& d) {
            return py::array_t<Real>(d.PA_z.size(), d.PA_z.data());
        }, "P - A displacement z")
        .def_property_readonly("PB_x", [](const PrimitivePairData& d) {
            return py::array_t<Real>(d.PB_x.size(), d.PB_x.data());
        }, "P - B displacement x")
        .def_property_readonly("PB_y", [](const PrimitivePairData& d) {
            return py::array_t<Real>(d.PB_y.size(), d.PB_y.data());
        }, "P - B displacement y")
        .def_property_readonly("PB_z", [](const PrimitivePairData& d) {
            return py::array_t<Real>(d.PB_z.size(), d.PB_z.data());
        }, "P - B displacement z")
        // Scalar fields
        .def_readonly("n_shells_a", &PrimitivePairData::n_shells_a,
                      "Number of shells in ShellSet A")
        .def_readonly("n_shells_b", &PrimitivePairData::n_shells_b,
                      "Number of shells in ShellSet B")
        .def_readonly("K_a", &PrimitivePairData::K_a,
                      "Primitives per shell in A")
        .def_readonly("K_b", &PrimitivePairData::K_b,
                      "Primitives per shell in B")
        .def_readonly("n_total_pairs", &PrimitivePairData::n_total_pairs,
                      "Total number of primitive pairs")
        // Methods
        .def("pair_index", &PrimitivePairData::pair_index,
             py::arg("shell_i"), py::arg("shell_j"),
             py::arg("prim_p"), py::arg("prim_q"),
             "Compute flat index for a specific primitive pair")
        .def("shell_pair_offset", &PrimitivePairData::shell_pair_offset,
             py::arg("shell_i"), py::arg("shell_j"),
             "Get offset to a shell pair's primitive data")
        .def("primitives_per_shell_pair", &PrimitivePairData::primitives_per_shell_pair,
             "Get K_a * K_b")
        .def("empty", &PrimitivePairData::empty,
             "Check if pair data is empty")
        .def("__repr__", [](const PrimitivePairData& d) {
            return "PrimitivePairData(n_total_pairs=" +
                   std::to_string(d.n_total_pairs) +
                   ", shells=" + std::to_string(d.n_shells_a) +
                   "x" + std::to_string(d.n_shells_b) +
                   ", K=" + std::to_string(d.K_a) +
                   "x" + std::to_string(d.K_b) + ")";
        });

    // ========================================================================
    // ShellPairMeta
    // ========================================================================
    py::class_<ShellPairMeta>(m, "ShellPairMeta", R"pbdoc(
        Metadata for a shell pair in an IntegralBuffer.
    )pbdoc")
        .def_readonly("offset", &ShellPairMeta::offset, "Offset into data buffer")
        .def_readonly("fi", &ShellPairMeta::fi, "Basis function index for shell A")
        .def_readonly("fj", &ShellPairMeta::fj, "Basis function index for shell B")
        .def_readonly("na", &ShellPairMeta::na, "Number of functions in shell A")
        .def_readonly("nb", &ShellPairMeta::nb, "Number of functions in shell B")
        .def("__repr__", [](const ShellPairMeta& m) {
            return "ShellPairMeta(fi=" + std::to_string(m.fi) +
                   ", fj=" + std::to_string(m.fj) +
                   ", na=" + std::to_string(m.na) +
                   ", nb=" + std::to_string(m.nb) + ")";
        });

    // ========================================================================
    // ShellQuartetMeta
    // ========================================================================
    py::class_<ShellQuartetMeta>(m, "ShellQuartetMeta", R"pbdoc(
        Metadata for a shell quartet in an IntegralBuffer.
    )pbdoc")
        .def_readonly("offset", &ShellQuartetMeta::offset, "Offset into data buffer")
        .def_readonly("fi", &ShellQuartetMeta::fi, "Basis function index for shell A")
        .def_readonly("fj", &ShellQuartetMeta::fj, "Basis function index for shell B")
        .def_readonly("fk", &ShellQuartetMeta::fk, "Basis function index for shell C")
        .def_readonly("fl", &ShellQuartetMeta::fl, "Basis function index for shell D")
        .def_readonly("na", &ShellQuartetMeta::na, "Number of functions in shell A")
        .def_readonly("nb", &ShellQuartetMeta::nb, "Number of functions in shell B")
        .def_readonly("nc", &ShellQuartetMeta::nc, "Number of functions in shell C")
        .def_readonly("nd", &ShellQuartetMeta::nd, "Number of functions in shell D")
        .def("__repr__", [](const ShellQuartetMeta& m) {
            return "ShellQuartetMeta(fi=" + std::to_string(m.fi) +
                   ", fj=" + std::to_string(m.fj) +
                   ", fk=" + std::to_string(m.fk) +
                   ", fl=" + std::to_string(m.fl) + ")";
        });

    // ========================================================================
    // IntegralBuffer
    // ========================================================================
    py::class_<IntegralBuffer>(m, "IntegralBuffer", R"pbdoc(
        Buffer holding computed integrals with metadata.

        Stores integral values along with shell pair/quartet metadata for
        efficient access. Supports both one-electron (pair) and two-electron
        (quartet) integrals.
    )pbdoc")
        .def("n_shell_quartets", &IntegralBuffer::n_shell_quartets,
             "Number of shell quartets in buffer")
        .def("n_shell_pairs", &IntegralBuffer::n_shell_pairs,
             "Number of shell pairs in buffer")
        .def("n_integrals", &IntegralBuffer::n_integrals,
             "Total number of integral values")
        .def("empty", &IntegralBuffer::empty,
             "Check if buffer is empty")
        .def("La", &IntegralBuffer::La, "Angular momentum of center A")
        .def("Lb", &IntegralBuffer::Lb, "Angular momentum of center B")
        .def("Lc", &IntegralBuffer::Lc, "Angular momentum of center C")
        .def("Ld", &IntegralBuffer::Ld, "Angular momentum of center D")
        .def("deriv_order", &IntegralBuffer::deriv_order, "Derivative order")
        .def("to_numpy", [](const IntegralBuffer& buf) {
            auto data = buf.data();
            return py::array_t<Real>(data.size(), data.data());
        }, "Copy integral data to a NumPy array")
        .def("quartet_meta", &IntegralBuffer::quartet_meta,
             py::arg("idx"),
             "Get metadata for quartet at index")
        .def("pair_meta", &IntegralBuffer::pair_meta,
             py::arg("idx"),
             "Get metadata for pair at index")
        .def("__repr__", [](const IntegralBuffer& buf) {
            return "IntegralBuffer(n_integrals=" +
                   std::to_string(buf.n_integrals()) +
                   ", n_pairs=" + std::to_string(buf.n_shell_pairs()) +
                   ", n_quartets=" + std::to_string(buf.n_shell_quartets()) + ")";
        });

    // ========================================================================
    // DeviceInfo
    // ========================================================================
    py::class_<DeviceInfo>(m, "DeviceInfo", R"pbdoc(
        Information about a compute device (GPU).
    )pbdoc")
        .def_readonly("name", &DeviceInfo::name, "Device name")
        .def_readonly("total_memory", &DeviceInfo::total_memory,
                      "Total device memory in bytes")
        .def_readonly("available_memory", &DeviceInfo::available_memory,
                      "Available device memory in bytes")
        .def_readonly("compute_capability", &DeviceInfo::compute_capability,
                      "SM version (CUDA)")
        .def_readonly("multiprocessor_count", &DeviceInfo::multiprocessor_count,
                      "Number of multiprocessors/compute units")
        .def_readonly("max_threads_per_block", &DeviceInfo::max_threads_per_block,
                      "Maximum threads per block")
        .def_readonly("warp_size", &DeviceInfo::warp_size,
                      "Warp/wavefront size")
        .def("__repr__", [](const DeviceInfo& d) {
            return "DeviceInfo(name='" + d.name +
                   "', memory=" + std::to_string(d.total_memory / (1024*1024)) +
                   " MB, SMs=" + std::to_string(d.multiprocessor_count) + ")";
        });

    // ========================================================================
    // StreamHandle
    // ========================================================================
    py::class_<StreamHandle>(m, "StreamHandle", R"pbdoc(
        Handle to an asynchronous execution stream.

        Wraps a CUDA stream for asynchronous kernel execution.
    )pbdoc")
        .def(py::init<>(), "Create a default (synchronous) stream handle")
        .def_static("create", &StreamHandle::create,
                    py::arg("backend"),
                    "Create an asynchronous stream for the given backend")
        .def("synchronize", &StreamHandle::synchronize,
             "Wait for all operations on this stream to complete")
        .def("valid", &StreamHandle::valid,
             "Check if this stream handle is valid")
        .def("backend", &StreamHandle::backend,
             "Get the backend type for this stream")
        .def("__repr__", [](const StreamHandle& s) {
            return "StreamHandle(valid=" + std::string(s.valid() ? "True" : "False") +
                   ", backend=" + std::string(backend_name(s.backend())) + ")";
        });

    // ========================================================================
    // CpuEngine
    // ========================================================================
    py::class_<engine::CpuEngine>(m, "CpuEngine", R"pbdoc(
        CPU backend engine for molecular integral computation.

        Provides direct access to CPU-based integral computation using
        optimized kernels with optional OpenMP parallelism.

        Parameters
        ----------
        basis : BasisSet
            The basis set to use for computations.
    )pbdoc")
        .def(py::init<const BasisSet&>(),
             py::arg("basis"),
             py::keep_alive<1, 2>(),
             "Create a CPU engine with the given basis set")
        .def("basis", &engine::CpuEngine::basis,
             py::return_value_policy::reference_internal,
             "Get the basis set")
        .def("max_angular_momentum", &engine::CpuEngine::max_angular_momentum,
             "Get maximum angular momentum from basis")
        .def("compute_shell_set_pair",
             [](engine::CpuEngine& eng, const Operator& op,
                const ShellSetPair& pair) {
                 Size nbf = eng.basis().n_basis_functions();
                 std::vector<Real> result(nbf * nbf, 0.0);
                 eng.compute_shell_set_pair(op, pair, result);
                 return py::array_t<Real>({nbf, nbf}, result.data());
             },
             py::arg("op"), py::arg("pair"),
             R"pbdoc(
        Compute 1e integrals for a ShellSetPair.

        Parameters
        ----------
        op : Operator
            One-electron operator.
        pair : ShellSetPair
            Shell set pair to compute.

        Returns
        -------
        numpy.ndarray
            Result matrix of shape (n_basis, n_basis).
    )pbdoc")
        .def("compute_and_consume",
             [](engine::CpuEngine& eng, const Operator& op,
                consumers::FockBuilder& consumer) {
                 eng.compute_and_consume(op, consumer);
             },
             py::arg("op"), py::arg("consumer"),
             "Compute and consume 2e integrals with FockBuilder")
        .def("compute_and_consume_parallel",
             [](engine::CpuEngine& eng, const Operator& op,
                consumers::FockBuilder& consumer, int n_threads) {
                 eng.compute_and_consume_parallel(op, consumer, n_threads);
             },
             py::arg("op"), py::arg("consumer"), py::arg("n_threads") = 0,
             "Parallel compute and consume 2e integrals with FockBuilder")
        .def("compute_1e",
             [](engine::CpuEngine& eng, const OneElectronOperator& op) {
                 Size nbf = eng.basis().n_basis_functions();
                 std::vector<Real> result(nbf * nbf, 0.0);
                 eng.compute_1e<0>(op, result);
                 return py::array_t<Real>({nbf, nbf}, result.data());
             },
             py::arg("op"),
             R"pbdoc(
        Compute one-electron integrals for all shell pairs.

        Parameters
        ----------
        op : OneElectronOperator
            Composed one-electron operator.

        Returns
        -------
        numpy.ndarray
            Result matrix of shape (n_basis, n_basis).
    )pbdoc")
        .def("compute_1e_parallel",
             [](engine::CpuEngine& eng, const OneElectronOperator& op,
                int n_threads) {
                 Size nbf = eng.basis().n_basis_functions();
                 std::vector<Real> result(nbf * nbf, 0.0);
                 eng.compute_1e_parallel<0>(op, result, n_threads);
                 return py::array_t<Real>({nbf, nbf}, result.data());
             },
             py::arg("op"), py::arg("n_threads") = 0,
             R"pbdoc(
        Parallel one-electron integral computation.

        Parameters
        ----------
        op : OneElectronOperator
            Composed one-electron operator.
        n_threads : int, default=0
            Number of threads (0 = auto-detect).

        Returns
        -------
        numpy.ndarray
            Result matrix of shape (n_basis, n_basis).
    )pbdoc")
        .def("__repr__", [](const engine::CpuEngine& e) {
            return "CpuEngine(n_basis=" +
                   std::to_string(e.basis().n_basis_functions()) +
                   ", max_am=" + std::to_string(e.max_angular_momentum()) + ")";
        });

    // ========================================================================
    // CudaEngine (conditional)
    // ========================================================================
#if LIBACCINT_USE_CUDA
    py::class_<CudaEngine>(m, "CudaEngine", R"pbdoc(
        GPU-accelerated engine for molecular integral computation.

        Uses CUDA to compute integrals on NVIDIA GPUs. Supports fused
        one-electron kernel (S+T+V in a single pass) and batched ERI
        computation.

        Parameters
        ----------
        basis : BasisSet
            The basis set to use for computations.
    )pbdoc")
        .def(py::init<const BasisSet&>(),
             py::arg("basis"),
             py::keep_alive<1, 2>(),
             "Create a CUDA engine with the given basis set")
        .def("basis", &CudaEngine::basis,
             py::return_value_policy::reference_internal,
             "Get the basis set")
        .def("max_angular_momentum", &CudaEngine::max_angular_momentum,
             "Get maximum angular momentum from basis")
        .def("is_initialized", &CudaEngine::is_initialized,
             "Check if GPU resources are initialized")
        .def("compute_overlap_matrix", [](CudaEngine& eng) {
            Size nbf = eng.basis().n_basis_functions();
            std::vector<Real> result(nbf * nbf, 0.0);
            eng.compute_overlap_matrix(result);
            return py::array_t<Real>({nbf, nbf}, result.data());
        }, "Compute overlap matrix S on GPU")
        .def("compute_kinetic_matrix", [](CudaEngine& eng) {
            Size nbf = eng.basis().n_basis_functions();
            std::vector<Real> result(nbf * nbf, 0.0);
            eng.compute_kinetic_matrix(result);
            return py::array_t<Real>({nbf, nbf}, result.data());
        }, "Compute kinetic energy matrix T on GPU")
        .def("compute_nuclear_matrix",
             [](CudaEngine& eng, const std::vector<data::Atom>& atoms) {
                 PointChargeParams charges;
                 for (const auto& atom : atoms) {
                     charges.x.push_back(atom.position.x);
                     charges.y.push_back(atom.position.y);
                     charges.z.push_back(atom.position.z);
                     charges.charge.push_back(
                         static_cast<Real>(atom.atomic_number));
                 }
                 Size nbf = eng.basis().n_basis_functions();
                 std::vector<Real> result(nbf * nbf, 0.0);
                 eng.compute_nuclear_matrix(charges, result);
                 return py::array_t<Real>({nbf, nbf}, result.data());
             },
             py::arg("atoms"),
             "Compute nuclear attraction matrix V on GPU")
        .def("compute_core_hamiltonian",
             [](CudaEngine& eng, const std::vector<data::Atom>& atoms) {
                 PointChargeParams charges;
                 for (const auto& atom : atoms) {
                     charges.x.push_back(atom.position.x);
                     charges.y.push_back(atom.position.y);
                     charges.z.push_back(atom.position.z);
                     charges.charge.push_back(
                         static_cast<Real>(atom.atomic_number));
                 }
                 Size nbf = eng.basis().n_basis_functions();
                 std::vector<Real> result(nbf * nbf, 0.0);
                 eng.compute_core_hamiltonian(charges, result);
                 return py::array_t<Real>({nbf, nbf}, result.data());
             },
             py::arg("atoms"),
             "Compute core Hamiltonian H = T + V on GPU")
        .def("compute_all_1e_fused",
             [](CudaEngine& eng, const std::vector<data::Atom>& atoms) {
                 PointChargeParams charges;
                 for (const auto& atom : atoms) {
                     charges.x.push_back(atom.position.x);
                     charges.y.push_back(atom.position.y);
                     charges.z.push_back(atom.position.z);
                     charges.charge.push_back(
                         static_cast<Real>(atom.atomic_number));
                 }
                 Size nbf = eng.basis().n_basis_functions();
                 std::vector<Real> S(nbf * nbf, 0.0);
                 std::vector<Real> T(nbf * nbf, 0.0);
                 std::vector<Real> V(nbf * nbf, 0.0);
                 eng.compute_all_1e_fused(charges, S, T, V);
                 return py::make_tuple(
                     py::array_t<Real>({nbf, nbf}, S.data()),
                     py::array_t<Real>({nbf, nbf}, T.data()),
                     py::array_t<Real>({nbf, nbf}, V.data()));
             },
             py::arg("atoms"),
             R"pbdoc(
        Compute all one-electron matrices in a single fused GPU kernel.

        This is the most efficient way to compute S, T, and V simultaneously,
        as it uploads shell data once and runs a fused kernel.

        Parameters
        ----------
        atoms : list of Atom
            Atoms defining nuclear charges and positions.

        Returns
        -------
        tuple of numpy.ndarray
            (S, T, V) overlap, kinetic, and nuclear attraction matrices.
    )pbdoc")
        .def("synchronize", &CudaEngine::synchronize,
             "Synchronize all GPU operations")
        .def("__repr__", [](const CudaEngine& e) {
            return "CudaEngine(n_basis=" +
                   std::to_string(e.basis().n_basis_functions()) +
                   ", initialized=" +
                   (e.is_initialized() ? "True" : "False") + ")";
        });
#endif  // LIBACCINT_USE_CUDA

    // ========================================================================
    // GpuFockBuilder (conditional)
    // ========================================================================
#if LIBACCINT_USE_CUDA
    py::class_<consumers::GpuFockBuilder>(m, "GpuFockBuilder", R"pbdoc(
        GPU-accelerated Fock matrix builder.

        Builds Coulomb (J) and exchange (K) matrices on the GPU using
        device-side accumulation with atomicAdd for the two-electron
        integral contributions.

        Parameters
        ----------
        nbf : int
            Number of basis functions.
    )pbdoc")
        .def(py::init<Size>(),
             py::arg("nbf"),
             "Create GPU Fock builder with default stream")
        .def("set_density",
             [](consumers::GpuFockBuilder& fb, py::array_t<Real> D) {
                 auto buf = D.request();
                 Size nbf = fb.nbf();
                 if (buf.size != static_cast<ssize_t>(nbf * nbf)) {
                     throw std::invalid_argument(
                         "Density matrix must have nbf*nbf elements");
                 }
                 fb.set_density(static_cast<const Real*>(buf.ptr), nbf);
             },
             py::arg("D"),
             "Upload density matrix to GPU")
        .def("get_coulomb_matrix",
             [](const consumers::GpuFockBuilder& fb) {
                 auto J = fb.get_coulomb_matrix();
                 Size nbf = fb.nbf();
                 return py::array_t<Real>({nbf, nbf}, J.data());
             },
             "Download Coulomb matrix J from GPU")
        .def("get_exchange_matrix",
             [](const consumers::GpuFockBuilder& fb) {
                 auto K = fb.get_exchange_matrix();
                 Size nbf = fb.nbf();
                 return py::array_t<Real>({nbf, nbf}, K.data());
             },
             "Download exchange matrix K from GPU")
        .def("get_fock_matrix",
             [](const consumers::GpuFockBuilder& fb,
                py::array_t<Real> H_core, Real exchange_fraction) {
                 auto buf = H_core.request();
                 Size nbf = fb.nbf();
                 auto F = fb.get_fock_matrix(
                     std::span<const Real>(
                         static_cast<const Real*>(buf.ptr),
                         static_cast<size_t>(buf.size)),
                     exchange_fraction);
                 return py::array_t<Real>({nbf, nbf}, F.data());
             },
             py::arg("H_core"), py::arg("exchange_fraction") = 1.0,
             "Compute Fock matrix F = H_core + J - fraction*K on GPU")
        .def("reset", &consumers::GpuFockBuilder::reset,
             "Zero J and K matrices on device")
        .def("synchronize", &consumers::GpuFockBuilder::synchronize,
             "Synchronize GPU operations")
        .def("nbf", &consumers::GpuFockBuilder::nbf,
             "Get number of basis functions")
        .def("__repr__", [](const consumers::GpuFockBuilder& fb) {
            return "GpuFockBuilder(nbf=" + std::to_string(fb.nbf()) + ")";
        });
#endif  // LIBACCINT_USE_CUDA
}
