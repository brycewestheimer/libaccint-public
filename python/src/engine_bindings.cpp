// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file engine_bindings.cpp
/// @brief pybind11 bindings for the Engine class

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <span>

#include <libaccint/engine/engine.hpp>
#include <libaccint/engine/dispatch_policy.hpp>
#include <libaccint/engine/integral_buffer.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/screening/screening_options.hpp>
#include <libaccint/data/builtin_basis.hpp>

#if LIBACCINT_USE_CUDA
#include <libaccint/engine/multi_gpu_engine.hpp>
#include <libaccint/consumers/multi_gpu_fock_builder.hpp>
#endif

namespace py = pybind11;

void bind_engine(py::module_& m) {
    using namespace libaccint;

    // ========================================================================
    // DispatchConfig struct
    // ========================================================================
    py::class_<DispatchConfig>(m, "DispatchConfig", R"pbdoc(
        Configuration for backend dispatch heuristics.

        Controls when the engine switches between CPU and GPU backends
        based on problem size.

        Attributes
        ----------
        min_gpu_batch_size : int
            Minimum batch size to consider GPU dispatch (default: 16).
        min_gpu_primitives : int
            Minimum total primitives to consider GPU dispatch (default: 1000).
        high_am_threshold : int
            Angular momentum threshold for GPU preference (default: 4).
        min_gpu_shells : int
            Minimum shell count for full-basis GPU dispatch (default: 10).
        enable_auto_tuning : bool
            Enable auto-tuning via KernelCalculator (default: False).
        auto_tune_min_batch : int
            Minimum batch size before auto-tuning engages (default: 100).
        n_gpu_slots : int
            Number of concurrent GPU execution slots. Each slot owns an
            independent CUDA stream and device buffers, enabling parallel
            GPU kernel execution from multiple host threads (default: 4).
    )pbdoc")
        .def(py::init<>(), "Create default config")
        .def_readwrite("min_gpu_batch_size",
                       &DispatchConfig::min_gpu_batch_size,
                       "Minimum batch size to consider GPU dispatch")
        .def_readwrite("min_gpu_primitives",
                       &DispatchConfig::min_gpu_primitives,
                       "Minimum primitives before considering GPU")
        .def_readwrite("high_am_threshold",
                       &DispatchConfig::high_am_threshold,
                       "Angular momentum threshold for GPU preference")
        .def_readwrite("min_gpu_shells",
                       &DispatchConfig::min_gpu_shells,
                       "Minimum shells before considering GPU")
        .def_readwrite("enable_auto_tuning",
                       &DispatchConfig::enable_auto_tuning,
                       "Enable auto-tuning via KernelCalculator")
        .def_readwrite("auto_tune_min_batch",
                       &DispatchConfig::auto_tune_min_batch,
                       "Minimum batch size before auto-tuning engages")
        .def_readwrite("n_gpu_slots",
                       &DispatchConfig::n_gpu_slots,
                       "Number of concurrent GPU execution slots (streams + buffers)")
        .def("__repr__", [](const DispatchConfig& c) {
            return "DispatchConfig(min_gpu_batch_size=" +
                   std::to_string(c.min_gpu_batch_size) +
                   ", min_gpu_shells=" +
                   std::to_string(c.min_gpu_shells) +
                   ", min_gpu_primitives=" +
                   std::to_string(c.min_gpu_primitives) +
                   ", high_am_threshold=" +
                   std::to_string(c.high_am_threshold) +
                   ", enable_auto_tuning=" +
                   (c.enable_auto_tuning ? "True" : "False") +
                   ", n_gpu_slots=" +
                   std::to_string(c.n_gpu_slots) + ")";
        });

    // ========================================================================
    // Engine class
    // ========================================================================
    py::class_<Engine>(m, "Engine", R"pbdoc(
        Unified computation orchestrator for molecular integrals.

        Engine provides a single API that automatically routes work to the
        optimal backend (CPU or GPU) based on dispatch heuristics.

        Parameters
        ----------
        basis : BasisSet
            The basis set to use for computations.
        config : DispatchConfig, optional
            Configuration for backend selection heuristics.

        Examples
        --------
        >>> basis = create_builtin_basis("sto-3g", atoms)
        >>> engine = Engine(basis)
        >>> S = engine.compute_overlap_matrix()
        >>> T = engine.compute_kinetic_matrix()
    )pbdoc")
        .def(py::init<const BasisSet&>(),
             py::arg("basis"),
             py::keep_alive<1, 2>(),
             "Create engine with default dispatch config")
        .def(py::init<const BasisSet&, DispatchConfig>(),
             py::arg("basis"), py::arg("config"),
             py::keep_alive<1, 2>(),
             "Create engine with custom dispatch config")

        // Basic accessors
        .def("basis", [](const Engine& e) -> const BasisSet* {
            return &e.basis();
        }, py::return_value_policy::reference_internal,
             "Get the basis set")
        .def("max_angular_momentum", &Engine::max_angular_momentum,
             "Get maximum angular momentum from basis")
        .def("gpu_available", &Engine::gpu_available,
             "Check if GPU backend is available")

        // Dispatch configuration
        .def("set_dispatch_config", &Engine::set_dispatch_config,
             py::arg("config"),
             "Set dispatch configuration")

        // ====================================================================
        // Full Matrix Computation Methods (return NumPy arrays)
        // ====================================================================
        .def("compute_overlap_matrix", [](Engine& engine, BackendHint hint) {
            std::vector<Real> result;
            engine.compute_overlap_matrix(result, hint);
            Size n = engine.basis().n_basis_functions();
            return py::array_t<Real>({n, n}, result.data());
        }, py::arg("hint") = BackendHint::Auto,
           R"pbdoc(
        Compute the full overlap matrix S.

        Parameters
        ----------
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        numpy.ndarray
            Overlap matrix of shape (n_basis, n_basis).
    )pbdoc")

        .def("compute_kinetic_matrix", [](Engine& engine, BackendHint hint) {
            std::vector<Real> result;
            engine.compute_kinetic_matrix(result, hint);
            Size n = engine.basis().n_basis_functions();
            return py::array_t<Real>({n, n}, result.data());
        }, py::arg("hint") = BackendHint::Auto,
           R"pbdoc(
        Compute the full kinetic energy matrix T.

        Parameters
        ----------
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        numpy.ndarray
            Kinetic energy matrix of shape (n_basis, n_basis).
    )pbdoc")

        .def("compute_nuclear_matrix",
             [](Engine& engine, const std::vector<data::Atom>& atoms, BackendHint hint) {
                 // Convert atoms to PointChargeParams
                 PointChargeParams charges;
                 for (const auto& atom : atoms) {
                     charges.x.push_back(atom.position.x);
                     charges.y.push_back(atom.position.y);
                     charges.z.push_back(atom.position.z);
                     charges.charge.push_back(static_cast<Real>(atom.atomic_number));
                 }
                 std::vector<Real> result;
                 engine.compute_nuclear_matrix(charges, result, hint);
                 Size n = engine.basis().n_basis_functions();
                 return py::array_t<Real>({n, n}, result.data());
             },
             py::arg("atoms"), py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute the full nuclear attraction matrix V.

        Parameters
        ----------
        atoms : list of Atom
            List of atoms defining nuclear charges and positions.
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        numpy.ndarray
            Nuclear attraction matrix of shape (n_basis, n_basis).
    )pbdoc")

        .def("compute_core_hamiltonian",
             [](Engine& engine, const std::vector<data::Atom>& atoms, BackendHint hint) {
                 // Convert atoms to PointChargeParams
                 PointChargeParams charges;
                 for (const auto& atom : atoms) {
                     charges.x.push_back(atom.position.x);
                     charges.y.push_back(atom.position.y);
                     charges.z.push_back(atom.position.z);
                     charges.charge.push_back(static_cast<Real>(atom.atomic_number));
                 }
                 std::vector<Real> result;
                 engine.compute_core_hamiltonian(charges, result, hint);
                 Size n = engine.basis().n_basis_functions();
                 return py::array_t<Real>({n, n}, result.data());
             },
             py::arg("atoms"), py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute the core Hamiltonian H = T + V.

        Parameters
        ----------
        atoms : list of Atom
            List of atoms defining nuclear charges and positions.
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        numpy.ndarray
            Core Hamiltonian matrix of shape (n_basis, n_basis).
    )pbdoc")

        // ====================================================================
        // Consumer-based computation (named aliases)
        // ====================================================================
        .def("compute_and_consume",
             [](Engine& engine, const Operator& op,
                consumers::FockBuilder& consumer, BackendHint hint) {
                 engine.compute(op, consumer, hint);
             },
             py::arg("op"), py::arg("consumer"), py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute two-electron integrals and consume with FockBuilder.

        Alias for ``compute(op, consumer, hint)``.

        This method computes all electron repulsion integrals and
        accumulates Coulomb (J) and exchange (K) contributions into
        the FockBuilder.

        Parameters
        ----------
        op : Operator
            Two-electron operator (e.g., Operator.coulomb()).
        consumer : FockBuilder
            FockBuilder to accumulate J and K matrices.
        hint : BackendHint, default=Auto
            Backend selection hint.
    )pbdoc")

        .def("compute_and_consume_parallel",
             [](Engine& engine, const Operator& op,
                consumers::FockBuilder& consumer, int n_threads, BackendHint hint) {
                 engine.compute_and_consume_parallel(op, consumer, n_threads, hint);
             },
             py::arg("op"), py::arg("consumer"),
             py::arg("n_threads") = 0, py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Parallel compute and consume two-electron integrals.

        Multi-threaded version using OpenMP.

        Parameters
        ----------
        op : Operator
            Two-electron operator.
        consumer : FockBuilder
            Thread-safe FockBuilder.
        n_threads : int, default=0
            Number of threads (0 = auto-detect).
        hint : BackendHint, default=Auto
            Backend selection hint.
    )pbdoc")

        // ====================================================================
        // Schwarz screening
        // ====================================================================
        .def("precompute_schwarz_bounds", [](Engine& engine) {
            engine.precompute_schwarz_bounds();
        }, "Precompute Schwarz bounds for all shell pairs")

        .def("schwarz_bounds_precomputed", &Engine::schwarz_bounds_precomputed,
             "Check if Schwarz bounds have been precomputed")

        .def("set_density_matrix", [](Engine& engine, py::array D, Size nbf) {
            if (!py::isinstance<py::array_t<Real>>(D)) {
                throw std::invalid_argument("Density matrix must have dtype float64");
            }
            auto buf = D.request();
            if (buf.ndim != 2) {
                throw std::invalid_argument("Density matrix must be 2D array");
            }
            if (buf.shape[0] != static_cast<py::ssize_t>(nbf) ||
                buf.shape[1] != static_cast<py::ssize_t>(nbf)) {
                throw std::invalid_argument(
                    "Density matrix shape must be (" + std::to_string(nbf) +
                    ", " + std::to_string(nbf) + ")");
            }
            const auto item_size = static_cast<py::ssize_t>(sizeof(Real));
            const bool c_contiguous =
                (buf.strides[1] == item_size) &&
                (buf.strides[0] == static_cast<py::ssize_t>(nbf) * item_size);
            if (!c_contiguous) {
                throw std::invalid_argument(
                    "Density matrix must be C-contiguous (row-major)");
            }
            engine.set_density_matrix(static_cast<const Real*>(buf.ptr), nbf);
        }, py::arg("D"), py::arg("nbf"),
            R"pbdoc(
            Set the density matrix for density-weighted screening.

            Parameters
            ----------
            D : numpy.ndarray
                Row-major density matrix (nbf x nbf).
            nbf : int
                Number of basis functions.
        )pbdoc")

        .def("density_matrix_set", &Engine::density_matrix_set,
             "Check if density matrix has been set for density-weighted screening")

        .def("compute_and_consume_screened",
             [](Engine& engine, const Operator& op,
                consumers::FockBuilder& consumer,
                const screening::ScreeningOptions& options,
                BackendHint hint) {
                 engine.compute_and_consume(op, consumer, options, hint);
             },
             py::arg("op"), py::arg("consumer"),
             py::arg("options"), py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute and consume with Schwarz screening.

        Uses pre-computed Schwarz bounds to skip negligible shell quartets.
        Schwarz bounds will be precomputed automatically if not already done.

        Parameters
        ----------
        op : Operator
            Two-electron operator.
        consumer : FockBuilder
            FockBuilder to accumulate J and K matrices.
        options : ScreeningOptions
            Screening parameters (threshold, preset, etc.).
        hint : BackendHint, default=Auto
            Backend selection hint.
    )pbdoc")

        .def("compute_and_consume_screened_parallel",
             [](Engine& engine, const Operator& op,
                consumers::FockBuilder& consumer,
                const screening::ScreeningOptions& options,
                int n_threads, BackendHint hint) {
                 engine.compute_and_consume_screened_parallel(
                     op, consumer, options, n_threads);
                 (void)hint;
             },
             py::arg("op"), py::arg("consumer"),
             py::arg("options"),
             py::arg("n_threads") = 0, py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Parallel screened compute and consume.

        Combines Schwarz screening with OpenMP parallelism.

        Parameters
        ----------
        op : Operator
            Two-electron operator.
        consumer : FockBuilder
            Thread-safe FockBuilder.
        options : ScreeningOptions
            Screening parameters.
        n_threads : int, default=0
            Number of threads (0 = auto-detect).
        hint : BackendHint, default=Auto
            Backend selection hint.
    )pbdoc")

        // ====================================================================
        // Direct Backend Access
        // ====================================================================
        .def("cpu_engine", [](Engine& e) -> engine::CpuEngine& {
            return e.cpu_engine();
        }, py::return_value_policy::reference_internal,
           "Get the CPU engine (always available)")

#if LIBACCINT_USE_CUDA
        .def("cuda_engine", [](Engine& e) -> CudaEngine* {
            return e.cuda_engine();
        }, py::return_value_policy::reference_internal,
           "Get the CUDA engine (None if GPU not available)")
#endif

        // ====================================================================
        // Parallel 1e computation
        // ====================================================================
        .def("compute_1e_parallel",
             [](Engine& engine, const OneElectronOperator& op, int n_threads) {
                 std::vector<Real> result;
                 engine.compute_1e_parallel<0>(op, result, n_threads);
                 Size n = engine.basis().n_basis_functions();
                 return py::array_t<Real>({n, n}, result.data());
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

        // ====================================================================
        // Batch compute API (non-consuming)
        // ====================================================================
        .def("compute_batch",
             [](Engine& engine, const Operator& op,
                const ShellSetQuartet& quartet, BackendHint hint) {
                 return engine.compute_batch(op, quartet, hint);
             },
             py::arg("op"), py::arg("quartet"),
             py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute two-electron integrals for a ShellSetQuartet.

        Returns an IntegralBuffer with computed integrals and metadata.

        Parameters
        ----------
        op : Operator
            Two-electron operator.
        quartet : ShellSetQuartet
            ShellSetQuartet to compute.
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        IntegralBuffer
            Buffer with computed integrals.
    )pbdoc")

        .def("compute_all_1e",
             [](Engine& engine, const Operator& op, BackendHint hint) {
                 return engine.compute_all_1e(op, hint);
             },
             py::arg("op"), py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute all one-electron integrals for the full basis set.

        Returns a list of IntegralBuffers, one per ShellSetPair.

        Parameters
        ----------
        op : Operator
            One-electron operator.
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        list of IntegralBuffer
            Integral buffers for all shell set pairs.
    )pbdoc")

        .def("compute_all_2e",
             [](Engine& engine, const Operator& op, BackendHint hint) {
                 return engine.compute_all_2e(op, hint);
             },
             py::arg("op"), py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute all two-electron integrals for the full basis set.

        Returns a list of IntegralBuffers, one per ShellSetQuartet.

        Parameters
        ----------
        op : Operator
            Two-electron operator.
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        list of IntegralBuffer
            Integral buffers for all shell set quartets.
    )pbdoc")

        // ====================================================================
        // Unified compute() API — mirrors C++ Engine::compute() overloads
        // ====================================================================

        // compute(op, ShellSetPair, hint) → IntegralBuffer
        .def("compute",
             [](Engine& engine, const Operator& op,
                const ShellSetPair& pair,
                BackendHint hint) {
                 return engine.compute(op, pair, hint);
             },
             py::arg("op"), py::arg("work_unit"),
             py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute one-electron integrals for a ShellSetPair.

        The engine routes work to the best available backend (CPU or GPU)
        based on dispatch heuristics and the hint.

        Parameters
        ----------
        op : Operator
            One-electron operator (e.g., Operator.overlap()).
        work_unit : ShellSetPair
            Shell pair work unit.
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        IntegralBuffer
            Buffer with computed integrals and scatter metadata.
    )pbdoc")

        // compute(op, ShellSetQuartet, hint) → IntegralBuffer
        .def("compute",
             [](Engine& engine, const Operator& op,
                const ShellSetQuartet& quartet,
                BackendHint hint) {
                 return engine.compute(op, quartet, hint);
             },
             py::arg("op"), py::arg("work_unit"),
             py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute two-electron integrals for a ShellSetQuartet.

        The engine routes work to the best available backend (CPU or GPU)
        based on dispatch heuristics and the hint.

        Parameters
        ----------
        op : Operator
            Two-electron operator (e.g., Operator.coulomb()).
        work_unit : ShellSetQuartet
            Shell quartet work unit.
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        IntegralBuffer
            Buffer with computed integrals and scatter metadata.
    )pbdoc")

        // compute(op, ShellSetQuartet, FockBuilder, hint) → None
        .def("compute",
             [](Engine& engine, const Operator& op,
                const ShellSetQuartet& quartet,
                consumers::FockBuilder& consumer,
                BackendHint hint) {
                 engine.compute(op, quartet, consumer, hint);
             },
             py::arg("op"), py::arg("work_unit"), py::arg("consumer"),
             py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute two-electron integrals for a ShellSetQuartet and accumulate
        into a FockBuilder consumer.

        Parameters
        ----------
        op : Operator
            Two-electron operator.
        work_unit : ShellSetQuartet
            Shell quartet work unit.
        consumer : FockBuilder
            Consumer that accumulates J and K contributions.
        hint : BackendHint, default=Auto
            Backend selection hint.
    )pbdoc")

        // compute(op, FockBuilder, hint) → None  (full basis)
        .def("compute",
             [](Engine& engine, const Operator& op,
                consumers::FockBuilder& consumer,
                BackendHint hint) {
                 engine.compute(op, consumer, hint);
             },
             py::arg("op"), py::arg("consumer"),
             py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute all two-electron integrals for the full basis and accumulate
        into a FockBuilder consumer.

        This is the most common entry point for Fock matrix construction.
        Equivalent to iterating over all ShellSetQuartets and consuming.

        Parameters
        ----------
        op : Operator
            Two-electron operator (e.g., Operator.coulomb()).
        consumer : FockBuilder
            Consumer that accumulates J and K contributions.
        hint : BackendHint, default=Auto
            Backend selection hint.
    )pbdoc")

        // ====================================================================
        // Named aliases (for backward compatibility)
        // ====================================================================
        .def("compute_quartet",
             [](Engine& engine, const Operator& op,
                const ShellSetQuartet& quartet,
                BackendHint hint) {
                 return engine.compute(op, quartet, hint);
             },
             py::arg("op"), py::arg("quartet"),
             py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute two-electron integrals for a ShellSetQuartet (non-consumer).

        Alias for ``compute(op, quartet, hint)``.

        Parameters
        ----------
        op : Operator
            Two-electron operator.
        quartet : ShellSetQuartet
            Shell quartet to compute.
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        IntegralBuffer
            Buffer with computed integrals.
    )pbdoc")

        .def("compute_pair",
             [](Engine& engine, const Operator& op,
                const ShellSetPair& pair,
                BackendHint hint) {
                 return engine.compute(op, pair, hint);
             },
             py::arg("op"), py::arg("pair"),
             py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute one-electron integrals for a ShellSetPair (non-consumer).

        Alias for ``compute(op, pair, hint)``.

        Parameters
        ----------
        op : Operator
            One-electron operator.
        pair : ShellSetPair
            Shell pair to compute.
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        IntegralBuffer
            Buffer with computed integrals.
    )pbdoc")

        .def("compute_pair_matrix",
             [](Engine& engine, const Operator& op,
                const ShellSetPair& pair,
                BackendHint hint) {
                 Size nbf = engine.basis().n_basis_functions();
                 std::vector<Real> result(nbf * nbf, 0.0);
                 engine.compute(op, pair, result, hint);
                 return py::array_t<Real>({nbf, nbf}, result.data());
             },
             py::arg("op"), py::arg("pair"),
             py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute one-electron integrals for a ShellSetPair and scatter into
        a full AO matrix.

        This uses the ShellSetPair work unit and backend-aware dispatch path
        (CPU/GPU according to dispatch policy + hint).

        Parameters
        ----------
        op : Operator
            One-electron operator.
        pair : ShellSetPair
            Shell pair work unit to compute.
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        numpy.ndarray
            AO matrix contribution of shape (n_basis, n_basis).
    )pbdoc")

        .def("compute_quartet_and_consume",
             [](Engine& engine, const Operator& op,
                const ShellSetQuartet& quartet,
                consumers::FockBuilder& consumer,
                BackendHint hint) {
                 engine.compute(op, quartet, consumer, hint);
             },
             py::arg("op"), py::arg("quartet"), py::arg("consumer"),
             py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute a single ShellSetQuartet and accumulate directly into a
        FockBuilder consumer.

        Alias for ``compute(op, quartet, consumer, hint)``.

        Parameters
        ----------
        op : Operator
            Two-electron operator.
        quartet : ShellSetQuartet
            Shell quartet work unit to compute.
        consumer : FockBuilder
            Consumer that receives ERI contributions.
        hint : BackendHint, default=Auto
            Backend selection hint.
    )pbdoc")

        .def("compute_quartets_and_consume",
             [](Engine& engine, const Operator& op,
                const std::vector<ShellSetQuartet>& quartets,
                consumers::FockBuilder& consumer,
                BackendHint hint) {
                 for (const auto& quartet : quartets) {
                     engine.compute(op, quartet, consumer, hint);
                 }
             },
             py::arg("op"), py::arg("quartets"), py::arg("consumer"),
             py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute a list of ShellSetQuartets and accumulate into a FockBuilder.

        Parameters
        ----------
        op : Operator
            Two-electron operator.
        quartets : list of ShellSetQuartet
            Work units to compute in input order.
        consumer : FockBuilder
            Consumer that receives ERI contributions.
        hint : BackendHint, default=Auto
            Backend selection hint.
    )pbdoc")

        // ====================================================================
        // ERI tensor/block (Phase 14 parity)
        // ====================================================================
        .def("compute_eri_tensor",
             [](Engine& engine, const Operator& op, BackendHint hint) {
                 auto tensor = engine.compute_eri_tensor(op, hint);
                 Size nbf = engine.basis().n_basis_functions();
                 return py::array_t<Real>(
                     {nbf, nbf, nbf, nbf}, tensor.data());
             },
             py::arg("op") = Operator::coulomb(),
             py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute the full 4-index ERI tensor.

        Returns (ij|kl) for all basis function indices i,j,k,l as a
        4D NumPy array of shape (nbf, nbf, nbf, nbf).

        Warning: allocates O(nbf^4) memory. Max 200 basis functions.

        Parameters
        ----------
        op : Operator, default=Operator.coulomb()
            Two-electron operator.
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        numpy.ndarray
            4D ERI tensor of shape (nbf, nbf, nbf, nbf).
    )pbdoc")

        .def("compute_eri_block",
             [](Engine& engine, const Operator& op,
                const ShellSetQuartet& quartet, BackendHint hint) {
                 auto block = engine.compute_eri_block(op, quartet, hint);
                 // Return as flat 1D array - shape depends on shell set sizes
                 return py::array_t<Real>(block.size(), block.data());
             },
             py::arg("op"), py::arg("quartet"),
             py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute ERIs for a single ShellSetQuartet as a block.

        Returns a flat array with local basis-function indexing.

        Parameters
        ----------
        op : Operator
            Two-electron operator.
        quartet : ShellSetQuartet
            Shell quartet to compute.
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        numpy.ndarray
            Flat array of ERI values.
    )pbdoc")

        // ====================================================================
        // Parallel and screened batch (Phase 14 parity)
        // ====================================================================
        .def("compute_batch_parallel",
             [](Engine& engine, const Operator& op,
                const std::vector<ShellSetQuartet>& quartets,
                int n_threads, BackendHint hint) {
                 return engine.compute_batch_parallel(
                     op, std::span<const ShellSetQuartet>(quartets),
                     n_threads, hint);
             },
             py::arg("op"), py::arg("quartets"),
             py::arg("n_threads") = 0,
             py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Parallel compute two-electron integrals for multiple ShellSetQuartets.

        Parameters
        ----------
        op : Operator
            Two-electron operator.
        quartets : list of ShellSetQuartet
            Shell quartets to compute.
        n_threads : int, default=0
            Number of threads (0 = auto).
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        list of IntegralBuffer
            Computed buffers in input order.
    )pbdoc")

        .def("compute_all_2e_parallel",
             [](Engine& engine, const Operator& op,
                int n_threads, BackendHint hint) {
                 return engine.compute_all_2e_parallel(op, n_threads, hint);
             },
             py::arg("op"),
             py::arg("n_threads") = 0,
             py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Parallel compute all two-electron integrals for the full basis.

        Parameters
        ----------
        op : Operator
            Two-electron operator.
        n_threads : int, default=0
            Number of threads (0 = auto).
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        list of IntegralBuffer
            Integral buffers for all shell set quartets.
    )pbdoc")

        .def("compute_batch_screened",
             [](Engine& engine, const Operator& op,
                const screening::ScreeningOptions& screening,
                BackendHint hint) {
                 return engine.compute_batch_screened(op, screening, hint);
             },
             py::arg("op"), py::arg("screening"),
             py::arg("hint") = BackendHint::Auto,
             R"pbdoc(
        Compute all two-electron integrals with Schwarz screening.

        Precomputes Schwarz bounds automatically. Empty IntegralBuffers
        in the output indicate screened-out quartets.

        Parameters
        ----------
        op : Operator
            Two-electron operator.
        screening : ScreeningOptions
            Screening parameters.
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        list of IntegralBuffer
            Buffers for all quartets (empty = screened out).
    )pbdoc")

        .def_static("compute_screening_statistics",
             [](const std::vector<IntegralBuffer>& results) {
                 return Engine::compute_screening_statistics(results);
             },
             py::arg("results"),
             R"pbdoc(
        Get screening statistics from batch computation results.

        Counts non-empty (computed) and empty (screened) IntegralBuffers.

        Parameters
        ----------
        results : list of IntegralBuffer
            Results from a screened computation.

        Returns
        -------
        ScreeningStatistics
            Statistics with computed/skipped/total counts.
    )pbdoc")

        .def("__repr__", [](const Engine& e) {
            return "Engine(n_basis=" +
                   std::to_string(e.basis().n_basis_functions()) +
                   ", gpu_available=" + (e.gpu_available() ? "True" : "False") +
                   ")";
        });

    // ========================================================================
    // Multi-GPU Bindings (CUDA only)
    // ========================================================================
#if LIBACCINT_USE_CUDA
    using namespace libaccint::engine;

    // MultiGPUConfig
    py::class_<MultiGPUConfig>(m, "MultiGPUConfig", R"pbdoc(
        Configuration for multi-GPU execution.

        Controls device selection, work distribution, and execution strategy
        for multi-GPU integral computation.
    )pbdoc")
        .def(py::init<>(), "Create default multi-GPU config")
        .def_readwrite("device_ids", &MultiGPUConfig::device_ids,
                       "Device IDs to use (empty = use all available)")
        .def_readwrite("enable_peer_access", &MultiGPUConfig::enable_peer_access,
                       "Enable peer-to-peer access between devices")
        .def_readwrite("streams_per_device", &MultiGPUConfig::streams_per_device,
                       "Number of streams per device")
        .def_readwrite("async_execution", &MultiGPUConfig::async_execution,
                       "Overlap compute and communication")
        .def_readwrite("collect_stats", &MultiGPUConfig::collect_stats,
                       "Collect timing statistics");

    // MultiGPUStats
    py::class_<MultiGPUStats>(m, "MultiGPUStats", R"pbdoc(
        Statistics from multi-GPU execution.
    )pbdoc")
        .def_readonly("total_time_ms", &MultiGPUStats::total_time_ms,
                      "Total wall clock time (ms)")
        .def_readonly("compute_time_ms", &MultiGPUStats::compute_time_ms,
                      "Time spent in kernels (ms)")
        .def_readonly("communication_time_ms", &MultiGPUStats::communication_time_ms,
                      "Time in data transfer (ms)")
        .def_readonly("reduction_time_ms", &MultiGPUStats::reduction_time_ms,
                      "Time in result reduction (ms)")
        .def_readonly("per_device_time_ms", &MultiGPUStats::per_device_time_ms,
                      "Compute time per device (ms)")
        .def_readonly("per_device_quartets", &MultiGPUStats::per_device_quartets,
                      "Quartets processed per device")
        .def("load_balance_efficiency", &MultiGPUStats::load_balance_efficiency,
             "Calculate load balance efficiency");

    // MultiGPUEngine
    py::class_<MultiGPUEngine>(m, "MultiGPUEngine", R"pbdoc(
        Multi-GPU engine for parallel molecular integral computation.

        Coordinates integral computation across multiple GPU devices with
        automatic work distribution and load balancing.

        Parameters
        ----------
        basis : BasisSet
            The basis set (shared across devices).
        config : MultiGPUConfig, optional
            Multi-GPU configuration.
    )pbdoc")
        .def(py::init<const BasisSet&, MultiGPUConfig>(),
             py::arg("basis"), py::arg("config") = MultiGPUConfig{},
             py::keep_alive<1, 2>(),
             "Create multi-GPU engine")
        .def("basis", [](const MultiGPUEngine& e) -> const BasisSet* {
            return &e.basis();
        }, py::return_value_policy::reference_internal,
           "Get the basis set")
        .def("device_ids", &MultiGPUEngine::device_ids,
             "Get active device IDs")
        .def("device_count", &MultiGPUEngine::device_count,
             "Get number of active devices")
        .def("config", &MultiGPUEngine::config,
             py::return_value_policy::reference_internal,
             "Get the configuration")
        .def("stats", &MultiGPUEngine::stats,
             py::return_value_policy::reference_internal,
             "Get the most recent execution statistics")
        .def("compute_overlap_matrix", [](MultiGPUEngine& engine) {
            std::vector<Real> result;
            engine.compute_overlap_matrix(result);
            Size n = engine.basis().n_basis_functions();
            return py::array_t<Real>({n, n}, result.data());
        }, "Compute the full overlap matrix using all devices")
        .def("compute_kinetic_matrix", [](MultiGPUEngine& engine) {
            std::vector<Real> result;
            engine.compute_kinetic_matrix(result);
            Size n = engine.basis().n_basis_functions();
            return py::array_t<Real>({n, n}, result.data());
        }, "Compute the full kinetic matrix using all devices")
        .def("synchronize_all", &MultiGPUEngine::synchronize_all,
             "Synchronize all devices")
        .def("summary", &MultiGPUEngine::summary,
             "Get a summary string")
        .def("__repr__", [](const MultiGPUEngine& e) {
            return "MultiGPUEngine(n_basis=" +
                   std::to_string(e.basis().n_basis_functions()) +
                   ", n_devices=" + std::to_string(e.device_count()) + ")";
        });

    // MultiGPUFockBuilder
    py::class_<consumers::MultiGPUFockBuilder>(m, "MultiGPUFockBuilder", R"pbdoc(
        Multi-GPU Fock matrix builder with parallel accumulation.

        Extends the single-GPU FockBuilder to support accumulation across
        multiple GPU devices with efficient result reduction.

        Parameters
        ----------
        nbf : int
            Number of basis functions.
        device_ids : list of int
            GPU device IDs to use.
    )pbdoc")
        .def(py::init<Size, const std::vector<int>&>(),
             py::arg("nbf"), py::arg("device_ids"),
             "Create multi-GPU Fock builder")
        .def("set_density", [](consumers::MultiGPUFockBuilder& fb, py::array D) {
            if (!py::isinstance<py::array_t<Real>>(D)) {
                throw std::invalid_argument("Density must have dtype float64");
            }
            auto info = D.request();
            if (info.ndim != 2) {
                throw std::invalid_argument("Density must be 2D array");
            }
            auto nbf = fb.nbf();
            if (info.shape[0] != static_cast<py::ssize_t>(nbf) ||
                info.shape[1] != static_cast<py::ssize_t>(nbf)) {
                throw std::invalid_argument(
                    "Density shape must be (" + std::to_string(nbf) +
                    ", " + std::to_string(nbf) + ")");
            }
            fb.set_density(static_cast<const Real*>(info.ptr), nbf);
        }, py::arg("density"),
           "Set the density matrix (replicated to all devices)")
        .def("get_coulomb_matrix", [](consumers::MultiGPUFockBuilder& fb) {
            auto J = fb.get_coulomb_matrix();
            Size n = fb.nbf();
            py::array_t<Real> result({n, n});
            std::copy(J.begin(), J.end(), result.mutable_data());
            return result;
        }, "Get reduced Coulomb matrix J")
        .def("get_exchange_matrix", [](consumers::MultiGPUFockBuilder& fb) {
            auto K = fb.get_exchange_matrix();
            Size n = fb.nbf();
            py::array_t<Real> result({n, n});
            std::copy(K.begin(), K.end(), result.mutable_data());
            return result;
        }, "Get reduced exchange matrix K")
        .def("reset", &consumers::MultiGPUFockBuilder::reset,
             "Reset J and K matrices on all devices")
        .def("synchronize", &consumers::MultiGPUFockBuilder::synchronize,
             "Synchronize all devices")
        .def("nbf", &consumers::MultiGPUFockBuilder::nbf,
             "Get the number of basis functions")
        .def("device_ids", &consumers::MultiGPUFockBuilder::device_ids,
             "Get device IDs")
        .def("__repr__", [](const consumers::MultiGPUFockBuilder& fb) {
            return "MultiGPUFockBuilder(nbf=" + std::to_string(fb.nbf()) +
                   ", n_devices=" +
                   std::to_string(fb.device_ids().size()) + ")";
        });

#endif  // LIBACCINT_USE_CUDA
}
