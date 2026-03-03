// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file consumer_bindings.cpp
/// @brief pybind11 bindings for consumer classes (FockBuilder, etc.)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <libaccint/consumers/fock_builder.hpp>

#if LIBACCINT_USE_CUDA
#include <libaccint/consumers/fock_builder_gpu.hpp>
#endif

namespace py = pybind11;

void bind_consumers(py::module_& m) {
    using namespace libaccint;
    using namespace libaccint::consumers;

    // ========================================================================
    // FockThreadingStrategy enum
    // ========================================================================
    py::enum_<FockThreadingStrategy>(m, "FockThreadingStrategy", R"pbdoc(
        Threading strategy for FockBuilder.

        Controls how thread-safety is handled during parallel Fock builds.
    )pbdoc")
        .value("Sequential", FockThreadingStrategy::Sequential,
               "Non-thread-safe (default)")
        .value("Atomic", FockThreadingStrategy::Atomic,
               "Uses atomic operations for thread safety")
        .value("ThreadLocal", FockThreadingStrategy::ThreadLocal,
               "Uses thread-local buffers with final reduction");

    // ========================================================================
    // FockBuilder class
    // ========================================================================
    py::class_<FockBuilder>(m, "FockBuilder", R"pbdoc(
        Builds Coulomb (J) and exchange (K) matrices from two-electron integrals.

        FockBuilder is a "consumer" that accumulates J and K contributions
        as integrals are computed. This enables memory-efficient Fock matrix
        construction without storing all ERIs.

        The accumulation uses the density matrix D:
            J_μν     += Σ_λσ (μν|λσ) D_λσ
            K_μλ     += Σ_νσ (μν|λσ) D_νσ

        Parameters
        ----------
        nbf : int
            Number of basis functions.

        Examples
        --------
        >>> fock = FockBuilder(7)  # STO-3G water
        >>> fock.set_density(D)    # Set density matrix
        >>> engine.compute_and_consume(Operator.coulomb(), fock)
        >>> J = fock.get_coulomb_matrix()
        >>> K = fock.get_exchange_matrix()
        >>> F = H_core + J - 0.5 * K  # For RHF
    )pbdoc")
        .def(py::init<Size>(), py::arg("nbf"),
             "Create FockBuilder for basis with nbf functions")

        .def("set_density", [](FockBuilder& fb, py::array D) {
            if (!py::isinstance<py::array_t<Real>>(D)) {
                throw std::invalid_argument("Density must have dtype float64");
            }
            auto info = D.request();
            if (info.ndim != 2) {
                throw std::invalid_argument("Density must be 2D array");
            }
            if (info.shape[0] != static_cast<py::ssize_t>(fb.nbf()) ||
                info.shape[1] != static_cast<py::ssize_t>(fb.nbf())) {
                throw std::invalid_argument(
                    "Density shape must be (" + std::to_string(fb.nbf()) +
                    ", " + std::to_string(fb.nbf()) + ")");
            }
            const auto item_size = static_cast<py::ssize_t>(sizeof(Real));
            const bool c_contiguous =
                (info.strides[1] == item_size) &&
                (info.strides[0] == static_cast<py::ssize_t>(fb.nbf()) * item_size);
            if (!c_contiguous) {
                throw std::invalid_argument(
                    "Density must be C-contiguous (row-major)");
            }
            fb.set_density(static_cast<const Real*>(info.ptr), fb.nbf());
        }, py::arg("density"),
           R"pbdoc(
        Set the density matrix for accumulation.

        Parameters
        ----------
        density : numpy.ndarray
            Density matrix of shape (nbf, nbf). Must be contiguous.
    )pbdoc")

        .def("get_coulomb_matrix", [](const FockBuilder& fb) {
            auto J = fb.get_coulomb_matrix();
            Size n = fb.nbf();
            py::array_t<Real> result({n, n});
            std::copy(J.begin(), J.end(), result.mutable_data());
            return result;
        }, R"pbdoc(
        Get the accumulated Coulomb matrix J.

        Returns
        -------
        numpy.ndarray
            Coulomb matrix of shape (nbf, nbf).
    )pbdoc")

        .def("get_exchange_matrix", [](const FockBuilder& fb) {
            auto K = fb.get_exchange_matrix();
            Size n = fb.nbf();
            py::array_t<Real> result({n, n});
            std::copy(K.begin(), K.end(), result.mutable_data());
            return result;
        }, R"pbdoc(
        Get the accumulated exchange matrix K.

        Returns
        -------
        numpy.ndarray
            Exchange matrix of shape (nbf, nbf).
    )pbdoc")

        .def("get_fock_matrix", [](const FockBuilder& fb,
                                   py::array H_core,
                                   Real exchange_fraction) {
            if (!py::isinstance<py::array_t<Real>>(H_core)) {
                throw std::invalid_argument("H_core must have dtype float64");
            }
            auto info = H_core.request();
            if (info.ndim != 2) {
                throw std::invalid_argument("H_core must be 2D array");
            }
            if (info.shape[0] != static_cast<py::ssize_t>(fb.nbf()) ||
                info.shape[1] != static_cast<py::ssize_t>(fb.nbf())) {
                throw std::invalid_argument(
                    "H_core shape must be (" + std::to_string(fb.nbf()) +
                    ", " + std::to_string(fb.nbf()) + ")");
            }
            const auto item_size = static_cast<py::ssize_t>(sizeof(Real));
            const bool c_contiguous =
                (info.strides[1] == item_size) &&
                (info.strides[0] == static_cast<py::ssize_t>(fb.nbf()) * item_size);
            if (!c_contiguous) {
                throw std::invalid_argument(
                    "H_core must be C-contiguous (row-major)");
            }
            std::span<const Real> h_span(static_cast<const Real*>(info.ptr),
                                          fb.nbf() * fb.nbf());
            auto F = fb.get_fock_matrix(h_span, exchange_fraction);
            Size n = fb.nbf();
            py::array_t<Real> result({n, n});
            std::copy(F.begin(), F.end(), result.mutable_data());
            return result;
        }, py::arg("H_core"), py::arg("exchange_fraction") = 1.0,
           R"pbdoc(
        Compute the Fock matrix F = H_core + J - exchange_fraction * K.

        Parameters
        ----------
        H_core : numpy.ndarray
            Core Hamiltonian matrix of shape (nbf, nbf).
        exchange_fraction : float, default=1.0
            Fraction of exact exchange (1.0 for RHF, 0.0 for pure DFT).

        Returns
        -------
        numpy.ndarray
            Fock matrix of shape (nbf, nbf).
    )pbdoc")

        .def("reset", &FockBuilder::reset,
             "Reset J and K matrices to zero")

        .def("nbf", &FockBuilder::nbf,
             "Get the number of basis functions")

        // Thread-safety configuration
        .def("set_threading_strategy", &FockBuilder::set_threading_strategy,
             py::arg("strategy"),
             R"pbdoc(
        Set the threading strategy for accumulation.

        Call before using in parallel regions. After changing, call reset().

        Parameters
        ----------
        strategy : FockThreadingStrategy
            The threading mode to use.
    )pbdoc")

        .def("threading_strategy", &FockBuilder::threading_strategy,
             "Get the current threading strategy")

        .def("prepare_parallel", &FockBuilder::prepare_parallel,
             py::arg("n_threads") = 0,
             R"pbdoc(
        Prepare thread-local buffers for parallel execution.

        Must be called before parallel region with ThreadLocal strategy.

        Parameters
        ----------
        n_threads : int, default=0
            Number of threads (0 = auto-detect).
    )pbdoc")

        .def("finalize_parallel", &FockBuilder::finalize_parallel,
             "Reduce thread-local buffers into main J/K matrices")

        .def("__repr__", [](const FockBuilder& fb) {
            return "FockBuilder(nbf=" + std::to_string(fb.nbf()) + ")";
        });

    // ========================================================================
    // GPU Consumer Bindings (CUDA only)
    // ========================================================================
#if LIBACCINT_USE_CUDA

    // GpuFockBuilder
    py::class_<GpuFockBuilder>(m, "GpuFockBuilder", R"pbdoc(
        GPU-accelerated Fock matrix builder using device-side accumulation.

        Accumulates Coulomb (J) and exchange (K) matrix contributions using
        atomic operations on GPU device memory.

        Parameters
        ----------
        nbf : int
            Number of basis functions.
    )pbdoc")
        .def(py::init<Size>(), py::arg("nbf"),
             "Create GpuFockBuilder for basis with nbf functions")
        .def("set_density", [](GpuFockBuilder& fb, py::array D) {
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
           "Set the density matrix (uploads to GPU)")
        .def("get_coulomb_matrix", [](const GpuFockBuilder& fb) {
            auto J = fb.get_coulomb_matrix();
            Size n = fb.nbf();
            py::array_t<Real> result({n, n});
            std::copy(J.begin(), J.end(), result.mutable_data());
            return result;
        }, "Get the Coulomb matrix J (downloads from GPU)")
        .def("get_exchange_matrix", [](const GpuFockBuilder& fb) {
            auto K = fb.get_exchange_matrix();
            Size n = fb.nbf();
            py::array_t<Real> result({n, n});
            std::copy(K.begin(), K.end(), result.mutable_data());
            return result;
        }, "Get the exchange matrix K (downloads from GPU)")
        .def("reset", &GpuFockBuilder::reset,
             "Reset J and K matrices to zero on GPU")
        .def("synchronize", &GpuFockBuilder::synchronize,
             "Synchronize all pending GPU operations")
        .def("nbf", &GpuFockBuilder::nbf,
             "Get the number of basis functions")
        .def("__repr__", [](const GpuFockBuilder& fb) {
            return "GpuFockBuilder(nbf=" + std::to_string(fb.nbf()) + ")";
        });

#endif  // LIBACCINT_USE_CUDA
}
