// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file buffer_bindings.cpp
/// @brief pybind11 bindings for integral buffers with NumPy array protocol

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>

namespace py = pybind11;

void bind_buffers(py::module_& m) {
    using namespace libaccint;

    // ========================================================================
    // OneElectronBuffer<0> (Energy only)
    // ========================================================================
    py::class_<OneElectronBuffer<0>>(m, "OneElectronBuffer", R"pbdoc(
        Buffer for one-electron integrals.

        Stores computed integrals for a shell pair (a, b).
        Provides NumPy array access for Python interoperability.

        The buffer uses row-major storage with shape (na, nb) where
        na and nb are the number of basis functions in shells a and b.

        Parameters
        ----------
        na : int
            Number of basis functions in shell A.
        nb : int
            Number of basis functions in shell B.

        Examples
        --------
        >>> buffer = OneElectronBuffer(6, 6)  # For d-d shell pair
        >>> buffer.resize(3, 3)  # Resize for p-p shell pair
        >>> arr = buffer.to_numpy()  # Get as NumPy array
    )pbdoc")
        .def(py::init<>(), "Create empty buffer")
        .def(py::init([](int na, int nb) {
            OneElectronBuffer<0> buf;
            buf.resize(na, nb);
            return buf;
        }), py::arg("na"), py::arg("nb"),
            "Create buffer with dimensions (na, nb)")
        .def("resize", &OneElectronBuffer<0>::resize,
             py::arg("na"), py::arg("nb"),
             "Resize buffer for given shell dimensions")
        .def("clear", &OneElectronBuffer<0>::clear,
             "Zero all elements in the buffer")
        .def("na", &OneElectronBuffer<0>::na, "Number of functions in shell A")
        .def("nb", &OneElectronBuffer<0>::nb, "Number of functions in shell B")
        .def("size", &OneElectronBuffer<0>::size, "Total number of elements")
        .def("empty", &OneElectronBuffer<0>::empty, "Check if buffer is empty")
        .def("__getitem__", [](const OneElectronBuffer<0>& buf,
                               std::tuple<int, int> idx) {
            return buf(std::get<0>(idx), std::get<1>(idx));
        }, py::arg("index"), "Access element at (a, b)")
        .def("__setitem__", [](OneElectronBuffer<0>& buf,
                               std::tuple<int, int> idx, Real val) {
            buf(std::get<0>(idx), std::get<1>(idx)) = val;
        }, py::arg("index"), py::arg("value"), "Set element at (a, b)")
        .def("to_numpy", [](const OneElectronBuffer<0>& buf) {
            auto data = buf.data();
            int na = buf.na(), nb = buf.nb();
            auto result = py::array_t<Real>({na, nb});
            auto r = result.mutable_unchecked<2>();
            Size idx = 0;
            for (int a = 0; a < na; ++a)
                for (int b = 0; b < nb; ++b)
                    r(a, b) = data[idx++];
            return result;
        }, "Convert to NumPy array of shape (na, nb)")
        .def("__repr__", [](const OneElectronBuffer<0>& buf) {
            return "OneElectronBuffer(na=" +
                   std::to_string(buf.na()) + ", nb=" +
                   std::to_string(buf.nb()) + ")";
        });

    // ========================================================================
    // TwoElectronBuffer<0> (Energy only)
    // ========================================================================
    py::class_<TwoElectronBuffer<0>>(m, "TwoElectronBuffer", R"pbdoc(
        Buffer for two-electron integrals.

        Stores computed electron repulsion integrals (ERIs) for a shell
        quartet (a, b | c, d).

        The buffer uses row-major storage with shape (na, nb, nc, nd).

        Parameters
        ----------
        na, nb, nc, nd : int
            Number of basis functions in each shell.

        Examples
        --------
        >>> buffer = TwoElectronBuffer(3, 3, 3, 3)  # For (p p | p p)
        >>> arr = buffer.to_numpy()  # Get as NumPy array
    )pbdoc")
        .def(py::init<>(), "Create empty buffer")
        .def(py::init([](int na, int nb, int nc, int nd) {
            TwoElectronBuffer<0> buf;
            buf.resize(na, nb, nc, nd);
            return buf;
        }), py::arg("na"), py::arg("nb"), py::arg("nc"), py::arg("nd"),
            "Create buffer with dimensions (na, nb, nc, nd)")
        .def("resize", &TwoElectronBuffer<0>::resize,
             py::arg("na"), py::arg("nb"), py::arg("nc"), py::arg("nd"),
             "Resize buffer for given shell dimensions")
        .def("clear", &TwoElectronBuffer<0>::clear,
             "Zero all elements in the buffer")
        .def("na", &TwoElectronBuffer<0>::na, "Number of functions in shell A")
        .def("nb", &TwoElectronBuffer<0>::nb, "Number of functions in shell B")
        .def("nc", &TwoElectronBuffer<0>::nc, "Number of functions in shell C")
        .def("nd", &TwoElectronBuffer<0>::nd, "Number of functions in shell D")
        .def("size", &TwoElectronBuffer<0>::size, "Total number of elements")
        .def("empty", &TwoElectronBuffer<0>::empty, "Check if buffer is empty")
        .def("n_integrals", &TwoElectronBuffer<0>::n_integrals,
             "Number of integrals (na*nb*nc*nd)")
        .def("__getitem__", [](const TwoElectronBuffer<0>& buf,
                               std::tuple<int, int, int, int> idx) {
            return buf(std::get<0>(idx), std::get<1>(idx),
                       std::get<2>(idx), std::get<3>(idx));
        }, py::arg("index"), "Access element at (a, b, c, d)")
        .def("__setitem__", [](TwoElectronBuffer<0>& buf,
                               std::tuple<int, int, int, int> idx, Real val) {
            buf(std::get<0>(idx), std::get<1>(idx),
                std::get<2>(idx), std::get<3>(idx)) = val;
        }, py::arg("index"), py::arg("value"), "Set element at (a, b, c, d)")
        .def("to_numpy", [](const TwoElectronBuffer<0>& buf) {
            auto data = buf.data();
            int na = buf.na(), nb = buf.nb(), nc = buf.nc(), nd = buf.nd();
            auto result = py::array_t<Real>({na, nb, nc, nd});
            auto r = result.mutable_unchecked<4>();
            Size idx = 0;
            for (int a = 0; a < na; ++a)
                for (int b = 0; b < nb; ++b)
                    for (int c = 0; c < nc; ++c)
                        for (int d = 0; d < nd; ++d)
                            r(a, b, c, d) = data[idx++];
            return result;
        }, "Convert to NumPy array of shape (na, nb, nc, nd)")
        .def("__repr__", [](const TwoElectronBuffer<0>& buf) {
            return "TwoElectronBuffer(na=" +
                   std::to_string(buf.na()) + ", nb=" +
                   std::to_string(buf.nb()) + ", nc=" +
                   std::to_string(buf.nc()) + ", nd=" +
                   std::to_string(buf.nd()) + ")";
        });

}
