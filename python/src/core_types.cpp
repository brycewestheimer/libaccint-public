// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file core_types.cpp
/// @brief pybind11 bindings for core LibAccInt types

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <libaccint/core/types.hpp>
#include <libaccint/core/backend.hpp>
#include <libaccint/engine/dispatch_policy.hpp>

namespace py = pybind11;

void bind_core_types(py::module_& m) {
    using namespace libaccint;

    // Type aliases as module attributes for documentation
    m.attr("Real") = py::float_();
    m.attr("Index") = py::int_();
    m.attr("Size") = py::int_();

    // ========================================================================
    // AngularMomentum enum
    // ========================================================================
    py::enum_<AngularMomentum>(m, "AngularMomentum", R"pbdoc(
        Angular momentum quantum number.

        This enum represents the orbital angular momentum (l) of a basis
        function shell. Values range from S (l=0) to I (l=6).
    )pbdoc")
        .value("S", AngularMomentum::S, "l=0, 1 Cartesian function")
        .value("P", AngularMomentum::P, "l=1, 3 Cartesian functions")
        .value("D", AngularMomentum::D, "l=2, 6 Cartesian functions")
        .value("F", AngularMomentum::F, "l=3, 10 Cartesian functions")
        .value("G", AngularMomentum::G, "l=4, 15 Cartesian functions")
        .value("H", AngularMomentum::H, "l=5, 21 Cartesian functions")
        .value("I", AngularMomentum::I, "l=6, 28 Cartesian functions");

    // ========================================================================
    // DerivativeOrder enum
    // ========================================================================
    py::enum_<DerivativeOrder>(m, "DerivativeOrder", R"pbdoc(
        Derivative order for integral computation.

        Controls integral computation order.
    )pbdoc")
        .value("Energy", DerivativeOrder::Energy, "Energy integrals (no derivatives)");

    // ========================================================================
    // Point3D class
    // ========================================================================
    py::class_<Point3D>(m, "Point3D", R"pbdoc(
        3D Cartesian point/vector.

        Represents a point in 3D space with x, y, z coordinates.

        Parameters
        ----------
        x : float, default=0.0
            X coordinate
        y : float, default=0.0
            Y coordinate
        z : float, default=0.0
            Z coordinate

        Examples
        --------
        >>> p = Point3D(1.0, 2.0, 3.0)
        >>> print(p.x, p.y, p.z)
        1.0 2.0 3.0
    )pbdoc")
        .def(py::init<>(), "Create point at origin")
        .def(py::init<Real, Real, Real>(),
             py::arg("x"), py::arg("y"), py::arg("z"),
             "Create point at (x, y, z)")
        .def(py::init([](py::array_t<Real> arr) {
            if (arr.size() != 3) {
                throw std::invalid_argument("Point3D requires exactly 3 elements");
            }
            auto r = arr.unchecked<1>();
            return Point3D(r(0), r(1), r(2));
        }), py::arg("coords"),
            "Create point from array-like with 3 elements")
        .def_readwrite("x", &Point3D::x, "X coordinate")
        .def_readwrite("y", &Point3D::y, "Y coordinate")
        .def_readwrite("z", &Point3D::z, "Z coordinate")
        .def("__getitem__", [](const Point3D& p, int i) {
            if (i < 0 || i > 2) throw py::index_error("Index out of range [0, 2]");
            return p[i];
        })
        .def("__setitem__", [](Point3D& p, int i, Real val) {
            if (i < 0 || i > 2) throw py::index_error("Index out of range [0, 2]");
            p[i] = val;
        })
        .def("distance_squared", &Point3D::distance_squared,
             py::arg("other"),
             "Compute squared distance to another point")
        .def("__repr__", [](const Point3D& p) {
            return "Point3D(" + std::to_string(p.x) + ", " +
                   std::to_string(p.y) + ", " + std::to_string(p.z) + ")";
        })
        .def("to_numpy", [](const Point3D& p) {
            py::array_t<Real> arr(3);
            auto buf = arr.mutable_unchecked<1>();
            buf(0) = p.x;
            buf(1) = p.y;
            buf(2) = p.z;
            return arr;
        }, "Convert to NumPy array");

    // ========================================================================
    // BackendType enum
    // ========================================================================
    py::enum_<BackendType>(m, "BackendType", R"pbdoc(
        Available compute backends.

        Specifies which hardware backend to use for integral computation.
    )pbdoc")
        .value("CPU", BackendType::CPU, "Host CPU (vectorized)")
        .value("CUDA", BackendType::CUDA, "NVIDIA CUDA");

    // ========================================================================
    // BackendHint enum
    // ========================================================================
    py::enum_<BackendHint>(m, "BackendHint", R"pbdoc(
        Hint for backend selection.

        Provides hints to the dispatch policy about which backend to prefer.
    )pbdoc")
        .value("Auto", BackendHint::Auto, "Automatic selection based on heuristics")
        .value("PreferCPU", BackendHint::PreferCPU, "Prefer CPU if suitable")
        .value("PreferGPU", BackendHint::PreferGPU, "Prefer GPU if available")
        .value("ForceCPU", BackendHint::ForceCPU, "Force CPU backend")
        .value("ForceGPU", BackendHint::ForceGPU, "Force GPU backend");

    // ========================================================================
    // Backend utility functions
    // ========================================================================
    m.def("is_backend_available", &is_backend_available,
          py::arg("backend"),
          R"pbdoc(
        Check if a compute backend is available at runtime.

        Parameters
        ----------
        backend : BackendType
            The backend to check.

        Returns
        -------
        bool
            True if the backend is available, False otherwise.

        Examples
        --------
        >>> libaccint.is_backend_available(BackendType.CPU)
        True
        >>> libaccint.is_backend_available(BackendType.CUDA)
        False  # If no CUDA device
    )pbdoc");

    m.def("backend_name", &backend_name,
          py::arg("backend"),
          R"pbdoc(
        Get the string name of a backend.

        Parameters
        ----------
        backend : BackendType
            The backend type.

        Returns
        -------
        str
            String representation ("CPU" or "CUDA").
    )pbdoc");

    // ========================================================================
    // Helper functions
    // ========================================================================
    m.def("n_cartesian", &n_cartesian,
          py::arg("l"),
          R"pbdoc(
        Number of Cartesian basis functions for angular momentum l.

        Returns (l+1)*(l+2)/2.

        Parameters
        ----------
        l : int
            Angular momentum quantum number.

        Returns
        -------
        int
            Number of Cartesian functions.
    )pbdoc");

    m.def("n_spherical", &n_spherical,
          py::arg("l"),
          R"pbdoc(
        Number of spherical basis functions for angular momentum l.

        Returns 2*l+1.

        Parameters
        ----------
        l : int
            Angular momentum quantum number.

        Returns
        -------
        int
            Number of spherical functions.
    )pbdoc");
}
