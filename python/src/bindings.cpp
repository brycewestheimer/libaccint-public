// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file bindings.cpp
/// @brief Main pybind11 module definition for LibAccInt Python bindings

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <libaccint/config.hpp>

namespace py = pybind11;

// Forward declarations for submodule binding functions
void bind_core_types(py::module_& m);
void bind_basis(py::module_& m);
void bind_operators(py::module_& m);
void bind_buffers(py::module_& m);
void bind_engine(py::module_& m);
void bind_consumers(py::module_& m);
void bind_screening(py::module_& m);
void bind_convenience(py::module_& m);
void bind_advanced(py::module_& m);

#if LIBACCINT_USE_MPI
void init_mpi_bindings(py::module_& m);
#endif

/// @brief Main Python module definition
PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        LibAccInt C++ Core Extension
        ----------------------------

        This module provides Python bindings for the LibAccInt C++ library,
        enabling high-performance molecular integral computation with GPU
        acceleration.

        For high-level usage, use the libaccint Python package which provides
        more Pythonic interfaces built on top of this core extension.
    )pbdoc";

    // Version information (derived from CMake-generated config.hpp)
    m.attr("__version__") = LIBACCINT_VERSION_FULL;

    // Bind all submodules
    bind_core_types(m);
    bind_basis(m);
    bind_operators(m);
    bind_buffers(m);
    bind_engine(m);
    bind_consumers(m);
    bind_screening(m);
    bind_convenience(m);
    bind_advanced(m);

#if LIBACCINT_USE_MPI
    init_mpi_bindings(m);
#endif
}
