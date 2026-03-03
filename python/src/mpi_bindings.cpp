// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file mpi_bindings.cpp
/// @brief pybind11 bindings for MPI-related types and utilities

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <libaccint/config.hpp>

#if LIBACCINT_USE_MPI
#include <libaccint/mpi/mpi_guard.hpp>
#endif

namespace py = pybind11;

#if LIBACCINT_USE_MPI

void init_mpi_bindings(py::module_& m) {
    using namespace libaccint::mpi;

    m.def("has_mpi", []() { return true; },
          "Returns True if LibAccInt was built with MPI support");

    py::class_<MPIGuard>(m, "MPIGuard", R"pbdoc(
        RAII guard for MPI initialization and finalization.

        Handles MPI lifecycle in a way that's compatible with both
        LibAccInt initializing MPI itself and external code initializing
        MPI before LibAccInt.

        If MPI is already initialized when MPIGuard is constructed, it will
        NOT call MPI_Finalize on destruction.
    )pbdoc")
        .def(py::init<>(),
             "Initialize MPI if not already initialized")
        .def("owns_finalize", &MPIGuard::owns_finalize,
             "Check if this guard owns the MPI finalization")
        .def("thread_level", &MPIGuard::thread_level,
             "Get the provided thread support level")
        .def("thread_support_satisfied", &MPIGuard::thread_support_satisfied,
             "Check if the requested thread support was provided")
        .def_static("is_initialized", &MPIGuard::is_initialized,
                    "Check if MPI is currently initialized")
        .def_static("is_finalized", &MPIGuard::is_finalized,
                    "Check if MPI has been finalized")
        .def_static("rank", []() { return MPIGuard::rank(); },
                    "Get the rank of the current process")
        .def_static("size", []() { return MPIGuard::size(); },
                    "Get the size of the communicator")
        .def_static("processor_name", &MPIGuard::processor_name,
                    "Get the processor name")
        .def_static("barrier", []() { MPIGuard::barrier(); },
                    "Barrier synchronization")
        .def_static("is_root", []() { return MPIGuard::is_root(); },
                    "Check if this is the root process (rank 0)");
}

#endif  // LIBACCINT_USE_MPI
