// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file convenience_api.cpp
/// @brief pybind11 bindings for convenience API functions

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/consumers/fock_builder.hpp>

namespace py = pybind11;

void bind_convenience(py::module_& m) {
    using namespace libaccint;
    using namespace libaccint::consumers;

    // ========================================================================
    // Convenience factory for atoms from list of tuples
    // ========================================================================
    m.def("atoms_from_xyz",
          [](const std::vector<std::tuple<int, Real, Real, Real>>& xyz_data) {
              std::vector<data::Atom> atoms;
              atoms.reserve(xyz_data.size());
              for (const auto& [z, x, y, zc] : xyz_data) {
                  atoms.push_back({z, Point3D(x, y, zc)});
              }
              return atoms;
          },
          py::arg("xyz_data"),
          R"pbdoc(
        Create list of atoms from (Z, x, y, z) tuples.

        Convenience function for quickly creating atom lists.

        Parameters
        ----------
        xyz_data : list of tuples
            List of (atomic_number, x, y, z) tuples.

        Returns
        -------
        list of Atom
            List of Atom objects.

        Examples
        --------
        >>> atoms = atoms_from_xyz([
        ...     (8, 0.0, 0.0, 0.0),
        ...     (1, 0.0, 1.43, -1.11),
        ...     (1, 0.0, -1.43, -1.11),
        ... ])
    )pbdoc");

    // ========================================================================
    // Quick matrix computation functions
    // ========================================================================
    m.def("quick_overlap",
          [](const BasisSet& basis, BackendHint hint) {
              Engine engine(basis);
              std::vector<Real> result;
              engine.compute_overlap_matrix(result, hint);
              Size n = basis.n_basis_functions();
              return py::array_t<Real>({n, n}, result.data());
          },
          py::arg("basis"), py::arg("hint") = BackendHint::Auto,
          R"pbdoc(
        Quickly compute overlap matrix from basis set.

        Creates a temporary Engine and computes S.

        Parameters
        ----------
        basis : BasisSet
            The basis set.
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        numpy.ndarray
            Overlap matrix of shape (n_basis, n_basis).
    )pbdoc");

    m.def("quick_kinetic",
          [](const BasisSet& basis, BackendHint hint) {
              Engine engine(basis);
              std::vector<Real> result;
              engine.compute_kinetic_matrix(result, hint);
              Size n = basis.n_basis_functions();
              return py::array_t<Real>({n, n}, result.data());
          },
          py::arg("basis"), py::arg("hint") = BackendHint::Auto,
          R"pbdoc(
        Quickly compute kinetic energy matrix from basis set.

        Creates a temporary Engine and computes T.

        Parameters
        ----------
        basis : BasisSet
            The basis set.
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        numpy.ndarray
            Kinetic energy matrix of shape (n_basis, n_basis).
    )pbdoc");

    m.def("quick_nuclear",
          [](const BasisSet& basis, const std::vector<data::Atom>& atoms,
             BackendHint hint) {
              Engine engine(basis);
              PointChargeParams charges;
              for (const auto& atom : atoms) {
                  charges.x.push_back(atom.position.x);
                  charges.y.push_back(atom.position.y);
                  charges.z.push_back(atom.position.z);
                  charges.charge.push_back(static_cast<Real>(atom.atomic_number));
              }
              std::vector<Real> result;
              engine.compute_nuclear_matrix(charges, result, hint);
              Size n = basis.n_basis_functions();
              return py::array_t<Real>({n, n}, result.data());
          },
          py::arg("basis"), py::arg("atoms"), py::arg("hint") = BackendHint::Auto,
          R"pbdoc(
        Quickly compute nuclear attraction matrix.

        Creates a temporary Engine and computes V.

        Parameters
        ----------
        basis : BasisSet
            The basis set.
        atoms : list of Atom
            Atoms with nuclear charges and positions.
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        numpy.ndarray
            Nuclear attraction matrix of shape (n_basis, n_basis).
    )pbdoc");

    m.def("quick_core_hamiltonian",
          [](const BasisSet& basis, const std::vector<data::Atom>& atoms,
             BackendHint hint) {
              Engine engine(basis);
              PointChargeParams charges;
              for (const auto& atom : atoms) {
                  charges.x.push_back(atom.position.x);
                  charges.y.push_back(atom.position.y);
                  charges.z.push_back(atom.position.z);
                  charges.charge.push_back(static_cast<Real>(atom.atomic_number));
              }
              std::vector<Real> result;
              engine.compute_core_hamiltonian(charges, result, hint);
              Size n = basis.n_basis_functions();
              return py::array_t<Real>({n, n}, result.data());
          },
          py::arg("basis"), py::arg("atoms"), py::arg("hint") = BackendHint::Auto,
          R"pbdoc(
        Quickly compute core Hamiltonian H = T + V.

        Creates a temporary Engine and computes H_core.

        Parameters
        ----------
        basis : BasisSet
            The basis set.
        atoms : list of Atom
            Atoms with nuclear charges and positions.
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        numpy.ndarray
            Core Hamiltonian matrix of shape (n_basis, n_basis).
    )pbdoc");

    // ========================================================================
    // High-level Fock build
    // ========================================================================
    m.def("quick_fock_build",
          [](const BasisSet& basis, py::array_t<Real> density,
             py::array_t<Real> H_core, Real exchange_fraction,
             BackendHint hint) {
              Size nbf = basis.n_basis_functions();

              // Validate density
              auto d_info = density.request();
              if (d_info.ndim != 2 ||
                  d_info.shape[0] != static_cast<py::ssize_t>(nbf) ||
                  d_info.shape[1] != static_cast<py::ssize_t>(nbf)) {
                  throw std::invalid_argument(
                      "Density must have shape (" + std::to_string(nbf) +
                      ", " + std::to_string(nbf) + ")");
              }

              // Create engine and FockBuilder
              Engine engine(basis);
              FockBuilder fock(nbf);
              fock.set_density(static_cast<const Real*>(d_info.ptr), nbf);

              // Compute ERIs and accumulate
              engine.compute_and_consume(Operator::coulomb(), fock, hint);

              // Build Fock matrix
              std::span<const Real> h_span(
                  static_cast<const Real*>(H_core.request().ptr), nbf * nbf);
              auto F = fock.get_fock_matrix(h_span, exchange_fraction);

              py::array_t<Real> result({nbf, nbf});
              std::copy(F.begin(), F.end(), result.mutable_data());
              return result;
          },
          py::arg("basis"), py::arg("density"), py::arg("H_core"),
          py::arg("exchange_fraction") = 1.0, py::arg("hint") = BackendHint::Auto,
          R"pbdoc(
        Build Fock matrix in one call.

        High-level convenience function that creates temporary Engine
        and FockBuilder, computes ERIs, and returns F = H_core + J - x*K.

        Parameters
        ----------
        basis : BasisSet
            The basis set.
        density : numpy.ndarray
            Density matrix of shape (nbf, nbf).
        H_core : numpy.ndarray
            Core Hamiltonian of shape (nbf, nbf).
        exchange_fraction : float, default=1.0
            Fraction of exact exchange (1.0 for HF).
        hint : BackendHint, default=Auto
            Backend selection hint.

        Returns
        -------
        numpy.ndarray
            Fock matrix of shape (nbf, nbf).
    )pbdoc");
}
