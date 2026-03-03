// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file basis_bindings.cpp
/// @brief pybind11 bindings for basis set types (Shell, ShellSet, BasisSet)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/data/basis_parser.hpp>

#include <memory>

namespace py = pybind11;

void bind_basis(py::module_& m) {
    using namespace libaccint;

    // ========================================================================
    // Atom struct (from data/builtin_basis.hpp)
    // ========================================================================
    py::class_<data::Atom>(m, "Atom", R"pbdoc(
        Atom specification for basis set construction.

        Represents an atom with its atomic number (Z) and 3D position
        in atomic units (Bohr).

        Parameters
        ----------
        atomic_number : int
            Atomic number (Z). E.g., 1 for H, 6 for C, 8 for O.
        position : Point3D or array-like
            Position in Bohr (atomic units).

        Examples
        --------
        >>> # Water molecule atoms
        >>> O = Atom(8, [0.0, 0.0, 0.0])
        >>> H1 = Atom(1, [0.0, 1.43, -1.11])
        >>> H2 = Atom(1, [0.0, -1.43, -1.11])
    )pbdoc")
        .def(py::init<>(), "Create empty atom")
        .def(py::init([](int z, Point3D pos) {
            return data::Atom{z, pos};
        }), py::arg("atomic_number"), py::arg("position"),
            "Create atom with atomic number and Point3D position")
        .def(py::init([](int z, py::array_t<Real> pos) {
            if (pos.size() != 3) {
                throw std::invalid_argument("Position must have exactly 3 elements");
            }
            auto r = pos.unchecked<1>();
            return data::Atom{z, Point3D(r(0), r(1), r(2))};
        }), py::arg("atomic_number"), py::arg("position"),
            "Create atom with atomic number and array position")
        .def(py::init([](int z, std::vector<Real> pos) {
            if (pos.size() != 3) {
                throw std::invalid_argument("Position must have exactly 3 elements");
            }
            return data::Atom{z, Point3D(pos[0], pos[1], pos[2])};
        }), py::arg("atomic_number"), py::arg("position"),
            "Create atom with atomic number and list position")
        .def_readwrite("atomic_number", &data::Atom::atomic_number,
                       "Atomic number (Z)")
        .def_readwrite("position", &data::Atom::position,
                       "Position in Bohr")
        .def("__repr__", [](const data::Atom& a) {
            return "Atom(" + std::to_string(a.atomic_number) + ", [" +
                   std::to_string(a.position.x) + ", " +
                   std::to_string(a.position.y) + ", " +
                   std::to_string(a.position.z) + "])";
        });

    // ========================================================================
    // Shell class
    // ========================================================================
    py::class_<Shell>(m, "Shell", R"pbdoc(
        Contracted Gaussian shell.

        A Shell represents a contracted Gaussian basis function centered at a
        point in space with a given angular momentum. It consists of:
          - A center position (Point3D)
          - An angular momentum quantum number (l)
          - A set of primitive Gaussian exponents (α_i)
          - A set of contraction coefficients (c_i)

        By default, coefficients are automatically normalized. Use the
        pre_normalized parameter to skip normalization for already-normalized
        coefficients.

        Parameters
        ----------
        angular_momentum : int or AngularMomentum
            Angular momentum quantum number (0=S, 1=P, 2=D, etc.)
        center : Point3D or array-like
            Shell center position in Bohr.
        exponents : array-like
            Primitive Gaussian exponents (must be positive).
        coefficients : array-like
            Contraction coefficients.
        pre_normalized : bool, default=False
            If True, skip normalization (coefficients already normalized).

        Examples
        --------
        >>> # Create an s-shell at origin
        >>> s_shell = Shell(0, [0, 0, 0], [3.42, 0.62], [0.15, 0.85])
    )pbdoc")
        .def(py::init([](int am, Point3D center,
                         std::vector<Real> exponents,
                         std::vector<Real> coefficients,
                         bool is_pre_normalized) {
            if (is_pre_normalized) {
                return Shell(libaccint::pre_normalized, am, center,
                            std::move(exponents), std::move(coefficients));
            } else {
                return Shell(am, center, std::move(exponents),
                            std::move(coefficients));
            }
        }),
             py::arg("angular_momentum"),
             py::arg("center"),
             py::arg("exponents"),
             py::arg("coefficients"),
             py::arg("pre_normalized") = false,
             "Create a shell with the given parameters")
        .def(py::init([](int am, py::array_t<Real> center,
                         py::array_t<Real> exponents,
                         py::array_t<Real> coefficients,
                         bool is_pre_normalized) {
            if (center.size() != 3) {
                throw std::invalid_argument("Center must have exactly 3 elements");
            }
            auto c = center.unchecked<1>();
            Point3D pt(c(0), c(1), c(2));

            std::vector<Real> exp_vec(exponents.data(),
                                      exponents.data() + exponents.size());
            std::vector<Real> coef_vec(coefficients.data(),
                                       coefficients.data() + coefficients.size());

            if (is_pre_normalized) {
                return Shell(libaccint::pre_normalized, am, pt,
                            std::move(exp_vec), std::move(coef_vec));
            } else {
                return Shell(am, pt, std::move(exp_vec), std::move(coef_vec));
            }
        }),
             py::arg("angular_momentum"),
             py::arg("center"),
             py::arg("exponents"),
             py::arg("coefficients"),
             py::arg("pre_normalized") = false,
             "Create a shell from NumPy arrays")
        .def("angular_momentum", &Shell::angular_momentum,
             "Get angular momentum as integer")
        .def("angular_momentum_enum", &Shell::angular_momentum_enum,
             "Get angular momentum as enum")
        .def("center", &Shell::center, py::return_value_policy::reference,
             "Get shell center position")
        .def("n_primitives", &Shell::n_primitives,
             "Get number of primitive Gaussians")
        .def("n_functions", &Shell::n_functions,
             "Get number of Cartesian basis functions")
        .def("valid", &Shell::valid,
             "Check if shell is valid (has primitives)")
        .def("atom_index", &Shell::atom_index,
             "Get atom index (-1 if not set)")
        .def("shell_index", &Shell::shell_index,
             "Get shell index within basis set (-1 if not set)")
        .def("function_index", &Shell::function_index,
             "Get starting basis function index (-1 if not set)")
        .def("exponents", [](const Shell& s) {
            auto exp = s.exponents();
            return py::array_t<Real>(exp.size(), exp.data());
        }, "Get exponents as NumPy array")
        .def("coefficients", [](const Shell& s) {
            auto coef = s.coefficients();
            return py::array_t<Real>(coef.size(), coef.data());
        }, "Get coefficients as NumPy array")
        .def("__repr__", [](const Shell& s) {
            std::string am_char(1, "SPDFGHI"[s.angular_momentum()]);
            return "Shell(" + am_char + ", n_prim=" +
                   std::to_string(s.n_primitives()) + ", center=[" +
                   std::to_string(s.center().x) + ", " +
                   std::to_string(s.center().y) + ", " +
                   std::to_string(s.center().z) + "])";
        });

    // ========================================================================
    // ShellSet class
    // ========================================================================
    py::class_<ShellSet>(m, "ShellSet", R"pbdoc(
        Collection of shells with identical angular momentum and primitive count.

        ShellSet groups shells by (angular_momentum, n_primitives) for efficient
        batched integral computation. All shells in a set share the same:
          - Angular momentum
          - Number of primitives
          - Contraction coefficient pattern

        This enables vectorized computation over multiple shells simultaneously.
    )pbdoc")
        .def("angular_momentum", &ShellSet::angular_momentum,
             "Get the common angular momentum")
        .def("n_primitives", &ShellSet::n_primitives_per_shell,
             "Get the common number of primitives")
        .def("n_shells", &ShellSet::n_shells,
             "Get number of shells in this set")
        .def("n_functions_per_shell", &ShellSet::n_functions_per_shell,
             "Get number of basis functions per shell")
        .def("shell", [](const ShellSet& ss, Size i) -> const Shell& {
            return ss.shell(i);
        }, py::arg("index"), py::return_value_policy::reference,
           "Get shell by index within this set")
        .def("__len__", &ShellSet::n_shells)
        .def("__repr__", [](const ShellSet& ss) {
            std::string am_char(1, "SPDFGHI"[ss.angular_momentum()]);
            return "ShellSet(AM=" + am_char + ", K=" +
                   std::to_string(ss.n_primitives_per_shell()) + ", n_shells=" +
                   std::to_string(ss.n_shells()) + ")";
        });

    // ========================================================================
    // BasisSet class
    // Note: BasisSet contains unique_ptr<ShellSet> so is non-copyable.
    // We use shared_ptr as holder to avoid copy issues.
    // ========================================================================
    py::class_<BasisSet, std::shared_ptr<BasisSet>>(m, "BasisSet", R"pbdoc(
        Collection of Shell objects organized into ShellSets.

        BasisSet is the primary container for a molecular basis set. It:
          - Assigns sequential shell indices and basis function offsets
          - Groups shells into ShellSets by (angular_momentum, n_primitives)
          - Provides iteration infrastructure for integral computation

        Parameters
        ----------
        shells : list of Shell
            List of shells to include in the basis set.

        Examples
        --------
        >>> # Create basis set from shells
        >>> shells = [Shell(0, [0,0,0], [1.0], [1.0])]
        >>> basis = BasisSet(shells)
        >>> print(basis.n_shells(), basis.n_basis_functions())
    )pbdoc")
        .def(py::init<>(), "Create empty basis set")
        .def(py::init<std::vector<Shell>>(),
             py::arg("shells"),
             "Create basis set from list of shells")
        .def("n_shells", &BasisSet::n_shells,
             "Get total number of shells")
        .def("n_basis_functions", &BasisSet::n_basis_functions,
             "Get total number of basis functions")
        .def("max_angular_momentum", &BasisSet::max_angular_momentum,
             "Get maximum angular momentum")
        .def("max_n_primitives", &BasisSet::max_n_primitives,
             "Get maximum number of primitives")
        .def("shell", [](const BasisSet& bs, Size i) -> const Shell& {
            return bs.shell(i);
        }, py::arg("index"), py::return_value_policy::reference,
           "Get shell by index")
        .def("n_shell_sets", &BasisSet::n_shell_sets,
             "Get number of ShellSets")
        .def("shell_set", [](const BasisSet& bs, int am, int n_prim) {
            return bs.shell_set(am, n_prim);
        }, py::arg("am"), py::arg("n_primitives"),
           py::return_value_policy::reference,
           "Look up ShellSet by angular momentum and primitive count")
        .def("shell_set_pairs", [](const BasisSet& bs) {
            return bs.shell_set_pairs();
        }, py::return_value_policy::reference_internal,
           R"pbdoc(
        Get all unique ShellSetPairs (upper-triangle).

        Returns the cached worklist of ShellSetPairs. The first call
        computes and caches all pairs; subsequent calls return in O(1).

        Returns
        -------
        list of ShellSetPair
            All unique shell set pairs.
    )pbdoc")
        .def("shell_set_quartets", [](const BasisSet& bs) {
            return bs.shell_set_quartets();
        }, py::return_value_policy::reference_internal,
           R"pbdoc(
        Get all unique ShellSetQuartets (upper-triangle).

        Returns the cached worklist of ShellSetQuartets. The first call
        computes and caches all quartets; subsequent calls return in O(1).

        Returns
        -------
        list of ShellSetQuartet
            All unique shell set quartets.
    )pbdoc")
        .def("shells_on_atom", [](const BasisSet& bs, Index atom_idx) {
            auto ptrs = bs.shells_on_atom(atom_idx);
            py::list result;
            for (const auto* s : ptrs) {
                result.append(py::cast(*s, py::return_value_policy::reference));
            }
            return result;
        }, py::arg("atom_idx"),
           R"pbdoc(
        Get all shells centered on a given atom.

        Parameters
        ----------
        atom_idx : int
            Atom index.

        Returns
        -------
        list of Shell
            Shells on the specified atom.
    )pbdoc")
        .def("__len__", &BasisSet::n_shells)
        .def("__repr__", [](const BasisSet& bs) {
            return "BasisSet(n_shells=" + std::to_string(bs.n_shells()) +
                   ", n_basis=" + std::to_string(bs.n_basis_functions()) +
                   ", max_am=" + std::to_string(bs.max_angular_momentum()) + ")";
        });

    // ========================================================================
    // Built-in basis set creation
    // ========================================================================
    m.def("create_builtin_basis",
          [](const std::string& name, const std::vector<data::Atom>& atoms) 
              -> std::shared_ptr<BasisSet> {
              return std::make_shared<BasisSet>(data::create_builtin_basis(name, atoms));
          },
          py::arg("name"), py::arg("atoms"),
          R"pbdoc(
        Create a named built-in basis set.

        Currently only "sto-3g" is supported as a built-in.

        Parameters
        ----------
        name : str
            Basis set name (case-insensitive). E.g., "sto-3g".
        atoms : list of Atom
            Atoms with atomic numbers and positions.

        Returns
        -------
        BasisSet
            The constructed basis set.

        Raises
        ------
        RuntimeError
            If the basis set name is not recognized.
            If an element is not supported by the basis set.

        Examples
        --------
        >>> atoms = [Atom(8, [0,0,0]), Atom(1, [0,1.4,-1.1])]
        >>> basis = create_builtin_basis("sto-3g", atoms)
    )pbdoc");

    m.def("load_basis_set",
          [](const std::string& name, const std::vector<data::Atom>& atoms)
              -> std::shared_ptr<BasisSet> {
              return std::make_shared<BasisSet>(data::load_basis_set(name, atoms));
          },
          py::arg("name"), py::arg("atoms"),
          R"pbdoc(
        Load a bundled Basis Set Exchange (BSE) basis by name.

        Parameters
        ----------
        name : str
            Basis set name (case-insensitive). Examples: "6-31G*", "cc-pVDZ".
            Pople star notation is supported: ``*`` maps to polarization,
            ``**`` maps to double polarization.
        atoms : list of Atom
            Atoms with atomic numbers and positions.

        Returns
        -------
        BasisSet
            The constructed basis set.

        Raises
        ------
        RuntimeError
            If the basis set name is unknown or not available for an element.
    )pbdoc");

    m.def("list_available_basis_sets",
          &data::list_available_basis_sets,
          R"pbdoc(
        List all bundled basis sets available for loading.

        Scans the data directory for JSON basis set files and returns
        their stem names sorted alphabetically.

        Returns
        -------
        list of str
            Sorted list of basis set names (e.g., ["3-21g", "6-31g", ...]).
    )pbdoc");
}
