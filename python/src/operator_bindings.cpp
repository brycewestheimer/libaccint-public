// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file operator_bindings.cpp
/// @brief pybind11 bindings for operator types

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <libaccint/operators/operator_types.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/operators/one_electron_operator.hpp>

namespace py = pybind11;

void bind_operators(py::module_& m) {
    using namespace libaccint;

    // ========================================================================
    // OperatorKind enum
    // ========================================================================
    py::enum_<OperatorKind>(m, "OperatorKind", R"pbdoc(
        Enumeration of supported integral operators.

        Includes both one-electron and two-electron operators.
    )pbdoc")
        // One-electron operators
        .value("Overlap", OperatorKind::Overlap, "Overlap operator (no parameters)")
        .value("Kinetic", OperatorKind::Kinetic, "Kinetic energy operator")
        .value("Nuclear", OperatorKind::Nuclear, "Nuclear attraction operator")
        .value("PointCharge", OperatorKind::PointCharge, "Point charge interaction")
        // Two-electron operators
        .value("Coulomb", OperatorKind::Coulomb, "Coulomb operator 1/r12")
        .value("ErfCoulomb", OperatorKind::ErfCoulomb, "Short-range erf(ωr12)/r12")
        .value("ErfcCoulomb", OperatorKind::ErfcCoulomb, "Long-range erfc(ωr12)/r12")
        // Advanced one-electron operators
        .value("DistributedMultipole", OperatorKind::DistributedMultipole, "Distributed multipole operator")
        .value("ProjectionOperator", OperatorKind::ProjectionOperator, "Projection operator")
        // Property integral operators
        .value("ElectricDipole", OperatorKind::ElectricDipole, "Electric dipole moment (3-component)")
        .value("ElectricQuadrupole", OperatorKind::ElectricQuadrupole, "Electric quadrupole moment (6-component)")
        .value("ElectricOctupole", OperatorKind::ElectricOctupole, "Electric octupole moment (10-component)")
        .value("LinearMomentum", OperatorKind::LinearMomentum, "Linear momentum (3-component, anti-Hermitian)")
        .value("AngularMomentum", OperatorKind::AngularMomentum, "Angular momentum (3-component, anti-Hermitian)");

    // ========================================================================
    // PointChargeParams struct
    // ========================================================================
    py::class_<PointChargeParams>(m, "PointChargeParams", R"pbdoc(
        Parameters for nuclear attraction or point charge operators.

        Structure-of-Arrays layout containing positions (x, y, z) and charges
        for each charge center.

        Parameters
        ----------
        x : array-like
            X-coordinates of charge centers.
        y : array-like
            Y-coordinates of charge centers.
        z : array-like
            Z-coordinates of charge centers.
        charges : array-like
            Charges (atomic numbers for nuclear, arbitrary for point charges).

        Examples
        --------
        >>> # Water nuclear charges
        >>> params = PointChargeParams(
        ...     x=[0.0, 0.0, 0.0],
        ...     y=[0.0, 1.43, -1.43],
        ...     z=[0.0, -1.11, -1.11],
        ...     charges=[8.0, 1.0, 1.0]
        ... )
    )pbdoc")
        .def(py::init<>(), "Create empty point charge parameters")
        .def(py::init([](std::vector<Real> x, std::vector<Real> y,
                         std::vector<Real> z, std::vector<Real> charges) {
            PointChargeParams params;
            params.x = std::move(x);
            params.y = std::move(y);
            params.z = std::move(z);
            params.charge = std::move(charges);
            return params;
        }),
             py::arg("x"), py::arg("y"), py::arg("z"), py::arg("charges"),
             "Create from coordinate and charge arrays")
        .def_readwrite("x", &PointChargeParams::x, "X-coordinates")
        .def_readwrite("y", &PointChargeParams::y, "Y-coordinates")
        .def_readwrite("z", &PointChargeParams::z, "Z-coordinates")
        .def_property("charges",
            [](const PointChargeParams& p) { return p.charge; },
            [](PointChargeParams& p, std::vector<Real> c) { p.charge = std::move(c); },
            "Charges")
        .def("n_centers", &PointChargeParams::n_centers,
             "Number of charge centers")
        .def("__len__", &PointChargeParams::n_centers)
        .def("__repr__", [](const PointChargeParams& p) {
            return "PointChargeParams(n_centers=" +
                   std::to_string(p.n_centers()) + ")";
        });

    // ========================================================================
    // RangeSeparatedParams struct
    // ========================================================================
    py::class_<RangeSeparatedParams>(m, "RangeSeparatedParams", R"pbdoc(
        Parameters for range-separated Coulomb operators.

        Contains the range-separation parameter omega.

        Parameters
        ----------
        omega : float
            Range-separation parameter ω in atomic units.
    )pbdoc")
        .def(py::init<>(), "Create with omega=0")
        .def(py::init([](Real omega) {
            return RangeSeparatedParams{omega};
        }), py::arg("omega"), "Create with specified omega")
        .def_readwrite("omega", &RangeSeparatedParams::omega,
                       "Range-separation parameter ω")
        .def("__repr__", [](const RangeSeparatedParams& p) {
            return "RangeSeparatedParams(omega=" +
                   std::to_string(p.omega) + ")";
        });

    // ========================================================================
    // OriginParams struct
    // ========================================================================
    py::class_<OriginParams>(m, "OriginParams", R"pbdoc(
        Parameters for origin-dependent property integrals.

        Contains the gauge/expansion origin for dipole, quadrupole,
        momentum, and other property integrals.

        Parameters
        ----------
        origin : array-like of 3 floats
            The gauge/expansion origin in atomic units. Default: [0, 0, 0].
    )pbdoc")
        .def(py::init<>(), "Create with default origin (0, 0, 0)")
        .def(py::init([](py::array_t<Real> origin) {
            if (origin.size() != 3) {
                throw std::invalid_argument("Origin must have exactly 3 elements");
            }
            auto buf = origin.unchecked<1>();
            OriginParams p;
            p.origin = {buf(0), buf(1), buf(2)};
            return p;
        }), py::arg("origin"),
             "Create with specified origin")
        .def_readwrite("origin", &OriginParams::origin, "Gauge/expansion origin")
        .def("__repr__", [](const OriginParams& p) {
            return "OriginParams(origin=[" +
                   std::to_string(p.origin[0]) + ", " +
                   std::to_string(p.origin[1]) + ", " +
                   std::to_string(p.origin[2]) + "])";
        });

    // ========================================================================
    // DistributedMultipoleParams struct
    // ========================================================================
    py::class_<DistributedMultipoleParams>(m, "DistributedMultipoleParams", R"pbdoc(
        Parameters for distributed multipole operators.

        Stores multipole site positions and moment components
        (charges, dipoles, quadrupoles) in Structure-of-Arrays layout.
    )pbdoc")
        .def(py::init<>(), "Create empty parameters")
        .def_readwrite("x", &DistributedMultipoleParams::x, "X-coordinates of sites")
        .def_readwrite("y", &DistributedMultipoleParams::y, "Y-coordinates of sites")
        .def_readwrite("z", &DistributedMultipoleParams::z, "Z-coordinates of sites")
        .def_readwrite("charges", &DistributedMultipoleParams::charges, "Site charges")
        .def_readwrite("dipole_x", &DistributedMultipoleParams::dipole_x, "Dipole x-components")
        .def_readwrite("dipole_y", &DistributedMultipoleParams::dipole_y, "Dipole y-components")
        .def_readwrite("dipole_z", &DistributedMultipoleParams::dipole_z, "Dipole z-components")
        .def_readwrite("quad_xx", &DistributedMultipoleParams::quad_xx, "Quadrupole xx-components")
        .def_readwrite("quad_xy", &DistributedMultipoleParams::quad_xy, "Quadrupole xy-components")
        .def_readwrite("quad_xz", &DistributedMultipoleParams::quad_xz, "Quadrupole xz-components")
        .def_readwrite("quad_yy", &DistributedMultipoleParams::quad_yy, "Quadrupole yy-components")
        .def_readwrite("quad_yz", &DistributedMultipoleParams::quad_yz, "Quadrupole yz-components")
        .def_readwrite("quad_zz", &DistributedMultipoleParams::quad_zz, "Quadrupole zz-components")
        .def("n_sites", &DistributedMultipoleParams::n_sites, "Number of multipole sites")
        .def("max_rank", &DistributedMultipoleParams::max_rank,
             "Maximum multipole rank (0=charge, 1=dipole, 2=quadrupole)")
        .def("is_valid", &DistributedMultipoleParams::is_valid,
             "Check internal consistency");

    // ========================================================================
    // ProjectionOperatorParams struct
    // ========================================================================
    py::class_<ProjectionOperatorParams>(m, "ProjectionOperatorParams", R"pbdoc(
        Parameters for projection operators.

        P = C * diag(w) * C^T where C is the coefficient matrix
        and w are the weights.
    )pbdoc")
        .def(py::init<>(), "Create empty parameters")
        .def_readwrite("coefficients", &ProjectionOperatorParams::coefficients,
                       "Flattened coefficient matrix (n_basis x n_projectors, column-major)")
        .def_readwrite("weights", &ProjectionOperatorParams::weights,
                       "Weight for each projector function")
        .def_readwrite("n_basis", &ProjectionOperatorParams::n_basis,
                       "Number of basis functions")
        .def_readwrite("n_projectors", &ProjectionOperatorParams::n_projectors,
                       "Number of projector functions")
        .def("is_valid", &ProjectionOperatorParams::is_valid,
             "Check internal consistency");

    // ========================================================================
    // Operator class
    // ========================================================================
    py::class_<Operator>(m, "Operator", R"pbdoc(
        Quantum chemical operator for integral computation.

        Operators are created using factory methods (e.g., overlap(), kinetic(),
        coulomb()). They encapsulate operator type and any associated parameters.

        Examples
        --------
        >>> S = Operator.overlap()   # Overlap operator
        >>> T = Operator.kinetic()   # Kinetic energy
        >>> V = Operator.nuclear(params)  # Nuclear attraction
        >>> J = Operator.coulomb()   # Electron repulsion
    )pbdoc")
        // Factory methods
        .def_static("overlap", &Operator::overlap,
                    "Create overlap operator (S)")
        .def_static("kinetic", &Operator::kinetic,
                    "Create kinetic energy operator (T)")
        .def_static("coulomb", &Operator::coulomb,
                    "Create Coulomb operator (1/r12)")
        .def_static("nuclear", &Operator::nuclear,
                    py::arg("params"),
                    "Create nuclear attraction operator with point charges")
        .def_static("point_charges", &Operator::point_charges,
                    py::arg("params"),
                    "Create point charge operator")
        .def_static("erf_coulomb", &Operator::erf_coulomb,
                    py::arg("omega"),
                    "Create erf-attenuated Coulomb operator erf(ωr12)/r12")
        .def_static("erfc_coulomb", &Operator::erfc_coulomb,
                    py::arg("omega"),
                    "Create erfc-attenuated Coulomb operator erfc(ωr12)/r12")
        // Property integral factory methods
        .def_static("linear_momentum", &Operator::linear_momentum,
                    py::arg("params") = OriginParams{},
                    "Create linear momentum operator -i∇ (3-component)")
        .def_static("angular_momentum", &Operator::angular_momentum,
                    py::arg("params") = OriginParams{},
                    "Create angular momentum operator r×(-i∇) (3-component)")
        .def_static("electric_dipole", &Operator::electric_dipole,
                    py::arg("params") = OriginParams{},
                    "Create electric dipole moment operator (3-component)")
        .def_static("electric_quadrupole", &Operator::electric_quadrupole,
                    py::arg("params") = OriginParams{},
                    "Create electric quadrupole moment operator (6-component)")
        .def_static("electric_octupole", &Operator::electric_octupole,
                    py::arg("params") = OriginParams{},
                    "Create electric octupole moment operator (10-component)")
        .def_static("distributed_multipole", &Operator::distributed_multipole,
                    py::arg("params"),
                    "Create distributed multipole operator")
        .def_static("projection", &Operator::projection,
                    py::arg("params"),
                    "Create projection operator")
        // Accessors
        .def("kind", &Operator::kind, "Get the operator kind")
        .def("is_one_electron", &Operator::is_one_electron,
             "Check if this is a one-electron operator")
        .def("is_two_electron", &Operator::is_two_electron,
             "Check if this is a two-electron operator")
        .def("__repr__", [](const Operator& op) {
            return "Operator(" + std::string(to_string(op.kind())) + ")";
        });

    // ========================================================================
    // OneElectronOperator class
    // ========================================================================
    py::class_<OneElectronOperator>(m, "OneElectronOperator", R"pbdoc(
        Composed one-electron operator for computing multiple integrals.

        Allows combining multiple operators (e.g., T + V) to compute the
        core Hamiltonian in a single pass.

        Parameters
        ----------
        op : Operator
            Initial one-electron operator.

        Examples
        --------
        >>> h_core = OneElectronOperator(Operator.kinetic())
        >>> h_core.add(Operator.nuclear(params))  # H = T + V
    )pbdoc")
        .def(py::init<Operator>(), py::arg("op"),
             "Create from a single operator")
        .def("add", &OneElectronOperator::add,
             py::arg("op"), py::arg("scale") = 1.0,
             "Add another operator to compose with optional scale factor")
        .def("n_contributions", &OneElectronOperator::n_contributions,
             "Get number of operators in composition")
        .def("__add__", [](const OneElectronOperator& a, const OneElectronOperator& b) {
            return a + b;
        }, "Combine two OneElectronOperators")
        .def("__mul__", [](const OneElectronOperator& a, Real s) {
            return a * s;
        }, "Scale operator by a factor")
        .def("__rmul__", [](const OneElectronOperator& a, Real s) {
            return s * a;
        }, "Scale operator by a factor (left multiply)")
        .def("__repr__", [](const OneElectronOperator& op) {
            return "OneElectronOperator(n_ops=" +
                   std::to_string(op.n_contributions()) + ")";
        });

    // Classification utility functions
    m.def("is_one_electron", &is_one_electron,
          py::arg("kind"),
          "Check if operator kind is one-electron");
    m.def("is_two_electron", &is_two_electron,
          py::arg("kind"),
          "Check if operator kind is two-electron");
    m.def("is_multi_component", &is_multi_component,
          py::arg("kind"),
          "Check if operator kind produces multi-component integrals");
    m.def("is_anti_hermitian", &is_anti_hermitian,
          py::arg("kind"),
          "Check if operator kind produces anti-Hermitian matrices");
    m.def("is_property_integral", &is_property_integral,
          py::arg("kind"),
          "Check if operator kind is a property integral");
    m.def("component_count", &component_count,
          py::arg("kind"),
          "Return number of integral components for the operator kind");
}
