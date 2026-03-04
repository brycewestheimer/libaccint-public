// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file screening_bindings.cpp
/// @brief Python bindings for screening module (Task 8.4.2)

#include <libaccint/screening/screening_options.hpp>
#include <libaccint/screening/schwarz_bounds.hpp>
#include <libaccint/screening/density_screening.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

namespace py = pybind11;

using namespace libaccint;
using namespace libaccint::screening;

void bind_screening(py::module_& m) {
    // =========================================================================
    // ScreeningPreset enum
    // =========================================================================
    py::enum_<ScreeningPreset>(m, "ScreeningPreset",
        R"pbdoc(Preset levels for screening thresholds.)pbdoc")
        .value("NONE", ScreeningPreset::None, "No screening (all quartets computed)")
        .value("LOOSE", ScreeningPreset::Loose, "1e-10 threshold, faster but less accurate")
        .value("NORMAL", ScreeningPreset::Normal, "1e-12 threshold, good balance (default)")
        .value("TIGHT", ScreeningPreset::Tight, "1e-14 threshold, maximum accuracy")
        .value("CUSTOM", ScreeningPreset::Custom, "User-specified threshold");

    // =========================================================================
    // ScreeningOptions struct
    // =========================================================================
    py::class_<ScreeningOptions>(m, "ScreeningOptions",
        R"pbdoc(
        Configuration options for Schwarz screening.

        Controls how screening is applied during integral computation.
        The Schwarz inequality states: |(ab|cd)| <= Q_ab * Q_cd
        Quartets where Q_ab * Q_cd < threshold are skipped.

        Usage::

            opts = ScreeningOptions.normal()
            opts.density_weighted = True
            engine.compute_and_consume(fock, density, opts)
        )pbdoc")
        .def(py::init<>(), "Create default screening options (Normal preset)")
        .def_readwrite("threshold", &ScreeningOptions::threshold,
            "Schwarz screening threshold. Lower = tighter = more accuracy but less speedup.")
        .def_readwrite("enabled", &ScreeningOptions::enabled,
            "Whether screening is enabled. When False, all quartets are computed.")
        .def_readwrite("density_weighted", &ScreeningOptions::density_weighted,
            "Whether to use density-weighted screening (tighter bounds during SCF).")
        .def_readwrite("use_permutation_symmetry", &ScreeningOptions::use_permutation_symmetry,
            "Whether to exploit 8-fold permutation symmetry (reduces computation up to 8x).")
        .def_readwrite("enable_statistics", &ScreeningOptions::enable_statistics,
            "Whether to collect screening statistics.")
        .def_readwrite("verbosity", &ScreeningOptions::verbosity,
            "Verbosity level: 0=silent, 1=summary, 2=detailed.")
        .def_static("none", &ScreeningOptions::none,
            "Create options with no screening (compute all quartets).")
        .def_static("loose", &ScreeningOptions::loose,
            "Create options with loose screening (1e-10).")
        .def_static("normal", &ScreeningOptions::normal,
            "Create options with normal screening (1e-12, default).")
        .def_static("tight", &ScreeningOptions::tight,
            "Create options with tight screening (1e-14).")
        .def_static("from_preset", &ScreeningOptions::from_preset,
            py::arg("preset"),
            "Create options from a ScreeningPreset.")
        .def("effective_threshold", &ScreeningOptions::effective_threshold,
            "Get the effective threshold (0 if screening disabled).")
        .def("passes_screening", &ScreeningOptions::passes_screening,
            py::arg("schwarz_product"),
            "Check if a Schwarz bound product passes screening.")
        .def("preset_name", &ScreeningOptions::preset_name,
            "Get the preset name as a string.")
        .def("validate", &ScreeningOptions::validate,
            "Validate options and print warnings for extreme values.")
        .def("__repr__", [](const ScreeningOptions& opts) {
            std::ostringstream oss;
            oss << "ScreeningOptions(preset=" << opts.preset_name()
                << ", threshold=" << opts.threshold
                << ", enabled=" << (opts.enabled ? "True" : "False")
                << ", density_weighted=" << (opts.density_weighted ? "True" : "False")
                << ", use_permutation_symmetry=" << (opts.use_permutation_symmetry ? "True" : "False")
                << ")";
            return oss.str();
        });

    // =========================================================================
    // ScreeningStatistics struct
    // =========================================================================
    py::class_<ScreeningStatistics>(m, "ScreeningStatistics",
        R"pbdoc(Statistics about screening effectiveness.)pbdoc")
        .def(py::init<>(), "Create empty screening statistics.")
        .def_readwrite("total_quartets", &ScreeningStatistics::total_quartets,
            "Total number of unique quartets considered.")
        .def_readwrite("computed_quartets", &ScreeningStatistics::computed_quartets,
            "Number of quartets that were computed.")
        .def_readwrite("skipped_quartets", &ScreeningStatistics::skipped_quartets,
            "Number of quartets that were screened out.")
        .def("efficiency", &ScreeningStatistics::efficiency,
            "Compute screening efficiency (0.0 to 1.0).")
        .def("skip_percentage", &ScreeningStatistics::skip_percentage,
            "Compute percentage of quartets skipped.")
        .def("reset", &ScreeningStatistics::reset,
            "Reset all counters to zero.")
        .def("__repr__", [](const ScreeningStatistics& stats) {
            std::ostringstream oss;
            oss << "ScreeningStatistics(total=" << stats.total_quartets
                << ", computed=" << stats.computed_quartets
                << ", skipped=" << stats.skipped_quartets
                << ", efficiency=" << (stats.efficiency() * 100.0) << "%)";
            return oss.str();
        });

    // =========================================================================
    // SchwarzBounds class (read-only query interface)
    // =========================================================================
    py::class_<SchwarzBounds>(m, "SchwarzBounds",
        R"pbdoc(
        Precomputed Schwarz bounds for shell pairs.

        Stores Q[i][j] = sqrt(max |(ij|ij)|) for O(1) lookup.
        Construct from a BasisSet, then query bounds and screening.
        )pbdoc")
        .def(py::init<const BasisSet&>(), py::arg("basis"),
            "Compute Schwarz bounds for a basis set.")
        .def("__call__", &SchwarzBounds::operator(),
            py::arg("i"), py::arg("j"),
            "Look up Schwarz bound Q_ij for shell pair (i, j).")
        .def("n_shells", &SchwarzBounds::n_shells,
            "Get the number of shells.")
        .def("passes_screening", &SchwarzBounds::passes_screening,
            py::arg("i"), py::arg("j"), py::arg("k"), py::arg("l"),
            py::arg("threshold"),
            "Check if a quartet passes Schwarz screening.")
        .def("quartet_bound", &SchwarzBounds::quartet_bound,
            py::arg("i"), py::arg("j"), py::arg("k"), py::arg("l"),
            "Get the Schwarz bound product Q_ij * Q_kl.")
        .def("max_bound", &SchwarzBounds::max_bound,
            "Get the maximum Schwarz bound.")
        .def("count_passing_quartets", &SchwarzBounds::count_passing_quartets,
            py::arg("threshold"),
            "Count unique quartets that pass screening at threshold.")
        .def("estimate_pass_fraction", &SchwarzBounds::estimate_pass_fraction,
            py::arg("threshold"),
            "Estimate the fraction of quartets that pass screening.")
        .def("__repr__", [](const SchwarzBounds& b) {
            std::ostringstream oss;
            oss << "SchwarzBounds(n_shells=" << b.n_shells()
                << ", max_bound=" << b.max_bound() << ")";
            return oss.str();
        });
}
