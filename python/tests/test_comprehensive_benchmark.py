# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

"""Comprehensive benchmark suite for libaccint Python bindings.

Benchmarks 1e integrals, 2e Fock builds, full SCF, and optionally compares
against PySCF timing.

Usage:
    PYTHONPATH=build/python-test/python:python python3 -m pytest python/tests/test_comprehensive_benchmark.py -v -s
    PYTHONPATH=build/python-test/python:python python3 -m pytest python/tests/test_comprehensive_benchmark.py -v -s -m "not slow"

The -s flag is important to see the printed timing tables.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pytest

import libaccint
from libaccint._core import (
    Atom,
    BasisSet,
    Engine,
    FockBuilder,
    Operator,
    ScreeningOptions,
    ScreeningPreset,
    create_builtin_basis,
)


# =============================================================================
# Timing utilities
# =============================================================================

def benchmark_operation(
    func: Callable,
    n_repeats: int = 5,
    n_warmup: int = 1,
    label: str = "",
) -> Dict[str, Any]:
    """Time a callable with warmup and multiple repeats.

    Parameters
    ----------
    func : callable
        Zero-argument callable to benchmark.
    n_repeats : int
        Number of timed repetitions.
    n_warmup : int
        Number of warmup calls (not timed).
    label : str
        Descriptive label for the benchmark.

    Returns
    -------
    dict with keys: label, mean, std, min, max, median, n, times
    """
    # Warmup
    for _ in range(n_warmup):
        func()

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times_arr = np.array(times)
    return {
        "label": label,
        "mean": float(np.mean(times_arr)),
        "std": float(np.std(times_arr)),
        "min": float(np.min(times_arr)),
        "max": float(np.max(times_arr)),
        "median": float(np.median(times_arr)),
        "n": n_repeats,
        "times": times,
    }


def format_time(seconds: float) -> str:
    """Format a time value with appropriate units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def print_benchmark_table(results: List[Dict], title: str = "") -> None:
    """Print a formatted benchmark results table."""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}")

    # Header
    print(f"{'Operation':<45} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'N':>4}")
    print("-" * 89)

    for r in results:
        print(
            f"{r['label']:<45} "
            f"{format_time(r['mean']):>10} "
            f"{format_time(r['std']):>10} "
            f"{format_time(r['min']):>10} "
            f"{format_time(r['max']):>10} "
            f"{r['n']:>4}"
        )
    print()


def print_comparison_table(
    lib_results: List[Dict],
    pyscf_results: List[Dict],
    title: str = "",
) -> None:
    """Print a side-by-side timing comparison table."""
    if title:
        print(f"\n{'=' * 90}")
        print(f"  {title}")
        print(f"{'=' * 90}")

    print(
        f"{'Operation':<35} "
        f"{'LibAccInt':>12} "
        f"{'PySCF':>12} "
        f"{'Ratio':>8} "
        f"{'Faster':>10}"
    )
    print("-" * 77)

    for lr, pr in zip(lib_results, pyscf_results):
        ratio = lr["mean"] / pr["mean"] if pr["mean"] > 0 else float("inf")
        if ratio < 1.0:
            faster = f"{1.0 / ratio:.1f}x lib"
        else:
            faster = f"{ratio:.1f}x pyscf"
        print(
            f"{lr['label']:<35} "
            f"{format_time(lr['mean']):>12} "
            f"{format_time(pr['mean']):>12} "
            f"{ratio:>8.2f} "
            f"{faster:>10}"
        )
    print()


# =============================================================================
# Molecule / basis data (mirrored from test_comprehensive_validation.py)
# =============================================================================

WATER_ATOMS_RAW = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 1.43233673, -1.10866041]),
    (1, [0.0, -1.43233673, -1.10866041]),
]

METHANE_ATOMS_RAW = [
    (6, [0.0, 0.0, 0.0]),
    (1, [1.18321596, 1.18321596, 1.18321596]),
    (1, [-1.18321596, -1.18321596, 1.18321596]),
    (1, [-1.18321596, 1.18321596, -1.18321596]),
    (1, [1.18321596, -1.18321596, -1.18321596]),
]

BOHR2ANG = 0.529177249


def _nuclear_repulsion(atoms_raw: list) -> float:
    e_nuc = 0.0
    for i in range(len(atoms_raw)):
        zi = atoms_raw[i][0]
        ri = np.array(atoms_raw[i][1])
        for j in range(i + 1, len(atoms_raw)):
            zj = atoms_raw[j][0]
            rj = np.array(atoms_raw[j][1])
            e_nuc += zi * zj / np.linalg.norm(ri - rj)
    return e_nuc


class BenchSpec(NamedTuple):
    mol_name: str
    basis_name: str
    basis_file: str
    atoms_raw: list
    n_electrons: int
    expected_nbf: int
    e_nuc: float


BENCH_SPECS = [
    BenchSpec("H2O", "sto-3g", "sto-3g.json", WATER_ATOMS_RAW,
              10, 7, _nuclear_repulsion(WATER_ATOMS_RAW)),
    BenchSpec("H2O", "6-31g", "6-31g.json", WATER_ATOMS_RAW,
              10, 13, _nuclear_repulsion(WATER_ATOMS_RAW)),
    BenchSpec("H2O", "aug-cc-pvdz", "aug-cc-pvdz.json", WATER_ATOMS_RAW,
              10, 41, _nuclear_repulsion(WATER_ATOMS_RAW)),
    BenchSpec("CH4", "sto-3g", "sto-3g.json", METHANE_ATOMS_RAW,
              10, 9, _nuclear_repulsion(METHANE_ATOMS_RAW)),
    BenchSpec("CH4", "6-31g", "6-31g.json", METHANE_ATOMS_RAW,
              10, 17, _nuclear_repulsion(METHANE_ATOMS_RAW)),
    BenchSpec("CH4", "aug-cc-pvdz", "aug-cc-pvdz.json", METHANE_ATOMS_RAW,
              10, 59, _nuclear_repulsion(METHANE_ATOMS_RAW)),
]

BENCH_IDS = [f"{s.mol_name}/{s.basis_name}" for s in BENCH_SPECS]

# STO-3G only (always available)
STO3G_BENCH_SPECS = [s for s in BENCH_SPECS if s.basis_name == "sto-3g"]
STO3G_BENCH_IDS = [f"{s.mol_name}/{s.basis_name}" for s in STO3G_BENCH_SPECS]


# =============================================================================
# Helpers
# =============================================================================

def _find_basis_file(filename: str) -> Optional[str]:
    candidates = [
        pathlib.Path(__file__).parent / ".." / ".." / "share" / "basis_sets" / filename,
        pathlib.Path("share") / "basis_sets" / filename,
        pathlib.Path("/home/westh/portfolio/programming/libaccint/share/basis_sets") / filename,
    ]
    project_root = os.environ.get("LIBACCINT_ROOT")
    if project_root:
        candidates.append(pathlib.Path(project_root) / "share" / "basis_sets" / filename)
    for p in candidates:
        resolved = p.resolve()
        if resolved.is_file():
            return str(resolved)
    return None


def _try_load_bse_basis(basis_file: str, atoms: List[Atom]) -> Optional[BasisSet]:
    path = _find_basis_file(basis_file)
    if path is None:
        return None
    try:
        from libaccint._core import BseJsonParser
        return BseJsonParser.parse_file(path, atoms)
    except (ImportError, AttributeError):
        pass
    try:
        from libaccint._core import parse_basis_file
        return parse_basis_file(path, atoms)
    except (ImportError, AttributeError):
        pass
    return None


def _make_atoms(atoms_raw: list) -> List[Atom]:
    return [Atom(z, pos) for z, pos in atoms_raw]


def _load_basis(spec: BenchSpec) -> Tuple[List[Atom], BasisSet]:
    atoms = _make_atoms(spec.atoms_raw)
    if spec.basis_name == "sto-3g":
        try:
            return atoms, create_builtin_basis("sto-3g", atoms)
        except RuntimeError:
            pass
    basis = _try_load_bse_basis(spec.basis_file, atoms)
    if basis is not None:
        return atoms, basis
    try:
        return atoms, create_builtin_basis(spec.basis_name, atoms)
    except RuntimeError:
        pytest.skip(f"Basis '{spec.basis_name}' not available")


def _run_rhf(
    S, T, V, n_electrons, E_nuc, engine, atoms,
    tol=1e-10, max_iter=200, use_screening=False,
):
    """Minimal RHF SCF for benchmarking."""
    from scipy.linalg import eigh as _eigh

    H = T + V
    nbf = S.shape[0]
    n_occ = n_electrons // 2

    s_vals, U = _eigh(S)
    mask = s_vals > 1e-10
    X = U[:, mask] @ np.diag(1.0 / np.sqrt(s_vals[mask])) @ U[:, mask].T

    if use_screening:
        engine.precompute_schwarz_bounds()

    Fp = X.T @ H @ X
    eps, Cp = _eigh(Fp)
    C = X @ Cp
    D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

    E_old = 0.0
    for iteration in range(max_iter):
        fb = FockBuilder(nbf)
        fb.set_density(D)
        if use_screening:
            engine.set_density_matrix(D, nbf)
            opts = ScreeningOptions.from_preset(ScreeningPreset.NORMAL)
            engine.compute_and_consume_screened(Operator.coulomb(), fb, opts)
        else:
            engine.compute_and_consume(Operator.coulomb(), fb)

        J = fb.get_coulomb_matrix()
        K = fb.get_exchange_matrix()
        F = H + J - 0.5 * K  # K from FockBuilder lacks the 1/2 factor

        E_elec = 0.5 * np.sum(D * (H + F))
        E_total = E_elec + E_nuc

        if abs(E_total - E_old) < tol and iteration > 0:
            return {"converged": True, "energy": E_total, "n_iter": iteration + 1}
        E_old = E_total

        Fp = X.T @ F @ X
        eps, Cp = _eigh(Fp)
        C = X @ Cp
        D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

    return {"converged": False, "energy": E_total, "n_iter": max_iter}


def _pyscf_available() -> bool:
    try:
        import pyscf  # noqa: F401
        return True
    except ImportError:
        return False


def _atoms_to_pyscf_string(atoms_raw: list) -> str:
    Z_TO_SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S"}
    parts = []
    for z, pos in atoms_raw:
        sym = Z_TO_SYM.get(z, f"X{z}")
        x_a, y_a, z_a = pos[0] * BOHR2ANG, pos[1] * BOHR2ANG, pos[2] * BOHR2ANG
        parts.append(f"{sym} {x_a:.12f} {y_a:.12f} {z_a:.12f}")
    return "; ".join(parts)


def _pyscf_basis_name(basis_name: str) -> str:
    return {"sto-3g": "sto-3g", "6-31g": "6-31g", "aug-cc-pvdz": "aug-cc-pvdz"}.get(
        basis_name, basis_name
    )


# =============================================================================
# Global results collector (populated during test session, dumped at end)
# =============================================================================

_ALL_RESULTS: Dict[str, Any] = {}


# =============================================================================
# Benchmarks: One-Electron Integrals
# =============================================================================

class TestBenchmarkOneElectron:
    """Benchmark 1e integral computation."""

    @pytest.mark.parametrize("spec", BENCH_SPECS, ids=BENCH_IDS)
    def test_bench_overlap(self, spec: BenchSpec):
        """Benchmark overlap matrix computation."""
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)

        r = benchmark_operation(
            engine.compute_overlap_matrix,
            n_repeats=10,
            label=f"S  {spec.mol_name}/{spec.basis_name} (nbf={spec.expected_nbf})",
        )
        _ALL_RESULTS[f"overlap_{spec.mol_name}_{spec.basis_name}"] = r
        print(f"\n  {r['label']}: {format_time(r['mean'])} +/- {format_time(r['std'])}")

    @pytest.mark.parametrize("spec", BENCH_SPECS, ids=BENCH_IDS)
    def test_bench_kinetic(self, spec: BenchSpec):
        """Benchmark kinetic energy matrix computation."""
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)

        r = benchmark_operation(
            engine.compute_kinetic_matrix,
            n_repeats=10,
            label=f"T  {spec.mol_name}/{spec.basis_name} (nbf={spec.expected_nbf})",
        )
        _ALL_RESULTS[f"kinetic_{spec.mol_name}_{spec.basis_name}"] = r
        print(f"\n  {r['label']}: {format_time(r['mean'])} +/- {format_time(r['std'])}")

    @pytest.mark.parametrize("spec", BENCH_SPECS, ids=BENCH_IDS)
    def test_bench_nuclear(self, spec: BenchSpec):
        """Benchmark nuclear attraction matrix computation."""
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)

        r = benchmark_operation(
            lambda: engine.compute_nuclear_matrix(atoms),
            n_repeats=10,
            label=f"V  {spec.mol_name}/{spec.basis_name} (nbf={spec.expected_nbf})",
        )
        _ALL_RESULTS[f"nuclear_{spec.mol_name}_{spec.basis_name}"] = r
        print(f"\n  {r['label']}: {format_time(r['mean'])} +/- {format_time(r['std'])}")

    @pytest.mark.parametrize("spec", BENCH_SPECS, ids=BENCH_IDS)
    def test_bench_core_hamiltonian(self, spec: BenchSpec):
        """Benchmark core Hamiltonian (T+V) computation."""
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)

        r = benchmark_operation(
            lambda: engine.compute_core_hamiltonian(atoms),
            n_repeats=10,
            label=f"H  {spec.mol_name}/{spec.basis_name} (nbf={spec.expected_nbf})",
        )
        _ALL_RESULTS[f"hcore_{spec.mol_name}_{spec.basis_name}"] = r
        print(f"\n  {r['label']}: {format_time(r['mean'])} +/- {format_time(r['std'])}")


# =============================================================================
# Benchmarks: Two-Electron Fock Build
# =============================================================================

class TestBenchmarkFockBuild:
    """Benchmark 2e Fock matrix construction."""

    @pytest.mark.parametrize("spec", BENCH_SPECS, ids=BENCH_IDS)
    def test_bench_fock_build(self, spec: BenchSpec):
        """Benchmark unscreened Fock build."""
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)
        nbf = basis.n_basis_functions()
        D = np.eye(nbf) / nbf

        def _build():
            fb = FockBuilder(nbf)
            fb.set_density(D)
            engine.compute_and_consume(Operator.coulomb(), fb)
            return fb.get_coulomb_matrix()

        r = benchmark_operation(
            _build,
            n_repeats=5,
            label=f"Fock(unscreened) {spec.mol_name}/{spec.basis_name}",
        )
        _ALL_RESULTS[f"fock_unscreened_{spec.mol_name}_{spec.basis_name}"] = r
        print(f"\n  {r['label']}: {format_time(r['mean'])} +/- {format_time(r['std'])}")


# =============================================================================
# Benchmarks: Screened vs Unscreened Fock Build
# =============================================================================

class TestBenchmarkScreening:
    """Compare screened vs unscreened Fock build timing."""

    @pytest.mark.parametrize("spec", BENCH_SPECS, ids=BENCH_IDS)
    def test_bench_screened_vs_unscreened(self, spec: BenchSpec):
        """Benchmark screened Fock build and compare to unscreened."""
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)
        nbf = basis.n_basis_functions()
        D = np.eye(nbf) / nbf

        # Unscreened
        def _build_unscreened():
            fb = FockBuilder(nbf)
            fb.set_density(D)
            engine.compute_and_consume(Operator.coulomb(), fb)
            return fb.get_coulomb_matrix()

        r_uns = benchmark_operation(
            _build_unscreened,
            n_repeats=5,
            label=f"Fock(unscr) {spec.mol_name}/{spec.basis_name}",
        )

        # Screened
        engine.precompute_schwarz_bounds()

        def _build_screened():
            engine.set_density_matrix(D, nbf)
            fb = FockBuilder(nbf)
            fb.set_density(D)
            opts = ScreeningOptions.from_preset(ScreeningPreset.NORMAL)
            engine.compute_and_consume_screened(Operator.coulomb(), fb, opts)
            return fb.get_coulomb_matrix()

        r_scr = benchmark_operation(
            _build_screened,
            n_repeats=5,
            label=f"Fock(screened) {spec.mol_name}/{spec.basis_name}",
        )

        _ALL_RESULTS[f"fock_unscr_{spec.mol_name}_{spec.basis_name}"] = r_uns
        _ALL_RESULTS[f"fock_scr_{spec.mol_name}_{spec.basis_name}"] = r_scr

        ratio = r_scr["mean"] / r_uns["mean"] if r_uns["mean"] > 0 else float("inf")
        speedup_str = f"{1.0 / ratio:.2f}x" if ratio < 1 else f"{ratio:.2f}x slower"

        print(
            f"\n  {spec.mol_name}/{spec.basis_name} (nbf={spec.expected_nbf}):"
            f"\n    Unscreened: {format_time(r_uns['mean'])}"
            f"\n    Screened:   {format_time(r_scr['mean'])}"
            f"\n    Screening:  {speedup_str}"
        )


# =============================================================================
# Benchmarks: Full RHF SCF
# =============================================================================

class TestBenchmarkSCF:
    """Benchmark full RHF SCF convergence."""

    @pytest.mark.parametrize("spec", BENCH_SPECS, ids=BENCH_IDS)
    def test_bench_scf(self, spec: BenchSpec):
        """Benchmark full SCF to convergence."""
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)

        S = engine.compute_overlap_matrix()
        T = engine.compute_kinetic_matrix()
        V = engine.compute_nuclear_matrix(atoms)

        def _run_scf():
            return _run_rhf(S, T, V, spec.n_electrons, spec.e_nuc, engine, atoms)

        r = benchmark_operation(
            _run_scf,
            n_repeats=3,
            n_warmup=1,
            label=f"SCF {spec.mol_name}/{spec.basis_name} (nbf={spec.expected_nbf})",
        )
        _ALL_RESULTS[f"scf_{spec.mol_name}_{spec.basis_name}"] = r

        # Also run once to get iteration count
        scf_result = _run_scf()
        print(
            f"\n  {r['label']}:"
            f"\n    Time:       {format_time(r['mean'])} +/- {format_time(r['std'])}"
            f"\n    Converged:  {scf_result['converged']}"
            f"\n    Iterations: {scf_result['n_iter']}"
            f"\n    Energy:     {scf_result['energy']:.10f}"
        )

    @pytest.mark.parametrize("spec", STO3G_BENCH_SPECS, ids=STO3G_BENCH_IDS)
    def test_bench_scf_screened(self, spec: BenchSpec):
        """Benchmark SCF with Schwarz screening."""
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)

        S = engine.compute_overlap_matrix()
        T = engine.compute_kinetic_matrix()
        V = engine.compute_nuclear_matrix(atoms)

        def _run_scf_screened():
            return _run_rhf(
                S, T, V, spec.n_electrons, spec.e_nuc, engine, atoms,
                use_screening=True,
            )

        r = benchmark_operation(
            _run_scf_screened,
            n_repeats=3,
            n_warmup=1,
            label=f"SCF(screened) {spec.mol_name}/{spec.basis_name}",
        )
        _ALL_RESULTS[f"scf_screened_{spec.mol_name}_{spec.basis_name}"] = r

        scf_result = _run_scf_screened()
        print(
            f"\n  {r['label']}:"
            f"\n    Time:       {format_time(r['mean'])} +/- {format_time(r['std'])}"
            f"\n    Converged:  {scf_result['converged']}"
            f"\n    Iterations: {scf_result['n_iter']}"
            f"\n    Energy:     {scf_result['energy']:.10f}"
        )


# =============================================================================
# Benchmarks: PySCF Comparison (slow)
# =============================================================================

@pytest.mark.slow
class TestBenchmarkPySCFComparison:
    """Compare libaccint timing against PySCF for identical operations.

    Requires PySCF. Run with: pytest -m slow
    """

    @pytest.fixture(autouse=True)
    def require_pyscf(self):
        if not _pyscf_available():
            pytest.skip("PySCF not installed")

    @pytest.mark.parametrize("spec", BENCH_SPECS, ids=BENCH_IDS)
    def test_bench_1e_vs_pyscf(self, spec: BenchSpec):
        """Compare 1e integral timing: libaccint vs PySCF."""
        from pyscf import gto

        # Setup libaccint
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)

        # Setup PySCF
        mol = gto.M(
            atom=_atoms_to_pyscf_string(spec.atoms_raw),
            basis=_pyscf_basis_name(spec.basis_name),
            unit="Angstrom",
        )

        # Benchmark overlap
        r_lib_s = benchmark_operation(
            engine.compute_overlap_matrix,
            n_repeats=10,
            label=f"S {spec.mol_name}/{spec.basis_name}",
        )
        r_pyscf_s = benchmark_operation(
            lambda: mol.intor("int1e_ovlp"),
            n_repeats=10,
            label=f"S {spec.mol_name}/{spec.basis_name}",
        )

        # Benchmark kinetic
        r_lib_t = benchmark_operation(
            engine.compute_kinetic_matrix,
            n_repeats=10,
            label=f"T {spec.mol_name}/{spec.basis_name}",
        )
        r_pyscf_t = benchmark_operation(
            lambda: mol.intor("int1e_kin"),
            n_repeats=10,
            label=f"T {spec.mol_name}/{spec.basis_name}",
        )

        # Benchmark nuclear
        r_lib_v = benchmark_operation(
            lambda: engine.compute_nuclear_matrix(atoms),
            n_repeats=10,
            label=f"V {spec.mol_name}/{spec.basis_name}",
        )
        r_pyscf_v = benchmark_operation(
            lambda: mol.intor("int1e_nuc"),
            n_repeats=10,
            label=f"V {spec.mol_name}/{spec.basis_name}",
        )

        _ALL_RESULTS[f"pyscf_cmp_S_{spec.mol_name}_{spec.basis_name}"] = {
            "lib": r_lib_s, "pyscf": r_pyscf_s,
        }
        _ALL_RESULTS[f"pyscf_cmp_T_{spec.mol_name}_{spec.basis_name}"] = {
            "lib": r_lib_t, "pyscf": r_pyscf_t,
        }
        _ALL_RESULTS[f"pyscf_cmp_V_{spec.mol_name}_{spec.basis_name}"] = {
            "lib": r_lib_v, "pyscf": r_pyscf_v,
        }

        print_comparison_table(
            [r_lib_s, r_lib_t, r_lib_v],
            [r_pyscf_s, r_pyscf_t, r_pyscf_v],
            title=f"1e Integrals: LibAccInt vs PySCF — {spec.mol_name}/{spec.basis_name}",
        )

    @pytest.mark.parametrize("spec", BENCH_SPECS, ids=BENCH_IDS)
    def test_bench_scf_vs_pyscf(self, spec: BenchSpec):
        """Compare full SCF timing: libaccint vs PySCF."""
        from pyscf import gto, scf

        # --- libaccint SCF ---
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)
        S = engine.compute_overlap_matrix()
        T = engine.compute_kinetic_matrix()
        V = engine.compute_nuclear_matrix(atoms)

        def _run_lib_scf():
            return _run_rhf(S, T, V, spec.n_electrons, spec.e_nuc, engine, atoms)

        r_lib = benchmark_operation(
            _run_lib_scf,
            n_repeats=3,
            n_warmup=1,
            label=f"SCF {spec.mol_name}/{spec.basis_name}",
        )

        # --- PySCF SCF ---
        mol = gto.M(
            atom=_atoms_to_pyscf_string(spec.atoms_raw),
            basis=_pyscf_basis_name(spec.basis_name),
            unit="Angstrom",
        )

        def _run_pyscf_scf():
            mf = scf.RHF(mol)
            mf.conv_tol = 1e-10
            mf.verbose = 0
            mf.kernel()
            return mf.e_tot

        r_pyscf = benchmark_operation(
            _run_pyscf_scf,
            n_repeats=3,
            n_warmup=1,
            label=f"SCF {spec.mol_name}/{spec.basis_name}",
        )

        _ALL_RESULTS[f"pyscf_cmp_scf_{spec.mol_name}_{spec.basis_name}"] = {
            "lib": r_lib, "pyscf": r_pyscf,
        }

        print_comparison_table(
            [r_lib],
            [r_pyscf],
            title=f"SCF: LibAccInt vs PySCF — {spec.mol_name}/{spec.basis_name}",
        )


# =============================================================================
# Summary Report (runs at end of session)
# =============================================================================

class TestBenchmarkSummary:
    """Print a summary of all benchmark results at the end."""

    def test_print_summary(self):
        """Print collected benchmark summary (runs last)."""
        if not _ALL_RESULTS:
            pytest.skip("No benchmark results collected")

        # Group results by category
        one_e = [v for k, v in _ALL_RESULTS.items()
                 if k.startswith(("overlap_", "kinetic_", "nuclear_", "hcore_"))
                 and isinstance(v, dict) and "label" in v]

        fock_uns = [v for k, v in _ALL_RESULTS.items()
                    if k.startswith("fock_unscreened_")
                    and isinstance(v, dict) and "label" in v]

        scf_res = [v for k, v in _ALL_RESULTS.items()
                   if k.startswith("scf_") and not k.startswith("scf_screened_")
                   and isinstance(v, dict) and "label" in v]

        if one_e:
            print_benchmark_table(one_e, "ONE-ELECTRON INTEGRALS")
        if fock_uns:
            print_benchmark_table(fock_uns, "TWO-ELECTRON FOCK BUILD (UNSCREENED)")
        if scf_res:
            print_benchmark_table(scf_res, "FULL RHF SCF")

        # Save to JSON if output dir exists or requested
        output_path = os.environ.get(
            "LIBACCINT_BENCH_OUTPUT",
            "python/tests/benchmark_results.json",
        )
        try:
            # Strip non-serialisable entries (numpy arrays, etc.)
            serialisable = {}
            for k, v in _ALL_RESULTS.items():
                if isinstance(v, dict):
                    clean = {}
                    for vk, vv in v.items():
                        if isinstance(vv, (int, float, str, bool, list)):
                            clean[vk] = vv
                        elif isinstance(vv, dict):
                            inner = {}
                            for ik, iv in vv.items():
                                if isinstance(iv, (int, float, str, bool, list)):
                                    inner[ik] = iv
                            clean[vk] = inner
                    serialisable[k] = clean

            with open(output_path, "w") as f:
                json.dump(serialisable, f, indent=2)
            print(f"\nBenchmark results saved to: {output_path}")
        except (OSError, TypeError) as e:
            print(f"\nCould not save benchmark results: {e}")
