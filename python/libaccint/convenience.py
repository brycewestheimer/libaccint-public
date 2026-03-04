# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

"""
Convenience API for LibAccInt.

Provides high-level, Pythonic functions for common operations.
"""

from typing import List, Optional, Union, Sequence
import numpy as np
from numpy.typing import NDArray

from ._core import (
    Atom as _Atom,
    BasisSet as _BasisSet,
    Engine as _Engine,
    Operator,
    PointChargeParams,
    OneElectronOperator,
    FockBuilder as _FockBuilder,
)


def basis_set(
    name: str,
    atoms: List[_Atom],
) -> _BasisSet:
    """
    Create a basis set for the given atoms.

    Supports 40+ bundled basis sets from the Basis Set Exchange, including:

    - **Pople:** STO-3G, STO-6G, 3-21G, 6-31G, 6-31G*, 6-31G**, 6-311G, ...
    - **Dunning:** cc-pVDZ, cc-pVTZ, cc-pVQZ, aug-cc-pVDZ, ...
    - **Karlsruhe def2:** def2-SVP, def2-TZVP, def2-QZVP, ...
    - **Auxiliary:** cc-pVTZ-JKFIT, def2-universal-jkfit, cc-pVDZ-RIFIT, ...

    Names are case-insensitive. Pople star notation is supported:
    ``"6-31G*"`` and ``"6-31G**"`` are resolved automatically.

    Use :func:`list_available_basis_sets` to see all available names.

    Parameters
    ----------
    name : str
        Basis set name (case-insensitive). Examples: ``"6-31G*"``,
        ``"cc-pVDZ"``, ``"def2-SVP"``.
    atoms : List[Atom]
        List of atoms with atomic numbers and positions.

    Returns
    -------
    BasisSet
        The constructed basis set.

    Examples
    --------
    >>> atoms = [Atom(8, [0, 0, 0]), Atom(1, [0, 1.43, -1.11])]
    >>> basis = basis_set("6-31G*", atoms)
    >>> print(basis.n_basis_functions())
    """
    from ._core import create_builtin_basis
    try:
        return create_builtin_basis(name, atoms)
    except Exception:
        # Fall back to bundled Basis Set Exchange (BSE) basis data when available.
        from ._core import load_basis_set
        return load_basis_set(name, atoms)


def list_available_basis_sets() -> List[str]:
    """
    List all bundled basis sets available for loading.

    Returns a sorted list of basis set stem names that can be passed
    to :func:`basis_set` or ``load_basis_set``.

    Returns
    -------
    List[str]
        Sorted list of basis set names (e.g., ``["3-21g", "6-31g", ...]``).

    Examples
    --------
    >>> names = list_available_basis_sets()
    >>> "sto-3g" in names
    True
    """
    from ._core import list_available_basis_sets as _list
    return _list()


def compute_overlap(
    basis_or_engine: Union[_BasisSet, _Engine],
) -> NDArray[np.float64]:
    """
    Compute the overlap matrix S.

    Parameters
    ----------
    basis_or_engine : BasisSet or Engine
        Either a BasisSet (Engine will be created) or an existing Engine.

    Returns
    -------
    numpy.ndarray
        Overlap matrix of shape (n_basis, n_basis).

    Examples
    --------
    >>> S = compute_overlap(basis)
    >>> print(S.shape)
    """
    if isinstance(basis_or_engine, _Engine):
        engine = basis_or_engine
    else:
        engine = _Engine(basis_or_engine)

    return engine.compute_overlap_matrix()


def compute_kinetic(
    basis_or_engine: Union[_BasisSet, _Engine],
) -> NDArray[np.float64]:
    """
    Compute the kinetic energy matrix T.

    Parameters
    ----------
    basis_or_engine : BasisSet or Engine
        Either a BasisSet (Engine will be created) or an existing Engine.

    Returns
    -------
    numpy.ndarray
        Kinetic energy matrix of shape (n_basis, n_basis).

    Examples
    --------
    >>> T = compute_kinetic(basis)
    >>> print(T.shape)
    """
    if isinstance(basis_or_engine, _Engine):
        engine = basis_or_engine
    else:
        engine = _Engine(basis_or_engine)

    return engine.compute_kinetic_matrix()


def compute_nuclear(
    basis_or_engine: Union[_BasisSet, _Engine],
    atoms: List[_Atom],
) -> NDArray[np.float64]:
    """
    Compute the nuclear attraction matrix V.

    Parameters
    ----------
    basis_or_engine : BasisSet or Engine
        Either a BasisSet (Engine will be created) or an existing Engine.
    atoms : List[Atom]
        List of atoms defining nuclear charges and positions.

    Returns
    -------
    numpy.ndarray
        Nuclear attraction matrix of shape (n_basis, n_basis).

    Examples
    --------
    >>> V = compute_nuclear(basis, atoms)
    >>> print(V.shape)
    """
    if isinstance(basis_or_engine, _Engine):
        engine = basis_or_engine
    else:
        engine = _Engine(basis_or_engine)

    return engine.compute_nuclear_matrix(atoms)


def compute_core_hamiltonian(
    basis_or_engine: Union[_BasisSet, _Engine],
    atoms: List[_Atom],
) -> NDArray[np.float64]:
    """
    Compute the core Hamiltonian H = T + V.

    Parameters
    ----------
    basis_or_engine : BasisSet or Engine
        Either a BasisSet (Engine will be created) or an existing Engine.
    atoms : List[Atom]
        List of atoms defining nuclear charges and positions.

    Returns
    -------
    numpy.ndarray
        Core Hamiltonian matrix of shape (n_basis, n_basis).

    Examples
    --------
    >>> H = compute_core_hamiltonian(basis, atoms)
    >>> # Equivalent to:
    >>> # H = compute_kinetic(basis) + compute_nuclear(basis, atoms)
    """
    if isinstance(basis_or_engine, _Engine):
        engine = basis_or_engine
    else:
        engine = _Engine(basis_or_engine)

    return engine.compute_core_hamiltonian(atoms)


def build_fock(
    engine: _Engine,
    density: NDArray[np.float64],
    core_hamiltonian: Optional[NDArray[np.float64]] = None,
    exchange_fraction: float = 1.0,
) -> NDArray[np.float64]:
    """
    Build the Fock matrix from density matrix.

    Computes F = H_core + J - exchange_fraction * K
    where J is the Coulomb matrix and K is the exchange matrix.

    Parameters
    ----------
    engine : Engine
        The integral engine.
    density : numpy.ndarray
        Density matrix of shape (n_basis, n_basis).
    core_hamiltonian : numpy.ndarray, optional
        Core Hamiltonian matrix. If not provided, only J - x*K is returned.
    exchange_fraction : float, default=1.0
        Fraction of exact exchange (1.0 for HF, 0.0 for pure DFT).

    Returns
    -------
    numpy.ndarray
        Fock matrix of shape (n_basis, n_basis).

    Examples
    --------
    >>> D = np.eye(7) * 0.1  # Dummy density
    >>> H = compute_core_hamiltonian(basis, atoms)
    >>> F = build_fock(engine, D, H)
    """
    nbf = engine.basis().n_basis_functions()

    if density.shape != (nbf, nbf):
        raise ValueError(
            f"Density matrix shape {density.shape} doesn't match "
            f"basis size ({nbf}, {nbf})"
        )

    # Ensure density is contiguous and C-order
    density_c = np.ascontiguousarray(density, dtype=np.float64)

    # Build Fock matrix using FockBuilder
    fock_builder = _FockBuilder(nbf)
    fock_builder.set_density(density_c)

    # Compute two-electron integrals and accumulate via unified compute()
    engine.compute(Operator.coulomb(), fock_builder)

    # Get J and K matrices
    J = fock_builder.get_coulomb_matrix()
    K = fock_builder.get_exchange_matrix()

    # Build Fock matrix
    if core_hamiltonian is not None:
        F = core_hamiltonian + J - exchange_fraction * K
    else:
        F = J - exchange_fraction * K

    return F


def compute_eri_tensor(
    engine: _Engine,
    op: Optional["Operator"] = None,
) -> NDArray[np.float64]:
    """Compute the full ERI tensor as a 4D NumPy array.

    Parameters
    ----------
    engine : Engine
        The integral engine.
    op : Operator, optional
        Two-electron operator. Defaults to Coulomb.

    Returns
    -------
    numpy.ndarray
        4D array of shape (nbf, nbf, nbf, nbf).
    """
    from ._core import Operator
    if op is None:
        op = Operator.coulomb()
    return engine.compute_eri_tensor(op)


def compute_eri_block(
    engine: _Engine,
    quartet: "ShellSetQuartet",
    op: Optional["Operator"] = None,
) -> NDArray[np.float64]:
    """Compute ERIs for a single ShellSetQuartet.

    Parameters
    ----------
    engine : Engine
        The integral engine.
    quartet : ShellSetQuartet
        Shell quartet to compute.
    op : Operator, optional
        Two-electron operator. Defaults to Coulomb.

    Returns
    -------
    numpy.ndarray
        Flat array of ERI values.
    """
    from ._core import Operator
    if op is None:
        op = Operator.coulomb()
    return engine.compute_eri_block(op, quartet)
