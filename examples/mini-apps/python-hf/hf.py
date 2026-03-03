#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import libaccint


BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


@dataclass
class MoleculeInput:
    label: str
    atoms: List[libaccint.Atom]


PRESET_MOLECULES = {
    "h2": {
        "label": "H2",
        "atoms": [
            (1, [0.0, 0.0, 0.0]),
            (1, [0.0, 0.0, 1.4]),
        ],
    },
    "h2o": {
        "label": "H2O",
        "atoms": [
            (8, [0.000000, 0.000000, 0.117176]),
            (1, [0.000000, 1.430665, -0.468706]),
            (1, [0.000000, -1.430665, -0.468706]),
        ],
    },
}


def parse_backend(backend: str) -> libaccint.BackendHint:
    value = backend.lower()
    mapping = {
        "auto": libaccint.BackendHint.Auto,
        "force-cpu": libaccint.BackendHint.ForceCPU,
        "prefer-cpu": libaccint.BackendHint.PreferCPU,
        "prefer-gpu": libaccint.BackendHint.PreferGPU,
        "force-gpu": libaccint.BackendHint.ForceGPU,
    }
    if value not in mapping:
        raise ValueError(f"invalid backend: {backend}")
    return mapping[value]


def parse_units_scale(units: str) -> float:
    value = units.lower()
    if value == "bohr":
        return 1.0
    if value in ("angstrom", "ang"):
        return BOHR_PER_ANGSTROM
    raise ValueError("units must be 'bohr' or 'angstrom'")


def atomic_number_from_token(token: str) -> int:
    token = token.strip()
    if token.isdigit():
        z = int(token)
        if z <= 0:
            raise ValueError("atomic number must be positive")
        return z

    key = token.capitalize()
    symbol_to_z = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
    }
    if key not in symbol_to_z:
        raise ValueError(f"unsupported atomic symbol in mini-app: {token}")
    return symbol_to_z[key]


def parse_molecule_spec(spec: str, units: str) -> MoleculeInput:
    text = spec.strip()
    if not text:
        raise ValueError("molecule specification is empty")

    key = text.lower()
    if key in PRESET_MOLECULES:
        data = PRESET_MOLECULES[key]
        atoms = [libaccint.Atom(z, xyz) for z, xyz in data["atoms"]]
        return MoleculeInput(label=data["label"], atoms=atoms)

    scale = parse_units_scale(units)
    atoms: List[libaccint.Atom] = []
    for raw_entry in text.split(";"):
        entry = raw_entry.strip()
        if not entry:
            continue
        parts = entry.replace(",", " ").split()
        if len(parts) != 4:
            raise ValueError(
                f"could not parse atom entry '{raw_entry}'. Expected 'Element x y z'."
            )
        z = atomic_number_from_token(parts[0])
        x, y, zc = (float(parts[1]), float(parts[2]), float(parts[3]))
        atoms.append(libaccint.Atom(z, [scale * x, scale * y, scale * zc]))

    if not atoms:
        raise ValueError("custom molecule specification produced zero atoms")

    return MoleculeInput(label="custom", atoms=atoms)


def electron_count(atoms: List[libaccint.Atom], charge: int) -> int:
    return sum(atom.atomic_number for atom in atoms) - charge


def compute_nuclear_repulsion(atoms: List[libaccint.Atom]) -> float:
    e_nuc = 0.0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            ai = atoms[i]
            aj = atoms[j]
            dx = ai.position.x - aj.position.x
            dy = ai.position.y - aj.position.y
            dz = ai.position.z - aj.position.z
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            e_nuc += ai.atomic_number * aj.atomic_number / r
    return e_nuc


def make_nuclear_params(atoms: List[libaccint.Atom]) -> libaccint.PointChargeParams:
    return libaccint.PointChargeParams(
        x=[atom.position.x for atom in atoms],
        y=[atom.position.y for atom in atoms],
        z=[atom.position.z for atom in atoms],
        charges=[float(atom.atomic_number) for atom in atoms],
    )


def symmetric_orthogonalization(S: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(S)
    inv_sqrt = 1.0 / np.sqrt(vals)
    return vecs @ np.diag(inv_sqrt) @ vecs.T


def scatter_pair_buffer_symmetric(buf, target: np.ndarray) -> None:
    flat = np.asarray(buf.to_numpy())
    for idx in range(buf.n_shell_pairs()):
        meta = buf.pair_meta(idx)
        n_vals = meta.na * meta.nb
        block = flat[meta.offset : meta.offset + n_vals].reshape(meta.na, meta.nb)

        i0 = meta.fi
        j0 = meta.fj
        target[i0 : i0 + meta.na, j0 : j0 + meta.nb] += block
        if i0 != j0:
            target[j0 : j0 + meta.nb, i0 : i0 + meta.na] += block.T


def build_1e_matrix_from_pair_buffers(
    engine: libaccint.Engine,
    op: libaccint.Operator,
    pairs,
    nbf: int,
    hint: libaccint.BackendHint,
) -> np.ndarray:
    mat = np.zeros((nbf, nbf), dtype=np.float64)
    for pair_batch in pairs:
        buf = engine.compute_pair(op, pair_batch, hint)
        scatter_pair_buffer_symmetric(buf, mat)
    return mat


def build_jk_from_quartet_buffers(
    engine: libaccint.Engine,
    quartets,
    D: np.ndarray,
    nbf: int,
    hint: libaccint.BackendHint,
) -> Tuple[np.ndarray, np.ndarray]:
    J = np.zeros((nbf, nbf), dtype=np.float64)
    K = np.zeros((nbf, nbf), dtype=np.float64)
    op = libaccint.Operator.coulomb()

    for quartet_batch in quartets:
        buf = engine.compute_quartet(op, quartet_batch, hint)
        flat = np.asarray(buf.to_numpy())

        for qi in range(buf.n_shell_quartets()):
            meta = buf.quartet_meta(qi)
            n_vals = meta.na * meta.nb * meta.nc * meta.nd
            block = flat[meta.offset : meta.offset + n_vals].reshape(
                meta.na, meta.nb, meta.nc, meta.nd
            )

            i0, j0, k0, l0 = meta.fi, meta.fj, meta.fk, meta.fl
            for a in range(meta.na):
                mu = i0 + a
                for b in range(meta.nb):
                    nu = j0 + b
                    for c in range(meta.nc):
                        lam = k0 + c
                        for d in range(meta.nd):
                            sig = l0 + d
                            g = block[a, b, c, d]
                            J[mu, nu] += g * D[lam, sig]
                            K[mu, lam] += g * D[nu, sig]

    return J, K


def build_jk_from_consumer(
    engine: libaccint.Engine,
    quartets,
    D: np.ndarray,
    nbf: int,
    hint: libaccint.BackendHint,
) -> Tuple[np.ndarray, np.ndarray]:
    fock = libaccint.FockBuilder(nbf)
    fock.set_density(np.ascontiguousarray(D, dtype=np.float64))

    op = libaccint.Operator.coulomb()
    for quartet_batch in quartets:
        engine.compute_quartet_and_consume(op, quartet_batch, fock, hint)

    return fock.get_coulomb_matrix(), fock.get_exchange_matrix()


def run_rhf(args: argparse.Namespace) -> float:
    hint = parse_backend(args.backend)
    mol = parse_molecule_spec(args.molecule, args.units)

    n_electrons = electron_count(mol.atoms, args.charge)
    if n_electrons <= 0:
        raise ValueError("electron count must be positive after applying charge")
    if n_electrons % 2 != 0:
        raise ValueError("mini-app supports only closed-shell RHF (even electron count)")

    n_occ = n_electrons // 2

    basis = libaccint.basis_set(args.basis, mol.atoms)
    nbf = basis.n_basis_functions()
    if n_occ > nbf:
        raise ValueError("not enough basis functions for occupied orbitals")

    config = libaccint.DispatchConfig()
    config.n_gpu_slots = args.gpu_slots
    engine = libaccint.Engine(basis, config)
    pairs = basis.shell_set_pairs()
    quartets = basis.shell_set_quartets()

    print("=" * 60)
    print("  LibAccInt Mini-App: Restricted Hartree-Fock (RHF)")
    print("=" * 60)
    print()
    print(f"Molecule:              {mol.label}")
    print(f"Atoms:                 {len(mol.atoms)}")
    print(f"Charge:                {args.charge}")
    print(f"Electrons:             {n_electrons}")
    print(f"Basis:                 {args.basis}")
    print(f"Basis functions:       {nbf}")
    print(f"ShellSetPairs:         {len(pairs)}")
    print(f"ShellSetQuartets:      {len(quartets)}")
    print(f"Requested backend:     {args.backend}")
    print(f"GPU available:         {'yes' if engine.gpu_available() else 'no'}")
    print(f"GPU slots:             {args.gpu_slots}")
    print(f"Two-electron mode:     {args.two_e_mode}")
    print()

    S = build_1e_matrix_from_pair_buffers(engine, libaccint.Operator.overlap(), pairs, nbf, hint)
    T = build_1e_matrix_from_pair_buffers(engine, libaccint.Operator.kinetic(), pairs, nbf, hint)
    nuclear_params = make_nuclear_params(mol.atoms)
    V = build_1e_matrix_from_pair_buffers(
        engine,
        libaccint.Operator.nuclear(nuclear_params),
        pairs,
        nbf,
        hint,
    )
    H = T + V

    X = symmetric_orthogonalization(S)
    H_prime = X.T @ H @ X
    eps, C_prime = np.linalg.eigh(H_prime)
    C = X @ C_prime

    e_nuc = compute_nuclear_repulsion(mol.atoms)
    e_old = 0.0

    print(f"Starting SCF iteration (max {args.max_iter} cycles)...")
    print("-" * 70)
    print(f"{'Iter':>5s}{'E_total (Hartree)':>22s}{'Delta_E':>18s}{'Status':>12s}")
    print("-" * 70)

    for iteration in range(1, args.max_iter + 1):
        C_occ = C[:, :n_occ]
        D = 2.0 * C_occ @ C_occ.T

        J_buffer = K_buffer = None
        J_consumer = K_consumer = None

        if args.two_e_mode in ("buffer", "compare"):
            J_buffer, K_buffer = build_jk_from_quartet_buffers(
                engine, quartets, D, nbf, hint
            )

        if args.two_e_mode in ("consumer", "compare"):
            J_consumer, K_consumer = build_jk_from_consumer(
                engine, quartets, D, nbf, hint
            )

        if args.two_e_mode in ("consumer", "compare"):
            J, K = J_consumer, K_consumer
        else:
            J, K = J_buffer, K_buffer

        if args.two_e_mode == "compare":
            diff_j = np.max(np.abs(J_buffer - J_consumer))
            diff_k = np.max(np.abs(K_buffer - K_consumer))
            print(
                f"  [compare] max|J_buffer-J_consumer|={diff_j:.3e}  "
                f"max|K_buffer-K_consumer|={diff_k:.3e}"
            )

        F = H + J - 0.5 * K

        F_prime = X.T @ F @ X
        eps, C_prime = np.linalg.eigh(F_prime)
        C = X @ C_prime

        e_elec = 0.5 * np.sum(D * (H + F))
        e_total = e_elec + e_nuc
        delta_e = abs(e_total - e_old)

        status = ""
        if delta_e < args.conv_thresh and iteration > 1:
            status = "CONV"

        print(f"{iteration:5d}{e_total:22.10f}{delta_e:18.4e}{status:>12s}")

        if status == "CONV":
            print("-" * 70)
            print()
            print(f"SCF converged in {iteration} iterations.")
            print()
            print(f"Electronic energy:  {e_elec:20.10f} Hartree")
            print(f"Nuclear repulsion:  {e_nuc:20.10f} Hartree")
            print(f"Total RHF energy:   {e_total:20.10f} Hartree")
            print()
            print("Orbital energies (Hartree):")
            for i in range(nbf):
                label = "(occupied)" if i < n_occ else "(virtual)"
                print(f"  {i + 1:3d}: {eps[i]:14.6f}  {label}")
            print()
            return float(e_total)

        e_old = e_total

    print("-" * 70)
    raise RuntimeError(
        f"SCF did not converge within {args.max_iter} iterations. Last energy: {e_old:.10f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Restricted Hartree-Fock mini-application using ShellSetPair/"
            "ShellSetQuartet work-unit APIs"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python hf.py --molecule h2o --basis sto-3g --two-e-mode buffer\n"
            "  python hf.py --molecule h2 --basis sto-3g --backend prefer-gpu\n"
            "  python hf.py --molecule \"H 0 0 0; H 0 0 1.4\" --units bohr --two-e-mode compare\n"
        ),
    )
    parser.add_argument(
        "--molecule",
        "-m",
        default="h2o",
        help=(
            "preset molecule ('h2', 'h2o') or inline geometry "
            "'Element x y z; Element x y z'"
        ),
    )
    parser.add_argument(
        "--units",
        default="bohr",
        choices=["bohr", "angstrom"],
        help="coordinate units for custom inline geometry (default: bohr)",
    )
    parser.add_argument(
        "--basis",
        "-b",
        default="sto-3g",
        help="basis set name (default: sto-3g)",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="total molecular charge (default: 0)",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "force-cpu", "prefer-cpu", "prefer-gpu", "force-gpu"],
        help="backend hint for dispatch (default: auto)",
    )
    parser.add_argument(
        "--two-e-mode",
        default="buffer",
        choices=["buffer", "consumer", "compare"],
        help="two-electron build mode: buffer|consumer|compare",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="maximum SCF iterations (default: 100)",
    )
    parser.add_argument(
        "--conv-thresh",
        type=float,
        default=1.0e-10,
        help="energy convergence threshold (default: 1e-10)",
    )
    parser.add_argument(
        "--gpu-slots",
        type=int,
        default=4,
        help="number of concurrent GPU execution slots (default: 4)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_rhf(args)


if __name__ == "__main__":
    main()
