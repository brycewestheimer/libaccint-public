#!/usr/bin/env python3
"""
Water Cluster PySCF Reference Data Generator

Generates reference integrals for (H2O)_N clusters (N = 1, 2, 4, 8) with the
aug-cc-pVTZ basis set using Cartesian Gaussians, for validating LibAccInt.

Output format: JSON with full S, T, V matrices, RHF energy, J/K from a stored
density matrix, and sampled ERI shell quartets.
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

try:
    import pyscf
    from pyscf import gto, scf
except ImportError:
    print("ERROR: PySCF is not installed.", file=sys.stderr)
    print("Install with: pip install pyscf", file=sys.stderr)
    sys.exit(1)


def make_water_cluster_geometry(n: int) -> List[List]:
    """
    Generate (H2O)_N cluster geometry in Bohr.

    N=1: single water at origin
      O at [0, 0, 0], H at [0, +/-1.43233673, -1.10866041]

    N>=2: O atoms at regular N-gon vertices in XY plane with nearest-neighbor
      O-O distance = 5.4 Bohr. Each water oriented with one H pointing toward
      the next O (H-bond donor).

    Returns list of [element, [x, y, z]] entries.
    """
    r_oh = 1.8088  # Bohr
    theta_hoh = 104.52 * math.pi / 180.0  # H-O-H angle in radians
    half_theta = theta_hoh / 2.0

    if n == 1:
        # Single water at origin: O at origin, Hs in YZ plane
        hy = r_oh * math.sin(half_theta)
        hz = -r_oh * math.cos(half_theta)
        return [
            ['O', [0.0, 0.0, 0.0]],
            ['H', [0.0, hy, hz]],
            ['H', [0.0, -hy, hz]],
        ]

    # N >= 2: place O atoms on a regular N-gon in XY plane
    # with nearest-neighbor O-O distance = 5.4 Bohr
    oo_dist = 5.4
    # Radius of circumscribing circle: R = d / (2 * sin(pi/N))
    R = oo_dist / (2.0 * math.sin(math.pi / n))

    atoms = []
    for i in range(n):
        angle_o = 2.0 * math.pi * i / n
        ox = R * math.cos(angle_o)
        oy = R * math.sin(angle_o)
        oz = 0.0
        atoms.append(['O', [ox, oy, oz]])

        # Direction toward next O (H-bond donor direction)
        next_i = (i + 1) % n
        angle_next = 2.0 * math.pi * next_i / n
        nx = R * math.cos(angle_next) - ox
        ny = R * math.sin(angle_next) - oy
        nlen = math.sqrt(nx * nx + ny * ny)
        dx = nx / nlen
        dy = ny / nlen

        # Perpendicular direction in XY plane (for second H)
        px = -dy
        py = dx

        # H1: donor H, pointing toward next O
        h1x = ox + r_oh * math.cos(half_theta) * dx
        h1y = oy + r_oh * math.cos(half_theta) * dy
        h1z = oz - r_oh * math.sin(half_theta)
        atoms.append(['H', [h1x, h1y, h1z]])

        # H2: other H, pointing away (perpendicular in XY, down in Z)
        h2x = ox + r_oh * math.cos(half_theta) * px
        h2y = oy + r_oh * math.cos(half_theta) * py
        h2z = oz + r_oh * math.sin(half_theta)
        atoms.append(['H', [h2x, h2y, h2z]])

    return atoms


def n_cartesian(L: int) -> int:
    """Number of Cartesian basis functions for angular momentum L."""
    return (L + 1) * (L + 2) // 2


def load_bse_basis_for_pyscf(bse_json_path: str) -> Dict:
    """
    Load a BSE JSON basis set file and convert it to a PySCF-compatible basis
    dictionary that produces one shell per contraction column — matching
    LibAccInt's BseJsonParser segmented contraction handling.
    """
    with open(bse_json_path) as f:
        bse = json.load(f)

    Z_TO_SYM = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N',
                8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al',
                14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar'}

    basis_dict = {}
    for z_str, el_data in bse['elements'].items():
        z = int(z_str)
        sym = Z_TO_SYM.get(z, f'X{z}')
        shells = []
        for shell_data in el_data['electron_shells']:
            am_list = shell_data['angular_momentum']
            exponents = [float(e) for e in shell_data['exponents']]
            coeff_cols = shell_data['coefficients']

            if len(am_list) == 1:
                am = am_list[0]
                for col in coeff_cols:
                    coefficients = [float(c) for c in col]
                    if all(c == 0.0 for c in coefficients):
                        continue
                    shell_entry = [am]
                    for exp, coef in zip(exponents, coefficients):
                        shell_entry.append([exp, coef])
                    shells.append(shell_entry)
            else:
                for col_idx, am in enumerate(am_list):
                    if col_idx >= len(coeff_cols):
                        break
                    coefficients = [float(c) for c in coeff_cols[col_idx]]
                    if all(c == 0.0 for c in coefficients):
                        continue
                    shell_entry = [am]
                    for exp, coef in zip(exponents, coefficients):
                        shell_entry.append([exp, coef])
                    shells.append(shell_entry)

        basis_dict[sym] = shells
    return basis_dict


def get_libaccint_shell_order(bse_json_path: str, atoms: List[List]) -> List[Dict]:
    """
    Determine the shell ordering that LibAccInt's BseJsonParser would produce.
    Returns a list of shell descriptors in LibAccInt order, each with:
      {'atom': int, 'L': int, 'nprim': int, 'coeffs': list, 'n_bf': int}
    """
    with open(bse_json_path) as f:
        bse = json.load(f)

    elem_to_z = {'H': '1', 'He': '2', 'Li': '3', 'Be': '4', 'B': '5',
                 'C': '6', 'N': '7', 'O': '8', 'F': '9', 'Ne': '10'}

    shells = []
    for atom_idx, (elem, _) in enumerate(atoms):
        z_str = elem_to_z[elem]
        el_data = bse['elements'][z_str]
        for shell_data in el_data['electron_shells']:
            am_list = shell_data['angular_momentum']
            exponents = [float(e) for e in shell_data['exponents']]
            coeff_cols = shell_data['coefficients']

            if len(am_list) == 1:
                am = am_list[0]
                for col in coeff_cols:
                    coefficients = [float(c) for c in col]
                    if all(c == 0.0 for c in coefficients):
                        continue
                    shells.append({
                        'atom': atom_idx,
                        'L': am,
                        'nprim': len(exponents),
                        'coeffs': coefficients,
                        'n_bf': n_cartesian(am),
                    })
            else:
                for col_idx, am in enumerate(am_list):
                    if col_idx >= len(coeff_cols):
                        break
                    coefficients = [float(c) for c in coeff_cols[col_idx]]
                    if all(c == 0.0 for c in coefficients):
                        continue
                    shells.append({
                        'atom': atom_idx,
                        'L': am,
                        'nprim': len(exponents),
                        'coeffs': coefficients,
                        'n_bf': n_cartesian(am),
                    })
    return shells


def compute_bf_permutation(mol: gto.Mole, libaccint_shells: List[Dict]) -> np.ndarray:
    """
    Compute the permutation that maps PySCF basis function indices to LibAccInt
    basis function indices.

    Returns perm such that: M_libaccint[i, j] = M_pyscf[perm[i], perm[j]]

    Both PySCF and LibAccInt have the same shells, but PySCF sorts by angular
    momentum within each atom while LibAccInt follows BSE JSON file order.
    Within each (atom, L) group, both orderings preserve the BSE file order,
    so we can match shells 1:1 by their position within each group.
    """
    from collections import defaultdict

    # Group PySCF shells by (atom, L), preserving order within each group
    pyscf_groups = defaultdict(list)
    for ish in range(mol.nbas):
        atom = int(mol.bas_atom(ish))
        L = int(mol.bas_angular(ish))
        ao_start = int(mol.ao_loc[ish])
        n_bf = int(mol.ao_loc[ish + 1]) - ao_start
        pyscf_groups[(atom, L)].append({
            'ao_start': ao_start, 'n_bf': n_bf,
        })

    # Group LibAccInt shells by (atom, L), preserving order
    libaccint_groups = defaultdict(list)
    li_bf = 0
    for sh in libaccint_shells:
        key = (sh['atom'], sh['L'])
        libaccint_groups[key].append({
            'bf_start': li_bf, 'n_bf': sh['n_bf'],
        })
        li_bf += sh['n_bf']

    # Build permutation by matching 1:1 within each (atom, L) group
    perm = np.zeros(mol.nao, dtype=int)
    for key in libaccint_groups:
        li_list = libaccint_groups[key]
        ps_list = pyscf_groups[key]
        assert len(li_list) == len(ps_list), \
            f"Shell count mismatch for {key}: LibAccInt={len(li_list)}, PySCF={len(ps_list)}"
        for li_sh, ps_sh in zip(li_list, ps_list):
            assert li_sh['n_bf'] == ps_sh['n_bf'], \
                f"BF count mismatch for {key}: {li_sh['n_bf']} vs {ps_sh['n_bf']}"
            for k in range(li_sh['n_bf']):
                perm[li_sh['bf_start'] + k] = ps_sh['ao_start'] + k

    return perm


def permute_matrix(M: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Permute a 2D matrix: M_out[i,j] = M_in[perm[i], perm[j]]"""
    return M[np.ix_(perm, perm)]


def renormalize_matrix(M: np.ndarray, S_diag: np.ndarray) -> np.ndarray:
    """
    Renormalize a matrix from PySCF's Cartesian convention (S_ii != 1 for d/f)
    to LibAccInt's convention (S_ii = 1 for all BFs).

    M_out[i,j] = M_in[i,j] / sqrt(S_diag[i] * S_diag[j])
    """
    inv_norm = 1.0 / np.sqrt(S_diag)
    return M * np.outer(inv_norm, inv_norm)


def find_bse_json(basis_name: str = 'aug-cc-pvtz') -> str:
    """Find the BSE JSON file for a given basis set name."""
    search_paths = [
        f'share/basis_sets/{basis_name}.json',
        f'../share/basis_sets/{basis_name}.json',
        f'../../share/basis_sets/{basis_name}.json',
    ]
    for p in search_paths:
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"BSE JSON file not found for {basis_name}")


def build_mol(atoms: List[List], basis: str = 'aug-cc-pvtz') -> gto.Mole:
    """
    Build a PySCF Mole object with Cartesian Gaussians in Bohr.

    Uses the BSE JSON file directly to construct the basis, ensuring one shell
    per contraction column — matching LibAccInt's BseJsonParser segmented
    contraction handling. Note: PySCF still reorders shells by angular momentum
    within each atom, so a permutation step is needed.
    """
    bse_path = find_bse_json(basis)
    basis_dict = load_bse_basis_for_pyscf(bse_path)

    mol = gto.Mole()
    mol.atom = atoms
    mol.basis = basis_dict
    mol.unit = 'Bohr'
    mol.cart = True
    mol.verbose = 0
    mol.build()
    return mol


def sample_eri_quartets(mol: gto.Mole, n_samples: int = 200,
                        seed: int = 42) -> List[Dict[str, Any]]:
    """
    Sample ERI shell quartets and compute per-quartet integral blocks.
    Uses shls_slice to avoid computing the full N^4 tensor.
    """
    rng = np.random.default_rng(seed)
    nsh = mol.nbas
    samples = []
    seen = set()

    # Generate unique random quartets
    attempts = 0
    while len(samples) < n_samples and attempts < n_samples * 10:
        attempts += 1
        i, j, k, l = rng.integers(0, nsh, size=4)
        key = (int(i), int(j), int(k), int(l))
        if key in seen:
            continue
        seen.add(key)

        i_s, i_e = int(mol.ao_loc[i]), int(mol.ao_loc[i + 1])
        j_s, j_e = int(mol.ao_loc[j]), int(mol.ao_loc[j + 1])
        k_s, k_e = int(mol.ao_loc[k]), int(mol.ao_loc[k + 1])
        l_s, l_e = int(mol.ao_loc[l]), int(mol.ao_loc[l + 1])

        block = mol.intor('int2e', shls_slice=(i, i+1, j, j+1, k, k+1, l, l+1))
        block = block.reshape(i_e - i_s, j_e - j_s, k_e - k_s, l_e - l_s)

        samples.append({
            'shells': [int(i), int(j), int(k), int(l)],
            'angular_momenta': [
                int(mol.bas_angular(i)), int(mol.bas_angular(j)),
                int(mol.bas_angular(k)), int(mol.bas_angular(l))
            ],
            'basis_ranges': [
                [i_s, i_e], [j_s, j_e], [k_s, k_e], [l_s, l_e]
            ],
            'shape': list(block.shape),
            'values': block.flatten().tolist()
        })

    return samples


def generate_reference(n_waters: int, verbose: bool = False) -> Dict[str, Any]:
    """Generate complete reference data for an (H2O)_N cluster.

    All matrices are permuted from PySCF's internal ordering to LibAccInt's
    BSE JSON file ordering so they can be compared element-by-element.
    """
    if verbose:
        print(f"\n=== (H2O)_{n_waters} with aug-cc-pVTZ ===")

    atoms = make_water_cluster_geometry(n_waters)
    mol = build_mol(atoms)
    bse_path = find_bse_json('aug-cc-pvtz')

    nbf = mol.nao
    nsh = mol.nbas
    n_electrons = 10 * n_waters

    if verbose:
        print(f"  Atoms: {mol.natm}, Shells: {nsh}, BF: {nbf}, Electrons: {n_electrons}")

    # Compute the permutation from PySCF to LibAccInt ordering
    if verbose:
        print("  Computing BF permutation (PySCF -> LibAccInt)...")
    libaccint_shells = get_libaccint_shell_order(bse_path, atoms)
    perm = compute_bf_permutation(mol, libaccint_shells)
    n_libaccint_shells = len(libaccint_shells)

    if verbose:
        is_identity = np.array_equal(perm, np.arange(nbf))
        print(f"  Permutation is {'identity' if is_identity else 'non-trivial'}")
        print(f"  LibAccInt shells: {n_libaccint_shells}, PySCF shells: {nsh}")

    # One-electron integrals (in PySCF ordering)
    if verbose:
        print("  Computing S, T, V matrices...")
    S_pyscf = mol.intor('int1e_ovlp')
    T_pyscf = mol.intor('int1e_kin')
    V_pyscf = mol.intor('int1e_nuc')

    # Get PySCF overlap diagonal for normalization correction
    # PySCF uses a Cartesian convention where S_ii != 1 for d/f functions.
    # LibAccInt normalizes each Cartesian function individually (S_ii = 1).
    S_diag_pyscf = np.diag(S_pyscf)

    # Permute to LibAccInt ordering, then renormalize
    S_perm = permute_matrix(S_pyscf, perm)
    S_diag_perm = np.diag(S_perm)
    S = renormalize_matrix(S_perm, S_diag_perm)
    T = renormalize_matrix(permute_matrix(T_pyscf, perm), S_diag_perm)
    V = renormalize_matrix(permute_matrix(V_pyscf, perm), S_diag_perm)

    if verbose:
        print(f"  S diagonal check: max|S_ii - 1| = {np.max(np.abs(np.diag(S) - 1.0)):.2e}")

    # RHF energy (basis-ordering-independent)
    if verbose:
        print("  Running RHF...")
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    e_rhf = mf.kernel()
    if verbose:
        print(f"  RHF energy: {e_rhf:.12f} Ha, converged: {mf.converged}")

    # Generate random symmetric density in LibAccInt ordering
    if verbose:
        print("  Generating random density matrix (LibAccInt ordering)...")
    rng = np.random.default_rng(42)
    D_raw = rng.standard_normal((nbf, nbf))
    D = (D_raw + D_raw.T) / 2.0
    D *= 0.5 / nbf

    # To compute J, K from D in PySCF, we need to:
    # 1. Convert D from LibAccInt convention (normalized, S_ii=1) to PySCF convention
    #    D_pyscf = D_lib / (N_i * N_j) where N_i = sqrt(S_pyscf_diag[i])
    # 2. Permute D from LibAccInt ordering to PySCF ordering
    # 3. Compute J, K in PySCF
    # 4. Permute and renormalize J, K to LibAccInt convention
    inv_perm = np.argsort(perm)
    inv_norm = 1.0 / np.sqrt(S_diag_perm)
    D_pyscf_liorder = D * np.outer(inv_norm, inv_norm)  # LibAccInt -> PySCF convention
    D_pyscf = permute_matrix(D_pyscf_liorder, inv_perm)

    if verbose:
        print("  Computing J, K from stored density...")
    J_pyscf, K_pyscf = scf.hf.get_jk(mol, D_pyscf)

    # Permute J, K to LibAccInt ordering and renormalize
    J = renormalize_matrix(permute_matrix(J_pyscf, perm), S_diag_perm)
    K = renormalize_matrix(permute_matrix(K_pyscf, perm), S_diag_perm)

    # Sampled ERI values via delta-density trick (normalization-correct)
    # For each sampled BF pair (k,l), compute J_ij = sum_{k,l} (ij|kl) D_kl
    # with D = delta(k,l) to get (ij|kl). Store specific (i,j,k,l) -> value.
    if verbose:
        print("  Computing sampled ERI elements via delta-density...")
    rng_eri = np.random.default_rng(123)
    n_eri_samples = min(20, nbf)
    eri_samples = []
    for _ in range(n_eri_samples):
        i, j, k, l = rng_eri.integers(0, nbf, size=4).tolist()
        # Compute (ij|kl) in LibAccInt convention via PySCF
        # D_pyscf delta at (k,l) in PySCF convention and ordering
        D_delta_lib = np.zeros((nbf, nbf))
        D_delta_lib[k, l] = 1.0
        D_delta_lib[l, k] = 1.0
        # Convert to PySCF convention
        D_delta_pyscf_li = D_delta_lib * np.outer(inv_norm, inv_norm)
        D_delta_pyscf = permute_matrix(D_delta_pyscf_li, inv_perm)
        J_delta_pyscf = scf.hf.get_jk(mol, D_delta_pyscf, hermi=1)[0]
        J_delta_lib = renormalize_matrix(permute_matrix(J_delta_pyscf, perm), S_diag_perm)
        # J_delta_lib[i,j] = (ij|kl) + (ij|lk) = 2*(ij|kl) if k!=l, else (ij|kk)
        val = J_delta_lib[i, j]
        if k != l:
            val /= 2.0
        eri_samples.append({
            'indices': [i, j, k, l],
            'value': float(val),
        })
    if verbose:
        print(f"  Computed {len(eri_samples)} ERI samples")

    # Shell information in LibAccInt ordering
    shells = []
    bf_start = 0
    for idx, sh in enumerate(libaccint_shells):
        shells.append({
            'index': idx,
            'atom': sh['atom'],
            'angular_momentum': sh['L'],
            'n_primitives': sh['nprim'],
            'ao_start': bf_start,
            'ao_end': bf_start + sh['n_bf'],
        })
        bf_start += sh['n_bf']

    data = {
        'format_version': '2.0',
        'generator': 'PySCF',
        'pyscf_version': pyscf.__version__,
        'generated_date': datetime.now().isoformat(),
        'cartesian': True,
        'unit': 'Bohr',
        'basis_set': 'aug-cc-pVTZ',
        'note': 'All matrices are in LibAccInt BF ordering (BSE JSON file order)',
        'molecule': {
            'name': f'water_{n_waters}',
            'description': f'(H2O)_{n_waters} cluster with aug-cc-pVTZ',
            'n_waters': n_waters,
            'n_atoms': mol.natm,
            'n_electrons': n_electrons,
            'atoms': [a[0] for a in atoms],
            'geometry_bohr': [a[1] for a in atoms],
        },
        'n_basis': nbf,
        'n_shells': n_libaccint_shells,
        'shells': shells,
        'integrals': {
            'overlap': {
                'matrix': S.flatten().tolist(),
                'shape': [nbf, nbf],
            },
            'kinetic': {
                'matrix': T.flatten().tolist(),
                'shape': [nbf, nbf],
            },
            'nuclear_attraction': {
                'matrix': V.flatten().tolist(),
                'shape': [nbf, nbf],
            },
        },
        'rhf': {
            'total_energy': e_rhf,
            'converged': bool(mf.converged),
        },
        'density_matrix': {
            'matrix': D.flatten().tolist(),
            'shape': [nbf, nbf],
            'description': 'Random symmetric density from numpy RNG seed=42, scaled by 0.5/nbf, in LibAccInt ordering',
        },
        'fock_from_density': {
            'coulomb_matrix': J.flatten().tolist(),
            'exchange_matrix': K.flatten().tolist(),
            'shape': [nbf, nbf],
        },
        'sampled_eri': eri_samples,
    }

    return data


def main():
    parser = argparse.ArgumentParser(
        description='Generate water cluster reference data using PySCF')
    parser.add_argument('--sizes', nargs='+', type=int, default=[1, 2, 4, 8],
                        help='Water cluster sizes N')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('tests/data/reference'),
                        help='Output directory')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Water Cluster Reference Generator")
    print(f"PySCF version: {pyscf.__version__}")
    print(f"Sizes: {args.sizes}")
    print(f"Output: {args.output_dir}")

    for n in args.sizes:
        print(f"\n[{n}] Generating (H2O)_{n} / aug-cc-pVTZ...")
        try:
            data = generate_reference(n, verbose=args.verbose)
            filename = f"water_{n}_aug_cc_pVTZ.json"
            output_path = args.output_dir / filename
            with open(output_path, 'w') as f:
                json.dump(data, f)
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  Written: {output_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue

    print("\nDone!")


if __name__ == '__main__':
    main()
