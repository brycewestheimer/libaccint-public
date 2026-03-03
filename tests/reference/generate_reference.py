#!/usr/bin/env python3
"""
PySCF Reference Integral Generator

Generates reference integral data for validating LibAccInt implementations.
Computes overlap (S), kinetic (T), nuclear attraction (V), and electron repulsion (ERI)
integrals for standard test molecules using PySCF.

Output format: JSON with metadata, full integral matrices, and individual shell pair blocks.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

# Try to import PySCF with helpful error message
try:
    import pyscf
    from pyscf import gto
except ImportError:
    print("ERROR: PySCF is not installed.", file=sys.stderr)
    print("Please install PySCF with: pip install pyscf", file=sys.stderr)
    print("Or: conda install -c pyscf pyscf", file=sys.stderr)
    sys.exit(1)


# Standard molecular geometries (in Angstrom)
MOLECULES = {
    'H2': {
        'atoms': [['H', [0.0, 0.0, 0.0]], ['H', [0.74, 0.0, 0.0]]],
        'description': 'Hydrogen molecule, bond length 0.74 A'
    },
    'H2O': {
        'atoms': [
            ['O', [0.0, 0.0, 0.0]],
            ['H', [0.758602, 0.0, 0.504284]],
            ['H', [0.758602, 0.0, -0.504284]]
        ],
        'description': 'Water molecule, O-H 0.96 A, angle 104.5 degrees'
    },
    'NH3': {
        'atoms': [
            ['N', [0.0, 0.0, 0.0]],
            ['H', [0.0, 0.9377, 0.3816]],
            ['H', [0.8121, -0.4689, 0.3816]],
            ['H', [-0.8121, -0.4689, 0.3816]]
        ],
        'description': 'Ammonia molecule, N-H 1.012 A, HNH angle 106.7 degrees'
    },
    'CH4': {
        'atoms': [
            ['C', [0.0, 0.0, 0.0]],
            ['H', [0.6276, 0.6276, 0.6276]],
            ['H', [-0.6276, -0.6276, 0.6276]],
            ['H', [-0.6276, 0.6276, -0.6276]],
            ['H', [0.6276, -0.6276, -0.6276]]
        ],
        'description': 'Methane molecule, C-H 1.089 A, tetrahedral geometry'
    }
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate reference integral data using PySCF',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--molecules',
        nargs='+',
        default=['H2', 'H2O', 'NH3', 'CH4'],
        choices=list(MOLECULES.keys()),
        help='Molecules to compute integrals for'
    )
    parser.add_argument(
        '--basis-sets',
        nargs='+',
        default=['STO-3G', 'cc-pVDZ'],
        help='Basis sets to use'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('tests/data/reference'),
        help='Output directory for JSON files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    return parser.parse_args()


def angstrom_to_bohr(coords: List[List]) -> List[List]:
    """Convert atomic coordinates from Angstrom to Bohr."""
    ANGSTROM_TO_BOHR = 1.8897261246257702
    return [[atom[0], [c * ANGSTROM_TO_BOHR for c in atom[1]]] for atom in coords]


def extract_shell_info(mol: gto.Mole) -> List[Dict[str, Any]]:
    """Extract shell information from PySCF molecule object."""
    shells = []
    for ish in range(mol.nbas):
        atom_index = mol.bas_atom(ish)
        angular_momentum = mol.bas_angular(ish)
        n_primitives = mol.bas_nprim(ish)
        n_contracted = mol.bas_nctr(ish)

        # Extract exponents and contraction coefficients
        exponents = mol.bas_exp(ish).tolist()
        coefficients = mol.bas_ctr_coeff(ish).tolist()

        shells.append({
            'index': ish,
            'atom': atom_index,
            'angular_momentum': angular_momentum,
            'n_primitives': n_primitives,
            'n_contracted': n_contracted,
            'exponents': exponents,
            'coefficients': coefficients
        })

    return shells


def compute_shell_pair_blocks(mol: gto.Mole, matrix: np.ndarray,
                               integral_type: str) -> List[Dict[str, Any]]:
    """Extract individual shell pair blocks from integral matrix."""
    blocks = []

    for ish in range(mol.nbas):
        # Get basis function range for shell i
        i_start = mol.ao_loc[ish]
        i_end = mol.ao_loc[ish + 1]
        i_size = i_end - i_start

        for jsh in range(mol.nbas):
            # Get basis function range for shell j
            j_start = mol.ao_loc[jsh]
            j_end = mol.ao_loc[jsh + 1]
            j_size = j_end - j_start

            # Extract block
            block = matrix[i_start:i_end, j_start:j_end]

            blocks.append({
                'shell_i': ish,
                'shell_j': jsh,
                'shell_i_angular_momentum': mol.bas_angular(ish),
                'shell_j_angular_momentum': mol.bas_angular(jsh),
                'shell_i_nprim': mol.bas_nprim(ish),
                'shell_j_nprim': mol.bas_nprim(jsh),
                'basis_range_i': [i_start, i_end],
                'basis_range_j': [j_start, j_end],
                'block_shape': [i_size, j_size],
                f'{integral_type}_block': block.flatten().tolist()
            })

    return blocks


def compute_shell_quartet_blocks(mol: gto.Mole, eri_tensor: np.ndarray) -> List[Dict[str, Any]]:
    """Extract individual shell quartet blocks from ERI tensor."""
    blocks = []

    # Only compute unique quartets (ish <= jsh, ksh <= lsh, ijsh <= klsh)
    for ish in range(mol.nbas):
        i_start = mol.ao_loc[ish]
        i_end = mol.ao_loc[ish + 1]
        i_size = i_end - i_start

        for jsh in range(ish + 1):
            j_start = mol.ao_loc[jsh]
            j_end = mol.ao_loc[jsh + 1]
            j_size = j_end - j_start

            for ksh in range(mol.nbas):
                k_start = mol.ao_loc[ksh]
                k_end = mol.ao_loc[ksh + 1]
                k_size = k_end - k_start

                for lsh in range(ksh + 1):
                    # Skip if klsh > ijsh (8-fold symmetry)
                    ijsh = ish * (ish + 1) // 2 + jsh
                    klsh = ksh * (ksh + 1) // 2 + lsh
                    if klsh > ijsh:
                        continue

                    l_start = mol.ao_loc[lsh]
                    l_end = mol.ao_loc[lsh + 1]
                    l_size = l_end - l_start

                    # Extract block
                    block = eri_tensor[i_start:i_end, j_start:j_end,
                                      k_start:k_end, l_start:l_end]

                    blocks.append({
                        'shell_i': ish,
                        'shell_j': jsh,
                        'shell_k': ksh,
                        'shell_l': lsh,
                        'shell_i_angular_momentum': mol.bas_angular(ish),
                        'shell_j_angular_momentum': mol.bas_angular(jsh),
                        'shell_k_angular_momentum': mol.bas_angular(ksh),
                        'shell_l_angular_momentum': mol.bas_angular(lsh),
                        'shell_i_nprim': mol.bas_nprim(ish),
                        'shell_j_nprim': mol.bas_nprim(jsh),
                        'shell_k_nprim': mol.bas_nprim(ksh),
                        'shell_l_nprim': mol.bas_nprim(lsh),
                        'basis_range_i': [i_start, i_end],
                        'basis_range_j': [j_start, j_end],
                        'basis_range_k': [k_start, k_end],
                        'basis_range_l': [l_start, l_end],
                        'block_shape': [i_size, j_size, k_size, l_size],
                        'eri_block': block.flatten().tolist()
                    })

    return blocks


def compute_integrals(mol_name: str, basis_set: str, verbose: bool = False) -> Dict[str, Any]:
    """Compute all integral types for a given molecule and basis set."""
    if verbose:
        print(f"\nComputing integrals for {mol_name}/{basis_set}...")

    # Build molecule
    mol_data = MOLECULES[mol_name]
    mol = gto.Mole()
    mol.atom = mol_data['atoms']
    mol.basis = basis_set
    mol.unit = 'Angstrom'
    mol.build()

    if verbose:
        print(f"  Molecule built: {mol.natm} atoms, {mol.nao} basis functions, {mol.nbas} shells")

    # Extract shell information
    shells = extract_shell_info(mol)

    # Compute integrals
    if verbose:
        print("  Computing overlap integrals...")
    overlap = mol.intor('int1e_ovlp')

    if verbose:
        print("  Computing kinetic energy integrals...")
    kinetic = mol.intor('int1e_kin')

    if verbose:
        print("  Computing nuclear attraction integrals...")
    nuclear = mol.intor('int1e_nuc')

    if verbose:
        print("  Computing electron repulsion integrals...")
    eri = mol.intor('int2e')

    if verbose:
        print("  Extracting shell pair blocks...")
    overlap_blocks = compute_shell_pair_blocks(mol, overlap, 'overlap')
    kinetic_blocks = compute_shell_pair_blocks(mol, kinetic, 'kinetic')
    nuclear_blocks = compute_shell_pair_blocks(mol, nuclear, 'nuclear_attraction')

    if verbose:
        print("  Extracting shell quartet blocks...")
    eri_blocks = compute_shell_quartet_blocks(mol, eri)

    # Convert coordinates to Bohr for output
    atoms_bohr = angstrom_to_bohr(mol_data['atoms'])

    # Build output data structure
    data = {
        'format_version': '1.0',
        'generator': 'PySCF',
        'pyscf_version': pyscf.__version__,
        'generated_date': datetime.now().isoformat(),
        'molecule': {
            'name': mol_name,
            'description': mol_data['description'],
            'atoms': [atom[0] for atom in mol_data['atoms']],
            'geometry_angstrom': [atom[1] for atom in mol_data['atoms']],
            'geometry_bohr': [atom[1] for atom in atoms_bohr],
            'charge': mol.charge,
            'spin': mol.spin
        },
        'basis_set': basis_set,
        'n_atoms': mol.natm,
        'n_basis': mol.nao,
        'n_shells': mol.nbas,
        'shells': shells,
        'integrals': {
            'overlap': {
                'matrix': overlap.flatten().tolist(),
                'shape': list(overlap.shape)
            },
            'kinetic': {
                'matrix': kinetic.flatten().tolist(),
                'shape': list(kinetic.shape)
            },
            'nuclear_attraction': {
                'matrix': nuclear.flatten().tolist(),
                'shape': list(nuclear.shape)
            },
            'electron_repulsion': {
                'tensor': eri.flatten().tolist(),
                'shape': list(eri.shape)
            }
        },
        'shell_pair_blocks': {
            'overlap': overlap_blocks,
            'kinetic': kinetic_blocks,
            'nuclear_attraction': nuclear_blocks
        },
        'shell_quartet_blocks': {
            'electron_repulsion': eri_blocks
        }
    }

    if verbose:
        print(f"  Complete: {len(overlap_blocks)} shell pairs, {len(eri_blocks)} shell quartets")

    return data


def main():
    """Main entry point."""
    args = parse_arguments()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"PySCF Reference Integral Generator")
    print(f"PySCF version: {pyscf.__version__}")
    print(f"Output directory: {args.output_dir}")
    print(f"Molecules: {', '.join(args.molecules)}")
    print(f"Basis sets: {', '.join(args.basis_sets)}")

    # Generate reference data for each molecule/basis combination
    total_count = len(args.molecules) * len(args.basis_sets)
    current = 0

    for mol_name in args.molecules:
        for basis_set in args.basis_sets:
            current += 1
            print(f"\n[{current}/{total_count}] Processing {mol_name}/{basis_set}...")

            try:
                # Compute integrals
                data = compute_integrals(mol_name, basis_set, args.verbose)

                # Generate output filename
                filename = f"{mol_name}_{basis_set.replace('-', '_')}.json"
                output_path = args.output_dir / filename

                # Write JSON file
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)

                print(f"  Written: {output_path}")
                print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")

            except Exception as e:
                print(f"  ERROR: Failed to compute integrals for {mol_name}/{basis_set}", file=sys.stderr)
                print(f"  {type(e).__name__}: {e}", file=sys.stderr)
                continue

    print(f"\nComplete! Generated {current} reference data files in {args.output_dir}")


if __name__ == '__main__':
    main()
