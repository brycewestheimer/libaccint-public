#!/usr/bin/env python3
"""Generate higher angular momentum reference data using PySCF.

This script generates reference integral data for f, g, and h functions
to validate LibAccInt's higher AM implementation.

Requirements:
    pip install pyscf numpy

Usage:
    python generate_higher_am_reference.py [--basis cc-pVTZ] [--output-dir tests/data/higher_am/]
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

try:
    import pyscf
    from pyscf import gto, scf
except ImportError:
    print("ERROR: PySCF is required for reference generation.")
    print("Install with: pip install pyscf")
    exit(1)


# Test molecules with various elements that use higher AM basis functions
MOLECULES = {
    # cc-pVTZ has d-functions on H, and d,f on C/N/O
    "H2O_TZ": {
        "atoms": "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0",
        "basis": "cc-pVTZ",
        "max_am": 3,  # f-functions on O
        "description": "Water molecule with cc-pVTZ (f-functions)"
    },
    # cc-pVQZ has up to g-functions
    "H2O_QZ": {
        "atoms": "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0",
        "basis": "cc-pVQZ",
        "max_am": 4,  # g-functions on O
        "description": "Water molecule with cc-pVQZ (g-functions)"
    },
    # Methane with higher AM
    "CH4_TZ": {
        "atoms": """
            C  0.000000  0.000000  0.000000
            H  0.626425  0.626425  0.626425
            H -0.626425 -0.626425  0.626425
            H -0.626425  0.626425 -0.626425
            H  0.626425 -0.626425 -0.626425
        """,
        "basis": "cc-pVTZ",
        "max_am": 3,
        "description": "Methane with cc-pVTZ (f-functions)"
    },
    # Argon for testing with d-block elements behavior
    "Ar_TZ": {
        "atoms": "Ar 0 0 0",
        "basis": "cc-pVTZ",
        "max_am": 3,
        "description": "Argon atom with cc-pVTZ"
    },
}


def get_am_symbol(l: int) -> str:
    """Get angular momentum symbol."""
    return "spdfghi"[l] if l < 7 else f"L{l}"


def analyze_basis_am(mol: gto.Mole) -> Dict[str, Any]:
    """Analyze angular momentum content of basis set."""
    am_counts = {}
    max_am = 0
    
    for ish in range(mol.nbas):
        l = mol.bas_angular(ish)
        sym = get_am_symbol(l)
        am_counts[sym] = am_counts.get(sym, 0) + 1
        max_am = max(max_am, l)
    
    return {
        "max_angular_momentum": max_am,
        "am_symbol": get_am_symbol(max_am),
        "shell_counts": am_counts,
        "n_basis": mol.nao,
        "n_shells": mol.nbas
    }


def compute_one_electron_integrals(mol: gto.Mole) -> Dict[str, np.ndarray]:
    """Compute one-electron integrals."""
    S = mol.intor('int1e_ovlp')  # Overlap
    T = mol.intor('int1e_kin')   # Kinetic
    V = mol.intor('int1e_nuc')   # Nuclear attraction
    
    return {
        "overlap": S,
        "kinetic": T,
        "nuclear": V
    }


def compute_eri_samples(mol: gto.Mole, n_samples: int = 100) -> List[Dict[str, Any]]:
    """Compute sample ERI values for validation.
    
    Instead of storing the full ERI tensor (which can be huge),
    we store sample values at specific indices.
    """
    n = mol.nao
    samples = []
    
    # Get specific ERI values for shells with higher AM
    # Focus on quartets involving f, g, h shells
    eri = mol.intor('int2e', aosym='s1')
    
    # Sample some representative values
    np.random.seed(42)  # Reproducible
    
    # Sample random indices
    for _ in range(n_samples):
        i, j, k, l = np.random.randint(0, n, 4)
        val = eri[i, j, k, l]
        if abs(val) > 1e-15:  # Only significant values
            samples.append({
                "indices": [int(i), int(j), int(k), int(l)],
                "value": float(val)
            })
    
    # Also add diagonal elements
    for i in range(min(n, 20)):
        val = eri[i, i, i, i]
        samples.append({
            "indices": [int(i), int(i), int(i), int(i)],
            "value": float(val)
        })
    
    return samples


def generate_reference_for_molecule(name: str, config: Dict) -> Dict[str, Any]:
    """Generate reference data for a single molecule."""
    print(f"\n{'='*60}")
    print(f"Generating reference for: {name}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}")
    
    # Build molecule
    mol = gto.Mole()
    mol.atom = config['atoms']
    mol.basis = config['basis']
    mol.build()
    
    # Analyze basis
    basis_info = analyze_basis_am(mol)
    print(f"Basis analysis:")
    print(f"  Max AM: {basis_info['am_symbol']} (L={basis_info['max_angular_momentum']})")
    print(f"  Number of basis functions: {basis_info['n_basis']}")
    print(f"  Shell counts: {basis_info['shell_counts']}")
    
    # One-electron integrals
    print("Computing one-electron integrals...")
    one_e = compute_one_electron_integrals(mol)
    
    # Sample ERIs
    print("Computing ERI samples...")
    eri_samples = compute_eri_samples(mol)
    
    # Run HF for reference energy
    print("Running HF calculation...")
    mf = scf.RHF(mol)
    energy = mf.kernel()
    print(f"  HF Energy: {energy:.12f} Ha")
    
    # Compile results
    result = {
        "molecule_name": name,
        "description": config['description'],
        "geometry": config['atoms'],
        "basis_set": config['basis'],
        "basis_info": basis_info,
        "pyscf_version": pyscf.__version__,
        "generation_timestamp": datetime.now().isoformat(),
        "one_electron_integrals": {
            "overlap": one_e["overlap"].tolist(),
            "kinetic": one_e["kinetic"].tolist(),
            "nuclear": one_e["nuclear"].tolist()
        },
        "eri_samples": eri_samples,
        "reference_energy": {
            "hf": float(energy)
        },
        "tolerances": {
            "one_electron": 1e-12,
            "two_electron": 1e-12,
            "energy": 1e-10
        }
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate higher AM reference data using PySCF"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/data/higher_am"),
        help="Output directory for reference files"
    )
    parser.add_argument(
        "--molecules",
        nargs="+",
        default=None,
        help="Specific molecules to generate (default: all)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select molecules
    molecules = args.molecules if args.molecules else list(MOLECULES.keys())
    
    print(f"LibAccInt Higher AM Reference Generator")
    print(f"PySCF version: {pyscf.__version__}")
    print(f"Output directory: {args.output_dir}")
    print(f"Molecules: {', '.join(molecules)}")
    
    # Generate reference for each molecule
    all_results = {}
    
    for mol_name in molecules:
        if mol_name not in MOLECULES:
            print(f"WARNING: Unknown molecule {mol_name}, skipping")
            continue
        
        try:
            result = generate_reference_for_molecule(mol_name, MOLECULES[mol_name])
            all_results[mol_name] = result
            
            # Save individual file
            output_file = args.output_dir / f"{mol_name}_reference.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Saved: {output_file}")
            
        except Exception as e:
            print(f"ERROR generating {mol_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined file
    combined_file = args.output_dir / "higher_am_reference_all.json"
    with open(combined_file, 'w') as f:
        json.dump({
            "format_version": "1.0",
            "generator": "PySCF",
            "pyscf_version": pyscf.__version__,
            "timestamp": datetime.now().isoformat(),
            "molecules": all_results
        }, f, indent=2)
    print(f"\nSaved combined reference: {combined_file}")
    
    # Summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    for name, result in all_results.items():
        basis_info = result["basis_info"]
        print(f"  {name}: max_am={basis_info['am_symbol']}, "
              f"nbasis={basis_info['n_basis']}, "
              f"E_HF={result['reference_energy']['hf']:.8f}")
    
    print(f"\nTotal: {len(all_results)} reference files generated")


if __name__ == "__main__":
    main()
