#!/usr/bin/env python3
"""Download basis set JSON files from the Basis Set Exchange (BSE).

Uses the basis_set_exchange Python package to fetch basis sets in MolSSI BSE
JSON schema format and saves them to share/basis_sets/. Covers Pople, Dunning,
and auxiliary basis sets for elements H-Kr (Z=1-36).

Usage:
    python scripts/download_basis_sets.py                    # Download all
    python scripts/download_basis_sets.py --list             # List configured sets
    python scripts/download_basis_sets.py --dry-run          # Show what would download

Requires: pip install basis_set_exchange

Reference: https://www.basissetexchange.org
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import basis_set_exchange as bse
except ImportError:
    print("ERROR: basis_set_exchange package not found.", file=sys.stderr)
    print("Install with: pip install basis_set_exchange", file=sys.stderr)
    sys.exit(1)

# Elements H-Kr (Z=1 through Z=36)
ELEMENTS_H_TO_KR = list(range(1, 37))

# Project root and output directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "share" / "basis_sets"

# Basis set definitions: (BSE name, output filename, category)
# BSE names must match exactly what basis_set_exchange recognizes
POPLE_BASIS_SETS: list[tuple[str, str]] = [
    ("STO-3G", "sto-3g.json"),
    ("STO-6G", "sto-6g.json"),
    ("3-21G", "3-21g.json"),
    ("6-31G", "6-31g.json"),
    ("6-31G*", "6-31g_st.json"),
    ("6-31G**", "6-31g_ss.json"),
    ("6-31+G*", "6-31+g_st.json"),
    ("6-31++G**", "6-31++g_ss.json"),
    ("6-311G", "6-311g.json"),
    ("6-311G*", "6-311g_st.json"),
    ("6-311G**", "6-311g_ss.json"),
    ("6-311+G*", "6-311+g_st.json"),
    ("6-311+G**", "6-311+g_ss.json"),
    ("6-311++G**", "6-311++g_ss.json"),
]

DUNNING_BASIS_SETS: list[tuple[str, str]] = [
    ("cc-pVDZ", "cc-pvdz.json"),
    ("cc-pVTZ", "cc-pvtz.json"),
    ("cc-pVQZ", "cc-pvqz.json"),
    ("cc-pV5Z", "cc-pv5z.json"),
    ("aug-cc-pVDZ", "aug-cc-pvdz.json"),
    ("aug-cc-pVTZ", "aug-cc-pvtz.json"),
    ("aug-cc-pVQZ", "aug-cc-pvqz.json"),
    ("aug-cc-pV5Z", "aug-cc-pv5z.json"),
]

KARLSRUHE_BASIS_SETS: list[tuple[str, str]] = [
    ("def2-SVP", "def2-svp.json"),
    ("def2-SVPD", "def2-svpd.json"),
    ("def2-TZVP", "def2-tzvp.json"),
    ("def2-TZVPD", "def2-tzvpd.json"),
    ("def2-TZVPP", "def2-tzvpp.json"),
    ("def2-TZVPPD", "def2-tzvppd.json"),
    ("def2-QZVP", "def2-qzvp.json"),
    ("def2-QZVPD", "def2-qzvpd.json"),
    ("def2-QZVPP", "def2-qzvpp.json"),
    ("def2-QZVPPD", "def2-qzvppd.json"),
]

AUXILIARY_BASIS_SETS: list[tuple[str, str]] = [
    ("cc-pVTZ-JKFIT", "cc-pvtz-jkfit.json"),
    ("cc-pVQZ-JKFIT", "cc-pvqz-jkfit.json"),
    ("cc-pV5Z-JKFIT", "cc-pv5z-jkfit.json"),
    ("cc-pVDZ-RIFIT", "cc-pvdz-rifit.json"),
    ("cc-pVTZ-RIFIT", "cc-pvtz-rifit.json"),
    ("cc-pVQZ-RIFIT", "cc-pvqz-rifit.json"),
    ("cc-pV5Z-RIFIT", "cc-pv5z-rifit.json"),
    ("def2-UNIVERSAL-JKFIT", "def2-universal-jkfit.json"),
    ("def2-SVP-RIFIT", "def2-svp-rifit.json"),
    ("def2-TZVP-RIFIT", "def2-tzvp-rifit.json"),
    ("def2-QZVP-RIFIT", "def2-qzvp-rifit.json"),
]

# Maximum angular momentum supported by the library (g-functions = 4)
MAX_ANGULAR_MOMENTUM = 4

ALL_BASIS_SETS = (
    POPLE_BASIS_SETS + DUNNING_BASIS_SETS + KARLSRUHE_BASIS_SETS + AUXILIARY_BASIS_SETS
)


def get_available_elements(bse_name: str, desired: list[int]) -> list[int]:
    """Get the intersection of desired elements and those available in BSE.

    Args:
        bse_name: BSE basis set name
        desired: List of desired atomic numbers

    Returns:
        Sorted list of available atomic numbers within the desired range
    """
    metadata = bse.get_metadata()

    # BSE metadata uses a normalized key format where * becomes _st_
    # Use bse.lookup_basis_by_role to find the canonical name, or
    # try multiple key forms
    key = bse_name.lower()

    # BSE internally converts * to _st_ in keys
    normalized = key.replace("*", "_st_")

    # Try exact match, then normalized
    for candidate in [key, normalized]:
        if candidate in metadata:
            info = metadata[candidate]
            versions = info.get("versions", {})
            if versions:
                latest = max(versions.keys())
                available = {int(e) for e in versions[latest].get("elements", [])}
                return sorted(set(desired) & available)

    # Fallback: request all and let BSE handle it
    return desired


def download_basis_set(
    bse_name: str,
    output_file: str,
    output_dir: Path,
    elements: list[int],
) -> dict[str, Any]:
    """Download a single basis set from BSE and save to disk.

    Args:
        bse_name: BSE basis set name (e.g., "STO-3G")
        output_file: Output filename (e.g., "sto-3g.json")
        output_dir: Output directory path
        elements: List of atomic numbers to include

    Returns:
        dict with download statistics
    """
    try:
        # Find available elements for this basis set within H-Kr
        available = get_available_elements(bse_name, elements)

        # Download from BSE in JSON format
        bs_json = bse.get_basis(bse_name, elements=available, fmt="json")
        data = json.loads(bs_json)

        # Filter out shells with AM > MAX_ANGULAR_MOMENTUM
        n_filtered = filter_high_am(data)

        # Count elements actually available
        n_elements = len(data.get("elements", {}))

        # Write to file with consistent formatting
        output_path = output_dir / output_file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")  # trailing newline

        return {
            "name": bse_name,
            "file": output_file,
            "n_elements": n_elements,
            "n_filtered": n_filtered,
            "status": "ok",
        }
    except Exception as e:
        return {
            "name": bse_name,
            "file": output_file,
            "n_elements": 0,
            "status": f"ERROR: {e}",
        }


def filter_high_am(data: dict[str, Any], max_am: int = MAX_ANGULAR_MOMENTUM) -> int:
    """Remove shells with angular momentum exceeding max_am from basis set data.

    Args:
        data: Parsed QCSchema JSON basis set dictionary (modified in place).
        max_am: Maximum allowed angular momentum (default: 4 for g-functions).

    Returns:
        Number of shells removed.
    """
    removed = 0
    elements = data.get("elements", {})
    for z_str, element_data in elements.items():
        shells = element_data.get("electron_shells", [])
        filtered = []
        for shell in shells:
            am_list = shell.get("angular_momentum", [])
            if any(am > max_am for am in am_list):
                removed += 1
            else:
                filtered.append(shell)
        element_data["electron_shells"] = filtered
    return removed


def list_available_sets() -> None:
    """Print configured basis sets grouped by category."""
    print("Configured basis sets:")
    print()
    for label, sets in [
        ("Pople", POPLE_BASIS_SETS),
        ("Dunning", DUNNING_BASIS_SETS),
        ("Karlsruhe def2", KARLSRUHE_BASIS_SETS),
        ("Auxiliary", AUXILIARY_BASIS_SETS),
    ]:
        print(f"  {label}:")
        for bse_name, filename in sets:
            print(f"    {bse_name:<20s}  →  {filename}")
        print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download basis sets from the Basis Set Exchange (BSE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured basis sets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without downloading",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )

    args = parser.parse_args()

    if args.list:
        list_available_sets()
        return 0

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Elements: H-Kr (Z=1-36, {len(ELEMENTS_H_TO_KR)} elements)")
    print(f"Basis sets to download: {len(ALL_BASIS_SETS)}")
    print()

    if args.dry_run:
        for bse_name, filename in ALL_BASIS_SETS:
            print(f"  Would download: {bse_name} -> {filename}")
        return 0

    results = []
    for i, (bse_name, filename) in enumerate(ALL_BASIS_SETS, 1):
        print(f"  [{i}/{len(ALL_BASIS_SETS)}] {bse_name}...", end=" ", flush=True)
        result = download_basis_set(bse_name, filename, output_dir, ELEMENTS_H_TO_KR)
        results.append(result)
        if result["status"] == "ok":
            msg = f"OK ({result['n_elements']} elements)"
            if result.get("n_filtered", 0) > 0:
                msg += f", {result['n_filtered']} high-AM shells removed"
            print(msg)
        else:
            print(result["status"])

    # Summary
    print()
    ok_results = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] != "ok"]
    print(f"Downloaded: {len(ok_results)}/{len(ALL_BASIS_SETS)} basis sets")
    if errors:
        print(f"Errors: {len(errors)}")
        for r in errors:
            print(f"  {r['name']}: {r['status']}")

    print("\nElement coverage:")
    for r in ok_results:
        print(f"  {r['file']}: {r['n_elements']} elements")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

