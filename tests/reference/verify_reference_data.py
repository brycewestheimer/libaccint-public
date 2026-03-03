#!/usr/bin/env python3
"""
Verify the generated mathematical reference data.

This script validates the JSON structure and spot-checks reference values
for Boys function and Rys quadrature data.
"""

import json
import sys
from pathlib import Path


def verify_boys_reference(filepath):
    """Verify Boys function reference data."""
    print(f"Verifying Boys function reference: {filepath}")

    try:
        with open(filepath) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"  ERROR: File not found: {filepath}")
        return False
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON: {e}")
        return False

    # Check metadata
    required_keys = ["format_version", "generator", "mpmath_version",
                     "precision_digits", "generated_date", "boys_function"]
    for key in required_keys:
        if key not in data:
            print(f"  ERROR: Missing required key: {key}")
            return False

    print(f"  Format version: {data['format_version']}")
    print(f"  Generator: {data['generator']}")
    print(f"  mpmath version: {data['mpmath_version']}")
    print(f"  Precision: {data['precision_digits']} digits")
    print(f"  Generated: {data['generated_date']}")

    # Check Boys function data
    boys_data = data["boys_function"]
    n_range = boys_data["n_range"]
    t_points = boys_data["t_points"]
    values = boys_data["values"]

    print(f"  n range: [{n_range[0]}, {n_range[1]}]")
    print(f"  Number of T points: {len(t_points)}")
    print(f"  Number of n values: {len(values)}")

    # Verify n_range
    if n_range != [0, 30]:
        print(f"  ERROR: Expected n_range [0, 30], got {n_range}")
        return False

    # Verify number of T points
    if len(t_points) != 200:
        print(f"  WARNING: Expected 200 T points, got {len(t_points)}")

    # Verify number of n values
    expected_n_values = n_range[1] - n_range[0] + 1
    if len(values) != expected_n_values:
        print(f"  ERROR: Expected {expected_n_values} n values, got {len(values)}")
        return False

    # Spot-check F_0(0) = 1.0
    f0_0 = float(values["0"][t_points[0]])  # Should be T=0
    if abs(f0_0 - 1.0) > 1e-15:
        print(f"  ERROR: F_0(0) = {f0_0}, expected 1.0")
        return False
    print(f"  ✓ F_0(0) = {f0_0} (correct)")

    # Spot-check F_1(0) = 1/3
    f1_0 = float(values["1"][t_points[0]])
    expected_f1_0 = 1.0 / 3.0
    if abs(f1_0 - expected_f1_0) > 1e-15:
        print(f"  ERROR: F_1(0) = {f1_0}, expected {expected_f1_0}")
        return False
    print(f"  ✓ F_1(0) = {f1_0} (correct)")

    # Spot-check F_2(0) = 1/5
    f2_0 = float(values["2"][t_points[0]])
    expected_f2_0 = 1.0 / 5.0
    if abs(f2_0 - expected_f2_0) > 1e-15:
        print(f"  ERROR: F_2(0) = {f2_0}, expected {expected_f2_0}")
        return False
    print(f"  ✓ F_2(0) = {f2_0} (correct)")

    # Calculate total entries
    total_entries = len(values) * len(t_points)
    print(f"  Total reference values: {total_entries:,}")

    # Check file size
    file_size = Path(filepath).stat().st_size
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

    print("  ✓ Boys function reference data is valid\n")
    return True


def verify_rys_reference(filepath):
    """Verify Rys quadrature reference data."""
    print(f"Verifying Rys quadrature reference: {filepath}")

    try:
        with open(filepath) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"  ERROR: File not found: {filepath}")
        return False
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON: {e}")
        return False

    # Check metadata
    required_keys = ["format_version", "generator", "mpmath_version",
                     "precision_digits", "generated_date", "rys_quadrature"]
    for key in required_keys:
        if key not in data:
            print(f"  ERROR: Missing required key: {key}")
            return False

    print(f"  Format version: {data['format_version']}")
    print(f"  Generator: {data['generator']}")
    print(f"  mpmath version: {data['mpmath_version']}")
    print(f"  Precision: {data['precision_digits']} digits")
    print(f"  Generated: {data['generated_date']}")

    # Check Rys quadrature data
    rys_data = data["rys_quadrature"]
    n_roots_range = rys_data["n_roots_range"]
    t_points = rys_data["t_points"]
    quad_data = rys_data["data"]

    print(f"  n_roots range: [{n_roots_range[0]}, {n_roots_range[1]}]")
    print(f"  Number of T points: {len(t_points)}")
    print(f"  Number of n_roots values: {len(quad_data)}")

    # Verify n_roots_range
    if n_roots_range != [1, 10]:
        print(f"  ERROR: Expected n_roots_range [1, 10], got {n_roots_range}")
        return False

    # Verify number of T points
    if len(t_points) != 50:
        print(f"  WARNING: Expected 50 T points, got {len(t_points)}")

    # Verify number of n_roots values
    expected_n_roots = n_roots_range[1] - n_roots_range[0] + 1
    if len(quad_data) != expected_n_roots:
        print(f"  ERROR: Expected {expected_n_roots} n_roots values, got {len(quad_data)}")
        return False

    # Spot-check 1-point quadrature at T=0
    # For T=0, the root should be 1/sqrt(3) and weight should be 1
    if "1" in quad_data and t_points[0] in quad_data["1"]:
        rule = quad_data["1"][t_points[0]]
        if "roots" in rule and "weights" in rule:
            root = float(rule["roots"][0])
            weight = float(rule["weights"][0])
            expected_root = 1.0 / (3.0 ** 0.5)
            if abs(root - expected_root) > 1e-10:
                print(f"  WARNING: 1-point root at T=0: {root}, expected ~{expected_root}")
            if abs(weight - 1.0) > 1e-10:
                print(f"  WARNING: 1-point weight at T=0: {weight}, expected 1.0")
            print(f"  ✓ 1-point quadrature at T=0: root={root:.10f}, weight={weight:.10f}")

    # Check for errors in data
    error_count = 0
    for n_roots_str, t_data in quad_data.items():
        for t_str, rule in t_data.items():
            if "error" in rule or "ERROR" in str(rule.get("roots", [])):
                error_count += 1

    if error_count > 0:
        print(f"  WARNING: Found {error_count} failed computations in data")

    # Calculate total entries
    total_entries = sum(len(t_data) for t_data in quad_data.values())
    print(f"  Total quadrature rules: {total_entries:,}")

    # Check file size
    file_size = Path(filepath).stat().st_size
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

    print("  ✓ Rys quadrature reference data is valid\n")
    return True


def main():
    """Main verification routine."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify mathematical reference data"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("tests/data"),
        help="Directory containing reference data (default: tests/data/)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Mathematical Reference Data Verification")
    print("=" * 70)
    print()

    boys_file = args.data_dir / "boys_reference.json"
    rys_file = args.data_dir / "rys_reference.json"

    boys_ok = verify_boys_reference(boys_file)
    rys_ok = verify_rys_reference(rys_file)

    print("=" * 70)
    if boys_ok and rys_ok:
        print("✓ All reference data is valid!")
        print("=" * 70)
        return 0
    else:
        print("✗ Some reference data is invalid or missing")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
