#!/usr/bin/env python3
"""
Generate high-precision reference data for Boys function and Rys quadrature.

This script uses mpmath for arbitrary-precision arithmetic to generate reference
data for mathematical functions used in LibAccInt molecular integral evaluation.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import mpmath as mp
except ImportError:
    print("ERROR: mpmath is not installed.", file=sys.stderr)
    print("Please install it with: pip install mpmath", file=sys.stderr)
    sys.exit(1)


def generate_t_points_boys(n_points=200):
    """
    Generate T points for Boys function with denser spacing near 0 and regime boundaries.

    Distribution:
    - 50 points in [0, 1] (dense near T=0, where F_n changes rapidly)
    - 50 points in [1, 10]
    - 50 points in [10, 35] (crossover regime)
    - 50 points in [35, 100] (asymptotic regime)
    """
    points = []

    # [0, 1]: dense near 0
    for i in range(50):
        t = (i / 49) ** 2  # Quadratic spacing for extra density near 0
        points.append(t)

    # [1, 10]: linear spacing
    for i in range(50):
        t = 1.0 + (i / 49) * 9.0
        points.append(t)

    # [10, 35]: linear spacing
    for i in range(50):
        t = 10.0 + (i / 49) * 25.0
        points.append(t)

    # [35, 100]: linear spacing
    for i in range(50):
        t = 35.0 + (i / 49) * 65.0
        points.append(t)

    return sorted(set(points))  # Remove any duplicates and sort


def generate_t_points_rys(n_points=50):
    """
    Generate T points for Rys quadrature with denser spacing near 0.

    T range: [0, 50] with denser spacing near 0 where roots change rapidly.
    """
    points = []

    # Use a combination of linear and quadratic spacing
    for i in range(n_points):
        # Mix of linear (70%) and quadratic (30%) spacing
        linear = (i / (n_points - 1)) * 50.0
        quadratic = ((i / (n_points - 1)) ** 1.5) * 50.0
        t = 0.3 * quadratic + 0.7 * linear
        points.append(t)

    return points


def boys_function(n, T, precision=50):
    """
    Compute Boys function F_n(T) with high precision.

    F_n(T) = integral_0^1 t^(2n) * exp(-T * t^2) dt

    For T = 0: F_n(0) = 1/(2n+1)
    For T > 0: F_n(T) = 0.5 * T^(-n-0.5) * gammainc(n+0.5, 0, T)
               where gammainc is the lower incomplete gamma function.
    """
    mp.dps = precision

    T_mp = mp.mpf(T)
    n_mp = mp.mpf(n)

    if T == 0:
        # F_n(0) = 1/(2n+1)
        return mp.mpf(1) / (2 * n_mp + 1)
    else:
        # F_n(T) = 0.5 * T^(-n-0.5) * gammainc(n+0.5, 0, T)
        # mpmath.gammainc(a, z1, z2) computes the generalized incomplete gamma function
        # For lower incomplete gamma: gammainc(a, 0, z)
        n_plus_half = n_mp + mp.mpf(0.5)
        gamma_lower = mp.gammainc(n_plus_half, 0, T_mp)
        result = mp.mpf(0.5) * mp.power(T_mp, -n_plus_half) * gamma_lower
        return result


def generate_boys_reference(n_max=30, n_points=200, precision=50):
    """
    Generate reference data for Boys function.

    Args:
        n_max: Maximum n value (inclusive)
        n_points: Number of T points to sample
        precision: Number of decimal digits for mpmath precision

    Returns:
        Dictionary with Boys function reference data
    """
    print(f"Generating Boys function reference data...")
    print(f"  n range: [0, {n_max}]")
    print(f"  T points: {n_points} points in [0, 100]")
    print(f"  Precision: {precision} digits")

    mp.dps = precision

    t_points = generate_t_points_boys(n_points)
    t_points_str = [str(mp.nstr(mp.mpf(t), precision)) for t in t_points]

    values = {}
    for n in range(n_max + 1):
        print(f"  Computing F_{n}(T)...", end='\r')
        values[str(n)] = {}
        for t in t_points:
            fn_t = boys_function(n, t, precision)
            t_str = str(mp.nstr(mp.mpf(t), precision))
            values[str(n)][t_str] = str(mp.nstr(fn_t, precision))

    print(f"  Computing F_{{0..{n_max}}}(T)... Done!")

    data = {
        "format_version": "1.0",
        "generator": "mpmath",
        "mpmath_version": mp.__version__,
        "precision_digits": precision,
        "generated_date": datetime.now().isoformat(),
        "boys_function": {
            "n_range": [0, n_max],
            "t_points": t_points_str,
            "n_t_points": len(t_points),
            "values": values
        }
    }

    return data


def compute_rys_roots_weights_golub_welsch(n_roots, T, precision=50):
    """
    Compute Rys quadrature roots and weights using the Golub-Welsch algorithm.

    The Rys quadrature is Gaussian quadrature on [0,1] with weight function
    w(t) = exp(-T*t^2). The moments are related to the Boys function:
    mu_{2k} = F_k(T) and mu_{2k+1} = F_k(T)/(2k+2) for even/odd moments.

    Args:
        n_roots: Number of quadrature points
        T: Parameter value
        precision: Number of decimal digits

    Returns:
        (roots, weights) as lists of mpmath numbers
    """
    mp.dps = precision

    # Compute moments: mu_k = integral_0^1 t^k * exp(-T*t^2) dt
    # We can relate these to Boys function values
    # mu_k = integral_0^1 t^k * exp(-T*t^2) dt
    # For even k=2m: mu_{2m} = F_m(T)
    # For odd k=2m+1: mu_{2m+1} = integral_0^1 t^(2m+1) * exp(-T*t^2) dt
    #                           = [make substitution u = T*t^2, du = 2Tt dt]
    #                           = (1/(2T)) * integral_0^T u^m * exp(-u) du
    #                           = (1/(2T)) * gamma(m+1) * (1 - gammainc(m+1, T)/gamma(m+1))
    #                           = (1/(2T)) * (gamma(m+1) - gammainc(m+1, 0, T))

    # Actually, for odd moments, there's a simpler relation:
    # mu_{2m+1} = (1 - exp(-T) * sum_{k=0}^m T^k/k!) / (2T) for the normalized form
    # But let's compute directly via numerical integration for robustness

    # Compute the first 2*n_roots moments
    moments = []
    for k in range(2 * n_roots):
        # mu_k = integral_0^1 t^k * exp(-T*t^2) dt
        integrand = lambda t: t**k * mp.exp(-T * t**2)
        mu_k = mp.quad(integrand, [0, 1])
        moments.append(mu_k)

    # Use the Golub-Welsch algorithm: construct the Jacobi matrix
    # from the three-term recurrence relation via the modified Chebyshev algorithm

    # Modified Chebyshev algorithm to get recurrence coefficients
    # P_{k+1}(x) = (x - alpha_k) * P_k(x) - beta_k * P_{k-1}(x)
    alpha = []
    beta = []
    sigma = [[mp.mpf(0) for _ in range(2 * n_roots + 1)] for _ in range(2 * n_roots + 1)]

    # Initialize sigma matrix with moments
    for l in range(2 * n_roots + 1):
        sigma[l][0] = moments[l] if l < len(moments) else mp.mpf(0)

    # Modified Chebyshev algorithm
    for k in range(n_roots):
        # Compute alpha_k
        if k == 0:
            alpha_k = sigma[1][k] / sigma[0][k]
        else:
            alpha_k = sigma[1][k] / sigma[0][k] - sigma[1][k-1] / sigma[0][k-1]
        alpha.append(alpha_k)

        # Compute beta_k
        if k == 0:
            beta_k = mp.mpf(0)
        else:
            beta_k = sigma[0][k] / sigma[0][k-1]
        beta.append(beta_k)

        # Update sigma for next iteration
        for l in range(2 * n_roots - k):
            sigma[l][k+1] = sigma[l+1][k] - alpha_k * sigma[l][k]
            if k > 0:
                sigma[l][k+1] -= beta_k * sigma[l][k-1]

    # Construct the Jacobi matrix
    # J = tridiag(beta_1, ..., beta_{n-1} | alpha_0, ..., alpha_{n-1} | beta_1, ..., beta_{n-1})
    J = mp.matrix(n_roots, n_roots)
    for i in range(n_roots):
        J[i, i] = alpha[i]
        if i < n_roots - 1:
            sqrt_beta = mp.sqrt(beta[i + 1])
            J[i, i + 1] = sqrt_beta
            J[i + 1, i] = sqrt_beta

    # Compute eigenvalues and eigenvectors
    # Roots are eigenvalues, weights are mu_0 * (first component of eigenvector)^2
    eigenvalues, eigenvectors = mp.eig(J)

    # Sort by eigenvalue (root)
    pairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(n_roots)]
    pairs.sort(key=lambda x: x[0])

    roots = [pair[0] for pair in pairs]
    weights = [moments[0] * pair[1][0]**2 for pair in pairs]

    return roots, weights


def generate_rys_reference(n_roots_max=10, n_points=50, precision=50):
    """
    Generate reference data for Rys quadrature.

    Args:
        n_roots_max: Maximum number of roots (inclusive)
        n_points: Number of T points to sample
        precision: Number of decimal digits for mpmath precision

    Returns:
        Dictionary with Rys quadrature reference data
    """
    print(f"\nGenerating Rys quadrature reference data...")
    print(f"  n_roots range: [1, {n_roots_max}]")
    print(f"  T points: {n_points} points in [0, 50]")
    print(f"  Precision: {precision} digits")
    print(f"  NOTE: This may take several minutes due to eigenvalue computations...")

    mp.dps = precision

    t_points = generate_t_points_rys(n_points)
    t_points_str = [str(mp.nstr(mp.mpf(t), precision)) for t in t_points]

    data_dict = {}

    for n_roots in range(1, n_roots_max + 1):
        print(f"  Computing Rys {n_roots}-point quadrature...", end='\r')
        data_dict[str(n_roots)] = {}

        for t_idx, t in enumerate(t_points):
            try:
                roots, weights = compute_rys_roots_weights_golub_welsch(n_roots, t, precision)

                t_str = str(mp.nstr(mp.mpf(t), precision))
                data_dict[str(n_roots)][t_str] = {
                    "roots": [str(mp.nstr(r, precision)) for r in roots],
                    "weights": [str(mp.nstr(w, precision)) for w in weights]
                }

                # Verify: sum of weights should approximately equal F_0(T)
                weight_sum = sum(weights)
                f0_t = boys_function(0, t, precision)
                rel_error = abs(weight_sum - f0_t) / (abs(f0_t) + 1e-100)

                if rel_error > 1e-30:  # Very strict tolerance for high precision
                    print(f"\n  WARNING: n_roots={n_roots}, T={t:.4f}: weight sum verification failed")
                    print(f"           sum(weights) = {weight_sum}, F_0(T) = {f0_t}, rel_error = {rel_error}")

            except Exception as e:
                print(f"\n  ERROR: Failed for n_roots={n_roots}, T={t:.4f}: {e}")
                # Store placeholder data
                data_dict[str(n_roots)][t_str] = {
                    "roots": ["ERROR"] * n_roots,
                    "weights": ["ERROR"] * n_roots,
                    "error": str(e)
                }

        print(f"  Computing Rys {n_roots}-point quadrature... Done!")

    print(f"  Computing Rys {{1..{n_roots_max}}}-point quadrature... All done!")

    data = {
        "format_version": "1.0",
        "generator": "mpmath",
        "mpmath_version": mp.__version__,
        "precision_digits": precision,
        "generated_date": datetime.now().isoformat(),
        "rys_quadrature": {
            "n_roots_range": [1, n_roots_max],
            "t_points": t_points_str,
            "n_t_points": len(t_points),
            "data": data_dict
        }
    }

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate high-precision reference data for Boys function and Rys quadrature"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/data"),
        help="Output directory for reference data files (default: tests/data/)"
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=50,
        help="Number of decimal digits for mpmath precision (default: 50)"
    )
    parser.add_argument(
        "--n-boys-points",
        type=int,
        default=200,
        help="Number of T points for Boys function (default: 200)"
    )
    parser.add_argument(
        "--n-rys-points",
        type=int,
        default=50,
        help="Number of T points for Rys quadrature (default: 50)"
    )
    parser.add_argument(
        "--boys-only",
        action="store_true",
        help="Generate only Boys function reference data"
    )
    parser.add_argument(
        "--rys-only",
        action="store_true",
        help="Generate only Rys quadrature reference data"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"LibAccInt Mathematical Reference Data Generator")
    print(f"=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Precision: {args.precision} decimal digits")
    print(f"mpmath version: {mp.__version__}")
    print()

    # Generate Boys function reference
    if not args.rys_only:
        boys_data = generate_boys_reference(
            n_max=30,
            n_points=args.n_boys_points,
            precision=args.precision
        )

        boys_file = args.output_dir / "boys_reference.json"
        print(f"\nWriting Boys function reference to {boys_file}...")
        with open(boys_file, 'w') as f:
            json.dump(boys_data, f, indent=2)
        print(f"  Wrote {boys_file.stat().st_size:,} bytes")

        # Report statistics
        n_values = len(boys_data["boys_function"]["values"])
        n_t_points = boys_data["boys_function"]["n_t_points"]
        total_entries = n_values * n_t_points
        print(f"  Generated {total_entries:,} reference values ({n_values} n values × {n_t_points} T points)")

    # Generate Rys quadrature reference
    if not args.boys_only:
        rys_data = generate_rys_reference(
            n_roots_max=10,
            n_points=args.n_rys_points,
            precision=args.precision
        )

        rys_file = args.output_dir / "rys_reference.json"
        print(f"\nWriting Rys quadrature reference to {rys_file}...")
        with open(rys_file, 'w') as f:
            json.dump(rys_data, f, indent=2)
        print(f"  Wrote {rys_file.stat().st_size:,} bytes")

        # Report statistics
        n_roots_values = len(rys_data["rys_quadrature"]["data"])
        n_t_points = rys_data["rys_quadrature"]["n_t_points"]
        total_entries = n_roots_values * n_t_points
        print(f"  Generated {total_entries:,} reference quadrature rules ({n_roots_values} n_roots values × {n_t_points} T points)")

    print(f"\n{'=' * 60}")
    print("Reference data generation complete!")


if __name__ == "__main__":
    main()
