#!/usr/bin/env python3
"""Generate Cartesian-to-spherical transformation coefficients for LibAccInt.

Uses the Schlegel & Frisch algorithm (mpmath precision) to produce C_F (l=3)
and C_G (l=4) coefficient arrays in Racah (regular) solid harmonic convention,
with PySCF m-ordering: [0, 1, -1, 2, -2, ..., l, -l].

Validates output against the known-correct C_D (l=2) array as a sanity check.

Cartesian ordering:
  for lx in range(l, -1, -1):
      for ly in range(l-lx, -1, -1):
          lz = l - lx - ly

Storage format: row-major C[sph_index * n_cart + cart_index].

Prerequisites:
  pip install mpmath

The cart2sph transformation is implemented inline using the Schlegel & Frisch
algorithm for real solid harmonics. See:
  Schlegel, H.B.; Frisch, M.J. Int. J. Quantum Chem. 54, 83-87 (1995).
"""

import sys
import mpmath

mpmath.mp.dps = 50  # High precision for coefficient generation


def xyz2sph_real(lx, ly, lz, m):
    """Compute the real solid harmonic transformation coefficient.

    Implements the Schlegel & Frisch algorithm for the transformation
    coefficient from Cartesian Gaussian (lx, ly, lz) to real spherical
    harmonic with quantum number m and l = lx + ly + lz.

    Returns the coefficient as an mpmath high-precision float.
    """
    l = lx + ly + lz
    am = abs(m)

    # Normalization prefactor
    prefactor = mpmath.sqrt(
        mpmath.factorial(l) * mpmath.factorial(l - am)
        / (mpmath.factorial(l + am))
        * mpmath.mpf(2) ** (-l)
        / mpmath.factorial(l)
    )

    # Additional factor for m != 0
    if m != 0:
        prefactor *= mpmath.sqrt(mpmath.mpf(2))

    result = mpmath.mpf(0)

    # Sum over valid indices
    for p in range((l - am) // 2 + 1):
        for q in range(p + 1):
            for s in range(am + 1):
                # Parity check
                if m >= 0:
                    if (am - s) % 2 != 0:
                        continue
                else:
                    if (am - s) % 2 != 1:
                        continue

                # Check if powers match
                x_pow = 2 * q + am - s
                y_pow = 2 * p - 2 * q + s
                z_pow = l - 2 * p - am

                if x_pow != lx or y_pow != ly or z_pow != lz:
                    continue

                sign = (-1) ** (p + (am - s) // 2 if m >= 0 else (am - s - 1) // 2)

                coeff = (
                    mpmath.binomial(l, p)
                    * mpmath.binomial(p, q)
                    * mpmath.binomial(am, s)
                )

                result += sign * coeff

    return prefactor * result


def cartesian_indices(l):
    """Return list of (lx, ly, lz) tuples in standard ordering."""
    indices = []
    for lx in range(l, -1, -1):
        for ly in range(l - lx, -1, -1):
            lz = l - lx - ly
            indices.append((lx, ly, lz))
    return indices


def pyscf_m_order(l):
    """Return m values in PySCF convention: [0, 1, -1, 2, -2, ..., l, -l]."""
    ms = [0]
    for m in range(1, l + 1):
        ms.append(m)
        ms.append(-m)
    return ms


def generate_coefficients(l):
    """Generate the (2l+1) x n_cart transformation matrix for angular momentum l.

    The raw xyz2sph_real output gives coefficients for normalized real spherical
    harmonics Y_l^m. Our library uses Racah (regular) solid harmonics, which
    differ by a factor of sqrt(4*pi/(2*l+1)).

    Rows are reordered from ascending m [-l..l] to PySCF convention [0,1,-1,2,-2,...].
    """
    cart_idx = cartesian_indices(l)
    n_cart = len(cart_idx)
    n_sph = 2 * l + 1

    # Build raw matrix in ascending m order [-l, ..., l]
    raw = {}
    for m in range(-l, l + 1):
        row = []
        for (lx, ly, lz) in cart_idx:
            row.append(xyz2sph_real(lx, ly, lz, m))
        raw[m] = row

    # Scale: convert from normalized spherical harmonics to Racah solid harmonics
    scale = mpmath.sqrt(mpmath.mpf(4) * mpmath.pi / (2 * l + 1))

    # Reorder rows to PySCF convention
    ms = pyscf_m_order(l)
    matrix = []
    for m in ms:
        row = [float(raw[m][j] * scale) for j in range(n_cart)]
        matrix.append(row)

    return matrix, cart_idx, ms


def format_cpp_array(name, matrix, cart_idx, ms, l):
    """Format matrix as C++ array literal."""
    n_cart = len(cart_idx)
    n_sph = len(ms)

    # Map m values to labels
    am_letter = 'SPDFG'[l]

    lines = []
    for i, m in enumerate(ms):
        sign = '' if m >= 0 else ''
        label = f"{am_letter}{m} (m={m})"
        # Build comment showing which Cartesian terms have nonzero coefficients
        terms = []
        for j, (lx, ly, lz) in enumerate(cart_idx):
            if abs(matrix[i][j]) > 1e-15:
                cart_name = 'x' * lx + 'y' * ly + 'z' * lz
                terms.append(cart_name)
        lines.append(f"    // {label}: {', '.join(terms)}")

        vals = []
        for j in range(n_cart):
            v = matrix[i][j]
            if abs(v) < 1e-18:
                vals.append("0.0")
            else:
                vals.append(f"{v:.16e}")

        # Format with trailing comma except last row
        suffix = "," if i < n_sph - 1 else ""
        lines.append(f"    {', '.join(vals)}{suffix}")

    return '\n'.join(lines)


def validate_against_C_D():
    """Validate our generation against the known-correct C_D coefficients."""
    # Known correct C_D from spherical_transform.cpp
    C_D_expected = [
        # d0 (m=0)
        [-0.5, 0.0, 0.0, -0.5, 0.0, 1.0],
        # d1 (m=1)
        [0.0, 0.0, 1.7320508075688772, 0.0, 0.0, 0.0],
        # d-1 (m=-1)
        [0.0, 0.0, 0.0, 0.0, 1.7320508075688772, 0.0],
        # d2 (m=2)
        [0.8660254037844386, 0.0, 0.0, -0.8660254037844386, 0.0, 0.0],
        # d-2 (m=-2)
        [0.0, 1.7320508075688772, 0.0, 0.0, 0.0, 0.0],
    ]

    matrix, cart_idx, ms = generate_coefficients(2)

    max_err = 0.0
    for i in range(len(ms)):
        for j in range(len(cart_idx)):
            err = abs(matrix[i][j] - C_D_expected[i][j])
            max_err = max(max_err, err)

    if max_err > 1e-13:
        print(f"VALIDATION FAILED: C_D max error = {max_err:.3e}", file=sys.stderr)
        print("Generated C_D:", file=sys.stderr)
        for i, m in enumerate(ms):
            print(f"  m={m}: {matrix[i]}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"C_D validation PASSED (max error = {max_err:.3e})")


def check_row_norms(matrix, label):
    """Print row norms of the transformation matrix (informational)."""
    for i, row in enumerate(matrix):
        norm = sum(v * v for v in row) ** 0.5
        if norm < 1e-15:
            print(f"  WARNING: {label} row {i} is zero!", file=sys.stderr)


def main():
    print("Generating spherical transformation coefficients")
    print("=" * 60)

    # Validate against known-correct D-type
    validate_against_C_D()
    print()

    # Generate F-type (l=3)
    print("F-type (l=3): 7 x 10")
    f_matrix, f_cart, f_ms = generate_coefficients(3)
    check_row_norms(f_matrix, "C_F")
    print()
    print("Cartesian order:", [('x' * lx + 'y' * ly + 'z' * lz)
                               for (lx, ly, lz) in f_cart])
    print()
    print("C++ array literal for C_F:")
    print(f"const std::array<double, 70> C_F = {{")
    print(format_cpp_array("C_F", f_matrix, f_cart, f_ms, 3))
    print(f"}};")
    print()

    # Generate G-type (l=4)
    print("G-type (l=4): 9 x 15")
    g_matrix, g_cart, g_ms = generate_coefficients(4)
    check_row_norms(g_matrix, "C_G")
    print()
    print("Cartesian order:", [('x' * lx + 'y' * ly + 'z' * lz)
                               for (lx, ly, lz) in g_cart])
    print()
    print("C++ array literal for C_G:")
    print(f"const std::array<double, 135> C_G = {{")
    print(format_cpp_array("C_G", g_matrix, g_cart, g_ms, 4))
    print(f"}};")


if __name__ == '__main__':
    main()
