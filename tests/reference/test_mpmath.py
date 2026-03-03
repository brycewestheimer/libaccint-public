#!/usr/bin/env python3
"""Quick test to verify mpmath is available and working."""

import sys

try:
    import mpmath as mp
    print(f"SUCCESS: mpmath version {mp.__version__} is installed")

    # Quick test of Boys function computation
    mp.dps = 50

    # Test F_0(0) = 1.0
    result = mp.mpf(1) / (2 * mp.mpf(0) + 1)
    print(f"F_0(0) = {result} (expected: 1.0)")

    # Test F_1(0) = 1/3
    result = mp.mpf(1) / (2 * mp.mpf(1) + 1)
    print(f"F_1(0) = {result} (expected: 0.333...)")

    # Test gammainc function
    result = mp.gammainc(0.5, 0, 1.0)
    print(f"gammainc(0.5, 0, 1.0) = {result}")

    print("\nAll basic tests passed!")
    sys.exit(0)

except ImportError as e:
    print(f"ERROR: mpmath is not installed", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    print(f"\nPlease install with: pip install mpmath", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Unexpected error testing mpmath", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1)
