#!/usr/bin/env python3
"""Simple test to verify generate_reference.py syntax."""
import ast
import sys

try:
    with open('generate_reference.py', 'r') as f:
        code = f.read()
    ast.parse(code)
    print("Syntax check: PASSED")
    sys.exit(0)
except SyntaxError as e:
    print(f"Syntax check: FAILED - {e}")
    sys.exit(1)
