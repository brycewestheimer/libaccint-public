#!/bin/bash
# Install requirements for reference data generation

echo "Installing mpmath for mathematical reference data generation..."
pip install mpmath

echo ""
echo "Verifying installation..."
python3 -c "import mpmath; print(f'mpmath version {mpmath.__version__} installed successfully')"

echo ""
echo "Installation complete! You can now run:"
echo "  python3 tests/reference/generate_math_reference.py"
