# Installation and Setup for Reference Data Generation

## Quick Start

```bash
# 1. Install mpmath
pip install mpmath

# 2. Generate reference data
cd /home/westh/portfolio/programming/libaccint
python3 tests/reference/generate_math_reference.py

# 3. Verify outputs
ls -lh tests/data/boys_reference.json
ls -lh tests/data/rys_reference.json
```

## Detailed Instructions

### Step 1: Install mpmath

The mathematical reference generation script requires `mpmath` for arbitrary-precision arithmetic:

```bash
pip install mpmath
```

Or if using conda:
```bash
conda install -c conda-forge mpmath
```

### Step 2: Verify Installation

Test that mpmath is installed and working:

```bash
python3 tests/reference/test_mpmath.py
```

Expected output:
```
SUCCESS: mpmath version X.X.X is installed
F_0(0) = 1.0 (expected: 1.0)
F_1(0) = 0.333... (expected: 0.333...)
gammainc(0.5, 0, 1.0) = ...
All basic tests passed!
```

### Step 3: Generate Reference Data

Run the generation script:

```bash
python3 tests/reference/generate_math_reference.py
```

This will:
1. Generate Boys function reference data (~30 seconds)
2. Generate Rys quadrature reference data (~5-10 minutes)
3. Save outputs to `tests/data/boys_reference.json` and `tests/data/rys_reference.json`

Expected output:
```
LibAccInt Mathematical Reference Data Generator
============================================================
Output directory: tests/data
Precision: 50 decimal digits
mpmath version: X.X.X

Generating Boys function reference data...
  n range: [0, 30]
  T points: 200 points in [0, 100]
  Precision: 50 digits
  Computing F_{0..30}(T)... Done!

Writing Boys function reference to tests/data/boys_reference.json...
  Wrote XXX,XXX bytes
  Generated 6,200 reference values (31 n values × 200 T points)

Generating Rys quadrature reference data...
  n_roots range: [1, 10]
  T points: 50 points in [0, 50]
  Precision: 50 digits
  NOTE: This may take several minutes due to eigenvalue computations...
  Computing Rys {1..10}-point quadrature... All done!

Writing Rys quadrature reference to tests/data/rys_reference.json...
  Wrote XXX,XXX bytes
  Generated 500 reference quadrature rules (10 n_roots values × 50 T points)

============================================================
Reference data generation complete!
```

### Step 4: Verify Outputs

Check that the files were created:

```bash
ls -lh tests/data/boys_reference.json tests/data/rys_reference.json
```

Validate JSON format:

```bash
python3 -m json.tool tests/data/boys_reference.json > /dev/null && echo "boys_reference.json is valid JSON"
python3 -m json.tool tests/data/rys_reference.json > /dev/null && echo "rys_reference.json is valid JSON"
```

Quick spot check of Boys function values:

```bash
python3 -c "import json; data=json.load(open('tests/data/boys_reference.json')); print('F_0(0) =', data['boys_function']['values']['0']['0.0']); print('Expected: 1.0')"
```

## Alternative Generation Options

### Generate Only Boys Function (Faster)

If you need to iterate quickly or test the script:

```bash
python3 tests/reference/generate_math_reference.py --boys-only
```

This completes in ~30 seconds instead of ~10 minutes.

### Generate with Different Precision

```bash
python3 tests/reference/generate_math_reference.py --precision 100
```

### Generate with More T Points

```bash
python3 tests/reference/generate_math_reference.py --n-boys-points 500 --n-rys-points 100
```

## Troubleshooting

### mpmath Not Found

If you see:
```
ERROR: mpmath is not installed.
Please install it with: pip install mpmath
```

Solution:
```bash
pip install mpmath
# Or
conda install -c conda-forge mpmath
```

### Permission Denied

If you see permission errors, try:
```bash
pip install --user mpmath
```

### Import Errors

If Python can't find mpmath after installation, check your Python version:
```bash
python3 --version
pip3 --version
```

Make sure both use the same Python installation.

### Slow Performance

Rys quadrature generation is computationally intensive (eigenvalue problems).
- Use `--boys-only` flag for faster iteration
- Use `--n-rys-points 25` to reduce computation time
- The full generation takes ~10 minutes on a modern laptop

### Memory Issues

If you encounter memory errors:
- Reduce precision: `--precision 30`
- Reduce points: `--n-boys-points 100 --n-rys-points 25`

## File Locations

After successful generation:

```
tests/
├── reference/
│   ├── generate_math_reference.py     (generation script)
│   ├── test_mpmath.py                  (verification script)
│   ├── install_requirements.sh         (installation helper)
│   ├── README.md                       (documentation)
│   └── INSTALLATION.md                 (this file)
└── data/
    ├── boys_reference.json             (Boys function reference data)
    └── rys_reference.json              (Rys quadrature reference data)
```

## Next Steps

After generating the reference data:

1. Run unit tests that depend on this data (when implemented)
2. Use the reference data to validate mathematical function implementations
3. Check the tracking file: `v1_release_development/phase-0/tracking/tracking-0.6.2-math-reference-data.md`
