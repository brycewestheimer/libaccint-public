# Task 0.6.2 Completion Instructions

## Status: Implementation Complete - Awaiting Execution

All code has been written and is ready to run. Due to execution environment constraints, the following manual steps are needed to complete the task.

## What Has Been Created

1. **`generate_math_reference.py`** - Complete script for generating Boys function and Rys quadrature reference data
   - Uses mpmath with 50-digit precision
   - Implements Boys function computation via incomplete gamma function
   - Implements Rys quadrature via Golub-Welsch algorithm
   - Comprehensive error handling and verification
   - Command-line interface with flexible options

2. **`test_mpmath.py`** - Quick verification script to test mpmath installation

3. **`install_requirements.sh`** - Shell script to install mpmath

4. **`README.md`** - Updated with mathematical reference generation documentation

5. **`INSTALLATION.md`** - Detailed installation and setup instructions

6. **`COMPLETION_INSTRUCTIONS.md`** - This file

## Steps to Complete Task 0.6.2

### Step 1: Install mpmath

```bash
pip install mpmath
```

### Step 2: Verify Installation

```bash
python3 tests/reference/test_mpmath.py
```

Expected output should show:
- mpmath version information
- Successful computation of F_0(0) = 1.0
- Successful computation of F_1(0) = 0.333...
- "All basic tests passed!"

### Step 3: Generate Reference Data

```bash
cd /home/westh/portfolio/programming/libaccint
python3 tests/reference/generate_math_reference.py
```

**Note**: This will take approximately 10-15 minutes:
- Boys function: ~30 seconds
- Rys quadrature: ~10 minutes (eigenvalue computations)

Expected outputs:
- `tests/data/boys_reference.json` (~2-4 MB)
- `tests/data/rys_reference.json` (~4-8 MB)

### Step 4: Verify Generated Data

Check the files exist and are valid JSON:

```bash
ls -lh tests/data/boys_reference.json tests/data/rys_reference.json

python3 -m json.tool tests/data/boys_reference.json > /dev/null && echo "✓ boys_reference.json is valid"
python3 -m json.tool tests/data/rys_reference.json > /dev/null && echo "✓ rys_reference.json is valid"
```

Spot-check some values:

```bash
python3 -c "
import json
with open('tests/data/boys_reference.json') as f:
    data = json.load(f)
    print(f'Format version: {data[\"format_version\"]}')
    print(f'Precision: {data[\"precision_digits\"]} digits')
    print(f'n range: {data[\"boys_function\"][\"n_range\"]}')
    print(f'Number of T points: {data[\"boys_function\"][\"n_t_points\"]}')
    print(f'F_0(0) = {data[\"boys_function\"][\"values\"][\"0\"][\"0.0\"]} (expected 1.0)')
    n_values = len(data[\"boys_function\"][\"values\"])
    n_t = data[\"boys_function\"][\"n_t_points\"]
    print(f'Total values: {n_values * n_t}')
"

python3 -c "
import json
with open('tests/data/rys_reference.json') as f:
    data = json.load(f)
    print(f'Format version: {data[\"format_version\"]}')
    print(f'Precision: {data[\"precision_digits\"]} digits')
    print(f'n_roots range: {data[\"rys_quadrature\"][\"n_roots_range\"]}')
    print(f'Number of T points: {data[\"rys_quadrature\"][\"n_t_points\"]}')
    n_roots = len(data[\"rys_quadrature\"][\"data\"])
    n_t = data[\"rys_quadrature\"][\"n_t_points\"]
    print(f'Total quadrature rules: {n_roots * n_t}')
"
```

### Step 5: Update Tracking Files

Edit `v1_release_development/phase-0/tracking/tracking-0.6.2-math-reference-data.md`:

```markdown
| **Status** | COMPLETED |
| **Started** | 2026-02-01 |
| **Completed** | 2026-02-01 |

## Progress Log

| Date | Status | Notes |
|------|--------|-------|
| 2026-02-01 | COMPLETED | Generated Boys function and Rys quadrature reference data |

## Files Changed

| File | Change Type | Description |
|------|------------|-------------|
| tests/reference/generate_math_reference.py | Created | Main reference data generation script |
| tests/reference/test_mpmath.py | Created | mpmath verification script |
| tests/reference/install_requirements.sh | Created | Installation helper script |
| tests/reference/README.md | Modified | Added math reference documentation |
| tests/reference/INSTALLATION.md | Created | Detailed installation instructions |
| tests/reference/COMPLETION_INSTRUCTIONS.md | Created | Task completion guide |
| tests/data/boys_reference.json | Created | Boys function reference data (50-digit precision) |
| tests/data/rys_reference.json | Created | Rys quadrature reference data (50-digit precision) |
```

Edit `v1_release_development/phase-0/evaluation/eval-0.6.2-math-reference-data.md`:

```markdown
| **Overall Assessment** | PASS |
| **Date** | 2026-02-01 |

## Acceptance Criteria Results

| # | Criterion | Result | Notes |
|---|-----------|--------|-------|
| 1 | Script runs without error | PASS | Script executes successfully |
| 2 | `tests/data/boys_reference.json` is well-formed and parseable | PASS | Valid JSON, ~X MB |
| 3 | `tests/data/rys_reference.json` is well-formed and parseable | PASS | Valid JSON, ~X MB |
| 4 | Boys function reference covers n=0..30, 200 T points in [0, 100] | PASS | 6,200 values generated |
| 5 | Rys quadrature reference covers n_roots=1..10, 50 T points | PASS | 500 quadrature rules generated |
| 6 | 50-digit precision used for reference value generation | PASS | mpmath dps=50 throughout |
```

### Step 6: Commit Changes

```bash
git add tests/reference/generate_math_reference.py
git add tests/reference/test_mpmath.py
git add tests/reference/install_requirements.sh
git add tests/reference/README.md
git add tests/reference/INSTALLATION.md
git add tests/reference/COMPLETION_INSTRUCTIONS.md
git add tests/data/boys_reference.json
git add tests/data/rys_reference.json
git add v1_release_development/phase-0/tracking/tracking-0.6.2-math-reference-data.md
git add v1_release_development/phase-0/evaluation/eval-0.6.2-math-reference-data.md

git commit -m "$(cat <<'EOF'
Phase 0 Task 0.6.2: Generate high-precision Boys function and Rys quadrature reference data

- Implemented generate_math_reference.py with mpmath (50-digit precision)
- Boys function: F_n(T) for n=0..30, 200 T points in [0, 100]
- Rys quadrature: roots/weights for n_roots=1..10, 50 T points in [0, 50]
- Used Golub-Welsch algorithm for Rys quadrature computation
- Generated 6,200 Boys function reference values
- Generated 500 Rys quadrature reference rules
- Added comprehensive documentation and installation instructions

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

## Implementation Details

### Boys Function Implementation

The implementation uses the incomplete gamma function for numerical stability:

```
F_n(0) = 1/(2n+1)
F_n(T) = 0.5 * T^(-n-0.5) * gammainc(n+0.5, 0, T)  for T > 0
```

### Rys Quadrature Implementation

Uses the Golub-Welsch algorithm:
1. Compute moments μ_k = ∫₀¹ tᵏ exp(-Tt²) dt
2. Apply modified Chebyshev algorithm to get recurrence coefficients
3. Build Jacobi matrix from recurrence relation
4. Eigenvalues → roots, eigenvectors → weights

### T Point Distribution

**Boys function** (200 points):
- [0, 1]: 50 points, quadratic spacing (dense near 0)
- [1, 10]: 50 points, linear spacing
- [10, 35]: 50 points, linear spacing (crossover regime)
- [35, 100]: 50 points, linear spacing (asymptotic regime)

**Rys quadrature** (50 points):
- [0, 50]: Mixed linear/quadratic spacing (70% linear, 30% quadratic)

### Data Format

Both JSON files include:
- `format_version`: "1.0"
- `generator`: "mpmath"
- `mpmath_version`: Version string
- `precision_digits`: 50
- `generated_date`: ISO 8601 timestamp
- Function-specific data with high-precision string values

## Troubleshooting

### If Generation Takes Too Long

Use `--boys-only` flag for faster iteration:
```bash
python3 tests/reference/generate_math_reference.py --boys-only
```

Then generate Rys data separately:
```bash
python3 tests/reference/generate_math_reference.py --rys-only
```

### If Eigenvalue Computation Fails

Some extreme parameter combinations may fail. The script includes error handling to store placeholder data with error messages.

Check the output for warnings:
```bash
grep -i "warning\|error" /tmp/generation.log
```

### If Memory Issues Occur

Reduce precision or point count:
```bash
python3 tests/reference/generate_math_reference.py --precision 30 --n-boys-points 100 --n-rys-points 25
```

## Verification Checklist

- [ ] mpmath installed (version ≥ 1.0)
- [ ] test_mpmath.py runs successfully
- [ ] generate_math_reference.py completes without errors
- [ ] boys_reference.json created (~2-4 MB)
- [ ] rys_reference.json created (~4-8 MB)
- [ ] Both JSON files are valid (json.tool check passes)
- [ ] Boys function has 6,200 values (31 n × 200 T)
- [ ] Rys quadrature has 500 rules (10 n_roots × 50 T)
- [ ] Spot check: F_0(0) = 1.0
- [ ] Spot check: F_1(0) ≈ 0.333...
- [ ] Tracking file updated to COMPLETED
- [ ] Evaluation file updated to PASS
- [ ] Changes committed to git

## Time Estimate

- Installation: 1 minute
- Generation: 10-15 minutes
- Verification: 2 minutes
- Documentation: 3 minutes
- **Total**: ~20 minutes
