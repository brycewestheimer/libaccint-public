# Quality Gates

This document describes the quality gates enforced in LibAccInt's CI/CD pipeline. All pull requests must pass these gates before merging to the main branch.

## Overview

Quality gates ensure code quality, test coverage, and consistency across the codebase. They are automatically enforced via GitHub Actions workflows.

## Gate Categories

### 1. Build Verification

| Criterion | Requirement |
|-----------|-------------|
| Linux build | GCC 11/12/13, Clang 15/16/17 |
| macOS build | AppleClang on macOS 13/14 |
| Windows build | MSVC 2022 |
| Build configurations | Debug and Release must pass |
| Compiler warnings | Treated as errors (-Werror) |

### 2. Test Requirements

| Criterion | Requirement |
|-----------|-------------|
| Unit tests | 100% pass rate required |
| Integration tests | 100% pass rate required |
| Test timeout | No test may exceed 5 minutes |

### 3. Code Coverage

| Criterion | Threshold |
|-----------|-----------|
| Line coverage | ≥ 80% |
| Branch coverage | ≥ 70% |
| New code coverage | ≥ 70% on changed lines |

Coverage is tracked via [Codecov](https://codecov.io) and reported on each PR.

### 4. Static Analysis

| Tool | Requirement |
|------|-------------|
| clang-tidy | No high-severity findings |
| clang-format | All files must be formatted |

### 5. Documentation

| Criterion | Requirement |
|-----------|-------------|
| Public API | All public functions documented |
| Sphinx build | No warnings (using -W flag) |
| Link check | No broken internal links |

## Workflow Status Checks

The following GitHub Actions workflows must pass:

1. **CPU CI** (`cpu-ci.yml`) - Multi-platform build and test
2. **CUDA CI** (`cuda-ci.yml`) - CUDA compilation check
3. **Python CI** (`python-ci.yml`) - Python bindings build and test
5. **Coverage** (`coverage.yml`) - Code coverage collection
6. **Quality Gate** (`quality-gate.yml`) - Static analysis and formatting

## Branch Protection

The `main` branch is protected with these requirements:

- All status checks must pass
- At least 1 approving review required
- Conversations must be resolved
- Signed commits recommended

## Override Process

In exceptional circumstances, quality gates may be overridden:

1. **Who can override**: Only maintainers with admin access
2. **When to override**: Critical security fixes, CI infrastructure issues
3. **Documentation**: Override reason must be documented in PR

### Steps to Override

1. Document the reason in the PR description
2. Get approval from a second maintainer
3. Use "Merge without waiting for requirements" option
4. Create follow-up issue to address skipped checks

## Improving Quality

If you encounter quality gate failures:

### Build Failures

```bash
# Reproduce locally
cmake --preset cpu-debug
cmake --build --preset cpu-debug
```

### Test Failures

```bash
# Run tests locally
ctest --test-dir build/cpu-debug --output-on-failure

# Run specific test
ctest --test-dir build/cpu-debug -R test_name -V
```

### Coverage Issues

```bash
# Build with coverage
cmake -B build/coverage -DCMAKE_CXX_FLAGS="--coverage"
cmake --build build/coverage
ctest --test-dir build/coverage

# Generate report
lcov --capture --directory build/coverage --output-file coverage.info
genhtml coverage.info --output-directory coverage-report
```

### Formatting Issues

```bash
# Check formatting
find src include -name "*.cpp" -o -name "*.hpp" | xargs clang-format --dry-run

# Fix formatting
find src include -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i
```

## See Also

- [Contributing Guide](../../CONTRIBUTING.md)
- [CI/CD Workflows](../../.github/workflows/)
- [Code Coverage Report](https://codecov.io/gh/libaccint/libaccint)
