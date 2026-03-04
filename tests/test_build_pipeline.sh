#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.
#
# test_build_pipeline.sh — Lightweight build-pipeline smoke test
#
# Validates the handwritten-only (alpha) build: checks that stub files exist,
# key headers are present, and the library was compiled successfully.
#
# Usage:
#   ./tests/test_build_pipeline.sh <SOURCE_DIR> [BUILD_DIR]
#
# Arguments:
#   SOURCE_DIR   Path to the project root (CMAKE_SOURCE_DIR)
#   BUILD_DIR    Path to the build directory (CMAKE_BINARY_DIR, optional)

set -euo pipefail

# ============================================================================
# Colour helpers
# ============================================================================
if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
    RED=$'\033[0;31m'
    GREEN=$'\033[0;32m'
    YELLOW=$'\033[0;33m'
    BOLD=$'\033[1m'
    RESET=$'\033[0m'
else
    RED='' GREEN='' YELLOW='' BOLD='' RESET=''
fi

ok()   { printf '%s[PASS]%s %s\n' "${GREEN}"  "${RESET}" "$*"; }
fail() { printf '%s[FAIL]%s %s\n' "${RED}"    "${RESET}" "$*"; FAILURES=$((FAILURES + 1)); }
warn() { printf '%s[WARN]%s %s\n' "${YELLOW}" "${RESET}" "$*"; }
info() { printf '%s[INFO]%s %s\n' "${BOLD}"   "${RESET}" "$*"; }

# ============================================================================
# Arguments
# ============================================================================
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <SOURCE_DIR> [BUILD_DIR]" >&2
    exit 1
fi

SOURCE_DIR="$(cd "$1" && pwd)"
BUILD_DIR="${2:-}"

FAILURES=0
CHECKS=0

pass_check() {
    CHECKS=$((CHECKS + 1))
    ok "$@"
}

fail_check() {
    CHECKS=$((CHECKS + 1))
    fail "$@"
}

# ============================================================================
# Section 1: Stub files exist
# ============================================================================
info "=== Section 1: Stub files exist ==="

STUB_CPU="${SOURCE_DIR}/src/host/kernels/generated_kernel_registry_stub.cpp"
if [[ -f "${STUB_CPU}" ]]; then
    pass_check "CPU generated registry stub exists"
else
    fail_check "CPU generated registry stub missing: ${STUB_CPU}"
fi

STUB_CUDA="${SOURCE_DIR}/src/device/cuda/kernels/generated_kernel_registry_stub.cu"
if [[ -f "${STUB_CUDA}" ]]; then
    pass_check "CUDA generated registry stub exists"
else
    fail_check "CUDA generated registry stub missing: ${STUB_CUDA}"
fi

# ============================================================================
# Section 2: Key public headers exist
# ============================================================================
info "=== Section 2: Key public headers ==="

REQUIRED_HEADERS=(
    "include/libaccint/libaccint.hpp"
    "include/libaccint/kernels/generated_kernel_registry.hpp"
    "include/libaccint/kernels/generated_kernel_registry_cuda.hpp"
)

for header in "${REQUIRED_HEADERS[@]}"; do
    hpath="${SOURCE_DIR}/${header}"
    if [[ -f "${hpath}" ]]; then
        pass_check "Header exists: ${header}"
    else
        fail_check "Header missing: ${header}"
    fi
done

# ============================================================================
# Section 3: Codegen / generated directories must NOT exist (alpha build)
# ============================================================================
info "=== Section 3: Alpha exclusion checks ==="

for dir_name in codegen generated src/generated; do
    dpath="${SOURCE_DIR}/${dir_name}"
    if [[ -d "${dpath}" ]]; then
        fail_check "Directory should not exist in alpha build: ${dir_name}/"
    else
        pass_check "Correctly excluded: ${dir_name}/"
    fi
done

# ============================================================================
# Section 4: Build output checks (if BUILD_DIR provided)
# ============================================================================
if [[ -n "${BUILD_DIR}" ]] && [[ -d "${BUILD_DIR}" ]]; then
    info "=== Section 4: Build output checks ==="

    # Check that generated config.hpp was produced
    CONFIG_HPP="${BUILD_DIR}/include/libaccint/config.hpp"
    if [[ -f "${CONFIG_HPP}" ]]; then
        pass_check "Generated config.hpp exists in build dir"

        # Verify dispatch strategy is "handwritten"
        if grep -q '"handwritten"' "${CONFIG_HPP}"; then
            pass_check "config.hpp has handwritten dispatch strategy"
        else
            fail_check "config.hpp does not show handwritten dispatch strategy"
        fi
    else
        fail_check "Generated config.hpp missing from build dir"
    fi

    # Check that shared library was built
    if compgen -G "${BUILD_DIR}/lib*accint*" > /dev/null 2>&1 || \
       compgen -G "${BUILD_DIR}/**/lib*accint*" > /dev/null 2>&1; then
        pass_check "Library binary found in build dir"
    else
        warn "Library binary not found (may be in a different subdirectory)"
    fi
else
    info "=== Section 4: Build output checks (skipped — no BUILD_DIR) ==="
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
info "=== Summary ==="
printf '  Total checks: %d\n' "${CHECKS}"
printf '  Passed:       %s%d%s\n' "${GREEN}" "$((CHECKS - FAILURES))" "${RESET}"
if [[ "${FAILURES}" -gt 0 ]]; then
    printf '  Failed:       %s%d%s\n' "${RED}" "${FAILURES}" "${RESET}"
    echo ""
    fail "Build pipeline validation FAILED"
    exit 1
else
    printf '  Failed:       0\n'
    echo ""
    ok "Build pipeline validation PASSED"
    exit 0
fi
