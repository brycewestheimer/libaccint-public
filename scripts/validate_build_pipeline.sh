#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.
#
# validate_build_pipeline.sh — End-to-end build pipeline validation
#
# Step 9.5 of the kernel codegen refactoring plan.
# Validates the full pipeline: codegen → configure → build → test.
#
# Usage:
#   ./scripts/validate_build_pipeline.sh [OPTIONS]
#
# Options:
#   --preset <name>   CMake configure/build/test preset (default: cpu-debug)
#   --skip-codegen    Skip code generation step
#   --clean           Remove build directory before configuring
#   --jobs <N>        Parallel build jobs (default: nproc)
#   --help            Show this help message

set -euo pipefail

# ============================================================================
# Colour helpers (disabled when not a terminal or NO_COLOR is set)
# ============================================================================
if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
    RED=$'\033[0;31m'
    GREEN=$'\033[0;32m'
    YELLOW=$'\033[0;33m'
    BLUE=$'\033[0;34m'
    BOLD=$'\033[1m'
    RESET=$'\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' BOLD='' RESET=''
fi

# ============================================================================
# Logging helpers
# ============================================================================
info()  { printf '%s[INFO]%s  %s\n'  "${BLUE}"   "${RESET}" "$*"; }
ok()    { printf '%s[ OK ]%s  %s\n'  "${GREEN}"  "${RESET}" "$*"; }
warn()  { printf '%s[WARN]%s  %s\n'  "${YELLOW}" "${RESET}" "$*"; }
fail()  { printf '%s[FAIL]%s  %s\n'  "${RED}"    "${RESET}" "$*"; }
stage() { printf '\n%s══════ %s ══════%s\n' "${BOLD}" "$*" "${RESET}"; }

die() { fail "$@"; exit 1; }

# ============================================================================
# Resolve project root (parent of scripts/)
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================================
# Defaults
# ============================================================================
PRESET="cpu-debug"
SKIP_CODEGEN=0
CLEAN_BUILD=0
JOBS=""
STAGE_TIMES=()
STAGE_NAMES=()
ERRORS=0

# ============================================================================
# Argument parsing
# ============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)
            PRESET="$2"; shift 2 ;;
        --skip-codegen)
            SKIP_CODEGEN=1; shift ;;
        --clean)
            CLEAN_BUILD=1; shift ;;
        --jobs)
            JOBS="$2"; shift 2 ;;
        --help|-h)
            head -20 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *)
            die "Unknown option: $1 (try --help)" ;;
    esac
done

# Default parallel jobs
if [[ -z "${JOBS}" ]]; then
    if command -v nproc &>/dev/null; then
        JOBS="$(nproc)"
    elif command -v sysctl &>/dev/null; then
        JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
    else
        JOBS=4
    fi
fi

BUILD_DIR="${PROJECT_ROOT}/build/${PRESET}"

# ============================================================================
# Timer helpers
# ============================================================================
_timer_start=""
timer_start() { _timer_start="$(date +%s)"; }
timer_stop() {
    local end
    end="$(date +%s)"
    local elapsed=$(( end - _timer_start ))
    STAGE_TIMES+=("${elapsed}")
    STAGE_NAMES+=("$1")
    info "$1 completed in ${elapsed}s"
}

# ============================================================================
# Stage 1: Check prerequisites
# ============================================================================
stage "Stage 1: Checking prerequisites"
timer_start

check_cmd() {
    local cmd="$1"
    local label="${2:-$1}"
    if command -v "${cmd}" &>/dev/null; then
        local ver
        ver="$("${cmd}" --version 2>&1 | head -1)" || ver="(unknown version)"
        ok "${label}: ${ver}"
    else
        fail "${label} not found"
        ERRORS=$((ERRORS + 1))
    fi
}

# Python 3
if command -v python3 &>/dev/null; then
    PY_VER="$(python3 --version 2>&1)"
    ok "Python: ${PY_VER}"
else
    fail "python3 not found"
    ERRORS=$((ERRORS + 1))
fi

# CMake ≥ 3.20
if command -v cmake &>/dev/null; then
    CMAKE_VER_STR="$(cmake --version | head -1)"
    CMAKE_VER="$(echo "${CMAKE_VER_STR}" | grep -oE '[0-9]+\.[0-9]+' | head -1)"
    CMAKE_MAJOR="${CMAKE_VER%%.*}"
    CMAKE_MINOR="${CMAKE_VER#*.}"
    if [[ "${CMAKE_MAJOR}" -gt 3 ]] || { [[ "${CMAKE_MAJOR}" -eq 3 ]] && [[ "${CMAKE_MINOR}" -ge 20 ]]; }; then
        ok "CMake: ${CMAKE_VER_STR}"
    else
        fail "CMake >= 3.20 required, found ${CMAKE_VER}"
        ERRORS=$((ERRORS + 1))
    fi
else
    fail "cmake not found"
    ERRORS=$((ERRORS + 1))
fi

# C++ compiler
CXX_CMD="${CXX:-}"
if [[ -z "${CXX_CMD}" ]]; then
    for candidate in g++ clang++ c++; do
        if command -v "${candidate}" &>/dev/null; then
            CXX_CMD="${candidate}"
            break
        fi
    done
fi
if [[ -n "${CXX_CMD}" ]] && command -v "${CXX_CMD}" &>/dev/null; then
    CXX_VER="$("${CXX_CMD}" --version 2>&1 | head -1)"
    ok "C++ compiler: ${CXX_VER}"
else
    fail "No C++ compiler found (set CXX env var)"
    ERRORS=$((ERRORS + 1))
fi

# Build tool (ninja preferred, make fallback)
if command -v ninja &>/dev/null; then
    ok "Build tool: $(ninja --version 2>&1 | head -1) (ninja)"
elif command -v make &>/dev/null; then
    warn "ninja not found; falling back to make"
    ok "Build tool: $(make --version 2>&1 | head -1)"
else
    fail "Neither ninja nor make found"
    ERRORS=$((ERRORS + 1))
fi

if [[ "${ERRORS}" -gt 0 ]]; then
    die "Prerequisite checks failed (${ERRORS} error(s))"
fi

timer_stop "Prerequisites"

# ============================================================================
# Stage 2: Code generation
# ============================================================================
if [[ "${SKIP_CODEGEN}" -eq 0 ]]; then
    stage "Stage 2: Code generation"
    timer_start

    CODEGEN_DIR="${PROJECT_ROOT}/codegen"

    # Install codegen package in editable mode if not already available
    if ! python3 -c "import libaccint_codegen" &>/dev/null; then
        info "Installing libaccint-codegen package..."
        python3 -m pip install -e "${CODEGEN_DIR}" --quiet || \
            die "Failed to install libaccint-codegen"
        ok "libaccint-codegen installed"
    else
        ok "libaccint-codegen already available"
    fi

    # Run codegen for both CPU and CUDA backends
    info "Running code generation (--max-am 4 --backends cpu cuda --all-k-ranges)..."
    python3 -m libaccint_codegen.cli \
        --max-am 4 \
        --backends cpu cuda \
        --all-k-ranges \
        --output "${PROJECT_ROOT}/generated" \
        || die "Code generation failed"

    # Verify output
    local_errors=0
    for backend in cpu cuda; do
        for k_range in small medium large; do
            dir="${PROJECT_ROOT}/generated/${backend}/${k_range}"
            if [[ -d "${dir}" ]]; then
                count="$(find "${dir}" -maxdepth 1 -type f | wc -l)"
                ok "${backend}/${k_range}: ${count} files"
            else
                fail "Missing directory: generated/${backend}/${k_range}"
                local_errors=$((local_errors + 1))
            fi
        done
    done

    # Check generated_sources.cmake
    cmake_file="${PROJECT_ROOT}/generated/cuda/generated_sources.cmake"
    if [[ -f "${cmake_file}" ]]; then
        ok "generated_sources.cmake exists"
    else
        fail "generated_sources.cmake missing"
        local_errors=$((local_errors + 1))
    fi

    if [[ "${local_errors}" -gt 0 ]]; then
        die "Code generation verification failed (${local_errors} error(s))"
    fi

    timer_stop "Code generation"
else
    info "Skipping code generation (--skip-codegen)"
fi

# ============================================================================
# Stage 3: CMake configure
# ============================================================================
stage "Stage 3: CMake configure (preset: ${PRESET})"
timer_start

if [[ "${CLEAN_BUILD}" -eq 1 ]] && [[ -d "${BUILD_DIR}" ]]; then
    info "Cleaning build directory: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

# Try preset first, fall back to manual configuration
if cmake --list-presets 2>/dev/null | grep -q "${PRESET}" 2>/dev/null; then
    info "Using CMake preset: ${PRESET}"
    cmake --preset "${PRESET}" -S "${PROJECT_ROOT}" \
        || die "CMake configure (preset) failed"
else
    warn "Preset '${PRESET}' not available, falling back to manual configuration"
    mkdir -p "${BUILD_DIR}"
    cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DLIBACCINT_USE_CUDA=OFF \
        -DLIBACCINT_USE_OPENMP=ON \
        -DLIBACCINT_BUILD_TESTS=ON \
        -DLIBACCINT_BUILD_EXAMPLES=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        || die "CMake configure (manual) failed"
fi

ok "CMake configuration succeeded"
timer_stop "CMake configure"

# ============================================================================
# Stage 4: Build
# ============================================================================
stage "Stage 4: Build"
timer_start

# Try preset-based build, fall back to direct cmake --build
if cmake --build --list-presets 2>/dev/null | grep -q "${PRESET}" 2>/dev/null; then
    cmake --build --preset "${PRESET}" --parallel "${JOBS}" \
        || die "Build failed"
else
    cmake --build "${BUILD_DIR}" --parallel "${JOBS}" \
        || die "Build failed"
fi

ok "Build succeeded"
timer_stop "Build"

# ============================================================================
# Stage 5: Run tests
# ============================================================================
stage "Stage 5: Run tests"
timer_start

TEST_RESULT_FILE="$(mktemp)"
trap 'rm -f "${TEST_RESULT_FILE}"' EXIT

# Try test preset first, fall back to ctest --test-dir
if ctest --list-presets 2>/dev/null | grep -q "${PRESET}" 2>/dev/null; then
    ctest --preset "${PRESET}" --output-on-failure --parallel "${JOBS}" \
        2>&1 | tee "${TEST_RESULT_FILE}" \
        || ERRORS=$((ERRORS + 1))
else
    ctest --test-dir "${BUILD_DIR}" --output-on-failure --parallel "${JOBS}" \
        2>&1 | tee "${TEST_RESULT_FILE}" \
        || ERRORS=$((ERRORS + 1))
fi

# Parse test results
TESTS_TOTAL=0
TESTS_PASSED=0
TESTS_FAILED=0

# ctest summary line: "XX% tests passed, N tests failed out of M"
if grep -qE 'tests passed' "${TEST_RESULT_FILE}"; then
    TESTS_TOTAL="$(grep -oE '[0-9]+ tests? failed out of [0-9]+' "${TEST_RESULT_FILE}" | grep -oE '[0-9]+$' || echo 0)"
    TESTS_FAILED="$(grep -oE '[0-9]+ tests? failed' "${TEST_RESULT_FILE}" | grep -oE '^[0-9]+' || echo 0)"
    TESTS_PASSED=$(( TESTS_TOTAL - TESTS_FAILED ))
fi

timer_stop "Tests"

# ============================================================================
# Summary
# ============================================================================
stage "Pipeline Summary"

printf '\n'
printf '  %-24s %s\n' "Preset:" "${PRESET}"
printf '  %-24s %s\n' "Build directory:" "${BUILD_DIR}"
printf '  %-24s %s\n' "Parallel jobs:" "${JOBS}"
printf '\n'

# Stage timings
printf '  %s%-24s %s%s\n' "${BOLD}" "Stage" "Time" "${RESET}"
printf '  %-24s %s\n' "------------------------" "-----"
for i in "${!STAGE_NAMES[@]}"; do
    printf '  %-24s %ss\n' "${STAGE_NAMES[$i]}" "${STAGE_TIMES[$i]}"
done

# Total time
TOTAL_TIME=0
for t in "${STAGE_TIMES[@]}"; do
    TOTAL_TIME=$((TOTAL_TIME + t))
done
printf '  %-24s %s\n' "------------------------" "-----"
printf '  %-24s %ss\n' "Total" "${TOTAL_TIME}"

printf '\n'

# Test summary
if [[ "${TESTS_TOTAL}" -gt 0 ]]; then
    printf '  %sTest Results:%s\n' "${BOLD}" "${RESET}"
    printf '    Total:  %d\n' "${TESTS_TOTAL}"
    printf '    Passed: %s%d%s\n' "${GREEN}" "${TESTS_PASSED}" "${RESET}"
    if [[ "${TESTS_FAILED}" -gt 0 ]]; then
        printf '    Failed: %s%d%s\n' "${RED}" "${TESTS_FAILED}" "${RESET}"
    else
        printf '    Failed: %d\n' "${TESTS_FAILED}"
    fi
    printf '\n'
fi

# Final verdict
if [[ "${ERRORS}" -gt 0 ]]; then
    fail "Pipeline validation FAILED"
    exit 1
else
    ok "Pipeline validation PASSED"
    exit 0
fi
