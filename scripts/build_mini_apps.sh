#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.
#
# build_mini_apps.sh — Build the main library, Python bindings, and C++ mini-app
#
# Builds everything needed to run both the C++ and Python HF mini-apps.
# Uses the all-release preset (AUTO detection for CUDA/HIP).
#
# Usage:
#   ./scripts/build_mini_apps.sh [OPTIONS]
#
# Options:
#   --clean           Remove build directories before configuring
#   --jobs <N>        Parallel build jobs (default: nproc)
#   --preset <NAME>   CMake preset to use (default: all-release)
#   --safe-wsl        Force low-memory CUDA-safe defaults (preset + jobs)
#   --skip-codegen    Skip code generation step
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
PRESET="all-release"
SKIP_CODEGEN=0
CLEAN_BUILD=0
JOBS=""
SAFE_WSL=0
STAGE_TIMES=()
STAGE_NAMES=()
ERRORS=0

# ============================================================================
# Argument parsing
# ============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-codegen)
            SKIP_CODEGEN=1; shift ;;
        --clean)
            CLEAN_BUILD=1; shift ;;
        --jobs)
            JOBS="$2"; shift 2 ;;
        --preset)
            PRESET="$2"; shift 2 ;;
        --safe-wsl)
            SAFE_WSL=1; shift ;;
        --help|-h)
            head -20 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *)
            die "Unknown option: $1 (try --help)" ;;
    esac
done

is_wsl() {
    [[ -n "${WSL_DISTRO_NAME:-}" ]] && return 0
    grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null
}

if [[ "${SAFE_WSL}" -eq 1 ]] && [[ "${PRESET}" == "all-release" ]]; then
    PRESET="cuda-release-safe"
fi

# Default parallel jobs
if [[ -z "${JOBS}" ]]; then
    if [[ "${SAFE_WSL}" -eq 1 ]]; then
        JOBS=1
    elif is_wsl && [[ "${PRESET}" == cuda* || "${PRESET}" == all-* ]]; then
        JOBS=1
    elif command -v nproc &>/dev/null; then
        JOBS="$(nproc)"
    elif command -v sysctl &>/dev/null; then
        JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
    else
        JOBS=4
    fi
fi

BUILD_DIR="${PROJECT_ROOT}/build/${PRESET}"
MINIAPP_BUILD_DIR="${PROJECT_ROOT}/examples/mini-apps/cpp-hf/build"

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
# Stage 3: Configure main library
# ============================================================================
stage "Stage 3: Configure main library (preset: ${PRESET})"
timer_start

if [[ "${CLEAN_BUILD}" -eq 1 ]] && [[ -d "${BUILD_DIR}" ]]; then
    info "Cleaning build directory: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

info "Using CMake preset: ${PRESET} with -DLIBACCINT_BUILD_PYTHON=ON"
cmake --preset "${PRESET}" -S "${PROJECT_ROOT}" \
    -DLIBACCINT_BUILD_PYTHON=ON \
    || die "CMake configure failed"

ok "CMake configuration succeeded"
timer_stop "Configure main library"

# ============================================================================
# Stage 4: Build main library
# ============================================================================
stage "Stage 4: Build main library"
timer_start

cmake --build --preset "${PRESET}" --parallel "${JOBS}" \
    || die "Build failed"

ok "Main library build succeeded"
timer_stop "Build main library"

# ============================================================================
# Stage 5: Install Python bindings
# ============================================================================
stage "Stage 5: Install Python bindings"
timer_start

pip install -e "${PROJECT_ROOT}/python/" \
    || die "Python bindings installation failed"

ok "Python bindings installed"
timer_stop "Install Python bindings"

# ============================================================================
# Stage 6: Build C++ mini-app (standalone)
# ============================================================================
stage "Stage 6: Build C++ mini-app"
timer_start

if [[ "${CLEAN_BUILD}" -eq 1 ]] && [[ -d "${MINIAPP_BUILD_DIR}" ]]; then
    info "Cleaning mini-app build directory: ${MINIAPP_BUILD_DIR}"
    rm -rf "${MINIAPP_BUILD_DIR}"
fi

MINIAPP_SRC="${PROJECT_ROOT}/examples/mini-apps/cpp-hf"

# Detect build generator
if command -v ninja &>/dev/null; then
    GENERATOR="Ninja"
else
    GENERATOR="Unix Makefiles"
fi

info "Configuring C++ mini-app (standalone build)..."
cmake -S "${MINIAPP_SRC}" -B "${MINIAPP_BUILD_DIR}" \
    -G "${GENERATOR}" \
    -DCMAKE_BUILD_TYPE=Release \
    || die "C++ mini-app configure failed"

info "Building C++ mini-app..."
cmake --build "${MINIAPP_BUILD_DIR}" --parallel "${JOBS}" \
    || die "C++ mini-app build failed"

ok "C++ mini-app build succeeded"
timer_stop "Build C++ mini-app"

# ============================================================================
# Summary
# ============================================================================
stage "Summary"

printf '\n'
printf '  %-24s %s\n' "Preset:" "${PRESET}"
printf '  %-24s %s\n' "Main build directory:" "${BUILD_DIR}"
printf '  %-24s %s\n' "Mini-app build dir:" "${MINIAPP_BUILD_DIR}"
printf '  %-24s %s\n' "Parallel jobs:" "${JOBS}"
printf '  %-24s %s\n' "Safe WSL mode:" "$([[ "${SAFE_WSL}" -eq 1 ]] && echo yes || echo no)"

# Detect GPU backends
GPU_INFO="CPU only"
if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    CUDA_ENABLED="$(grep -E '^LIBACCINT_USE_CUDA:' "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null | cut -d= -f2 || echo "OFF")"
    if [[ "${CUDA_ENABLED}" == "ON" ]]; then
        GPU_INFO="CUDA"
    fi
fi
printf '  %-24s %s\n' "Detected backends:" "${GPU_INFO}"

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

# Usage examples
printf '  %sTo run the mini-apps:%s\n' "${BOLD}" "${RESET}"
printf '    C++ mini-app:    ./examples/mini-apps/cpp-hf/build/hf_miniapp --molecule h2o --basis sto-3g\n'
printf '    Python mini-app: python examples/mini-apps/python-hf/hf.py --molecule h2o --basis sto-3g\n'

printf '\n'
ok "All builds completed successfully"
