#!/usr/bin/env bash
# build_wheels.sh — Build manylinux wheels using cibuildwheel
# Task 26.2.1: PyPI wheel build script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default configuration
PYTHON_VERSIONS="${PYTHON_VERSIONS:-cp39 cp310 cp311 cp312}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/wheelhouse}"
PLATFORM="${PLATFORM:-auto}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Build Python wheels for LibAccInt using cibuildwheel.

Options:
    -p, --platform PLATFORM   Build platform (linux, macos, windows, auto)
    -o, --output DIR          Output directory (default: ./wheelhouse)
    -v, --versions VERSIONS   Python versions (default: "cp39 cp310 cp311 cp312")
    --test-only               Only run tests on existing wheels
    -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--platform) PLATFORM="$2"; shift 2 ;;
        -o|--output) OUTPUT_DIR="$2"; shift 2 ;;
        -v|--versions) PYTHON_VERSIONS="$2"; shift 2 ;;
        --test-only) TEST_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Check dependencies
if ! command -v cibuildwheel &>/dev/null; then
    echo "Installing cibuildwheel..."
    pip install cibuildwheel
fi

# Build version string for CIBW_BUILD
BUILD_SPEC=""
for ver in $PYTHON_VERSIONS; do
    BUILD_SPEC="${BUILD_SPEC:+${BUILD_SPEC} }${ver}-*"
done

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "LibAccInt Wheel Builder"
echo "============================================"
echo "Project root: $PROJECT_ROOT"
echo "Platform: $PLATFORM"
echo "Python versions: $PYTHON_VERSIONS"
echo "Output: $OUTPUT_DIR"
echo "============================================"

# Set environment variables
export CIBW_BUILD="$BUILD_SPEC"
export CIBW_SKIP="*-win32 *-manylinux_i686 *-musllinux*"
export CIBW_BUILD_FRONTEND="build"
export CIBW_ENVIRONMENT="CMAKE_BUILD_PARALLEL_LEVEL=4"

# Platform-specific settings
export CIBW_BEFORE_ALL_LINUX="yum install -y ninja-build || apt-get install -y ninja-build"
export CIBW_BEFORE_ALL_MACOS="brew install ninja libomp"
export CIBW_BEFORE_ALL_WINDOWS="choco install ninja"

# Test commands
export CIBW_TEST_COMMAND="python -c 'import libaccint; print(libaccint.__version__)'"

if [[ "${TEST_ONLY:-}" == "1" ]]; then
    echo "Test-only mode — testing existing wheels in $OUTPUT_DIR"
    for whl in "$OUTPUT_DIR"/*.whl; do
        echo "Testing: $(basename "$whl")"
        pip install "$whl" && python -c "import libaccint; print(libaccint.__version__)"
    done
    exit 0
fi

# Run cibuildwheel
cd "$PROJECT_ROOT"

if [[ "$PLATFORM" == "auto" ]]; then
    cibuildwheel --output-dir "$OUTPUT_DIR" python/
else
    cibuildwheel --platform "$PLATFORM" --output-dir "$OUTPUT_DIR" python/
fi

echo "============================================"
echo "Build complete! Wheels in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*.whl 2>/dev/null || echo "No wheels found"
echo "============================================"
