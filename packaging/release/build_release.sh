#!/usr/bin/env bash
# build_release.sh — Build binary release artifacts
# Task 26.3.3: Binary release artifact builder for Linux and macOS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VERSION="$(python3 "$PROJECT_ROOT/scripts/version_info.py" --field runtime)"

# Determine platform
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
PLATFORM="${OS}-${ARCH}"

# Output settings
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build/release-package}"
STAGING_DIR="${BUILD_DIR}/staging"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/dist}"
ARTIFACT_NAME="libaccint-${VERSION}-${PLATFORM}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Build binary release artifacts for LibAccInt.

Options:
    -v, --version VERSION   Override version (default: from CMakeLists.txt)
    -o, --output DIR        Output directory (default: ./dist)
    --no-test               Skip running tests
    --no-static             Skip static library
    -h, --help              Show this help

Produces:
    \${OUTPUT_DIR}/libaccint-\${VERSION}-\${PLATFORM}.tar.gz
    Contents: lib/, include/, share/, cmake/, pkgconfig/, LICENSE, README.md
EOF
}

RUN_TESTS=1
BUILD_STATIC=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--version) VERSION="$2"; ARTIFACT_NAME="libaccint-${VERSION}-${PLATFORM}"; shift 2 ;;
        -o|--output) OUTPUT_DIR="$2"; shift 2 ;;
        --no-test) RUN_TESTS=0; shift ;;
        --no-static) BUILD_STATIC=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

echo "============================================"
echo "LibAccInt Release Builder"
echo "============================================"
echo "Version: ${VERSION}"
echo "Platform: ${PLATFORM}"
echo "Output: ${OUTPUT_DIR}/${ARTIFACT_NAME}.tar.gz"
echo "============================================"

cpu_count() {
    if command -v getconf >/dev/null 2>&1; then
        getconf _NPROCESSORS_ONLN 2>/dev/null && return 0
    fi
    if command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu 2>/dev/null && return 0
    fi
    echo 1
}

sha256_file() {
    local file="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file"
    else
        shasum -a 256 "$file"
    fi
}

# Step 1: Configure and build
echo ""
echo ">>> Configuring build..."
cmake -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$STAGING_DIR" \
    -DLIBACCINT_RELOCATABLE=ON \
    -DLIBACCINT_ALLOW_FETCHCONTENT=OFF \
    -DLIBACCINT_USE_CUDA=OFF \
    -DLIBACCINT_USE_OPENMP=ON \
    -DLIBACCINT_BUILD_TESTS=ON \
    -DLIBACCINT_BUILD_EXAMPLES=OFF \
    -DLIBACCINT_BUILD_BENCHMARKS=OFF \
    -S "$PROJECT_ROOT"

echo ""
echo ">>> Building..."
cmake --build "$BUILD_DIR" --parallel

# Step 2: Run tests
if [[ "$RUN_TESTS" == "1" ]]; then
    echo ""
    echo ">>> Running tests..."
    ctest --test-dir "$BUILD_DIR" --output-on-failure --parallel "$(cpu_count)"
fi

# Step 3: Install to staging
echo ""
echo ">>> Installing to staging directory..."
cmake --install "$BUILD_DIR"

# Step 4: Add metadata
echo ""
echo ">>> Adding release metadata..."
cp "$PROJECT_ROOT/LICENSE" "$STAGING_DIR/"
cp "$PROJECT_ROOT/README.md" "$STAGING_DIR/"
cp "$PROJECT_ROOT/CHANGELOG.md" "$STAGING_DIR/" 2>/dev/null || true

# Create version file
cat > "$STAGING_DIR/VERSION" <<EOF
LibAccInt ${VERSION}
Platform: ${PLATFORM}
Build date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Compiler: $(${CXX:-g++} --version | head -1)
EOF

# Step 5: Package
echo ""
echo ">>> Creating release archive..."
mkdir -p "$OUTPUT_DIR"

cd "$STAGING_DIR/.."
mv staging "$ARTIFACT_NAME"
tar -czvf "${OUTPUT_DIR}/${ARTIFACT_NAME}.tar.gz" "$ARTIFACT_NAME"
mv "$ARTIFACT_NAME" staging

# Generate checksum
cd "$OUTPUT_DIR"
sha256_file "${ARTIFACT_NAME}.tar.gz" > "${ARTIFACT_NAME}.tar.gz.sha256"

echo ""
echo "============================================"
echo "Release artifact created:"
echo "  ${OUTPUT_DIR}/${ARTIFACT_NAME}.tar.gz"
echo "  ${OUTPUT_DIR}/${ARTIFACT_NAME}.tar.gz.sha256"
echo ""
ls -lh "${OUTPUT_DIR}/${ARTIFACT_NAME}".*
echo "============================================"
