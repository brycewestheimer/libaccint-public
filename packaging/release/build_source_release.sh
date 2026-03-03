#!/usr/bin/env bash
# build_source_release.sh — Build deterministic source release artifact
# Matches the source archive naming/layout used by .github/workflows/release.yml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="$(python3 "$PROJECT_ROOT/scripts/version_info.py" --field runtime)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/dist}"
ARCHIVE_NAME="libaccint-${VERSION}.tar.gz"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Build the deterministic source archive used for release packaging.

Options:
    -o, --output DIR   Output directory (default: ./dist)
    -h, --help         Show this help

Produces:
    \${OUTPUT_DIR}/libaccint-\${VERSION}.tar.gz
    \${OUTPUT_DIR}/libaccint-\${VERSION}.tar.gz.sha256
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]]; then
    echo "Refusing to build source release from a dirty worktree." >&2
    echo "Commit or stash changes first so the archive matches HEAD exactly." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

git -C "$PROJECT_ROOT" archive \
    --format=tar.gz \
    --prefix="libaccint-${VERSION}/" \
    -o "${OUTPUT_DIR}/${ARCHIVE_NAME}" \
    HEAD

if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${OUTPUT_DIR}/${ARCHIVE_NAME}" > "${OUTPUT_DIR}/${ARCHIVE_NAME}.sha256"
else
    shasum -a 256 "${OUTPUT_DIR}/${ARCHIVE_NAME}" > "${OUTPUT_DIR}/${ARCHIVE_NAME}.sha256"
fi

echo "Created ${OUTPUT_DIR}/${ARCHIVE_NAME}"
echo "Created ${OUTPUT_DIR}/${ARCHIVE_NAME}.sha256"
