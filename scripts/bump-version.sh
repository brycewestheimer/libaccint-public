#!/usr/bin/env bash
# bump-version.sh — Semantic version bumping wrapper
# Task 26.4.1: Semantic versioning automation
#
# Convenience wrapper around scripts/bump_version.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [major|minor|patch] [OPTIONS]
       $(basename "$0") --version X.Y.Z [OPTIONS]

Bump the LibAccInt version across all project files.

Commands:
    major               Bump major version (1.0.0 -> 2.0.0)
    minor               Bump minor version (1.0.0 -> 1.1.0)
    patch               Bump patch version (1.0.0 -> 1.0.1)

Options:
    -v, --version VER   Set explicit version (e.g., 1.2.3 or 1.2.3-beta.1)
    -n, --dry-run       Show what would be changed without making changes
    --tag               Create git tag after version bump
    --commit            Create git commit after version bump
    -h, --help          Show this help

Examples:
    $(basename "$0") patch                    # 1.0.0 -> 1.0.1
    $(basename "$0") minor --tag --commit     # 1.0.0 -> 1.1.0, commit and tag
    $(basename "$0") --version 2.0.0-rc.1     # Set explicit version
EOF
}

BUMP_TYPE=""
EXPLICIT_VERSION=""
DRY_RUN=""
CREATE_TAG=0
CREATE_COMMIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        major|minor|patch) BUMP_TYPE="$1"; shift ;;
        -v|--version) EXPLICIT_VERSION="$2"; shift 2 ;;
        -n|--dry-run) DRY_RUN="--dry-run"; shift ;;
        --tag) CREATE_TAG=1; shift ;;
        --commit) CREATE_COMMIT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "$BUMP_TYPE" && -z "$EXPLICIT_VERSION" ]]; then
    echo "ERROR: Specify a bump type (major, minor, patch) or explicit version"
    usage
    exit 1
fi

# Build arguments for bump_version.py
ARGS=()
if [[ -n "$EXPLICIT_VERSION" ]]; then
    ARGS+=(--version "$EXPLICIT_VERSION")
elif [[ -n "$BUMP_TYPE" ]]; then
    ARGS+=(--bump "$BUMP_TYPE")
fi

if [[ -n "$DRY_RUN" ]]; then
    ARGS+=("$DRY_RUN")
fi

ARGS+=(--root "$PROJECT_ROOT")

# Run the Python version bumper
python3 "$SCRIPT_DIR/bump_version.py" "${ARGS[@]}"

# Post-bump actions (only if not dry-run)
if [[ -z "$DRY_RUN" ]]; then
    # Get the new version
    NEW_VERSION=$(grep -oP 'project\(LibAccInt\s+VERSION\s+\K[0-9]+\.[0-9]+\.[0-9]+' "$PROJECT_ROOT/CMakeLists.txt")

    if [[ "$CREATE_COMMIT" == "1" ]]; then
        echo ""
        echo "Creating git commit..."
        cd "$PROJECT_ROOT"
        git add -A
        git commit -m "chore(release): bump version to ${NEW_VERSION}"
        echo "Committed version bump to ${NEW_VERSION}"
    fi

    if [[ "$CREATE_TAG" == "1" ]]; then
        echo ""
        echo "Creating git tag..."
        cd "$PROJECT_ROOT"
        git tag -a "v${NEW_VERSION}" -m "Release ${NEW_VERSION}"
        echo "Tagged v${NEW_VERSION}"
    fi
fi
