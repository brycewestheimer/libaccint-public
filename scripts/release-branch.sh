#!/usr/bin/env bash
# release-branch.sh — Release branch workflow automation
# Task 26.4.2: Release branch management
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [OPTIONS]

Release branch workflow management for LibAccInt.

Commands:
    create <version>     Create a release branch (e.g., release/1.2)
    prepare <version>    Prepare release: bump version, update changelog
    finalize <version>   Tag and merge release branch back to main
    hotfix <version>     Create hotfix branch from release tag
    status               Show current release branches and tags

Options:
    --dry-run, -n        Show what would be done without making changes
    -h, --help           Show this help

Examples:
    $(basename "$0") create 1.2.0
    $(basename "$0") prepare 1.2.0
    $(basename "$0") finalize 1.2.0
    $(basename "$0") hotfix 1.2.1
EOF
}

DRY_RUN=0

# Parse global options
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) ARGS+=("$arg") ;;
    esac
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
    usage
    exit 1
fi

COMMAND="${ARGS[0]}"

run_cmd() {
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY RUN] $*"
    else
        echo ">>> $*"
        "$@"
    fi
}

check_clean_tree() {
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "ERROR: Working tree is not clean. Commit or stash changes first."
        exit 1
    fi
}

cmd_create() {
    local VERSION="${ARGS[1]:-}"
    if [[ -z "$VERSION" ]]; then
        echo "ERROR: Version required. Usage: $(basename "$0") create <version>"
        exit 1
    fi

    # Extract major.minor for branch name
    local BRANCH_VERSION
    BRANCH_VERSION=$(echo "$VERSION" | grep -oP '^\d+\.\d+')
    local BRANCH_NAME="release/${BRANCH_VERSION}"

    echo "Creating release branch: ${BRANCH_NAME}"

    check_clean_tree

    run_cmd git checkout main
    run_cmd git pull origin main
    run_cmd git checkout -b "$BRANCH_NAME"
    run_cmd git push -u origin "$BRANCH_NAME"

    echo ""
    echo "Release branch '${BRANCH_NAME}' created."
    echo ""
    echo "Next steps:"
    echo "  1. Run: $(basename "$0") prepare ${VERSION}"
    echo "  2. Review and test the release branch"
    echo "  3. Run: $(basename "$0") finalize ${VERSION}"
}

cmd_prepare() {
    local VERSION="${ARGS[1]:-}"
    if [[ -z "$VERSION" ]]; then
        echo "ERROR: Version required."
        exit 1
    fi

    local BRANCH_VERSION
    BRANCH_VERSION=$(echo "$VERSION" | grep -oP '^\d+\.\d+')
    local BRANCH_NAME="release/${BRANCH_VERSION}"

    echo "Preparing release ${VERSION} on branch ${BRANCH_NAME}"

    check_clean_tree

    # Switch to release branch
    run_cmd git checkout "$BRANCH_NAME"

    # Bump version
    echo "Bumping version to ${VERSION}..."
    run_cmd python3 "$PROJECT_ROOT/scripts/bump_version.py" --version "$VERSION"

    # Validate version consistency
    run_cmd python3 "$PROJECT_ROOT/scripts/validate_version.py"

    # Commit version bump
    run_cmd git add -A
    run_cmd git commit -m "chore(release): bump version to ${VERSION}"

    echo ""
    echo "Release ${VERSION} prepared on branch ${BRANCH_NAME}."
    echo ""
    echo "Next steps:"
    echo "  1. Build and test: cmake --build --preset cpu-release && ctest --test-dir build/cpu-release"
    echo "  2. Review CHANGELOG.md"
    echo "  3. Push: git push origin ${BRANCH_NAME}"
    echo "  4. Run: $(basename "$0") finalize ${VERSION}"
}

cmd_finalize() {
    local VERSION="${ARGS[1]:-}"
    if [[ -z "$VERSION" ]]; then
        echo "ERROR: Version required."
        exit 1
    fi

    local BRANCH_VERSION
    BRANCH_VERSION=$(echo "$VERSION" | grep -oP '^\d+\.\d+')
    local BRANCH_NAME="release/${BRANCH_VERSION}"
    local TAG_NAME="v${VERSION}"

    echo "Finalizing release ${VERSION}"

    check_clean_tree

    # Ensure we're on the release branch
    run_cmd git checkout "$BRANCH_NAME"

    # Create annotated tag
    run_cmd git tag -a "$TAG_NAME" -m "Release ${VERSION}"

    # Merge back to main
    run_cmd git checkout main
    run_cmd git merge --no-ff "$BRANCH_NAME" -m "chore(release): merge ${BRANCH_NAME} for ${TAG_NAME}"

    # Push everything
    run_cmd git push origin main
    run_cmd git push origin "$TAG_NAME"
    run_cmd git push origin "$BRANCH_NAME"

    echo ""
    echo "Release ${VERSION} finalized!"
    echo "  Tag: ${TAG_NAME}"
    echo "  Branch: ${BRANCH_NAME} merged to main"
    echo ""
    echo "The release workflow will be triggered automatically by the tag push."
}

cmd_hotfix() {
    local VERSION="${ARGS[1]:-}"
    if [[ -z "$VERSION" ]]; then
        echo "ERROR: Version required."
        exit 1
    fi

    # Determine the base tag (previous patch version)
    local MAJOR MINOR PATCH
    MAJOR=$(echo "$VERSION" | cut -d. -f1)
    MINOR=$(echo "$VERSION" | cut -d. -f2)
    PATCH=$(echo "$VERSION" | cut -d. -f3)

    if [[ "$PATCH" -le 0 ]]; then
        echo "ERROR: Hotfix version must have patch > 0 (e.g., 1.2.1)"
        exit 1
    fi

    local BASE_TAG="v${MAJOR}.${MINOR}.$((PATCH - 1))"
    local BRANCH_NAME="hotfix/${VERSION}"

    echo "Creating hotfix branch: ${BRANCH_NAME} from ${BASE_TAG}"

    check_clean_tree

    # Create hotfix branch from the previous release tag
    run_cmd git checkout -b "$BRANCH_NAME" "$BASE_TAG"
    run_cmd git push -u origin "$BRANCH_NAME"

    echo ""
    echo "Hotfix branch '${BRANCH_NAME}' created from ${BASE_TAG}."
    echo ""
    echo "Next steps:"
    echo "  1. Apply fixes on this branch"
    echo "  2. Run: $(basename "$0") prepare ${VERSION}"
    echo "  3. Run: $(basename "$0") finalize ${VERSION}"
}

cmd_status() {
    echo "=== Release Status ==="
    echo ""
    echo "Release branches:"
    git branch -a --list '*release/*' 2>/dev/null | head -20 || echo "  (none)"
    echo ""
    echo "Hotfix branches:"
    git branch -a --list '*hotfix/*' 2>/dev/null | head -20 || echo "  (none)"
    echo ""
    echo "Recent tags:"
    git tag --sort=-v:refname | head -10 || echo "  (none)"
    echo ""
    echo "Current branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "Current version: $(grep -oP 'project\(LibAccInt\s+VERSION\s+\K[0-9]+\.[0-9]+\.[0-9]+' "$PROJECT_ROOT/CMakeLists.txt")"
}

case "$COMMAND" in
    create)   cmd_create ;;
    prepare)  cmd_prepare ;;
    finalize) cmd_finalize ;;
    hotfix)   cmd_hotfix ;;
    status)   cmd_status ;;
    *)
        echo "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac
