#!/usr/bin/env python3
"""Release-gate validator for version + packaging metadata consistency.

Phase 5 scope:
- BUILD-H1: Version consistency across code/docs/packaging/changelog
- PACKAGING-M1: Detect unreplaced hash placeholders and malformed SHA256s
"""

import re
import sys
from pathlib import Path

from version_info import load_version_info

VERSION_PATTERNS = {
    "python/pyproject.toml": (
        r'version\s*=\s*"(\d+\.\d+\.\d+[^"]*)"',
        "python",
    ),
    "python/setup.py": (
        r'version\s*=\s*"(\d+\.\d+\.\d+[^"]*)"',
        "python",
    ),
    "python/libaccint/__init__.py": (
        r'__version__\s*=\s*"(\d+\.\d+\.\d+[^"]*)"',
        "python",
    ),
    "docs/conf.py": (
        r"version\s*=\s*'([0-9]+\.[0-9]+\.[0-9]+[^']*)'",
        "base",
    ),
    "docs/conf.py#release": (
        r"release\s*=\s*'([0-9]+\.[0-9]+\.[0-9]+[^']*)'",
        "runtime",
    ),
    "docs/api/API_REFERENCE.md": (
        r'##\s+Version:\s+([0-9]+\.[0-9]+\.[0-9]+[^\s]*)',
        "runtime",
    ),
    "docs/user_guide/FAQ.md": (
        r'##\s+Version:\s+([0-9]+\.[0-9]+\.[0-9]+[^\s]*)',
        "runtime",
    ),
    "packaging/conda/meta.yaml": (
        r'\{\%\s*set\s+version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+[^"]*)"\s*\%\}',
        "python",
    ),
    "packaging/pypi/pyproject.toml": (
        r'version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+[^"]*)"',
        "python",
    ),
    "packaging/spack/packages/libaccint/package.py#version": (
        r'version\("([0-9]+\.[0-9]+\.[0-9]+[^"]*)"\s*,\s*sha256=',
        "runtime",
    ),
}

CHANGELOG_RELEASE_PATTERN = (
    r'^##\s+\[(\d+\.\d+\.\d+(?:-[A-Za-z0-9.]+)?)\]\s*-\s*\d{4}-\d{2}-\d{2}\s*$'
)
SHA256_PATTERN = re.compile(r'^[a-f0-9]{64}$')
PLACEHOLDER_TOKENS = (
    "UPDATE_WITH_RELEASE_HASH",
    "RELEASE_HASH",
    "TODO",
    "PLACEHOLDER",
)


def extract_value(file_path: Path, pattern: str) -> str | None:
    """Extract version from a file using the given pattern."""
    if not file_path.exists():
        return None

    content = file_path.read_text()
    match = re.search(pattern, content)
    if match:
        return match.group(1)
    return None


def parse_changelog_latest_release(content: str) -> str | None:
    """Extract latest released version heading from CHANGELOG.md."""
    for line in content.splitlines():
        match = re.match(CHANGELOG_RELEASE_PATTERN, line.strip())
        if match:
            return match.group(1)
    return None


def extract_packaging_hashes(root_dir: Path) -> dict[str, str | None]:
    """Extract hash fields used by packaging recipes."""
    hashes: dict[str, str | None] = {
        "packaging/conda/meta.yaml": None,
        "packaging/spack/packages/libaccint/package.py": None,
    }

    conda_path = root_dir / "packaging/conda/meta.yaml"
    if conda_path.exists():
        conda_content = conda_path.read_text()
        match = re.search(r'\bsha256\s*:\s*([^\s#]+)', conda_content)
        if match:
            hashes["packaging/conda/meta.yaml"] = match.group(1).strip('"\'')

    spack_path = root_dir / "packaging/spack/packages/libaccint/package.py"
    if spack_path.exists():
        spack_content = spack_path.read_text()
        match = re.search(
            r'version\("[^"]+"\s*,\s*sha256\s*=\s*"([^"]+)"\)',
            spack_content,
        )
        if match:
            hashes["packaging/spack/packages/libaccint/package.py"] = match.group(1)

    return hashes


def is_placeholder_hash(value: str) -> bool:
    upper = value.upper()
    return any(token in upper for token in PLACEHOLDER_TOKENS)


def main():
    check_hashes = "--check-hashes" in sys.argv[1:]
    root_dir = Path(__file__).parent.parent.resolve()
    print(f"Checking release-gate consistency in: {root_dir}")

    try:
        expected_versions = load_version_info(root_dir)
    except ValueError as exc:
        print(f"\n✗ {exc}")
        return 1

    print("Canonical versions:")
    for key, value in expected_versions.items():
        print(f"  {key}: {value}")

    versions: dict[str, str] = {}
    errors: list[str] = []

    for file_rel, (pattern, field) in VERSION_PATTERNS.items():
        file_key = file_rel.split("#", 1)[0]
        file_path = root_dir / file_rel
        if "#" in file_rel:
            file_path = root_dir / file_key

        version = extract_value(file_path, pattern)
        if version is not None:
            versions[file_rel] = version
            print(f"  {file_rel}: {version}")
            expected = expected_versions[field]
            if version != expected:
                errors.append(
                    f"Version mismatch in {file_rel}: "
                    f"{version} != {expected}"
                )
        else:
            print(f"  {file_rel}: NOT FOUND")
            errors.append(
                "Missing or unreadable version marker in "
                f"{file_rel}"
            )

    changelog_path = root_dir / "CHANGELOG.md"
    if changelog_path.exists():
        changelog_content = changelog_path.read_text()
        latest_release = parse_changelog_latest_release(changelog_content)
        print(
            "  CHANGELOG.md latest released entry: "
            f"{latest_release or 'NOT FOUND'}"
        )
        if latest_release is None:
            errors.append(
                "CHANGELOG.md has no released version heading"
            )
        elif latest_release != expected_versions["runtime"]:
            errors.append(
                "CHANGELOG latest release mismatch: "
                f"{latest_release} != {expected_versions['runtime']}"
            )
    else:
        errors.append("Missing CHANGELOG.md")

    if check_hashes:
        hashes = extract_packaging_hashes(root_dir)
        for hash_file, hash_value in hashes.items():
            if hash_value is None:
                print(f"  {hash_file} hash: NOT FOUND")
                errors.append(f"Missing sha256 in {hash_file}")
                continue

            print(f"  {hash_file} hash: {hash_value}")
            if is_placeholder_hash(hash_value):
                errors.append(
                    "Unreplaced hash placeholder in "
                    f"{hash_file}: {hash_value}"
                )
                continue
            if not SHA256_PATTERN.match(hash_value):
                errors.append(
                    f"Invalid sha256 format in {hash_file}: "
                    f"{hash_value}"
                )

    if errors:
        print("\n✗ Release-gate validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(
        "\n✓ Release-gate version/hash checks passed for "
        f"{expected_versions['runtime']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
