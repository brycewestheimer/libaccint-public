#!/usr/bin/env python3
"""
Version bumping script for LibAccInt.

Updates version numbers across all project files to maintain consistency.
Supports semantic versioning with pre-release identifiers.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional


VERSION_FILES = {
    "CMakeLists.txt": r'project\(LibAccInt\s+VERSION\s+(\d+\.\d+\.\d+)',
    "python/pyproject.toml": r'version\s*=\s*"(\d+\.\d+\.\d+[^"]*)"',
    "python/setup.py": r'version\s*=\s*"(\d+\.\d+\.\d+[^"]*)"',
    "python/libaccint/__init__.py": r'__version__\s*=\s*"(\d+\.\d+\.\d+[^"]*)"',
    "python/src/bindings.cpp": r'm\.attr\("__version__"\)\s*=\s*"(\d+\.\d+\.\d+[^"]*)"',
    "docs/conf.py": r"(?:version|release)\s*=\s*'(\d+\.\d+\.\d+[^']*)'",
}


def parse_version(version_str: str) -> tuple[int, int, int, Optional[str]]:
    """Parse a version string into components."""
    # Match: 1.2.3 or 1.2.3-beta.1
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$', version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")
    
    major, minor, patch, prerelease = match.groups()
    return int(major), int(minor), int(patch), prerelease


def format_version(major: int, minor: int, patch: int, prerelease: Optional[str] = None) -> str:
    """Format version components into a string."""
    version = f"{major}.{minor}.{patch}"
    if prerelease:
        version += f"-{prerelease}"
    return version


def get_current_version(root_dir: Path) -> str:
    """Extract current version from CMakeLists.txt."""
    cmake_file = root_dir / "CMakeLists.txt"
    content = cmake_file.read_text()
    
    match = re.search(VERSION_FILES["CMakeLists.txt"], content)
    if match:
        return match.group(1)
    
    raise RuntimeError("Could not find version in CMakeLists.txt")


def update_file(file_path: Path, pattern: str, new_version: str, dry_run: bool = False) -> bool:
    """Update version in a single file."""
    if not file_path.exists():
        print(f"  Skipping {file_path} (not found)")
        return False
    
    content = file_path.read_text()
    
    def replace_version(match):
        full_match = match.group(0)
        old_version = match.group(1)
        return full_match.replace(old_version, new_version)
    
    new_content, count = re.subn(pattern, replace_version, content)
    
    if count == 0:
        print(f"  Warning: No version pattern found in {file_path}")
        return False
    
    if dry_run:
        print(f"  Would update {file_path}")
    else:
        file_path.write_text(new_content)
        print(f"  Updated {file_path}")
    
    return True


def bump_version(current: str, bump_type: str) -> str:
    """Bump version according to type."""
    major, minor, patch, prerelease = parse_version(current)
    
    if bump_type == "major":
        return format_version(major + 1, 0, 0)
    elif bump_type == "minor":
        return format_version(major, minor + 1, 0)
    elif bump_type == "patch":
        return format_version(major, minor, patch + 1)
    else:
        raise ValueError(f"Unknown bump type: {bump_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Bump version numbers across LibAccInt project files"
    )
    parser.add_argument(
        "--version", "-v",
        help="Explicit version to set (e.g., 1.2.3 or 1.2.3-beta.1)"
    )
    parser.add_argument(
        "--bump", "-b",
        choices=["major", "minor", "patch"],
        help="Bump type (alternative to explicit version)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be changed without making changes"
    )
    parser.add_argument(
        "--root", "-r",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    if not args.version and not args.bump:
        parser.error("Either --version or --bump must be specified")
    
    root_dir = args.root.resolve()
    print(f"Project root: {root_dir}")
    
    current_version = get_current_version(root_dir)
    print(f"Current version: {current_version}")
    
    if args.version:
        new_version = args.version.lstrip("v")
    else:
        new_version = bump_version(current_version, args.bump)
    
    print(f"New version: {new_version}")
    
    if args.dry_run:
        print("\nDry run - no files will be modified:")
    else:
        print("\nUpdating files:")
    
    success = True
    for file_rel, pattern in VERSION_FILES.items():
        file_path = root_dir / file_rel
        if not update_file(file_path, pattern, new_version, args.dry_run):
            success = False
    
    if success:
        print(f"\n✓ Version updated to {new_version}")
        return 0
    else:
        print("\n⚠ Some files could not be updated")
        return 1


if __name__ == "__main__":
    sys.exit(main())
