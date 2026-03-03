#!/usr/bin/env python3
"""Single source of truth for LibAccInt release version mapping."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_PATTERN = re.compile(
    r"project\(LibAccInt\s+VERSION\s+(\d+\.\d+\.\d+)", re.MULTILINE
)
PRERELEASE_PATTERN = re.compile(
    r'set\(LIBACCINT_PRERELEASE\s+"([^"]*)"\)', re.MULTILINE
)


def to_python_version(base: str, prerelease: str) -> str:
    if not prerelease:
        return base

    match = re.fullmatch(r"(alpha|beta|rc)\.(\d+)", prerelease)
    if not match:
        raise ValueError(f"Unsupported prerelease format: {prerelease}")

    label, number = match.groups()
    suffix = {"alpha": "a", "beta": "b", "rc": "rc"}[label]
    return f"{base}{suffix}{number}"


def load_version_info(root: Path) -> dict[str, str]:
    cmake_path = root / "CMakeLists.txt"
    content = cmake_path.read_text(encoding="utf-8")

    project_match = PROJECT_PATTERN.search(content)
    if not project_match:
        raise ValueError("Could not parse project version from CMakeLists.txt")

    prerelease_match = PRERELEASE_PATTERN.search(content)
    base = project_match.group(1)
    prerelease = prerelease_match.group(1) if prerelease_match else ""
    runtime = f"{base}-{prerelease}" if prerelease else base

    return {
        "base": base,
        "prerelease": prerelease,
        "runtime": runtime,
        "python": to_python_version(base, prerelease),
        "tag": f"v{runtime}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--field",
        choices=["base", "prerelease", "runtime", "python", "tag"],
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    info = load_version_info(root)

    if args.json:
        print(json.dumps(info))
        return 0

    if args.field:
        print(info[args.field])
        return 0

    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
