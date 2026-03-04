#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

"""
Legacy setup.py for pip install -e . workflow.

This file provides backward compatibility for editable installs.
For production builds, pyproject.toml with scikit-build-core is preferred.
"""

import os
import sys
import subprocess
import warnings
from pathlib import Path

warnings.warn(
    "setup.py is deprecated. Use pyproject.toml with pip install instead.",
    DeprecationWarning,
    stacklevel=2,
)

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    """Custom build_ext that runs CMake for the C++ extension."""

    def run(self):
        # Check for CMake
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build this extension")

        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        cwd = Path().absolute()
        build_temp = Path(self.build_temp).absolute()
        build_temp.mkdir(parents=True, exist_ok=True)

        # Extension output directory
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        extdir.mkdir(parents=True, exist_ok=True)

        # CMake configuration
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DLIBACCINT_BUILD_PYTHON=ON",
            f"-DLIBACCINT_BUILD_TESTS=OFF",
            f"-DLIBACCINT_RELOCATABLE=ON",
            f"-DLIBACCINT_ALLOW_FETCHCONTENT=OFF",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]

        build_args = ["--config", "Release", "--", "-j4"]

        os.chdir(str(build_temp))

        # Run CMake configure from the project root
        project_root = cwd.parent  # Assuming python/ is a subdirectory
        self.spawn(["cmake", str(project_root)] + cmake_args)

        # Build
        self.spawn(["cmake", "--build", "."] + build_args)

        os.chdir(str(cwd))


class CMakeExtension:
    """Dummy extension class for CMake builds."""

    def __init__(self, name, sourcedir=""):
        self.name = name
        self.sourcedir = os.path.abspath(sourcedir)


setup(
    name="libaccint",
    version="0.1.0a2",
    author="Bryce M. Westheimer",
    description="High-performance molecular integral library with GPU acceleration",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
    ext_modules=[CMakeExtension("libaccint._core")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
    zip_safe=False,
)
