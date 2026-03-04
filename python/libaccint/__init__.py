# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

"""
LibAccInt - High-performance molecular integral library with GPU acceleration.

This package provides Python bindings for the LibAccInt C++ library, enabling
fast molecular integral computation with NumPy integration.

Basic Usage
-----------
>>> import libaccint
>>> atoms = [
...     libaccint.Atom(8, [0.0, 0.0, 0.0]),
...     libaccint.Atom(1, [0.0, 1.43, -1.11]),
...     libaccint.Atom(1, [0.0, -1.43, -1.11]),
... ]
>>> basis = libaccint.basis_set("sto-3g", atoms)
>>> engine = libaccint.Engine(basis)
>>> S = engine.compute_overlap_matrix()
"""

__version__ = "0.1.0a2"  # Must match CMakeLists.txt VERSION (PEP 440 alpha)
__author__ = "Bryce M. Westheimer"

# Import from C++ extension.
# Bind all exported non-private symbols dynamically so Python package import
# stays compatible with older/partial extension builds.
from . import _core as _core

for _name in dir(_core):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_core, _name)

# Normalize semver prerelease tag from C++ binding to PEP 440
_raw = getattr(_core, "__version__", __version__)
if "-alpha." in _raw:
    __version__ = _raw.replace("-alpha.", "a")
elif "-beta." in _raw:
    __version__ = _raw.replace("-beta.", "b")
elif "-rc." in _raw:
    __version__ = _raw.replace("-rc.", "rc")
else:
    __version__ = _raw


def _missing_backend_bool() -> bool:
    return False


def _missing_device_count() -> int:
    return 0


def _missing_device_info(*_args, **_kwargs):
    return None


def _version_fallback() -> str:
    return getattr(_core, "__version__", __version__)


# Optional symbols that may not exist in older extension binaries.
for _opt_name in (
    "ThreadConfig",
    "ShellSetKey",
    "ShellSetPair",
    "ShellSetQuartet",
    "PrimitivePairData",
    "IntegralBuffer",
    "ShellQuartetMeta",
    "ShellPairMeta",
    "CpuEngine",
    "DeviceInfo",
    "StreamHandle",
    "BackendError",
    "CudaEngine",
    "GpuFockBuilder",
):
    globals().setdefault(_opt_name, None)

# Utility fallbacks.
globals().setdefault("version", _version_fallback)
globals().setdefault("has_cuda_backend", _missing_backend_bool)
globals().setdefault("has_openmp", _missing_backend_bool)
globals().setdefault("get_device_info", _missing_device_info)
globals().setdefault("get_device_count", _missing_device_count)

# Import convenience API
from .convenience import (
    basis_set,
    list_available_basis_sets,
    compute_overlap,
    compute_kinetic,
    compute_nuclear,
    compute_core_hamiltonian,
    build_fock,
    compute_eri_tensor,
    compute_eri_block,
)

# Public API
__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core types
    "Real",
    "Index",
    "Size",
    "AngularMomentum",
    "DerivativeOrder",
    "Point3D",
    "n_cartesian",
    "n_spherical",
    # Basis
    "Shell",
    "ShellSet",
    "BasisSet",
    "Atom",
    "create_builtin_basis",
    # Operators
    "OperatorKind",
    "Operator",
    "PointChargeParams",
    "RangeSeparatedParams",
    "OriginParams",
    "DistributedMultipoleParams",
    "ProjectionOperatorParams",
    "OneElectronOperator",
    "is_one_electron",
    "is_two_electron",
    "is_multi_component",
    "is_anti_hermitian",
    "is_property_integral",
    "component_count",
    # Buffers
    "OneElectronBuffer",
    "TwoElectronBuffer",
    # Backend
    "BackendType",
    "BackendHint",
    "is_backend_available",
    "backend_name",
    # Engine
    "Engine",
    "DispatchConfig",
    # Consumers
    "FockBuilder",
    "FockThreadingStrategy",
    # Screening
    "ScreeningPreset",
    "ScreeningOptions",
    "ScreeningStatistics",
    "SchwarzBounds",
    # Convenience API (from _core)
    "atoms_from_xyz",
    "quick_overlap",
    "quick_kinetic",
    "quick_nuclear",
    "quick_core_hamiltonian",
    "quick_fock_build",
    # Advanced types
    "ThreadConfig",
    "ShellSetKey",
    "ShellSetPair",
    "ShellSetQuartet",
    "PrimitivePairData",
    "IntegralBuffer",
    "ShellQuartetMeta",
    "ShellPairMeta",
    "CpuEngine",
    "DeviceInfo",
    "StreamHandle",
    "BackendError",
    # CUDA-conditional types
    "CudaEngine",
    "GpuFockBuilder",
    # Utility functions
    "version",
    "has_cuda_backend",
    "has_openmp",
    "get_device_info",
    "get_device_count",
    # Convenience API (Python wrappers)
    "basis_set",
    "list_available_basis_sets",
    "compute_overlap",
    "compute_kinetic",
    "compute_nuclear",
    "compute_core_hamiltonian",
    "build_fock",
    "compute_eri_tensor",
    "compute_eri_block",
]
