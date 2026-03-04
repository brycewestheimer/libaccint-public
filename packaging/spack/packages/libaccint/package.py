# Copyright Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause
#
# Spack package definition for LibAccInt

from spack.package import *


class Libaccint(CMakePackage, CudaPackage):
    """High-performance molecular integral library with GPU acceleration.

    LibAccInt is a C++20 library for computing molecular integrals used in
    quantum chemistry calculations. It supports overlap, kinetic, nuclear,
    and electron repulsion integrals with GPU acceleration via CUDA,
    density fitting, MPI, and thread-parallel CPU execution.
    """

    homepage = "https://github.com/brycewestheimer/libaccint-public"
    url = (
        "https://github.com/brycewestheimer/libaccint-public/releases/download/"
        "v0.1.0-alpha.2/libaccint-0.1.0-alpha.2.tar.gz"
    )
    git = "https://github.com/brycewestheimer/libaccint-public.git"

    maintainers("brycewestheimer")

    license("BSD-3-Clause")
    generator = "Ninja"

    version("main", branch="main")
    version("0.1.0-alpha.2", sha256="b415493c20fa708648091b49ce77c6b151374b5baf350ba81facf66ce52f51b8")

    variant("cuda", default=False, description="Enable CUDA GPU backend")
    variant("mpi", default=False, description="Enable MPI backend")
    variant("openmp", default=True, description="Enable OpenMP parallelization")
    variant("python", default=False, description="Build Python bindings")
    variant("shared", default=True, description="Build shared libraries")
    variant("examples", default=False, description="Build example programs")
    variant("tests", default=False, description="Build test suite")

    depends_on("cmake@3.25:", type="build")
    depends_on("ninja", type="build")
    depends_on("pkgconfig", type="build")
    depends_on("nlohmann-json", type=("build", "link"))
    depends_on("eigen", when="+tests", type=("build", "link"))
    depends_on("eigen", when="+examples", type=("build", "link"))

    depends_on("llvm-openmp", when="+openmp %apple-clang")

    depends_on("cuda@11.0:", when="+cuda")

    depends_on("mpi", when="+mpi")

    depends_on("python@3.9:", when="+python", type=("build", "run"))
    depends_on("py-pybind11@2.11:", when="+python", type="build")
    depends_on("py-numpy@1.20:", when="+python", type=("build", "run"))

    depends_on("googletest@1.14:", when="+tests", type="build")

    def cmake_args(self):
        args = [
            self.define_from_variant("LIBACCINT_USE_CUDA", "cuda"),
            self.define_from_variant("LIBACCINT_USE_MPI", "mpi"),
            self.define_from_variant("LIBACCINT_USE_OPENMP", "openmp"),
            self.define_from_variant("LIBACCINT_BUILD_PYTHON", "python"),
            self.define_from_variant("LIBACCINT_BUILD_EXAMPLES", "examples"),
            self.define_from_variant("LIBACCINT_BUILD_TESTS", "tests"),
            self.define_from_variant("BUILD_SHARED_LIBS", "shared"),
            self.define("LIBACCINT_ALLOW_FETCHCONTENT", "OFF"),
            self.define("LIBACCINT_RELOCATABLE", "ON"),
        ]

        if "+python" in self.spec:
            args.append(
                self.define(
                    "Python3_EXECUTABLE",
                    self.spec["python"].prefix.bin.python,
                )
            )

        if "+cuda" in self.spec and self.spec.variants["cuda_arch"].value != ("none",):
            args.append(
                self.define(
                    "CMAKE_CUDA_ARCHITECTURES",
                    ";".join(self.spec.variants["cuda_arch"].value),
                )
            )

        return args

    @run_after("install")
    def install_test(self):
        """Smoke test: verify library and headers are installed."""
        assert self.prefix.lib.join("libaccint.so").exists() or \
               self.prefix.lib.join("libaccint.a").exists() or \
               self.prefix.lib.join("libaccint.dylib").exists(), \
               "Library not found after installation"
        assert (
            self.prefix.include.join("libaccint", "libaccint.hpp").exists()
        ), "Headers not found after installation"
