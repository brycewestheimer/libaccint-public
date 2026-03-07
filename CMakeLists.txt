cmake_minimum_required(VERSION 3.25)

project(LibAccInt
    VERSION 0.1.0  # Canonical version — keep in sync with python/pyproject.toml, python/setup.py, python/libaccint/__init__.py
    DESCRIPTION "High-performance molecular integral library with GPU acceleration"
    LANGUAGES CXX
)

# Pre-release suffix for alpha/beta/rc builds (empty string for stable releases)
set(LIBACCINT_PRERELEASE "alpha.2")
set(LIBACCINT_VERSION_FULL "${PROJECT_VERSION}-${LIBACCINT_PRERELEASE}")

# =============================================================================
# Build Options
# =============================================================================

option(LIBACCINT_USE_CUDA "Enable CUDA backend (AUTO=detect, ON=require, OFF=disable)" AUTO)
set(LIBACCINT_USE_MPI "AUTO" CACHE STRING
    "Enable MPI backend (AUTO=detect, ON=require, OFF=disable)")
set_property(CACHE LIBACCINT_USE_MPI PROPERTY STRINGS "AUTO" "ON" "OFF")
option(LIBACCINT_BUILD_TESTS "Build test suite" ON)
option(LIBACCINT_BUILD_PYTHON "Build Python bindings" OFF)  # Disabled by default until implemented
option(LIBACCINT_BUILD_EXAMPLES "Build example programs" ON)
option(LIBACCINT_USE_OPENMP "Enable OpenMP for CPU parallelization" ON)
option(LIBACCINT_BUILD_BENCHMARKS "Build benchmark suite" OFF)
option(LIBACCINT_USE_ASAN "Enable AddressSanitizer for debug builds" OFF)
option(LIBACCINT_GPU_ONLY "Build with GPU-only execution (no CPU fallback)" OFF)
option(LIBACCINT_TRACE_DISPATCH "Enable verbose dispatch tracing" OFF)
option(LIBACCINT_ALLOW_FETCHCONTENT
    "Allow CMake to fetch third-party dependencies from the network"
    ON)
# Generated registry options — reserved for future use (generated kernels not
# included in this alpha release; stubs are always compiled instead).
option(LIBACCINT_ENABLE_CPU_GENERATED_REGISTRY
    "Build the full CPU generated-kernel registry (reserved for future release)"
    OFF)
option(LIBACCINT_ENABLE_CUDA_GENERATED_REGISTRY
    "Build the CUDA generated-kernel registry TU (reserved for future release)"
    OFF)
set(LIBACCINT_CUDA_HEAVY_TU_COMPILE_JOBS "1" CACHE STRING
    "Max concurrent compilation jobs for heavy CUDA translation units under Ninja (0 disables throttling)")
set(LIBACCINT_CUDA_HEAVY_TU_OPT_LEVEL "3" CACHE STRING
    "Optimization level override for heavy CUDA translation units (0-3; 3 keeps default release optimization)")
set_property(CACHE LIBACCINT_CUDA_HEAVY_TU_OPT_LEVEL PROPERTY STRINGS "0" "1" "2" "3")

set(LIBACCINT_KERNEL_DISPATCH "handwritten" CACHE STRING
    "Kernel dispatch strategy: handwritten, optimal")
set_property(CACHE LIBACCINT_KERNEL_DISPATCH PROPERTY STRINGS
    "handwritten" "optimal")

# Code generation scope options
set(LIBACCINT_MAX_AM "4" CACHE STRING
    "Maximum angular momentum for generated kernels (stable support this cycle: 0=S, 1=P, 2=D, 3=F, 4=G)")
set_property(CACHE LIBACCINT_MAX_AM PROPERTY STRINGS "0" "1" "2" "3" "4")
set(LIBACCINT_K_RANGES "3,6" CACHE STRING
    "K-range boundaries for contraction-aware dispatch (comma-separated, e.g. '3,6')")

# Validate and parse LIBACCINT_K_RANGES into concrete threshold values.
string(REPLACE "," ";" _LIBACCINT_K_RANGES_LIST "${LIBACCINT_K_RANGES}")
list(LENGTH _LIBACCINT_K_RANGES_LIST _LIBACCINT_K_RANGES_LEN)
if(NOT _LIBACCINT_K_RANGES_LEN EQUAL 2)
    message(FATAL_ERROR
        "LIBACCINT_K_RANGES must contain exactly two comma-separated integers (got '${LIBACCINT_K_RANGES}')")
endif()

list(GET _LIBACCINT_K_RANGES_LIST 0 LIBACCINT_SMALL_K_MAX)
list(GET _LIBACCINT_K_RANGES_LIST 1 LIBACCINT_MEDIUM_K_MAX)
string(STRIP "${LIBACCINT_SMALL_K_MAX}" LIBACCINT_SMALL_K_MAX)
string(STRIP "${LIBACCINT_MEDIUM_K_MAX}" LIBACCINT_MEDIUM_K_MAX)

if(NOT LIBACCINT_SMALL_K_MAX MATCHES "^[0-9]+$" OR
   NOT LIBACCINT_MEDIUM_K_MAX MATCHES "^[0-9]+$")
    message(FATAL_ERROR
        "LIBACCINT_K_RANGES must be numeric (got '${LIBACCINT_K_RANGES}')")
endif()

if(LIBACCINT_SMALL_K_MAX LESS 1 OR LIBACCINT_MEDIUM_K_MAX LESS 1 OR
   LIBACCINT_SMALL_K_MAX GREATER_EQUAL LIBACCINT_MEDIUM_K_MAX)
    message(FATAL_ERROR
        "LIBACCINT_K_RANGES requires 1 <= small < medium (got '${LIBACCINT_K_RANGES}')")
endif()

# Validate LIBACCINT_MAX_AM
if(LIBACCINT_MAX_AM LESS 0 OR LIBACCINT_MAX_AM GREATER 4)
    message(FATAL_ERROR "LIBACCINT_MAX_AM must be between 0 and 4 for the stable contract in this cycle (got ${LIBACCINT_MAX_AM})")
endif()

if(NOT LIBACCINT_CUDA_HEAVY_TU_COMPILE_JOBS MATCHES "^[0-9]+$")
    message(FATAL_ERROR
        "LIBACCINT_CUDA_HEAVY_TU_COMPILE_JOBS must be a non-negative integer (got '${LIBACCINT_CUDA_HEAVY_TU_COMPILE_JOBS}')")
endif()

if(NOT LIBACCINT_CUDA_HEAVY_TU_OPT_LEVEL MATCHES "^[0-3]$")
    message(FATAL_ERROR
        "LIBACCINT_CUDA_HEAVY_TU_OPT_LEVEL must be one of 0,1,2,3 (got '${LIBACCINT_CUDA_HEAVY_TU_OPT_LEVEL}')")
endif()

# Map AM number to human-readable label
set(_AM_LABEL_LIST "S" "P" "D" "F" "G")
list(GET _AM_LABEL_LIST ${LIBACCINT_MAX_AM} LIBACCINT_MAX_AM_LABEL)

# =============================================================================
# Project Setup
# =============================================================================

# Add CMake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Include CMake modules
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CompilerFlags)
include(BackendDetection)
include(VectorizationDetection)
include(CodeGeneration)
include(version)
include(abi-version)

# =============================================================================
# Backend Detection
# =============================================================================

detect_cuda_backend()
detect_mpi_backend()
detect_vectorization()
libaccint_add_codegen_target()

# Report configuration
message(STATUS "==============================================")
message(STATUS "LibAccInt Configuration Summary")
message(STATUS "==============================================")
message(STATUS "Version: ${LIBACCINT_VERSION_FULL}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CUDA backend: ${LIBACCINT_USE_CUDA}")
message(STATUS "MPI backend: ${LIBACCINT_USE_MPI}")
message(STATUS "OpenMP: ${LIBACCINT_USE_OPENMP}")
message(STATUS "Vectorization: ${LIBACCINT_VECTOR_ISA} (width=${LIBACCINT_VECTOR_WIDTH})")
message(STATUS "Build tests: ${LIBACCINT_BUILD_TESTS}")
message(STATUS "Build Python: ${LIBACCINT_BUILD_PYTHON}")
message(STATUS "Build examples: ${LIBACCINT_BUILD_EXAMPLES}")
message(STATUS "Build benchmarks: ${LIBACCINT_BUILD_BENCHMARKS}")
message(STATUS "GPU only: ${LIBACCINT_GPU_ONLY}")
message(STATUS "CPU generated registry: ${LIBACCINT_ENABLE_CPU_GENERATED_REGISTRY}")
message(STATUS "CUDA generated registry: ${LIBACCINT_ENABLE_CUDA_GENERATED_REGISTRY}")
message(STATUS "CUDA heavy TU compile jobs: ${LIBACCINT_CUDA_HEAVY_TU_COMPILE_JOBS}")
message(STATUS "CUDA heavy TU optimization level: ${LIBACCINT_CUDA_HEAVY_TU_OPT_LEVEL}")
message(STATUS "Kernel dispatch: ${LIBACCINT_KERNEL_DISPATCH}")
message(STATUS "Max AM (codegen): ${LIBACCINT_MAX_AM} (${LIBACCINT_MAX_AM_LABEL}-functions)")
message(STATUS "K-ranges (codegen): ${LIBACCINT_K_RANGES}")
message(STATUS "==============================================")

# =============================================================================
# Dependencies
# =============================================================================

include(FetchContent)

# GoogleTest for testing
if(LIBACCINT_BUILD_TESTS)
    find_package(GTest CONFIG QUIET NO_CMAKE_PACKAGE_REGISTRY)
    if(NOT GTest_FOUND)
        if(NOT LIBACCINT_ALLOW_FETCHCONTENT)
            message(FATAL_ERROR
                "GTest not found and LIBACCINT_ALLOW_FETCHCONTENT=OFF. "
                "Install GTest or disable LIBACCINT_BUILD_TESTS.")
        endif()
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
            GIT_SHALLOW TRUE
        )
        # Prevent GoogleTest from overriding our compiler/linker options
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
endif()

# nlohmann/json for basis set parsing
find_package(nlohmann_json CONFIG QUIET NO_CMAKE_PACKAGE_REGISTRY)
if(NOT nlohmann_json_FOUND)
    if(NOT LIBACCINT_ALLOW_FETCHCONTENT)
        message(FATAL_ERROR
            "nlohmann_json not found and LIBACCINT_ALLOW_FETCHCONTENT=OFF. "
            "Install nlohmann_json or enable network fetches.")
    endif()
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE
    )
    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    set(JSON_Install OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(nlohmann_json)
    # Export FetchContent-provided target so install(EXPORT) succeeds for static lib
    install(TARGETS nlohmann_json EXPORT LibAccIntTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# Eigen for linear algebra in tests and examples (header-only)
if(LIBACCINT_BUILD_TESTS OR LIBACCINT_BUILD_EXAMPLES)
    find_package(Eigen3 CONFIG QUIET NO_CMAKE_PACKAGE_REGISTRY)
    if(NOT Eigen3_FOUND)
        if(NOT LIBACCINT_ALLOW_FETCHCONTENT)
            message(FATAL_ERROR
                "Eigen3 not found and LIBACCINT_ALLOW_FETCHCONTENT=OFF. "
                "Install Eigen3 or disable tests/examples.")
        endif()
        FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.0
            GIT_SHALLOW TRUE
        )
        set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(eigen)
    endif()
endif()

# Google Benchmark for benchmarking
if(LIBACCINT_BUILD_BENCHMARKS)
    include(FetchContent)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
        GIT_SHALLOW TRUE
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

# =============================================================================
# Generate Configuration Header
# =============================================================================

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/libaccint/config.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/libaccint/config.hpp"
    @ONLY
)

# =============================================================================
# Library Source Files
# =============================================================================

# Shared/common source files (CPU + common utilities)
set(LIBACCINT_SHARED_SOURCES
    src/shared/libaccint.cpp
    src/shared/backend.cpp
    src/shared/math/gaussian_product.cpp
    src/shared/math/boys_function.cpp
    src/shared/math/boys_function_float.cpp
    src/shared/math/boys_range_separated.cpp
    src/shared/math/normalization.cpp
    src/shared/math/rys_quadrature.cpp
    src/shared/math/spherical_transform.cpp
    src/shared/basis/shell.cpp
    src/shared/basis/shell_set.cpp
    src/shared/basis/shell_set_pair.cpp
    src/shared/basis/shell_set_triple.cpp
    src/shared/operators/operator.cpp
    src/shared/operators/one_electron_operator.cpp
    src/shared/basis/basis_set.cpp
    src/shared/utils/matrix_assembly.cpp
    src/shared/consumers/fock_builder.cpp
    src/shared/data/builtin_basis.cpp
    src/shared/data/basis_parser.cpp
    src/shared/data/bse_json_parser.cpp
    src/shared/memory/memory_pool.cpp
    src/shared/kernels/registry_key.cpp
    src/shared/kernels/cost_model.cpp
    src/shared/kernels/optimal_dispatch_table.cpp
    src/shared/screening/screened_quartet_iterator.cpp
    src/shared/screening/schwarz_bounds.cpp
    src/shared/screening/density_screening.cpp
    src/shared/math/interaction_tensors.cpp
    src/shared/operators/projection_operator.cpp
    src/shared/data/auxiliary_basis_parser.cpp
    src/shared/data/auxiliary_basis_data.cpp
    src/shared/data/auxiliary_basis_selector.cpp
    src/shared/df/three_center_storage.cpp
    src/shared/df/out_of_core_storage.cpp
    src/shared/basis/auxiliary_basis_set.cpp
    src/shared/consumers/df_fock_builder.cpp
    src/shared/consumers/mixed_precision_fock_builder.cpp
    src/shared/utils/logging.cpp
    src/shared/utils/diagnostics.cpp
    src/shared/utils/input_validation.cpp
)

# CPU-specific host source files
set(LIBACCINT_HOST_SOURCES
    src/host/kernels/overlap_kernel.cpp
    src/host/kernels/kinetic_kernel.cpp
    src/host/kernels/nuclear_kernel.cpp
    src/host/engine/cpu_engine.cpp
    src/host/engine/dispatch_policy.cpp
    src/host/engine/kernel_calculator.cpp
    src/host/engine/dispatch_registry.cpp
    src/host/kernels/eri_kernel.cpp
    src/host/kernels/eri_erf_coulomb_kernel.cpp
    src/host/kernels/eri_erfc_coulomb_kernel.cpp
    src/host/kernels/multipole_kernel.cpp
    src/host/kernels/momentum_kernel.cpp
    src/host/kernels/distributed_multipole_kernel.cpp
    src/host/kernels/three_center_eri_kernel.cpp
    src/host/kernels/two_center_coulomb_kernel.cpp
)

list(APPEND LIBACCINT_HOST_SOURCES
    src/host/kernels/generated_kernel_registry_stub.cpp
)

# CUDA-specific device source files
set(LIBACCINT_CUDA_SOURCES
    src/device/cuda/device_memory.cu
    src/device/cuda/device_manager.cu
    src/device/cuda/multi_device_memory.cu
    src/device/cuda/device_resource_tracker.cu
    src/device/cuda/stream_management.cu
    src/device/cuda/gpu_execution_slot.cu
    src/device/cuda/math/boys_device.cu
    src/device/cuda/basis/device_data.cu
    src/device/cuda/operators/device_operator_data.cu
    src/device/cuda/kernels/overlap_kernel.cu
    src/device/cuda/kernels/kinetic_kernel.cu
    src/device/cuda/kernels/nuclear_kernel.cu
    src/device/cuda/kernels/fused_1e_kernel.cu
    src/device/cuda/kernels/eri_kernel.cu
    src/device/cuda/kernels/eri_kernel_warp.cu
    src/device/cuda/kernels/eri_scatter_kernel.cu
    src/device/cuda/engine/cuda_engine.cu
    src/device/cuda/engine/multi_gpu_engine.cu
    src/device/cuda/consumers/fock_builder_gpu.cu
    src/device/cuda/consumers/multi_gpu_fock_builder.cu
    src/device/cuda/kernels/multipole_kernel.cu
    src/device/cuda/kernels/momentum_kernel.cu
    src/device/cuda/kernels/distributed_multipole_kernel.cu
)

list(APPEND LIBACCINT_CUDA_SOURCES
    src/device/cuda/kernels/generated_kernel_registry_stub.cu
)

# MPI-specific distributed source files
set(LIBACCINT_MPI_SOURCES
    src/mpi/mpi_guard.cpp
    src/mpi/mpi_fock_builder.cpp
    src/mpi/mpi_engine.cpp
)

# Combine all sources
set(LIBACCINT_SOURCES
    ${LIBACCINT_SHARED_SOURCES}
    ${LIBACCINT_HOST_SOURCES}
)

if(LIBACCINT_USE_CUDA)
    list(APPEND LIBACCINT_SOURCES ${LIBACCINT_CUDA_SOURCES})
endif()

if(LIBACCINT_USE_MPI)
    list(APPEND LIBACCINT_SOURCES ${LIBACCINT_MPI_SOURCES})
endif()

# Throttle known heavy CUDA translation units under Ninja to avoid WSL2
# process/memory spikes during compilation.
if(LIBACCINT_USE_CUDA AND CMAKE_GENERATOR MATCHES "Ninja")
    if(NOT LIBACCINT_CUDA_HEAVY_TU_COMPILE_JOBS STREQUAL "0")
        set_property(
            GLOBAL APPEND PROPERTY JOB_POOLS
            libaccint_cuda_heavy=${LIBACCINT_CUDA_HEAVY_TU_COMPILE_JOBS}
        )

        set(_LIBACCINT_CUDA_HEAVY_TU_SOURCES
            src/device/cuda/kernels/eri_kernel.cu
        )

        set_source_files_properties(
            ${_LIBACCINT_CUDA_HEAVY_TU_SOURCES}
            PROPERTIES
                JOB_POOL_COMPILE libaccint_cuda_heavy
        )

        if(NOT LIBACCINT_CUDA_HEAVY_TU_OPT_LEVEL STREQUAL "3")
            set_source_files_properties(
                ${_LIBACCINT_CUDA_HEAVY_TU_SOURCES}
                PROPERTIES
                    COMPILE_OPTIONS "-O${LIBACCINT_CUDA_HEAVY_TU_OPT_LEVEL}"
            )
        endif()
    endif()
endif()

# =============================================================================
# Library Targets
# =============================================================================

# Shared library
add_library(libaccint SHARED ${LIBACCINT_SOURCES})
add_library(LibAccInt::libaccint ALIAS libaccint)

# Static library
add_library(libaccint_static STATIC ${LIBACCINT_SOURCES})
add_library(LibAccInt::libaccint_static ALIAS libaccint_static)

# Set library output name (without 'lib' prefix duplication)
# SOVERSION 0 = unstable ABI (pre-1.0). Will increment to 1 on first stable release.
# Until then, ABI may change between any releases. Downstream consumers should
# pin to exact versions or rebuild from source.
set_target_properties(libaccint PROPERTIES
    OUTPUT_NAME accint
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

set_target_properties(libaccint_static PROPERTIES
    OUTPUT_NAME accint
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(libaccint
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_include_directories(libaccint_static
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Apply compiler flags
apply_libaccint_compile_flags(libaccint)
apply_libaccint_compile_flags(libaccint_static)

# Dispatch tracing (zero overhead when disabled)
if(LIBACCINT_TRACE_DISPATCH)
    target_compile_definitions(libaccint PRIVATE LIBACCINT_TRACE_DISPATCH=1)
    target_compile_definitions(libaccint_static PRIVATE LIBACCINT_TRACE_DISPATCH=1)
endif()

# Link nlohmann/json (required for basis set parsing, header-only)
target_link_libraries(libaccint PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(libaccint_static PRIVATE nlohmann_json::nlohmann_json)
target_include_directories(libaccint PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_include_directories(libaccint_static PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# Compile-time path for installed basis set data
option(LIBACCINT_RELOCATABLE "Use relative data path for relocatable packages" OFF)
if(LIBACCINT_RELOCATABLE)
    set(_datadir_value "../${CMAKE_INSTALL_DATADIR}/libaccint/basis_sets")
else()
    set(_datadir_value "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/libaccint/basis_sets")
endif()
target_compile_definitions(libaccint PRIVATE "LIBACCINT_INSTALL_DATADIR=\"${_datadir_value}\"")
target_compile_definitions(libaccint_static PRIVATE "LIBACCINT_INSTALL_DATADIR=\"${_datadir_value}\"")

# Link CUDA libraries if enabled
if(LIBACCINT_USE_CUDA)
    target_link_libraries(libaccint PUBLIC CUDA::cudart CUDA::cublas)
    target_link_libraries(libaccint_static PUBLIC CUDA::cudart CUDA::cublas)
endif()

if(LIBACCINT_USE_MPI)
    target_link_libraries(libaccint PUBLIC MPI::MPI_CXX)
    target_link_libraries(libaccint_static PUBLIC MPI::MPI_CXX)
endif()

if(CMAKE_DL_LIBS)
    target_link_libraries(libaccint PRIVATE ${CMAKE_DL_LIBS})
    target_link_libraries(libaccint_static PRIVATE ${CMAKE_DL_LIBS})
endif()

# =============================================================================
# Installation
# =============================================================================

# Install library targets
install(TARGETS libaccint libaccint_static
    EXPORT LibAccIntTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install basis set data files
install(DIRECTORY share/basis_sets
    DESTINATION ${CMAKE_INSTALL_DATADIR}/libaccint
    FILES_MATCHING
    PATTERN "*.json"
    PATTERN "*.md"
)

# Install headers
install(DIRECTORY include/libaccint
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Install generated config header
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/libaccint/config.hpp"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libaccint
)

# Install CMake package configuration
install(EXPORT LibAccIntTargets
    FILE LibAccIntTargets.cmake
    NAMESPACE LibAccInt::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibAccInt
)

# Generate and install package config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LibAccIntConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/LibAccIntConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibAccInt
)

# Generate and install package version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/LibAccIntConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/LibAccIntConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/LibAccIntConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibAccInt
)

# =============================================================================
# pkg-config Support (Task 26.3.2)
# =============================================================================

# Build pkg-config private libs and requires
set(LIBACCINT_PC_PRIVATE_LIBS "")
set(LIBACCINT_PC_REQUIRES "")

if(LIBACCINT_USE_OPENMP)
    string(APPEND LIBACCINT_PC_PRIVATE_LIBS " ${OpenMP_CXX_FLAGS}")
endif()

if(LIBACCINT_USE_CUDA)
    string(APPEND LIBACCINT_PC_PRIVATE_LIBS " -lcudart -lcublas")
endif()

find_package(BLAS QUIET)
if(BLAS_FOUND)
    string(APPEND LIBACCINT_PC_PRIVATE_LIBS " -lblas")
endif()

find_package(LAPACK QUIET)
if(LAPACK_FOUND)
    string(APPEND LIBACCINT_PC_PRIVATE_LIBS " -llapack")
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/libaccint.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libaccint.pc"
    @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libaccint.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# =============================================================================
# ABI Stability (Task 26.4.3)
# =============================================================================

# Note: Symbol visibility and version scripts are available but not applied by
# default. To enable them, set LIBACCINT_ENABLE_SYMBOL_VERSIONING=ON.
# This requires adding LIBACCINT_EXPORT macros to all public API symbols first.
option(LIBACCINT_ENABLE_SYMBOL_VERSIONING "Enable symbol visibility and linker version script" OFF)

if(LIBACCINT_ENABLE_SYMBOL_VERSIONING)
    apply_symbol_visibility(libaccint)
    apply_symbol_visibility(libaccint_static)
    apply_version_script(libaccint)
endif()

# Add ABI check targets (always available for comparison)
add_abi_check_target(libaccint)

# =============================================================================
# Subdirectories
# =============================================================================

if(LIBACCINT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(LIBACCINT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(LIBACCINT_BUILD_BENCHMARKS)
    # add_subdirectory(benchmarks)  # Enable when benchmarks are implemented
endif()

if(LIBACCINT_BUILD_PYTHON)
    add_subdirectory(python)
endif()
